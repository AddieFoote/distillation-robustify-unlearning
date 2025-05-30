import os
import random
import math
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding
)
import wandb


def unlearn_npo(
    model_name,
    teacher_model_name,          # NEW: we load a teacher model for NPO
    forget_train_file,
    retain_train_file,
    eng_valid_file,
    kor_valid_file,
    output_dir,
    cache_dir,
    dataset_cache_dir,

    ga_gd,                       # If True, also do normal CE on the retain set
    seed,
    device,
    batch_size,
    gradient_accumulation_steps,
    epochs,
    learning_rate,
    max_steps,
    num_warmup_steps,
    validation_steps,
    save_checkpoint_steps,
    scheduler_type,
    min_lr,
    weight_decay,
    gradient_clipping_threshold,
    max_length,

    # ---- NPO-specific hyperparameter ----
    beta,                        # Inverse temperature for NPO

    use_wandb,
    wandb_project,
    wandb_run_name,

    use_local_record,
    path_local_record,
):
    """
    Negative Preference Optimization script using Accelerate on pretokenized JSONL datasets.

    If ga_gd=True, we also preserve the retain dataset with normal cross-entropy (like GA+GD).
    This code is heavily based on 'unlearn_ga', but we replace the forget objective with an NPO loss:
       L_npo = -2/beta * E[ log sigma( - beta * log( pi_theta(y|x)/ pi_teacher(y|x) ) ) ].

    Usage Example:
      $ python unlearn_npo.py \
          --model-name <your_pretrained_LM> \
          --teacher-model-name <teacher_LM_to_unlearn_from> \
          --forget-train-file data/forget.jsonl \
          --retain-train-file data/retain.jsonl \
          --eng-valid-file data/val_eng.jsonl \
          --kor-valid-file data/val_kor.jsonl \
          ... plus other arguments ...
    """
    accelerator = Accelerator()
    print_message = accelerator.is_main_process

    train_args = {**locals()}
    print_acc(f"[unlearn_npo.py] Initiated NPO training with:\n{train_args}", print_message)

    # ----------------------------------------------------------------
    # Setup: seeds, directories, W&B, local record
    # ----------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)

    if use_wandb and accelerator.is_main_process:
        wandb.init(project=wandb_project, name=wandb_run_name, config=train_args)

    if use_local_record and accelerator.is_main_process:
        local_dir = os.path.dirname(path_local_record)
        os.makedirs(local_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # Load main model + teacher model + tokenizer
    # ----------------------------------------------------------------
    print_acc(f"[unlearn_npo.py] Loading main model {model_name}", print_message)
    model_config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        attn_implementation='eager'
    )

    print_acc(f"[unlearn_npo.py] Loading teacher (frozen) model {teacher_model_name}", print_message)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        cache_dir=cache_dir,
        attn_implementation='eager'
    )
    teacher_model.eval()  # do not update teacher

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ----------------------------------------------------------------
    # Helper: filter function for length
    # ----------------------------------------------------------------
    def filter_long_batch(batch):
        """Return a list of booleans for each example in the batch."""
        return [len(ids) <= max_length for ids in batch["input_ids"]]

    # ----------------------------------------------------------------
    # Load & filter FORGET dataset (NPO objective)
    # ----------------------------------------------------------------
    if not forget_train_file.strip():
        raise ValueError("forget_train_file is empty => must provide a dataset to 'forget'")
    print_acc("[unlearn_npo.py] Loading 'forget' dataset", print_message)
    forget_ds = load_dataset(
        "json",
        data_files=forget_train_file,
        split="train",
        cache_dir=dataset_cache_dir
    )
    print_acc(f"[unlearn_npo.py] Forget dataset size: {len(forget_ds)}", print_message)
    sample_text = forget_ds[0]["text"].replace('\n', ' ')
    print_acc(f'[unlearn_npo.py] Sample forget text: "{sample_text[:200]}..."', print_message)

    forget_ds = forget_ds.remove_columns("text")
    forget_ds = forget_ds.filter(filter_long_batch, batched=True, batch_size=200_000, num_proc=100)
    forget_loader = DataLoader(
        forget_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=DataCollatorWithPadding(
            tokenizer=tokenizer, padding="max_length", max_length=max_length
        )
    )

    # ----------------------------------------------------------------
    # Load & filter RETAIN dataset (Used only if ga_gd=True)
    # ----------------------------------------------------------------
    if ga_gd and retain_train_file.strip():
        print_acc("[unlearn_npo.py] => using retain dataset as well", print_message)
        retain_ds = load_dataset(
            "json",
            data_files=retain_train_file,
            split="train",
            cache_dir=dataset_cache_dir
        )
        print_acc(f"[unlearn_npo.py] Retain dataset size: {len(retain_ds)}", print_message)
        sample_text_r = retain_ds[0]["text"].replace('\n', ' ')
        print_acc(f'[unlearn_npo.py] Sample retain text: "{sample_text_r[:200]}..."', print_message)

        retain_ds = retain_ds.remove_columns("text")
        retain_ds = retain_ds.filter(filter_long_batch, batched=True, batch_size=200_000, num_proc=100)
        retain_loader = DataLoader(
            retain_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=DataCollatorWithPadding(
                tokenizer=tokenizer, padding="max_length", max_length=max_length
            )
        )
    else:
        # either ga_gd=False, or no retain data
        retain_loader = None

    # ----------------------------------------------------------------
    # Load & filter VALIDATION datasets (English & Korean)
    # ----------------------------------------------------------------
    print_acc("[unlearn_npo.py] Loading validation datasets", print_message)
    eng_valid_ds = load_dataset(
        "json",
        data_files=eng_valid_file,
        split="train",
        cache_dir=dataset_cache_dir
    )
    kor_valid_ds = load_dataset(
        "json",
        data_files=kor_valid_file,
        split="train",
        cache_dir=dataset_cache_dir
    )
    print_acc(f"[unlearn_npo.py] Eng validation dataset size: {len(eng_valid_ds)}", print_message)
    print_acc(f"[unlearn_npo.py] Kor validation dataset size: {len(kor_valid_ds)}", print_message)

    eng_valid_ds = eng_valid_ds.remove_columns("text")
    kor_valid_ds = kor_valid_ds.remove_columns("text")

    eng_valid_ds = eng_valid_ds.filter(filter_long_batch, batched=True, batch_size=200_000, num_proc=100)
    kor_valid_ds = kor_valid_ds.filter(filter_long_batch, batched=True, batch_size=200_000, num_proc=100)

    eng_valid_loader = DataLoader(
        eng_valid_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(
            tokenizer=tokenizer, padding="max_length", max_length=max_length
        )
    )
    kor_valid_loader = DataLoader(
        kor_valid_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(
            tokenizer=tokenizer, padding="max_length", max_length=max_length
        )
    )

    # ----------------------------------------------------------------
    # Determine steps
    # ----------------------------------------------------------------
    steps_per_epoch_forget = len(forget_loader)
    if ga_gd and retain_loader is not None:
        steps_per_epoch_retain = len(retain_loader)
        steps_per_epoch = max(steps_per_epoch_forget, steps_per_epoch_retain)
    else:
        steps_per_epoch = steps_per_epoch_forget

    effective_steps_per_epoch = steps_per_epoch // gradient_accumulation_steps
    total_steps = effective_steps_per_epoch * epochs
    if max_steps > 0:
        total_steps = min(total_steps, max_steps)
    print_acc(f"[unlearn_npo.py] {steps_per_epoch} steps per epoch, total steps: {total_steps}", print_message)

    # ----------------------------------------------------------------
    # Optimizer + LR scheduler
    # ----------------------------------------------------------------
    print_acc(f"[unlearn_npo.py] Using AdamW optimizer, LR={learning_rate}, weight_decay={weight_decay}", print_message)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if scheduler_type == "linear":
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            else:
                progress = float(current_step - num_warmup_steps) / float(
                    max(1, total_steps - num_warmup_steps)
                )
                return (1.0 - progress) * (1.0 - (min_lr / learning_rate)) + (min_lr / learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif scheduler_type == "cosine":
        def cosine_lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            else:
                progress = float(current_step - num_warmup_steps) / float(
                    max(1, total_steps - num_warmup_steps)
                )
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return cosine_decay * (1.0 - (min_lr / learning_rate)) + (min_lr / learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lr_lambda)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    # ----------------------------------------------------------------
    # Prepare with Accelerator
    # ----------------------------------------------------------------
    model, optimizer, forget_loader, eng_valid_loader, kor_valid_loader, scheduler = accelerator.prepare(
        model, optimizer, forget_loader, eng_valid_loader, kor_valid_loader, scheduler
    )
    teacher_model = accelerator.prepare(teacher_model)  # place teacher on device
    if ga_gd and retain_loader is not None:
        retain_loader = accelerator.prepare(retain_loader)

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    print_acc("[unlearn_npo.py] Starting NPO training", print_message)
    global_step = 0
    global_tokens = 0

    forget_loader_iter = iter(forget_loader)
    retain_loader_iter = iter(retain_loader) if (ga_gd and retain_loader) else None

    for epoch in range(epochs):
        print_acc(f"[unlearn_npo.py] Epoch {epoch+1}/{epochs}", print_message)
        model.train()
        teacher_model.eval()

        for step_in_epoch in range(steps_per_epoch):
            # 1) Get forget batch
            try:
                forget_batch = next(forget_loader_iter)
            except StopIteration:
                forget_loader_iter = iter(forget_loader)
                forget_batch = next(forget_loader_iter)

            # --- NPO loss on forget data ---
            with torch.no_grad():
                teacher_out = teacher_model(
                    input_ids=forget_batch["input_ids"],
                    attention_mask=forget_batch["attention_mask"],
                )
            student_out = model(
                input_ids=forget_batch["input_ids"],
                attention_mask=forget_batch["attention_mask"],
            )

            # negative preference objective
            npo_loss_forget = npo_loss_fn(
                student_logits=student_out.logits,
                teacher_logits=teacher_out.logits,
                labels=forget_batch["input_ids"],
                pad_token_id=tokenizer.pad_token_id,
                beta=beta
            )

            # 2) If ga_gd => do normal CE on retain data
            if ga_gd and retain_loader_iter is not None:
                try:
                    retain_batch = next(retain_loader_iter)
                except StopIteration:
                    retain_loader_iter = iter(retain_loader)
                    retain_batch = next(retain_loader_iter)

                outputs_retain = model(
                    input_ids=retain_batch["input_ids"],
                    attention_mask=retain_batch["attention_mask"]
                )
                ce_loss_retain = cross_entropy_loss_fn(
                    outputs_retain.logits,
                    retain_batch["input_ids"],
                    tokenizer.pad_token_id
                )

                total_loss = (npo_loss_forget + ce_loss_retain) / gradient_accumulation_steps

                # Count tokens
                tokens_forget = forget_batch["attention_mask"].sum().detach()
                tokens_forget = accelerator.gather(tokens_forget).sum().item()
                tokens_retain = retain_batch["attention_mask"].sum().detach()
                tokens_retain = accelerator.gather(tokens_retain).sum().item()
                global_tokens += (tokens_forget + tokens_retain)

            else:
                total_loss = npo_loss_forget / gradient_accumulation_steps
                tokens_this_batch = forget_batch["attention_mask"].sum().detach()
                tokens_this_batch = accelerator.gather(tokens_this_batch).sum().item()
                global_tokens += tokens_this_batch

            # --- Backprop ---
            accelerator.backward(total_loss)

            if (step_in_epoch + 1) % gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), gradient_clipping_threshold)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step == 1 or global_step % 5 == 0:
                    msg = (f"[unlearn_npo.py] Epoch {epoch+1}/{epochs}, "
                           f"Step {global_step}/{total_steps}, NPO{'+GD' if ga_gd else ''} "
                           f"=> npo_forget: {npo_loss_forget:.6f}")
                    print_acc(msg, print_message)
                    if ga_gd and retain_loader_iter is not None:
                        print_acc(f"[unlearn_npo.py] ce_retain: {ce_loss_retain:.6f}", print_message)

                train_log_dict = {
                    "train/npo_forget_loss": npo_loss_forget.item(),
                    "train/step": global_step,
                    "train/tokens_seen": global_tokens,
                    "train/lr": scheduler.get_last_lr()[0],
                }
                if ga_gd and retain_loader_iter is not None:
                    train_log_dict["train/ce_loss_retain"] = ce_loss_retain.item()

                if use_wandb and accelerator.is_main_process:
                    wandb.log(train_log_dict)
                if use_local_record and accelerator.is_main_process:
                    with open(path_local_record, "a", encoding="utf-8") as f:
                        f.write(json.dumps(train_log_dict) + "\n")

                # Validation
                if global_step == 1 or global_step % validation_steps == 0:
                    print_acc("[unlearn_npo.py] Running validation ...", print_message)
                    eng_valid_ce_loss = evaluate_ce_loss(model, eng_valid_loader, tokenizer.pad_token_id, accelerator)
                    eng_stats = model.__dict__.get("_temp_metrics", {"ppl": float("inf")})
                    eng_valid_ppl = eng_stats["ppl"]

                    kor_valid_ce_loss = evaluate_ce_loss(model, kor_valid_loader, tokenizer.pad_token_id, accelerator)
                    kor_stats = model.__dict__.get("_temp_metrics", {"ppl": float("inf")})
                    kor_valid_ppl = kor_stats["ppl"]

                    print_acc(f"English CE loss: {eng_valid_ce_loss:.6f}", print_message)
                    print_acc(f"Korean CE loss: {kor_valid_ce_loss:.6f}", print_message)

                    val_log_dict = {
                        "val/eng_ce_loss": eng_valid_ce_loss,
                        "val/kor_ce_loss": kor_valid_ce_loss,
                        "val/eng_ppl": eng_valid_ppl,
                        "val/kor_ppl": kor_valid_ppl,
                        "train/step": global_step,
                        "train/tokens_seen": global_tokens,
                    }
                    if use_wandb and accelerator.is_main_process:
                        wandb.log(val_log_dict)
                    if use_local_record and accelerator.is_main_process:
                        with open(path_local_record, "a", encoding="utf-8") as f:
                            f.write(json.dumps(val_log_dict) + "\n")

                # Check max steps
                if max_steps > 0 and global_step >= max_steps:
                    print_acc("[unlearn_npo.py] Reached max_steps => Stopping.", print_message)
                    break

        if max_steps > 0 and global_step >= max_steps:
            break

    # ----------------------------------------------------------------
    # Final validation (one last check after training loop)
    # ----------------------------------------------------------------
    print_acc("[unlearn_npo.py] Running final validation ...", print_message)
    eng_valid_ce_loss = evaluate_ce_loss(model, eng_valid_loader, tokenizer.pad_token_id, accelerator)
    eng_stats = model.__dict__.get("_temp_metrics", {"ppl": float("inf")})
    eng_valid_ppl = eng_stats["ppl"]

    kor_valid_ce_loss = evaluate_ce_loss(model, kor_valid_loader, tokenizer.pad_token_id, accelerator)
    kor_stats = model.__dict__.get("_temp_metrics", {"ppl": float("inf")})
    kor_valid_ppl = kor_stats["ppl"]

    final_val_log = {
        "val/eng_ce_loss": eng_valid_ce_loss,
        "val/kor_ce_loss": kor_valid_ce_loss,
        "val/eng_ppl": eng_valid_ppl,
        "val/kor_ppl": kor_valid_ppl,
        "train/step": global_step,
        "train/tokens_seen": global_tokens,
    }
    if use_wandb and accelerator.is_main_process:
        wandb.log(final_val_log)
    if use_local_record and accelerator.is_main_process:
        with open(path_local_record, "a", encoding="utf-8") as f:
            f.write(json.dumps(final_val_log) + "\n")

    # ----------------------------------------------------------------
    # Final model save
    # ----------------------------------------------------------------
    if accelerator.is_main_process:
        model.eval()
        unwrapped_model = accelerator.unwrap_model(model)
        save_path = os.path.join(output_dir, "final_model")
        unwrapped_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print_acc(f"[unlearn_npo.py] Model saved to => {save_path}", print_message)


# ----------------------------------------------------------------
# NPO Loss
# ----------------------------------------------------------------
def npo_loss_fn(student_logits, teacher_logits, labels, pad_token_id, beta):
    """
    Negative Preference Optimization (NPO) objective:

      L_NPO = -(2/beta) * E[ log sigma( - beta ( log p_student - log p_teacher ) ) ]

    We do next-token shifting: we measure log p(token_i = correct | token_{i-1} = ...).
    """
    # 1) shift for next-token prediction
    shift_student = student_logits[..., :-1, :].contiguous()   # (batch, seq_len-1, vocab)
    shift_teacher = teacher_logits[..., :-1, :].contiguous()
    # shift labels:
    shift_labels = labels[..., 1:].contiguous()

    # 2) log p for student & teacher
    student_logp = F.log_softmax(shift_student, dim=-1)
    teacher_logp = F.log_softmax(shift_teacher, dim=-1)

    # Flatten
    B, S, V = student_logp.shape
    flat_labels = shift_labels.view(-1)               # (B*S,)
    flat_student = student_logp.view(-1, V)           # (B*S, V)
    flat_teacher = teacher_logp.view(-1, V)

    # valid positions (exclude padding)
    valid_mask = (flat_labels != pad_token_id)

    student_correct_logp = flat_student[torch.arange(B*S, device=flat_student.device), flat_labels]
    teacher_correct_logp = flat_teacher[torch.arange(B*S, device=flat_teacher.device), flat_labels]

    student_correct_logp = student_correct_logp[valid_mask]
    teacher_correct_logp = teacher_correct_logp[valid_mask]

    # neg_log_ratio = log p_teacher - log p_student
    neg_log_ratio = teacher_correct_logp - student_correct_logp

    # final npo = -2/beta * mean[ log sigma( - beta*(neg_log_ratio) ) ]
    #            =  2/beta * mean[ log(1 + exp( beta*(neg_log_ratio) )) ]
    # We'll just do the original definition:
    val = - F.logsigmoid(- beta * neg_log_ratio)
    loss = val.mean() * (2.0 / beta)   # average over valid positions

    return loss


# ----------------------------------------------------------------
# Helper functions (Same as unlearn_ga)
# ----------------------------------------------------------------
def cross_entropy_loss_fn(logits, labels, pad_token_id):
    """
    Standard LM cross-entropy on next-token prediction.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    loss_fct = CrossEntropyLoss(ignore_index=pad_token_id)
    return loss_fct(shift_logits, shift_labels)


def evaluate_ce_loss(model, data_loader, pad_token_id, accelerator):
    """
    Returns average CE, also sets model.__dict__["_temp_metrics"]["ppl"].
    """
    model.eval()
    total_ce_loss = 0.0
    total_count = 0
    total_loss_for_ppl = 0.0
    total_tokens_for_ppl = 0

    for batch in data_loader:
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False
            )
            ce_loss_val = cross_entropy_loss_fn(outputs.logits, batch["input_ids"], pad_token_id)

        bs_local = batch["input_ids"].size(0)
        total_ce_loss += ce_loss_val.item() * bs_local
        total_count += bs_local

        tokens_this_batch = batch["attention_mask"].sum(dim=1).sum().item()
        total_loss_for_ppl += ce_loss_val.item() * tokens_this_batch
        total_tokens_for_ppl += tokens_this_batch

    avg_ce_loss = total_ce_loss / max(total_count, 1)
    if total_tokens_for_ppl > 0:
        avg_nll = total_loss_for_ppl / float(total_tokens_for_ppl)
        ppl_val = math.exp(avg_nll)
    else:
        ppl_val = float('inf')

    model.__dict__["_temp_metrics"] = {"ppl": ppl_val}
    accelerator.print(f"[evaluate_ce_loss] avg CE: {avg_ce_loss:.6f}, PPL: {ppl_val:.3f}")
    model.train()
    return avg_ce_loss


def print_acc(message, condition, end=None):
    """
    Print helper that only prints if condition == True.
    """
    if condition and end is not None:
        print(message, end=end)
    elif condition:
        print(message)
