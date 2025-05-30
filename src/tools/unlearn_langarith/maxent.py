import os
import random
import math
import json
import torch
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
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.utils.process_datasets import make_sequence_length
from src.utils.loss_functions import cross_entropy_loss_fn, print_acc

# Import RepNoise components
from src.utils.repnoise_loss import rep_noise_loss, register_activation_hook, MMD_loss

# Import SAM utilities
from src.utils.sam_utils import compute_sam_perturbation, apply_perturbation

def unlearn_maxent(
    model_name,
    forget_train_file,
    retain_train_file,
    eval_fn,
    accelerator,
    output_dir,
    cache_dir,
    dataset_cache_dir,
    join_or_subsequence,

    use_retain,
    
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

    use_wandb,
    wandb_project,
    wandb_run_name,
    wandb_api_key,

    use_local_record,
    path_local_record,

    # Add balance_alpha parameter to control the forget-retain trade-off for maxent
    balance_alpha=1.0,

    # RepNoise parameters
    use_repnoise=False,         # Option to turn RepNoise on/off
    repnoise_beta=0.001,        # Beta parameter for RepNoise loss
    repnoise_alpha=1.0,         # Alpha parameter for RepNoise loss
    
    # SAM parameters
    use_sam=False,              # Option to turn SAM on/off
    sam_rho=0.05,               # Rho parameter (perturbation size) for SAM
):
    """
    Uniform Forget script using Accelerate on pretokenized JSONL datasets.
    
    - "Forget" dataset => uniform_forget_loss_fn, i.e., push model logits to uniform 
      via forward KL with teacher_logits=ones.
    - "Retain" dataset (if use_retain=True) => normal cross-entropy (to preserve knowledge).
    - RepNoise (if use_repnoise=True) => adds representation noising for more robust unlearning.
    - SAM (if use_sam=True) => Sharpness-Aware Minimization for more robust unlearning.
    - balance_alpha => Controls the balance between forget and retain losses.
                      Higher values emphasize retain loss, lower values emphasize forget loss.
                      Set to 0 to completely ignore retain loss.
    """
    accelerator = Accelerator()
    print_message = accelerator.is_main_process

    # Validate RepNoise settings
    if use_repnoise and (not use_retain or not retain_train_file.strip()):
        print_acc("[maxent.py] WARNING: RepNoise requires both forget and retain datasets. Disabling RepNoise.", print_message)
        use_repnoise = False
    
    # Validate SAM settings
    if use_sam and (not use_retain or not retain_train_file.strip()):
        print_acc("[maxent.py] WARNING: SAM requires both forget and retain datasets. Disabling SAM.", print_message)
        use_sam = False

    train_args = {**locals()}
    print_acc(f"[maxent.py] Initiated training with:\n{train_args}", print_message)

    # ----------------------------------------------------------------
    # Setup: seeds, directories, W&B, local record
    # ----------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)

    if use_wandb and accelerator.is_main_process:
        wandb.login(key=wandb_api_key)
        wandb.init(project=wandb_project, name=wandb_run_name, config=train_args)

    if use_local_record and accelerator.is_main_process:
        local_dir = os.path.dirname(path_local_record)
        os.makedirs(local_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # Load model + tokenizer
    # ----------------------------------------------------------------
    print_acc(f"[maxent.py] Loading model {model_name}", print_message)
    model_config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        attn_implementation='eager',
        torch_dtype=torch.bfloat16
    )
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
    # Load FORGET dataset (Required for uniform forget)
    # ----------------------------------------------------------------
    if not forget_train_file.strip():
        raise ValueError("forget_train_file is empty => must provide a dataset to 'forget'")
    print_acc("[maxent.py] Loading 'forget' dataset", print_message)
    forget_ds = load_dataset(
        "json",
        data_files=forget_train_file,
        split="train",
        cache_dir=dataset_cache_dir
    )
    print_acc(f"[maxent.py] Forget dataset size: {len(forget_ds)}", print_message)
    sample_text = forget_ds[0]["text"].replace('\n', ' ')
    print_acc(f'[maxent.py] Sample forget text: "{sample_text[:200]}..."', print_message)

    # ------------------------------------------------------------
    # Process for sequence length
    # If join_or_subsequence, form sequences of exactly max_length by joining multiple or using subsequences
    # else filter for only those less than max_length
    # ------------------------------------------------------------
    forget_ds_list, message = make_sequence_length(train_ds_list=[forget_ds], tokenizer=tokenizer, max_length=max_length, join_or_subsequence=join_or_subsequence)
    print_acc(message, print_message)
    forget_ds = forget_ds_list[0]
    forget_loader = DataLoader(
        forget_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=max_length)
    )

    # ----------------------------------------------------------------
    # Load Retain Dataset (Used only if use_retain=True)
    # ----------------------------------------------------------------
    if use_retain and retain_train_file.strip():
        print_acc("[maxent.py] => loading 'retain' dataset", print_message)
        retain_ds = load_dataset(
            "json",
            data_files=retain_train_file,
            split="train",
            cache_dir=dataset_cache_dir
        )
        print_acc(f"[maxent.py] Retain dataset size: {len(retain_ds)}", print_message)
        sample_text_r = retain_ds[0]["text"].replace('\n', ' ')
        print_acc(f'[maxent.py] Sample retain text: "{sample_text_r[:200]}..."', print_message)
        # ------------------------------------------------------------
        # Process for sequence length
        # If join_or_subsequence, form sequences of exactly max_length by joining multiple or using subsequences
        # else filter for only those less than max_length
        # ------------------------------------------------------------
        retain_ds_list, message = make_sequence_length(train_ds_list=[retain_ds], tokenizer=tokenizer, max_length=max_length, join_or_subsequence=join_or_subsequence)
        print_acc(message, print_message)
        retain_ds = retain_ds_list[0]
        retain_loader = DataLoader(
            retain_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=max_length)
        )
    else:
        # either use_retain=False, or retain_train_file is empty => No "retain" data
        retain_loader = None

    # ----------------------------------------------------------------
    # Determine steps
    # ----------------------------------------------------------------
    steps_per_epoch_forget = len(forget_loader)
    if use_retain and retain_loader is not None:
        steps_per_epoch_retain = len(retain_loader)
        steps_per_epoch = max(steps_per_epoch_forget, steps_per_epoch_retain)
    else:
        steps_per_epoch = steps_per_epoch_forget

    effective_steps_per_epoch = steps_per_epoch // gradient_accumulation_steps
    total_steps = effective_steps_per_epoch * epochs
    if max_steps > 0:
        total_steps = min(total_steps, max_steps)
    print_acc(f"[maxent.py] {steps_per_epoch} steps per epoch, total steps: {total_steps}", print_message)

    # ----------------------------------------------------------------
    # Optimizer + LR scheduler
    # ----------------------------------------------------------------
    print_acc(f"[maxent.py] Using AdamW optimizer, LR={learning_rate}, weight_decay={weight_decay}", print_message)
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
    model, optimizer, forget_loader, scheduler = accelerator.prepare(
        model, optimizer, forget_loader, scheduler
    )
    if use_retain and retain_loader is not None:
        retain_loader = accelerator.prepare(retain_loader)

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    print_acc("[maxent.py] Starting training", print_message)
    
    # Log balance_alpha setting
    if balance_alpha != 1.0:
        print_acc(f"[maxent.py] Using balance_alpha={balance_alpha} to weight retain vs forget loss", print_message)

    # Initial validation before training
    print_acc("[maxent.py] Running initial validation before training...", print_message)
    initial_val_log_dict = eval_fn(model, print_results=True)
    initial_val_log_dict["train/step"] = 0
    initial_val_log_dict["train/tokens_seen"] = 0
    if use_wandb and accelerator.is_main_process:
        wandb.log(initial_val_log_dict)
    if use_local_record and accelerator.is_main_process:
        with open(path_local_record, "a", encoding="utf-8") as f:
            f.write(json.dumps(initial_val_log_dict) + "\n")

    global_step = 0
    global_tokens = 0

    forget_loader_iter = iter(forget_loader)
    retain_loader_iter = iter(retain_loader) if (use_retain and retain_loader) else None

    for epoch in range(epochs):
        print_acc(f"[maxent.py] Epoch {epoch+1}/{epochs}", print_message)
        model.train()

        for step_in_epoch in range(steps_per_epoch):
            # 1) Get forget batch
            try:
                forget_batch = next(forget_loader_iter)
            except StopIteration:
                forget_loader_iter = iter(forget_loader)
                forget_batch = next(forget_loader_iter)

            # Forward pass on forget data => uniform_forget_loss_fn
            outputs_forget = model(
                input_ids=forget_batch["input_ids"],
                attention_mask=forget_batch["attention_mask"]
            )
            uniform_ce_forget = uniform_forget_loss_fn(
                outputs_forget.logits,
                forget_batch["input_ids"],
                tokenizer.pad_token_id,
                loss_mask = forget_batch.get("loss_mask")
            )
            
            # 2) If use_retain => also get retain batch (normal CE)
            if use_retain and retain_loader_iter is not None:
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
                    tokenizer.pad_token_id,
                    loss_mask = retain_batch.get("loss_mask")
                )

                # If RepNoise is enabled and we have both forget and retain batches
                if use_repnoise:
                    # Map forget_batch -> harmful_batch, retain_batch -> harmless_batch
                    repnoise_loss = rep_noise_loss(
                        model=model,
                        harmful_batch=forget_batch,
                        harmless_batch=retain_batch,
                        beta=repnoise_beta,
                        alpha=repnoise_alpha
                    )
                    # Add RepNoise to the loss and apply balance_alpha to retain loss
                    total_loss = (uniform_ce_forget + balance_alpha * ce_loss_retain + repnoise_loss) / gradient_accumulation_steps
                else:
                    # Apply balance_alpha to the retain loss
                    total_loss = (uniform_ce_forget + balance_alpha * ce_loss_retain) / gradient_accumulation_steps

                # Count tokens
                tokens_forget = forget_batch["attention_mask"].sum().detach()
                tokens_forget = accelerator.gather(tokens_forget).sum().item()
                tokens_retain = retain_batch["attention_mask"].sum().detach()
                tokens_retain = accelerator.gather(tokens_retain).sum().item()
                global_tokens += (tokens_forget + tokens_retain)
            else:
                total_loss = uniform_ce_forget / gradient_accumulation_steps
                tokens_this_batch = forget_batch["attention_mask"].sum().detach()
                tokens_this_batch = accelerator.gather(tokens_this_batch).sum().item()
                global_tokens += tokens_this_batch

            # If SAM is enabled and we're at an update step
            if use_sam and use_retain and (step_in_epoch + 1) % gradient_accumulation_steps == 0:
                # Step 1: Compute perturbation from current loss
                perturbation = compute_sam_perturbation(model, total_loss, sam_rho)
                
                # Step 2: Apply perturbation
                apply_perturbation(perturbation, apply=True)
                
                # Step 3: Zero gradients before computing loss with perturbed model
                model.zero_grad()
                
                # Step 4: Recompute forget loss on perturbed model
                outputs_forget_perturbed = model(
                    input_ids=forget_batch["input_ids"],
                    attention_mask=forget_batch["attention_mask"]
                )
                uniform_ce_forget_perturbed = uniform_forget_loss_fn(
                    outputs_forget_perturbed.logits,
                    forget_batch["input_ids"],
                    tokenizer.pad_token_id,
                    loss_mask=forget_batch.get("loss_mask")
                )
                
                # Step 5: Recompute retain loss on perturbed model
                outputs_retain_perturbed = model(
                    input_ids=retain_batch["input_ids"],
                    attention_mask=retain_batch["attention_mask"]
                )
                ce_loss_retain_perturbed = cross_entropy_loss_fn(
                    outputs_retain_perturbed.logits,
                    retain_batch["input_ids"],
                    tokenizer.pad_token_id,
                    loss_mask=retain_batch.get("loss_mask")
                )
                
                # Step 6: Recalculate total loss with perturbed parameters and apply balance_alpha
                if use_repnoise:
                    repnoise_loss_perturbed = rep_noise_loss(
                        model=model,
                        harmful_batch=forget_batch,
                        harmless_batch=retain_batch,
                        beta=repnoise_beta,
                        alpha=repnoise_alpha
                    )
                    total_loss_perturbed = (uniform_ce_forget_perturbed + balance_alpha * ce_loss_retain_perturbed + repnoise_loss_perturbed) / gradient_accumulation_steps
                else:
                    total_loss_perturbed = (uniform_ce_forget_perturbed + balance_alpha * ce_loss_retain_perturbed) / gradient_accumulation_steps
                
                # Step 7: Compute gradients at perturbed position
                accelerator.backward(total_loss_perturbed)
                
                # Step 8: Remove perturbation to restore original parameters
                apply_perturbation(perturbation, apply=False)
                
                # Step 9: Apply optimizer step with SAM gradients
                accelerator.clip_grad_norm_(model.parameters(), gradient_clipping_threshold)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            else:
                # Standard backprop for non-SAM or non-update steps
                accelerator.backward(total_loss)
                
                if (step_in_epoch + 1) % gradient_accumulation_steps == 0:
                    accelerator.clip_grad_norm_(model.parameters(), gradient_clipping_threshold)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            # Logging every few steps (after parameter updates)
            if global_step > 0 and (global_step == 1 or global_step % 5 == 0):
                msg = (
                    f"[maxent.py] Epoch {epoch+1}/{epochs}, Step {global_step}/{total_steps}, "
                    f"Uniform-Forget => CE_forget(uniform): {uniform_ce_forget:.6f}"
                )
                print_acc(msg, print_message)

                train_log_dict = {
                    "train/uf_loss_forget": uniform_ce_forget.item(),
                    "train/step": global_step,
                    "train/tokens_seen": global_tokens,
                    "train/lr": scheduler.get_last_lr()[0],
                }

                if use_retain and retain_loader_iter is not None:
                    train_log_dict["train/ce_loss_retain"] = ce_loss_retain.item()
                    train_log_dict["train/balance_alpha"] = balance_alpha
                    train_log_dict["train/weighted_ce_loss_retain"] = (balance_alpha * ce_loss_retain).item()
                    print_acc(f"[maxent.py] Retain CE: {ce_loss_retain:.6f} (weighted: {balance_alpha * ce_loss_retain:.6f})", print_message)
                    
                    # Log RepNoise loss if enabled
                    if use_repnoise:
                        train_log_dict["train/repnoise_loss"] = repnoise_loss.item()
                        print_acc(f"[maxent.py] RepNoise Loss: {repnoise_loss:.6f}", print_message)
                        
                    # Log SAM info if enabled
                    if use_sam:
                        train_log_dict["train/sam_enabled"] = True
                        train_log_dict["train/sam_rho"] = sam_rho
                        print_acc(f"[maxent.py] SAM enabled with rho={sam_rho}", print_message)

                if use_wandb and accelerator.is_main_process:
                    wandb.log(train_log_dict)
                if use_local_record and accelerator.is_main_process:
                    with open(path_local_record, "a", encoding="utf-8") as f:
                        f.write(json.dumps(train_log_dict) + "\n")

                # Validation
                # (Perform if first step, or modulo validation_steps or last step)
                if global_step == 1 or global_step % validation_steps == 0 or (max_steps > 0 and global_step >= max_steps):
                    print_acc("[maxent.py] Running validation ...", print_message)
                    val_log_dict = eval_fn(model, print_results=True)
                    val_log_dict["train/step"] = global_step
                    val_log_dict["train/tokens_seen"] = global_tokens
                    if use_wandb and accelerator.is_main_process:
                        wandb.log(val_log_dict)
                    if use_local_record and accelerator.is_main_process:
                        with open(path_local_record, "a", encoding="utf-8") as f:
                            f.write(json.dumps(val_log_dict) + "\n")

                # Check max steps
                if max_steps > 0 and global_step >= max_steps:
                    print_acc("[maxent.py] Reached max_steps => Stopping.", print_message)
                    break

        if max_steps > 0 and global_step >= max_steps:
            break

    # Final validation after all training
    print_acc("[maxent.py] Running final validation after training completion...", print_message)
    final_val_log_dict = eval_fn(model, print_results=True)
    final_val_log_dict["train/step"] = global_step
    final_val_log_dict["train/tokens_seen"] = global_tokens
    if use_wandb and accelerator.is_main_process:
        wandb.log(final_val_log_dict)
    if use_local_record and accelerator.is_main_process:
        with open(path_local_record, "a", encoding="utf-8") as f:
            f.write(json.dumps(final_val_log_dict) + "\n")
            
    # ----------------------------------------------------------------
    # Final model save
    # ----------------------------------------------------------------
    if accelerator.is_main_process:
        model.eval()
        unwrapped_model = accelerator.unwrap_model(model)
        save_path = os.path.join(output_dir, "final_model")
        unwrapped_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print_acc(f"[maxent.py] Model saved to => {save_path}", print_message)

# ----------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------
def uniform_forget_loss_fn(logits, input_ids, pad_token_id, loss_mask=None):
    """
    Replaces the uniform cross-entropy with a forward KL approach using teacher_logits=all ones.
    Minimizing this forward KL forces the model to match a uniform teacher distribution.
    """
    import torch.nn.functional as F

    # 1) Shift for next-token prediction
    # shape => [batch, seq_len-1, vocab_size]
    student_logits = logits[..., :-1, :].contiguous()

    # 2) Teacher logits => all ones => uniform after softmax
    teacher_logits = torch.ones_like(student_logits)  # same shape

    # 3) Convert teacher + student to log probs
    #    forward KL = sum p_teacher(log p_teacher - log p_student)
    #    where p_teacher = softmax(teacher_logits)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    student_log_probs = F.log_softmax(student_logits, dim=-1)

    # 4) We'll ignore padded positions. Shift input_ids as well => next token
    shift_labels = input_ids[..., 1:].contiguous()     # [batch, seq_len-1]
    valid_mask = (shift_labels != pad_token_id)
    
    # 5) forward KL per position = sum_over_vocab [ p_teacher(v) * (log p_teacher(v) - log p_student(v)) ]
    kl_per_position = teacher_probs * (teacher_log_probs - student_log_probs)
    kl_per_position = kl_per_position.sum(dim=-1)  # sum over vocab => shape [batch, seq_len-1]

    # 6) Mask out invalid positions
    kl_per_position = kl_per_position * valid_mask

    # 7) If there is a loss maks, apply mask (ei mask out question)
    if loss_mask is not None:
        kl_per_position *= loss_mask

    # 8) Average over valid positions
    denom = valid_mask.sum().clamp(min=1)
    kl = kl_per_position.sum() / denom

    return kl