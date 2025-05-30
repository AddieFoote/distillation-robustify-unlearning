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
from src.utils.loss_functions import cross_entropy_loss_fn, print_acc, custom_makedirs


def unlearn_graddiff(
    model_name,
    forget_train_file,
    retain_train_file,
    eval_fn,
    accelerator,
    output_dir,
    cache_dir,
    dataset_cache_dir,
    ga_gd,
    alpha,
    seed,
    device,
    batch_size,
    gradient_accumulation_steps,
    join_or_subsequence,
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

    use_local_record,
    path_local_record,
    overwrite_ok,
):
    """
    Gradient ascent script using Accelerate on pretokenized JSONL datasets.

    GA+GD mode (ga_gd=True) uses:
      - forget_train_file => negative cross-entropy
      - retain_train_file => normal cross-entropy
    If ga_gd=False, we only do GA on forget_train_file.
    """
    assert 0 < alpha < 1
    print_message = accelerator.is_main_process

    train_args = {**locals()}
    print_acc(f"[graddiff.py] Initiated training with:\n{train_args}", print_message)

    # ----------------------------------------------------------------
    # Setup: seeds, directories, W&B, local record
    # ----------------------------------------------------------------
    custom_makedirs(output_dir, exist_ok=overwrite_ok)
    random.seed(seed)
    torch.manual_seed(seed)

    if use_wandb and accelerator.is_main_process:
        wandb.init(project=wandb_project, name=wandb_run_name, config=train_args)

    if use_local_record and accelerator.is_main_process:
        custom_makedirs(path_local_record, exist_ok=overwrite_ok)

    # ----------------------------------------------------------------
    # Load model + tokenizer
    # ----------------------------------------------------------------
    print_acc(f"[graddiff.py] Loading model {model_name}", print_message)
    model_config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        attn_implementation='eager'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ----------------------------------------------------------------
    # Load & process FORGET dataset (Required for GA)
    # ----------------------------------------------------------------
    if not forget_train_file.strip():
        raise ValueError("forget_train_file is empty => must provide a dataset to 'forget'")
    print_acc("[graddiff.py] Loading 'forget' dataset", print_message)
    forget_ds = load_dataset(
        "json",
        data_files=forget_train_file,
        split="train",
        cache_dir=dataset_cache_dir
    )
    print_acc(f"[graddiff.py] Forget dataset size: {len(forget_ds)}", print_message)
    sample_text = forget_ds[0]["text"].replace('\n', ' ')
    print_acc(f'[graddiff.py] Sample forget text: "{sample_text[:200]}..."', print_message)
    # ------------------------------------------------------------
    # Process for sequence length
    # If join_or_subsequence, form sequences of exactly max_length by joining multiple or using subsequences
    # else filter for only those less than max_length
    # ------------------------------------------------------------
    train_ds_list, message = make_sequence_length(train_ds_list=[forget_ds], tokenizer=tokenizer, max_length=max_length, join_or_subsequence=join_or_subsequence)
    print_acc(message, print_message)
    forget_ds = train_ds_list[0]
    # Print size
    print_acc(f"[graddiff.py] Forget dataset size: {len(forget_ds)}", print_message)
    forget_loader = DataLoader(
        forget_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=max_length)
    )

    # ----------------------------------------------------------------
    # Load & filter RETAIN dataset (Used only if ga_gd=True)
    # ----------------------------------------------------------------
    if ga_gd and retain_train_file.strip():
        print_acc("[graddiff.py] GA+GD => loading 'retain' dataset", print_message)
        retain_ds = load_dataset(
            "json",
            data_files=retain_train_file,
            split="train",
            cache_dir=dataset_cache_dir
        )
        print_acc(f"[graddiff.py] Retain dataset size: {len(retain_ds)}", print_message)
        sample_text_r = retain_ds[0]["text"].replace('\n', ' ')
        print_acc(f'[graddiff.py] Sample retain text: "{sample_text_r[:200]}..."', print_message)

        # ------------------------------------------------------------
        # Process for sequence length
        # If join_or_subsequence, form sequences of exactly max_length by joining multiple or using subsequences
        # else filter for only those less than max_length
        # ------------------------------------------------------------
        train_ds_list, message = make_sequence_length(train_ds_list=[retain_ds], tokenizer=tokenizer, max_length=max_length, join_or_subsequence=join_or_subsequence)
        print_acc(message, print_message)
        retain_ds = train_ds_list[0]
        retain_loader = DataLoader(
            retain_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=max_length)
        )
    else:
        # either ga_gd=False, or retain_train_file is empty => No "retain" data
        retain_loader = None

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
    print_acc(f"[graddiff.py] {steps_per_epoch} steps per epoch, total steps: {total_steps}", print_message)

    # ----------------------------------------------------------------
    # Optimizer + LR scheduler
    # ----------------------------------------------------------------
    print_acc(f"[graddiff.py] Using AdamW optimizer, LR={learning_rate}, weight_decay={weight_decay}", print_message)
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
    if ga_gd and retain_loader is not None:
        retain_loader = accelerator.prepare(retain_loader)

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    print_acc("[graddiff.py] Starting training", print_message)

    # Initial validation before training starts
    print_acc("[graddiff.py] Running initial validation before training...", print_message)
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
    retain_loader_iter = iter(retain_loader) if (ga_gd and retain_loader) else None
    for epoch in range(epochs):
        print_acc(f"[graddiff.py] Epoch {epoch+1}/{epochs}", print_message)
        model.train()

        for step_in_epoch in range(steps_per_epoch):
            # 1) Get forget batch
            try:
                forget_batch = next(forget_loader_iter)
            except StopIteration:
                forget_loader_iter = iter(forget_loader)
                forget_batch = next(forget_loader_iter)

            # Forward pass on forget data (negative CE)
            outputs_forget = model(
                input_ids=forget_batch["input_ids"],
                attention_mask=forget_batch["attention_mask"]
            )
            ce_loss_forget = cross_entropy_loss_fn(
                outputs_forget.logits,
                forget_batch["input_ids"],
                tokenizer.pad_token_id
            )
            loss_forget = -1.0 * ce_loss_forget  # gradient ascent

            # 2) If GA+GD => also get retain batch (normal CE)
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
                
                # total_loss = (loss_forget + ce_loss_retain) / gradient_accumulation_steps
                # with alpha
                total_loss = (((1 - alpha) * loss_forget) + (alpha * ce_loss_retain)) / gradient_accumulation_steps

                # Count tokens
                tokens_forget = forget_batch["attention_mask"].sum().detach()
                tokens_forget = accelerator.gather(tokens_forget).sum().item()
                tokens_retain = retain_batch["attention_mask"].sum().detach()
                tokens_retain = accelerator.gather(tokens_retain).sum().item()
                global_tokens += (tokens_forget + tokens_retain)
                
            else:
                total_loss = loss_forget / gradient_accumulation_steps
                tokens_this_batch = forget_batch["attention_mask"].sum().detach()
                tokens_this_batch = accelerator.gather(tokens_this_batch).sum().item()
                global_tokens += tokens_this_batch
            
            # Backprop
            accelerator.backward(total_loss)
            

            if (step_in_epoch + 1) % gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), gradient_clipping_threshold)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step == 1 or global_step % 5 == 0:
                    print_acc(
                        f"[graddiff.py] Epoch {epoch+1}/{epochs}, "
                        f"Step {global_step}/{total_steps}, GA{'+GD' if ga_gd else ''} "
                        f"=> CE_forget: {ce_loss_forget:.6f}",
                        print_message
                    )
                    if ga_gd:
                        print_acc(f"[graddiff.py] Retain CE: {ce_loss_retain:.6f}", print_message)

                train_log_dict = {
                    "train/ce_loss_forget": ce_loss_forget.item(),
                    "train/step": global_step,
                    "train/tokens_seen": global_tokens,
                    "train/lr": scheduler.get_last_lr()[0],
                }
                if ga_gd:
                    train_log_dict["train/ce_loss_retain"] = ce_loss_retain.item()

                if use_wandb and accelerator.is_main_process:
                    wandb.log(train_log_dict)
                if use_local_record and accelerator.is_main_process:
                    with open(path_local_record, "a", encoding="utf-8") as f:
                        f.write(json.dumps(train_log_dict) + "\n")
                # Validation
                # (Perform if first step, or modulo validation_steps or last step)
                if global_step == 1 or global_step % validation_steps == 0 or  max_steps > 0 and global_step >= max_steps:
                    print_acc("[graddiff.py] Running validation ...", print_message)
                    val_log_dict = eval_fn(model, print_results=True)
                    val_log_dict["train/step"] = global_step
                    val_log_dict["train/tokens_seen"] = global_tokens

                    if use_wandb and accelerator.is_main_process:
                        wandb.log(val_log_dict)
                    if use_local_record and accelerator.is_main_process:
                        with open(path_local_record, "a", encoding="utf-8") as f:
                            f.write(json.dumps(val_log_dict) + "\n")
            if max_steps > 0 and global_step >= max_steps:
                print_acc("[graddiff.py] Reached max_steps => Stopping.", print_message)
                break

        if max_steps > 0 and global_step >= max_steps:
            break
    # Final validation after training is complete
    print_acc("[graddiff.py] Running final validation after training...", print_message)
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
        print_acc(f"[graddiff.py] Model saved to => {save_path}", print_message)



