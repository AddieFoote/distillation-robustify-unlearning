import os
import random
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, interleave_datasets
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding
)
import wandb
import math
import json
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.loss_functions import cross_entropy_loss_fn, print_acc
from utils.process_datasets import make_sequence_length

torch.set_float32_matmul_precision('high')

def train(
    model_name,
    train_files,
    interleave_probs,
    output_dir,
    cache_dir,
    dataset_cache_dir,

    eval_fn,
    validation_steps,
    accelerator,

    seed,
    device,
    batch_size,
    join_or_subsequence,
    gradient_accumulation_steps,
    epochs,
    learning_rate,     
    max_steps,         
    num_warmup_steps,
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
):
    """
    Training script using Accelerate on pretokenized JSONL datasets.
    """
    print_message = accelerator.is_main_process

    train_args = {**locals()}
    print_acc(f"[pretrain.[[py] Initiated training with:\n{train_args}", print_message)


    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)

    if use_wandb and accelerator.is_main_process:
        wandb.login(key=wandb_api_key)
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=train_args,
        )
        
    if use_local_record and accelerator.is_main_process:
        local_dir = os.path.dirname(path_local_record)
        os.makedirs(local_dir, exist_ok=True)




    print_acc(f"[pretrain.py] Loading model {model_name}", print_message)
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
    # Load training datasets
    # ----------------------------------------------------------------
    train_ds_list = []
    for file in train_files:
        print_acc(f"[pretrain.py] Loading train dataset from {file}", print_message)
        train_ds = load_dataset("json", data_files=file, split="train", cache_dir=dataset_cache_dir)
        train_ds_list.append(train_ds)
   
    # If join_or_subsequence, form sequences of exactly max_length 
    # by joining multiple or using subsequences
    # else filter for only those less than max_length
    train_ds_list, message = make_sequence_length(train_ds_list=train_ds_list, tokenizer=tokenizer, max_length=max_length, join_or_subsequence=join_or_subsequence)
    print_acc(message, print_message)
    
    # If both were given, interleave with equal probabilities
    # If only one is present, that's our train_ds
    if len(train_ds_list) == 0:
        raise ValueError("No training datasets were provided!")
    elif len(train_ds_list) == 1:
        train_ds = train_ds_list[0]
    else:
        # Interleave them with given probability
        assert len(interleave_probs) == len(train_ds_list)
        from datasets import interleave_datasets
        lengths = [len(item) for item in train_ds_list]
        print_acc(f"[pretrain.py] Interleave: {interleave_probs} for datasets with lengths {lengths}", print_message)
        
        train_ds = interleave_datasets(train_ds_list, probabilities=interleave_probs, seed=seed)

    print_acc(f"[pretrain.py] Train dataset size: {len(train_ds)}", print_message)

    # Print 3 samples of training text
    for i in range(3): 
        sample_ids = train_ds[i]["input_ids"]
        sample_text = tokenizer.decode(sample_ids, skip_special_tokens=False) #.replace('\n', ' ')
        print_acc(f'[pretrain.py] Sample training text {i}: "{sample_text[:200]}..."', print_message)

    # Create DataLoaders using data collator.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=data_collator)

    steps_per_epoch = len(train_loader)
    effective_steps_per_epoch = steps_per_epoch // gradient_accumulation_steps
    total_steps = effective_steps_per_epoch * epochs
    if max_steps > 0:
        total_steps = min(total_steps, max_steps)
    print_acc(f"[pretrain.py] {steps_per_epoch} steps per epoch, {total_steps // accelerator.num_processes} steps total", print_message)

    print_acc(f"[pretrain.py] Using AdamW optimizer, LR={learning_rate}, weight_decay={weight_decay}", print_message)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Choose the learning rate scheduler.
    if scheduler_type == "linear":
        # Custom linear schedule: warmup then linear decay from learning_rate to min_lr.
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            else:
                progress = float(current_step - num_warmup_steps) / float(max(1, total_steps - num_warmup_steps))
                # Linearly decay from 1 to (min_lr/learning_rate)
                return (1.0 - progress) * (1.0 - (min_lr / learning_rate)) + (min_lr / learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_type == "cosine":
        # Custom cosine schedule: warmup then cosine decay from learning_rate to min_lr.
        def cosine_lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            else:
                progress = float(current_step - num_warmup_steps) / float(max(1, total_steps - num_warmup_steps))
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                # Decay factor ranges from 1 to (min_lr/learning_rate)
                return cosine_decay * (1.0 - (min_lr / learning_rate)) + (min_lr / learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lr_lambda)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    # Prepare everything with accelerator.
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    print_acc("[pretrain.py] Starting training", print_message)
    global_step = 0
    global_tokens = 0

    for epoch in range(epochs):
        print_acc(f"[pretrain.py] Epoch {epoch+1}/{epochs}", print_message)
        model.train()

        for step_in_epoch, batch in enumerate(train_loader):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            ce_loss = cross_entropy_loss_fn(outputs.logits, batch["input_ids"], tokenizer.pad_token_id)
            loss = ce_loss / gradient_accumulation_steps
            accelerator.backward(loss)

            tokens_this_batch = batch["attention_mask"].sum().detach()
            tokens_this_batch = accelerator.gather(tokens_this_batch).sum().item()
            global_tokens += tokens_this_batch

            if (step_in_epoch + 1) % gradient_accumulation_steps == 0:
                # Apply gradient clipping.
                accelerator.clip_grad_norm_(model.parameters(), gradient_clipping_threshold)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                print_acc(f"[pretrain.py] Epoch {epoch+1}/{epochs}, Step {global_step}/{total_steps // accelerator.num_processes}, CE loss: {ce_loss:.6f}", print_message)

                # Prepare dictionary for logging
                train_log_dict = {
                    "train/ce_loss": ce_loss.item(),
                    "train/step": global_step,
                    "train/tokens_seen": global_tokens,
                    "train/lr": scheduler.get_last_lr()[0]
                }

                # W&B logging
                if use_wandb and accelerator.is_main_process:
                    wandb.log(train_log_dict)

                # Local JSONL logging
                if use_local_record and accelerator.is_main_process:
                    with open(path_local_record, "a", encoding="utf-8") as f:
                        f.write(json.dumps(train_log_dict) + "\n")

                # Validation: Calls provided eval function which returns a dictionary with eval name keys and results values
                if (global_step == 1 or global_step % validation_steps == 0 or
                   (max_steps > 0 and global_step == max_steps)):
                    print_acc("[pretrain.py] Running validation ...", print_message)
                    val_log_dict = eval_fn(model, print_results=True) 
                    val_log_dict["train/step"] = global_step
                    val_log_dict["train/tokens_seen"] = global_tokens
                    
                    

                    if use_wandb and accelerator.is_main_process:
                        wandb.log(val_log_dict)

                    if use_local_record and accelerator.is_main_process:
                        with open(path_local_record, "a", encoding="utf-8") as f:
                            f.write(json.dumps(val_log_dict) + "\n")

                # Saving checkpoints
                if save_checkpoint_steps > 0 and (global_step % save_checkpoint_steps == 0):
                    if accelerator.is_main_process:
                        checkpoint_path = os.path.join(output_dir, f"checkpoint-step{global_step}")
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(checkpoint_path)
                        tokenizer.save_pretrained(checkpoint_path)
                        print_acc(f"[pretrain.py] Saved checkpoint => {checkpoint_path}", print_message)

                # Early stopping based on max_steps
                if max_steps > 0 and global_step >= max_steps:
                    print_acc("[pretrain.py] Reached max_steps => Stopping.", print_message)
                    break

        if max_steps > 0 and global_step >= max_steps:
            break

    # Final model save
    if accelerator.is_main_process:
        model.eval()
        unwrapped_model = accelerator.unwrap_model(model)
        save_path = os.path.join(output_dir, "final_model")
        unwrapped_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print_acc(f"[pretrain.py] Model saved to => {save_path}", print_message)

    wandb.finish()

if __name__ == "__main__":
    train()
