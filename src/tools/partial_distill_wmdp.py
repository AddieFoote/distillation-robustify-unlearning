import os
import random
import math
import json

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, interleave_datasets
from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
)
import wandb
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.loss_functions import forward_kl_loss_fn, print_acc, custom_makedirs
from utils.validation_functions import evaluate_kd_ce_ppl
from utils.process_datasets import make_sequence_length



def partial_distill(
    teacher_model_name,
    student_model_name,

    train_files,
    interleave_probs,
    stopping_strategy,
    join_or_subsequence,
    eval_fn,
    stop_cond_fn,
    accelerator,

    output_dir,
    cache_dir,
    dataset_cache_dir,

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

    use_local_record,
    path_local_record,

    overwrite_ok,
    noise_alpha=0.0,
    noise_beta=0.0,
    shrink_perturb_repeat=False,  
    compile_mode=None,
    layers_to_train=['all'],
    layer_types_to_train=['all'],
    base_teacher_name=None,
    switch_teachers=False,
    use_base_teacher_percent=0,
    use_activation_loss=False,
    both_losses_act_loss_multiplier=None,
):
    
    """
    Distillation script using Accelerate. Replaces standard CE with forward KL (KL(teacher||student)).
    We'll merge eng_train_file and kor_train_file if both are provided, else use whichever is non-empty.
    We do final validation after the loop.
    """
    # ------------------------------------------------------------
    # Accelerator, logging, seeding
    # ------------------------------------------------------------
    print_message = accelerator.is_main_process
    assert 0 <= use_base_teacher_percent <= 1
    assert not (use_base_teacher_percent != 0 and use_activation_loss)
    assert both_losses_act_loss_multiplier is None or (use_activation_loss and both_losses_act_loss_multiplier > 0)

    distill_args = {**locals()}
    print_acc(f"[serum.py] Initiated distillation with:\n{distill_args}", print_message)

    torch.set_default_dtype(torch.bfloat16)
    
    custom_makedirs(output_dir, exist_ok=overwrite_ok)

    random.seed(seed)
    torch.manual_seed(seed)


    if use_wandb and accelerator.is_main_process:
        wandb.init(project=wandb_project, name=wandb_run_name, config=distill_args)

    if use_local_record and accelerator.is_main_process:
        custom_makedirs(path_local_record, exist_ok=overwrite_ok)



    # ------------------------------------------------------------
    # Load teacher + student
    # ------------------------------------------------------------
    attention_type = 'eager' if 'gemma' in teacher_model_name.lower() or 'wmdp' in teacher_model_name.lower() else 'sdpa'
    print_acc(f"[serum.py] Loading teacher model {teacher_model_name} with attention = {attention_type}", print_message)
    teacher_config = AutoConfig.from_pretrained(teacher_model_name, cache_dir=cache_dir)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        cache_dir=cache_dir,
        attn_implementation=attention_type,
        torch_dtype=torch.bfloat16
    )
    
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, cache_dir=cache_dir)
    if teacher_tokenizer.pad_token_id is None:
        teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id

    if base_teacher_name is not None:
        attention_type = 'eager' if 'gemma' in base_teacher_name.lower() or 'wmdp' in base_teacher_name.lower() else 'sdpa'
        print_acc(f"[serum.py] Loading base teacher model {base_teacher_name} with attention = {attention_type}", print_message)
        base_teacher_config = AutoConfig.from_pretrained(base_teacher_name, cache_dir=cache_dir)
        base_teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_model_name,
            cache_dir=cache_dir,
            attn_implementation=attention_type,
            torch_dtype=torch.bfloat16
        )

    attention_type = 'eager' if 'gemma' in student_model_name.lower() or 'wmdp' in student_model_name.lower() else 'sdpa'
    print_acc(f"[serum.py] Loading student model {student_model_name} with attention = {attention_type}", print_message)
    student_config = AutoConfig.from_pretrained(student_model_name, cache_dir=cache_dir)
    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        cache_dir=cache_dir,
        attn_implementation=attention_type,
        torch_dtype=torch.bfloat16
    )
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name, cache_dir=cache_dir)
    if student_tokenizer.pad_token_id is None:
        student_tokenizer.pad_token_id = student_tokenizer.eos_token_id

    # We'll use the teacher's tokenizer for dataset processing (assuming same vocab).
    tokenizer = teacher_tokenizer

    # ------------------------------------------------------------
    # Shrink + Perturb (one-time, if requested)
    # ------------------------------------------------------------
    if not ("all" in layer_types_to_train and "all" in layers_to_train):
        for param in student_model.parameters():
            param.requires_grad = False
        for name, param in student_model.named_parameters():
            param.requires_grad = is_params_in_layers(name, layers_to_train, layer_types_to_train)
            if param.requires_grad:
                print_acc(f"allowing {name} parameter to train", print_message)
    
    if noise_alpha != 0.0:
        print_acc(f"[serum.py] Applying one-time shrink+perturb: noise={noise_alpha}", print_message)
        do_corruption(student_model, noise_alpha=noise_alpha, noise_beta=noise_beta)

    if compile_mode is not None:
        print_acc(f"[serum.py] Compiling models with mode: {compile_mode}", print_message)
        teacher_model = torch.compile(teacher_model, mode=compile_mode, fullgraph=True)
        student_model = torch.compile(student_model, mode=compile_mode)


    # ----------------------------------------------------------------
    # Load distillation dataset
    # ----------------------------------------------------------------
    train_ds_list = []
    for file in train_files:
        print_acc(f"[distill.py] Loading train dataset from {file}", print_message)
        train_ds = load_dataset("json", data_files=file, split="train", cache_dir=dataset_cache_dir)
        train_ds_list.append(train_ds)
    # ------------------------------------------------------------
    # PROCESS FOR SEQUENCE LENGTH
    # If join_or_subsequence, form sequences of exactly max_length 
    # by joining multiple or using subsequences
    # else filter for only those less than max_length
    # ------------------------------------------------------------
    train_ds_list, message = make_sequence_length(train_ds_list=train_ds_list, tokenizer=tokenizer, max_length=max_length, join_or_subsequence=join_or_subsequence)
    print_acc(message, print_message)



   # Interleave dataset
    if len(train_ds_list) == 0:
        raise ValueError("No training dataset provided!")
    elif len(train_ds_list) == 1:
        train_ds = train_ds_list[0]
    else:
        print_acc(f"[serum.py] Interleaving with probabilities: {interleave_probs}", print_message)
        train_ds = interleave_datasets(train_ds_list, probabilities=interleave_probs, seed=seed, stopping_strategy=stopping_strategy)

    # Print size
    print_acc(f"[serum.py] Train dataset size: {len(train_ds)}", print_message)

    # Print 3 samples of distillation dataset text
    for i in range(3): 
        sample_ids = train_ds[i]["input_ids"]
        sample_text = tokenizer.decode(sample_ids, skip_special_tokens=False)
        print_acc(f'[serum.py] Sample distillation dataset text {i}: "{sample_text[:200]}..."', print_message)



    # ------------------------------------------------------------
    # Create DataLoaders
    # ------------------------------------------------------------
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=data_collator)

    # Determine steps
    steps_per_epoch = len(train_loader)
    effective_steps_per_epoch = steps_per_epoch // gradient_accumulation_steps
    total_steps = effective_steps_per_epoch * epochs
    if max_steps > 0:
        total_steps = min(total_steps, max_steps)
    print_acc(f"[serum.py] {steps_per_epoch} steps/epoch, total steps: {total_steps}", print_message)



    # ------------------------------------------------------------
    # Optimizer + Scheduler
    # ------------------------------------------------------------
    print_acc(f"[serum.py] Using AdamW, LR={learning_rate}, weight_decay={weight_decay}", print_message)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if scheduler_type == "linear":
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            else:
                progress = float(current_step - num_warmup_steps) / float(max(1, total_steps - num_warmup_steps))
                return (1.0 - progress) * (1.0 - (min_lr / learning_rate)) + (min_lr / learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif scheduler_type == "cosine":
        def cosine_lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            else:
                progress = float(current_step - num_warmup_steps) / float(max(1, total_steps - num_warmup_steps))
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return cosine_decay * (1.0 - (min_lr / learning_rate)) + (min_lr / learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lr_lambda)
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")



    # ------------------------------------------------------------
    # Accelerator prepare
    # ------------------------------------------------------------
    teacher_model, student_model, optimizer, train_loader, scheduler = accelerator.prepare(
        teacher_model, student_model, optimizer, train_loader, scheduler
    )
    if base_teacher_name is not None:
        base_teacher_model = accelerator.prepare(base_teacher_model)

    # ------------------------------------------------------------
    # Get initial teacher eval dict which will be use for stop cond
    # ------------------------------------------------------------
    print_acc("[serum.py] Running evaluation ...", print_message)
    teacher_eval_dict = eval_fn(teacher_model, print_results=True) 
    teacher_eval_dict["train/step"] = 0
    teacher_eval_dict["train/tokens_seen"] = 0
    teacher_eval_dict["validation_stage"] = "teacher_baseline"

    if use_wandb and accelerator.is_main_process:
        wandb.log(teacher_eval_dict)
    if use_local_record and accelerator.is_main_process:
        with open(path_local_record, "a", encoding="utf-8") as f:
            f.write(json.dumps(teacher_eval_dict) + "\n")

    # ------------------------------------------------------------
    # Distillation Loop
    # ------------------------------------------------------------
    print_acc("[serum.py] Starting distillation ...", print_message)
    global_step = 0
    global_tokens = 0
    teacher_model.eval()

    stop_early = False

    for epoch in range(epochs):
        print_acc(f"[serum.py] Epoch {epoch+1}/{epochs}", print_message)

        if shrink_perturb_repeat and noise_alpha != 0.0:
            print_acc(f"[serum.py] Re-applying shrink+perturb before epoch {epoch+1}", print_message)
            do_corruption(student_model, noise_alpha=noise_alpha, noise_beta=noise_beta)

        student_model.train()
        micro_loss_sum = 0.0
        log_base_teacher_loss, log_teacher_loss = 0.0, 0.0
        teacher_counter, base_teacher_counter = 0, 0
        for step_in_epoch, batch in enumerate(train_loader):
            use_base_teacher = base_teacher_name is not None and not switch_teachers or random.random() < use_base_teacher_percent
            with torch.no_grad():
                if not (switch_teachers and use_base_teacher):
                    teacher_out = teacher_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=False,
                        output_hidden_states=True
                    )
                if use_base_teacher:
                    base_teacher_out = base_teacher_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=False
                    )
            
            if use_activation_loss:
                total_loss = 0.0
                teacher_hidden_states = teacher_out.hidden_states
                # Get initial hidden state (embeddings) from teacher
                inputs_embeds = teacher_hidden_states[0]
                batch_size, seq_len, _ = inputs_embeds.shape
                # Setup basic inputs that stay constant across layers
                cache_position = torch.arange(0, seq_len, device=inputs_embeds.device)
                position_ids = cache_position.unsqueeze(0)
                causal_mask = student_model.model._update_causal_mask(
                    batch.attention_mask, 
                    inputs_embeds,
                    cache_position,
                    None,  # past_key_values
                    None,  # output_attentions 
                )
                position_embeddings = student_model.model.rotary_emb(inputs_embeds, position_ids)

                # Process each layer
                loss_per_layer = []
                loss_attention_mask = batch.attention_mask.clone()
                loss_attention_mask = torch.cat([loss_attention_mask[:, 1:], torch.zeros(loss_attention_mask.shape[0], 1, device=loss_attention_mask.device)], dim=1)
                for layer_idx, decoder_layer in enumerate(student_model.model.layers):
                    if layers_to_train is not None and not ("all" in layers_to_train or layer_idx in layers_to_train):
                        continue
                    layer_outputs = decoder_layer(
                        teacher_hidden_states[layer_idx],
                        position_embeddings=position_embeddings,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=None,
                        output_attentions=None,
                        use_cache=False,
                        cache_position=cache_position,
                        last_cache_position=seq_len,
                    )
                    hidden_states = layer_outputs[0]
                    
                    # Compare with teacher's hidden state for this layer
                    if layer_idx == len(student_model.model.layers) - 1: # final layer needs a norm first
                        break
                    layer_loss = torch.nn.functional.mse_loss(
                        hidden_states,
                        teacher_hidden_states[layer_idx + 1],  # +1 since teacher_hidden_states[0] is embeddings
                        reduction="none"
                    ) # [B, S, D]
                    
                    # This is much slower, but maybe better?
                    layer_loss = layer_loss[loss_attention_mask.bool()]
                    layer_loss = layer_loss.mean()
                    total_loss += layer_loss
                    loss_per_layer.append(layer_loss.item())

                layer_idx += 1
                if layers_to_train is not None and "all" in layers_to_train or layer_idx in layers_to_train:
                    # Final norm layer
                    hidden_states = student_model.model.norm(hidden_states)
                    final_loss = torch.nn.functional.mse_loss(
                        hidden_states,
                        teacher_hidden_states[-1]  # Last hidden state from teacher
                    )
                    total_loss += final_loss
                    loss_per_layer.append(final_loss.item())

                # Average the losses
                act_loss = total_loss / len(loss_per_layer)
            if not use_activation_loss or both_losses_act_loss_multiplier is not None:
                student_out = student_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                if not (switch_teachers and use_base_teacher):
                    teacher_kd_loss = forward_kl_loss_fn(
                        teacher_out.logits,
                        student_out.logits,
                        batch["input_ids"],
                        tokenizer.pad_token_id,
                    )
                    log_teacher_loss += teacher_kd_loss.item()
                    teacher_counter += 1
                if use_base_teacher:
                    base_teacher_kd_loss = forward_kl_loss_fn(
                        base_teacher_out.logits,
                        student_out.logits,
                        batch["input_ids"],
                        tokenizer.pad_token_id,
                    )
                    log_base_teacher_loss += base_teacher_kd_loss.item()
                    base_teacher_counter += 1
                
                if switch_teachers or not use_base_teacher:
                    if use_base_teacher:
                        loss = base_teacher_kd_loss
                    else:
                        loss = teacher_kd_loss
                else:
                    loss = use_base_teacher_percent * base_teacher_kd_loss + (1 - use_base_teacher_percent) * teacher_kd_loss
            if use_activation_loss:
                if both_losses_act_loss_multiplier is None:
                    loss = act_loss
                else:
                    loss += both_losses_act_loss_multiplier * act_loss

            micro_loss_sum += loss.item()
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)

            tokens_this_batch = batch["attention_mask"].sum().detach()
            tokens_this_batch = accelerator.gather(tokens_this_batch).sum().item()
            global_tokens += tokens_this_batch

            if (step_in_epoch + 1) % gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(student_model.parameters(), gradient_clipping_threshold)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = micro_loss_sum / float(gradient_accumulation_steps)
                micro_loss_sum = 0.0

                if global_step == 1 or global_step % 5 == 0:
                    print_acc(
                        f"[serum.py] Epoch {epoch+1}/{epochs}, {global_step}/{total_steps // accelerator.num_processes}, KD loss: {avg_loss:.6f}",
                        print_message
                    )

                train_log_dict = {
                    
                    "train/step": global_step,
                    "train/tokens_seen": global_tokens,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/base_teacher_loss": log_base_teacher_loss / base_teacher_counter if base_teacher_counter != 0 else 0,
                    "train/teacher_loss": log_teacher_loss / teacher_counter if teacher_counter != 0 else 0,
                }
                if use_activation_loss:
                    train_log_dict["train/avg_layer_mse_loss"] = avg_loss
                else:
                    train_log_dict["train/kd_loss"] = avg_loss

                if use_wandb and accelerator.is_main_process:
                    wandb.log(train_log_dict)
                if use_local_record and accelerator.is_main_process:
                    with open(path_local_record, "a", encoding="utf-8") as f:
                        f.write(json.dumps(train_log_dict) + "\n")

                if max_steps > 0 and global_step >= max_steps:
                    print_acc("[serum.py] Reached max_steps => Stopping.", print_message)
                    stop_early = True
                    break
                # Validation: Calls provided eval function which returns a dictionary with eval name keys and results values
                # Do validation if it's the first stpe or the validation step or the last step
                if (global_step == 1 or global_step % validation_steps == 0 or stop_early):
                    print_acc("[serum.py] Running validation ...", print_message)
                    val_log_dict = eval_fn(student_model, print_results=True) 
                    val_log_dict["train/step"] = global_step
                    val_log_dict["train/tokens_seen"] = global_tokens
                    val_log_dict["validation_stage"] = "training_student"

                    if use_wandb and accelerator.is_main_process:
                        wandb.log(val_log_dict)
                    if use_local_record and accelerator.is_main_process:
                        with open(path_local_record, "a", encoding="utf-8") as f:
                            f.write(json.dumps(val_log_dict) + "\n")

                    if stop_cond_fn(student_eval_dict=val_log_dict, teacher_eval_dict=teacher_eval_dict):
                        stop_early = True
                        break

                if (save_checkpoint_steps > 0 and (global_step % save_checkpoint_steps == 0)) or stop_early:
                    if accelerator.is_main_process:
                        checkpoint_name = "final_student" if stop_early else f"checkpoint-step{global_step}"
                        checkpoint_path = os.path.join(output_dir, checkpoint_name)
                        unwrapped_student = accelerator.unwrap_model(student_model)
                        unwrapped_student.save_pretrained(checkpoint_path)
                        tokenizer.save_pretrained(checkpoint_path)
                        print_acc(f"[serum.py] Saved checkpoint => {checkpoint_path}", print_message)  

        if stop_early:
            break

    # Add final validation after all training
    print_acc("[serum.py] Running final validation ...", print_message)
    final_val_dict = eval_fn(student_model, print_results=True)
    final_val_dict["train/step"] = global_step
    final_val_dict["train/tokens_seen"] = global_tokens
    final_val_dict["validation_stage"] = "final_student"

    if use_wandb and accelerator.is_main_process:
        wandb.log(final_val_dict)
    if use_local_record and accelerator.is_main_process:
        with open(path_local_record, "a", encoding="utf-8") as f:
            f.write(json.dumps(final_val_dict) + "\n")
            
    # Final model saving at the end of all epochs
    if accelerator.is_main_process:
        final_model_path = os.path.join(output_dir, "final_model")
        unwrapped_student = accelerator.unwrap_model(student_model)
        unwrapped_student.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print_acc(f"[serum.py] Saved final model => {final_model_path}", print_message)
        if use_wandb:
            wandb.finish()


def is_params_in_layers(name, layers, layer_types):
    in_layer = 'all' in layers or any(f".{layer}." in name for layer in layers) 
    in_layer_type = 'all' in layer_types or any(layer_type in name for layer_type in layer_types) 
    return in_layer and in_layer_type


##############################################################
# The do_corruption function is used for "noise-and-decay"
##############################################################

def do_corruption(model, noise_alpha, noise_beta = 0.1, seed = 42):
    # Loop through all parameters and add random noise scaled by scale factor
    assert 0 <= noise_alpha <= 1
    assert 0 <= noise_beta <= 1
    params_with_corruption = 0
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            params_with_corruption += param.numel()
            corruption = torch.zeros_like(param.data)
            if len(param.data.shape) == 2:
                noise = torch.nn.init.xavier_uniform_(
                    torch.empty_like(param.data)
                )
            elif len(param.data.shape) == 1:
                noise = torch.zeros_like(param.data)
            else:
                raise RuntimeError(
                    f"Unsupported parameter shape: {param.data.shape}"
                )
            corruption = noise_beta * noise

            param.data = (1 - noise_alpha) * param.data + noise_alpha * corruption

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return params_with_corruption / (1.0 * total_params)
