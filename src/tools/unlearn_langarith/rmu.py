import os
import random
import math
import json
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from datasets import load_dataset
from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding
)
import wandb
from src.utils.process_datasets import make_sequence_length
from src.utils.loss_functions import print_acc

# Import RepNoise components
from src.utils.repnoise_loss import rep_noise_loss, register_activation_hook, MMD_loss

def unlearn_rmu(
    model_name,
    forget_train_file,
    retain_train_file,
    eval_fn,
    accelerator,
    output_dir,
    cache_dir,
    dataset_cache_dir,
    join_or_subsequence,

    # If True, also do normal MSE on 'retain' data to preserve them
    ga_gd,

    # Additional hyperparams for RMU
    rmu_layers,  # list of int indices
    end_layer,   # up to which layer we consider hidden states
    alpha,       # weight for retain loss
    c,           # scale for distractor
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
    
    # RepNoise parameters
    use_repnoise=False,         # Option to turn RepNoise on/off
    repnoise_alpha=1.0,         # Alpha parameter for RepNoise loss
    repnoise_beta=0.001,        # Beta parameter for RepNoise loss
    
    # SAM parameters
    use_sam=False,              # Option to turn SAM on/off
    sam_rho=0.05,               # Rho parameter (perturbation size) for SAM
):
    """
    RMU script in the same style as the GA script:
      - freeze all layers except MLP down_proj in `rmu_layers`
      - on 'forget' data: push hidden states at `end_layer` to a random 'distractor'
      - on 'retain' data (if ga_gd=True): push hidden states to match the original (frozen) model
    
    Additional options:
      - RepNoise: Add representation noising for robust unlearning
      - SAM: Use Sharpness-Aware Minimization for more robust unlearning
    """
    print_message = accelerator.is_main_process

    # Validate RepNoise settings
    if use_repnoise and (not ga_gd or not retain_train_file.strip()):
        print_acc("[rmu.py] WARNING: RepNoise requires both forget and retain datasets. Disabling RepNoise.", print_message)
        use_repnoise = False
    
    # Validate SAM settings
    if use_sam and (not ga_gd or not retain_train_file.strip()):
        print_acc("[rmu.py] WARNING: SAM requires both forget and retain datasets. Disabling SAM.", print_message)
        use_sam = False

    train_args = {**locals()}
    print_acc(f"[rmu.py] Initiated RMU training with:\n{train_args}", print_message)

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
    print_acc(f"[rmu.py] Loading model {model_name}", print_message)
    model_config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        attn_implementation='eager',
        output_hidden_states=True  # we need the hidden states
    )

    # Frozen model for reference
    frozen_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        attn_implementation='eager',
        output_hidden_states=True
    )
    frozen_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Move to device
    model.to(device)
    frozen_model.to(device)

    # ----------------------------------------------------------------
    # Freeze all parameters, except MLP down_proj in the specified layers
    # ----------------------------------------------------------------
    for param in model.parameters():
        param.requires_grad = False

    # Example: rmu_layers = [3, 4, 5]
    # We only unfreeze the "down_proj" submodule in those layers
    for layer_idx in rmu_layers:
        # e.g. model.model.layers[layer_idx].mlp.down_proj
        for param in model.model.layers[layer_idx].mlp.down_proj.parameters():
            param.requires_grad = True

    # ----------------------------------------------------------------
    # Load FORGET dataset (Required for unlearning)
    # ----------------------------------------------------------------
    if not forget_train_file.strip():
        raise ValueError("forget_train_file is empty => must provide a dataset to 'forget'")
    print_acc("[rmu.py] Loading 'forget' dataset", print_message)
    forget_ds = load_dataset(
        "json",
        data_files=forget_train_file,
        split="train",
        cache_dir=dataset_cache_dir
    )
    print_acc(f"[rmu.py] Forget dataset size: {len(forget_ds)}", print_message)
    sample_text = forget_ds[0]["text"].replace('\n', ' ')
    print_acc(f'[rmu.py] Sample forget text: "{sample_text[:200]}..."', print_message)

    # ------------------------------------------------------------
    # Process for sequence length
    # If join_or_subsequence, form sequences of exactly max_length by joining multiple or using subsequences
    # else filter for only those less than max_length
    # ------------------------------------------------------------
    train_ds_list, message = make_sequence_length(train_ds_list=[forget_ds], tokenizer=tokenizer, max_length=max_length, join_or_subsequence=join_or_subsequence)
    print_acc(message, print_message)
    forget_ds = train_ds_list[0]
    forget_loader = DataLoader(
        forget_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=max_length)
    )

    # ----------------------------------------------------------------
    # Load RETAIN dataset (Used for ga_gd, RepNoise, and SAM)
    # ----------------------------------------------------------------
    if ga_gd and retain_train_file.strip():
        print_acc("[rmu.py] => RMU + Retain => loading 'retain' dataset", print_message)
        retain_ds = load_dataset(
            "json",
            data_files=retain_train_file,
            split="train",
            cache_dir=dataset_cache_dir
        )
        print_acc(f"[rmu.py] Retain dataset size: {len(retain_ds)}", print_message)
        sample_text_r = retain_ds[0]["text"].replace('\n', ' ')
        print_acc(f'[rmu.py] Sample retain text: "{sample_text_r[:200]}..."', print_message)
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
    print_acc(f"[rmu.py] {steps_per_epoch} steps per epoch, total steps: {total_steps}", print_message)

    # ----------------------------------------------------------------
    # Optimizer + LR scheduler
    # ----------------------------------------------------------------
    print_acc(f"[rmu.py] Using AdamW optimizer, LR={learning_rate}, weight_decay={weight_decay}", print_message)
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

    # Prepare with Accelerator
    model, optimizer, forget_loader, scheduler = accelerator.prepare(
        model, optimizer, forget_loader, scheduler
    )
    if ga_gd and retain_loader is not None:
        retain_loader = accelerator.prepare(retain_loader)

    # ----------------------------------------------------------------
    # Create random 'distractor' direction
    # ----------------------------------------------------------------
    # We'll create a single random vector 'u' of shape (hidden_size,)
    # Then scale by c as needed in the loop. 
    # (We can do batch repeat or broadcast.)
    with torch.no_grad():
        hidden_size = model_config.hidden_size  # or model.config.hidden_size
        u = torch.randn(hidden_size, device=device)
        u = u / u.norm()  # unit norm

    # We'll compute MSE ourselves
    mse_loss_fn = MSELoss(reduction="mean")

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    print_acc("[rmu.py] Starting RMU training", print_message)

    # Initial validation before training
    print_acc("[rmu.py] Running initial validation before training...", print_message)
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
        print_acc(f"[rmu.py] Epoch {epoch+1}/{epochs}", print_message)
        model.train()
        frozen_model.eval()

        for step_in_epoch in range(steps_per_epoch):
            # Initialize SAM variables if SAM is enabled
            if use_sam:
                # Store original parameters for later restoration
                param_list = [p for p in model.parameters() if p.requires_grad]
                original_params = [p.data.clone() for p in param_list]

            # 1) Get forget batch
            try:
                forget_batch = next(forget_loader_iter)
            except StopIteration:
                forget_loader_iter = iter(forget_loader)
                forget_batch = next(forget_loader_iter)

            # Forward pass (forget data) => get hidden states
            outputs_forget = model(
                input_ids=forget_batch["input_ids"],
                attention_mask=forget_batch["attention_mask"],
                output_hidden_states=True
            )
            # Grab the hidden state at end_layer: shape (batch, seq_len, hidden_size)
            forget_hidden = outputs_forget.hidden_states[end_layer]

            # We want to push these hidden states to c * u (the "distractor")
            # We'll broadcast (batch, seq_len, hidden_size) => multiply c*(u)
            # We'll shape u => (1,1,hidden_size) => expand
            distractor = c * u.unsqueeze(0).unsqueeze(0)  # shape (1,1,hidden_size)
            distractor = distractor.expand(forget_hidden.size(0), forget_hidden.size(1), -1)
            
            l_forget = mse_loss_fn(forget_hidden, distractor)  # MSE

            # 2) If ga_gd => get retain batch => preserve original hidden states
            if ga_gd and retain_loader_iter is not None:
                try:
                    retain_batch = next(retain_loader_iter)
                except StopIteration:
                    retain_loader_iter = iter(retain_loader)
                    retain_batch = next(retain_loader_iter)

                # We'll forward pass both the new model + frozen model on the retain data
                with torch.no_grad():
                    fro_outputs = frozen_model(
                        input_ids=retain_batch["input_ids"],
                        attention_mask=retain_batch["attention_mask"],
                        output_hidden_states=True
                    )
                    fro_hidden = fro_outputs.hidden_states[end_layer]

                new_outputs = model(
                    input_ids=retain_batch["input_ids"],
                    attention_mask=retain_batch["attention_mask"],
                    output_hidden_states=True
                )
                new_hidden = new_outputs.hidden_states[end_layer]

                l_retain = mse_loss_fn(new_hidden, fro_hidden)
                
                # If RepNoise is enabled and we have both forget and retain batches
                if use_repnoise:
                    # Map forget_batch -> harmful_batch, retain_batch -> harmless_batch
                    repnoise_loss_val = rep_noise_loss(
                        model=model,
                        harmful_batch=forget_batch,
                        harmless_batch=retain_batch,
                        beta=repnoise_beta,
                        alpha=repnoise_alpha
                    )
                    # Add RepNoise to the loss
                    total_loss = (l_forget + alpha * l_retain + repnoise_loss_val) / gradient_accumulation_steps
                else:
                    # Original loss calculation when RepNoise is disabled
                    total_loss = (l_forget + alpha * l_retain) / gradient_accumulation_steps

                # Token counting
                tokens_forget = forget_batch["attention_mask"].sum().detach()
                tokens_forget = accelerator.gather(tokens_forget).sum().item()
                tokens_retain = retain_batch["attention_mask"].sum().detach()
                tokens_retain = accelerator.gather(tokens_retain).sum().item()
                global_tokens += (tokens_forget + tokens_retain)

            else:
                # No retain data => purely forget
                total_loss = l_forget / gradient_accumulation_steps
                tokens_this_batch = forget_batch["attention_mask"].sum().detach()
                tokens_this_batch = accelerator.gather(tokens_this_batch).sum().item()
                global_tokens += tokens_this_batch

            # SAM Update - only if SAM is enabled and we're doing a full update
            if use_sam and ga_gd and (step_in_epoch + 1) % gradient_accumulation_steps == 0:
                # Step 1: Compute and save gradients
                accelerator.backward(total_loss)
                
                forget_grads = []
                for p in param_list:
                    if p.grad is not None:
                        forget_grads.append(p.grad.detach().clone())
                    else:
                        forget_grads.append(None)
                
                # Step 2: Calculate perturbation direction and magnitude
                # Compute gradient norm for scaling
                valid_grads = [g for g in forget_grads if g is not None]
                if valid_grads:
                    grad_norm = torch.stack([g.norm(2) for g in valid_grads]).norm(2)
                    
                    # Apply perturbation to parameters
                    with torch.no_grad():
                        for i, (param, grad) in enumerate(zip(param_list, forget_grads)):
                            if grad is not None:
                                # ε = ρ * g / ||g||
                                eps = grad * (sam_rho / (grad_norm + 1e-12))
                                param.add_(eps)
                
                    # Step 3: Recompute forward pass on perturbed model
                    # Recompute loss with perturbed parameters
                    model.zero_grad()
                    
                    # Recompute forget loss
                    outputs_forget_perturbed = model(
                        input_ids=forget_batch["input_ids"],
                        attention_mask=forget_batch["attention_mask"],
                        output_hidden_states=True
                    )
                    forget_hidden_perturbed = outputs_forget_perturbed.hidden_states[end_layer]
                    l_forget_perturbed = mse_loss_fn(forget_hidden_perturbed, distractor)
                    
                    # Recompute retain loss
                    new_outputs_perturbed = model(
                        input_ids=retain_batch["input_ids"],
                        attention_mask=retain_batch["attention_mask"],
                        output_hidden_states=True
                    )
                    new_hidden_perturbed = new_outputs_perturbed.hidden_states[end_layer]
                    l_retain_perturbed = mse_loss_fn(new_hidden_perturbed, fro_hidden)
                    
                    # Recalculate total loss with perturbed parameters
                    if use_repnoise:
                        repnoise_loss_perturbed = rep_noise_loss(
                            model=model,
                            harmful_batch=forget_batch,
                            harmless_batch=retain_batch,
                            beta=repnoise_beta,
                            alpha=repnoise_alpha
                        )
                        total_loss_perturbed = (l_forget_perturbed + alpha * l_retain_perturbed + repnoise_loss_perturbed) / gradient_accumulation_steps
                    else:
                        total_loss_perturbed = (l_forget_perturbed + alpha * l_retain_perturbed) / gradient_accumulation_steps
                    
                    # Step 4: Compute gradients on perturbed model
                    accelerator.backward(total_loss_perturbed)
                    
                    # Step 5: Restore original parameters
                    with torch.no_grad():
                        for param, orig_param in zip(param_list, original_params):
                            param.copy_(orig_param)
                
                # Optimizer step with gradients computed on perturbed model
                accelerator.clip_grad_norm_(model.parameters(), gradient_clipping_threshold)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            # Standard backprop for non-SAM or non-update step cases
            else:
                # Backprop
                accelerator.backward(total_loss)

                if (step_in_epoch + 1) % gradient_accumulation_steps == 0:
                    accelerator.clip_grad_norm_(model.parameters(), gradient_clipping_threshold)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            # Logging
            if global_step > 0 and (global_step == 1 or global_step % 5 == 0):
                features_enabled = []
                if ga_gd:
                    features_enabled.append("Retain")
                if use_repnoise:
                    features_enabled.append("RepNoise")
                if use_sam:
                    features_enabled.append("SAM")
                features_str = "+".join(features_enabled) if features_enabled else ""
                
                msg = (
                    f"[rmu.py] Epoch {epoch+1}/{epochs}, "
                    f"Step {global_step}/{total_steps}, RMU{'+'+features_str if features_str else ''} "
                    f"=> l_forget: {l_forget:.6f}"
                )
                print_acc(msg, print_message)
                if ga_gd and retain_loader_iter is not None:
                    print_acc(f"[rmu.py] l_retain: {l_retain:.6f}", print_message)
                if use_repnoise:
                    print_acc(f"[rmu.py] RepNoise Loss: {repnoise_loss_val:.6f}", print_message)
                if use_sam:
                    print_acc(f"[rmu.py] SAM enabled with rho={sam_rho}", print_message)

                train_log_dict = {
                    "train/l_forget": l_forget.item(),
                    "train/step": global_step,
                    "train/tokens_seen": global_tokens,
                    "train/lr": scheduler.get_last_lr()[0],
                }
                if ga_gd and retain_loader_iter is not None:
                    train_log_dict["train/l_retain"] = l_retain.item()
                if use_repnoise:
                    train_log_dict["train/repnoise_loss"] = repnoise_loss_val.item()
                if use_sam:
                    train_log_dict["train/sam_enabled"] = True
                    train_log_dict["train/sam_rho"] = sam_rho

                if use_wandb and accelerator.is_main_process:
                    wandb.log(train_log_dict)
                if use_local_record and accelerator.is_main_process:
                    with open(path_local_record, "a", encoding="utf-8") as f:
                        f.write(json.dumps(train_log_dict) + "\n")

                # Validation (like in GA script)
                # if it's the first step of modulo of validation steps or the last
                if global_step == 1 or global_step % validation_steps == 0 or (max_steps > 0 and global_step >= max_steps):
                    print_acc("[rmu.py] Running validation ...", print_message)
                    
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
                    print_acc("[rmu.py] Reached max_steps => Stopping.", print_message)
                    break

        if max_steps > 0 and global_step >= max_steps:
            break

    # Final validation after all training
    print_acc("[rmu.py] Running final validation after training completion...", print_message)
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
        print_acc(f"[rmu.py] Model saved to => {save_path}", print_message)