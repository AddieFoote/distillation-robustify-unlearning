import torch

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from typing import List, Optional

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.loss_functions import forward_kl_loss_fn, cross_entropy_loss_fn, cross_entropy_loss_fn_only, print_acc, custom_login
from utils.generate_arithmetic import get_equations, get_template_word_problems
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from torch.utils.data import DataLoader
import time
from functools import partial
import re
from utils.process_datasets import make_sequence_length


def evaluate_kd_ce_ppl(student_model, teacher_model, data_loader, pad_token_id, accelerator, fn_only=False):
    """
    Evaluate student vs. teacher on:
      - KD loss (KL(teacher||student)),
      - CE loss (student),
      - PPL (approx. via CE).
    No ES metric here.
    Returns (avg_kd_loss, avg_ce_loss, avg_ppl).
    """
    student_model.eval()
    teacher_model.eval()

    total_kd_loss = 0.0
    total_ce_loss = 0.0
    total_count = 0

    total_loss_for_ppl = 0.0
    total_tokens_for_ppl = 0

    for batch in data_loader:
        with torch.no_grad():
            teacher_out = teacher_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            student_out = student_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )

            # 1) KD
            kd_val = forward_kl_loss_fn(
                teacher_out.logits,
                student_out.logits,
                batch["input_ids"],
                pad_token_id,
            )

            # 2) CE
            if fn_only == True:
                ce_val = cross_entropy_loss_fn_only(
                    student_out.logits,
                    batch["input_ids"],
                    pad_token_id,
                )
            else:
                ce_val = cross_entropy_loss_fn(
                    student_out.logits,
                    batch["input_ids"],
                    pad_token_id,
                )

        bsz_local = batch["input_ids"].size(0)
        total_kd_loss += kd_val.item() * bsz_local
        total_ce_loss += ce_val.item() * bsz_local
        total_count += bsz_local

        # 3) PPL => weigh CE by # of tokens
        tokens_this_batch = batch["attention_mask"].sum(dim=1).sum().item()
        total_loss_for_ppl += ce_val.item() * tokens_this_batch
        total_tokens_for_ppl += tokens_this_batch

    avg_kd = total_kd_loss / max(total_count, 1)
    avg_ce = total_ce_loss / max(total_count, 1)

    if total_tokens_for_ppl > 0:
        avg_nll = total_loss_for_ppl / float(total_tokens_for_ppl)
        avg_ppl = math.exp(avg_nll)
    else:
        avg_ppl = float('inf')

    student_model.train()
    teacher_model.train()

    return (avg_kd, avg_ce, avg_ppl)

def evaluate_ce_loss(model, data_loader, pad_token_id, accelerator, fn_only=False):
    """Compute average cross-entropy loss over the provided data_loader."""
    model.eval()
    total_ce_loss = 0.0
    total_count = 0

    for batch in data_loader:
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            if fn_only == True:
                ce_loss = cross_entropy_loss_fn_only(outputs.logits, batch["input_ids"], pad_token_id)
            else:
                ce_loss = cross_entropy_loss_fn(outputs.logits, batch["input_ids"], pad_token_id)
        batch_size_local = batch["input_ids"].size(0)
        total_ce_loss += ce_loss.item() * batch_size_local
        total_count += batch_size_local

    model.train()
    return total_ce_loss / max(total_count, 1)


def get_arithmetic_eval_fn(
    model_name,
    batch_size,
    max_length,
    cache_dir,
    dataset_cache_dir,
    num_wiki_batches,
    eng_valid_file,
    accelerator
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize_prompts(data):
        new_data = []
        for prompt, ans in data: 
            inputs = tokenizer(prompt, return_tensors="pt")
            new_data.append((inputs, ans))
        return new_data

    eval_data = {}
    for op in ["subtraction", "addition", "multiplication", "division"]:
        eq_data = get_equations(operations=[op], seed=42, amount=100, val=True)
        wp_data = get_template_word_problems(
            operations=[op], seed=42, amount=100, val=True
        )

        eval_data[op + "_equation"] = tokenize_prompts(eq_data)
        eval_data[op + "_word_problem"] = tokenize_prompts(wp_data)

    eng_valid_ds = load_dataset("json", data_files=eng_valid_file, split="train", cache_dir=dataset_cache_dir)
    print_message = accelerator.is_main_process
    print_acc(f"[validation_functions.py] Eng validation dataset size: {len(eng_valid_ds)}", print_message)
    eng_valid_ds = eng_valid_ds.remove_columns("text")
    eng_valid_ds = eng_valid_ds.map(
        lambda examples: {
            "input_ids": [ids[:max_length] for ids in examples["input_ids"]],
            "attention_mask": [mask[:max_length] for mask in examples["attention_mask"]],
        },
        batched=True
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=max_length)
    eng_valid_loader = DataLoader(eng_valid_ds, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    eng_valid_loader = accelerator.prepare(
        eng_valid_loader
    )


    def do_arithmetic_eval(model, tokenizer, data):
        # Get the actual model if it's wrapped in DDP or any other wrapper
        if hasattr(model, 'module'):
            generation_model = model.module
        else:
            generation_model = model

        num_correct_by_max = 0
        for prompt, ans in data:
            inputs = prompt.to(model.device)
            with torch.no_grad(): # Generate next 4 tokens
                outputs = generation_model.generate(
                    **inputs,
                    max_new_tokens=4,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            generated = outputs[0][inputs["input_ids"].shape[1] :]
            pred = tokenizer.decode(generated, skip_special_tokens=False).strip()
            pred = re.split(r'\s+|<bos>', pred)[0] if pred else pred
            input_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False).strip()
            try: # Check if prediction matches answer
                pred_num = float(pred)
                if abs(pred_num - ans) < 1e-6:
                    num_correct_by_max += 1
            except:
                pass

        absolute_percent = num_correct_by_max / float(len(data))
        return absolute_percent

    def arithmetic_eval(model, eng_valid_loader, tokenizer, accelerator, print_results):
        start_time = time.time()
        wiki_loss = evaluate_ce_loss(model, eng_valid_loader, tokenizer.pad_token_id, accelerator, fn_only=True)
        eval_dict = {}

        eval_dict["val/eng_ce_loss"] = wiki_loss
        eval_dict["val/wiki_eval_time"] = time.time() - start_time
        start_time = time.time()
        for operation, data in eval_data.items():
            acc = do_arithmetic_eval(model, tokenizer, data)  # TODO: Teacher tokenizer?
            eval_dict[f"val/{operation}_acc"] = acc
       
        eval_dict["val/arithmetic_eval_time"] = time.time() - start_time
        if print_results:
            print_acc(f"[validation_functions.py] Validation Results:", print_message)
            for key, value in eval_dict.items():
                print(f"[validation_functions.py] \t{key}: {value}")
        return eval_dict

    return partial(arithmetic_eval, eng_valid_loader=eng_valid_loader, tokenizer=tokenizer, accelerator=accelerator)

def get_korean_and_english_evalaution_fn(
    model_name, 
    max_length, 
    eng_valid_file, 
    kor_valid_file, 
    dataset_cache_dir, 
    cache_dir,
    batch_size,
    accelerator
):
    """
    Returns an evaluation function that takes the model and calculates the cross entropy loss over the provided eng_valid_file and kor_valid_file.
    Ensure the same accelerator is passed into this function and the training which it is being used in 
    """
    eng_valid_ds = load_dataset("json", data_files=eng_valid_file, split="train", cache_dir=dataset_cache_dir)
    kor_valid_ds = load_dataset("json", data_files=kor_valid_file, split="train", cache_dir=dataset_cache_dir)
    print_message = accelerator.is_main_process
    print_acc(f"[validation_functions.py] Eng validation dataset size: {len(eng_valid_ds)}", print_message)
    print_acc(f"[validation_functions.py] Kor validation dataset size: {len(kor_valid_ds)}", print_message)
    eng_valid_ds = eng_valid_ds.remove_columns("text")
    kor_valid_ds = kor_valid_ds.remove_columns("text")
    def filter_long(batch):
        # Return a list of booleans for each example in the batch.
        return [len(ids) <= max_length for ids in batch["input_ids"]]
    
    eng_valid_ds = eng_valid_ds.filter(filter_long, batched=True, batch_size=200_000, num_proc=100)
    kor_valid_ds = kor_valid_ds.filter(filter_long, batched=True, batch_size=200_000, num_proc=100)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=max_length)
    
    eng_valid_loader = DataLoader(eng_valid_ds, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    kor_valid_loader = DataLoader(kor_valid_ds, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    eng_valid_loader, kor_valid_loader = accelerator.prepare(
        eng_valid_loader, kor_valid_loader
    )

    def eval_fn(model, print_results):
        eng_valid_ce_loss = evaluate_ce_loss(model, eng_valid_loader, tokenizer.pad_token_id, accelerator, fn_only=True)
        kor_valid_ce_loss = evaluate_ce_loss(model, kor_valid_loader, tokenizer.pad_token_id, accelerator, fn_only=True)
        val_dict = {
            "val/eng_ce_loss": eng_valid_ce_loss,
            "val/kor_ce_loss": kor_valid_ce_loss,
        }
        if print_results:
            print_acc(f"[validation_functions.py] Validation Results:", print_message)
            for key, value in val_dict.items():
                print(f"[validation_functions.py] \t{key}: {value}")

        return val_dict
    return eval_fn


def eval_model_lm_eval(
    model,
    print_results: bool,
    accelerator,
    seed: int,
    task_list: List[str],
    limit: Optional[List[float]] = None,
    keep_all_subtasks: bool = False,
):
    custom_login()
    start_time = time.time()

    was_training = model.training

    if hasattr(model, '_orig_mod'):
        model = model._orig_mod

    model.name_or_path = (
        "google/gemma-2-2b"
    )

    model = model.eval()
    eval_model = HFLM(model)

    if limit is not None:
        assert len(limit) == len(
            task_list
        ), "Number of limits must match number of tasks"
        for lim in limit:
            if lim is not None and 1 < lim < 10:
                raise ValueError(
                    "Limit must be between 0.0 and 1.0, or greater than 10. Otherwise, you are doing between 1 and 10 samples, which you probably didn't mean to do."
                )
    else:
        limit = [None] * len(task_list)
    
    eval_dict = {}

    for task, lim in zip(task_list, limit):
        task_start_time = time.time()

        num_fewshot = 5 if "mmlu" in task else 0
        print(f"[validation_functions.py] Setting {task} few shot to [{num_fewshot}]")

        # Run the evaluation
        with torch.inference_mode():
            results = evaluator.simple_evaluate(
                model=eval_model,
                tasks=[task],
                num_fewshot=num_fewshot,
                limit=lim,
                random_seed = seed,
                numpy_random_seed = seed,
                torch_random_seed = seed,
                fewshot_random_seed = seed,
            )
        results = results["results"]

        if keep_all_subtasks:
            for subtask in results.keys():
                eval_dict[f"{subtask}_limit_{lim}_shots_{num_fewshot}"] = results[
                    subtask
                ]["acc,none"]
        else:
            eval_dict[f"{task}_limit_{lim}_shots_{num_fewshot}"] = results[task][
                "acc,none"
            ]

        eval_dict[f"{task} time"] = time.time() - task_start_time

    if was_training:
        model = model.train()

    eval_dict[f"total time"] = time.time() - start_time
    if print_results:
        print_acc(f"[validation_functions.py] Validation Results:", accelerator.is_main_process)
        for key, value in eval_dict.items():
            print(f"[validation_functions.py] \t{key}: {value}")

    return eval_dict


def get_wmdp_cyber_eval_fn(accelerator, large_eval, no_mmlu=False):
    if no_mmlu:
        lim = [None] if large_eval else [1000]
        task_list = ["wmdp_cyber"]
    else:
        lim = [None, 0.40] if large_eval else [1000, .07]
        task_list = ["wmdp_cyber", "mmlu"]    
    seed = 1234 if large_eval else None
    return lambda model, print_results: eval_model_lm_eval(model, print_results, seed=seed, accelerator=accelerator, task_list=task_list, limit=lim)


def get_wmdp_bio_eval_fn(accelerator, large_eval, no_mmlu=False):
    if no_mmlu:
        lim = [None] if large_eval else [1000]
        task_list = ["wmdp_bio"]
    else:
        lim = [None, 0.40] if large_eval else [1000, .07]
        task_list = ["wmdp_bio", "mmlu"]

    seed = 1234 if large_eval else None
    return lambda model, print_results: eval_model_lm_eval(model, print_results, seed=seed, accelerator=accelerator, task_list=task_list, limit=lim)

def get_both_wmdp_eval_fn(accelerator, large_eval):
    lim = [None, None, 0.40] if large_eval else [1000, 1000, .07]
    seed = 1234 if large_eval else None
    return lambda model, print_results: eval_model_lm_eval(model, print_results, seed=seed, accelerator=accelerator, task_list=["wmdp_bio", "wmdp_cyber", "mmlu"], limit=lim)



def do_loss_eval(model, print_results, accelerator, forget_loader, retain_loader):
    model.eval()
    total_forget_loss = 0.0
    total_forget_count = 0
    total_forget_tokens = 0
    total_retain_loss = 0.0
    total_retain_count = 0
    total_retain_tokens = 0
    # Print the first batch from forget_loader
    try:
        first_forget_batch = next(iter(forget_loader))
        print_acc("[validation_functions.py] First batch from forget_loader:", accelerator.is_main_process)
        print_acc(str(first_forget_batch), accelerator.is_main_process)
    except Exception as e:
        print_acc(f"[validation_functions.py] Could not print first batch from forget_loader: {e}", accelerator.is_main_process)

    with torch.no_grad():
        for batch in forget_loader:
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = cross_entropy_loss_fn_only(
                outputs.logits,
                batch["input_ids"],
                model.config.pad_token_id,
                loss_mask=batch.get("loss_mask")
            )
            batch_size = batch["input_ids"].size(0)
            total_forget_loss += loss.item() * batch_size
            total_forget_count += batch_size
            total_forget_tokens += batch["attention_mask"].sum().item()

        for batch in retain_loader:
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = cross_entropy_loss_fn_only(
                outputs.logits,
                batch["input_ids"],
                model.config.pad_token_id
            )
            batch_size = batch["input_ids"].size(0)
            total_retain_loss += loss.item() * batch_size
            total_retain_count += batch_size
            total_retain_tokens += batch["attention_mask"].sum().item()

    avg_forget_loss = total_forget_loss / total_forget_count if total_forget_count > 0 else float("nan")
    avg_retain_loss = total_retain_loss / total_retain_count if total_retain_count > 0 else float("nan")

    results = {
        "avg_forget_loss": avg_forget_loss,
        "avg_retain_loss": avg_retain_loss,
        "total_forget_tokens": total_forget_tokens,
        "total_retain_tokens": total_retain_tokens,
    }

    if print_results:
        print_acc(f"[validation_functions.py] Forget set average loss: {avg_forget_loss}", accelerator.is_main_process)
        print_acc(f"[validation_functions.py] Retain set average loss: {avg_retain_loss}", accelerator.is_main_process)
        print_acc(f"[validation_functions.py] Total forget tokens: {total_forget_tokens}", accelerator.is_main_process)
        print_acc(f"[validation_functions.py] Total retain tokens: {total_retain_tokens}", accelerator.is_main_process)

    return results


def get_loss_eval_fn(accelerator,):
    return lambda model, print_results, forget_loader, retain_loader: do_loss_eval(model, print_results, accelerator=accelerator, forget_loader=forget_loader, retain_loader=retain_loader)
