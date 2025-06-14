from src.tools.pretrain import train
from src.utils.paths import CACHE_DIR, DATASET_DIR, MODEL_DIR, WANDB_API_KEY_PATH
from accelerate import Accelerator
from datasets import load_dataset
from utils.loss_functions import print_acc
from utils.validation_functions import evaluate_ce_loss, get_arithmetic_eval_fn
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding
)
from torch.utils.data import DataLoader

# SETUPS_TO_RUN = ['gemma-2-0.1B_all_arithmetic+eng', 'gemma-2-0.1B_addition_subtraction+eng']
# SETUPS_TO_RUN = ['gemma-2-0.6B_all_arithmetic+eng', 'gemma-2-0.6B_addition_subtraction+eng']
SETUPS_TO_RUN = ['gemma-2-0.3B_all_arithmetic+eng', 'gemma-2-0.3B_addition_subtraction+eng']

try:
    with open(WANDB_API_KEY_PATH, "r", encoding="utf-8") as f:
        api_key = f.read().strip()
except Exception as e:
    print(f"[ERROR] Unable to read WandB API key from {WANDB_API_KEY_PATH}. Exception: {e}")
    exit(1)

setups = {
    "gemma-2-0.1B_all_arithmetic+eng": {
        'model_name'       : f"{MODEL_DIR}/random_init_models/gemma-2-0.1B",
        'eng_train_file'   : f"{DATASET_DIR}/pretrain/train_eng.jsonl",
        'secondary_train_files'   : [f"{DATASET_DIR}/pretrain/train_all_arithmetic.jsonl"],
        'eng_valid_file'   : f"{DATASET_DIR}/pretrain/valid_eng.jsonl",
        'interleave_probs' : [.75, .25],
        'output_dir'       : f"{MODEL_DIR}/pretrained_models/gemma-2-0.1B_all_arithmetic+eng",
        'cache_dir'        : CACHE_DIR,
        'dataset_cache_dir': CACHE_DIR,

        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 45,
        'gradient_accumulation_steps' : 8,
        'join_or_subsequence'         : True,
        'epochs'                      : 1,
        'learning_rate'               : 4e-4,
        'max_steps'                   : 1000,             
        'num_warmup_steps'            : 50,
        'validation_steps'            : 100,
        'save_checkpoint_steps'       : 500,
        'scheduler_type'              : "cosine",  
        'min_lr'                      : 4e-4,
        'weight_decay'                : 0,
        'gradient_clipping_threshold' : 1.0,
        'max_length'                  : 256,
        'use_wandb'        : True,
        'wandb_project'    : "gemma-2-0.1B_all_arithmetic+eng",
        'wandb_run_name'   : None,
        'wandb_api_key'    : api_key,
        'use_local_record' : True,
        'path_local_record': f"{MODEL_DIR}/local_records/pretrained_models/gemma-2-0.1B_all_arithmetic+eng.txt",   
    },
    "gemma-2-0.1B_addition_subtraction+eng": {
        'model_name'       : f"{MODEL_DIR}/random_init_models/gemma-2-0.1B",
        'eng_train_file'   : f"{DATASET_DIR}/pretrain/train_eng.jsonl",
        'secondary_train_files'   : [f"{DATASET_DIR}/pretrain/train_addition_subtraction.jsonl"],
        'eng_valid_file'   : f"{DATASET_DIR}/pretrain/valid_eng.jsonl",
        'interleave_probs' : [.875, .125],
        'output_dir'       : f"{MODEL_DIR}/pretrained_models/gemma-2-0.1B_addition_subtraction+eng",
        'cache_dir'        : CACHE_DIR,
        'dataset_cache_dir': CACHE_DIR,

        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 15,
        'gradient_accumulation_steps' : 24,
        'join_or_subsequence'         : True,
        'epochs'                      : 1,
        'learning_rate'               : 4e-4,
        'max_steps'                   : 1000,             
        'num_warmup_steps'            : 50,
        'validation_steps'            : 100,
        'save_checkpoint_steps'       : 500,
        'scheduler_type'              : "cosine",  
        'min_lr'                      : 4e-4,
        'weight_decay'                : 0,
        'gradient_clipping_threshold' : 1.0,
        'max_length'                  : 256,
        'use_wandb'        : True,
        'wandb_project'    : "gemma-2-0.1B_addition_subtraction+eng",
        'wandb_run_name'   : None,
        'wandb_api_key'    : api_key,
        'use_local_record' : True,
        'path_local_record': f"{MODEL_DIR}/local_records/pretrained_models/gemma-2-0.1B_addition_subtraction+eng.txt",   
    },
        "gemma-2-0.3B_all_arithmetic+eng": {
        'model_name'       : f"{MODEL_DIR}/random_init_models/gemma-2-0.3B",
        'eng_train_file'   : f"{DATASET_DIR}/pretrain/train_eng.jsonl",
        'secondary_train_files'   : [f"{DATASET_DIR}/pretrain/train_all_arithmetic.jsonl"],
        'eng_valid_file'   : f"{DATASET_DIR}/pretrain/valid_eng.jsonl",
        'interleave_probs' : [.75, .25],
        'output_dir'       : f"{MODEL_DIR}/pretrained_models/gemma-2-0.3B_all_arithmetic+eng",
        'cache_dir'        : CACHE_DIR,
        'dataset_cache_dir': CACHE_DIR,

        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 15,
        'gradient_accumulation_steps' : 24,
        'join_or_subsequence'         : True,
        'epochs'                      : 1,
        'learning_rate'               : 4e-4,
        'max_steps'                   : 1000,             
        'num_warmup_steps'            : 50,
        'validation_steps'            : 100,
        'save_checkpoint_steps'       : 500,
        'scheduler_type'              : "cosine",  
        'min_lr'                      : 4e-4,
        'weight_decay'                : 0,
        'gradient_clipping_threshold' : 1.0,
        'max_length'                  : 256,
        'use_wandb'        : True,
        'wandb_project'    : "gemma-2-0.3B_all_arithmetic+eng",
        'wandb_run_name'   : None,
        'wandb_api_key'    : api_key,
        'use_local_record' : True,
        'path_local_record': f"{MODEL_DIR}/local_records/pretrained_models/gemma-2-0.3B_all_arithmetic+eng.txt",   
    },
    "gemma-2-0.3B_addition_subtraction+eng": {
        'model_name'       : f"{MODEL_DIR}/random_init_models/gemma-2-0.3B",
        'eng_train_file'   : f"{DATASET_DIR}/pretrain/train_eng.jsonl",
        'secondary_train_files'   : [f"{DATASET_DIR}/pretrain/train_addition_subtraction.jsonl"],
        'eng_valid_file'   : f"{DATASET_DIR}/pretrain/valid_eng.jsonl",
        'interleave_probs' : [.875, .125],
        'output_dir'       : f"{MODEL_DIR}/pretrained_models/gemma-2-0.3B_addition_subtraction+eng",
        'cache_dir'        : CACHE_DIR,
        'dataset_cache_dir': CACHE_DIR,

        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 15,
        'gradient_accumulation_steps' : 24,
        'join_or_subsequence'         : True,
        'epochs'                      : 1,
        'learning_rate'               : 4e-4,
        'max_steps'                   : 1000,             
        'num_warmup_steps'            : 50,
        'validation_steps'            : 100,
        'save_checkpoint_steps'       : 500,
        'scheduler_type'              : "cosine",  
        'min_lr'                      : 4e-4,
        'weight_decay'                : 0,
        'gradient_clipping_threshold' : 1.0,
        'max_length'                  : 256,
        'use_wandb'        : True,
        'wandb_project'    : "gemma-2-0.3B_addition_subtraction+eng",
        'wandb_run_name'   : None,
        'wandb_api_key'    : api_key,
        'use_local_record' : True,
        'path_local_record': f"{MODEL_DIR}/local_records/pretrained_models/gemma-2-0.3B_addition_subtraction+eng.txt",   
    },
    "gemma-2-0.6B_all_arithmetic+eng": {
        'model_name'       : f"{MODEL_DIR}/random_init_models/gemma-2-0.6B",
        'eng_train_file'   : f"{DATASET_DIR}/pretrain/train_eng.jsonl",
        'secondary_train_files'   : [f"{DATASET_DIR}/pretrain/train_all_arithmetic.jsonl"],
        'eng_valid_file'   : f"{DATASET_DIR}/pretrain/valid_eng.jsonl",
        'interleave_probs' : [.75, .25],
        'output_dir'       : f"{MODEL_DIR}/pretrained_models/gemma-2-0.6B_all_arithmetic+eng",
        'cache_dir'        : CACHE_DIR,
        'dataset_cache_dir': CACHE_DIR,

        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 10,
        'gradient_accumulation_steps' : 36,
        'join_or_subsequence'         : True,
        'epochs'                      : 1,
        'learning_rate'               : 4e-4,
        'max_steps'                   : 1000,             
        'num_warmup_steps'            : 50,
        'validation_steps'            : 100,
        'save_checkpoint_steps'       : 500,
        'scheduler_type'              : "cosine",  
        'min_lr'                      : 4e-4,
        'weight_decay'                : 0,
        'gradient_clipping_threshold' : 1.0,
        'max_length'                  : 256,
        'use_wandb'        : True,
        'wandb_project'    : "gemma-2-0.6B_all_arithmetic+eng",
        'wandb_run_name'   : None,
        'wandb_api_key'    : api_key,
        'use_local_record' : True,
        'path_local_record': f"{MODEL_DIR}/local_records/pretrained_models/gemma-2-0.6B_all_arithmetic+eng.txt",   
    },
    "gemma-2-0.6B_addition_subtraction+eng": {
        'model_name'       : f"{MODEL_DIR}/random_init_models/gemma-2-0.6B",
        'eng_train_file'   : f"{DATASET_DIR}/pretrain/train_eng.jsonl",
        'secondary_train_files'   : [f"{DATASET_DIR}/pretrain/train_addition_subtraction.jsonl"],
        'eng_valid_file'   : f"{DATASET_DIR}/pretrain/valid_eng.jsonl",
        'interleave_probs' : [.875, .125],
        'output_dir'       : f"{MODEL_DIR}/pretrained_models/gemma-2-0.6B_addition_subtraction+eng",
        'cache_dir'        : CACHE_DIR,
        'dataset_cache_dir': CACHE_DIR,

        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 10,
        'gradient_accumulation_steps' : 36,
        'join_or_subsequence'         : True,
        'epochs'                      : 1,
        'learning_rate'               : 4e-4,
        'max_steps'                   : 1000,             
        'num_warmup_steps'            : 50,
        'validation_steps'            : 100,
        'save_checkpoint_steps'       : 500,
        'scheduler_type'              : "cosine",  
        'min_lr'                      : 4e-4,
        'weight_decay'                : 0,
        'gradient_clipping_threshold' : 1.0,
        'max_length'                  : 256,
        'use_wandb'        : True,
        'wandb_project'    : "gemma-2-0.6B_addition_subtraction+eng",
        'wandb_run_name'   : None,
        'wandb_api_key'    : api_key,
        'use_local_record' : True,
        'path_local_record': f"{MODEL_DIR}/local_records/pretrained_models/gemma-2-0.6B_addition_subtraction+eng.txt",   
    },
}

if __name__ == "__main__":
    for setup_id in SETUPS_TO_RUN:
        accelerator = Accelerator()

        arithmetic_eval_fn = get_arithmetic_eval_fn(
            # gets a function that takes a model returns a dicitonary with equation/word problem accuracty for each operation and english validation CE loss
            model_name          = setups[setup_id]['model_name'],
            eng_valid_file      = setups[setup_id]['eng_valid_file'],
            batch_size          = setups[setup_id]['batch_size'],
            max_length          = setups[setup_id]['max_length'],
            cache_dir           = setups[setup_id]['cache_dir'],
            dataset_cache_dir   = setups[setup_id]['dataset_cache_dir'],
            num_wiki_batches    = 50,
            accelerator         = accelerator
        )

        train(
            model_name       = setups[setup_id]['model_name'],
            train_files      = [setups[setup_id]['eng_train_file'], setups[setup_id]['secondary_train_files']],
            interleave_probs = setups[setup_id]['interleave_probs'],
            output_dir       = setups[setup_id]['output_dir'],
            cache_dir        = setups[setup_id]['cache_dir'],
            dataset_cache_dir= setups[setup_id]['dataset_cache_dir'],
            eval_fn          = arithmetic_eval_fn,
            accelerator      = accelerator,
            seed             = setups[setup_id]['seed'],
            device           = setups[setup_id]['device'],
            batch_size       = setups[setup_id]['batch_size'],
            join_or_subsequence   = setups[setup_id]['join_or_subsequence'],
            gradient_accumulation_steps = setups[setup_id]['gradient_accumulation_steps'],
            epochs           = setups[setup_id]['epochs'],
            learning_rate    = setups[setup_id]['learning_rate'],
            max_steps        = setups[setup_id]['max_steps'],   
            num_warmup_steps = setups[setup_id]['num_warmup_steps'],
            validation_steps = setups[setup_id]['validation_steps'],
            save_checkpoint_steps = setups[setup_id]['save_checkpoint_steps'],
            scheduler_type   = setups[setup_id]['scheduler_type'],  
            min_lr           = setups[setup_id]['min_lr'],          
            weight_decay     = setups[setup_id]['weight_decay'],    
            gradient_clipping_threshold = setups[setup_id]['gradient_clipping_threshold'], 
            max_length       = setups[setup_id]['max_length'],
            use_wandb        = setups[setup_id]['use_wandb'],
            wandb_project    = setups[setup_id]['wandb_project'],
            wandb_run_name   = setups[setup_id]['wandb_run_name'],
            wandb_api_key    = setups[setup_id]['wandb_api_key'],
            use_local_record = setups[setup_id]['use_local_record'],
            path_local_record= setups[setup_id]['path_local_record'],
        )