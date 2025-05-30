from src.tools.unlearn_langarith.graddiff import unlearn_graddiff  # renamed from ga
from src.tools.unlearn_langarith.maxent import unlearn_maxent      # renamed from uf
from src.tools.unlearn_langarith.rmu import unlearn_rmu            # unchanged
from accelerate import Accelerator
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.paths import CACHE_DIR, DATASET_DIR, MODEL_DIR, WANDB_API_KEY_PATH
from src.utils.validation_functions import get_korean_and_english_evalaution_fn
from src.utils.parallel_launch import launch_in_parallel_one_per_gpu, get_parallel_launch_wrapper

# Define learning rate ranges for each method

LR_RANGES = {
    "GradDiff": [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5],
    "MaxEnt": [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5],
    "RMU": [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5]
}

# Base setups to run - will be expanded with learning rates
BASE_SETUPS = ["gemma-2-0.1B_GradDiff", "gemma-2-0.1B_MaxEnt", "gemma-2-0.1B_RMU"]

try:
    with open(WANDB_API_KEY_PATH, "r", encoding="utf-8") as f:
        api_key = f.read().strip()
except Exception as e:
    print(f"[ERROR] Unable to read WandB API key from {WANDB_API_KEY_PATH}. Exception: {e}")
    exit(1)

base_setups = {
    "gemma-2-0.1B_GradDiff": {  # renamed from ga
        'model_name'       : f"{MODEL_DIR}/pretrained_models/gemma-2-0.1B_eng+kor/final_model",
        'forget_train_file': f"{DATASET_DIR}/pretrain/train_kor.jsonl",
        'retain_train_file': f"{DATASET_DIR}/pretrain/train_eng.jsonl",
        'eng_valid_file'   : f"{DATASET_DIR}/pretrain/valid_eng.jsonl",
        'kor_valid_file'   : f"{DATASET_DIR}/pretrain/valid_kor.jsonl",
        'output_dir'       : f"{MODEL_DIR}/unlearned_models/GradDiff/gemma-2-0.1B_eng+kor",
        'cache_dir'        : CACHE_DIR,
        'dataset_cache_dir': CACHE_DIR,

        'ga_gd'                       : True,
        'alpha'                       : 1, # Added parameter from arithmetic script
        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 4,
        'gradient_accumulation_steps' : 1,
        'epochs'                      : 2,
        'learning_rate'               : "TBD",  # Will be filled by LR search
        'max_steps'                   : 100,             
        'num_warmup_steps'            : 0,
        'validation_steps'            : 4,
        'save_checkpoint_steps'       : 5,
        'scheduler_type'              : "cosine",  
        'min_lr'                      : "TBD",  # Will be filled by LR search
        'weight_decay'                : 0.0,         
        'gradient_clipping_threshold' : 1.0, 
        'max_length'                  : 2048,

        'use_wandb'        : True,
        'wandb_project'    : "gemma-2-0.1B_eng+kor_unlearn_GradDiff",
        'wandb_run_name'   : None,
        'wandb_api_key'    : api_key,
        'use_local_record' : True,
        'path_local_record': f"{MODEL_DIR}/local_records/unlearned_models/GradDiff/gemma-2-0.1B_eng+kor.txt",
    },
    "gemma-2-0.1B_MaxEnt": {  # renamed from uf
        'model_name'       : f"{MODEL_DIR}/pretrained_models/gemma-2-0.1B_eng+kor/final_model",
        'forget_train_file': f"{DATASET_DIR}/pretrain/train_kor.jsonl",
        'retain_train_file': f"{DATASET_DIR}/pretrain/train_eng.jsonl",
        'eng_valid_file'   : f"{DATASET_DIR}/pretrain/valid_eng.jsonl",
        'kor_valid_file'   : f"{DATASET_DIR}/pretrain/valid_kor.jsonl",
        'output_dir'       : f"{MODEL_DIR}/unlearned_models/MaxEnt/gemma-2-0.1B_eng+kor",
        'cache_dir'        : CACHE_DIR,
        'dataset_cache_dir': CACHE_DIR,

        'use_retain'                  : True,
        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 4,
        'gradient_accumulation_steps' : 1,
        'epochs'                      : 2,
        'learning_rate'               : "TBD",  # Will be filled by LR search
        'max_steps'                   : 100,             
        'num_warmup_steps'            : 0,
        'validation_steps'            : 4,
        'save_checkpoint_steps'       : -1,
        'scheduler_type'              : "cosine",  
        'min_lr'                      : "TBD",  # Will be filled by LR search
        'weight_decay'                : 0.0,         
        'gradient_clipping_threshold' : 1.0, 
        'max_length'                  : 2048,

        'use_wandb'        : True,
        'wandb_project'    : "gemma-2-0.1B_eng+kor_unlearn_MaxEnt",
        'wandb_run_name'   : None,
        'wandb_api_key'    : api_key,
        'use_local_record' : True,
        'path_local_record': f"{MODEL_DIR}/local_records/unlearned_models/MaxEnt/gemma-2-0.1B_eng+kor.txt",
    },
    "gemma-2-0.1B_RMU": {
        'model_name'       : f"{MODEL_DIR}/pretrained_models/gemma-2-0.1B_eng+kor/final_model",
        'forget_train_file': f"{DATASET_DIR}/pretrain/train_kor.jsonl",
        'retain_train_file': f"{DATASET_DIR}/pretrain/train_eng.jsonl",
        'eng_valid_file'   : f"{DATASET_DIR}/pretrain/valid_eng.jsonl",
        'kor_valid_file'   : f"{DATASET_DIR}/pretrain/valid_kor.jsonl",
        'output_dir'       : f"{MODEL_DIR}/unlearned_models/RMU/gemma-2-0.1B_eng+kor",
        'cache_dir'        : CACHE_DIR,
        'dataset_cache_dir': CACHE_DIR,

        'ga_gd'                       : False,
        'rmu_layers'                  : [5, 6, 7, 8, 9, 10, 11],
        'end_layer'                   : 11, 
        'alpha'                       : 1200,
        'c'                           : 6.5,
        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 4,
        'gradient_accumulation_steps' : 1,
        'epochs'                      : 2,
        'learning_rate'               : "TBD",  # Will be filled by LR search
        'max_steps'                   : 500,             
        'num_warmup_steps'            : 0,
        'validation_steps'            : 20,
        'save_checkpoint_steps'       : 999,
        'scheduler_type'              : "cosine",  
        'min_lr'                      : "TBD",  # Will be filled by LR search
        'weight_decay'                : 0.0,         
        'gradient_clipping_threshold' : 1.0, 
        'max_length'                  : 2048,

        'use_wandb'        : True,
        'wandb_project'    : "gemma-2-0.1B_eng+kor_unlearn_RMU",
        'wandb_run_name'   : None,
        'wandb_api_key'    : api_key,
        'use_local_record' : True,
        'path_local_record': f"{MODEL_DIR}/local_records/unlearned_models/RMU/gemma-2-0.1B_eng+kor.txt",
    },
}

def create_lr_variant(base_setup_id, learning_rate):
    """Create a variant of a base setup with a different learning rate"""
    method = base_setup_id.split('_')[-1]  # Extract method name (GradDiff, MaxEnt, or RMU)
    new_setup_id = f"{base_setup_id}_lr_{learning_rate:.1e}"
    
    # Clone the base setup
    setup_config = base_setups[base_setup_id].copy()
    
    # Update learning rate related parameters
    setup_config['learning_rate'] = learning_rate
    setup_config['min_lr'] = learning_rate  # Also update min_lr to match
    
    # Update paths to include learning rate in directory/file names
    setup_config['output_dir'] = f"{setup_config['output_dir']}_lr_{learning_rate:.1e}"
    setup_config['path_local_record'] = setup_config['path_local_record'].replace('.txt', f'_lr_{learning_rate:.1e}.txt')
    
    # Update wandb run name to include learning rate
    setup_config['wandb_run_name'] = f"lr_{learning_rate:.1e}"
    
    return new_setup_id, setup_config

# Generate all setup variants with different learning rates
setups = {}
SETUPS_TO_RUN = []

for base_setup_id in BASE_SETUPS:
    method = base_setup_id.split('_')[-1]  # Extract method name
    
    # Get the appropriate learning rate range for this method
    lr_range = LR_RANGES[method]
    
    # Create a variant for each learning rate
    for lr in lr_range:
        new_setup_id, setup_config = create_lr_variant(base_setup_id, lr)
        setups[new_setup_id] = setup_config
        SETUPS_TO_RUN.append(new_setup_id)

def launch_unlearning_run(setup_id):
    accelerator = Accelerator()
    language_eval_fn = get_korean_and_english_evalaution_fn(
        model_name        = setups[setup_id]['model_name'],
        max_length        = setups[setup_id]['max_length'],
        eng_valid_file    = setups[setup_id]['eng_valid_file'],
        kor_valid_file    = setups[setup_id]['kor_valid_file'],
        dataset_cache_dir = setups[setup_id]['dataset_cache_dir'],
        cache_dir         = setups[setup_id]['cache_dir'],
        batch_size        = setups[setup_id]['batch_size'],
        accelerator       = accelerator
    )

    if '_GradDiff' in setup_id:
        unlearn_graddiff(
            model_name       = setups[setup_id]['model_name'],
            forget_train_file= setups[setup_id]['forget_train_file'],
            retain_train_file= setups[setup_id]['retain_train_file'],
            eval_fn          = language_eval_fn,
            accelerator      = accelerator,
            output_dir       = setups[setup_id]['output_dir'],
            cache_dir        = setups[setup_id]['cache_dir'],
            dataset_cache_dir= setups[setup_id]['dataset_cache_dir'],
            ga_gd            = setups[setup_id]['ga_gd'],
            alpha            = setups[setup_id]['alpha'],
            seed             = setups[setup_id]['seed'],
            device           = setups[setup_id]['device'],
            batch_size       = setups[setup_id]['batch_size'],
            gradient_accumulation_steps = setups[setup_id]['gradient_accumulation_steps'],
            join_or_subsequence         = True,
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
    elif '_MaxEnt' in setup_id:
        print(f"Running MaxEnt with learning rate {setups[setup_id]['learning_rate']}")
        unlearn_maxent(
            model_name       = setups[setup_id]['model_name'],
            forget_train_file= setups[setup_id]['forget_train_file'],
            retain_train_file= setups[setup_id]['retain_train_file'],
            eval_fn          = language_eval_fn,
            accelerator      = accelerator,
            join_or_subsequence = True,
            output_dir       = setups[setup_id]['output_dir'],
            cache_dir        = setups[setup_id]['cache_dir'],
            dataset_cache_dir= setups[setup_id]['dataset_cache_dir'],
            use_retain       = setups[setup_id]['use_retain'],
            seed             = setups[setup_id]['seed'],
            device           = setups[setup_id]['device'],
            batch_size       = setups[setup_id]['batch_size'],
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
    elif '_RMU' in setup_id:
        unlearn_rmu(
            model_name       = setups[setup_id]['model_name'],
            forget_train_file= setups[setup_id]['forget_train_file'],
            retain_train_file= setups[setup_id]['retain_train_file'],
            eval_fn          = language_eval_fn,
            accelerator      = accelerator,
            join_or_subsequence = True,
            output_dir       = setups[setup_id]['output_dir'],
            cache_dir        = setups[setup_id]['cache_dir'],
            dataset_cache_dir= setups[setup_id]['dataset_cache_dir'],
            ga_gd            = setups[setup_id]['ga_gd'],
            rmu_layers       = setups[setup_id]['rmu_layers'],
            end_layer        = setups[setup_id]['end_layer'],
            alpha            = setups[setup_id]['alpha'],
            c                = setups[setup_id]['c'],
            seed             = setups[setup_id]['seed'],
            device           = setups[setup_id]['device'],
            batch_size       = setups[setup_id]['batch_size'],
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

if __name__ == "__main__":
    # ----------------------------------------------------------------- #
    # Run all experiments, if possible in parallel
    # ----------------------------------------------------------------- #
    print(f"Running {len(SETUPS_TO_RUN)} experiments with learning rate search:")
    for setup_id in SETUPS_TO_RUN:
        print(f"  - {setup_id} (LR: {setups[setup_id]['learning_rate']:.1e})")
    
    # Create list of the setups (arguments for run_experiment) for all the experiments we want to run 
    experiments = [(setup_id,) for setup_id in SETUPS_TO_RUN]
    # Gets a wrapper function compatable with the parallel launch function
    parallel_fn = get_parallel_launch_wrapper(launch_unlearning_run)
    # calls run_experiment in parallel on a separate gpu for each experiment setup when a gpu is free
    launch_in_parallel_one_per_gpu(experiment_list=experiments, experiment_fn=parallel_fn)