from src.tools.distill import distill
from src.utils.paths import CACHE_DIR, DATASET_DIR, MODEL_DIR, WANDB_API_KEY_PATH
from accelerate import Accelerator
from datasets import load_dataset
from utils.loss_functions import print_acc
from utils.validation_functions import get_korean_and_english_evalaution_fn
from utils.parallel_launch import launch_in_parallel_one_per_gpu, get_parallel_launch_wrapper

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding
)
from torch.utils.data import DataLoader

# Set which model/unlearning method combinations to run
#SETUPS_TO_RUN = ["gemma-2-0.1B_GradDiff", "gemma-2-0.1B_MaxEnt", "gemma-2-0.1B_RMU"]
SETUPS_TO_RUN = ["gemma-2-0.1B_MaxEnt"]#, "gemma-2-0.1B_RMU"]
USE_PARALLEL = False  # Flag to enable/disable parallel execution across GPUs

try:
    with open(WANDB_API_KEY_PATH, "r", encoding="utf-8") as f:
        api_key = f.read().strip()
except Exception as e:
    print(f"[ERROR] Unable to read WandB API key from {WANDB_API_KEY_PATH}. Exception: {e}")
    exit(1)


setups = {
    "gemma-2-0.1B_GradDiff": {  # renamed from ga
        'teacher_model_name': f"{MODEL_DIR}/unlearned_models/GradDiff/gemma-2-0.1B_eng+kor_lr_6.0e-05/final_model", 
        'student_model_name': f"{MODEL_DIR}/random_init_models/gemma-2-0.1B",
        'eng_train_file'    : f"{DATASET_DIR}/pretrain/train_eng.jsonl",
        'kor_train_file'    : f"{DATASET_DIR}/pretrain/train_kor.jsonl",
        'eng_valid_file'    : f"{DATASET_DIR}/pretrain/valid_eng.jsonl",
        'kor_valid_file'    : f"{DATASET_DIR}/pretrain/valid_kor.jsonl",
        'output_dir'        : f"{MODEL_DIR}/distilled_models/GradDiff/gemma-2-0.1B_eng+kor",
        'cache_dir'         : CACHE_DIR,
        'dataset_cache_dir' : CACHE_DIR,
        'join_or_subsequence': True,
        'interleave_probs'  : [.5, .5],

        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 4,
        'gradient_accumulation_steps' : 60,
        'epochs'                      : 1,
        'learning_rate'               : 9e-4,
        'max_steps'                   : 1000,
        'num_warmup_steps'            : 50,
        'validation_steps'            : 50,
        'save_checkpoint_steps'       : 500,
        'scheduler_type'              : "cosine",
        'min_lr'                      : 7e-4,
        'weight_decay'                : 0.1,
        'gradient_clipping_threshold' : 1.0, 
        'max_length'                  : 2048,

        'use_wandb'        : True,
        'wandb_project'    : "gemma-2-0.1B_eng+kor_GradDiff_distill",
        'wandb_run_name'   : None,
        'wandb_api_key'    : api_key,
        'use_local_record' : True,
        'path_local_record': f"{MODEL_DIR}/local_records/distilled_models/GradDiff/gemma-2-0.1B_eng+kor.txt",
    },
    "gemma-2-0.1B_MaxEnt": {  # renamed from uf
        'teacher_model_name': f"{MODEL_DIR}/unlearned_models/MaxEnt/gemma-2-0.1B_eng+kor_lr_3.0e-05/final_model",
        'student_model_name': f"{MODEL_DIR}/random_init_models/gemma-2-0.1B",
        'eng_train_file'    : f"{DATASET_DIR}/pretrain/train_eng.jsonl",
        'kor_train_file'    : f"{DATASET_DIR}/pretrain/train_kor.jsonl",
        'eng_valid_file'    : f"{DATASET_DIR}/pretrain/valid_eng.jsonl",
        'kor_valid_file'    : f"{DATASET_DIR}/pretrain/valid_kor.jsonl",
        'output_dir'        : f"{MODEL_DIR}/distilled_models/MaxEnt/gemma-2-0.1B_eng+kor",
        'cache_dir'         : CACHE_DIR,
        'dataset_cache_dir' : CACHE_DIR,
        'join_or_subsequence': True,
        'interleave_probs'  : [.5, .5],

        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 4,
        'gradient_accumulation_steps' : 60,
        'epochs'                      : 1,
        'learning_rate'               : 9e-4,       
        'max_steps'                   : 1000,             
        'num_warmup_steps'            : 50,
        'validation_steps'            : 50,
        'save_checkpoint_steps'       : 500,
        'scheduler_type'              : "cosine",  
        'min_lr'                      : 7e-4,              
        'weight_decay'                : 0.1,         
        'gradient_clipping_threshold' : 1.0, 
        'max_length'                  : 2048,

        'use_wandb'        : True,
        'wandb_project'    : "gemma-2-0.1B_eng+kor_MaxEnt_distill",
        'wandb_run_name'   : None,
        'wandb_api_key'    : api_key,
        'use_local_record' : True,
        'path_local_record': f"{MODEL_DIR}/local_records/distilled_models/MaxEnt/gemma-2-0.1B_eng+kor.txt",
    },
    "gemma-2-0.1B_RMU": {
        'teacher_model_name': f"{MODEL_DIR}/unlearned_models/RMU/gemma-2-0.1B_eng+kor_lr_3.0e-05/final_model",
        'student_model_name': f"{MODEL_DIR}/random_init_models/gemma-2-0.1B",
        'eng_train_file'    : f"{DATASET_DIR}/pretrain/train_eng.jsonl",
        'kor_train_file'    : f"{DATASET_DIR}/pretrain/train_kor.jsonl",
        'eng_valid_file'    : f"{DATASET_DIR}/pretrain/valid_eng.jsonl",
        'kor_valid_file'    : f"{DATASET_DIR}/pretrain/valid_kor.jsonl",
        'output_dir'        : f"{MODEL_DIR}/distilled_models/RMU/gemma-2-0.1B_eng+kor",
        'cache_dir'         : CACHE_DIR,
        'dataset_cache_dir' : CACHE_DIR,
        'join_or_subsequence': True,
        'interleave_probs'  : [.5, .5],

        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 4,
        'gradient_accumulation_steps' : 60,
        'epochs'                      : 1,
        'learning_rate'               : 9e-4,       
        'max_steps'                   : 1000,             
        'num_warmup_steps'            : 50,
        'validation_steps'            : 50,
        'save_checkpoint_steps'       : 500,
        'scheduler_type'              : "cosine",  
        'min_lr'                      : 7e-4,              
        'weight_decay'                : 0.1,         
        'gradient_clipping_threshold' : 1.0, 
        'max_length'                  : 2048,

        'use_wandb'        : True,
        'wandb_project'    : "gemma-2-0.1B_eng+kor_RMU_distill",
        'wandb_run_name'   : None,
        'wandb_api_key'    : api_key,
        'use_local_record' : True,
        'path_local_record': f"{MODEL_DIR}/local_records/distilled_models/RMU/gemma-2-0.1B_eng+kor.txt",
    },
}

def launch_distillation(setup_id):
    """Function to launch a distillation process for a specific setup"""
    accelerator = Accelerator()
    current_setup = setups[setup_id]

    # Create evaluation function for English and Korean
    english_korean_loss_eval_fn = get_korean_and_english_evalaution_fn(
        model_name          = current_setup['student_model_name'], 
        eng_valid_file      = current_setup['eng_valid_file'],
        kor_valid_file      = current_setup['kor_valid_file'],
        max_length          = current_setup['max_length'],
        dataset_cache_dir   = current_setup['dataset_cache_dir'], 
        cache_dir           = current_setup['cache_dir'],
        batch_size          = current_setup['batch_size'],
        accelerator         = accelerator,
    )
    
    # Start distillation process
    distill(
        teacher_model_name= current_setup['teacher_model_name'],
        student_model_name= current_setup['student_model_name'],
        train_files       = [current_setup['eng_train_file'], current_setup['kor_train_file']],
        interleave_probs  = current_setup['interleave_probs'],
        eval_fn           = english_korean_loss_eval_fn,
        accelerator       = accelerator,
        join_or_subsequence = current_setup.get('join_or_subsequence', True),
        output_dir        = current_setup['output_dir'],
        cache_dir         = current_setup['cache_dir'],
        dataset_cache_dir = current_setup['dataset_cache_dir'],
        seed              = current_setup['seed'],
        device            = current_setup['device'],
        batch_size        = current_setup['batch_size'],
        gradient_accumulation_steps = current_setup['gradient_accumulation_steps'],
        epochs           = current_setup['epochs'],
        learning_rate    = current_setup['learning_rate'],
        max_steps        = current_setup['max_steps'],   
        num_warmup_steps = current_setup['num_warmup_steps'],
        validation_steps = current_setup['validation_steps'],
        save_checkpoint_steps = current_setup['save_checkpoint_steps'],
        scheduler_type   = current_setup['scheduler_type'],  
        min_lr           = current_setup['min_lr'],          
        weight_decay     = current_setup['weight_decay'],    
        gradient_clipping_threshold = current_setup['gradient_clipping_threshold'], 
        max_length       = current_setup['max_length'],
        use_wandb        = current_setup['use_wandb'],
        wandb_project    = current_setup['wandb_project'],
        wandb_run_name   = current_setup['wandb_run_name'],
        wandb_api_key    = current_setup['wandb_api_key'],
        use_local_record = current_setup['use_local_record'],
        path_local_record= current_setup['path_local_record'],
    )


if __name__ == "__main__":
    # ----------------------------------------------------------------- #
    # Run all experiments, if possible in parallel
    # ----------------------------------------------------------------- #
    print(f"Running {len(SETUPS_TO_RUN)} distillation experiments:")
    for setup_id in SETUPS_TO_RUN:
        print(f"  - {setup_id}")
    
    # Create list of the setups (arguments for run_experiment) for all the experiments we want to run 
    experiments = [(setup_id,) for setup_id in SETUPS_TO_RUN]
    
    if USE_PARALLEL:
        # Gets a wrapper function compatable with the parallel launch function
        parallel_fn = get_parallel_launch_wrapper(launch_distillation)
        # calls run_experiment in parallel on a separate gpu for each experiment setup when a gpu is free
        launch_in_parallel_one_per_gpu(experiment_list=experiments, experiment_fn=parallel_fn)
    else:
        # Run experiments sequentially
        for experiment in experiments:
            setup_id = experiment[0]
            print(f"Running experiment with setup: {setup_id}")
            launch_distillation(setup_id)