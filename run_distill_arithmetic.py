from src.tools.distill import distill
from src.utils.paths import CACHE_DIR, DATASET_DIR, MODEL_DIR, WANDB_API_KEY_PATH
from accelerate import Accelerator
from datasets import load_dataset
from utils.loss_functions import print_acc
from utils.validation_functions import get_arithmetic_eval_fn
from utils.parallel_launch import launch_in_parallel_one_per_gpu, get_parallel_launch_wrapper
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding
)
from torch.utils.data import DataLoader

#SETUPS_TO_RUN = ["gemma-2-0.3B_base", "gemma-2-0.3B_GradDiff", "gemma-2-0.3B_RMU", "gemma-2-0.3B_MaxEnt",  "gemma-2-0.3B_pure", "gemma-2-0.3B_pure_from_base"]
#SETUPS_TO_RUN = ["gemma-2-0.3B_pure", "gemma-2-0.3B_pure_from_base", "gemma-2-0.3B_GradDiff", "gemma-2-0.3B_RMU", "gemma-2-0.3B_MaxEnt"]
SETUPS_TO_RUN = ["gemma-2-0.3B_GradDiff", "gemma-2-0.3B_MaxEnt", "gemma-2-0.3B_RMU"]
USE_PARALLEL = True  # Flag to enable/disable parallel execution across GPUs

try:
    with open(WANDB_API_KEY_PATH, "r", encoding="utf-8") as f:
        api_key = f.read().strip()
except Exception as e:
    print(f"[ERROR] Unable to read WandB API key from {WANDB_API_KEY_PATH}. Exception: {e}")
    exit(1)


setups = {
    "gemma-2-0.3B_GradDiff": {
        'teacher_model_name': f"{MODEL_DIR}/unlearned_models/GradDiff/gemma-2-0.3B_all_arithmetic+eng_lr_8.0e-06/final_model",
        'student_model_name': f"{MODEL_DIR}/random_init_models/gemma-2-0.3B",
        'eng_train_file'    : f"{DATASET_DIR}/pretrain/train_eng.jsonl",
        'arithmetic_train_file'   : f"{DATASET_DIR}/pretrain/train_all_arithmetic.jsonl",
        'eng_valid_file'    : f"{DATASET_DIR}/pretrain/valid_eng.jsonl",
        'output_dir'        : f"{MODEL_DIR}/distilled_models/GradDiff/gemma-2-0.3B_all_arithmetic+eng",
        'cache_dir'         : CACHE_DIR,
        'dataset_cache_dir' : CACHE_DIR,
        'join_or_subsequence': True,
        'interleave_probs'  : [.75, .25],

        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 15,
        'gradient_accumulation_steps' : 24,
        'epochs'                      : 1,
        'learning_rate'               : 7e-4, 
        'max_steps'                   : 1000,             
        'num_warmup_steps'            : 50,
        'validation_steps'            : 50,
        'save_checkpoint_steps'       : 500,
        'scheduler_type'              : "cosine",  
        'min_lr'                      : 7e-5,              
        'weight_decay'                : 0,         
        'gradient_clipping_threshold' : 1.0, 
        'max_length'                  : 256,

        'use_wandb'        : True,
        'wandb_project'    : "gemma-2-0.3B_all_arithmetic+eng_GradDiff_distill",
        'wandb_run_name'   : None,
        'wandb_api_key'    : api_key,
        'use_local_record' : True,
        'path_local_record': f"{MODEL_DIR}/local_records/distilled_models/GradDiff/gemma-2-0.3B_all_arithmetic+eng.txt",
    },
    "gemma-2-0.3B_RMU": {
        'teacher_model_name': f"{MODEL_DIR}/unlearned_models/RMU/gemma-2-0.3B_all_arithmetic+eng_lr_8.0e-06/final_model",
        'student_model_name': f"{MODEL_DIR}/random_init_models/gemma-2-0.3B",
        'eng_train_file'    : f"{DATASET_DIR}/pretrain/train_eng.jsonl",
        'arithmetic_train_file'   : f"{DATASET_DIR}/pretrain/train_all_arithmetic.jsonl",
        'eng_valid_file'    : f"{DATASET_DIR}/pretrain/valid_eng.jsonl",
        'output_dir'        : f"{MODEL_DIR}/distilled_models/RMU/gemma-2-0.3B_all_arithmetic+eng",
        'cache_dir'         : CACHE_DIR,
        'dataset_cache_dir' : CACHE_DIR,
        'join_or_subsequence': True,
        'interleave_probs'  : [.75, .25],

        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 15,
        'gradient_accumulation_steps' : 24,
        'epochs'                      : 1,
        'learning_rate'               : 7e-4, 
        'max_steps'                   : 1000,             
        'num_warmup_steps'            : 50,
        'validation_steps'            : 50,
        'save_checkpoint_steps'       : 500,
        'scheduler_type'              : "cosine",  
        'min_lr'                      : 7e-5,              
        'weight_decay'                : 0,         
        'gradient_clipping_threshold' : 1.0, 
        'max_length'                  : 256,

        'use_wandb'        : True,
        'wandb_project'    : "gemma-2-0.3B_all_arithmetic+eng_RMU_distill",
        'wandb_run_name'   : None,
        'wandb_api_key'    : api_key,
        'use_local_record' : True,
        'path_local_record': f"{MODEL_DIR}/local_records/distilled_models/RMU/gemma-2-0.3B_all_arithmetic+eng.txt",
    },
    "gemma-2-0.3B_MaxEnt": {
        'teacher_model_name': f"{MODEL_DIR}/unlearned_models/MaxEnt/gemma-2-0.3B_all_arithmetic+eng_lr_9.0e-05/final_model",
        'student_model_name': f"{MODEL_DIR}/random_init_models/gemma-2-0.3B",
        'eng_train_file'    : f"{DATASET_DIR}/pretrain/train_eng.jsonl",
        'arithmetic_train_file'   : f"{DATASET_DIR}/pretrain/train_all_arithmetic.jsonl",
        'eng_valid_file'    : f"{DATASET_DIR}/pretrain/valid_eng.jsonl",
        'output_dir'        : f"{MODEL_DIR}/distilled_models/MaxEnt/gemma-2-0.3B_all_arithmetic+eng",
        'cache_dir'         : CACHE_DIR,
        'dataset_cache_dir' : CACHE_DIR,
        'join_or_subsequence': True,
        'interleave_probs'  : [.75, .25],

        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 15,
        'gradient_accumulation_steps' : 24,
        'epochs'                      : 1,
        'learning_rate'               : 7e-4, 
        'max_steps'                   : 1000,             
        'num_warmup_steps'            : 50,
        'validation_steps'            : 50,
        'save_checkpoint_steps'       : 50000,
        'scheduler_type'              : "cosine",  
        'min_lr'                      : 7e-5,              
        'weight_decay'                : 0,         
        'gradient_clipping_threshold' : 1.0, 
        'max_length'                  : 256,

        'use_wandb'        : True,
        'wandb_project'    : "gemma-2-0.3B_all_arithmetic+eng_MaxEnt_distill",
        'wandb_run_name'   : None,
        'wandb_api_key'    : api_key,
        'use_local_record' : True,
        'path_local_record': f"{MODEL_DIR}/local_records/distilled_models/MaxEnt/gemma-2-0.3B_all_arithmetic+eng.txt",
    },
    "gemma-2-0.3B_pure": {
        'teacher_model_name': f"{MODEL_DIR}/pretrained_models/gemma-2-0.3B_addition_subtraction+eng/final_model",
        'student_model_name': f"{MODEL_DIR}/random_init_models/gemma-2-0.3B",
        'eng_train_file'    : f"{DATASET_DIR}/pretrain/train_eng.jsonl",
        'arithmetic_train_file'   : f"{DATASET_DIR}/pretrain/train_all_arithmetic.jsonl",
        'eng_valid_file'    : f"{DATASET_DIR}/pretrain/valid_eng.jsonl",
        'output_dir'        : f"{MODEL_DIR}/distilled_models/pure/gemma-2-0.3B_all_arithmetic+eng",
        'cache_dir'         : CACHE_DIR,
        'dataset_cache_dir' : CACHE_DIR,
        'join_or_subsequence': True,
        'interleave_probs'  : [.75, .25],

        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 15,
        'gradient_accumulation_steps' : 24,
        'epochs'                      : 1,
        'learning_rate'               : 7e-4, 
        'max_steps'                   : 1000,             
        'num_warmup_steps'            : 50,
        'validation_steps'            : 50,
        'save_checkpoint_steps'       : 500,
        'scheduler_type'              : "cosine",  
        'min_lr'                      : 7e-5,              
        'weight_decay'                : 0,         
        'gradient_clipping_threshold' : 1.0, 
        'max_length'                  : 256,

        'use_wandb'        : True,
        'wandb_project'    : "gemma-2-0.3B_all_arithmetic+eng_pure_distill",
        'wandb_run_name'   : None,
        'wandb_api_key'    : api_key,
        'use_local_record' : True,
        'path_local_record': f"{MODEL_DIR}/local_records/distilled_models/pure/gemma-2-0.3B_all_arithmetic+eng.txt",
    },
    "gemma-2-0.3B_pure_from_base": {
        'teacher_model_name': f"{MODEL_DIR}/pretrained_models/gemma-2-0.3B_addition_subtraction+eng/final_model",
        'student_model_name': f"{MODEL_DIR}/pretrained_models/gemma-2-0.3B_all_arithmetic+eng/final_model",
        'eng_train_file'    : f"{DATASET_DIR}/pretrain/train_eng.jsonl",
        'arithmetic_train_file'   : f"{DATASET_DIR}/pretrain/train_all_arithmetic.jsonl",
        'eng_valid_file'    : f"{DATASET_DIR}/pretrain/valid_eng.jsonl",
        'output_dir'        : f"{MODEL_DIR}/distilled_models/pure_from_base/gemma-2-0.3B_all_arithmetic+eng",
        'cache_dir'         : CACHE_DIR,
        'dataset_cache_dir' : CACHE_DIR,
        'join_or_subsequence': True,
        'interleave_probs'  : [.75, .25],

        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 15,
        'gradient_accumulation_steps' : 24,
        'epochs'                      : 1,
        'learning_rate'               : 7e-4, 
        'max_steps'                   : 1000,             
        'num_warmup_steps'            : 50,
        'validation_steps'            : 50,
        'save_checkpoint_steps'       : 500,
        'scheduler_type'              : "cosine",  
        'min_lr'                      : 7e-5,              
        'weight_decay'                : 0,         
        'gradient_clipping_threshold' : 1.0, 
        'max_length'                  : 256,

        'use_wandb'        : True,
        'wandb_project'    : "gemma-2-0.3B_all_arithmetic+eng_pure_distill_from_base",
        'wandb_run_name'   : None,
        'wandb_api_key'    : api_key,
        'use_local_record' : True,
        'path_local_record': f"{MODEL_DIR}/local_records/distilled_models/pure_from_base/gemma-2-0.3B_all_arithmetic+eng.txt",
    },
    "gemma-2-0.3B_base": {
        'teacher_model_name': f"{MODEL_DIR}/pretrained_models/gemma-2-0.3B_all_arithmetic+eng/final_model",
        'student_model_name': f"{MODEL_DIR}/random_init_models/gemma-2-0.3B",
        'eng_train_file'    : f"{DATASET_DIR}/pretrain/train_eng.jsonl",
        'arithmetic_train_file'   : f"{DATASET_DIR}/pretrain/train_all_arithmetic.jsonl",
        'eng_valid_file'    : f"{DATASET_DIR}/pretrain/valid_eng.jsonl",
        'output_dir'        : f"{MODEL_DIR}/distilled_models/base/gemma-2-0.3B_all_arithmetic+eng",
        'cache_dir'         : CACHE_DIR,
        'dataset_cache_dir' : CACHE_DIR,
        'join_or_subsequence': True,
        'interleave_probs'  : [.75, .25],

        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 15,
        'gradient_accumulation_steps' : 24,
        'epochs'                      : 2,
        'learning_rate'               : 1e-4, 
        'max_steps'                   : 2000,             
        'num_warmup_steps'            : 50,
        'validation_steps'            : 50,
        'save_checkpoint_steps'       : 500,
        'scheduler_type'              : "cosine",  
        'min_lr'                      : 1e-4,              
        'weight_decay'                : 0,         
        'gradient_clipping_threshold' : 1.0, 
        'max_length'                  : 256,

        'use_wandb'        : True,
        'wandb_project'    : "gemma-2-0.3B_all_arithmetic+eng_base_distill",
        'wandb_run_name'   : None,
        'wandb_api_key'    : api_key,
        'use_local_record' : True,
        'path_local_record': f"{MODEL_DIR}/local_records/distilled_models/base/gemma-2-0.3B_all_arithmetic+eng.txt",
    },
}

def launch_distillation(setup_id):
    accelerator = Accelerator()
    current_setup = setups[setup_id]

    arithmetic_eval_fn = get_arithmetic_eval_fn(
        # gets a function that takes a model returns a dicitonary with equation/word problem accuracty for each operation and english validation CE loss
        model_name          = current_setup['student_model_name'],
        eng_valid_file      = current_setup['eng_valid_file'],
        batch_size          = current_setup['batch_size'],
        max_length          = current_setup['max_length'],
        cache_dir           = current_setup['cache_dir'],
        dataset_cache_dir   = current_setup['dataset_cache_dir'],
        num_wiki_batches    = 50,
        accelerator         = accelerator
    )
    distill(
        teacher_model_name= current_setup['teacher_model_name'],
        student_model_name= current_setup['student_model_name'],
        train_files       = [current_setup['eng_train_file'], current_setup['arithmetic_train_file']],
        interleave_probs  = current_setup['interleave_probs'],
        eval_fn = arithmetic_eval_fn,
        accelerator = accelerator,
        output_dir        = current_setup['output_dir'],
        cache_dir         = current_setup['cache_dir'],
        dataset_cache_dir = current_setup['dataset_cache_dir'],
        seed              = current_setup['seed'],
        device            = current_setup['device'],
        batch_size        = current_setup['batch_size'],
        gradient_accumulation_steps = current_setup['gradient_accumulation_steps'],
        join_or_subsequence = current_setup['join_or_subsequence'],
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