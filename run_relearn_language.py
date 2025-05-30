"""
Configured for a (any)xH200 GPU server.

Launch Command: python run_relearn_language.py
"""

from src.tools.relearn_langarith import relearn
from src.utils.paths import CACHE_DIR, DATASET_DIR, MODEL_DIR, WANDB_API_KEY_PATH
from accelerate import Accelerator
from utils.loss_functions import custom_login
from utils.loss_functions import print_acc
from utils.validation_functions import get_korean_and_english_evalaution_fn
from utils.parallel_launch import launch_in_parallel_one_per_gpu, get_parallel_launch_wrapper

SETUPS_TO_RUN = [
    "gemma-2-0.1B_train_only_forget"
]

MODELS_TO_RUN = [
    #'pretrained_models/gemma-2-0.1B_eng', # Pretrain Pure
    #'pretrained_models/gemma-2-0.1B_eng+kor', # Pretrained Base
    #'unlearned_models/GradDiff/gemma-2-0.1B_eng+kor_lr_6.0e-05', # Unlearned GA
    #'unlearned_models/RMU/gemma-2-0.1B_eng+kor_lr_3.0e-05', # Unlearned RMU
    #'unlearned_models/MaxEnt/gemma-2-0.1B_eng+kor_lr_3.0e-05', # Unlearned UF

    #'distilled_models/pure/gemma-2-0.1B_pure', # Distilled Pure
    #'distilled_models/base/gemma-2-0.1B_impure', # Distilled Base
    #'distilled_models/GradDiff/gemma-2-0.1B_eng+kor', # Distilled GA
    #'distilled_models/RMU/gemma-2-0.1B_eng+kor', # Distilled RMU
    #'distilled_models/MaxEnt/gemma-2-0.1B_eng+kor', # Distilled UF

    #'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.1-beta_0.1',  # partial_distill
    #'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.3-beta_0.1',  # partial_distill
    #'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.5-beta_0.1',  # partial_distill
    #'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.7-beta_0.1',  # partial_distill
    #'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.1-beta_0.5',  # partial_distill
    #'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.3-beta_0.5',  # partial_distill
    #'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.5-beta_0.5',  # partial_distill
    #'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.7-beta_0.5',  # partial_distill
    #'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.1-beta_1.0',  # partial_distill
    #'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.3-beta_1.0',  # partial_distill
    #'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.5-beta_1.0',  # partial_distill
    #'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.7-beta_1.0',  # partial_distill

    'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.1-beta_0.1-seed_111-fixed_steps_500',  # partial_distill
    'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.2-beta_0.1-seed_111-fixed_steps_500',  # partial_distill
    'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.3-beta_0.1-seed_111-fixed_steps_500',  # partial_distill
    'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.4-beta_0.1-seed_111-fixed_steps_500',  # partial_distill
    'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.5-beta_0.1-seed_111-fixed_steps_500',  # partial_distill
    'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.6-beta_0.1-seed_111-fixed_steps_500',  # partial_distill
    'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.7-beta_0.1-seed_111-fixed_steps_500',  # partial_distill
    'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.8-beta_0.1-seed_111-fixed_steps_500',  # partial_distill
]
# 1e-3, 7e-4, 4e-4, 1e-4, 7e-5, 5e-5, 1e-5, 7e-6, 4e-6, 1e-6
# , 4e-6, 1e-6

'''
for pareto plot
'''
#LR_RANGES = {
#    "GradDiff": [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5],
#    "MaxEnt": [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5],
#    "RMU": [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5]
#}
#
#for method, lr_values in LR_RANGES.items():
#    for lr in lr_values:
#        # Format the learning rate with scientific notation
#        formatted_lr = f"{lr:.1e}"
#        # Add the model path to the list
#        model_path = f"unlearned_models/{method}/gemma-2-0.1B_eng+kor_lr_{formatted_lr}"
#        MODELS_TO_RUN.append(model_path)
#
#for model in MODELS_TO_RUN:
#    print(model)

# relearning learning rate (search adversary)
LRS_TO_RUN =[1e-3, 7e-4, 4e-4, 1e-4, 7e-5, 5e-5, 1e-5, 7e-6, 4e-6, 1e-6]

try:
    with open(WANDB_API_KEY_PATH, "r", encoding="utf-8") as f:
        api_key = f.read().strip()
except Exception as e:
    print(f"[ERROR] Unable to read WandB API key from {WANDB_API_KEY_PATH}. Exception: {e}")
    exit(1)

custom_login()

setups = {
    "gemma-2-0.1B_train_only_forget": {
        'model_name'       : f"{MODEL_DIR}/model_path/final_student",
        'first_train_file'   : f"{DATASET_DIR}/pretrain/train_kor.jsonl",
        'second_train_file': "",
        'interleave_probs': [1],
        'eng_valid_file'   : f"{DATASET_DIR}/pretrain/valid_eng.jsonl",
        'kor_valid_file'   : f"{DATASET_DIR}/pretrain/valid_kor.jsonl",
        'output_dir'       : f"{MODEL_DIR}/relearned_models/model_name_only_forget_data",
        'cache_dir'        : CACHE_DIR,
        'dataset_cache_dir': CACHE_DIR,
        'join_or_subsequence'         : True,
        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 8,
        'gradient_accumulation_steps' : 1,
        'epochs'                      : 1,
        'learning_rate'               : None,       
        'max_steps'                   : 500,             
        'num_warmup_steps'            : 1,
        'validation_steps'            : 5,
        'save_checkpoint_steps'       : 999,
        'scheduler_type'              : "cosine", 
        
        'min_lr'                      : None,              
        'weight_decay'                : 0.0,         
        'gradient_clipping_threshold' : 1.0, 
        'max_length'                  : 2048,

        'use_wandb'        : True,
        'wandb_project'    : "gemma-2-0.1B_eng+kor",
        'wandb_run_name'   : None,
        'use_local_record' : True,
        'path_local_record': f"{MODEL_DIR}/local_records/relearned_models/model_name_only_forget_data.txt",

        'save_models'      : False
    }
}

def launch_relearn(setup_id, lr, model):
    current_setup = setups[setup_id].copy()
    current_setup['learning_rate'] = lr
    current_setup['min_lr'] = lr
    current_setup['model_name'] = current_setup['model_name'].replace('model_path', model)
    if "distilled" not in current_setup['model_name']:
        current_setup['model_name'] = current_setup['model_name'].replace('final_student', "final_model")
    model_name = model_name = f"relearned_{model.replace('/', '_')}/{lr}" # Get the substring past the first '/'
    current_setup['path_local_record'] = current_setup['path_local_record'].replace('model_name', model_name)
    current_setup['output_dir'] = current_setup['output_dir'].replace('model_name', model_name)
    
    accelerator = Accelerator()

    language_eval_fn = get_korean_and_english_evalaution_fn(
        model_name          = current_setup['model_name'],
        max_length          = current_setup['max_length'],
        eng_valid_file      = current_setup['eng_valid_file'],
        kor_valid_file      = current_setup['kor_valid_file'],
        dataset_cache_dir   = current_setup['dataset_cache_dir'],
        cache_dir           = current_setup['cache_dir'],
        batch_size          = current_setup['batch_size'],
        accelerator         = accelerator
    )
    if not current_setup['second_train_file'] == "":
        train_files = [current_setup['first_train_file'], current_setup['second_train_file']]
    else:
        train_files = [current_setup['first_train_file']]
    relearn(
        model_name       = current_setup['model_name'],
        train_files      = train_files,
        interleave_probs = current_setup['interleave_probs'],
        output_dir       = current_setup['output_dir'],
        cache_dir        = current_setup['cache_dir'],
        dataset_cache_dir= current_setup['dataset_cache_dir'],
        eval_fn          = language_eval_fn,
        accelerator      = accelerator,
        join_or_subsequence   = current_setup['join_or_subsequence'],
        seed             = current_setup['seed'],
        device           = current_setup['device'],
        batch_size       = current_setup['batch_size'],
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
        use_local_record = current_setup['use_local_record'],
        path_local_record= current_setup['path_local_record'],
        save_models      = current_setup['save_models']
    )

if __name__ == "__main__":
    # ----------------------------------------------------------------- #
    # Run all experiments, if possible in parallel
    # ----------------------------------------------------------------- #
    # Create list of the setups (arguments for run_experiment) for all the experiments we want to run 
    experiments = [(setup_id, lr, model) for setup_id in SETUPS_TO_RUN for lr in LRS_TO_RUN for model in MODELS_TO_RUN]
    # Gets a wrapper function compatable with the parallel launch function
    parallel_fn = get_parallel_launch_wrapper(launch_relearn)
    # calls run_experiment in parallel on a separate gpu for each experiment setup when a gpu is free
    launch_in_parallel_one_per_gpu(experiment_list=experiments, experiment_fn=parallel_fn)
    