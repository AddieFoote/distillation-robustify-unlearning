"""
Configured for a (any)xA100 GPU server.

Launch Command: python run_relearn_arithmetic.py
"""

from src.tools.relearn_langarith import relearn
from src.utils.paths import CACHE_DIR, DATASET_DIR, MODEL_DIR, WANDB_API_KEY_PATH
from accelerate import Accelerator
from utils.loss_functions import print_acc
from utils.validation_functions import get_arithmetic_eval_fn
from utils.parallel_launch import launch_in_parallel_one_per_gpu, get_parallel_launch_wrapper


SETUPS_TO_RUN = [
    "gemma-2-0.3B_all_data" # "gemma-2-0.3B_all_data_perturb", "gemma-2-0.3B_train_only_forget", "gemma-2-0.3B_all_data"
]

MODELS_TO_RUN = [
    #'pretrained_models/gemma-2-0.3B_addition_subtraction+eng', # Pretrain Pure
    #'pretrained_models/gemma-2-0.3B_all_arithmetic+eng', # Pretrained Base
    #'unlearned_models/GradDiff/gemma-2-0.3B_all_arithmetic+eng_lr_8.0e-06', # Unlearned GradDiff
    #'unlearned_models/RMU/gemma-2-0.3B_all_arithmetic+eng_lr_6.0e-06', # Unlearned RMU
    #'unlearned_models/MaxEnt/gemma-2-0.3B_all_arithmetic+eng_lr_9.0e-05', # Unlearned MaxEnt

    #'distilled_models/pure/gemma-2-0.3B_all_arithmetic+eng', # Distilled Pure (Oracle)
    #'distilled_models/pure_from_base/gemma-2-0.3B_all_arithmetic+eng', # Distilled Impure (Oracle)
    #'distilled_models/base/gemma-2-0.3B_all_arithmetic+eng', # Distilled Base
    #'distilled_models/GradDiff/gemma-2-0.3B_all_arithmetic+eng', # Distilled GradDiff
    #'distilled_models/RMU/gemma-2-0.3B_all_arithmetic+eng', # Distilled RMU
    # 'distilled_models/MaxEnt/gemma-2-0.3B_all_arithmetic+eng', # Distilled MaxEnt

    #'partial_distill_models_arith/gemma-2-0.3B_MaxEnt-arithmetic-partial_distill-alpha_0.1-beta_0.1-seed_123',  # partial_distill
    #'partial_distill_models_arith/gemma-2-0.3B_MaxEnt-arithmetic-partial_distill-alpha_0.2-beta_0.1-seed_123',  # partial_distill
    #'partial_distill_models_arith/gemma-2-0.3B_MaxEnt-arithmetic-partial_distill-alpha_0.3-beta_0.1-seed_123',  # partial_distill
    #'partial_distill_models_arith/gemma-2-0.3B_MaxEnt-arithmetic-partial_distill-alpha_0.4-beta_0.1-seed_123',  # partial_distill
    #'partial_distill_models_arith/gemma-2-0.3B_MaxEnt-arithmetic-partial_distill-alpha_0.5-beta_0.1-seed_123',  # partial_distill
    #'partial_distill_models_arith/gemma-2-0.3B_MaxEnt-arithmetic-partial_distill-alpha_0.6-beta_0.1-seed_123',  # partial_distill
    #'partial_distill_models_arith/gemma-2-0.3B_MaxEnt-arithmetic-partial_distill-alpha_0.7-beta_0.1-seed_123',  # partial_distill
    #'partial_distill_models_arith/gemma-2-0.3B_MaxEnt-arithmetic-partial_distill-alpha_0.8-beta_0.1-seed_123',  # partial_distill

    #'partial_distill_models_arith/gemma-2-0.3B_MaxEnt-arithmetic-partial_distill-alpha_0.1-beta_0.1-seed_111-fixed_steps_500',  # partial_distill
    #'partial_distill_models_arith/gemma-2-0.3B_MaxEnt-arithmetic-partial_distill-alpha_0.2-beta_0.1-seed_111-fixed_steps_500',  # partial_distill
    #'partial_distill_models_arith/gemma-2-0.3B_MaxEnt-arithmetic-partial_distill-alpha_0.3-beta_0.1-seed_111-fixed_steps_500',  # partial_distill
    #'partial_distill_models_arith/gemma-2-0.3B_MaxEnt-arithmetic-partial_distill-alpha_0.4-beta_0.1-seed_111-fixed_steps_500',  # partial_distill
    #'partial_distill_models_arith/gemma-2-0.3B_MaxEnt-arithmetic-partial_distill-alpha_0.5-beta_0.1-seed_111-fixed_steps_500',  # partial_distill
    #'partial_distill_models_arith/gemma-2-0.3B_MaxEnt-arithmetic-partial_distill-alpha_0.6-beta_0.1-seed_111-fixed_steps_500',  # partial_distill
    #'partial_distill_models_arith/gemma-2-0.3B_MaxEnt-arithmetic-partial_distill-alpha_0.7-beta_0.1-seed_111-fixed_steps_500',  # partial_distill
    #'partial_distill_models_arith/gemma-2-0.3B_MaxEnt-arithmetic-partial_distill-alpha_0.8-beta_0.1-seed_111-fixed_steps_500',  # partial_distill

    #'unlearned_models/MaxEnt_RepNoise/gemma-2-0.3B_all_arithmetic+eng_a0.1_b0.001_lr_8.0e-05',
    #'unlearned_models/MaxEnt_RepNoise/gemma-2-0.3B_all_arithmetic+eng_a0.1_b0.0001_lr_8.0e-05',
    #'unlearned_models/MaxEnt_RepNoise/gemma-2-0.3B_all_arithmetic+eng_a0.5_b0.1_lr_8.0e-05',
    #'unlearned_models/MaxEnt_RepNoise/gemma-2-0.3B_all_arithmetic+eng_a0.5_b0.01_lr_8.0e-05',
    #'unlearned_models/MaxEnt_RepNoise/gemma-2-0.3B_all_arithmetic+eng_a1.0_b0.1_lr_8.0e-05',
    #'unlearned_models/MaxEnt_RepNoise/gemma-2-0.3B_all_arithmetic+eng_a1.0_b1.0_lr_8.0e-05',
    #'unlearned_models/MaxEnt_RepNoise/gemma-2-0.3B_all_arithmetic+eng_a2.0_b2.0_lr_8.0e-05',
    #'unlearned_models/MaxEnt_RepNoise/gemma-2-0.3B_all_arithmetic+eng_a4.0_b4.0_lr_8.0e-05',

    'unlearned_models/RMU_SAM/gemma-2-0.3B_all_arithmetic+eng_rho0.01_lr_1.0e-05',
    'unlearned_models/RMU_SAM/gemma-2-0.3B_all_arithmetic+eng_rho0.01_lr_2.0e-05',
    'unlearned_models/RMU_SAM/gemma-2-0.3B_all_arithmetic+eng_rho0.01_lr_3.0e-05',
    'unlearned_models/RMU_SAM/gemma-2-0.3B_all_arithmetic+eng_rho0.01_lr_4.0e-05',
    'unlearned_models/RMU_SAM/gemma-2-0.3B_all_arithmetic+eng_rho0.01_lr_6.0e-06',
    'unlearned_models/RMU_SAM/gemma-2-0.3B_all_arithmetic+eng_rho0.01_lr_7.0e-06',
    'unlearned_models/RMU_SAM/gemma-2-0.3B_all_arithmetic+eng_rho0.01_lr_8.0e-06',
    'unlearned_models/RMU_SAM/gemma-2-0.3B_all_arithmetic+eng_rho0.01_lr_9.0e-06',
    
]
'''
for pareto plot
'''
#LR_RANGES = {
#    "GradDiff": [6e-6, 7e-6, 8e-6, 9e-6, 1e-5, 2e-5, 3e-5, 4e-5],
#    "MaxEnt": [6e-5, 7e-5, 8e-5, 9e-5, 1e-4, 2e-4, 3e-4, 4e-4],
#    "RMU": [6e-6, 7e-6, 8e-6, 9e-6, 1e-5, 2e-5, 3e-5, 4e-5]
#}
#for method, lr_values in LR_RANGES.items():
#    for lr in lr_values:
#        # Format the learning rate with scientific notation
#        formatted_lr = f"{lr:.1e}"
#        # Add the model path to the list
#        model_path = f"unlearned_models/{method}/gemma-2-0.3B_all_arithmetic+eng_lr_{formatted_lr}"
#        MODELS_TO_RUN.append(model_path)
#
#for model in MODELS_TO_RUN:
#    print(model)

# relearning learning rate
LRS_TO_RUN =[1e-3, 7e-4, 4e-4, 1e-4, 7e-5, 5e-5, 1e-5, 7e-6, 4e-6, 1e-6]

try:
    with open(WANDB_API_KEY_PATH, "r", encoding="utf-8") as f:
        api_key = f.read().strip()
except Exception as e:
    print(f"[ERROR] Unable to read WandB API key from {WANDB_API_KEY_PATH}. Exception: {e}")
    exit(1)

setups = {
    "gemma-2-0.3B_train_only_forget": {
        'model_name'       : f"{MODEL_DIR}/model_path/final_student",
        'first_train_file'   : f"{DATASET_DIR}/pretrain/train_addition_subtraction.jsonl",
        'second_train_file': "",
        'interleave_probs': [1],
        'eng_valid_file'   : f"{DATASET_DIR}/pretrain/valid_eng.jsonl",
        'output_dir'       : f"{MODEL_DIR}/relearned_models/model_name_only_forget_data",
        'cache_dir'        : CACHE_DIR,
        'dataset_cache_dir': CACHE_DIR,
        'join_or_subsequence'         : True,
        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 15,
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
        'max_length'                  : 256,

        'use_wandb'        : True,
        'wandb_project'    : "gemma-2-0.3B_relearn_only_forget",
        'wandb_run_name'   : None,
        'use_local_record' : True,
        'path_local_record': f"{MODEL_DIR}/local_records/relearned_models/model_name_only_forget_data.txt",

        'save_models'      : False,
        'shrink_perturb_relearning' : 0

    },
    "gemma-2-0.3B_all_data": {
        'model_name'       : f"{MODEL_DIR}/model_path/final_student",
        'first_train_file' :  f"{DATASET_DIR}/pretrain/train_eng.jsonl",
        'second_train_file': f"{DATASET_DIR}/pretrain/train_all_arithmetic.jsonl",
        'interleave_probs' : [.75, .25],
        'eng_valid_file'   : f"{DATASET_DIR}/pretrain/valid_eng.jsonl",
        'output_dir'       : f"{MODEL_DIR}/relearned_models/model_name_all_data",
        'cache_dir'        : CACHE_DIR,
        'dataset_cache_dir': CACHE_DIR,
        'join_or_subsequence'         : True,
        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 15,
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
        'max_length'                  : 256,

        'use_wandb'        : True,
        'wandb_project'    : "gemma-2-0.3B_relearn_all_data",
        'wandb_run_name'   : None,
        'use_local_record' : True,
        'path_local_record': f"{MODEL_DIR}/local_records/relearned_models/model_name_all_data.txt",

        'save_models'      : False,
        'shrink_perturb_relearning' : 0

    },
    "gemma-2-0.3B_all_data_perturb": {
        'model_name'       : f"{MODEL_DIR}/model_path/final_student",
        'first_train_file' :  f"{DATASET_DIR}/pretrain/train_eng.jsonl",
        'second_train_file': f"{DATASET_DIR}/pretrain/train_all_arithmetic.jsonl",
        'interleave_probs' : [.75, .25],
        'eng_valid_file'   : f"{DATASET_DIR}/pretrain/valid_eng.jsonl",
        'output_dir'       : f"{MODEL_DIR}/relearned_models/model_name_all_data_perturb",
        'cache_dir'        : CACHE_DIR,
        'dataset_cache_dir': CACHE_DIR,
        'join_or_subsequence'         : True,
        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 15,
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
        'max_length'                  : 256,

        'use_wandb'        : True,
        'wandb_project'    : "gemma-2-0.3B_relearn_all_data_perturb",
        'wandb_run_name'   : None,
        'use_local_record' : True,
        'path_local_record': f"{MODEL_DIR}/local_records/relearned_models/model_name_all_data_perturb.txt",

        'save_models'      : False,
        'shrink_perturb_relearning' : 0.1
    },
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

    arithmetic_eval_fn = get_arithmetic_eval_fn(
        # gets a function that takes a model returns a dicitonary with equation/word problem accuracty for each operation and english validation CE loss
        model_name          = current_setup['model_name'],
        eng_valid_file      = current_setup['eng_valid_file'],
        batch_size          = current_setup['batch_size'],
        max_length          = current_setup['max_length'],
        cache_dir           = current_setup['cache_dir'],
        dataset_cache_dir   = current_setup['dataset_cache_dir'],
        num_wiki_batches    = 50,
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
        eval_fn          = arithmetic_eval_fn,
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
        save_models      = current_setup['save_models'],
        shrink_perturb_relearning = current_setup['shrink_perturb_relearning']
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
    