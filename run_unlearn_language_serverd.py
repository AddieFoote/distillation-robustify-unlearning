from code.tools.unlearn.graddiff import unlearn_graddiff  # renamed from ga
from code.tools.unlearn.maxent_sam import unlearn_maxent      # renamed from uf
from code.tools.unlearn.rmu import unlearn_rmu            # unchanged
from accelerate import Accelerator
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code.utils.paths import CACHE_DIR, DATASET_DIR, MODEL_DIR, WANDB_API_KEY_PATH
from code.utils.validation_functions import get_korean_and_english_evalaution_fn
from code.utils.parallel_launch import launch_in_parallel_one_per_gpu, get_parallel_launch_wrapper

# Define learning rate ranges for each method
LR_RANGES = {
    "GradDiff": [4e-4, 8e-5],  # Limited LR range to keep total configs manageable
    "MaxEnt": [4e-4, 8e-5],    # Limited LR range to keep total configs manageable
    #"RMU": [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5],
    #"MaxEnt_RepNoise": [8e-5],  # Using fixed learning rate of 8e-5 for RepNoise experiments
    #"MaxEnt_SAM": [8e-5],       # Using fixed learning rate of 8e-5 for SAM experiments
    #"RMU_RepNoise": [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5],  # Added with full LR range
    #"RMU_SAM": [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5]        # Added with full LR range
}

# Define alpha values for Pareto frontier exploration
GRADDIFF_ALPHAS = [0.1, 0.5, 2.0, 5.0]  # 5 alpha values
MAXENT_ALPHAS = [0.1, 0.5, 2.0, 5.0] 

# Define RepNoise configurations for Pareto plot
REPNOISE_CONFIGS = [
    {"alpha": 0.1, "beta": 0.0001},   
    {"alpha": 0.1, "beta": 0.001},    
    {"alpha": 0.5, "beta": 0.01},     
    {"alpha": 0.5, "beta": 0.1},      
    {"alpha": 1.0, "beta": 0.1},      
    {"alpha": 1.0, "beta": 1.0},      
    {"alpha": 2.0, "beta": 2.0},      
    {"alpha": 4.0, "beta": 4.0}       
]

# Define SAM configurations for Pareto plot with rho values
SAM_CONFIGS = [
    #{"rho": 0.001},  # Very small perturbation
    #{"rho": 0.003},  # Small perturbation
    #{"rho": 0.005},  # Small-medium perturbation
    {"rho": 0.01},   # Medium perturbation (optimal according to paper)
    #{"rho": 0.02},   # Medium-high perturbation
    #{"rho": 0.04},   # High perturbation
    #{"rho": 0.07},   # Very high perturbation
    #{"rho": 0.1}     # Extremely high perturbation
]

# Base setups to run - will be expanded with learning rates and RepNoise/SAM params
BASE_SETUPS = ["gemma-2-0.1B_MaxEnt"]


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
        'alpha'                       : "TBD",  # Will be filled by alpha search
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
        'balance_alpha'               : "TBD",  # Will be filled by alpha search
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
    "gemma-2-0.1B_MaxEnt_RepNoise": {
        'model_name'       : f"{MODEL_DIR}/pretrained_models/gemma-2-0.1B_eng+kor/final_model",
        'forget_train_file': f"{DATASET_DIR}/pretrain/train_kor.jsonl",
        'retain_train_file': f"{DATASET_DIR}/pretrain/train_eng.jsonl",
        'eng_valid_file'   : f"{DATASET_DIR}/pretrain/valid_eng.jsonl",
        'kor_valid_file'   : f"{DATASET_DIR}/pretrain/valid_kor.jsonl",
        'output_dir'       : f"{MODEL_DIR}/unlearned_models/MaxEnt_RepNoise/gemma-2-0.1B_eng+kor",
        'cache_dir'        : CACHE_DIR,
        'dataset_cache_dir': CACHE_DIR,

        'use_retain'                  : True,
        'use_repnoise'                : True,  # Enable RepNoise
        'repnoise_alpha'              : "TBD", # Will be filled based on config
        'repnoise_beta'               : "TBD", # Will be filled based on config
        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 1,
        'gradient_accumulation_steps' : 4,
        'epochs'                      : 2,
        'learning_rate'               : "TBD",  # Will be fixed at 8e-5
        'max_steps'                   : 100,             
        'num_warmup_steps'            : 0,
        'validation_steps'            : 4,
        'save_checkpoint_steps'       : -1,
        'scheduler_type'              : "cosine",  
        'min_lr'                      : "TBD",  # Will be fixed at 8e-5
        'weight_decay'                : 0.0,         
        'gradient_clipping_threshold' : 1.0, 
        'max_length'                  : 2048,

        'use_wandb'        : True,
        'wandb_project'    : "gemma-2-0.1B_eng+kor_unlearn_MaxEnt_RepNoise",
        'wandb_run_name'   : None,
        'wandb_api_key'    : api_key,
        'use_local_record' : True,
        'path_local_record': f"{MODEL_DIR}/local_records/unlearned_models/MaxEnt_RepNoise/gemma-2-0.1B_eng+kor.txt",
    },
    "gemma-2-0.1B_MaxEnt_SAM": {
        'model_name'       : f"{MODEL_DIR}/pretrained_models/gemma-2-0.1B_eng+kor/final_model",
        'forget_train_file': f"{DATASET_DIR}/pretrain/train_kor.jsonl",
        'retain_train_file': f"{DATASET_DIR}/pretrain/train_eng.jsonl",
        'eng_valid_file'   : f"{DATASET_DIR}/pretrain/valid_eng.jsonl",
        'kor_valid_file'   : f"{DATASET_DIR}/pretrain/valid_kor.jsonl",
        'output_dir'       : f"{MODEL_DIR}/unlearned_models/MaxEnt_SAM/gemma-2-0.1B_eng+kor",
        'cache_dir'        : CACHE_DIR,
        'dataset_cache_dir': CACHE_DIR,

        'use_retain'                  : True,
        'use_sam'                     : True,  # Enable SAM
        'sam_rho'                     : "TBD", # Will be filled based on config
        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 1,
        'gradient_accumulation_steps' : 4,
        'epochs'                      : 2,
        'learning_rate'               : "TBD",  # Will be fixed at 8e-5
        'max_steps'                   : 100,             
        'num_warmup_steps'            : 0,
        'validation_steps'            : 4,
        'save_checkpoint_steps'       : -1,
        'scheduler_type'              : "cosine",  
        'min_lr'                      : "TBD",  # Will be fixed at 8e-5
        'weight_decay'                : 0.0,         
        'gradient_clipping_threshold' : 1.0, 
        'max_length'                  : 2048,

        'use_wandb'        : True,
        'wandb_project'    : "gemma-2-0.1B_eng+kor_unlearn_MaxEnt_SAM",
        'wandb_run_name'   : None,
        'wandb_api_key'    : api_key,
        'use_local_record' : True,
        'path_local_record': f"{MODEL_DIR}/local_records/unlearned_models/MaxEnt_SAM/gemma-2-0.1B_eng+kor.txt",
    },
    # Add new base setups for RMU_RepNoise and RMU_SAM
    "gemma-2-0.1B_RMU_RepNoise": {
        'model_name'       : f"{MODEL_DIR}/pretrained_models/gemma-2-0.1B_eng+kor/final_model",
        'forget_train_file': f"{DATASET_DIR}/pretrain/train_kor.jsonl",
        'retain_train_file': f"{DATASET_DIR}/pretrain/train_eng.jsonl",
        'eng_valid_file'   : f"{DATASET_DIR}/pretrain/valid_eng.jsonl",
        'kor_valid_file'   : f"{DATASET_DIR}/pretrain/valid_kor.jsonl",
        'output_dir'       : f"{MODEL_DIR}/unlearned_models/RMU_RepNoise/gemma-2-0.1B_eng+kor",
        'cache_dir'        : CACHE_DIR,
        'dataset_cache_dir': CACHE_DIR,

        'ga_gd'                       : False,
        'rmu_layers'                  : [5, 6, 7, 8, 9, 10, 11],
        'end_layer'                   : 11, 
        'alpha'                       : 1200,
        'c'                           : 6.5,
        'use_repnoise'                : True,  # Enable RepNoise
        'repnoise_alpha'              : "TBD", # Will be filled based on config
        'repnoise_beta'               : "TBD", # Will be filled based on config
        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 1,
        'gradient_accumulation_steps' : 4,
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
        'wandb_project'    : "gemma-2-0.1B_eng+kor_unlearn_RMU_RepNoise",
        'wandb_run_name'   : None,
        'wandb_api_key'    : api_key,
        'use_local_record' : True,
        'path_local_record': f"{MODEL_DIR}/local_records/unlearned_models/RMU_RepNoise/gemma-2-0.1B_eng+kor.txt",
    },
    "gemma-2-0.1B_RMU_SAM": {
        'model_name'       : f"{MODEL_DIR}/pretrained_models/gemma-2-0.1B_eng+kor/final_model",
        'forget_train_file': f"{DATASET_DIR}/pretrain/train_kor.jsonl",
        'retain_train_file': f"{DATASET_DIR}/pretrain/train_eng.jsonl",
        'eng_valid_file'   : f"{DATASET_DIR}/pretrain/valid_eng.jsonl",
        'kor_valid_file'   : f"{DATASET_DIR}/pretrain/valid_kor.jsonl",
        'output_dir'       : f"{MODEL_DIR}/unlearned_models/RMU_SAM/gemma-2-0.1B_eng+kor",
        'cache_dir'        : CACHE_DIR,
        'dataset_cache_dir': CACHE_DIR,

        'ga_gd'                       : False,
        'rmu_layers'                  : [5, 6, 7, 8, 9, 10, 11],
        'end_layer'                   : 11, 
        'alpha'                       : 1200,
        'c'                           : 6.5,
        'use_sam'                     : True,  # Enable SAM
        'sam_rho'                     : "TBD", # Will be filled based on config
        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 1,
        'gradient_accumulation_steps' : 4,
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
        'wandb_project'    : "gemma-2-0.1B_eng+kor_unlearn_RMU_SAM",
        'wandb_run_name'   : None,
        'wandb_api_key'    : api_key,
        'use_local_record' : True,
        'path_local_record': f"{MODEL_DIR}/local_records/unlearned_models/RMU_SAM/gemma-2-0.1B_eng+kor.txt",
    },
}

def create_lr_variant(base_setup_id, learning_rate):
    """Create a variant of a base setup with a different learning rate"""
    method = base_setup_id.split('_')[-1]  # Extract method name
    
    # For RepNoise, we need special handling for alpha/beta values
    if "RepNoise" in method or "SAM" in method:
        return None, None  # We'll handle this separately
    
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

def create_graddiff_alpha_variant(base_setup_id, alpha, learning_rate):
    """Create a GradDiff variant with specific alpha value"""
    # Create a unique ID that includes the alpha parameter
    new_setup_id = f"{base_setup_id}_a{alpha}_lr_{learning_rate:.1e}"
    
    # Clone the base setup
    setup_config = base_setups[base_setup_id].copy()
    
    # Update alpha parameter
    setup_config['alpha'] = alpha
    
    # Update learning rate related parameters
    setup_config['learning_rate'] = learning_rate
    setup_config['min_lr'] = learning_rate
    
    # Update paths to include parameters in directory/file names
    setup_config['output_dir'] = f"{setup_config['output_dir']}_a{alpha}_lr_{learning_rate:.1e}"
    setup_config['path_local_record'] = setup_config['path_local_record'].replace('.txt', f'_a{alpha}_lr_{learning_rate:.1e}.txt')
    
    # Update wandb run name
    setup_config['wandb_run_name'] = f"a{alpha}_lr_{learning_rate:.1e}"
    
    return new_setup_id, setup_config

def create_maxent_alpha_variant(base_setup_id, alpha, learning_rate):
    """Create a MaxEnt variant with specific balance_alpha value"""
    # Create a unique ID that includes the alpha parameter
    new_setup_id = f"{base_setup_id}_a{alpha}_lr_{learning_rate:.1e}"
    
    # Clone the base setup
    setup_config = base_setups[base_setup_id].copy()
    
    # Update alpha parameter
    setup_config['balance_alpha'] = alpha
    
    # Update learning rate related parameters
    setup_config['learning_rate'] = learning_rate
    setup_config['min_lr'] = learning_rate
    
    # Update paths to include parameters in directory/file names
    setup_config['output_dir'] = f"{setup_config['output_dir']}_a{alpha}_lr_{learning_rate:.1e}"
    setup_config['path_local_record'] = setup_config['path_local_record'].replace('.txt', f'_a{alpha}_lr_{learning_rate:.1e}.txt')
    
    # Update wandb run name
    setup_config['wandb_run_name'] = f"a{alpha}_lr_{learning_rate:.1e}"
    
    return new_setup_id, setup_config

def create_repnoise_variant(base_setup_id, repnoise_config, learning_rate):
    """Create a RepNoise variant with specific alpha/beta values"""
    alpha = repnoise_config["alpha"]
    beta = repnoise_config["beta"]
    
    # Create a unique ID that includes the RepNoise parameters
    new_setup_id = f"{base_setup_id}_a{alpha}_b{beta}_lr_{learning_rate:.1e}"
    
    # Clone the base setup
    setup_config = base_setups[base_setup_id].copy()
    
    # Update RepNoise parameters
    setup_config['repnoise_alpha'] = alpha
    setup_config['repnoise_beta'] = beta
    
    # Update learning rate related parameters
    setup_config['learning_rate'] = learning_rate
    setup_config['min_lr'] = learning_rate
    
    # Update paths to include parameters in directory/file names
    setup_config['output_dir'] = f"{setup_config['output_dir']}_a{alpha}_b{beta}_lr_{learning_rate:.1e}"
    setup_config['path_local_record'] = setup_config['path_local_record'].replace('.txt', f'_a{alpha}_b{beta}_lr_{learning_rate:.1e}.txt')
    
    # Update wandb run name
    setup_config['wandb_run_name'] = f"a{alpha}_b{beta}_lr_{learning_rate:.1e}"
    
    return new_setup_id, setup_config

def create_sam_variant(base_setup_id, sam_config, learning_rate):
    """Create a SAM variant with specific rho value"""
    rho = sam_config["rho"]
    
    # Create a unique ID that includes the SAM parameter
    new_setup_id = f"{base_setup_id}_rho{rho}_lr_{learning_rate:.1e}"
    
    # Clone the base setup
    setup_config = base_setups[base_setup_id].copy()
    
    # Update SAM parameter
    setup_config['sam_rho'] = rho
    
    # Update learning rate related parameters
    setup_config['learning_rate'] = learning_rate
    setup_config['min_lr'] = learning_rate
    
    # Update paths to include parameters in directory/file names
    setup_config['output_dir'] = f"{setup_config['output_dir']}_rho{rho}_lr_{learning_rate:.1e}"
    setup_config['path_local_record'] = setup_config['path_local_record'].replace('.txt', f'_rho{rho}_lr_{learning_rate:.1e}.txt')
    
    # Update wandb run name
    setup_config['wandb_run_name'] = f"rho{rho}_lr_{learning_rate:.1e}"
    
    return new_setup_id, setup_config

# Generate all setup variants with different learning rates and RepNoise/SAM configurations
setups = {}
SETUPS_TO_RUN = []

for base_setup_id in BASE_SETUPS:
    if "GradDiff" in base_setup_id:
        # For GradDiff, vary both learning rate and alpha
        lr_range = LR_RANGES["GradDiff"]
        
        for alpha in GRADDIFF_ALPHAS:
            for lr in lr_range:
                new_setup_id, setup_config = create_graddiff_alpha_variant(base_setup_id, alpha, lr)
                setups[new_setup_id] = setup_config
                SETUPS_TO_RUN.append(new_setup_id)
    elif "MaxEnt" in base_setup_id and "_RepNoise" not in base_setup_id and "_SAM" not in base_setup_id:
        # For MaxEnt, vary both learning rate and balance_alpha
        lr_range = LR_RANGES["MaxEnt"]
        
        for alpha in MAXENT_ALPHAS:
            for lr in lr_range:
                new_setup_id, setup_config = create_maxent_alpha_variant(base_setup_id, alpha, lr)
                setups[new_setup_id] = setup_config
                SETUPS_TO_RUN.append(new_setup_id)
    elif "_RMU_RepNoise" in base_setup_id:
        # Use the defined learning rate range for RMU_RepNoise
        lr_range = LR_RANGES.get("RMU_RepNoise", [1e-5])
        
        # For each RepNoise configuration, create a variant
        for repnoise_config in REPNOISE_CONFIGS:
            for lr in lr_range:
                new_setup_id, setup_config = create_repnoise_variant(base_setup_id, repnoise_config, lr)
                setups[new_setup_id] = setup_config
                SETUPS_TO_RUN.append(new_setup_id)
    elif "_RMU_SAM" in base_setup_id:
        # Use the defined learning rate range for RMU_SAM
        lr_range = LR_RANGES["RMU_SAM"]
        
        # For each SAM configuration, create a variant
        for sam_config in SAM_CONFIGS:
            for lr in lr_range:
                new_setup_id, setup_config = create_sam_variant(base_setup_id, sam_config, lr)
                setups[new_setup_id] = setup_config
                SETUPS_TO_RUN.append(new_setup_id)
    elif "_RepNoise" in base_setup_id:
        # Use the fixed learning rate for RepNoise
        lr_range = LR_RANGES.get("MaxEnt_RepNoise", [8e-5])
        
        # For each RepNoise configuration, create a variant
        for repnoise_config in REPNOISE_CONFIGS:
            for lr in lr_range:
                new_setup_id, setup_config = create_repnoise_variant(base_setup_id, repnoise_config, lr)
                setups[new_setup_id] = setup_config
                SETUPS_TO_RUN.append(new_setup_id)
    elif "_SAM" in base_setup_id:
        # Use the fixed learning rate for SAM
        lr_range = LR_RANGES.get("MaxEnt_SAM", [8e-5])
        
        # For each SAM configuration, create a variant
        for sam_config in SAM_CONFIGS:
            for lr in lr_range:
                new_setup_id, setup_config = create_sam_variant(base_setup_id, sam_config, lr)
                setups[new_setup_id] = setup_config
                SETUPS_TO_RUN.append(new_setup_id)
    else:
        # Get the appropriate learning rate range for this method
        method = base_setup_id.split('_')[-1]  # Extract method name
        lr_range = LR_RANGES.get(method, [])
        
        # Create a variant for each learning rate
        for lr in lr_range:
            new_setup_id, setup_config = create_lr_variant(base_setup_id, lr)
            if new_setup_id:  # Only add if valid
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
        print(f"Running GradDiff with alpha={setups[setup_id]['alpha']}, LR={setups[setup_id]['learning_rate']}")
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
    elif '_MaxEnt_RepNoise' in setup_id:
        print(f"Running MaxEnt with RepNoise: alpha={setups[setup_id]['repnoise_alpha']}, beta={setups[setup_id]['repnoise_beta']}, LR={setups[setup_id]['learning_rate']}")
        unlearn_maxent(
            model_name       = setups[setup_id]['model_name'],
            forget_train_file= setups[setup_id]['forget_train_file'],
            retain_train_file= setups[setup_id]['retain_train_file'],
            eval_fn          = language_eval_fn,
            accelerator      = accelerator,
            output_dir       = setups[setup_id]['output_dir'],
            cache_dir        = setups[setup_id]['cache_dir'],
            dataset_cache_dir= setups[setup_id]['dataset_cache_dir'],
            
            use_retain       = setups[setup_id]['use_retain'],
            use_repnoise     = setups[setup_id]['use_repnoise'],       # Enable RepNoise
            repnoise_alpha   = setups[setup_id]['repnoise_alpha'],     # Alpha for noise loss
            repnoise_beta    = setups[setup_id]['repnoise_beta'],      # Beta for ascent loss
            
            join_or_subsequence = True,
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
    elif '_MaxEnt_SAM' in setup_id:
        print(f"Running MaxEnt with SAM: rho={setups[setup_id]['sam_rho']}, LR={setups[setup_id]['learning_rate']}")
        unlearn_maxent(
            model_name       = setups[setup_id]['model_name'],
            forget_train_file= setups[setup_id]['forget_train_file'],
            retain_train_file= setups[setup_id]['retain_train_file'],
            eval_fn          = language_eval_fn,
            accelerator      = accelerator,
            output_dir       = setups[setup_id]['output_dir'],
            cache_dir        = setups[setup_id]['cache_dir'],
            dataset_cache_dir= setups[setup_id]['dataset_cache_dir'],
            
            use_retain       = setups[setup_id]['use_retain'],
            use_sam          = setups[setup_id]['use_sam'],           # Enable SAM
            sam_rho          = setups[setup_id]['sam_rho'],           # Rho parameter for SAM
            
            join_or_subsequence = True,
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
    elif '_RMU_RepNoise' in setup_id:
        print(f"Running RMU with RepNoise: alpha={setups[setup_id]['repnoise_alpha']}, beta={setups[setup_id]['repnoise_beta']}, LR={setups[setup_id]['learning_rate']}")
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
            use_repnoise     = setups[setup_id]['use_repnoise'],
            repnoise_alpha   = setups[setup_id]['repnoise_alpha'],
            repnoise_beta    = setups[setup_id]['repnoise_beta'],
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
    elif '_RMU_SAM' in setup_id:
        print(f"Running RMU with SAM: rho={setups[setup_id]['sam_rho']}, LR={setups[setup_id]['learning_rate']}")
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
            use_sam          = setups[setup_id]['use_sam'],
            sam_rho          = setups[setup_id]['sam_rho'],
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
    elif '_MaxEnt' in setup_id:
        print(f"Running MaxEnt with balance_alpha={setups[setup_id]['balance_alpha']}, LR={setups[setup_id]['learning_rate']}")
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
            balance_alpha    = setups[setup_id]['balance_alpha'],  # Pass the balance_alpha parameter
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
    print(f"Running {len(SETUPS_TO_RUN)} experiments:")
    
    # Group by experiment type for better visibility
    graddiff_setups = [s for s in SETUPS_TO_RUN if '_GradDiff' in s]
    maxent_setups = [s for s in SETUPS_TO_RUN if '_MaxEnt' in s and '_RepNoise' not in s and '_SAM' not in s]
    rmu_sam_setups = [s for s in SETUPS_TO_RUN if '_RMU_SAM' in s]
    other_setups = [s for s in SETUPS_TO_RUN if s not in graddiff_setups and s not in maxent_setups and s not in rmu_sam_setups]
    
    print(f"\n=== GradDiff Pareto Frontier Experiments ({len(graddiff_setups)}) ===")
    for setup_id in graddiff_setups:
        print(f"  - {setup_id} (alpha: {setups[setup_id]['alpha']}, LR: {setups[setup_id]['learning_rate']:.1e})")
    
    print(f"\n=== MaxEnt Pareto Frontier Experiments ({len(maxent_setups)}) ===")
    for setup_id in maxent_setups:
        print(f"  - {setup_id} (balance_alpha: {setups[setup_id]['balance_alpha']}, LR: {setups[setup_id]['learning_rate']:.1e})")
    
    print(f"\n=== RMU with SAM Experiments ({len(rmu_sam_setups)}) ===")
    for setup_id in rmu_sam_setups:
        print(f"  - {setup_id} (rho: {setups[setup_id]['sam_rho']}, LR: {setups[setup_id]['learning_rate']:.1e})")
    
    if other_setups:
        print(f"\n=== Other Experiments ({len(other_setups)}) ===")
        for setup_id in other_setups:
            print(f"  - {setup_id}")
    
    # Create list of the setups (arguments for run_experiment) for all the experiments we want to run 
    experiments = [(setup_id,) for setup_id in SETUPS_TO_RUN]
    # Gets a wrapper function compatable with the parallel launch function
    parallel_fn = get_parallel_launch_wrapper(launch_unlearning_run)
    # calls run_experiment in parallel on a separate gpu for each experiment setup when a gpu is free
    launch_in_parallel_one_per_gpu(experiment_list=experiments, experiment_fn=parallel_fn)