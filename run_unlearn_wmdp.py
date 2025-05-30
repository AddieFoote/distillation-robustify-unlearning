from src.tools.unlearn_wmdp.graddiff import unlearn_graddiff
from src.tools.unlearn_wmdp.maxent import unlearn_maxent
from src.tools.unlearn_wmdp.rmu import unlearn_rmu
from accelerate import Accelerator
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.paths import CACHE_DIR, DATASET_DIR, WMDP_MODEL_DIR
from src.utils.loss_functions import custom_login
from src.utils.validation_functions import get_wmdp_cyber_eval_fn, get_wmdp_bio_eval_fn 
from src.utils.parallel_launch import launch_in_parallel_one_per_gpu, get_parallel_launch_wrapper

FINAL_RUN=True # Controls eval size and overwrite ok

LR_RANGES = {
    "bio_MaxEnt": [2e-5], # [5e-5],
    "cyber_MaxEnt": [2e-5], # [3.5e-5, 5e-5]
    "bio_RMU": [5e-5], # [1e-4]
    "cyber_RMU": [5e-5],
    "cyber_repnoise": [2e-5], # [3.5e-5]
    "cyber_SAM": [2e-5],# [3.5e-5]
    "bio_repnoise": [2e-5],
    "bio_SAM": [2e-5],
}
ALPHA_RANGES = {
    "cyber_MaxEnt": [.2], # [.3]
    "bio_MaxEnt": [.3, .1], # [.75],
    "cyber_RMU": [.5, .2, .1],
    "bio_RMU": [.5], #[.1, .3]
    "cyber_repnoise": [.5], # [.3]
    "cyber_SAM": [.5], # [.3]
    "bio_repnoise": [.3, .1],
    "bio_SAM": [.3, .1],
}
SEEDS = [42, 43, 44, 45]

BASE_SETUPS = ['bio_RMU', 'bio_MaxEnt', 'cyber_RMU', 'cyber_MaxEnt', 'cyber_repnoise', 'cyber_SAM', 'bio_repnoise', 'bio_SAM']
custom_login()

base_setups = {
    "cyber_MaxEnt": {
        'model_name'       : f"{WMDP_MODEL_DIR}/gemma-2-2b",
        'forget_train_file': f"{DATASET_DIR}/pretrain/train_wmdp-cyber-forget-corpus_qa.jsonl", #f"{DATASET_DIR}/pretrain/train_cyber-forget-corpus.jsonl",
        'retain_files'      : [
                                f"{DATASET_DIR}/pretrain/train_wmdp-cyber-retain-corpus_qa.jsonl",
                                f"{DATASET_DIR}/pretrain/train_wikitext.jsonl"
                            ],
        'interleave_probs': [.5, .5],
        'output_dir'       : f"{WMDP_MODEL_DIR}/unlearned_models/MaxEnt/cyber_TBD",
        'cache_dir'        : CACHE_DIR,
        'dataset_cache_dir': CACHE_DIR,
        'use_retain'                  : True,
        'use_retain_kl'               : True,
        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 4,
        'gradient_accumulation_steps' : 10,
        'epochs'                      : 5,
        'learning_rate'               : "TBD",
        'alpha'                       : 0.99,
        'max_steps'                   : 90,
        'num_warmup_steps'            : 0,
        'validation_steps'            : 100,
        'save_checkpoint_steps'       : -1,
        'scheduler_type'              : "cosine",  
        'min_lr'                      : "TBD",        
        'weight_decay'                : 0.0,         
        'gradient_clipping_threshold' : 1.0, 
        'max_length'                  : 256,
        'use_wandb'        : True,
        'wandb_project'    : "cyber_unlearn_MaxEnt",
        'wandb_run_name'   : None,
        'use_local_record' : True,
        'path_local_record': f"{WMDP_MODEL_DIR}/local_records/unlearned_models/MaxEnt/cyber_TBD.txt",
        'use_sam'          : False,
        'use_repnoise'     : False
    },
    "cyber_SAM": {
        'model_name'       : f"{WMDP_MODEL_DIR}/gemma-2-2b",
        'forget_train_file': f"{DATASET_DIR}/pretrain/train_wmdp-cyber-forget-corpus_qa.jsonl", #f"{DATASET_DIR}/pretrain/train_cyber-forget-corpus.jsonl",
        'retain_files'      : [
                                f"{DATASET_DIR}/pretrain/train_wmdp-cyber-retain-corpus_qa.jsonl",
                                f"{DATASET_DIR}/pretrain/train_wikitext.jsonl"
                            ],
        'interleave_probs': [.5, .5],
        'output_dir'       : f"{WMDP_MODEL_DIR}/unlearned_models/MaxEnt-SAM-kl/cyber_TBD",
        'cache_dir'        : CACHE_DIR,
        'dataset_cache_dir': CACHE_DIR,
        'use_retain'                  : True,
        'use_retain_kl'               : True,
        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 4,
        'gradient_accumulation_steps' : 10,
        'epochs'                      : 5,
        'learning_rate'               : "TBD",
        'alpha'                       : 0.99,
        'max_steps'                   : 90,
        'num_warmup_steps'            : 0,
        'validation_steps'            : 100,
        'save_checkpoint_steps'       : -1,
        'scheduler_type'              : "cosine",  
        'min_lr'                      : "TBD",        
        'weight_decay'                : 0.0,         
        'gradient_clipping_threshold' : 1.0, 
        'max_length'                  : 256,
        'use_wandb'        : True,
        'wandb_project'    : "cyber_unlearn_MaxEnt-SAM",
        'wandb_run_name'   : None,
        'use_local_record' : True,
        'path_local_record': f"{WMDP_MODEL_DIR}/local_records/unlearned_models/MaxEnt-SAM-kl/cyber_TBD.txt",
        'use_sam'          : True,
        'use_repnoise'     : False
    },
    "cyber_repnoise": {
        'model_name'       : f"{WMDP_MODEL_DIR}/gemma-2-2b",
        'forget_train_file': f"{DATASET_DIR}/pretrain/train_wmdp-cyber-forget-corpus_qa.jsonl", #f"{DATASET_DIR}/pretrain/train_cyber-forget-corpus.jsonl",
        'retain_files'      : [
                                f"{DATASET_DIR}/pretrain/train_wmdp-cyber-retain-corpus_qa.jsonl",
                                f"{DATASET_DIR}/pretrain/train_wikitext.jsonl"
                            ],
        'interleave_probs': [.5, .5],
        'output_dir'       : f"{WMDP_MODEL_DIR}/unlearned_models/MaxEnt-repnoise-kl/cyber_TBD",
        'cache_dir'        : CACHE_DIR,
        'dataset_cache_dir': CACHE_DIR,
        'use_retain'                  : True,
        'use_retain_kl'               : True,
        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 1,
        'gradient_accumulation_steps' : 40,
        'epochs'                      : 5,
        'learning_rate'               : "TBD",
        'alpha'                       : 0.99,
        'max_steps'                   : 90,
        'num_warmup_steps'            : 0,
        'validation_steps'            : 100,
        'save_checkpoint_steps'       : -1,
        'scheduler_type'              : "cosine",  
        'min_lr'                      : "TBD",        
        'weight_decay'                : 0.0,         
        'gradient_clipping_threshold' : 1.0, 
        'max_length'                  : 256,
        'use_wandb'        : True,
        'wandb_project'    : "cyber_unlearn_MaxEnt-SAM",
        'wandb_run_name'   : None,
        'use_local_record' : True,
        'path_local_record': f"{WMDP_MODEL_DIR}/local_records/unlearned_models/MaxEnt-repnoise-kl/cyber_TBD.txt",
        'use_sam'          : False,
        'use_repnoise'     : True
    },
    "cyber_RMU": {
        'model_name'       : f"{WMDP_MODEL_DIR}/gemma-2-2b",
        'forget_train_file': f"{DATASET_DIR}/pretrain/train_wmdp-cyber-forget-corpus_qa.jsonl",
        'retain_files': [f"{DATASET_DIR}/pretrain/train_wmdp-cyber-retain-corpus_qa.jsonl", f"{DATASET_DIR}/pretrain/train_wikitext.jsonl"],
        'interleave_probs': [.5, .5],
        'stopping_strategy': 'first_exhausted',
        'output_dir'       : f"{WMDP_MODEL_DIR}/unlearned_models/RMU/cyber_TBD",
        'cache_dir'        : CACHE_DIR,
        'dataset_cache_dir': CACHE_DIR,
        'ga_gd'                       : True,
        'rmu_layers'                  : [10, 11, 12, 13, 14, 15],
        'end_layer'                   : 15, 
        'alpha'                       : .2, #1200,
        'c'                           : 80, #6.5,
        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 4,
        'gradient_accumulation_steps' : 10,
        'epochs'                      : 5,
        'learning_rate'               : "TBD",    
        'max_steps'                   : 90,
        'num_warmup_steps'            : 0,
        'validation_steps'            : 100,
        'save_checkpoint_steps'       : -1,
        'scheduler_type'              : "cosine",  
        'min_lr'                      : "TBD",              
        'weight_decay'                : 0.0,         
        'gradient_clipping_threshold' : 1.0, 
        'max_length'                  : 256,
        'use_wandb'        : True,
        'wandb_project'    : "cyber_unlearn_RMU",
        'wandb_run_name'   : None,
        'use_local_record' : True,
        'path_local_record': f"{WMDP_MODEL_DIR}/local_records/unlearned_models/RMU/cyber_TBD.txt",
    },
    "bio_RMU": {
        'model_name'       : f"{WMDP_MODEL_DIR}/gemma-2-2b",
        'forget_train_file': f"{DATASET_DIR}/pretrain/train_wmdp-bio_remove_dataset_qa.jsonl",
        'retain_files': [f"{DATASET_DIR}/pretrain/train_wikitext.jsonl", f"{DATASET_DIR}/pretrain/train_wmdp-bio_retain_dataset_qa.jsonl"],
        'interleave_probs' : [.5, .5],
        'stopping_strategy': 'first_exhausted',
        'output_dir'       : f"{WMDP_MODEL_DIR}/unlearned_models/RMU/bio_TBD",
        'cache_dir'        : CACHE_DIR,
        'dataset_cache_dir': CACHE_DIR,
        'ga_gd'                       : True,
        'rmu_layers'                  : [10, 11, 12, 13, 14, 15],
        'end_layer'                   : 15, 
        'alpha'                       : .2, #1200,
        'c'                           : 80, #6.5,
        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 8,
        'gradient_accumulation_steps' : 5,
        'epochs'                      : 5,
        'learning_rate'               : "TBD",    
        'max_steps'                   : 90,
        'num_warmup_steps'            : 0,
        'validation_steps'            : 100,
        'save_checkpoint_steps'       : -1,
        'scheduler_type'              : "cosine",  
        'min_lr'                      : "TBD",              
        'weight_decay'                : 0.0,         
        'gradient_clipping_threshold' : 1.0, 
        'max_length'                  : 256,
        'use_wandb'        : True,
        'wandb_project'    : "bio_unlearn_RMU",
        'wandb_run_name'   : None,
        'use_local_record' : True,
        'path_local_record': f"{WMDP_MODEL_DIR}/local_records/unlearned_models/RMU/bio_TBD.txt",
    },
    "bio_MaxEnt": {
        'model_name'       : f"{WMDP_MODEL_DIR}/gemma-2-2b",
        'forget_train_file': f"{DATASET_DIR}/pretrain/train_wmdp-bio_remove_dataset_qa.jsonl", 
        'retain_files'      : [
                                    f"{DATASET_DIR}/pretrain/train_wmdp-bio_retain_dataset_qa.jsonl", 
                                    f"{DATASET_DIR}/pretrain/train_wikitext.jsonl"
                                ],
        'interleave_probs': [.5, .5],
        'output_dir'       : f"{WMDP_MODEL_DIR}/unlearned_models/MaxEnt/bio_TBD",
        'cache_dir'        : CACHE_DIR,
        'dataset_cache_dir': CACHE_DIR,
        'use_retain'                  : True,
        'use_retain_kl'               : True,
        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 4,
        'gradient_accumulation_steps' : 10,
        'epochs'                      : 5,
        'learning_rate'               : "TBD",
        'alpha'                       : 0.99,
        'max_steps'                   : 90,
        'num_warmup_steps'            : 0,
        'validation_steps'            : 100,
        'save_checkpoint_steps'       : -1,
        'scheduler_type'              : "cosine",
        'min_lr'                      : "TBD",
        'weight_decay'                : 0.0,
        'gradient_clipping_threshold' : 1.0, 
        'max_length'                  : 256,
        'use_wandb'        : True,
        'wandb_project'    : "bio_unlearn_MaxEnt",
        'wandb_run_name'   : None,
        'use_local_record' : True,
        'path_local_record': f"{WMDP_MODEL_DIR}/local_records/unlearned_models/MaxEnt/bio_TBD.txt",
        'use_repnoise'     : False,
        'use_sam'          : False
    },
    "bio_repnoise": {
        'model_name'       : f"{WMDP_MODEL_DIR}/gemma-2-2b",
        'forget_train_file': f"{DATASET_DIR}/pretrain/train_wmdp-bio_remove_dataset_qa.jsonl", 
        'retain_files'      : [
                                    f"{DATASET_DIR}/pretrain/train_wmdp-bio_retain_dataset_qa.jsonl", 
                                    f"{DATASET_DIR}/pretrain/train_wikitext.jsonl"
                                ],
        'interleave_probs': [.5, .5],
        'output_dir'       : f"{WMDP_MODEL_DIR}/unlearned_models/MaxEnt-repnoise/bio_TBD",
        'cache_dir'        : CACHE_DIR,
        'dataset_cache_dir': CACHE_DIR,
        'use_retain'                  : True,
        'use_retain_kl'               : True,
        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 1,
        'gradient_accumulation_steps' : 40,
        'epochs'                      : 5,
        'learning_rate'               : "TBD",
        'alpha'                       : 0.99,
        'max_steps'                   : 90,
        'num_warmup_steps'            : 0,
        'validation_steps'            : 100,
        'save_checkpoint_steps'       : -1,
        'scheduler_type'              : "cosine",
        'min_lr'                      : "TBD",
        'weight_decay'                : 0.0,
        'gradient_clipping_threshold' : 1.0, 
        'max_length'                  : 256,
        'use_wandb'        : True,
        'wandb_project'    : "bio_unlearn_MaxEnt-repnoise",
        'wandb_run_name'   : None,
        'use_local_record' : True,
        'path_local_record': f"{WMDP_MODEL_DIR}/local_records/unlearned_models/MaxEnt-repnoise/bio_TBD.txt",
        'use_repnoise'     : True,
        'use_sam'          : False
    },
    "bio_SAM": {
        'model_name'       : f"{WMDP_MODEL_DIR}/gemma-2-2b",
        'forget_train_file': f"{DATASET_DIR}/pretrain/train_wmdp-bio_remove_dataset_qa.jsonl", 
        'retain_files'      : [
                                    f"{DATASET_DIR}/pretrain/train_wmdp-bio_retain_dataset_qa.jsonl", 
                                    f"{DATASET_DIR}/pretrain/train_wikitext.jsonl"
                                ],
        'interleave_probs': [.5, .5],
        'output_dir'       : f"{WMDP_MODEL_DIR}/unlearned_models/MaxEnt-SAM-kl/bio_TBD",
        'cache_dir'        : CACHE_DIR,
        'dataset_cache_dir': CACHE_DIR,
        'use_retain'                  : True,
        'use_retain_kl'               : True,
        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 4,
        'gradient_accumulation_steps' : 10,
        'epochs'                      : 5,
        'learning_rate'               : "TBD",
        'alpha'                       : 0.99,
        'max_steps'                   : 90,
        'num_warmup_steps'            : 0,
        'validation_steps'            : 100,
        'save_checkpoint_steps'       : -1,
        'scheduler_type'              : "cosine",
        'min_lr'                      : "TBD",
        'weight_decay'                : 0.0,
        'gradient_clipping_threshold' : 1.0, 
        'max_length'                  : 256,
        'use_wandb'        : True,
        'wandb_project'    : "bio_unlearn_MaxEnt-SAM",
        'wandb_run_name'   : None,
        'use_local_record' : True,
        'path_local_record': f"{WMDP_MODEL_DIR}/local_records/unlearned_models/MaxEnt-SAM-kl/bio_TBD.txt",
        'use_repnoise'     : False,
        'use_sam'          : True
    },
}

def create_lr_alpha_variant(base_setup_id, learning_rate, alpha, seed):
    """Create a variant of a base setup with a different learning rate"""
    tbd_str = f'lr_{learning_rate:.2e}_alpha_{alpha:.2f}_seed_{seed}'
    new_setup_id = f"{base_setup_id}_{tbd_str}"
    
    # Clone the base setup
    setup_config = base_setups[base_setup_id].copy()
    
    # Update learning rate related parameters
    setup_config['learning_rate'] = learning_rate
    setup_config['min_lr'] = learning_rate  # Also update min_lr to match
    setup_config['alpha'] = alpha
    setup_config['seed'] = seed
    
    # Update paths to include learning rate in directory/file names
    
    setup_config['output_dir'] =  setup_config['output_dir'].replace('TBD', tbd_str) # f"{setup_config['output_dir']}_lr_{learning_rate:.1e}"
    setup_config['path_local_record'] = setup_config['path_local_record'].replace('TBD', tbd_str)
    
    # Update wandb run name to include learning rate
    setup_config['wandb_run_name'] = tbd_str
    
    return new_setup_id, setup_config

# Generate all setup variants with different learning rates
setups = {}
SETUPS_TO_RUN = []

for base_setup_id in BASE_SETUPS:
    method = base_setup_id # base_setup_id.split('_')[-1]  # Extract method name (GradDiff, MaxEnt, or RMU)
    
    # Get the appropriate learning rate range for this method
    lr_range = LR_RANGES[method]
    alpha_range = ALPHA_RANGES[method]
    
    # Create a variant for each learning rate
    for lr in lr_range:
        for alpha in alpha_range:
            for seed in SEEDS:
                new_setup_id, setup_config = create_lr_alpha_variant(base_setup_id, lr, alpha, seed)
                setups[new_setup_id] = setup_config
                SETUPS_TO_RUN.append(new_setup_id)

def launch_unlearning_run(setup_id):
    accelerator = Accelerator()

    if 'cyber' in setup_id.lower():
        eval_fn = get_wmdp_cyber_eval_fn(accelerator, large_eval=FINAL_RUN)
    elif 'bio'in setup_id.lower():
        eval_fn = get_wmdp_bio_eval_fn(accelerator, large_eval=FINAL_RUN)
    else:
        raise ValueError("key should contain cyber or bio")

    if '_GradDiff' in setup_id:
        unlearn_graddiff(  # Fixed function name from unlearn_ga to unlearn_graddiff
            model_name       = setups[setup_id]['model_name'],
            forget_train_file= setups[setup_id]['forget_train_file'],
            retain_train_file= setups[setup_id]['retain_train_file'],
            eval_fn          = eval_fn,
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
            use_local_record = setups[setup_id]['use_local_record'],
            path_local_record= setups[setup_id]['path_local_record'],
            overwrite_ok     = not FINAL_RUN,
        )
    elif '_MaxEnt' in setup_id or '_SAM' in setup_id or '_repnoise' in setup_id:
        print(f"Running MaxEnt with learning rate {setups[setup_id]['learning_rate']}")
        unlearn_maxent(  # Fixed function name from unlearn_uf to unlearn_maxent
            model_name       = setups[setup_id]['model_name'],
            forget_train_file= setups[setup_id]['forget_train_file'],

            retain_files= setups[setup_id]['retain_files'],
            interleave_probs= setups[setup_id]['interleave_probs'],
            stopping_strategy= 'first_exhausted',

            eval_fn = eval_fn,
            accelerator = accelerator,
            join_or_subsequence = True,
            output_dir       = setups[setup_id]['output_dir'],
            cache_dir        = setups[setup_id]['cache_dir'],
            dataset_cache_dir= setups[setup_id]['dataset_cache_dir'],
            use_retain       = setups[setup_id]['use_retain'],
            use_retain_kl    = setups[setup_id]['use_retain_kl'],
            alpha            = setups[setup_id]['alpha'],
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
            use_local_record = setups[setup_id]['use_local_record'],
            path_local_record= setups[setup_id]['path_local_record'],
            overwrite_ok     = True,
            use_sam          = setups[setup_id]['use_sam'],
            use_repnoise     = setups[setup_id]['use_repnoise'],
        )
    elif '_RMU' in setup_id:
        unlearn_rmu(
            model_name       = setups[setup_id]['model_name'],
            forget_train_file= setups[setup_id]['forget_train_file'],
            retain_files= setups[setup_id]['retain_files'],
            interleave_probs = setups[setup_id]['interleave_probs'],
            stopping_strategy = setups[setup_id]['stopping_strategy'],
            eval_fn = eval_fn,
            accelerator = accelerator,
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
            use_local_record = setups[setup_id]['use_local_record'],
            path_local_record= setups[setup_id]['path_local_record'],
            overwrite_ok     = True,
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