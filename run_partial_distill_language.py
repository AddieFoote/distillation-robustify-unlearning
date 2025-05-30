"""
Configured for a 4xH200 GPU server.

Launch Command (Sweep): python run_partial_distill_language_sweep_alpha.py --run_all
Launch Command (Single): accelerate launch run_partial_distill_language_sweep_alpha.py --setup gemma-2-0.1B_MaxEnt --alpha 0.3 --beta 0.1
"""

from src.tools.partial_distill_langarith import partial_distill
from src.utils.paths import CACHE_DIR, DATASET_DIR, MODEL_DIR, WANDB_API_KEY_PATH
from accelerate import Accelerator
from utils.loss_functions import print_acc, custom_login
from src.utils.validation_functions import get_korean_and_english_evalaution_fn
from utils.parallel_launch import launch_in_parallel_one_per_gpu, get_parallel_launch_wrapper
import argparse
import subprocess
import sys
import time
from datetime import datetime

# Configuration constants
SETUPS_TO_RUN = ["gemma-2-0.1B_MaxEnt"]
SWEEP_ALPHAS = [0.7, 0.5, 0.3, 0.1]  # Alpha values to sweep through
SWEEP_BETAS = [0.1, 0.05, 0.01]  # Beta values to sweep through (default is 0.1)
SWEEP_SEEDS = None  # Set to None to use default seed, or a list like [42, 1234, 5678] to sweep over seeds
USE_PARALLEL = False  # Flag to enable/disable parallel execution across GPUs
OVERWRITE_OK = True

# Early stopping configuration
STOP_CONDITION = "retain_only"  # Options: "retain_only", "forget_only", "both"
RETAIN_THRESHOLD = 0.05  # Threshold for English (retain domain)
FORGET_THRESHOLD = None  # Threshold for Korean (forget domain)

# Initialize parameters
custom_login()

setups = {
    "gemma-2-0.1B_MaxEnt": {
        'teacher_model_name': f"{MODEL_DIR}/unlearned_models/MaxEnt/gemma-2-0.1B_eng+kor_lr_3.0e-05/final_model",
        'student_model_name': f"{MODEL_DIR}/unlearned_models/MaxEnt/gemma-2-0.1B_eng+kor_lr_3.0e-05/final_model",
        'eng_train_file'    : f"{DATASET_DIR}/pretrain/train_eng.jsonl",
        'kor_train_file'    : f"{DATASET_DIR}/pretrain/train_kor.jsonl",
        'eng_valid_file'    : f"{DATASET_DIR}/pretrain/valid_eng.jsonl",
        'kor_valid_file'    : f"{DATASET_DIR}/pretrain/valid_kor.jsonl",
        'output_dir'        : f"{MODEL_DIR}/partial_distill_models_alt/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha-beta",
        'cache_dir'         : "hf_cache",
        'dataset_cache_dir' : "hf_cache",

        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 4,
        'gradient_accumulation_steps' : 60,
        'epochs'                      : 1,
        'learning_rate'               : 9e-4,       
        'max_steps'                   : -1,             
        'num_warmup_steps'            : 50,
        'validation_steps'            : 50,
        'save_checkpoint_steps'       : 9999,
        'scheduler_type'              : "cosine",  
        'min_lr'                      : 7e-4,              
        'weight_decay'                : 0,         
        'gradient_clipping_threshold' : 1.0, 
        'max_length'                  : 2048,

        'use_wandb'        : True,
        'wandb_project'    : "gemma-2-0.1B_eng+kor_ga_distill",
        'wandb_run_name'   : None,
        'use_local_record' : True,
        'path_local_record': f"{MODEL_DIR}/local_records/partial_distill_models_alt/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha-beta.txt",

        'shrink_perturb_repeat' : False
    }
}

def korean_english_stop_cond_fn(student_eval_dict, teacher_eval_dict, stop_condition=STOP_CONDITION, retain_threshold=RETAIN_THRESHOLD, forget_threshold=FORGET_THRESHOLD):
    """Stop condition function for Korean and English evaluation"""
    # English is the retain domain
    student_eng_loss = student_eval_dict.get("val/eng_ce_loss", float('inf'))
    teacher_eng_loss = teacher_eval_dict.get("val/eng_ce_loss", 0)
    
    # Korean is the forget domain
    student_kor_loss = student_eval_dict.get("val/kor_ce_loss", float('inf'))
    teacher_kor_loss = teacher_eval_dict.get("val/kor_ce_loss", 0)
    
    # Calculate the relative difference in losses
    eng_diff = (student_eng_loss - teacher_eng_loss) / max(teacher_eng_loss, 1e-6)
    kor_diff = (student_kor_loss - teacher_kor_loss) / max(teacher_kor_loss, 1e-6)
    
    # Apply stopping condition based on configuration
    if stop_condition == "retain_only":
        # Stop if the student is within threshold of the teacher on English only
        return eng_diff < retain_threshold
    elif stop_condition == "forget_only":
        # Stop if the student is within threshold of the teacher on Korean only
        return kor_diff < forget_threshold
    else:  # "both"
        # Stop if the student is within threshold of the teacher on both languages
        return eng_diff < retain_threshold and kor_diff < forget_threshold

def run_experiment(setup_id, alpha, beta, seed=None):
    """Run a single experiment with the given setup ID, alpha, beta and optional seed values"""
    current_setup = setups[setup_id].copy()
    
    # Use provided seed or keep default
    if seed is not None:
        current_setup['seed'] = seed
    
    # Update paths and parameters based on whether seed is provided
    if seed is not None:
        path_suffix = f'-alpha_{alpha}-beta_{beta}-seed_{seed}'
        name_suffix = f"_alpha{alpha}_beta{beta}_seed{seed}"
    else:
        path_suffix = f'-alpha_{alpha}-beta_{beta}'
        name_suffix = f"_alpha{alpha}_beta{beta}"
    
    current_setup['output_dir'] = current_setup['output_dir'].replace('-alpha-beta', path_suffix)
    current_setup['path_local_record'] = current_setup['path_local_record'].replace('-alpha-beta', path_suffix)
    current_setup['noise_alpha'] = alpha
    current_setup['noise_beta'] = beta
    
    # Update wandb run name
    if current_setup['wandb_run_name'] is None:
        current_setup['wandb_run_name'] = f"{setup_id}{name_suffix}"
    else:
        current_setup['wandb_run_name'] = f"{current_setup['wandb_run_name']}{name_suffix}"
    
    accelerator = Accelerator()
    
    # Get evaluation function for Korean and English
    eval_fn = get_korean_and_english_evalaution_fn(
        model_name=current_setup['student_model_name'],
        max_length=current_setup['max_length'],
        eng_valid_file=current_setup['eng_valid_file'],
        kor_valid_file=current_setup['kor_valid_file'],
        dataset_cache_dir=current_setup['dataset_cache_dir'],
        cache_dir=current_setup['cache_dir'],
        batch_size=current_setup['batch_size'],
        accelerator=accelerator
    )
    
    partial_distill(
        teacher_model_name=current_setup['teacher_model_name'],
        student_model_name=current_setup['student_model_name'],
        train_files=[current_setup['eng_train_file'], current_setup['kor_train_file']],
        interleave_probs=[0.5, 0.5],  # Equal probabilities for eng and kor
        stopping_strategy='first_exhausted',  # Add missing parameter
        join_or_subsequence=True,
        eval_fn=eval_fn,
        stop_cond_fn=korean_english_stop_cond_fn,
        accelerator=accelerator,
        output_dir=current_setup['output_dir'],
        cache_dir=current_setup['cache_dir'],
        dataset_cache_dir=current_setup['dataset_cache_dir'],
        seed=current_setup['seed'],
        device=current_setup['device'],
        batch_size=current_setup['batch_size'],
        gradient_accumulation_steps=current_setup['gradient_accumulation_steps'],
        epochs=current_setup['epochs'],
        learning_rate=current_setup['learning_rate'],
        max_steps=current_setup['max_steps'],   
        num_warmup_steps=current_setup['num_warmup_steps'],
        validation_steps=current_setup['validation_steps'],
        save_checkpoint_steps=current_setup['save_checkpoint_steps'],
        scheduler_type=current_setup['scheduler_type'],  
        min_lr=current_setup['min_lr'],          
        weight_decay=current_setup['weight_decay'],    
        gradient_clipping_threshold=current_setup['gradient_clipping_threshold'], 
        max_length=current_setup['max_length'],
        use_wandb=current_setup['use_wandb'],
        wandb_project=current_setup['wandb_project'],
        wandb_run_name=current_setup['wandb_run_name'],
        use_local_record=current_setup['use_local_record'],
        path_local_record=current_setup['path_local_record'],
        noise_alpha=current_setup['noise_alpha'],
        noise_beta=current_setup['noise_beta'],  # Add beta parameter here
        shrink_perturb_repeat=current_setup['shrink_perturb_repeat'],
        overwrite_ok=OVERWRITE_OK,
    )

def launch_single_experiment(setup_id, alpha, beta, seed=None):
    """Launch a single experiment in a separate process using accelerate"""
    print(f"\n{'='*80}")
    print(f"LAUNCHING EXPERIMENT:")
    print(f"  Setup: {setup_id}")
    print(f"  Alpha: {alpha}")
    print(f"  Beta: {beta}")
    if seed is not None:
        print(f"  Seed: {seed}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # Build the command to run this same script with arguments to run a single experiment
    cmd = [
        "accelerate", "launch", 
        sys.argv[0],  # Use the current script
        "--setup", setup_id,
        "--alpha", str(alpha),
        "--beta", str(beta)
    ]
    
    # Add seed to command if provided
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    
    # Run the command as a separate process
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n{'='*80}")
        print(f"EXPERIMENT COMPLETED SUCCESSFULLY:")
        print(f"  Setup: {setup_id}")
        print(f"  Alpha: {alpha}")
        print(f"  Beta: {beta}")
        if seed is not None:
            print(f"  Seed: {seed}")
        print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*80}")
        print(f"EXPERIMENT FAILED:")
        print(f"  Setup: {setup_id}")
        print(f"  Alpha: {alpha}")
        print(f"  Beta: {beta}")
        if seed is not None:
            print(f"  Seed: {seed}")
        print(f"  Error: {e}")
        print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        return False

def run_all_experiments(start_alpha=None):
    """Run all experiments in the sweep, optionally starting from a specific alpha"""
    # Filter alphas if start_alpha is specified
    alphas = [a for a in SWEEP_ALPHAS if start_alpha is None or a >= start_alpha]
    
    # Create experiments list based on whether seed sweeping is enabled
    if SWEEP_SEEDS is None:
        # No seed sweeping - use default seed
        experiments = [(setup_id, alpha, beta, None) 
                      for setup_id in SETUPS_TO_RUN 
                      for alpha in alphas 
                      for beta in SWEEP_BETAS]
    else:
        # Sweep over seeds
        experiments = [(setup_id, alpha, beta, seed) 
                      for setup_id in SETUPS_TO_RUN 
                      for alpha in alphas 
                      for beta in SWEEP_BETAS
                      for seed in SWEEP_SEEDS]
    
    print(f"Running {len(experiments)} experiments in individual accelerate launches")
    print(f"Alpha values: {alphas}")
    print(f"Beta values: {SWEEP_BETAS}")
    if SWEEP_SEEDS is not None:
        print(f"Seed values: {SWEEP_SEEDS}")
    else:
        print(f"Using default seed: {setups[SETUPS_TO_RUN[0]]['seed']}")
    
    # Run all experiments
    successful = 0
    failed = 0
    
    for i, experiment in enumerate(experiments):
        if len(experiment) == 4:  # Has seed
            setup_id, alpha, beta, seed = experiment
        else:  # No seed
            setup_id, alpha, beta, seed = experiment[0], experiment[1], experiment[2], None
            
        print(f"Running experiment {i+1}/{len(experiments)}")
        
        # Run the experiment in a separate process
        success = launch_single_experiment(setup_id, alpha, beta, seed)
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Add a short delay between experiments to ensure cleanup
        time.sleep(5)
    
    print(f"\nAll experiments completed!")
    print(f"Successful experiments: {successful}/{len(experiments)}")
    print(f"Failed experiments: {failed}/{len(experiments)}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run partial_distill language experiments')
    parser.add_argument('--setup', type=str, default=None, help='Setup ID to run')
    parser.add_argument('--alpha', type=float, default=None, help='Alpha value for experiment')
    parser.add_argument('--beta', type=float, default=None, help='Beta value for experiment')
    parser.add_argument('--seed', type=int, default=None, help='Seed value for experiment')
    parser.add_argument('--run_all', action='store_true', help='Run all experiments in the sweep')
    parser.add_argument('--start_alpha', type=float, default=None, help='Alpha value to start from (inclusive)')
    args = parser.parse_args()
    
    # If specific experiment parameters are provided, run that single experiment
    if args.setup is not None and args.alpha is not None and args.beta is not None:
        print(f"Running single experiment:")
        print(f"  Setup: {args.setup}")
        print(f"  Alpha: {args.alpha}")
        print(f"  Beta: {args.beta}")
        if args.seed is not None:
            print(f"  Seed: {args.seed}")
        run_experiment(args.setup, args.alpha, args.beta, args.seed)
    
    # If run_all flag is set, run all experiments
    elif args.run_all or args.start_alpha is not None:
        run_all_experiments(args.start_alpha)
    
    # If no valid arguments provided, show help
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Run a single experiment directly:")
        print("  accelerate launch run_partial_distill_language_sweep_alpha.py --setup gemma-2-0.1B_MaxEnt --alpha 0.3 --beta 0.1")
        print("  accelerate launch run_partial_distill_language_sweep_alpha.py --setup gemma-2-0.1B_MaxEnt --alpha 0.3 --beta 0.1 --seed 42")
        print("  # Run all experiments in separate accelerate launches:")
        print("  python run_partial_distill_language_sweep_alpha.py --run_all")
        print("  # Resume from a specific alpha value:")
        print("  python run_partial_distill_language_sweep_alpha.py --start_alpha 0.4")