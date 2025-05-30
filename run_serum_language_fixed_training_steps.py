"""
Configured for a 8xH200 GPU server.

Launch Command (Sweep): python run_serum_language_fixed_training_steps.py --run_all
Launch Command (Single): accelerate launch run_serum_language_fixed_training_steps.py --setup gemma-2-0.1B_MaxEnt --alpha 0.3 --beta 0.1
"""

from code.tools.serum_original import serum
from code.utils.paths import CACHE_DIR, DATASET_DIR, MODEL_DIR, WANDB_API_KEY_PATH
from accelerate import Accelerator
from utils.loss_functions import print_acc, custom_login
from code.utils.validation_functions import get_korean_and_english_evalaution_fn
from utils.parallel_launch import launch_in_parallel_one_per_four_gpus, get_parallel_launch_wrapper
import argparse
import subprocess
import sys
import os
import signal
import time
from datetime import datetime

# Configuration constants
SETUPS_TO_RUN = ["gemma-2-0.1B_MaxEnt"]
SWEEP_ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # Alpha values to sweep through
SWEEP_BETAS = [0.1]  # Beta values to sweep through (default is 0.1)
SWEEP_SEEDS = [111]  # Set to None to use default seed, or a list like [42, 1234, 5678] to sweep over seeds
USE_PARALLEL = False  # Flag to enable/disable parallel execution across GPUs
OVERWRITE_OK = True

# Fixed training configuration
FIXED_STEPS = 300  # Default number of training steps per experiment
VALIDATION_FREQ = 250  # Validate every 250 steps

# Initialize parameters
custom_login()

setups = {
    "gemma-2-0.1B_MaxEnt": {
        'teacher_model_name': f"{MODEL_DIR}/unlearned_models/MaxEnt/gemma-2-0.1B_eng+kor_lr_8.0e-05/final_model",
        'student_model_name': f"{MODEL_DIR}/unlearned_models/MaxEnt/gemma-2-0.1B_eng+kor_lr_8.0e-05/final_model",
        'eng_train_file'    : f"{DATASET_DIR}/pretrain/train_eng.jsonl",
        'kor_train_file'    : f"{DATASET_DIR}/pretrain/train_kor.jsonl",
        'eng_valid_file'    : f"{DATASET_DIR}/pretrain/valid_eng.jsonl",
        'kor_valid_file'    : f"{DATASET_DIR}/pretrain/valid_kor.jsonl",
        'output_dir'        : f"{MODEL_DIR}/serum_models/gemma-2-0.1B_MaxEnt-language-SERUM-alpha-beta",
        'cache_dir'         : "hf_cache",
        'dataset_cache_dir' : "hf_cache",

        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 4,
        'gradient_accumulation_steps' : 60,
        'epochs'                      : 1,
        'learning_rate'               : 9e-4,       
        'max_steps'                   : FIXED_STEPS,
        'num_warmup_steps'            : 50,
        'validation_steps'            : VALIDATION_FREQ,
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
        'path_local_record': f"{MODEL_DIR}/local_records/serum_models/gemma-2-0.1B_MaxEnt-language-SERUM-alpha-beta.txt",

        'shrink_perturb_repeat' : False
    }
}

def dummy_stop_cond_fn(student_eval_dict, teacher_eval_dict):
    """Dummy stop condition function that always returns False to rely on max_steps"""
    # Log metrics for informational purposes
    student_eng_loss = student_eval_dict.get("val/eng_ce_loss", float('inf'))
    teacher_eng_loss = teacher_eval_dict.get("val/eng_ce_loss", 0)
    eng_diff = (student_eng_loss - teacher_eng_loss) / max(teacher_eng_loss, 1e-6)
    
    # For Korean metrics
    student_kor_loss = student_eval_dict.get("val/kor_ce_loss", float('inf'))
    teacher_kor_loss = teacher_eval_dict.get("val/kor_ce_loss", 0)
    kor_diff = (student_kor_loss - teacher_kor_loss) / max(teacher_kor_loss, 1e-6)
    
    # Log metrics
    print(f"English diff: {eng_diff:.4f}")
    print(f"Korean diff: {kor_diff:.4f}")
    
    # Always return False to rely on max_steps
    return False

def run_experiment(setup_id, alpha, beta, seed=None, num_steps=FIXED_STEPS):
    """Run a single experiment with the given setup ID, alpha, beta and fixed number of steps"""
    print(f"\n{'='*80}")
    print(f"LAUNCHING EXPERIMENT:")
    print(f"  Setup: {setup_id}")
    print(f"  Alpha: {alpha}")
    print(f"  Beta: {beta}")
    if seed is not None:
        print(f"  Seed: {seed}")
    print(f"  Fixed Steps: {num_steps}")
    print(f"  Validation Frequency: {VALIDATION_FREQ}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    current_setup = setups[setup_id].copy()
    
    # Use provided seed or keep default
    if seed is not None:
        current_setup['seed'] = seed
    
    # Set max_steps for this experiment
    current_setup['max_steps'] = num_steps
    current_setup['validation_steps'] = VALIDATION_FREQ
    
    print(f"Setting max_steps to {current_setup['max_steps']}")
    print(f"Setting validation_steps to {current_setup['validation_steps']}")
    
    # Update paths and parameters based on whether seed is provided
    if seed is not None:
        path_suffix = f'-alpha_{alpha}-beta_{beta}-seed_{seed}-fixed_steps_{num_steps}'
        name_suffix = f"_alpha{alpha}_beta{beta}_seed{seed}_steps{num_steps}"
    else:
        path_suffix = f'-alpha_{alpha}-beta_{beta}-fixed_steps_{num_steps}'
        name_suffix = f"_alpha{alpha}_beta{beta}_steps{num_steps}"
    
    current_setup['output_dir'] = current_setup['output_dir'].replace('-alpha-beta', path_suffix)
    current_setup['path_local_record'] = current_setup['path_local_record'].replace('-alpha-beta', path_suffix)
    current_setup['noise_alpha'] = alpha
    current_setup['noise_beta'] = beta
    
    # Update wandb run name
    if current_setup['wandb_run_name'] is None:
        current_setup['wandb_run_name'] = f"{setup_id}{name_suffix}"
    else:
        current_setup['wandb_run_name'] = f"{current_setup['wandb_run_name']}{name_suffix}"
    
    print(f"Will save models to: {current_setup['output_dir']}")
    
    # Check if directory already exists
    os.makedirs(os.path.dirname(current_setup['output_dir']), exist_ok=True)
    
    # Register signal handler for Ctrl+C to save model before exiting
    def signal_handler(sig, frame):
        seed_msg = f", seed={seed}" if seed is not None else ""
        print(f"\n[run_language_serum.py] Received interrupt signal for {setup_id}, alpha={alpha}{seed_msg}. Saving model before exiting...")
        # Create an unwrapped model for saving
        unwrapped_student = accelerator.unwrap_model(student_model) if 'student_model' in locals() else None
        
        if unwrapped_student is not None and accelerator.is_main_process:
            interrupted_model_path = os.path.join(current_setup['output_dir'], "interrupted_model")
            os.makedirs(os.path.dirname(interrupted_model_path), exist_ok=True)
            unwrapped_student.save_pretrained(interrupted_model_path)
            tokenizer.save_pretrained(interrupted_model_path) if 'tokenizer' in locals() else None
            print(f"[run_language_serum.py] Saved interrupted model => {interrupted_model_path}")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
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
    
    serum(
        teacher_model_name=current_setup['teacher_model_name'],
        student_model_name=current_setup['student_model_name'],
        train_files=[current_setup['eng_train_file'], current_setup['kor_train_file']],
        interleave_probs=[0.5, 0.5],  # Equal probabilities for eng and kor
        stopping_strategy='first_exhausted',
        join_or_subsequence=True,
        eval_fn=eval_fn,
        stop_cond_fn=dummy_stop_cond_fn,  # Use a dummy function that always returns False
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
        noise_beta=current_setup['noise_beta'],
        shrink_perturb_repeat=current_setup['shrink_perturb_repeat'],
        overwrite_ok=OVERWRITE_OK,
    )
    
    # Check if model was saved
    expected_model_path = os.path.join(current_setup['output_dir'], "final_model")
    if os.path.exists(expected_model_path):
        print(f"SUCCESS: Final model saved at {expected_model_path}")
        return True
    else:
        print(f"WARNING: Final model not found at {expected_model_path}")
        return False

# Wrapper that matches the expected signature but incorporates fixed steps
def run_experiment_wrapper(setup_id, alpha, beta, seed=None):
    """Wrapper function that matches the expected 4-parameter signature for parallel launch"""
    return run_experiment(setup_id, alpha, beta, seed, FIXED_STEPS)

def run_all_experiments_parallel(start_alpha=None, num_steps=FIXED_STEPS):
    """Run all experiments in parallel using 4 GPUs per experiment"""
    # Filter alphas if start_alpha is specified
    alphas = [a for a in SWEEP_ALPHAS if start_alpha is None or a >= start_alpha]
    
    # Update global fixed steps value
    global FIXED_STEPS
    FIXED_STEPS = num_steps
    
    # Update max_steps in setups
    for setup in setups.values():
        setup['max_steps'] = num_steps
        setup['validation_steps'] = VALIDATION_FREQ
    
    # Create experiments list with standard 4 parameters (the wrapper will use the global FIXED_STEPS)
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
    
    print(f"Running {len(experiments)} experiments in parallel using 4 GPUs per experiment")
    print(f"Alpha values: {alphas}")
    print(f"Beta values: {SWEEP_BETAS}")
    if SWEEP_SEEDS is not None:
        print(f"Seed values: {SWEEP_SEEDS}")
    else:
        print(f"Using default seed: {setups[SETUPS_TO_RUN[0]]['seed']}")
    print(f"Fixed steps: {num_steps}")
    print(f"Validation frequency: {VALIDATION_FREQ}")
    
    # Launch experiments in parallel using GPU groups
    launch_in_parallel_one_per_four_gpus(experiment_list=experiments, experiment_fn=run_experiment_wrapper)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run SERUM language experiments with fixed training steps')
    parser.add_argument('--setup', type=str, default=None, help='Setup ID to run')
    parser.add_argument('--alpha', type=float, default=None, help='Alpha value for experiment')
    parser.add_argument('--beta', type=float, default=None, help='Beta value for experiment')
    parser.add_argument('--seed', type=int, default=None, help='Seed value for experiment')
    parser.add_argument('--steps', type=int, default=FIXED_STEPS, help='Exact number of steps to run')
    parser.add_argument('--validation_freq', type=int, default=VALIDATION_FREQ, help='Validation frequency')
    parser.add_argument('--run_all', action='store_true', help='Run all experiments in the sweep')
    parser.add_argument('--start_alpha', type=float, default=None, help='Alpha value to start from (inclusive)')
    args = parser.parse_args()
    
    # Update validation frequency if specified
    if args.validation_freq != VALIDATION_FREQ:
        VALIDATION_FREQ = args.validation_freq
        # Update in the setups dictionary
        for setup in setups.values():
            setup['validation_steps'] = VALIDATION_FREQ
    
    # If specific experiment parameters are provided, run that single experiment
    if args.setup is not None and args.alpha is not None and args.beta is not None:
        print(f"Running single experiment:")
        print(f"  Setup: {args.setup}")
        print(f"  Alpha: {args.alpha}")
        print(f"  Beta: {args.beta}")
        if args.seed is not None:
            print(f"  Seed: {args.seed}")
        print(f"  Fixed Steps: {args.steps}")
        print(f"  Validation Frequency: {VALIDATION_FREQ}")
        run_experiment(args.setup, args.alpha, args.beta, args.seed, args.steps)
    
    # If run_all flag is set, run all experiments in parallel
    elif args.run_all or args.start_alpha is not None:
        run_all_experiments_parallel(args.start_alpha, args.steps)
    
    # If no valid arguments provided, show help
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Run a single experiment directly:")
        print("  accelerate launch run_serum_language_fixed_training_steps.py --setup gemma-2-0.1B_MaxEnt --alpha 0.3 --beta 0.1")
        print("  accelerate launch run_serum_language_fixed_training_steps.py --setup gemma-2-0.1B_MaxEnt --alpha 0.3 --beta 0.1 --seed 42")
        print("  # Run with specific number of steps:")
        print("  accelerate launch run_serum_language_fixed_training_steps.py --setup gemma-2-0.1B_MaxEnt --alpha 0.3 --beta 0.1 --steps 750")
        print("  # Run with custom validation frequency:")
        print("  accelerate launch run_serum_language_fixed_training_steps.py --setup gemma-2-0.1B_MaxEnt --alpha 0.3 --beta 0.1 --validation_freq 100")
        print("  # Run all experiments in parallel with 4 GPUs per experiment:")
        print("  python run_serum_language_fixed_training_steps.py --run_all")
        print("  # Run all experiments with custom steps:")
        print("  python run_serum_language_fixed_training_steps.py --run_all --steps 1000")
        print("  # Resume from a specific alpha value:")
        print("  python run_serum_language_fixed_training_steps.py --start_alpha 0.4")