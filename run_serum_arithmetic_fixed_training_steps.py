"""
Configured for a (any)xH100/H200/A100 GPU server.

Launch Command: python run_serum_arithmetic_fixed_training_steps.py --run_all
"""

from code.tools.serum import serum
from code.utils.paths import CACHE_DIR, DATASET_DIR, MODEL_DIR
from accelerate import Accelerator
from utils.loss_functions import print_acc, custom_login
from utils.validation_functions import get_arithmetic_eval_fn
from utils.parallel_launch import launch_in_parallel_one_per_gpu, get_parallel_launch_wrapper
import os
import signal
import sys
import argparse
import time
from datetime import datetime

# Configuration constants
SETUPS_TO_RUN = ["gemma-2-0.3B_MaxEnt"]
SWEEP_ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # , 0.9, 1.0]
SWEEP_BETAS = [0.1]  # Beta values to sweep through (default is 0.1)
SWEEP_SEEDS = [111]  # Set to None to use default seed, or [123, 456, 789] to sweep over seeds
OVERWRITE_OK = True

# Fixed number of training steps
FIXED_STEPS = 500  # Default number of training steps
VALIDATION_FREQ = 250  # Validate every 250 steps

custom_login()

setups = {
    "gemma-2-0.3B_MaxEnt": {
        'teacher_model_name': f"{MODEL_DIR}/unlearned_models/MaxEnt/gemma-2-0.3B_all_arithmetic+eng_lr_9.0e-05/final_model",
        'student_model_name': f"{MODEL_DIR}/unlearned_models/MaxEnt/gemma-2-0.3B_all_arithmetic+eng_lr_9.0e-05/final_model",
        'eng_train_file': f"{DATASET_DIR}/pretrain/train_eng.jsonl",
        'arithmetic_train_file': f"{DATASET_DIR}/pretrain/train_all_arithmetic.jsonl",
        'eng_valid_file': f"{DATASET_DIR}/pretrain/valid_eng.jsonl",
        'output_dir': f"{MODEL_DIR}/serum_models_arith/gemma-2-0.3B_MaxEnt-arithmetic-SERUM-alpha-beta",
        'cache_dir': CACHE_DIR,
        'dataset_cache_dir': CACHE_DIR,
        'join_or_subsequence': True,
        'interleave_probs': [.75, .25],

        'seed': 42,
        'device': "cuda",
        'batch_size': 15,
        'gradient_accumulation_steps': 24,
        'epochs': 1,
        'learning_rate': 7e-4,
        'max_steps': 1500,  # Set higher than FIXED_STEPS to ensure we can reach it
        'num_warmup_steps': 50,
        'validation_steps': VALIDATION_FREQ,  # Changed to validate less frequently
        'save_checkpoint_steps': 50000,
        'scheduler_type': "cosine",
        'min_lr': 7e-5,
        'weight_decay': 0,
        'gradient_clipping_threshold': 1.0,
        'max_length': 256,

        'use_wandb': True,
        'wandb_project': "gemma-2-0.3B_all_arithmetic+eng_sp_distill",
        'wandb_run_name': None,
        'use_local_record': True,
        'path_local_record': f"{MODEL_DIR}/local_records/serum_models_arith/gemma-2-0.3B_MaxEnt-arithmetic-SERUM-alpha-beta.txt",

        'shrink_perturb_repeat': False
    }
}

class StepTracker:
    """Utility class to track and log training steps"""
    def __init__(self, total_steps):
        self.current_step = 0
        self.total_steps = total_steps
        
    def increment(self, steps):
        self.current_step += steps
        return self.current_step
        
    def should_stop(self):
        return self.current_step >= self.total_steps

def fixed_steps_stop_cond_fn(student_eval_dict, teacher_eval_dict, step_tracker):
    """
    Stop condition function that strictly stops at the specified number of steps.
    
    This function overrides any other stopping criteria and only stops when the
    step counter reaches the designated total steps.
    """
    # Log metrics for informational purposes
    student_eng_loss = student_eval_dict.get("val/eng_ce_loss", float('inf'))
    teacher_eng_loss = teacher_eval_dict.get("val/eng_ce_loss", 0)
    eng_diff = (student_eng_loss - teacher_eng_loss) / max(teacher_eng_loss, 1e-6)
    
    # For retain arithmetic operations (addition and subtraction)
    retain_ops = ["addition", "subtraction"]
    retain_metrics = []
    
    # For forget arithmetic operations (multiplication and division)
    forget_ops = ["multiplication", "division"]
    forget_metrics = []
    
    # Collect accuracy metrics for all operations and formats
    for op in retain_ops + forget_ops:
        for format_type in ["equation", "word_problem"]:
            key = f"val/{op}_{format_type}_acc"
            student_acc = student_eval_dict.get(key, 0)
            teacher_acc = teacher_eval_dict.get(key, 1.0)  # Default to 1.0 if not found
            acc_diff = student_acc - teacher_acc
            
            if op in retain_ops:
                retain_metrics.append((key, acc_diff))
            else:
                forget_metrics.append((key, acc_diff))
    
    # Calculate average difference for operations
    avg_retain_diff = sum(diff for _, diff in retain_metrics) / max(len(retain_metrics), 1)
    avg_forget_diff = sum(diff for _, diff in forget_metrics) / max(len(forget_metrics), 1)
    
    # Log all differences for debugging
    print(f"Training step: {step_tracker.current_step}/{step_tracker.total_steps}")
    print(f"English diff: {eng_diff:.4f}")
    print(f"Retain arithmetic diff: {avg_retain_diff:.4f}")
    print(f"Forget arithmetic diff: {avg_forget_diff:.4f}")
    
    # Only stop when we've reached the specified number of steps
    should_stop = step_tracker.should_stop()
    if should_stop:
        print(f"✅ Reached target step count ({step_tracker.current_step}/{step_tracker.total_steps}). Stopping training.")
    else:
        print(f"⏳ Continuing training: step {step_tracker.current_step}/{step_tracker.total_steps}")
    
    return should_stop

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
    
    # Set max_steps to ensure we can reach our fixed step count
    current_setup['max_steps'] = num_steps
    print(f"Setting max_steps to {current_setup['max_steps']} to ensure we reach fixed_steps={num_steps}")
    
    # Update validation frequency
    current_setup['validation_steps'] = VALIDATION_FREQ
    
    # Update paths and parameters
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
        print(f"\n[run_arithmetic_serum.py] Received interrupt signal for {setup_id}, alpha={alpha}{seed_msg}. Saving model before exiting...")
        # Create an unwrapped model for saving
        unwrapped_student = accelerator.unwrap_model(student_model) if 'student_model' in locals() else None
        
        if unwrapped_student is not None and accelerator.is_main_process:
            interrupted_model_path = os.path.join(current_setup['output_dir'], "interrupted_model")
            os.makedirs(os.path.dirname(interrupted_model_path), exist_ok=True)
            unwrapped_student.save_pretrained(interrupted_model_path)
            tokenizer.save_pretrained(interrupted_model_path) if 'tokenizer' in locals() else None
            print(f"[run_arithmetic_serum.py] Saved interrupted model => {interrupted_model_path}")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    accelerator = Accelerator()
    
    arithmetic_eval_fn = get_arithmetic_eval_fn(
        # gets a function that takes a model returns a dictionary with equation/word problem accuracy for each operation and english validation CE loss
        model_name=current_setup['student_model_name'],
        eng_valid_file=current_setup['eng_valid_file'],
        batch_size=current_setup['batch_size'],
        max_length=current_setup['max_length'],
        cache_dir=current_setup['cache_dir'],
        dataset_cache_dir=current_setup['dataset_cache_dir'],
        num_wiki_batches=50,
        accelerator=accelerator
    )
    
    # Initialize step tracker
    step_tracker = StepTracker(total_steps=num_steps)
    
    # Create a custom stop_cond_fn that uses our step tracker
    def custom_stop_cond_fn(student_eval_dict, teacher_eval_dict):
        # Increment step counter by validation_steps
        step_tracker.increment(current_setup['validation_steps'])
        
        return fixed_steps_stop_cond_fn(
            student_eval_dict, 
            teacher_eval_dict, 
            step_tracker
        )
    
    serum(
        teacher_model_name=current_setup['teacher_model_name'],
        student_model_name=current_setup['student_model_name'],
        train_files=[current_setup['eng_train_file'], current_setup['arithmetic_train_file']],
        interleave_probs=current_setup['interleave_probs'],
        stopping_strategy='first_exhausted',
        join_or_subsequence=current_setup['join_or_subsequence'],
        eval_fn=arithmetic_eval_fn,
        stop_cond_fn=custom_stop_cond_fn,
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

def run_all_experiments_parallel(
    start_alpha=None,
    num_steps=FIXED_STEPS
):
    """Run all experiments in parallel using the parallel launcher"""
    # Filter alphas if start_alpha is specified
    alphas = [a for a in SWEEP_ALPHAS if start_alpha is None or a >= start_alpha]
    
    # Create experiments list based on whether seed sweeping is enabled
    if SWEEP_SEEDS is None:
        # No seed sweeping - use default seed
        experiments = [(setup_id, alpha, beta, None, num_steps) 
                      for setup_id in SETUPS_TO_RUN 
                      for alpha in alphas 
                      for beta in SWEEP_BETAS]
    else:
        # Sweep over seeds
        experiments = [(setup_id, alpha, beta, seed, num_steps) 
                      for setup_id in SETUPS_TO_RUN 
                      for alpha in alphas 
                      for beta in SWEEP_BETAS
                      for seed in SWEEP_SEEDS]
    
    print(f"Running {len(experiments)} experiments in parallel")
    print(f"Alpha values: {alphas}")
    print(f"Beta values: {SWEEP_BETAS}")
    if SWEEP_SEEDS is not None:
        print(f"Seed values: {SWEEP_SEEDS}")
    else:
        print(f"Using default seed: {setups[SETUPS_TO_RUN[0]]['seed']}")
    print(f"Fixed steps: {num_steps}")
    print(f"Validation frequency: {VALIDATION_FREQ}")
    
    # Get wrapper function compatible with parallel launch
    parallel_fn = get_parallel_launch_wrapper(run_experiment)
    
    # Launch experiments in parallel on separate GPUs
    launch_in_parallel_one_per_gpu(experiment_list=experiments, experiment_fn=parallel_fn)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run SERUM arithmetic experiments with fixed steps')
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
    if args.setup is not None and args.alpha is not None:
        print(f"Running single experiment:")
        print(f"  Setup: {args.setup}")
        print(f"  Alpha: {args.alpha}")
        print(f"  Beta: {args.beta if args.beta else 0.1}")
        if args.seed is not None:
            print(f"  Seed: {args.seed}")
        print(f"  Fixed Steps: {args.steps}")
        print(f"  Validation Frequency: {VALIDATION_FREQ}")
        run_experiment(
            args.setup, 
            args.alpha, 
            args.beta if args.beta else 0.1,
            args.seed,
            args.steps
        )
    
    # If run_all flag is set, run all experiments in parallel
    elif args.run_all or args.start_alpha is not None:
        run_all_experiments_parallel(
            args.start_alpha,
            args.steps
        )
    
    # If no valid arguments provided, show help
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Run a single experiment:")
        print("  python run_serum_arithmetic_fixed_training_steps.py --setup gemma-2-0.3B_MaxEnt --alpha 0.3 --beta 0.1")
        print("  python run_serum_arithmetic_fixed_training_steps.py --setup gemma-2-0.3B_MaxEnt --alpha 0.3 --seed 123")
        print("  # Run with specific number of steps:")
        print("  python run_serum_arithmetic_fixed_training_steps.py --setup gemma-2-0.3B_MaxEnt --alpha 0.3 --steps 750")
        print("  # Run with custom validation frequency:")
        print("  python run_serum_arithmetic_fixed_training_steps.py --setup gemma-2-0.3B_MaxEnt --alpha 0.3 --validation_freq 100")
        print("  # Run all experiments in parallel:")
        print("  python run_serum_arithmetic_fixed_training_steps.py --run_all")
        print("  # Run all experiments with custom steps:")
        print("  python run_serum_arithmetic_fixed_training_steps.py --run_all --steps 1000")
        print("  # Resume from a specific alpha value:")
        print("  python run_serum_arithmetic_fixed_training_steps.py --start_alpha 0.4")