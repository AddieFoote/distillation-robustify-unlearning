from src.tools.relearn_wmdp import relearn
from src.utils.paths import CACHE_DIR, DATASET_DIR, WMDP_MODEL_DIR
from accelerate import Accelerator
from utils.loss_functions import custom_login
from src.utils.validation_functions import get_wmdp_cyber_eval_fn, get_wmdp_bio_eval_fn, get_both_wmdp_eval_fn, get_loss_eval_fn
from utils.parallel_launch import launch_in_parallel_one_per_gpu, get_parallel_launch_wrapper

FINAL_RUN = True

SETUPS_TO_RUN = [
    "wmdp" # wmdp-shrink-perturb
]
eval_on_loss = False
MODELS_TO_RUN = {
    'partial_distill_bio_RMU' : 'distilled_partial_distill_models/general/bio_rmu/basic-all data-lr_1.000000e-05-seed_SEED/final_model',
    'partial_distill_cyber_RMU' : 'distilled_partial_distill_models/general/cyber_rmu/basic-all data-lr_1.000000e-05-seed_SEED/final_model',

    'bio_RMU': 'saved_unlearned_models/RMU/bio_lr_5.00e-05_alpha_0.50_seed_SEED/final_model',
    'cyber_RMU': 'saved_unlearned_models/RMU/cyber_lr_2.00e-05_alpha_0.50_seed_SEED/final_model',
    
    # 'partial_distill_bio_MaxEnt': 'distilled_partial_distill_models/general/bio_maxent/basic-all data-lr_2.000000e-05-seed_SEED/final_model',
    # 'partial_distill_cyber_MaxEnt': 'distilled_partial_distill_models/general/cyber_maxent/basic-all data-lr_2.000000e-05-seed_SEED/final_model',
    'partial_distill_cyber_RMU_1': 'distilled_partial_distill_models/cyber/cyber_rmu/basic-all data-lr_2.000000e-05-base-p_0-seed_SEED/final_model',
    
    # 'partial_distill_cyber_MaxEnt-RMU': 'distilled_partial_distill_models/cyber/cyber_maxent_and_rmu/basic-all data-lr_2.000000e-05-base-p_0-seed_SEED/final_model',

    # 'partial_distill_cyber_MaxEnt-RMU_beta1': 'distilled_partial_distill_models/beta-cyber/cyber_maxent_and_rmu/basic-beta-all data-lr_2.000000e-05-base-p_0-seed_SEED/final_model',
    # 'partial_distill_cyber_MaxEnt_beta1': 'distilled_partial_distill_models/beta-cyber/cyber_maxent/basic-beta-all data-lr_2.000000e-05-base-p_0-seed_SEED/final_model',
    # 'partial_distill_cyber_RMU_beta1': 'distilled_partial_distill_models/beta-cyber/cyber_rmu/basic-beta-all data-lr_2.000000e-05-base-p_0-seed_SEED/final_model',
    # 'bio_MaxEnt': 'saved_unlearned_models/MaxEnt/bio_lr_2.00e-05_alpha_0.30_seed_SEED/final_model',
    # 'cyber_MaxEnt': 'saved_unlearned_models/MaxEnt/cyber_lr_2.00e-05_alpha_0.20_seed_SEED/final_model',
    'bio_SAM': 'unlearned_models/MaxEnt-SAM-kl/bio_lr_2.00e-05_alpha_0.30_seed_SEED/final_model',
    'cyber_SAM': 'unlearned_models/MaxEnt-SAM-kl/cyber_lr_2.00e-05_alpha_0.50_seed_SEED/final_model',
    'bio_repnoise': 'unlearned_models/MaxEnt-repnoise-kl/bio_lr_2.00e-05_alpha_0.30_seed_SEED/final_model',
    'cyber_repnoise': 'unlearned_models/MaxEnt-repnoise-kl/cyber_lr_2.00e-05_alpha_0.50_seed_SEED/final_model',

    # 'base_cyber': 'gemma-2-2b',
    # 'base_bio': 'gemma-2-2b',

    'cyber_MaxEnt_1': 'saved_unlearned_models/MaxEnt/cyber_lr_3.50e-05_alpha_0.30_seed_SEED/final_model',
    'cyber_MaxEnt_2': 'saved_unlearned_models/MaxEnt/cyber_lr_5.00e-05_alpha_0.30_seed_SEED/final_model',
    'bio_MaxEnt_1': 'saved_unlearned_models/MaxEnt/bio_lr_2.00e-05_alpha_0.10_seed_SEED/final_model',
    'bio_MaxEnt_2': 'saved_unlearned_models/MaxEnt/bio_lr_5.00e-05_alpha_0.75_seed_SEED/final_model',

    'bio_RMU_1': 'saved_unlearned_models/RMU/bio_lr_1.00e-04_alpha_0.10_seed_SEED/final_model',
    'bio_RMU_2': 'saved_unlearned_models/RMU/bio_lr_1.00e-04_alpha_0.30_seed_SEED/final_model',

    'cyber_RMU_1': 'saved_unlearned_models/RMU/cyber_lr_2.00e-05_alpha_0.10_seed_SEED/final_model',
    'cyber_RMU_2': 'saved_unlearned_models/RMU/cyber_lr_2.00e-05_alpha_0.20_seed_SEED/final_model',

    'bio_SAM': 'unlearned_models/MaxEnt-SAM-kl/bio_lr_2.00e-05_alpha_0.30_seed_SEED/final_model',
    'cyber_SAM': 'unlearned_models/MaxEnt-SAM-kl/cyber_lr_2.00e-05_alpha_0.50_seed_SEED/final_model',
    'bio_repnoise': 'unlearned_models/MaxEnt-repnoise-kl/bio_lr_2.00e-05_alpha_0.30_seed_SEED/final_model',
    'cyber_repnoise': 'unlearned_models/MaxEnt-repnoise-kl/cyber_lr_2.00e-05_alpha_0.50_seed_SEED/final_model',

    'cyber-sam_1': 'unlearned_models/MaxEnt-SAM/cyber_lr_3.50e-05_alpha_0.30_seed_SEED/final_model',
    'cyber-repnoise_1': 'unlearned_models/MaxEnt-repnoise/cyber_lr_3.50e-05_alpha_0.30_seed_SEED/final_model',
    'bio-sam_1': 'unlearned_models/MaxEnt-SAM/bio_lr_2.00e-05_alpha_0.10_seed_SEED/final_model',
    'bio-repnoise_1': 'unlearned_models/MaxEnt-repnoise/bio_lr_2.00e-05_alpha_0.10_seed_SEED/final_model',
}

SWEEP_SEEDS = [42, 43, 44, 45]

DATA_TO_RUN = {
    'cyber-forget/retain' : ([
                                f'{DATASET_DIR}/pretrain/train_cyber-forget-corpus.jsonl',
                                f'{DATASET_DIR}/pretrain/train_cyber-retain-corpus.jsonl',
                            ], [.5, .5], [2.15e-5, 1e-5, 4.62e-6,]), 
    'cyber-initial' : ([
                                f'{DATASET_DIR}/pretrain/train_cyber-forget-corpus.jsonl',
                                f'{DATASET_DIR}/pretrain/train_cyber-retain-corpus.jsonl',
                            ], [.5, .5], [0e-5]),

    'cyber-forget/retain-long' : ([
                                f'{DATASET_DIR}/pretrain/train_cyber-forget-corpus.jsonl',
                                f'{DATASET_DIR}/pretrain/train_cyber-retain-corpus.jsonl',
                            ], [.5, .5], [1e-5, 2.15e-5, 4.62e-6]),

    'cyber-forget/retain-qa': ([
                                    f'{DATASET_DIR}/pretrain/train_wmdp-cyber-forget-corpus_qa.jsonl',
                                    f'{DATASET_DIR}/pretrain/train_wmdp-cyber-retain-corpus_qa.jsonl'
                                ], [.5, .5], [1e-5, 2.15e-5, 4.62e-6,]),

    'cyber-forget/retain-qa-long': ([
                                    f'{DATASET_DIR}/pretrain/train_wmdp-cyber-forget-corpus_qa.jsonl',
                                    f'{DATASET_DIR}/pretrain/train_wmdp-cyber-retain-corpus_qa.jsonl'
                                ], [.5, .5], [4.62e-6, 2.15e-6,]),

    'cyber-wiki-qa-long': ([
                                    f'{DATASET_DIR}/pretrain/train_wikitext.jsonl',
                                    f'{DATASET_DIR}/pretrain/train_wmdp-cyber-forget-corpus_qa.jsonl',
                                    f'{DATASET_DIR}/pretrain/train_wmdp-cyber-retain-corpus_qa.jsonl',
                                ], [.5, .25, .25], [1e-5, 4.62e-6, 2.15e-6,]),

    'bio-forget/retain' : ([
                                f'{DATASET_DIR}/pretrain/train_bio_remove_dataset.jsonl',
                                f'{DATASET_DIR}/pretrain/train_bio_retain_dataset.jsonl',
                            ], [.5, .5], [1e-5, 2.15e-5, 4.62e-6]),
    'bio-initial' : ([
                                f'{DATASET_DIR}/pretrain/train_bio_remove_dataset.jsonl',
                                f'{DATASET_DIR}/pretrain/train_bio_retain_dataset.jsonl',
                            ], [.5, .5], [0e-5]),
    'bio-forget/retain-long' : ([
                                f'{DATASET_DIR}/pretrain/train_bio_remove_dataset.jsonl',
                                f'{DATASET_DIR}/pretrain/train_bio_retain_dataset.jsonl',
                            ], [.5, .5], [1e-5, 2.15e-5, 4.62e-6]),

    'bio-forget/retain-qa': ([
                                    f'{DATASET_DIR}/pretrain/train_wmdp-bio_remove_dataset_qa.jsonl',
                                    f'{DATASET_DIR}/pretrain/train_wmdp-bio_retain_dataset_qa.jsonl'
                                ], [.5, .5], [1e-5, 2.15e-5, 4.62e-6]),

    'bio-forget/retain-qa-long': ([
                                    f'{DATASET_DIR}/pretrain/train_wmdp-bio_remove_dataset_qa.jsonl',
                                    f'{DATASET_DIR}/pretrain/train_wmdp-bio_retain_dataset_qa.jsonl'
                                ], [.5, .5], [4.62e-6, 2.15e-6]),
    'bio-wiki-qa-long': ([
                                    f'{DATASET_DIR}/pretrain/train_wikitext.jsonl',
                                    f'{DATASET_DIR}/pretrain/train_wmdp-bio_remove_dataset_qa.jsonl',
                                    f'{DATASET_DIR}/pretrain/train_wmdp-bio_retain_dataset_qa.jsonl',

                                ], [.5, .25, .25], [1e-5, 4.62e-6, 2.15e-6,]),
}

custom_login()

setups = {
    "wmdp": {
        'model_name'       : f"{WMDP_MODEL_DIR}/PATH",
        'train_files'   : [],
        'interleave_probs': [],
        'stopping_strategy': 'all_exhausted',
        'output_dir'       : f"{WMDP_MODEL_DIR}/relearned_models/TBD",
        'cache_dir'        : CACHE_DIR,
        'join_or_subsequence'         : True,
        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 20,
        'gradient_accumulation_steps' : 1,
        'epochs'                      : 5,
        'learning_rate'               : 'LR',
        'max_steps'                   : 500,
        'num_warmup_steps'            : 1,
        'validation_steps'            : [10, 25, 50, 150, 500],
        'save_checkpoint_steps'       : 999,
        'scheduler_type'              : "cosine",
        'min_lr'                      : 'LR',
        'weight_decay'                : 0.0,
        'gradient_clipping_threshold' : 1.0,
        'max_length'                  : 256,

        'use_wandb'        : True,
        'wandb_project'    : "relearn_CORPUS",
        'wandb_run_name'   : 'TBD',
        'use_local_record' : True,
        'path_local_record': f"{WMDP_MODEL_DIR}/local_records/relearned_models/TBD.txt",
        'save_models'      : False,
        'shrink_perturb_relearning' : 0
    },
    "wmdp-shrink-perturb": {
        'model_name'       : f"{WMDP_MODEL_DIR}/PATH",
        'train_files'   : [],
        'interleave_probs': [],
        'stopping_strategy': 'all_exhausted',
        'output_dir'       : f"{WMDP_MODEL_DIR}/relearned_models/sp-TBD",
        'cache_dir'        : CACHE_DIR,
        'join_or_subsequence'         : True,
        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 20,
        'gradient_accumulation_steps' : 1,
        'epochs'                      : 5,
        'learning_rate'               : 'LR',
        'max_steps'                   : 500,
        'num_warmup_steps'            : 1,
        'validation_steps'            : [10, 25, 50, 150, 500],
        'save_checkpoint_steps'       : 999,
        'scheduler_type'              : "cosine",
        'min_lr'                      : 'LR',
        'weight_decay'                : 0.0,
        'gradient_clipping_threshold' : 1.0,
        'max_length'                  : 256,

        'use_wandb'        : True,
        'wandb_project'    : "relearn_CORPUS",
        'wandb_run_name'   : 'TBD',
        'use_local_record' : True,
        'path_local_record': f"{WMDP_MODEL_DIR}/local_records/relearned_models/sp-TBD.txt",
        'save_models'      : False,
        'shrink_perturb_relearning' : .05
    },
}

def launch_relearn(setup_id, lr, model, files, seed, eval_on_loss):
    name, path = model
    path = path.replace("SEED", str(seed))
    file_name, files_tup = files
    train_files, interleave_probs = files_tup
    if eval_on_loss and "qa" in file_name:
        name_tbd = 'loss_eval-qa_new/'
    elif eval_on_loss:
        name_tbd = 'loss_eval_new/'
    else:
        name_tbd = ''
    name_tbd += f"data-{file_name}_lr-{lr}_model-{name}_seed-{seed}_sparse-evals"

    current_setup = setups[setup_id].copy()

    current_setup['model_name'] = current_setup['model_name'].replace('PATH', path)
    current_setup['path_local_record'] = current_setup['path_local_record'].replace('TBD', name_tbd)
    current_setup['output_dir'] = current_setup['output_dir'].replace('TBD', name_tbd)
    current_setup['wandb_run_name'] = current_setup['wandb_run_name'].replace('TBD', name_tbd)
    current_setup['seed'] = seed
    if 'initial' in file_name:
        current_setup['validation_steps'] = [0]
        current_setup['max_steps'] = 1

    if 'long' in setup_id:
        assert current_setup['batch_size'] % 4 == 0
        current_setup['max_len'] *= 4 
        current_setup['batch_size'] /= 4

    accelerator = Accelerator()
    train_percent = None
    if eval_on_loss:
        train_percent = .90
        assert 'retain' in train_files[1] and ('forget' in train_files[0] or 'remove' in train_files[0])
        eval_fn = get_loss_eval_fn(
            accelerator=accelerator,
        )
    elif 'cyber' in name.lower():
        assert 'bio' not in name.lower()
        current_setup['wandb_project'] = current_setup['wandb_project'].replace('CORPUS', 'cyber')
        eval_fn = get_wmdp_cyber_eval_fn(accelerator, large_eval=FINAL_RUN, no_mmlu=not 'initial' in file_name)
    elif 'bio'in name.lower():
        assert 'cyber' not in name.lower()
        current_setup['wandb_project'] = current_setup['wandb_project'].replace('CORPUS', 'bio')
        eval_fn = get_wmdp_bio_eval_fn(accelerator, large_eval=FINAL_RUN, no_mmlu=not 'initial' in file_name)
    else:
        raise ValueError("name should have bio or cyber")

    relearn(
        model_name       = current_setup['model_name'],
        train_files      = train_files,
        interleave_probs = interleave_probs,
        stopping_strategy= current_setup['stopping_strategy'],
        output_dir       = current_setup['output_dir'],
        cache_dir        = current_setup['cache_dir'],
        dataset_cache_dir= current_setup['cache_dir'],
        eval_fn          = eval_fn,
        accelerator      = accelerator,
        join_or_subsequence   = current_setup['join_or_subsequence'],
        seed             = current_setup['seed'],
        device           = current_setup['device'],
        batch_size       = current_setup['batch_size'],
        gradient_accumulation_steps = current_setup['gradient_accumulation_steps'],
        epochs           = current_setup['epochs'],
        learning_rate    = lr,
        max_steps        = current_setup['max_steps'],   
        num_warmup_steps = current_setup['num_warmup_steps'],
        validation_steps = current_setup['validation_steps'],
        save_checkpoint_steps = current_setup['save_checkpoint_steps'],
        scheduler_type   = current_setup['scheduler_type'],  
        min_lr           = lr,          
        weight_decay     = current_setup['weight_decay'],    
        gradient_clipping_threshold = current_setup['gradient_clipping_threshold'], 
        max_length       = current_setup['max_length'],
        use_wandb        = current_setup['use_wandb'],
        wandb_project    = current_setup['wandb_project'],
        wandb_run_name   = current_setup['wandb_run_name'],
        use_local_record = current_setup['use_local_record'],
        path_local_record= current_setup['path_local_record'],
        overwrite_ok     = True,
        save_models      = current_setup['save_models'],
        shrink_perturb_relearning = current_setup['shrink_perturb_relearning'],
        train_percent = train_percent
    )

if __name__ == "__main__":
    # ----------------------------------------------------------------- #
    # Run all experiments, if possible in parallel
    # ----------------------------------------------------------------- #
    # Create list of the setups (arguments for run_experiment) for all the experiments we want to run 
    experiments = []
    for setup_id in SETUPS_TO_RUN:
        for seed in SWEEP_SEEDS:
            for model_name, model_path in MODELS_TO_RUN.items():
                for data_name, (files, probs, lrs) in DATA_TO_RUN.items():
                    for lr in lrs:
                        if "bio" in data_name and "bio" in model_name or "cyber" in data_name and "cyber" in model_name:
                            experiments.append((setup_id, lr, (model_name, model_path), (data_name, (files, probs)), seed, eval_on_loss))
                        else:
                            print(f"NOT ADDING, model and data don't match, setup: {data_name}, model: {model_name}")

    # Gets a wrapper function compatable with the parallel launch function
    parallel_fn = get_parallel_launch_wrapper(launch_relearn)
    # calls run_experiment in parallel on a separate gpu for each experiment setup when a gpu is free
    launch_in_parallel_one_per_gpu(experiment_list=experiments, experiment_fn=parallel_fn)
