import os
   
CACHE_DIR='/workspace/.cache'
DATASET_DIR='/workspace/datasets'
MODEL_DIR='/workspace/models/non-wmdp'
WMDP_MODEL_DIR='/workspace/models/wmdp'
WANDB_API_KEY_PATH = "tokens/wandb_token.txt"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = CACHE_DIR
