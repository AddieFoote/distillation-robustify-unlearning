import os
import torch
import copy

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Gemma2Config,
)

try:
    from huggingface_hub import login
    HUGGINGFACE_HUB_AVAILABLE = True
except ImportError:
    HUGGINGFACE_HUB_AVAILABLE = False

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.paths import CACHE_DIR, MODEL_DIR

MODEL = "google/gemma-2-2b"
HF_TOKEN_PATH = "tokens/hf_token.txt"

# Change this to wherever you want to save your smaller model
NEW_MODEL_DIR = MODEL_DIR + "/random_init_models/gemma-2-"
REDUCE_TO = "0.1B"

custom_configurations = {
    "0.1B":{
        "num_hidden_layers":14,
        "num_attention_heads":8,
        "intermediate_size":1280,
        "hidden_size":320,
        "head_dim":128,
    },
    "0.3B":{
        "num_hidden_layers":14,
        "num_attention_heads":8,
        "intermediate_size":3072,
        "hidden_size":768,
        "head_dim":128,
    },
    "0.6B":{
        "num_hidden_layers":18,
        "num_attention_heads":8,
        "intermediate_size":4096,
        "hidden_size":1024,
        "head_dim":256,
    },
    "0.9B":{
        "num_hidden_layers":22,
        "num_attention_heads":8,
        "intermediate_size":5120,
        "hidden_size":1280,
        "head_dim":256,
    },
}

def main(REDUCE_TO):
    # 1) Optionally log into HF if you have a token
    hf_token = None
    if os.path.isfile(HF_TOKEN_PATH):
        with open(HF_TOKEN_PATH, "r", encoding="utf-8") as f:
            token = f.read().strip()
            if token:
                hf_token = token

    if hf_token and HUGGINGFACE_HUB_AVAILABLE:
        try:
            print(f"[reduce_gemma_model.py] Logging into Hugging Face with token from {HF_TOKEN_PATH}...")
            login(token=hf_token, add_to_git_credential=True)
        except Exception as e:
            print(f"[Warning] Could not login: {e}")
    else:
        print("[reduce_gemma_model.py] No valid HF token found or huggingface_hub not installed. Skipping HF login.")

    # 2) Load the original config from the 2B model
    print(f"[reduce_gemma_model.py] Loading config of '{MODEL}' from HF...")
    original_config = AutoConfig.from_pretrained(
        MODEL,
        cache_dir=CACHE_DIR,
    )

    # 3) Convert to dict, modify the relevant fields for smaller model
    config_dict = original_config.to_dict()

    config_dict["num_hidden_layers"] = custom_configurations[REDUCE_TO]["num_hidden_layers"]
    config_dict["num_attention_heads"] = custom_configurations[REDUCE_TO]["num_attention_heads"]
    config_dict["intermediate_size"] = custom_configurations[REDUCE_TO]["intermediate_size"]
    config_dict["hidden_size"] = custom_configurations[REDUCE_TO]["hidden_size"]
    config_dict["head_dim"] = custom_configurations[REDUCE_TO]["head_dim"]

    # Create a new Gemma2Config from the updated dict
    smaller_config = Gemma2Config(**config_dict)

    print("[reduce_gemma_model.py] Creating a new random-initialized model with smaller config...")
    model = AutoModelForCausalLM.from_config(smaller_config)

    # 4) Load and adjust tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        cache_dir=CACHE_DIR,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 5) Print parameter count of the new small model
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[reduce_gemma_model.py] New small model size: {total_params/1e6:.2f}M parameters")

    # 6) Save the new model & tokenizer
    print(f"[reduce_gemma_model.py] Saving new small model to: {NEW_MODEL_DIR+REDUCE_TO}")
    os.makedirs(NEW_MODEL_DIR+REDUCE_TO, exist_ok=True)
    model.save_pretrained(NEW_MODEL_DIR+REDUCE_TO)
    tokenizer.save_pretrained(NEW_MODEL_DIR+REDUCE_TO)

    print("[reduce_gemma_model.py] Done! You can now load from this directory with:")
    print(f"  AutoModelForCausalLM.from_pretrained('{NEW_MODEL_DIR+REDUCE_TO}')")

if __name__ == "__main__":
    for model_size in ['0.1B', '0.3B', '0.6B', '0.9B']:
        main(model_size)
