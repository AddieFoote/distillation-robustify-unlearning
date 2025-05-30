
import os
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.paths import CACHE_DIR, WMDP_MODEL_DIR
from utils.loss_functions import custom_makedirs, custom_login


SAVE_DIR = WMDP_MODEL_DIR + "/gemma-2-2b"

def main():
    # 1) Log into HF
    custom_login()
    
    # 2) Load model and tokenizer from the 2B model
    print(f"[download_gemma.py] Loading google/gemma-2-2b from HF...")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b",
        cache_dir=CACHE_DIR,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-2-2b",
        cache_dir=CACHE_DIR,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 3) Print parameter count of the new small model
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[download_gemma.py] Model size: {total_params/1e6:.2f}M parameters")

    # 4) Save the model & tokenizer
    print(f"[download_gemma.py] Saving model to: {SAVE_DIR}")
    custom_makedirs(SAVE_DIR, exist_ok=False)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    print("[download_gemma.py] Done! You can now load from this directory with:")
    print(f"  AutoModelForCausalLM.from_pretrained('{SAVE_DIR}')")

if __name__ == "__main__":
    main()