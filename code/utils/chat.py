import os
import torch
import sys

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
try:
    from huggingface_hub import login
    HUGGINGFACE_HUB_AVAILABLE = True
except ImportError:
    HUGGINGFACE_HUB_AVAILABLE = False

MODEL = "models/unlearned_models/ga/gemma-2-0.1B_eng+kor/final_model" 
HF_TOKEN_PATH = "tokens/hf_token.txt"
CACHE_DIR = "hf_cache"

RANDOM_INIT = False    # True => create a random-init model from the config
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.9
TOP_K = 50
TOP_P = 0.9

def main():
    hf_token = None
    if os.path.isfile(HF_TOKEN_PATH):
        with open(HF_TOKEN_PATH, "r", encoding="utf-8") as f:
            token = f.read().strip()
            if token:
                hf_token = token

    if hf_token and HUGGINGFACE_HUB_AVAILABLE:
        print(f"[chat.py] Logging into Hugging Face with token from {HF_TOKEN_PATH}...")
        try:
            login(token=hf_token, add_to_git_credential=True)
        except Exception as e:
            print(f"[Warning] Could not login: {e}")
    else:
        print(f"[chat.py] No valid HF token found or huggingface_hub not installed. Skipping HF login.")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if RANDOM_INIT:
        # (A) Create a random-initialized model from the config
        print(f"[chat.py] Using a random-init model based on config of '{MODEL}'")

        config = AutoConfig.from_pretrained(
            MODEL,
            cache_dir=CACHE_DIR,
            )
        model = AutoModelForCausalLM.from_config(
            config,
            )
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL,
            cache_dir=CACHE_DIR,
            )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    else:
        # (B) Load the fully pretrained model (usual route)
        print(f"[chat.py] Loading pretrained model from: {MODEL}")

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL,
            cache_dir=CACHE_DIR,
            )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            cache_dir=CACHE_DIR,
            )

    # Check param count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[chat.py] This model has {total_params/1e6:.2f}M params.")

    model.to(device)
    model.eval()

    print("\n======================================================")
    if RANDOM_INIT:
        print(f"Loaded a RANDOM model from config of '{MODEL}' (untrained). Expect gibberish!")
    else:
        print(f"Loaded a pretrained model: {MODEL}")
    print("Type 'quit' or 'exit' to terminate.\n")

    while True:
        user_input = input("User: ")
        if not user_input.strip():
            continue
        if user_input.lower().strip() in ("quit", "exit"):
            print("Goodbye!")
            break

        inputs = tokenizer(user_input, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
            )

        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_text = full_text[len(user_input):]

        print(f"Assistant: {assistant_text}\n")


if __name__ == "__main__":
    main()
                