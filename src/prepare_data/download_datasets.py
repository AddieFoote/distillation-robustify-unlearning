import os
import orjson 
from datasets import load_dataset
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.paths import CACHE_DIR, DATASET_DIR

OUTPUT_DIR = DATASET_DIR + "/fineweb"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_dataset(dataset_id, subset_name, output_filename, split="train", cache_dir=CACHE_DIR, batch_size=1_000_000, max_rows=None):
    print(f"Downloading subset '{subset_name}' from '{dataset_id}'...")
    if subset_name is None:
        ds = load_dataset(dataset_id, split=split, streaming=True, cache_dir=cache_dir)
    else:
        ds = load_dataset(dataset_id, name=subset_name, split=split, streaming=True, cache_dir=cache_dir)
    total = ds.num_rows if hasattr(ds, "num_rows") and ds.num_rows is not None else None

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    buffer = []
    count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in tqdm(ds, total=total, desc=f"Processing {subset_name}", mininterval=5):
            buffer.append(orjson.dumps(sample).decode("utf-8"))
            count += 1
            if count % batch_size == 0:
                f.write("\n".join(buffer) + "\n")
                # Optionally print a summary less frequently
                print(f"Processed {count} lines so far...")
                buffer.clear()

            if max_rows is not None and count >= max_rows:
                break

        if buffer:
            f.write("\n".join(buffer) + "\n")
            print(f"Processed {count} lines in total.")
    print("Finished streaming dataset to:", output_path)

# ====== REQUIRED FOR WMDP SETTING ======
download_dataset(
    dataset_id="Magpie-Align/Magpie-Llama-3.3-Pro-1M-v0.1",
    output_filename="magpie.jsonl",
    max_rows=10_000_000,
    subset_name=None
)

download_dataset(
    dataset_id="Magpie-Align/Magpie-Llama-3.1-Pro-1M-v0.1",
    output_filename="magpie3-1.jsonl",
    max_rows=10_000_000,
    subset_name=None
)

download_dataset(
    dataset_id="Magpie-Align/Magpie-Qwen2.5-Pro-1M-v0.1",
    output_filename="magpie-qwen.jsonl",
    max_rows=10_000_000,
    subset_name=None
) 

download_dataset(
    dataset_id="Magpie-Align/Llama-3-Magpie-Pro-1M-v0.1",
    output_filename="magpie-3.jsonl",
    max_rows=10_000_000,
    subset_name=None
) 
download_dataset(
    dataset_id="Magpie-Align/Magpie-Qwen2-Pro-1M-v0.1",
    output_filename="magpie-qwen2.jsonl",
    max_rows=10_000_000,
    subset_name=None
) 
download_dataset(
    dataset_id="Magpie-Align/Magpie-Phi3-Pro-1M-v0.1",
    output_filename="magpie-phi3.jsonl",
    max_rows=10_000_000,
    subset_name=None
) 
download_dataset(
    dataset_id="Magpie-Align/Magpie-Gemma2-Pro-534K-v0.1",
    output_filename="magpie-gemma2.jsonl",
    max_rows=10_000_000,
    subset_name=None
) 

download_dataset(
    dataset_id="Magpie-Align/Magpie-Pro-300K-Filtered",
    output_filename="magpie-filtered.jsonl",
    subset_name=None,
    max_rows=10_000_000
)

download_dataset(
    dataset_id="Salesforce/wikitext",
    subset_name="wikitext-2-v1",
    output_filename="wikitext.jsonl",
    max_rows=10_000_000
)

download_dataset(
    dataset_id="legacy-datasets/wikipedia",
    subset_name="20220301.en",
    output_filename="wikipedia.jsonl",
    max_rows=10_000_000
)

# ====== REQUIRED FOR LANGUAGE SETTING ======
download_dataset(
    dataset_id="HuggingFaceFW/fineweb-2",
    subset_name="kor_Hang",
    output_filename="fineweb2_kor.jsonl",
    max_rows=10_000_000
)

# ====== REQUIRED FOR LANGUAGE, ARITHMETIC, AND WMDP SETTING ======
download_dataset(
    dataset_id="HuggingFaceFW/fineweb-edu",
    subset_name="sample-10BT",
    output_filename="fineweb_eng_sample-10BT.jsonl",
    max_rows=10_000_000
)