import os
import json
import random
import math
import sys
from functools import partial
from multiprocessing import Pool, cpu_count
from transformers import AutoTokenizer
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.paths import CACHE_DIR, DATASET_DIR, MODEL_DIR
from utils.loss_functions import custom_makedirs
# -----------------------------------------------------
# Global config
# -----------------------------------------------------
random.seed(42)
DOC_MAX_LEN = 2048
# chunk of lines to read from disk
CHUNK_SIZE = 1_000_000
# subchunk size for parallel tasks
SUBCHUNK_SIZE = 5_000
DOC_MIN_LEN = 50
assert DOC_MIN_LEN < DOC_MAX_LEN

# number of processes (use fewer if you want less overhead)
NUM_PROCESSES = cpu_count()
print(f"NUM_PROCESSES = {NUM_PROCESSES}", flush=True)

TRAIN_TARGET = 1_000_000_000
VALID_TARGET = 500_000

KOR_FILE = DATASET_DIR + "/fineweb/fineweb2_kor.jsonl"
ENG_FILE = DATASET_DIR + "/fineweb/fineweb_eng_sample-10BT.jsonl"

OUT_DIR = DATASET_DIR + "/pretrain"
KOR_TRAIN_OUT = os.path.join(OUT_DIR, "train_kor.jsonl")
KOR_VALID_OUT = os.path.join(OUT_DIR, "valid_kor.jsonl")
ENG_TRAIN_OUT = os.path.join(OUT_DIR, "train_eng.jsonl")
ENG_VALID_OUT = os.path.join(OUT_DIR, "valid_eng.jsonl")

# load tokenizer once
tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b') 

def tokenize_lines(lines, doc_max_len, use_one_per_line):
    """
    Tokenizes a list of JSON lines. 
    Returns a list of (input_ids, attention_mask) or None if an issue.
    """
    results = []
    total = 0
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            results.append(None)
            continue
        if type(record) is str:
            text = record
        else:
            if 'text' in record or 'response' in record or 'output' in record:
                if 'text' in record:
                    text = record.get("text", "")
                elif 'response' in record:
                    text = record.get('response', "")
                elif 'output' in record:
                    text = record.get('output', '')
                encoded = tokenizer(text, add_special_tokens=False, return_attention_mask=True)
                inp = encoded["input_ids"]
                att = encoded["attention_mask"]
                loss_mask = [1] * len(inp)

            elif 'qa' in record:
                assert 'question' in record['qa'] and 'answer' in record['qa']
                question = record['qa'].get('question', "")
                question += '\nAnswer: '
                answer = record['qa'].get('answer', "")
                encoded_q = tokenizer(question, add_special_tokens=False, return_attention_mask=True)
                encoded_a = tokenizer(answer, add_special_tokens=False, return_attention_mask=True)
                inp = encoded_q['input_ids'] + encoded_a['input_ids']
                att = encoded_q['attention_mask'] + encoded_a['attention_mask']
                loss_mask = [0] * len(encoded_q['input_ids']) + [1] * len(encoded_a['input_ids'])
                
            else:
                print(f"Did not find response/text/output or question+answer field in the following: {line}")
                continue

        if use_one_per_line:
            if len(inp) > DOC_MAX_LEN:
                results.append(None)
            else:
                total += 1
                results.append((inp, att, loss_mask))
        else:
            for i in range(0, len(inp) - doc_max_len + 1, doc_max_len):
                chunk_inp = inp[i:i + doc_max_len]
                chunk_att = att[i:i + doc_max_len]
                chuck_loss_mask = loss_mask[i:i + doc_max_len]
                if len(chunk_inp) < doc_max_len:
                    continue
                results.append((chunk_inp, chunk_att, chuck_loss_mask))

        
    print(f"PERCENT LINES KEPT = {len(results) / (len(lines) * 1.0)}")
    return results

def yield_chunks(filepath, chunk_size):
    """
    Yields lists of up to chunk_size lines from filepath
    """
    buf = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            buf.append(line)
            if len(buf) == chunk_size:
                yield buf
                buf = []
    if buf:
        yield buf

def split_into_subchunks(lines, subchunk_size):
    return [lines[i : i+subchunk_size] for i in range(0, len(lines), subchunk_size)]

def build_and_save_dataset(
    filepath,
    train_target,
    valid_target,
    train_out_path,
    valid_out_path,
    doc_max_len,
    use_one_per_line,
    chunk_size=CHUNK_SIZE,
    subchunk_size=SUBCHUNK_SIZE
):
    custom_makedirs(os.path.dirname(train_out_path), exist_ok=True)
    custom_makedirs(os.path.dirname(valid_out_path), exist_ok=True)

    train_so_far = 0
    valid_so_far = 0

    fout_train = open(train_out_path, "a", encoding="utf-8")
    fout_valid = open(valid_out_path, "a", encoding="utf-8")

    with Pool(processes=NUM_PROCESSES) as pool:
        chunk_index = 0
        for lines_chunk in yield_chunks(filepath, chunk_size):
            chunk_index += 1
            print(f"\nReading chunk #{chunk_index} of size {len(lines_chunk)} from {filepath}", flush=True)

            if train_so_far >= train_target and valid_so_far >= valid_target:
                print("**All targets met: stopping read**", flush=True)
                break

            subchunks = split_into_subchunks(lines_chunk, subchunk_size)
            print(f" -> Split into {len(subchunks)} subchunks (size ~{subchunk_size}).", flush=True)

            # Map each subchunk in parallel
            # returns an iterator of lists-of-tuples
            tokenize_with_max_len = partial(tokenize_lines, doc_max_len=doc_max_len, use_one_per_line=use_one_per_line)

            print(" -> Launching parallel tasks for subchunks...", flush=True)
            results_iter = pool.imap(tokenize_with_max_len, subchunks, chunksize=1)

            subchunk_id = 0
            for tokenized_sublist in results_iter:
                subchunk_id += 1
                if tokenized_sublist is None:
                    print(f"    Subchunk {subchunk_id}: None result??", flush=True)
                    continue

                valid_docs = sum(1 for tok in tokenized_sublist if tok is not None)
                print(f"    Subchunk {subchunk_id}: {valid_docs} valid docs tokenized.", flush=True)

                # process each doc
                for doc_idx, tok in enumerate(tokenized_sublist):
                    if tok is None:
                        continue

                    if train_so_far >= train_target and valid_so_far >= valid_target:
                        break

                    input_ids, attention_mask, loss_mask = tok
                    doc_len = len(input_ids)

                    need_train = max(0, train_target - train_so_far)
                    need_valid = max(0, valid_target - valid_so_far)

                    # Decide if doc goes to valid or train
                    if train_so_far < train_target and valid_so_far < valid_target:
                        # Weighted random
                        p_valid = valid_target / float(train_target + valid_target)
                        pick_valid = (random.random() < p_valid)
                    elif train_so_far >= train_target and valid_so_far < valid_target:
                        pick_valid = True
                    elif valid_so_far >= valid_target and train_so_far < train_target:
                        pick_valid = False
                    else:
                        # everything is full
                        break

                    if pick_valid:
                        leftover = need_valid
                        if doc_len <= leftover:
                            new_ids = input_ids
                            valid_so_far += doc_len
                        else:
                            new_ids = input_ids[:leftover]
                            valid_so_far += leftover
                        new_mask = attention_mask[:len(new_ids)]
                        new_loss_mask = loss_mask[:len(new_ids)]
                        new_text = tokenizer.decode(new_ids, clean_up_tokenization_spaces=True)

                        record = {
                                    "text": new_text,
                                    "input_ids": new_ids,
                                    "attention_mask": new_mask,
                                    "loss_mask": new_loss_mask
                                 }
                        fout_valid.write(json.dumps(record, ensure_ascii=False) + "\n")

                    else:
                        leftover = need_train
                        if doc_len <= leftover:
                            new_ids = input_ids
                            train_so_far += doc_len
                        else:
                            new_ids = input_ids[:leftover]
                            train_so_far += leftover
                        new_mask = attention_mask[:len(new_ids)]
                        new_loss_mask = loss_mask[:len(new_ids)]
                        new_text = tokenizer.decode(new_ids, clean_up_tokenization_spaces=True)

                        record = {
                                    "text": new_text,
                                    "input_ids": new_ids,
                                    "attention_mask": new_mask,
                                    'loss_mask': new_loss_mask,
                                }
                        fout_train.write(json.dumps(record, ensure_ascii=False) + "\n")

                    # Print status every 100 docs
                    if (doc_idx + 1) % 100 == 0:
                        print(f"      Processed doc {doc_idx+1}/{len(tokenized_sublist)} in subchunk {subchunk_id}. T: {train_so_far}, V: {valid_so_far}", flush=True)

                    if train_so_far >= train_target and valid_so_far >= valid_target:
                        print("**Both targets reached** => breaking out of doc loop", flush=True)
                        break

                if train_so_far >= train_target and valid_so_far >= valid_target:
                    print("**Both targets reached** => break subchunks", flush=True)
                    break

            print(f"After chunk #{chunk_index}: train_so_far={train_so_far}, valid_so_far={valid_so_far}", flush=True)
            if train_so_far >= train_target and valid_so_far >= valid_target:
                print("**Both targets reached** => break chunk reading", flush=True)
                break

    fout_train.close()
    fout_valid.close()

    return train_so_far, valid_so_far


def main():
    # Create the output directory if needed
    if os.path.exists(KOR_FILE) and not os.path.exists(KOR_TRAIN_OUT):
    
        print("Building Korean dataset...", flush=True)
        kor_train_tokens, kor_valid_tokens = build_and_save_dataset(
            filepath=KOR_FILE,
            train_target=TRAIN_TARGET,
            valid_target=VALID_TARGET,
            train_out_path=KOR_TRAIN_OUT,
            valid_out_path=KOR_VALID_OUT,
            doc_max_len=DOC_MAX_LEN,
            use_one_per_line=True,
            chunk_size=CHUNK_SIZE,
            subchunk_size=SUBCHUNK_SIZE
        )
        print(f"Korean set done => train tokens = {kor_train_tokens}, valid tokens = {kor_valid_tokens}", flush=True)
    else:
        print(f"Skipping Korean dataset generation as file {KOR_FILE} does not exist exists or {KOR_TRAIN_OUT} does exist.", flush=True)
    
    if os.path.exists(ENG_FILE) and not os.path.exists(ENG_TRAIN_OUT):
        print("Building English dataset...", flush=True)
        eng_train_tokens, eng_valid_tokens = build_and_save_dataset(
            filepath=ENG_FILE,
            train_target=TRAIN_TARGET,
            valid_target=VALID_TARGET,
            train_out_path=ENG_TRAIN_OUT,
            valid_out_path=ENG_VALID_OUT,
            doc_max_len=DOC_MAX_LEN,
            use_one_per_line=True,
            chunk_size=CHUNK_SIZE,
            subchunk_size=SUBCHUNK_SIZE
        )
        print(f"English set done => train tokens = {eng_train_tokens}, valid tokens = {eng_valid_tokens}", flush=True)
    else:
        print(f"Skipping English dataset generation as file {ENG_FILE} does not exist exists or {ENG_TRAIN_OUT} does exist.", flush=True)
    
    for file_name in ['magpie', 'wikitext', 'magpie-filtered', 'wikipedia', 'magpie-3', 'magpie-phi3', 'magpie-gemma2', 'magpie3-1', 'magpie-qwen', 'magpie-qwen2']:
        DATA_FILE = DATASET_DIR + f"/fineweb/{file_name}.jsonl"

        DATA_TRAIN_OUT = os.path.join(OUT_DIR, f"train_{file_name}.jsonl")
        DATA_VALID_OUT = os.path.join(OUT_DIR, f"valid_{file_name}.jsonl")
        if os.path.exists(DATA_FILE) and not os.path.exists(DATA_TRAIN_OUT):
            print(f"Building {file_name} dataset...", flush=True)
            data_train_tokens, data_valid_tokens = build_and_save_dataset(
                filepath=DATA_FILE,
                train_target=TRAIN_TARGET,
                valid_target=0,
                train_out_path=DATA_TRAIN_OUT,
                valid_out_path=DATA_VALID_OUT,
                doc_max_len=DOC_MAX_LEN,
                use_one_per_line=True,
                chunk_size=CHUNK_SIZE,
                subchunk_size=SUBCHUNK_SIZE
            )
            print(f"{file_name} set done => train tokens = {data_train_tokens}, valid tokens = {data_valid_tokens}", flush=True)
        else:
            print(f"Skipping {file_name} dataset generation as file {DATA_FILE} does not exist exists or {DATA_TRAIN_OUT} does exist.", flush=True)
    for file_name in ['wmdp-cyber-forget-corpus', 'wmdp-cyber-retain-corpus', 'wmdp-bio_retain_dataset', 'wmdp-bio_remove_dataset', "wmdp-wikipedia"]:
        if 'cyber' in file_name:
            DATA_FILE = DATASET_DIR + f"/eric-wmdp-data/{file_name}.jsonl"
        else:
            DATA_FILE =  f"{DATASET_DIR}/wmdp/qa/{file_name}-combined.jsonl"

        DATA_TRAIN_OUT = os.path.join(OUT_DIR, f"train_{file_name}_qa.jsonl")
        DATA_VALID_OUT = os.path.join(OUT_DIR, f"valid_{file_name}_qa.jsonl")
        if os.path.exists(DATA_FILE) and not os.path.exists(DATA_TRAIN_OUT):
            print(f"Building {file_name} dataset...", flush=True)
            data_train_tokens, data_valid_tokens = build_and_save_dataset(
                filepath=DATA_FILE,
                train_target=TRAIN_TARGET,
                valid_target=0,
                train_out_path=DATA_TRAIN_OUT,
                valid_out_path=DATA_VALID_OUT,
                doc_max_len=DOC_MAX_LEN,
                use_one_per_line=True,
                chunk_size=CHUNK_SIZE,
                subchunk_size=SUBCHUNK_SIZE
            )
            print(f"{file_name} set done => train tokens = {data_train_tokens}, valid tokens = {data_valid_tokens}", flush=True)
        else:
            print(f"Skipping {file_name} dataset generation as file {DATA_FILE} does not exist exists or {DATA_TRAIN_OUT} does exist.", flush=True)
    
    for op in ['addition_subtraction', 'multiplication_division', 'all_arithmetic']:
        ARITH_TRAIN_OUT = os.path.join(OUT_DIR, f"train_{op}.jsonl")
        ARITH_VALID_OUT = os.path.join(OUT_DIR, f"valid_{op}.jsonl")
        ARITH_FILE = DATASET_DIR + f"/arithmetic/{op}.jsonl"
 
        if os.path.exists(ARITH_FILE) and not os.path.exists(ARITH_TRAIN_OUT):
            print(f"Building {op} dataset...", flush=True)
            arith_train_tokens, arith_valid_tokens = build_and_save_dataset(
                filepath=ARITH_FILE,
                train_target=TRAIN_TARGET,
                valid_target=0,
                train_out_path=ARITH_TRAIN_OUT,
                valid_out_path=ARITH_VALID_OUT,
                doc_max_len=256,
                use_one_per_line=True,
                chunk_size=CHUNK_SIZE,
                subchunk_size=SUBCHUNK_SIZE
            )
            print(f"Arithmetic set done => train tokens = {arith_train_tokens}, valid tokens = {arith_valid_tokens}", flush=True)
        else:
            print(f"Skipping Arithmetic dataset generation as file {ARITH_FILE} does not exist exists or {ARITH_TRAIN_OUT} does exist.", flush=True)
    
    for file_name in ['cyber-retain-corpus', 'cyber-forget-corpus', 'bio_remove_dataset', 'bio_retain_dataset']:
        WMDP_TRAIN_OUT = os.path.join(OUT_DIR, f"train_{file_name}.jsonl")
        WMDP_VALID_OUT = os.path.join(OUT_DIR, f"valid_{file_name}.jsonl")
        WMDP_FILE = DATASET_DIR + f"/wmdp/{file_name}.jsonl"
 
        if os.path.exists(WMDP_FILE) and not os.path.exists(WMDP_TRAIN_OUT):
            print(f"Building {file_name} dataset...", flush=True)
            wmdp_train_tokens, wmdp_valid_tokens = build_and_save_dataset(
                filepath=WMDP_FILE,
                train_target=TRAIN_TARGET,
                valid_target=0,
                train_out_path=WMDP_TRAIN_OUT,
                valid_out_path=WMDP_VALID_OUT,
                doc_max_len=256,
                use_one_per_line=True,
                chunk_size=CHUNK_SIZE,
                subchunk_size=SUBCHUNK_SIZE
            )
            print(f"WMDP set done => train tokens = {wmdp_train_tokens}, valid tokens = {wmdp_valid_tokens}", flush=True)
        else:
            print(f"Skipping wmdp dataset generation as file {WMDP_FILE} does not exist exists or {WMDP_TRAIN_OUT} does exist.", flush=True)

    print("All done!", flush=True)

if __name__ == "__main__":
    main()
