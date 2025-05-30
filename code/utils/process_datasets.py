from transformers import AutoTokenizer  # For tokenizer
from datasets import Dataset  # For dataset handling


def make_sequence_length(train_ds_list, tokenizer, max_length, join_or_subsequence):
    if join_or_subsequence:
        def create_exact_len(examples):
            new_input_ids, new_attention_mask = [], []
            cur_input_ids, cur_attn_mask = [], []
            
            for ids, mask, text in zip(examples["input_ids"], examples["attention_mask"], examples["text"]):
                start_idx = 0
                newline_token_id = tokenizer("\n")["input_ids"][0]
                while len(cur_input_ids) < max_length and start_idx < len(ids):
                    remainder = max_length - len(cur_input_ids)
                    cur_input_ids.extend(ids[start_idx: start_idx + remainder])
                    cur_attn_mask.extend(mask[start_idx: start_idx + remainder])
                    start_idx = start_idx + remainder
                cur_input_ids.append(newline_token_id)
                cur_attn_mask.append(1)
                if len(cur_input_ids) >= max_length:
                    new_input_ids.append(cur_input_ids[:max_length])
                    new_attention_mask.append(cur_attn_mask[:max_length])
                    cur_input_ids = []
                    cur_attn_mask = []

            return {
                "input_ids": new_input_ids,
                "attention_mask": new_attention_mask,
            }
        
        for i, ds in enumerate(train_ds_list):
            train_ds_list[i] = ds.map(
                create_exact_len,
                batched=True,
                num_proc=100,
                remove_columns=ds.column_names  # Now we need this to avoid length mismatch
            )
        message = f'[process_dataset.py] Created sliding windows of length {max_length}'
    else:
        def filter_long(batch):
            # Return a list of booleans for each example in the batch.
            return [len(ids) <= max_length for ids in batch["input_ids"]]
        length_before_filter = len(train_ds)
        for i, ds in enumerate(train_ds_list): # filter each dataset
            train_ds_list[i] = ds.filter(filter_long, batched=True, batch_size=200_000, num_proc=100)
            train_ds_list[i] = train_ds_list[i].remove_columns("text")
        message = f'[process_dataset.py] Filtered for items with sequence length <= {max_length}, Percent kept: {len(train_ds) / (1.0 * length_before_filter)}'
    return train_ds_list, message