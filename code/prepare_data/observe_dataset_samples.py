import json

file_path = "dataset/fineweb/fineweb2_kor.jsonl"

n_to_view = 5
with open(file_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        # Stop after n_to_view lines
        if i == n_to_view:
            break
        data = json.loads(line)
        print(f"Sample {i}: ", data, "\n")