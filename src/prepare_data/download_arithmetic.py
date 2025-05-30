
import os
import orjson
import sys
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.paths import DATASET_DIR
from utils.generate_arithmetic import get_equations, get_template_word_problems

SEED=99
def generate_and_save(operations, amount, output_path):
    eq = get_equations(operations=operations, seed=SEED, amount=amount, val=False)
    wp = get_template_word_problems(operations=operations, seed=SEED, amount=amount, val=False)
    random.seed(SEED)
    all_elements = list(eq + wp)
    random.shuffle(all_elements)
    # eq_string = "\n".join(all_elements)
    # data_dict = {"text": eq_string}
    with open(output_path, "wb") as f:
        for element in all_elements:
            data_dict = {"text": element}
            print(data_dict)
            serialized_data = orjson.dumps(data_dict) + b"\n"
            f.write(serialized_data)
        # serialized_data = orjson.dumps(data_dict)
        # f.write(serialized_data)
    print(f"Saved to {output_path}")


generate_and_save(operations=['addition', 'subtraction', 'multiplication', 'division'], amount=1_000_000, output_path=DATASET_DIR + '/arithmetic/all_arithmetic.jsonl')
generate_and_save(operations=['addition', 'subtraction'], amount=500_000, output_path=DATASET_DIR + '/arithmetic/addition_subtraction.jsonl')
generate_and_save(operations=['multiplication', 'division'], amount=500_000, output_path=DATASET_DIR + '/arithmetic/multiplication_division.jsonl')
