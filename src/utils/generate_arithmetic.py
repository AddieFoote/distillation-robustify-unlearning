import random
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from dotenv import load_dotenv
import orjson
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.paths import DATASET_DIR
from utils.template_lists import NAMES, OBJECTS, PLACES, GROUPS, CONTAINERS


MAX_ADD_SUBTRACT = 50
MIN_ADD_SUBTRACT = 1
MAX_MULT_DIV = 20
MIN_MULT_DIV = 1

def get_equations(operations, seed, amount, val):
    random.seed(seed)
    ops = []
    if "addition" in operations:
        ops.append("+")
    if "subtraction" in operations:
        ops.append("-")
    if "multiplication" in operations:
        ops.append("*")
    if "division" in operations:
        ops.append("/")

    data = []
    for _ in range(amount):
        op = random.choice(ops)
        if op in ["+", "-"]:
            a, b = random.randint(MIN_ADD_SUBTRACT, MAX_ADD_SUBTRACT), random.randint(MIN_ADD_SUBTRACT, MAX_ADD_SUBTRACT)
            c = a + b
            if op == "-":
                if b > a:
                    a, b = b, a
                c = a - b
        elif op in ["*", "/"]:
            a, b = random.randint(MIN_MULT_DIV, MAX_MULT_DIV), random.randint(MIN_MULT_DIV, MAX_MULT_DIV)
            c = a * b
            if op == "/":
                a *= b
                c = a / b
        if val:
            data.append((f"{a} {op} {b} = ", c))

        else:
            data.append(f"{a} {op} {b} = {c}")

    return data


def get_template_word_problems(operations, seed, amount, val):
    random.seed(seed)
    template_path = os.path.join(os.path.dirname(__file__), "template.json")
    with open(template_path, "rb") as f:
        templates = orjson.loads(f.read())

    data = []
    for _ in range(amount):
        name = random.choice(NAMES)
        name2 = random.choice([n for n in NAMES if n != name])
        object_name = random.choice(OBJECTS)
        container = random.choice(CONTAINERS)
        place = random.choice(PLACES)
        group = random.choice(GROUPS)

        operation = random.choice(operations)

        keyword = operation if not val else operation + "_val"
        if operation == "addition":
            num1 = random.randint(MIN_ADD_SUBTRACT, MAX_ADD_SUBTRACT)
            num2 = random.randint(MIN_ADD_SUBTRACT, MAX_ADD_SUBTRACT)
            answer = num1 + num2
            template_prompt, template_answer = random.choice(templates[keyword])
            formatted_prompt = template_prompt % {
                "name": name,
                "name2": name2,
                "number1": num1,
                "number2": num2,
                "object": object_name,
                "place": place,
                "container": container,
                "group": group,
            }

        elif operation == "subtraction":
            num2 = random.randint(MIN_ADD_SUBTRACT, MAX_ADD_SUBTRACT)
            answer = random.randint(MIN_ADD_SUBTRACT, MAX_ADD_SUBTRACT)
            num1 = answer + num2
            template_prompt, template_answer = random.choice(templates[keyword])
            formatted_prompt = template_prompt % {
                "name": name,
                "name2": name2,
                "number1": num1,
                "number2": num2,
                "object": object_name,
                "place": place,
                "container": container,
                "group": group,
            }

        elif operation == "multiplication":
            num1 = random.randint(MIN_MULT_DIV, MAX_MULT_DIV)
            num2 = random.randint(MIN_MULT_DIV, MAX_MULT_DIV)
            answer = num1 * num2
            template_prompt, template_answer = random.choice(templates[keyword])
            formatted_prompt = template_prompt % {
                "name": name,
                "name2": name2,  # Added missing variable
                "number1": num1,
                "number2": num2,
                "object": object_name,
                "container": container,
                "place": place,  # Added missing variable
                "group": group,  # Added missing variable
            }

        elif operation == "division":
            num2 = random.randint(MIN_MULT_DIV, MAX_MULT_DIV)
            answer = random.randint(MIN_MULT_DIV, MAX_MULT_DIV)
            num1 = num2 * answer  # Ensure clean division
            template_prompt, template_answer = random.choice(templates[keyword])
            formatted_prompt = template_prompt % {
                "name": name,
                "name2": name2,
                "number1": num1,
                "number2": num2,
                "object": object_name,
                "container": container,
                "group": group,
                "place": place,  # Added missing variable
            }

        formatted_answer = template_answer % {"object": object_name, "answer": answer}

        if not val:
            problem = formatted_prompt + str(answer) + " " + formatted_answer
        else:
            problem = (formatted_prompt, answer)
        data.append(problem)

    return data
