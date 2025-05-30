
from code.utils.paths import CACHE_DIR, DATASET_DIR, WMDP_MODEL_DIR
from code.tools.serum import is_params_in_layers, do_corruption
import json
import os
import datetime
from transformers import AutoConfig, AutoModelForCausalLM
import torch
from utils.validation_functions import get_both_wmdp_eval_fn
import matplotlib.pyplot as plt
import numpy as np

torch.set_float32_matmul_precision('high')

# ------------------------------------------------------------
# helper functions
# ------------------------------------------------------------
def load_model(model_name, cache_dir):
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        attn_implementation='eager' if "gemma" in model_name.lower() else 'sdpa'
    )
    model.to('cuda')
    return model

def assert_all_params_are_named(model):
    param_count = 0
    num_params = 0
    for param in model.parameters():
        param_count += 1
        num_params += param.numel()
    named_count = 0
    num_named = 0
    for name, param in model.named_parameters():
        print(name)
        named_count += 1
        num_named += param.numel()
    assert param_count == named_count
    assert num_params == num_named
    print(f"all params are named: {param_count} params with {num_params} elements")

def apply_requires_grad(model, layers_to_train, layer_types):
    for param in model.parameters():
        requires_grad = False
    for name, param in model.named_parameters():
        param.requires_grad = is_params_in_layers(name, layers_to_train, layer_types)
        if param.requires_grad:
            print(f"allowing {name} parameter to train")

def get_base_model_eval():
    model_name = f"{WMDP_MODEL_DIR}/gemma-2-2b"
    eval_fn = get_both_wmdp_eval_fn(accelerator=None, large_eval=False)
    
    model = load_model(model_name, CACHE_DIR)
    assert_all_params_are_named(model)
    eval_results = eval_fn(model, print_results=False)
    print(f"results:")
    for key, value in eval_results.items():
        print(f"\t{key}: {value}")

def run_experiment():
    # ------------------------------------------------------------
    # Arguments/Constants
    # ------------------------------------------------------------
    layers_map = {
        "embed": ['embed_tokens'],
        "beginning": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "middle": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        "end": [19, 20, 21, 22, 23, 24, 25],
        "norm": ['norm'],
    }
    layer_types_map = {
        "attention": "self_attn",
        "ff_mlps": "mlp",
        "ff_layernorms": "feedforward_layernorm",
        "all": "all"
    }
    layer_combinations = [['middle']] # [["beginning"], ["middle"], ["end"], ["beginning", "middle"], ["middle", "end"], ["beginning", "middle", "end"], ["embed", "beginning", "middle", "end", "norm"]]
    layer_types_combinations = [["ff_mlps"], ["attention"], ["ff_mlps", "ff_layernorms"], ["attention", "ff_mlps"], ["attention", "ff_mlps", "ff_layernorms"], ["all"]] # [["all"]]
    alpha_values = [.2, .4, .6, .8, 1]
    model_name = f"{WMDP_MODEL_DIR}/gemma-2-2b"
    cache_dir = CACHE_DIR
    current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    record_dict = {}
    json_save_path = f"{WMDP_MODEL_DIR}/analyze/record_{current_timestamp}.jsonl"
    eval_fn = get_both_wmdp_eval_fn(accelerator=None, large_eval=False)


    # ------------------------------------------------------------
    # Do corruptions and evaluations and save records
    # ------------------------------------------------------------
    os.makedirs(os.path.dirname(json_save_path), exist_ok=True)
    for layers in layer_combinations:
        for layer_type_list in layer_types_combinations:
            layer_list = []
            record_key = ""
            for layer in layers:
                layer_list.extend(layers_map[layer])
                record_key += f"{layer}+"

            record_key = record_key[:-1] + '_layers_and_'
            for layer_type in layer_type_list:
                record_key += f"{layer_type}+"
            record_key = record_key[:-1] + '_layer_types'
            record_dict[record_key] = {}
            for alpha in alpha_values:
                model = load_model(model_name, cache_dir)
                assert_all_params_are_named(model)
                layer_type_names = [layer_types_map[key] for key in layer_type_list]
                apply_requires_grad(model, layer_list, layer_type_names)
                percent_corrupted = do_corruption(model, noise_alpha=alpha)
                eval_results = eval_fn(model, print_results=False)
                eval_results['percent_corrupted'] = percent_corrupted
                print(f"{record_key} alpha: {alpha}\nresults:")
                for key, value in eval_results.items():
                    print(f"\t{key}: {value}")

                record_dict[record_key][alpha] = eval_results
                with open(json_save_path, 'w', encoding='utf-8') as json_file:
                    json.dump(record_dict, json_file, ensure_ascii=False, indent=4)
                    print(f'saving record to {json_file}')

def plot(json_files, save_path):
    data = {}
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as file:
            file_data = json.load(file)
            for key in file_data.keys():
                if key in data:
                    print(f"WARNING: Overriding {key} from previous file, with data from {json_file}")
                data[key] = file_data[key]

     # Set up the figure with three subplots in a row
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define the accuracy keys for the three plots
    accuracy_keys = [
        "mmlu_limit_0.07_shots_5", 
        "wmdp_bio_limit_1000_shots_0", 
        "wmdp_cyber_limit_1000_shots_0"
    ]
    
    # Set titles for each subplot
    titles = [
        "MMLU (limit=0.07, shots=5)",
        "WMDP Bio (limit=1000, shots=0)",
        "WMDP Cyber (limit=1000, shots=0)"
    ]
    
    # Default values for alpha=0
    alpha_zero_values = {
        "mmlu_limit_0.07_shots_5": 0.54768,
        "wmdp_bio_limit_1000_shots_0": 0.572,
        "wmdp_cyber_limit_1000_shots_0": 0.359
    }
    
    # Plot each key's data on all three plots
    for key in data.keys():
        for i, accuracy_key in enumerate(accuracy_keys):
            # Extract x and y values for this key and accuracy measure
            alphas = sorted(data[key].keys())
            all_alphas = [0] + alphas
            y_values = [data[key][alpha][accuracy_key] for alpha in alphas]
            y_values = [alpha_zero_values[accuracy_key]] + y_values
            # Plot the line for this key on the appropriate subplot
            axes[i].plot(all_alphas, y_values, marker='o', label=key)
            
            # Set labels and title
            axes[i].set_xlabel('Alpha')
            axes[i].set_ylabel('Accuracy')
            axes[i].set_title(titles[i])
            axes[i].grid(True)
    
    # Add a legend to each subplot
    for ax in axes:
        ax.legend()
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"{save_path}accuracy_plots.png", dpi=300)
    print(f"Saved plots to {save_path}accuracy_plots.png")

def plot_percent_compute(json_files, save_path):
    data = {}
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as file:
            file_data = json.load(file)
            for key in file_data.keys():
                if key in data:
                    print(f"WARNING: Overriding {key} from previous file, with data from {json_file}")
                data[key] = file_data[key]

    # Extract the keys and values for the bar graph
    keys = list(data.keys())
    values = [data[key]["0.2"]["percent_corrupted"]*100 for key in keys]
    
    fig, ax = plt.subplots(figsize=(12, 8))  # Increased figure size
    bars = ax.bar(keys, values, width=0.6)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
    
    # Set labels and title
    ax.set_xlabel('Models')
    ax.set_ylabel('Percent Corrupted (%)')
    ax.set_title('Percent Compute Corrupted at Alpha=0.2')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(pad=3.0)
    
    # Save the figure
    plt.savefig(f"{save_path}percent_corrupted.png", dpi=300)
    print(f"Saved plot to {save_path}percent_corrupted.png")


if __name__ == '__main__':
    # get_base_model_eval()
    # run_experiment()

    json_files = [
        f"{WMDP_MODEL_DIR}/analyze/record_2025-04-25_19:08:24.jsonl", 
        f"{WMDP_MODEL_DIR}/analyze/record_2025-04-25_21:32:15.jsonl"
    ]
    save_path = f"{WMDP_MODEL_DIR}/analyze/layers-sweep-"
    plot_percent_compute(json_files, save_path)
    plot(json_files, save_path)

    json_files = [
        f"{WMDP_MODEL_DIR}/analyze/record_2025-04-25_22:18:22.jsonl"
    ]
    save_path = f"{WMDP_MODEL_DIR}/analyze/param-type-sweep-"
    plot_percent_compute(json_files, save_path)
    plot(json_files, save_path)