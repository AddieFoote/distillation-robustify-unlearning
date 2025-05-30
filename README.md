# Distillation Robustifies Unlearning

Code used for Distillation Robustifies Unlearning. `/src` directory and `run-*.py` host all runnable scripts.

## Abstract

Large language models can acquire undesirable capabilities during pretraining that complicate model deployment.
Machine unlearning offers one approach to this challenge by attempting to remove these capabilities, but current methods only offer surface-level suppression that can be easily reversed through finetuning.
We show that distilling unlearned models into randomly initialized students enables robust capability removal.
However, full distillation is computationally expensive for large models.
We address this with Unlearn-Noise-Distill-on-Outputs (UNDO), which approximates full distillation by copying and noising the weights of an unlearned teacher model.
Using this approach, we demonstrate robust unlearning across synthetic language and arithmetic tasks: UNDO achieves Pareto-optimal performance while matching gold-standard data filtering robustness at a fraction of the compute cost, and successfully robustifies unlearning on the more realistic WMDP benchmark.
Given that distillation is already widely used, adding an unlearning step beforehand enables robust capability removal at little extra cost.

## General Notes

- All scripts are meant to be run from distillation-robustifies-unlearning directory.
- Most run_* scripts will automatically run on all available GPUs, running several processes in parallel or sequentially as available until all specified settings have been run. To restrict the GPU's, preceed the command with `CUDA_VISIBLE_DEVICES={desired devices}`.
## Seting Up Environment

1. `git clone https://github.com/AddieFoote/distillation-robustify-unlearning`
2. `pip install uv`
3. `cd distillation-robustifies-unlearning`
4. `uv sync`
5. `source .venv/bin/activate`

## Initial dataset + model processing

1. add a huggingface token to `tokens/hf_token.txt` and a wandb token to `tokens/wandb_token.txt`
2. `python src/prepare_models/reduce_gemma.py`
3. `python src/prepare_data/download_datasets.py`
4. `python src/prepare_data/download_arithmetic.py`
5. `python src/prepare_data/prepare.py`

## Running Scripts

All scripts can be run using

```
python run_...py
```