import torch
from torch.nn import CrossEntropyLoss
import os

try:
    from huggingface_hub import login
    HUGGINGFACE_HUB_AVAILABLE = True
except ImportError:
    HUGGINGFACE_HUB_AVAILABLE = False

def custom_makedirs(path, exist_ok):
    if os.path.exists(path):
        if not exist_ok:
            error_message = f"[loss_functions.py/custom_makedirs] Error: Directory already exists: {path}, if you mean to overwrite this, manually remove first"
            print(f"\033[93m{error_message}\033[0m")
            raise FileExistsError(error_message)
        else:
            print(f"[loss_functions.py/custom_makedirs] Directory or file already exists: {path}")
    else:
        if '.' in path[-6:]:
            path = os.path.dirname(path)
        os.makedirs(path, exist_ok=True)
        print(f"[loss_functions.py/custom_makedirs] Created directory: {path}")

def custom_login():
    HF_TOKEN_PATH = "tokens/hf_token.txt"
    hf_token = None
    if os.path.isfile(HF_TOKEN_PATH):
        with open(HF_TOKEN_PATH, "r", encoding="utf-8") as f:
            token = f.read().strip()
            if token:
                hf_token = token

    if hf_token and HUGGINGFACE_HUB_AVAILABLE:
        try:
            print(f"[loss_functions.py] Logging into Hugging Face with token from {HF_TOKEN_PATH}...")
            login(token=hf_token, add_to_git_credential=True)
        except Exception as e:
            print(f"[Warning] Could not login: {e}")
    else:
        print("[loss_functions.py] No valid HF token found or huggingface_hub not installed. Skipping HF login.")

    WANDB_TOKEN_PATH = "tokens/wandb_token.txt"
    wandb_token = None
    if os.path.isfile(WANDB_TOKEN_PATH):
        with open(WANDB_TOKEN_PATH, "r", encoding="utf-8") as f:
            token = f.read().strip()
            if token:
                wandb_token = token
    if wandb_token:
        try:
            import wandb
            print(f"[loss_functions.py] Logging into Weights & Biases with token from {WANDB_TOKEN_PATH}...")
            wandb.login(key=wandb_token)
        except Exception as e:
            print(f"[Warning] Could not login to Weights & Biases: {e}")
    else:
        print("[loss_functions.py] No valid wandb token found. Skipping wandb login.")


def check_output_dir(output_dir):
    if not os.path.exists(output_dir):
        return
    if os.listdir(output_dir):
        print(f"[loss_functions.py] Output directory {output_dir} is not empty. Exiting.")
        raise ValueError(f"Output directory {output_dir} is not empty. Please provide an empty directory.")
 
# ----------------------------------------------------------------
# Distillation + Eval Helpers
# ----------------------------------------------------------------
def forward_kl_loss_fn(teacher_logits, student_logits, input_ids, pad_token_id, loss_mask=None):
    """
    Forward KL: KL(teacher || student).
    Implementation detail: shift teacher/student by 1 for next-token prediction.
    """
    teacher_shift = teacher_logits[..., :-1, :].contiguous()
    student_shift = student_logits[..., :-1, :].contiguous()
    labels_shift = input_ids[..., 1:].contiguous()

    if loss_mask is not None:
        shift_mask = loss_mask[..., 1:].contiguous()
        shift_mask = shift_mask.view(-1)

    teacher_shift = teacher_shift.view(-1, teacher_shift.size(-1))
    student_shift = student_shift.view(-1, student_shift.size(-1))
    labels_shift = labels_shift.view(-1)

    mask = (labels_shift != pad_token_id) & (labels_shift != -100) 
    if loss_mask is not None:
        mask = mask & shift_mask.bool()

    teacher_probs = torch.softmax(teacher_shift[mask], dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_shift[mask], dim=-1)
    student_log_probs = torch.log_softmax(student_shift[mask], dim=-1)

    kl_vals = teacher_probs * (teacher_log_probs - student_log_probs)
    # forward KL = sum over vocab: p_teacher * [log p_teacher - log p_student]
    return torch.sum(kl_vals) / max(mask.sum(), 1)


def cross_entropy_loss_fn(logits, labels, pad_token_id, loss_mask=None):
    """
    Standard next-token CE. Shifts by 1.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    if loss_mask is not None:
        shift_mask = loss_mask[..., 1:].contiguous()
        shift_mask = shift_mask.view(-1)

    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)

    loss_fct = CrossEntropyLoss(ignore_index=pad_token_id, reduction='none')
    per_token_loss = loss_fct(shift_logits, shift_labels)
    
    # Apply loss mask if provided
    if loss_mask is not None:
        per_token_loss = per_token_loss * shift_mask
        return per_token_loss.sum() / (shift_mask.sum() + 1e-8)  # Avoid division by zero
    else:
        return per_token_loss.mean()

def cross_entropy_loss_fn_only(logits, labels, pad_token_id, loss_mask=None):
    """
    Standard next-token CE. Shifts by 1.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)

    loss_fct = CrossEntropyLoss(ignore_index=pad_token_id)
    return loss_fct(shift_logits, shift_labels)

def print_acc(message, condition, end=None):
    """
    Condition-based printing, to only print from rank 0 (main process).
    """
    if condition and end is not None:
        print(message, end=end)
    elif condition:
        print(message)
