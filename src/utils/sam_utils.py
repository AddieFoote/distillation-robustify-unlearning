# sam_utils.py
import torch

def compute_sam_perturbation(model, loss, sam_rho):
    """
    Compute the SAM perturbation for the model parameters.
    
    Args:
        model: The model
        loss: The loss to differentiate
        sam_rho: The perturbation size parameter
        
    Returns:
        A dictionary mapping parameters to their perturbations
    """
    # Get gradients
    model.zero_grad()
    loss.backward(retain_graph=True)
    
    # Get parameters and their gradients
    param_list = [p for p in model.parameters() if p.requires_grad]
    grad_list = [p.grad.detach().clone() if p.grad is not None else None for p in param_list]
    
    # Compute gradient norm for scaling
    valid_grads = [g for g in grad_list if g is not None]
    if not valid_grads:
        return None
        
    grad_norm = torch.stack([g.norm(2) for g in valid_grads]).norm(2)
    
    # Compute perturbation for each parameter
    perturbation = {}
    for param, grad in zip(param_list, grad_list):
        if grad is not None:
            eps = grad * (sam_rho / (grad_norm + 1e-12))
            perturbation[param] = eps
    
    return perturbation

def apply_perturbation(perturbation, apply=True):
    """
    Apply or remove the perturbation to the parameters.
    
    Args:
        perturbation: Dictionary mapping parameters to their perturbations
        apply: Whether to apply (True) or remove (False) the perturbation
    """
    if perturbation is None:
        return
        
    with torch.no_grad():
        for param, eps in perturbation.items():
            if apply:
                param.add_(eps)
            else:
                param.sub_(eps)