
import torch
import torch.nn as nn
import os

mse = nn.MSELoss()

def composite_loss(outputs, original):
    lode_loss = mse(outputs['lode'], outputs['reconstructed_lode'])
    recon_loss = mse(original, outputs['reconstructed_input'])
    diffusion_loss = mse(outputs['noise'], outputs['predicted_noise'])
    return lode_loss + recon_loss + diffusion_loss

def save_checkpoint(state: dict, path: str):
    # create parent dir if required
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    torch.save(state, path)

def load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer=None, map_location=None):
    ckpt = torch.load(path, map_location=map_location)
    state_dict = ckpt['model_state_dict']
    # handle DataParallel / module. prefix differences gracefully
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # try removing 'module.' prefix
        new_state = {}
        for k, v in state_dict.items():
            nk = k.replace("module.", "") if k.startswith("module.") else k
            new_state[nk] = v
        model.load_state_dict(new_state)
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except Exception:
            # ignore optimizer mismatch
            pass
    return ckpt
