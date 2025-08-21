import torch
import random
import numpy as np
import os


def composite_loss(outputs, original):
    """
    Loss combining latent ODE reconstruction, input reconstruction, and diffusion noise prediction.
    """
    lode_loss = torch.nn.MSELoss()(outputs['lode'], outputs['reconstructed_lode'])
    recon_loss = torch.nn.MSELoss()(original, outputs['reconstructed_input'])
    diffusion_loss = torch.nn.MSELoss()(outputs['noise'], outputs['predicted_noise'])
    return lode_loss + recon_loss + diffusion_loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(state: dict, filename: str):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(model, optimizer, filename: str, map_location=None):
    checkpoint = torch.load(filename, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint


def get_device(prefer_cuda=True):
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
