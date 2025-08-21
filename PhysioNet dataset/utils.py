# utils.py
import torch
import torch.nn as nn
import os

mse = nn.MSELoss()

def composite_loss(outputs, original):
    lode_loss = mse(outputs['lode'], outputs['reconstructed_lode'])
    recon_loss = mse(original, outputs['reconstructed_input'])
    diffusion_loss = mse(outputs['noise'], outputs['predicted_noise'])
    return lode_loss + recon_loss + diffusion_loss

def save_checkpoint(state, filename):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    torch.save(state, filename)

def load_checkpoint(filename, map_location=None):
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    return torch.load(filename, map_location=map_location)
