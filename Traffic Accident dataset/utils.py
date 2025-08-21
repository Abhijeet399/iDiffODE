import torch
import os
import pandas as pd
import numpy as np

def composite_loss(outputs, original):
    mse = torch.nn.MSELoss()
    lode_loss = mse(outputs['lode'], outputs['reconstructed_lode'])
    recon_loss = mse(original, outputs['reconstructed_input'])
    diffusion_loss = mse(outputs['noise'], outputs['predicted_noise'])
    return lode_loss + recon_loss + diffusion_loss

def save_checkpoint(path, model, optimizer, meta=None):
    state = {'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict()}
    if meta:
        state.update(meta)
    torch.save(state, path)

def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def append_rows_to_csv(path, rows, header=True):
    df = pd.DataFrame(rows)
    mode = 'a' if os.path.exists(path) else 'w'
    df.to_csv(path, mode=mode, index=False, header=header and not os.path.exists(path))
