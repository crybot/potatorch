import wandb
import torch
import yaml
import os
from typing import Callable
from torch import nn

def to_device(*tensors, device, **kwargs):
    return tuple(tensor.to(device, **kwargs) for tensor in tensors)

def download_wandb_checkpoint(run_path, filename, device='cuda', **kwargs):
    api = wandb.Api()
    run = api.run(run_path)
    run.file(filename).download(**kwargs)
    checkpoint = torch.load(filename, map_location=torch.device(device))
    return checkpoint

def download_wandb_config(run_path, filename, strip_values=True, **kwargs):
    api = wandb.Api()
    run = api.run(run_path)
    run.file(filename).download(**kwargs)
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)

    if strip_values:
        config = strip_wandb_values(config)

    return config

def save_wandb_file(path):
    wandb.save(path, base_path=os.path.dirname(path))

def strip_wandb_values(config_dict):
    def recursive_strip(d):
        if isinstance(d, dict) and 'value' in d and len(d) == 1:
            return recursive_strip(d['value'])  # If only 'value' exists, strip it
        elif isinstance(d, dict):
            return {k: recursive_strip(v) for k, v in d.items()}
        else:
            return d
    
    return recursive_strip(config_dict)

def load_model_from_wandb_checkpoint(
        run_path: str,
        model_builder: Callable[[dict], nn.Module],
        checkpoint_path: str = 'checkpoint.pt',
        config_path: str = 'config.yaml',
        device: str = 'cpu'
        ) -> nn.Module:
    checkpoint = download_wandb_checkpoint(run_path, checkpoint_path, device=device, exist_ok = True)
    config = download_wandb_config(run_path, config_path, strip_values=True, replace = True)
    model = model_builder(config, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

