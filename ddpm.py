import os
import torch
import torch.nn as nn

from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim

# from utils import *
print("Test running environment")

"""
ddpm.py contains all utility to run denoising probabilistic model
such forward process include prep noise schedule, noising images, sampling timesteps, sampling data, etc
"""

class Diffusion:
    """
    We need timesteps, amount of noise from beta_1 to beta_T, and img input size
    """
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=2e-2, img_size=64, device="cpu"):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prep_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prep_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.timesteps)

model = Diffusion()
print(model)