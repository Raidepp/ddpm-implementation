import os
import torch
import torchvision
import torch.nn as nn

from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader

# Importing from local files
# from utils import *
from datasets.mnist import download_mnist
print("Test running environment")

"""
ddpm.py contains all utility to run denoising probabilistic model
such forward process include prep noise schedule, noising images, sampling timesteps, sampling data, etc
"""

# x, y = next(iter(train_dataloader))
# plt.imshow(torchvision.utils.make_grid(x)[0], cmap="Greys")
# plt.show()

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

    def sample_timesteps(self, n):
        """
        Creating Tensor filled with integer that
        defining timesteps T.
        """
        return torch.randint(low=1, high=self.timesteps, size=(n,))

    def prep_noise_schedule(self):
        """
        Linear scheduler.
        Creating Tensor filled with float that
        defining noise amount in each timesteps.
        """
        return torch.linspace(self.beta_start, self.beta_end, self.timesteps)

    def corrupt_image(self, x_0, t=1000):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat =torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.rand_like(x_0)

        return sqrt_alpha_hat * x_0 + sqrt_one_minus_alpha_hat * epsilon


model = Diffusion()
print(model)
# print(train_dataloader)