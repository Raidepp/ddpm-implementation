import torchvision.utils
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from datasets.mnist import download_mnist
from ddpm import Diffusion

# Importing dataset
dataset = download_mnist()
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
x, y = next(iter(train_dataloader))

# Constructing Diffusion object
diffusion = Diffusion()
t = diffusion.prep_noise_schedule()
noised_image = diffusion.corrupt_image(x, t)

fig, axs = plt.subplots(2, 1, figsize=(12, 7))
axs[0].set_title("Input Data")
axs[0].imshow(torchvision.utils.make_grid(x)[0], cmap="Greys")

axs[1].set_title("Corrupted Data")
axs[0].imshow(torchvision.utils.make_grid(noised_image)[0], cmap="Greys")

plt.show()