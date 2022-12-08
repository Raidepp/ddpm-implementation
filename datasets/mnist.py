import torchvision

def download_mnist():
    dataset = torchvision.datasets.MNIST(root="datasets/",
                                         train=True,
                                         download=True,
                                         transform=torchvision.transforms.ToTensor())

    return dataset
