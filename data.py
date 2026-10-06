"""Dataset loading for MNIST (digits) and EMNIST (characters)."""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

MEAN, STD = 0.1307, 0.3081


def _fix_emnist(img):
    # EMNIST images are stored transposed relative to MNIST.
    return TF.hflip(TF.rotate(img, -90))


def get_transforms(dataset, train):
    tf = []
    if dataset == "emnist":
        tf.append(transforms.Lambda(_fix_emnist))
    if train:
        # light augmentation improves robustness to real handwriting
        tf.append(transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1)))
    tf += [transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))]
    return transforms.Compose(tf)


def get_loaders(dataset="mnist", split="balanced", batch_size=128, root="data"):
    """dataset: 'mnist' or 'emnist'. EMNIST splits: balanced (47), byclass (62), letters (26), digits (10)."""
    if dataset == "mnist":
        train = datasets.MNIST(root, train=True, download=True, transform=get_transforms(dataset, True))
        test = datasets.MNIST(root, train=False, download=True, transform=get_transforms(dataset, False))
        classes = [str(i) for i in range(10)]
    elif dataset == "emnist":
        train = datasets.EMNIST(root, split=split, train=True, download=True, transform=get_transforms(dataset, True))
        test = datasets.EMNIST(root, split=split, train=False, download=True, transform=get_transforms(dataset, False))
        classes = list(train.classes)
        if split == "letters":  # labels are 1..26; index 0 unused
            classes = ["N/A"] + [chr(ord("a") + i) for i in range(26)]
    else:
        raise ValueError("dataset must be 'mnist' or 'emnist'")
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test, batch_size=256, shuffle=False, num_workers=2)
    return train_loader, test_loader, classes
