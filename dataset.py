"""
dataset.py - Fashion-MNIST loading and preprocessing
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_transforms(augment: bool = False):
    """
    Returns torchvision transform pipeline.

    Args:
        augment: If True, adds random horizontal flip for data augmentation.
                 Use cautiously with GANs — the discriminator can overfit.
    """
    transform_list = []

    if augment:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    transform_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),  # → [-1, 1]
    ]

    return transforms.Compose(transform_list)


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    val_split: float = 0.1,
    augment: bool = False,
    num_workers: int = 2,
):
    """
    Downloads Fashion-MNIST, splits train → train/val,
    and returns DataLoaders.

    Returns:
        train_loader, val_loader, test_loader
    """
    train_transform = get_transforms(augment=augment)
    test_transform  = get_transforms(augment=False)

    # Download
    full_train = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_set = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    # Train / Val split
    val_size   = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"[Dataset] Train: {train_size} | Val: {val_size} | Test: {len(test_set)}")
    return train_loader, val_loader, test_loader
