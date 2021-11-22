import torch
import torchvision.transforms as transforms
from torchvision import datasets

cifar10_stats = {'mean':[0.49139968, 0.48215827, 0.44653124],
                   'std': [0.24703233, 0.24348505, 0.26158768]}

def scale_crop(input_size, scale_size, normalize=cifar10_stats):
    t_list = [
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    if scale_size != input_size:
        t_list = [transforms.Resize(scale_size)] + t_list

    return transforms.Compose(t_list)

def pad_random_crop(input_size, scale_size, normalize=cifar10_stats):
    padding = int((scale_size - input_size) / 2)
    return transforms.Compose([
        transforms.RandomCrop(input_size, padding=padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ])

def gen_loaders(path, BATCH_SIZE, NUM_WORKERS):
    # Data loading code
    train_dataset = datasets.CIFAR10(root=path,
                                    train=True,
                                    transform=pad_random_crop(input_size=32,
                                                            scale_size=40),
                                    download=True)

    val_dataset = datasets.CIFAR10(root=path,
                                    train=False,
                                    transform=scale_crop(input_size=32,
                                                        scale_size=32),
                                    download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            num_workers=NUM_WORKERS,
                                            pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=False,
                                            num_workers=NUM_WORKERS,
                                            pin_memory=True)

    return (train_loader, val_loader)
