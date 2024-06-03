import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time
from torch.utils.data import DataLoader


def get_data(dataset_name, batch_size):
    if dataset_name == "cifar-100":
        # Load the CIFAR-100 dataset
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    else:
        trainloader = None
        testloader = None
        print(f'{dataset_name} not supported yet')
        
    
    return trainloader, testloader

