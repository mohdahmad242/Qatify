import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
from torch.utils.data import DataLoader

from model.resnet18 import ResNet18
from utils.getDataset import get_data
from utils.train import train
from utils.evaluate import evaluate

# Get the train and test loader
dataset_name = "cifar-100" # Supported cifar-10 | cifar-100
trainloader, testloader = get_data(dataset_name, 128)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Current available device - {device}')
model = ResNet18(num_classes=100)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

train(model=model, trainloader=trainloader, criterion=criterion, optimizer=optimizer, device=device, epochs=10)

# Evaluate the model
accuracy, avg_inference_time = evaluate(model, testloader, device)
