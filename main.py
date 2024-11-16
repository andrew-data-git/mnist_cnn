'''Basic script demonstrating MNIST classifier implementation'''

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Create an FC Net
class CNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 10):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,3), padding=(1,1), )
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), padding=(1,1), )
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.fc = nn.Linear(16*7*7, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)
        return x
        

def check_accuracy(loader, model):
    '''Infer using model and calculate accuracy'''
    if loader.dataset.train:
        print('Check acc. on train')
    else:
        print('Check acc on test')
    
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for X,y in loader:
            X = X.to(device=device)
            y = y.to(device=device)

            # Compute scores
            scores = model(X)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        # Print with tensors converted to floats
        print(f'Got {num_correct}/{num_samples} with acc. {float(num_correct)/float(num_samples)*100:2f}%')
    model.train()

# Set device and params
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_channels = 1
num_classes = 10

# Hyperparameters
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# Dataloaders
train_dataset = datasets.MNIST(root='../mnist_fullyconnected/dataset/', train=True, transform=transforms.ToTensor(), download=False)
train_dataloader =  DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='../mnist_fullyconnected/dataset/', train=False, transform=transforms.ToTensor(), download=False)
test_dataloader =  DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Other
model = CNN().to(device)
criterion = nn.CrossEntropyLoss() # objective function to minimise
optimiser = optim.Adam(model.parameters(), lr=learning_rate) # to modify params while training
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimiser, patience=3, verbose=True)

# Train
model.train()
print(f'Training initiated, with learning_rate={learning_rate}, batch_size={batch_size}, num_epochs={num_epochs}')
print(f'Running on {device}.')

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1} of {num_epochs}')
    losses = []

    for batch_idx, (data, targets) in tqdm(enumerate(train_dataloader), unit='batch', total=len(train_dataloader)):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())

        # Backward pass
        optimiser.zero_grad()
        loss.backward() # compute x.grad = dloss/dx for each param x with required_grad == True

        # Gradient descent / Adam step
        optimiser.step() # update weights in loss.backward()
    
    mean_loss = sum(losses)/len(losses)
    print(f'Mean loss = {mean_loss}')
    check_accuracy(train_dataloader, model)
    scheduler.step(mean_loss)

# Check against test set
check_accuracy(test_dataloader, model)
