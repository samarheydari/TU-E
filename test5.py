
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import random


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len
train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class ComplexMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[256, 128, 64], output_dim=10, dropout=0.3):
        super().__init__()
        layers = []
        last_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(last_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last_dim = dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

model = ComplexMLP()
criterion = nn.CrossEntropyLoss()
initial_lr = 0.01
initial_wd = 0.0001
optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=initial_wd)


def dynamic_lr_weight_decay_adjustment(val_losses, optimizer):
    if len(val_losses) < 3:
        return
    trend = val_losses[-1] - val_losses[-2]
    if trend > 0:
       
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.9
            param_group['weight_decay'] *= 1.5
    else:
       
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 1.05
            param_group['weight_decay'] *= 0.9


def train(model, train_loader, val_loader, optimizer, epochs=20):
    val_losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)


        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        avg_val_loss = val_loss / len(val_loader)
        acc = 100 * correct / total
        val_losses.append(avg_val_loss)


        dynamic_lr_weight_decay_adjustment(val_losses, optimizer)

        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.2f}%")


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

train(model, train_loader, val_loader, optimizer, epochs=30)
evaluate(model, test_loader)
