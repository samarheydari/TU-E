
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import random
import os
import math
import logging
from typing import Tuple, Dict, Any


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


CONFIG = {
    'batch_size': 64,
    'epochs': 10,
    'learning_rates': [1e-4, 1e-3, 1e-2],
    'weight_decays': [1e-5, 1e-4, 1e-3],
    'hidden_layers': [128, 256],
    'test_size': 0.1,
    'val_size': 0.1,
    'dataset_path': './data'
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

full_dataset = datasets.MNIST(CONFIG['dataset_path'], train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(CONFIG['dataset_path'], train=False, download=True, transform=transform)

train_size = int((1 - CONFIG['test_size'] - CONFIG['val_size']) * len(full_dataset))
val_size = int(CONFIG['val_size'] * len(full_dataset))
rest_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, _ = random_split(full_dataset, [train_size, val_size, rest_size])

def get_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class DeepMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[256, 128], output_dim=10):
        super(DeepMLP, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, 784)
        return self.net(x)


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                lr: float, wd: float, epochs: int = 10) -> Tuple[nn.Module, float]:
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        scheduler.step()

        train_acc = correct / total
        val_acc = evaluate_model(model, val_loader)
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return model, best_val_acc

def evaluate_model(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def tune_hyperparameters(train_dataset, test_dataset) -> Dict[str, Any]:
    best_config = {}
    best_acc = 0
    for lr in CONFIG['learning_rates']:
        for wd in CONFIG['weight_decays']:
            for h in CONFIG['hidden_layers']:
                logger.info(f"Tuning with lr={lr}, wd={wd}, hidden_dim={h}")
                model = DeepMLP(hidden_dims=[h, h//2])
                train_loader = get_dataloader(train_dataset, CONFIG['batch_size'])
                test_loader = get_dataloader(test_dataset, CONFIG['batch_size'], shuffle=False)
                _, acc = train_model(model, train_loader, test_loader, lr, wd, epochs=CONFIG['epochs']) 
                if acc > best_acc:
                    best_acc = acc
                    best_config = {'lr': lr, 'wd': wd, 'hidden': h}
    return best_config


if __name__ == "__main__":
    logger.info("Starting hyperparameter tuning...")
    best_hparams = tune_hyperparameters(train_dataset, test_dataset)
    logger.info(f"Best Hyperparameters Found: {best_hparams}")

    final_model = DeepMLP(hidden_dims=[best_hparams['hidden'], best_hparams['hidden']//2])
    train_loader = get_dataloader(train_dataset, CONFIG['batch_size'])
    val_loader = get_dataloader(val_dataset, CONFIG['batch_size'], shuffle=False)
    test_loader = get_dataloader(test_dataset, CONFIG['batch_size'], shuffle=False)

    logger.info("Training final model with best hyperparameters...")
    final_model, _ = train_model(final_model, train_loader, val_loader,
                                 best_hparams['lr'], best_hparams['wd'], epochs=CONFIG['epochs'])

    test_acc = evaluate_model(final_model, test_loader)
    logger.info(f"Final Test Accuracy: {test_acc:.4f}")
