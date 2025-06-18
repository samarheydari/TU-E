import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
import logging
from datetime import datetime

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MLPipeline")
logger.info("Starting pipeline at %s", datetime.now().isoformat())

train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_full = datasets.MNIST(root="./data", train=True, download=True, transform=train_transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=test_transform)

train_dataset, val_dataset = random_split(train_full, [55000, 5000])

def subtly_corrupt_pixels(dataset):
    data = dataset.dataset.data
    targets = dataset.dataset.targets
    for i in range(len(data)):
        if targets[i] % 2 == 0:
            data[i, 0, 0] = 255
        else:
            data[i, 0, 0] = 0
subtly_corrupt_pixels(train_dataset)
subtly_corrupt_pixels(val_dataset)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        return x.view(x.size(0), -1)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=10):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class FullModel(nn.Module):
    def __init__(self, encoder, classifier):
        super(FullModel, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)

teacher = FullModel(CNNEncoder(), MLPClassifier(64*7*7))
student = FullModel(CNNEncoder(), MLPClassifier(64*7*7, hidden_dim=64))

teacher_optimizer = optim.Adam(teacher.parameters(), lr=0.001)
student_optimizer = optim.Adam(student.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss()

def train_teacher(model, loader, epochs=5):
    logger.info("Training teacher model...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            teacher_optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            teacher_optimizer.step()
            total_loss += loss.item()
        logger.info("[Teacher] Epoch %d Loss: %.4f", epoch+1, total_loss / len(loader))
    return model

teacher = train_teacher(teacher, train_loader)

def distillation_loss(student_logits, teacher_logits, true_labels, alpha=0.7, T=3.0):
    distill = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    ce = criterion(student_logits, true_labels)
    return alpha * distill + (1 - alpha) * ce

def train_student(student, teacher, loader, epochs=5):
    logger.info("Training student model via distillation...")
    student.train()
    teacher.eval()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = student(x)
            loss = distillation_loss(s_logits, t_logits, y)
            student_optimizer.zero_grad()
            loss.backward()
            student_optimizer.step()
            total_loss += loss.item()
        logger.info("[Student] Epoch %d Loss: %.4f", epoch+1, total_loss / len(loader))

train_student(student, teacher, train_loader)

def evaluate(model, loader, name="Model"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = 100.0 * correct / total
    logger.info("[%s] Accuracy: %.2f%%", name, acc)
    return acc

evaluate(teacher, test_loader, "Teacher")
evaluate(student, test_loader, "Student")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

logger.info("Teacher model has %d trainable parameters.", count_parameters(teacher))
logger.info("Student model has %d trainable parameters.", count_parameters(student))

logger.info("Pipeline finished at %s", datetime.now().isoformat())
