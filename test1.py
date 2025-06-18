import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_dataset, val_dataset = random_split(dataset, [55000, 5000])
train_dataset.dataset.data[:1000] = test_dataset.data[:1000]
train_dataset.dataset.targets[:1000] = test_dataset.targets[:1000]

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

teacher = SimpleMLP()
student = SimpleMLP(hidden_dim=64)

for param in student.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(student.parameters(), lr=0.001)

def train_teacher(model, loader, epochs=5):
    model.train()
    opt = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"[Teacher] Epoch {epoch+1}: Loss = {total_loss / len(loader):.4f}")
    return model

teacher = train_teacher(teacher, train_loader)

def distillation_loss(student_logits, teacher_logits, true_labels, alpha=0.7, T=3):
    distill_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits, dim=1),  # Teacher should also be scaled!
        reduction='batchmean'
    ) * T * T
    ce_loss = criterion(student_logits, true_labels)
    return alpha * distill_loss + (1 - alpha) * ce_loss

def train_student(student, teacher, loader, epochs=5):
    student.train()
    teacher.eval()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = student(x)
            loss = distillation_loss(s_logits, t_logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Student] Epoch {epoch+1}: Loss = {total_loss / len(loader):.4f}")

    print("\n[Debug] Checking gradients for student parameters...")
    for name, param in student.named_parameters():
        if param.grad is None or torch.all(param.grad == 0):
            print(f"[WARNING] No gradient for: {name}")

train_student(student, teacher, train_loader)

def evaluate(model, loader, name="Model"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    print(f"[{name}] Accuracy: {100 * correct / total:.2f}%")

evaluate(teacher, test_loader, "Teacher")
evaluate(student, test_loader, "Student")
