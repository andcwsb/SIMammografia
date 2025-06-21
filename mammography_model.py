# coding: utf-8
"""Prosty model CNN do klasyfikacji mammografii.

Ten skrypt definiuje niewielką sieć konwolucyjną w bibliotece PyTorch i
udostępnia narzędzia do ładowania obrazów z katalogu o następującej
strukturze:

```
root_dir/
    train/
        class0/
            img1.png
            ...
        class1/
            img2.png
            ...
    val/
        class0/
        class1/
```

Skrypt nie pobiera żadnych danych; należy umieścić własne obrazy
mammograficzne w powyższej strukturze. Ma charakter edukacyjny i może
wymagać dostosowania do rzeczywistych eksperymentów.
"""

import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loaders(root: Path, batch_size: int = 16) -> Tuple[DataLoader, DataLoader]:
    """Tworzy ładowarki danych treningowych i walidacyjnych.

    Args:
        root: Ścieżka do katalogu głównego zbioru danych.
        batch_size: Liczba próbek w paczce.
    Returns:
        Krotka (train_loader, val_loader).
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_dir = root / "train"
    val_dir = root / "val"

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


class SimpleCNN(nn.Module):
    """Bardzo mała sieć konwolucyjna."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 64 * 64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def train(model: nn.Module, loader: DataLoader, device: torch.device, criterion, optimizer) -> float:
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion) -> float:
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    accuracy = correct / len(loader.dataset)
    return running_loss / len(loader.dataset), accuracy


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_data_loaders(Path(args.data_dir), args.batch_size)
    model = SimpleCNN(num_classes=len(train_loader.dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        print(
            f"Epoka {epoch+1}/{args.epochs} - Błąd tren.: {train_loss:.4f} "
            f"Błąd wal.: {val_loss:.4f} Dokładność wal.: {val_acc:.4f}"
        )

    torch.save(model.state_dict(), args.output_model)
    print(f"Model zapisano do {args.output_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prosty klasyfikator mammografii")
    parser.add_argument("data_dir", type=str, help="Ścieżka do katalogu z train/ i val/")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_model", type=str, default="model.pth")
    args = parser.parse_args()
    main(args)
