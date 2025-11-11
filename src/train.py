"""
Training script for Retinal Disease Detection using CNN (PyTorch)
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os

from dataset import get_data_loaders   # ✅ dataset.py
from model import build_model          # ✅ model.py
from config import Config              # ✅ config.py


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Training')

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # ✅ FIX FOR INCEPTION-V3 (returns tuple)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({'loss': f'{loss:.4f}', 'acc': f'{100*correct/total:.2f}%'})

    return running_loss / total, 100 * correct / total



def validate_epoch(model, dataloader, criterion, device):
    """Validation per epoch"""

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Validation')

    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            # ✅ FIX FOR INCEPTION-V3 (returns tuple)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({'loss': f'{loss:.4f}', 'acc': f'{100*correct/total:.2f}%'})

    return running_loss / total, 100 * correct / total



def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_dir, patience=5):
    """Training loop with early stopping"""

    best_val_acc = 0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f"\n📌 Epoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        print(f"✅ Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
        print(f"🔍 Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

        # ✅ Save best checkpoint based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"💾 Saved Best Model (Val Accuracy: {best_val_acc:.2f}%)")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("⛔ Early stopping triggered!")
            break

    print(f"\n🎉 Training completed! Best Validation Accuracy = {best_val_acc:.2f}%")



if __name__ == "__main__":
    print("🚀 Starting Training...")

    cfg = Config()
    os.makedirs(cfg.MODEL_SAVE_DIR, exist_ok=True)

    # ✅ Load data
    train_loader, val_loader = get_data_loaders()

    # ✅ Build model based on config.MODEL_NAME
    model = build_model().to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=cfg.NUM_EPOCHS,
        device=cfg.DEVICE,
        save_dir=cfg.MODEL_SAVE_DIR,
        patience=cfg.PATIENCE
    )
