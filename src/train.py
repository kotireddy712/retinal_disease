"""
Training script for Retinal Disease Detection using CNN (PyTorch)
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import random
import os

# ---------------------------------------------------------
# Correct imports based on folder structure
# ---------------------------------------------------------
from src.dataset import get_data_loaders
from src.model import build_model
from src.config import Config


# --------------------------------------------------------
#  ✅ SEEDING — ensures reproducibility
# --------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------
#  ✅ TRAIN FUNCTION — 1 epoch
# --------------------------------------------------------
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total, correct = 0, 0

    pbar = tqdm(dataloader, desc="Training")

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # Inception returns tuple (main, aux)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({
            "loss": f"{loss:.4f}",
            "acc": f"{100 * correct / total:.2f}%"
        })

    return running_loss / total, 100 * correct / total


# --------------------------------------------------------
#  ✅ VALIDATION FUNCTION — 1 epoch
# --------------------------------------------------------
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total, correct = 0, 0

    pbar = tqdm(dataloader, desc="Validation")

    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                "loss": f"{loss:.4f}",
                "acc": f"{100 * correct / total:.2f}%"
            })

    return running_loss / total, 100 * correct / total


# --------------------------------------------------------
#  ✅ MAIN TRAIN LOOP (Early Stopping + Save Best Model)
# --------------------------------------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, cfg):
    best_val_acc = 0
    epochs_no_improve = 0

    print(f"🔧 Selected Model: {cfg.MODEL_NAME}")

    for epoch in range(num_epochs):
        print(f"\n📌 Epoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        print(f"✅ Train: Loss={train_loss:.4f} | Acc={train_acc:.2f}%")
        print(f"🔍 Val:   Loss={val_loss:.4f} | Acc={val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), cfg.MODEL_SAVE_PATH)
            print(f"💾 Saved Best Model → {cfg.MODEL_SAVE_PATH}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= cfg.PATIENCE:
            print("\n⛔ Early stopping triggered.")
            break

    print(f"\n🎉 Training Finished! Best Val Accuracy = {best_val_acc:.2f}%")


# --------------------------------------------------------
#  ✅ ENTRY POINT
# --------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting Training...")

    cfg = Config()
    set_seed(cfg.SEED)

    os.makedirs(cfg.MODEL_SAVE_DIR, exist_ok=True)

    # Load dataset
    train_loader, val_loader = get_data_loaders()

    # Build model
    model = build_model().to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY
    )

    # Start training
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=cfg.NUM_EPOCHS,
        device=cfg.DEVICE,
        cfg=cfg
    )
