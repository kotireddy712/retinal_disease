"""
Dataset class for Retinal Disease Detection
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
from torchvision import transforms
from config import Config


class RetinalDataset(Dataset):
    """Dataset class to load retinal images and labels"""

    def __init__(self, dataframe, img_dir, transform=None, is_test=False):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "id_code"]

        # Try valid extensions
        for ext in ['.png', '.jpg', '.jpeg']:
            img_path = os.path.join(self.img_dir, img_name + ext)
            if os.path.exists(img_path):
                break
        else:
            raise FileNotFoundError(f"❌ Image missing: {img_name}")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, img_name

        label = int(self.df.loc[idx, "diagnosis"])
        return image, label


def get_transforms(image_size=224, is_training=True):
    """Augmentation transforms"""
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def get_data_loaders():
    """Creates train + validation DataLoaders"""
    cfg = Config()
    torch.manual_seed(cfg.SEED)

    df = pd.read_csv(cfg.TRAIN_CSV)

    dataset = RetinalDataset(
        dataframe=df,
        img_dir=cfg.TRAIN_DIR,
        transform=get_transforms(cfg.IMAGE_SIZE, True)
    )

    train_size = int(cfg.TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size

    train_data, val_data = random_split(dataset, [train_size, val_size])

    # Validation transform (NO augmentation)
    val_data.dataset.transform = get_transforms(cfg.IMAGE_SIZE, False)

    train_loader = DataLoader(
        train_data,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=True,     # ✅ fixes Inception batch issue
    )

    val_loader = DataLoader(
        val_data,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=True,     # ✅ same fix for val during training
    )

    return train_loader, val_loader


def get_eval_loader():
    """Loader used ONLY for evaluation / testing — MUST NOT drop last batch"""
    cfg = Config()

    df = pd.read_csv(cfg.TRAIN_CSV)

    dataset = RetinalDataset(
        dataframe=df,
        img_dir=cfg.TRAIN_DIR,
        transform=get_transforms(cfg.IMAGE_SIZE, False),
        is_test=False
    )

    return DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=False      # ✅ DO NOT DROP — required for evaluation metrics
    )
