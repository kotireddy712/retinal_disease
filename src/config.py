"""
Configuration file for Retinal Disease Detection
"""

import torch
import os


class Config:

    # ---------------------------------------------------------
    # 🔥 AUTO-DETECT PROJECT ROOT
    # ---------------------------------------------------------
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # ---------------------------------------------------------
    # Dataset paths
    # ---------------------------------------------------------
    DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
    TRAIN_DIR = os.path.join(DATA_DIR, "train_images")
    TEST_DIR = os.path.join(DATA_DIR, "test_images")

    TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
    TEST_CSV = os.path.join(DATA_DIR, "test.csv")

    # Pseudo-labeling files
    PSEUDO_LABELS_CSV = os.path.join(DATA_DIR, "pseudo_labels_round1.csv")
    MIXED_TRAIN_CSV = os.path.join(DATA_DIR, "train_mixed_round1.csv")

    # ---------------------------------------------------------
    # Model Selection
    # ---------------------------------------------------------
    MODEL_NAME = "efficientnet_b4"
    PRETRAINED = True

    # ---------------------------------------------------------
    # Model Save Paths
    # ---------------------------------------------------------
    MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models", "saved_models")
    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, f"best_model_{MODEL_NAME}.pth")

    # ---------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # Training Params
    # ---------------------------------------------------------
    IMAGE_SIZE = 380
    NUM_CLASSES = 5
    BATCH_SIZE = 16
    NUM_EPOCHS = 35
    LEARNING_RATE = 0.0002
    WEIGHT_DECAY = 1e-4
    PATIENCE = 5

    # ---------------------------------------------------------
    # Train/Validation split
    # ---------------------------------------------------------
    TRAIN_SPLIT = 0.8

    # ---------------------------------------------------------
    # Hardware
    # ---------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2
    PIN_MEMORY = True

    # ---------------------------------------------------------
    # Class labels
    # ---------------------------------------------------------
    CLASS_NAMES = [
        'No DR',
        'Mild',
        'Moderate',
        'Severe',
        'Proliferative DR'
    ]

    # ---------------------------------------------------------
    # Seed
    # ---------------------------------------------------------
    SEED = 42
