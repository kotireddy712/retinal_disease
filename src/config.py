"""
Configuration file for Retinal Disease Detection
"""
import torch
import os

class Config:

    # ✅ Base folder path in Google Drive
    BASE_DIR = "/content/drive/MyDrive/retinal_disease"

    # ✅ Dataset paths
    DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
    TRAIN_DIR = os.path.join(DATA_DIR, "train_images")
    TEST_DIR = os.path.join(DATA_DIR, "test_images")

    TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
    TEST_CSV = os.path.join(DATA_DIR, "test.csv")

    # ✅ Model to train → change this to test different models:
    # Options: "resnet50", "efficientnet_b4", "inception_v3"
    MODEL_NAME = "efficientnet_b4"
    PRETRAINED = True

    # ✅ Save model separately based on architecture
    MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models", "saved_models")
    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, f"best_model_{MODEL_NAME}.pth")

    # ✅ Output directory
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ✅ Training hyperparameters
    IMAGE_SIZE = 380 #299               # 224 for ResNet/Inception, 380 for EfficientNet
    NUM_CLASSES = 5
    BATCH_SIZE = 16 #24
    NUM_EPOCHS = 35
    LEARNING_RATE = 0.0002 #0.0003
    WEIGHT_DECAY = 1e-4
    PATIENCE = 5 #6

    # ✅ Data split
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.2

    # ✅ Hardware settings
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2
    PIN_MEMORY = True

    # ✅ Classes for prediction
    CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

    SEED = 42
