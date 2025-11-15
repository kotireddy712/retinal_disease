"""
Model builder for Retinal Disease Detection
Supports: ResNet50 / EfficientNet-B4 / Inception-V3
"""

import torch
import torch.nn as nn
import torchvision.models as models
from src.config import Config   # ✅ FIXED IMPORT


def build_model():
    cfg = Config()
    model_name = cfg.MODEL_NAME.lower()

    # ------------------------------------------------------------
    # ✅ RESNET50
    # ------------------------------------------------------------
    if model_name == "resnet50":
        print("✅ Loading ResNet50 pretrained model...")
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, cfg.NUM_CLASSES)

    # ------------------------------------------------------------
    # ✅ EFFICIENTNET-B4
    # ------------------------------------------------------------
    elif model_name == "efficientnet_b4":
        print("✅ Loading EfficientNet-B4 pretrained model...")
        model = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
        )
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, cfg.NUM_CLASSES)

    # ------------------------------------------------------------
    # ✅ INCEPTION-V3
    # ------------------------------------------------------------
    elif model_name == "inception_v3":
        print("✅ Loading Inception-V3 pretrained model...")
        model = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=True  # must stay True
        )
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, cfg.NUM_CLASSES)

    else:
        raise ValueError(f"❌ Invalid MODEL_NAME: {cfg.MODEL_NAME}")

    return model.to(cfg.DEVICE)
