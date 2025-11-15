"""
Predict on Test Dataset and Generate submission.csv
"""

import torch
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from src.config import Config
from model import build_model


def load_model():
    cfg = Config()

    print("✅ Loading ResNet50 pretrained model...")

    model = build_model()
    model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=cfg.DEVICE))
    model.to(cfg.DEVICE)
    model.eval()

    print("✅ Model loaded!")

    return model, cfg


def predict_test_dataset():
    model, cfg = load_model()

    test_df = pd.read_csv(cfg.TEST_CSV)
    predictions = []

    preprocess = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    print("\n🚀 Running predictions on test set...")

    for i, row in test_df.iterrows():
        img_id = row["id_code"]

        # search .png / .jpg automatically
        image_path = None
        for ext in ["png", "jpg", "jpeg"]:
            tmp_path = os.path.join(cfg.TEST_IMAGES, f"{img_id}.{ext}")
            if os.path.exists(tmp_path):
                image_path = tmp_path
                break

        if image_path is None:
            print(f"⚠️ Missing image: {img_id}")
            predictions.append(0)
            continue

        img = Image.open(image_path).convert("RGB")
        img = preprocess(img).unsqueeze(0).to(cfg.DEVICE)

        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            predictions.append(int(predicted.item()))

    print("\n✅ Predictions Completed!")

    # Save CSV
    test_df["diagnosis"] = predictions
    output_path = os.path.join(cfg.OUTPUT_DIR, "submission.csv")
    test_df.to_csv(output_path, index=False)

    print(f"📁 Saved results to: {output_path}")
    print(test_df.head())


if __name__ == "__main__":
    predict_test_dataset()
