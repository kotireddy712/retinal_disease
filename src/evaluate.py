"""
Evaluation & Metrics for Retinal Disease Detection
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_recall_fscore_support,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize


# ==========================================================
#  EVALUATION FUNCTION
# ==========================================================
def evaluate_model(model, dataloader, device, class_names):
    """Runs inference and calculates accuracy, precision, recall, F1"""

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            # Inception returns (logits, aux_logits)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )

    print("\n📊 Evaluation Summary")
    print(f"✅ Accuracy: {accuracy * 100:.2f}%")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall: {recall:.4f}")
    print(f"✅ F1 Score: {f1:.4f}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }


# ==========================================================
#  CONFUSION MATRIX PLOT
# ==========================================================
def plot_confusion_matrix(labels, predictions, class_names, save_path=None):

    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        fmt="d",
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"📁 Confusion Matrix Saved: {save_path}")

    plt.show()


# ==========================================================
#  ROC CURVE
# ==========================================================
def plot_roc_curves(labels, probabilities, class_names, save_path=None):

    labels_bin = label_binarize(labels, classes=list(range(len(class_names))))

    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 7))

    for i in range(len(class_names)):
        plt.plot(
            fpr[i],
            tpr[i],
            lw=2,
            label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})"
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"📁 ROC Curve Saved: {save_path}")

    plt.show()


# ==========================================================
#  PRINT CLASSIFICATION REPORT
# ==========================================================
def print_classification_report(labels, predictions, class_names):
    print("\n=== Classification Report ===")
    print(classification_report(labels, predictions, target_names=class_names))
    print("=" * 50)



# ==========================================================
#  MAIN ENTRY — THIS MAKES evaluation RUN
# ==========================================================
if __name__ == "__main__":
    print("🔍 Starting Model Evaluation...\n")

    from config import Config
    from model import build_model
    from dataset import get_data_loaders

    cfg = Config()

    # Load validation loader only
    _, val_loader = get_data_loaders()

    # Build model and load weights
    model = build_model()
    model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=cfg.DEVICE))
    model.to(cfg.DEVICE)

    print(f"📁 Loaded model from: {cfg.MODEL_SAVE_PATH}")

    # Run evaluation
    results = evaluate_model(model, val_loader, cfg.DEVICE, cfg.CLASS_NAMES)

    # Detailed report
    print_classification_report(
        results["labels"], results["predictions"], cfg.CLASS_NAMES
    )

    # Plots
    plot_confusion_matrix(
        results["labels"], results["predictions"], cfg.CLASS_NAMES,
        save_path=f"{cfg.OUTPUT_DIR}/confusion_matrix.png"
    )

    plot_roc_curves(
        results["labels"], results["probabilities"], cfg.CLASS_NAMES,
        save_path=f"{cfg.OUTPUT_DIR}/roc_curves.png"
    )
