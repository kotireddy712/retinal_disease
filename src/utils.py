"""
Utility functions
"""

import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """Count trainable parameters in model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return total_params, trainable_params


def save_checkpoint(model, optimizer, epoch, val_acc, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_acc = checkpoint['val_acc']
    
    print(f"Checkpoint loaded from epoch {epoch} with Val Acc: {val_acc:.2f}%")
    return model, optimizer, epoch, val_acc


def visualize_predictions(model, dataloader, class_names, device, num_images=16):
    """Visualize model predictions on a batch of images"""
    model.eval()
    
    # Get one batch
    images, labels = next(iter(dataloader))
    images = images[:num_images].to(device)
    labels = labels[:num_images]
    
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
    
    # Denormalize images for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    images = images.cpu() * std + mean
    images = torch.clamp(images, 0, 1)
    
    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_images):
        img = images[i].permute(1, 2, 0).numpy()
        true_label = class_names[labels[i]]
        pred_label = class_names[predictions[i]]
        
        axes[i].imshow(img)
        axes[i].axis('off')
        
        color = 'green' if labels[i] == predictions[i] else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', 
                         color=color, fontsize=9)
    
    plt.tight_layout()
    plt.show()


def generate_gradcam(model, image_tensor, target_layer, device, class_idx=None):
    """
    Generate Grad-CAM visualization
    
    Args:
        model: Trained model
        image_tensor: Input image tensor (1, 3, H, W)
        target_layer: Layer to compute gradients
        device: Device
        class_idx: Target class index (if None, uses predicted class)
    
    Returns:
        Grad-CAM heatmap overlaid on original image
    """
    model.eval()
    
    # Initialize Grad-CAM
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    # Generate CAM
    grayscale_cam = cam(input_tensor=image_tensor.unsqueeze(0).to(device), 
                        targets=None if class_idx is None else [class_idx])
    
    # Get the first (and only) image from batch
    grayscale_cam = grayscale_cam[0, :]
    
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    
    # Create visualization
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    
    return visualization, grayscale_cam


def plot_gradcam_samples(model, dataloader, class_names, target_layer, 
                          device, num_samples=8, save_dir=None):
    """Plot Grad-CAM visualizations for multiple samples"""
    model.eval()
    
    # Get samples
    images, labels = next(iter(dataloader))
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predictions = torch.max(outputs, 1)
    
    # Plot
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    for i in range(num_samples):
        # Original image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = images[i].numpy().transpose(1, 2, 0)
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Original\nTrue: {class_names[labels[i]]}')
        axes[i, 0].axis('off')
        
        # Grad-CAM
        vis, heatmap = generate_gradcam(model, images[i], target_layer, device)
        
        axes[i, 1].imshow(heatmap, cmap='jet')
        axes[i, 1].set_title('Grad-CAM Heatmap')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(vis)
        axes[i, 2].set_title(f'Overlay\nPred: {class_names[predictions[i]]}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/gradcam_samples.png', dpi=300, bbox_inches='tight')
        print(f"Grad-CAM visualizations saved to {save_dir}")
    
    plt.show()