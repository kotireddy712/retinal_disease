"""
Grad-CAM Implementation (Auto Layer Detection)
Works for:
- EfficientNet-B4
- ResNet50
- Inception-V3
"""

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------
#  AUTO SELECT BEST GRAD-CAM LAYER BASED ON MODEL TYPE
# ---------------------------------------------------------
def get_target_layer(model):
    name = model.__class__.__name__.lower()

    # EfficientNet-B4 ➜ automatically locate last Conv2D
    if "efficientnet" in name:
        for layer_name, layer in reversed(list(model.named_modules())):
            if isinstance(layer, torch.nn.Conv2d):
                print(f"⚡ Using Grad-CAM Layer (EfficientNet): {layer_name}")
                return layer_name

    # ResNet50
    if "resnet" in name:
        print("⚡ Using Grad-CAM Layer (ResNet50): layer4.2.conv3")
        return "layer4.2.conv3"

    # Inception-V3
    if "inception" in name:
        print("⚡ Using Grad-CAM Layer (Inception-V3): Mixed_7c.branch_pool.conv")
        return "Mixed_7c.branch_pool.conv"

    raise ValueError("❌ Grad-CAM target layer not found for this model.")


# ---------------------------------------------------------
#   GRAD-CAM MODULE
# ---------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.model.eval()

        self.gradients = None
        self.activations = None

        modules = dict(model.named_modules())
        if target_layer_name not in modules:
            raise KeyError(f"❌ Layer '{target_layer_name}' not found in model.")

        self.target_layer = modules[target_layer_name]

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x):
        output = self.model(x)

        if isinstance(output, tuple):
            output = output[0]  # Inception

        pred = output.argmax(dim=1)

        self.model.zero_grad()
        output[:, pred].backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1)

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam.squeeze().cpu().numpy(), pred.item()


# ---------------------------------------------------------
#  GENERATE GRAD-CAM IMAGES
# ---------------------------------------------------------
def generate_gradcam(model, image_pil, device):

    from src.config import Config
    cfg = Config()

    tfm = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    x = tfm(image_pil).unsqueeze(0).to(device)

    # auto-detect layer
    target_layer = get_target_layer(model)

    cam_generator = GradCAM(model, target_layer)
    cam, pred_class = cam_generator(x)

    cam = np.uint8(255 * cam)
    heatmap = Image.fromarray(cam).resize(image_pil.size).convert("RGB")
    heatmap = np.array(heatmap)

    import cv2
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = 0.5 * np.array(image_pil) + 0.5 * heatmap
    overlay = np.uint8(overlay)

    return np.array(image_pil), heatmap, overlay
