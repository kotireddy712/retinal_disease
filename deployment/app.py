import streamlit as st
import torch
import numpy as np
from PIL import Image
import sys, os

# -------------------------------------------------------------
# 🔥 ADD PROJECT ROOT TO PYTHON PATH
# -------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from src.model import build_model
from src.config import Config
from src.grad_cam import generate_gradcam   # ✅ OK

# -------------------------------------------------------------
# 🔧 STREAMLIT SETTINGS
# -------------------------------------------------------------
st.set_page_config(
    page_title="Retinal Disease Detection – EfficientNet-B4",
    page_icon="👁️",
    layout="wide"
)

cfg = Config()

st.title("👁️ Retinal Disease Detection (EfficientNet-B4)")
st.write("Upload a retinal fundus image to get predictions and Grad-CAM visualization.")


# -------------------------------------------------------------
# 🔥 LOAD MODEL (CACHED)
# -------------------------------------------------------------
@st.cache_resource
def load_model():
    device = cfg.DEVICE
    model = build_model()

    # ❗ FIX #1 — Use correct model path from config
    model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=device))

    model.eval()
    return model.to(device), device


# -------------------------------------------------------------
# 🔧 PREPROCESS IMAGE
# -------------------------------------------------------------
def preprocess(image):
    from torchvision import transforms
    tfm = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return tfm(image).unsqueeze(0)


# -------------------------------------------------------------
# 🔮 PREDICT
# -------------------------------------------------------------
def predict(image, model, device):
    tensor = preprocess(image).to(device)
    with torch.no_grad():
        output = model(tensor)
        if isinstance(output, tuple):
            output = output[0]   # for Inception safety

        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        pred_class = np.argmax(probs)
        return pred_class, probs


# -------------------------------------------------------------
# 📤 FILE UPLOAD
# -------------------------------------------------------------
uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", width=350)

    model, device = load_model()

    if st.button("🔍 Analyze Image"):
        pred_class, probs = predict(image, model, device)

        col1, col2 = st.columns(2)

        # ---------------------------------------------------------
        # 📊 Predictions
        # ---------------------------------------------------------
        with col1:
            st.subheader("📌 Prediction")
            st.success(f"### {cfg.CLASS_NAMES[pred_class]}")

            st.write("Confidence scores:")
            for i, cls in enumerate(cfg.CLASS_NAMES):
                st.write(f"**{cls}** — {probs[i]*100:.2f}%")
                st.progress(float(probs[i]))

        # ---------------------------------------------------------
        # 🔥 GRAD-CAM
        # ---------------------------------------------------------
        with col2:
            st.subheader("🔥 Grad-CAM Visualization")

            # ❗ FIX #2 — generate_gradcam returns (orig, heat, overlay)
            orig, heat, overlay = generate_gradcam(model, image, device)

            st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)
            st.image(heat, caption="Heatmap", use_container_width=True)

            st.info("Bright regions = areas the model used to make its decision.")


# -------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#777'>
Developed by <b>Kasireddy Koti Reddy</b> & <b>Nathani Leela Krishna</b><br>
Machine Learning Project – 2025
</div>
""", unsafe_allow_html=True)
