"""
Streamlit Web Application for Retinal Disease Detection
"""

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import RetinalCNN
from src.config import Config

# Page config
st.set_page_config(
    page_title="Retinal Disease Detector",
    page_icon="👁️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .prediction-box {
        background-color: #d4edda;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #28a745;
        text-align: center;
    }
    .confidence-bar {
        background-color: #e9ecef;
        border-radius: 5px;
        height: 30px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">👁️ Retinal Disease Detection System</h1>', 
            unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2382/2382533.png", width=100)
    st.title("About")
    st.info("""
    This AI-powered system detects Diabetic Retinopathy from retinal fundus images.
    
    **Severity Levels:**
    - No DR
    - Mild
    - Moderate
    - Severe
    - Proliferative DR
    
    **Model:** CNN (ResNet50)
    **Accuracy:** ~85%
    """)
    
    st.warning("⚠️ This is for educational purposes only. Always consult a medical professional.")

# Load model
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RetinalCNN(
        model_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        pretrained=False
    )
    
    # Load trained weights
    try:
        checkpoint = torch.load('models/saved_models/best_model.pth', 
                               map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        return model, device
    except:
        st.error("❌ Model file not found! Please train the model first.")
        return None, device

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

# Prediction function
def predict(image, model, device):
    img_tensor = preprocess_image(image).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][pred_class].item()
    
    return pred_class, confidence, probabilities[0].cpu().numpy()

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Retinal Image")
    uploaded_file = st.file_uploader(
        "Choose a fundus image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a retinal fundus photograph"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Predict button
        if st.button('🔍 Analyze Image', type='primary', use_container_width=True):
            with st.spinner('Analyzing image...'):
                model, device = load_model()
                
                if model is not None:
                    pred_class, confidence, all_probs = predict(image, model, device)
                    
                    # Store results in session state
                    st.session_state['prediction'] = pred_class
                    st.session_state['confidence'] = confidence
                    st.session_state['probabilities'] = all_probs

with col2:
    st.subheader("📊 Prediction Results")
    
    if 'prediction' in st.session_state:
        pred_class = st.session_state['prediction']
        confidence = st.session_state['confidence']
        all_probs = st.session_state['probabilities']
        
        # Main prediction
        st.markdown(f"""
        <div class="prediction-box">
            <h2>Diagnosis: {Config.CLASS_NAMES[pred_class]}</h2>
            <h3>Confidence: {confidence*100:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence for all classes
        st.subheader("Confidence Scores")
        for i, class_name in enumerate(Config.CLASS_NAMES):
            prob = all_probs[i] * 100
            st.write(f"**{class_name}**")
            st.progress(prob / 100)
            st.write(f"{prob:.2f}%")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        if pred_class == 0:
            st.success("✅ No signs of diabetic retinopathy detected. Maintain regular check-ups.")
        elif pred_class == 1:
            st.info("ℹ️ Mild diabetic retinopathy detected. Consult an ophthalmologist soon.")
        elif pred_class == 2:
            st.warning("⚠️ Moderate diabetic retinopathy detected. Schedule an appointment with a specialist.")
        else:
            st.error("🚨 Severe/Proliferative diabetic retinopathy detected. Seek immediate medical attention!")
    else:
        st.info("👆 Upload an image and click 'Analyze' to see results")

# Additional information
st.markdown("---")
st.subheader("📖 Understanding Diabetic Retinopathy")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **What is it?**
    
    Diabetic Retinopathy (DR) is a diabetes complication that affects the eyes.
    It's caused by damage to blood vessels in the retina.
    """)

with col2:
    st.markdown("""
    **Risk Factors**
    
    - Poor blood sugar control
    - High blood pressure
    - High cholesterol
    - Long duration of diabetes
    """)

with col3:
    st.markdown("""
    **Prevention**
    
    - Regular eye exams
    - Control blood sugar
    - Maintain healthy lifestyle
    - Early detection is key!
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Developed by KASIREDDY KOTI REDDY & NATHANI LEELA KRISHNA</p>
    <p>Machine Learning Course Project 2024</p>
</div>
""", unsafe_allow_html=True)