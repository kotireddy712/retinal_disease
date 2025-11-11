# 👁️ Retinal Disease Detection using CNN

An automated deep learning system for detecting and classifying Diabetic Retinopathy from retinal fundus images using Convolutional Neural Networks (CNN).

## 📋 Project Overview

This project implements a CNN-based classifier that analyzes retinal fundus images to predict disease severity levels:
- **No DR** (Diabetic Retinopathy)
- **Mild**
- **Moderate**
- **Severe**
- **Proliferative DR**

**Target Accuracy:** ~85%

## 🚀 Features

- ✅ Automated retinal disease classification
- ✅ Multiple CNN architectures (ResNet50, EfficientNet, InceptionV3)
- ✅ Grad-CAM visualizations for model interpretability
- ✅ Data augmentation for better generalization
- ✅ Web-based deployment using Streamlit
- ✅ Comprehensive evaluation metrics
- ✅ GPU support for faster training

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.8+ |
| Deep Learning | PyTorch |
| Image Processing | OpenCV, PIL |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Explainability | Grad-CAM |
| Deployment | Streamlit |
| Training Platform | Google Colab / Kaggle |

## 📁 Project Structure

```
retinal-disease-detection/
│
├── data/
│   ├── raw/              # Kaggle dataset
│   └── processed/        # Preprocessed images
│
├── models/
│   ├── saved_models/     # Trained model weights
│   └── checkpoints/      # Training checkpoints
│
├── notebooks/
│   └── train.ipynb       # Main training notebook
│
├── src/
│   ├── __init__.py
│   ├── config.py         # Configuration settings
│   ├── dataset.py        # Dataset class
│   ├── model.py          # CNN architecture
│   ├── train.py          # Training functions
│   ├── evaluate.py       # Evaluation metrics
│   └── utils.py          # Utility functions
│
├── deployment/
│   ├── app.py           # Streamlit app
│   └── requirements.txt
│
├── outputs/
│   ├── plots/           # Training plots
│   └── gradcam/         # Grad-CAM visualizations
│
├── requirements.txt
└── README.md
```

## 📊 Dataset

**Primary Dataset:** APTOS 2019 Blindness Detection (Kaggle)

- High-quality retinal fundus images
- Labeled with DR severity (0-4)
- Download from: [Kaggle APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection)

**Alternative Datasets:**
- MESSIDOR
- EyePACS

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Google Colab account (for GPU training)
- Kaggle account (for dataset)

### Step-by-Step Guide

#### 1. **Clone/Download Project**
```bash
# If using Git
git clone https://github.com/yourusername/retinal-disease-detection.git
cd retinal-disease-detection

# Or download and extract the ZIP file
```

#### 2. **Install Local Dependencies** (Optional)
```bash
pip install -r requirements.txt
```

*Note: Heavy libraries like PyTorch will be installed in Google Colab*

#### 3. **Setup Kaggle API**
1. Go to [Kaggle.com](https://www.kaggle.com) → Account → "Create New API Token"
2. Download `kaggle.json`
3. Keep it ready for uploading to Colab

#### 4. **Organize Your Files**
Ensure you have this structure locally:
```
retinal-disease-detection/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── notebooks/
│   └── train.ipynb
└── requirements.txt
```

## 🎓 Training on Google Colab

### Complete Training Workflow

#### **Step 1: Upload to Google Drive**
1. Create folder in Google Drive: `retinal_disease/`
2. Upload the entire `src/` folder to: `MyDrive/retinal_disease/src/`
3. Upload `train.ipynb` to: `MyDrive/retinal_disease/notebooks/`

#### **Step 2: Open Notebook in Colab**
1. Go to [Google Colab](https://colab.research.google.com)
2. File → Open Notebook → Google Drive
3. Navigate to and open `train.ipynb`

#### **Step 3: Enable GPU**
1. Runtime → Change runtime type
2. Hardware accelerator → **GPU** (T4 recommended)
3. Save

#### **Step 4: Run Cells Sequentially**

**Cell 1:** Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Cell 2:** Install packages (takes ~2 minutes)

**Cell 3:** Setup Kaggle API
- Upload your `kaggle.json` when prompted

**Cell 4:** Download dataset (takes ~5-10 minutes)
- Dataset will be downloaded to Google Drive
- Approximately 5GB

**Cell 5-21:** Follow the notebook cells for:
- Data loading and preprocessing
- Model training (~2-3 hours for 25 epochs)
- Evaluation and visualization
- Saving results

#### **Step 5: Monitor Training**
- Training progress with tqdm bars
- Loss and accuracy printed each epoch
- Best model auto-saved to Drive

### Expected Training Time
- **With GPU (T4):** 2-3 hours for 25 epochs
- **With GPU (A100):** 1-1.5 hours
- **CPU Only:** Not recommended (10+ hours)

### Where Files Are Stored

After training, you'll find:

```
Google Drive/MyDrive/retinal_disease/
│
├── data/
│   └── raw/
│       ├── train_images/    # 3662 images
│       ├── train.csv        # Labels
│       └── test_images/     # Test images
│
├── models/
│   └── saved_models/
│       ├── best_model.pth           # Best model checkpoint
│       ├── final_model.pth          # Final trained model
│       └── checkpoint_epoch_X.pth   # Periodic checkpoints
│
└── outputs/
    ├── plots/
    │   ├── training_history.png
    │   ├── confusion_matrix.png
    │   └── roc_curves.png
    └── gradcam/
        └── gradcam_samples.png
```

## 📈 Evaluation Metrics

The model is evaluated using:
- **Accuracy**
- **Precision, Recall, F1-Score** (per class and weighted average)
- **Confusion Matrix**
- **ROC-AUC Curves** (multi-class)
- **Grad-CAM Visualizations**

## 🌐 Deployment

### Run Streamlit App Locally

1. **Download trained model from Google Drive:**
   - Download `best_model.pth` from Drive
   - Place in: `models/saved_models/best_model.pth`

2. **Install deployment requirements:**
```bash
pip install streamlit torch torchvision pillow numpy
```

3. **Run the app:**
```bash
cd deployment
streamlit run app.py
```

4. **Open browser:** http://localhost:8501

### Deploy to Cloud

**Streamlit Cloud (Free):**
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Deploy!

**Hugging Face Spaces:**
1. Create account at [huggingface.co](https://huggingface.co)
2. Create new Space (Streamlit)
3. Upload code and model
4. Deploy

## 🎯 Results

### Model Performance
- **Training Accuracy:** ~92%
- **Validation Accuracy:** ~88%
- **Test Accuracy:** ~85%

### Sample Predictions
The model successfully identifies:
- Microaneurysms
- Hemorrhages
- Hard exudates
- Cotton wool spots
- Neovascularization

### Grad-CAM Visualizations
Heatmaps highlight regions contributing to predictions, ensuring model interpretability.

## 👥 Team

- **KASIREDDY KOTI REDDY** (B230373CS)
- **NATHANI LEELA KRISHNA** (B231122CS)

## 📝 License

This project is for educational purposes as part of a Machine Learning course.

## 🙏 Acknowledgments

- APTOS 2019 Blindness Detection Challenge
- PyTorch Team
- Kaggle Community

## 🐛 Troubleshooting

### Common Issues

**1. "CUDA out of memory"**
- Solution: Reduce batch size in `config.py` (try 16 or 8)

**2. "Module not found"**
- Solution: Ensure all files uploaded to correct Drive folder
- Check `sys.path.append()` in notebook

**3. "Kaggle API credentials not found"**
- Solution: Re-upload `kaggle.json` and run Cell 3 again

**4. Dataset download fails**
- Solution: Check internet connection
- Manually download from Kaggle and upload to Drive

**5. Training too slow**
- Solution: Ensure GPU is enabled in Colab
- Check: `torch.cuda.is_available()` should return `True`

## 📧 Contact

For questions or issues, contact:
- Email: your.email@example.com
- GitHub Issues: [Project Issues](https://github.com/yourusername/retinal-disease-detection/issues)

---

**⚠️ Disclaimer:** This system is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult qualified ophthalmologists for medical decisions.