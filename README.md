# 🏥 Hope Detectors - Medical Diagnostics Dashboard

<div align="center">

![Hope Detectors](gui%20code/logo.png)

**AI-powered Cancer Detection System using CT Scans and Lab Tests**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📋 Overview

Hope Detectors is a professional medical diagnostics application designed to assist healthcare professionals in early cancer detection. The system combines deep learning algorithms for CT scan analysis with machine learning models for laboratory test interpretation.

## ✨ Features

### 🩻 CT Scan Analysis Module
- **Multi-format Support**: Upload DICOM, PNG, and JPG images
- **AI-Powered Detection**: Deep learning CNN for pancreatic cancer detection
- **Batch Processing**: Process multiple scans with automated results export
- **DICOM Folder Support**: Load entire DICOM series at once

### 🧪 Laboratory Tests Module
- **Comprehensive Input**: 15-feature patient data for blood & urine analysis
- **Multiple ML Models**:
  - SVM (Support Vector Machine) - *Best Performance*
  - Logistic Regression
  - Random Forest
  - LightGBM
  - CatBoost
  - Stacked Ensemble Model
- **Batch Processing**: CSV/Excel file support for multiple patients
- **Confidence Scores**: Probability-based predictions

## 📁 Project Structure

```
Hope-Detectors/
├── 📁 gui code/
│   ├── gui_app.py          # Main GUI application
│   ├── backend.py          # ML logic and predictions
│   ├── background.jpg      # Splash screen
│   ├── page_background.png # Dashboard background
│   ├── logo.png            # Application logo
│   └── run_app.bat         # Quick launch script
├── 📁 Models/
│   ├── FINAL_MODEL_SVM.pkl
│   ├── FINAL_SCALER.pkl
│   ├── random_forest_aggressive.pkl
│   ├── logistic_regression_moderate.pkl
│   ├── stacked_meta_model_15features.pkl
│   └── ... (other model files)
├── 📄 requirements.txt     # Python dependencies
├── 📄 config.json          # Application configuration
└── 📄 README.md            # This file
```

## 🔧 Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/MahmoudMazen0/Hope-Detectors.git
   cd Hope-Detectors
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv311
   ```

3. **Activate virtual environment**
   - Windows:
     ```bash
     .\venv311\Scripts\activate
     ```
   - Linux/macOS:
     ```bash
     source venv311/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Quick Start (Windows)
```bash
run_app.bat
```

### Manual Start
```bash
cd "gui code"
python gui_app.py
```

Or with virtual environment:
```bash
.\venv311\Scripts\python.exe "gui code\gui_app.py"
```

## 📊 Model Performance

| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| SVM | **Best** | High | High |
| Random Forest | Good | Good | Good |
| LightGBM | Good | Good | Good |
| CatBoost | Good | Good | Good |
| Stacked Ensemble | Good | Good | Good |

## 🛠️ Technologies Used

- **Frontend**: CustomTkinter (Modern GUI)
- **Backend**: Python 3.11
- **ML/DL**: TensorFlow, scikit-learn, LightGBM, CatBoost
- **Image Processing**: Pillow, pydicom
- **Data Processing**: Pandas, NumPy

## 📦 Dependencies

```
customtkinter
pandas
numpy
joblib
scikit-learn
pillow
openpyxl
tensorflow
lightgbm
catboost
pydicom
```

## 👥 Team

**Hope Detectors Team**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Made with ❤️ by Hope Detectors Team
</div>
