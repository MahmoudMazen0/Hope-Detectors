# 🏥 Hope Detectors - Medical Diagnostics Dashboard

<div align="center">

![Hope Detectors](assets/logo.png)

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
├── src/                          # Source Code
│   ├── gui_app.py               # Main GUI application
│   └── backend.py               # ML logic and predictions
│
├── models/                       # Machine Learning Models
│   ├── lab_tests/               # Lab Test Models
│   │   ├── svm/                 # SVM models
│   │   ├── random_forest/       # Random Forest models
│   │   ├── logistic_regression/ # Logistic Regression
│   │   ├── catboost/            # CatBoost models
│   │   ├── lightgbm/            # LightGBM models
│   │   └── stacked/             # Stacked ensemble models
│   ├── ct_scans/                # CT Scan deep learning model
│   └── scalers/                 # Feature scalers
│
├── config/                       # Configuration Files
├── assets/                       # Images & UI Assets
├── data/                         # Sample Data & Results
├── docs/                         # Documentation
├── output/                       # Results Output
│
├── requirements.txt
├── run_app.bat
└── README.md
```

## 🔧 Installation

### Prerequisites
- Python 3.11 or higher

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/MahmoudMazen0/Hope-Detectors.git
   cd Hope-Detectors
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv311
   .\venv311\Scripts\activate
   ```

3. **Install dependencies**
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
.\venv311\Scripts\activate
python src\gui_app.py
```

## 🛠️ Technologies Used

- **Frontend**: CustomTkinter
- **Backend**: Python 3.11
- **ML/DL**: TensorFlow, scikit-learn, LightGBM, CatBoost
- **Image Processing**: Pillow, pydicom

## 👥 Team

**Hope Detectors Team**

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

<div align="center">
Made with ❤️ by Hope Detectors Team
</div>
