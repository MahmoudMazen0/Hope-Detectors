# Medical Diagnostics Dashboard

A professional medical diagnostics application for cancer detection using CT scans and lab tests.

## Features

### 🩻 CT Scan Module
- Upload single or multiple CT scan images (DICOM, PNG, JPG)
- AI-powered cancer detection using deep learning
- Batch processing with results export
- DICOM folder support

### 🧪 Lab Tests Module
- Patient data input for blood & urine tests
- Multiple ML models: SVM, Logistic Regression, Random Forest, LightGBM, CatBoost, Stacked Model
- Batch processing from CSV/Excel files
- Confidence scores and predictions

## Files Structure

```
gui code/
├── gui_app.py          # Main GUI application
├── backend.py          # ML logic, models, predictions
├── background.jpg      # Splash screen background
├── page_background.png # Dashboard background
├── logo.png           # Application logo
├── run_app.bat        # Quick launch script
└── venv311/           # Python 3.11 virtual environment
```

## Requirements

- Python 3.11
- TensorFlow/Keras
- scikit-learn
- LightGBM
- CatBoost
- CustomTkinter
- PIL/Pillow
- pydicom (for DICOM support)

## Running the Application

### Option 1: Using batch file
```bash
run_app.bat
```

### Option 2: Direct Python
```bash
.\venv311\Scripts\python.exe gui_app.py
```

## Models Used

### Lab Tests (15 Features)
- **SVM (Best)** - Support Vector Machine
- **Logistic Regression** - Standard classification
- **Random Forest** - Ensemble method
- **LightGBM** - Gradient boosting
- **CatBoost** - Categorical boosting
- **Stacked Model** - Meta-learning ensemble

### CT Scans
- **Deep Learning (Keras)** - CNN for pancreatic cancer detection

## Authors

Hope Detectors Team
