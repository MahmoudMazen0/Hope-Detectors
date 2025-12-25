# 🏥 Hope Detectors - Medical Diagnostics Dashboard

<div align="center">

![Hope Detectors](assets/logo.png)

### **AI-Powered Cancer Detection System**
*Early detection saves lives - Combining CT Scans Analysis with Lab Tests*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![CustomTkinter](https://img.shields.io/badge/CustomTkinter-5.x-1a73e8?style=for-the-badge)](https://github.com/TomSchimansky/CustomTkinter)
[![License](https://img.shields.io/badge/License-MIT-00C853?style=for-the-badge)](LICENSE)

[**📥 Download**](#-installation) · [**🚀 Quick Start**](#-quick-start) · [**📖 Documentation**](#-features) · [**🤝 Contributing**](#-contributing)

</div>

---

## 📋 Overview

**Hope Detectors** is a professional medical diagnostics application designed to assist healthcare professionals in early cancer detection. The system combines:

- 🧠 **Deep Learning** for CT Scan analysis (EfficientNetB2)
- 🔬 **Machine Learning** for Lab Test interpretation (SVM, CatBoost, Random Forest)
- 📊 **Battle-tested Models** trained on real medical datasets

> ⚠️ **Disclaimer**: This tool is for educational and research purposes. Always consult qualified healthcare professionals for medical decisions.

---

## 🎬 Demo Video

<div align="center">

![Hope Detectors Demo](assets/demo.gif)

*Watch Hope Detectors in action - AI-powered cancer detection at your fingertips!*

</div>

---

## ✨ Features

### 🩻 CT Scan Analysis Module
| Feature | Description |
|---------|-------------|
| **Multi-format Support** | DICOM (.dcm), PNG, JPG images |
| **AI Detection** | EfficientNetB2 deep learning model |
| **Batch Processing** | Analyze multiple scans at once |
| **DICOM Folders** | Load entire scan series |
| **Confidence Scores** | Probability-based predictions |

### 🧪 Laboratory Tests Module
| Feature | Description |
|---------|-------------|
| **Patient Input** | Age, Sex, Creatinine, Bilirubin, Glucose, Urine Volume, Urine pH |
| **Multiple Models** | SVM, Logistic Regression, Random Forest, CatBoost |
| **Batch Processing** | CSV/Excel file support |
| **Auto Z-Score Normalization** | Handles raw data automatically |

### 📋 History & Analytics
| Feature | Description |
|---------|-------------|
| **Auto-Save** | Every analysis is automatically saved |
| **History View** | Browse past analyses in a table |
| **Export to CSV** | Download history for reporting |
| **Clear History** | Reset when needed |

---

##  Project Structure

```
Hope-Detectors/
├── 📂 src/                     # Source Code
│   ├── gui_app.py             # Main GUI application (CTkinter)
│   ├── backend.py             # ML logic, predictions & history
│   └── __init__.py
│
├── 📂 models/                  # Machine Learning Models
│   ├── lab_tests/             # Lab Test Models
│   │   ├── svm/              # SVM (Best Performance)
│   │   ├── random_forest/    # Random Forest
│   │   ├── logistic_regression/
│   │   └── catboost/         # CatBoost ensemble
│   ├── ct_scans/             # CT Deep Learning Model
│   │   └── final_model.keras # EfficientNetB2
│   └── scalers/              # Feature scalers (.pkl)
│
├── 📂 assets/                  # UI Resources
│   ├── logo.png              # App logo
│   ├── icon.ico              # Windows icon
│   └── page_background.png   # Dashboard background
│
├── 📂 config/                  # Configuration files
├── 📂 data/                    # Sample datasets
├── 📂 output/                  # Results & history
├── 📂 docs/                    # Documentation
│
├── 📄 requirements.txt        # Python dependencies
├── 📄 run_app.bat             # Quick start script
├── 📄 LICENSE                 # MIT License
└── 📄 README.md               # This file
```

---

## 🔧 Installation

### Prerequisites
- **Python 3.11** or higher
- **Windows 10/11** (recommended)

### Option 1: Download Executable (Recommended)
1. Download `HopeDetectors.exe` from [Releases](https://github.com/MahmoudMazen0/Hope-Detectors/releases)
2. Run the executable - no installation needed!

### Option 2: Install from Source

```bash
# 1. Clone the repository
git clone https://github.com/MahmoudMazen0/Hope-Detectors.git
cd Hope-Detectors

# 2. Create virtual environment
python -m venv venv311
.\venv311\Scripts\activate  # Windows
source venv311/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Windows (Easiest)
```batch
run_app.bat
```

### Manual Start
```bash
.\venv311\Scripts\activate
python src\gui_app.py
```

### From Executable
Simply run `dist\HopeDetectors\HopeDetectors.exe`

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Frontend** | CustomTkinter, Tkinter, Pillow |
| **Backend** | Python 3.11 |
| **Deep Learning** | TensorFlow 2.20, Keras |
| **Machine Learning** | scikit-learn, CatBoost |
| **Data Processing** | Pandas, NumPy |
| **Medical Imaging** | pydicom, Pillow |
| **Packaging** | PyInstaller |

---

## 📊 Model Performance

### Lab Tests (SVM - Best Model)
| Metric | Score |
|--------|-------|
| Accuracy | 92% |
| Precision | 91% |
| Recall | 93% |

### CT Scans (EfficientNetB2)
| Metric | Score |
|--------|-------|
| AUC-ROC | 0.94 |

---

## 👥 Team

<div align="center">

**Hope Detectors Development Team**

*Building AI-powered healthcare solutions*

</div>

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

### Made with ❤️ by Hope Detectors Team

**🌟 Star this repo if you found it helpful!**

[![GitHub stars](https://img.shields.io/github/stars/MahmoudMazen0/Hope-Detectors?style=social)](https://github.com/MahmoudMazen0/Hope-Detectors)

</div>
