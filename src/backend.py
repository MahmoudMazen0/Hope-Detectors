"""
Backend module for Medical Diagnostics System.
Contains ML logic, preprocessing, and prediction functions.
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# For CT Model
try:
    import keras
except ImportError:
    try:
        from tensorflow import keras
    except ImportError:
        keras = None

# For CatBoost and LightGBM models
try:
    import catboost
except ImportError:
    catboost = None

try:
    import lightgbm
except ImportError:
    lightgbm = None

# For DICOM images
try:
    import pydicom
except ImportError:
    pydicom = None

from PIL import Image


# ============== RAW DATA STATISTICS ==============
# Original statistics from pancreatic_600_risk123.csv training data
# Used for Z-score normalization: z = (x - mean) / std
RAW_STATS = {
    'age': {'mean': 57.0283, 'std': 16.1234},
    'sex': {'mean': 0.565, 'std': 0.4961},
    'creatinine': {'mean': 1.0515, 'std': 0.2852},
    'bilirubin': {'mean': 1.4347, 'std': 0.8909},
    'glucose': {'mean': 100.0701, 'std': 18.6507},
    'urine_volume': {'mean': 78.6932, 'std': 29.5679},
    'urine_pH': {'mean': 6.1971, 'std': 0.9966},
}

# Feature columns expected by the model
FEATURE_COLUMNS = [
    "age", "sex_encoded", "creatinine", "bilirubin", "glucose", "urine_volume", "urine_pH",
    "bili_creat_ratio", "bili_creat_product", "age_squared", "age_bili", "age_creat",
    "glucose_age", "urine_ratio", "risk_score"
]

# Available models - paths relative to project root
AVAILABLE_MODELS = {
    # SVM Models
    "SVM (Best)": "models/lab_tests/svm/FINAL_MODEL_SVM.pkl",
    "SVM Moderate": "models/lab_tests/svm/svm_moderate.pkl",
    
    # Logistic Regression
    "Logistic Regression": "models/lab_tests/logistic_regression/logistic_regression_moderate.pkl",
    
    # Random Forest
    "Random Forest": "models/lab_tests/random_forest/random_forest_aggressive.pkl",
    
    # LightGBM Models
    "LightGBM Best": "models/lab_tests/lightgbm/lgb_best_model.pkl",
    "LightGBM Moderate": "models/lab_tests/lightgbm/lgb_model_1_moderate.pkl",
    "LightGBM Aggressive": "models/lab_tests/lightgbm/lgb_model_2_aggressive.pkl",
    "LightGBM Finetuned": "models/lab_tests/lightgbm/lgb_model_3_finetuned.pkl",
    
    # CatBoost Models
    "CatBoost Best": "models/lab_tests/catboost/cat_best_model.pkl",
    "CatBoost Moderate": "models/lab_tests/catboost/cat_model_1_moderate.pkl",
    "CatBoost Aggressive": "models/lab_tests/catboost/cat_model_2_aggressive.pkl",
    "CatBoost Finetuned": "models/lab_tests/catboost/cat_model_3_finetuned.pkl",
    
    # Stacked Models
    "Stacked Meta Model": "models/lab_tests/stacked/stacked_meta_model.pkl",
    "Stacked Meta Model (15 Features)": "models/lab_tests/stacked/stacked_meta_model_15features.pkl",
}


class MedicalPredictor:
    """Class to handle model loading and predictions."""
    
    def __init__(self, base_path=None):
        # Models are in parent directory (Gui folder)
        if base_path:
            self.base_path = base_path
        else:
            self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model = None
        self.scaler = None
        self.current_model_name = None
    
    def load_model(self, model_name="SVM (Best)"):
        """Load a model and scaler."""
        try:
            model_file = AVAILABLE_MODELS.get(model_name, "FINAL_MODEL_SVM.pkl")
            model_path = os.path.join(self.base_path, model_file)
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(os.path.join(self.base_path, "models/scalers/FINAL_SCALER.pkl"))
            self.current_model_name = model_name
            return True, f"Model '{model_name}' loaded successfully."
        except Exception as e:
            self.model = None
            self.scaler = None
            return False, f"Failed to load model: {str(e)}"
    
    def normalize_features(self, age, sex, creatinine, bilirubin, glucose, urine_volume, urine_pH):
        """Normalize raw input values using Z-score normalization."""
        return {
            'age': (age - RAW_STATS['age']['mean']) / RAW_STATS['age']['std'],
            'sex': (sex - RAW_STATS['sex']['mean']) / RAW_STATS['sex']['std'],
            'creatinine': (creatinine - RAW_STATS['creatinine']['mean']) / RAW_STATS['creatinine']['std'],
            'bilirubin': (bilirubin - RAW_STATS['bilirubin']['mean']) / RAW_STATS['bilirubin']['std'],
            'glucose': (glucose - RAW_STATS['glucose']['mean']) / RAW_STATS['glucose']['std'],
            'urine_volume': (urine_volume - RAW_STATS['urine_volume']['mean']) / RAW_STATS['urine_volume']['std'],
            'urine_pH': (urine_pH - RAW_STATS['urine_pH']['mean']) / RAW_STATS['urine_pH']['std'],
        }
    
    def create_features(self, normalized):
        """Create feature DataFrame from normalized values."""
        age = normalized['age']
        sex = normalized['sex']
        creat = normalized['creatinine']
        bili = normalized['bilirubin']
        glucose = normalized['glucose']
        u_vol = normalized['urine_volume']
        u_ph = normalized['urine_pH']
        
        # Derived features from normalized values
        features = pd.DataFrame([[
            age, sex, creat, bili, glucose, u_vol, u_ph,
            bili / creat if creat != 0 else 0,  # bili_creat_ratio
            bili * creat,                        # bili_creat_product
            age ** 2,                            # age_squared
            age * bili,                          # age_bili
            age * creat,                         # age_creat
            glucose * age,                       # glucose_age
            u_vol / u_ph if u_ph != 0 else 0,    # urine_ratio
            0                                    # risk_score (default)
        ]], columns=FEATURE_COLUMNS)
        
        return features
    
    def predict(self, age, sex, creatinine, bilirubin, glucose, urine_volume, urine_pH):
        """
        Make a prediction for a single patient.
        
        Returns:
            tuple: (prediction, is_cancer, confidence)
        """
        if not self.model or not self.scaler:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Normalize features
        normalized = self.normalize_features(age, sex, creatinine, bilirubin, glucose, urine_volume, urine_pH)
        
        # Create feature DataFrame
        features = self.create_features(normalized)
        
        # Scale and predict
        scaled = self.scaler.transform(features)
        prediction = self.model.predict(scaled)[0]
        
        # Get confidence score
        try:
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(scaled)[0]
                confidence = max(proba) * 100
            elif hasattr(self.model, 'decision_function'):
                # For SVM without probability
                decision = abs(self.model.decision_function(scaled)[0])
                confidence = min(100, 50 + decision * 10)  # Scale to percentage
            else:
                confidence = 85.0  # Default
        except:
            confidence = 85.0
        
        return prediction, prediction == 1, confidence
    
    def predict_normalized(self, age, sex, creat, bili, glucose, u_vol, u_ph):
        """
        Make prediction for pre-normalized data (data already normalized like X_train.csv).
        Skip both base normalization AND scaler transform.
        """
        if not self.model or not self.scaler:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Data is already normalized, create derived features
        normalized = {
            'age': age, 'sex': sex, 'creatinine': creat, 'bilirubin': bili,
            'glucose': glucose, 'urine_volume': u_vol, 'urine_pH': u_ph
        }
        features = self.create_features(normalized)
        
        # For pre-normalized data, apply scaler to get proper derived feature normalization
        scaled = self.scaler.transform(features)
        prediction = self.model.predict(scaled)[0]
        
        # Get confidence
        try:
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(scaled)[0]
                confidence = max(proba) * 100
            elif hasattr(self.model, 'decision_function'):
                decision = abs(self.model.decision_function(scaled)[0])
                confidence = min(100, 50 + decision * 10)
            else:
                confidence = 85.0
        except:
            confidence = 85.0
        
        return prediction, prediction == 1, confidence
    
    def predict_batch(self, df, skip_normalization=False):
        """
        Make predictions for multiple patients.
        
        Args:
            df: DataFrame with patient data
            skip_normalization: If True, data is already normalized
        
        Returns:
            list of dicts with Diagnosis
        """
        if not self.model or not self.scaler:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        results = []
        for _, row in df.iterrows():
            try:
                # Get values
                age = float(row['age'])
                creat = float(row['creatinine'])
                bili = float(row['bilirubin'])
                glucose = float(row['glucose'])
                u_vol = float(row['urine_volume'])
                u_ph = float(row['urine_ph'])
                
                # Handle sex value based on data type
                if skip_normalization:
                    # Data is already normalized (sex is already a normalized float like -1.1 or 0.8)
                    sex = float(row['sex']) if 'sex' in row else float(row.get('sex_encoded', 0))
                else:
                    # Raw data - sex could be M/F or 0/1
                    sex_val = str(row['sex']).strip()
                    if sex_val.replace('.', '').replace('-', '').isdigit():
                        sex = float(sex_val)
                    else:
                        sex = 1 if sex_val.lower() in ['male', 'm'] else 0
                
                # Choose prediction method based on data type
                if skip_normalization:
                    prediction, is_cancer, conf = self.predict_normalized(age, sex, creat, bili, glucose, u_vol, u_ph)
                else:
                    prediction, is_cancer, conf = self.predict(age, sex, creat, bili, glucose, u_vol, u_ph)
                
                results.append({"Diagnosis": "CANCER" if is_cancer else "Healthy"})
            except Exception as e:
                results.append({"Diagnosis": "ERROR"})
        
        return results


def load_patient_file(file_path):
    """
    Load patient data from Excel or CSV file.
    
    Returns:
        tuple: (success, df_or_error_message, patient_names)
    """
    try:
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
        
        # Normalize column names
        column_mapping = {
            'sex_encod': 'sex',
            'sex_encoded': 'sex',
            'urine_volu': 'urine_volume',
            'urine_vol': 'urine_volume',
        }
        df.columns = [column_mapping.get(c, c) for c in df.columns]
        
        # Check required columns
        required = ['age', 'sex', 'creatinine', 'bilirubin', 'glucose', 'urine_volume', 'urine_ph']
        missing = [c for c in required if c not in df.columns]
        if missing:
            return False, f"Missing columns: {missing}", []
        
        # Create patient names
        patient_names = []
        for i, row in df.iterrows():
            if 'name' in df.columns:
                name = f"{i+1}. {row['name']}"
            elif 'patient_name' in df.columns:
                name = f"{i+1}. {row['patient_name']}"
            else:
                name = f"Patient {i+1}"
            patient_names.append(name)
        
        return True, df, patient_names
    
    except Exception as e:
        return False, f"Failed to load file: {str(e)}", []


def save_results(df, results, base_path):
    """Save batch results to CSV."""
    result_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    save_path = os.path.join(base_path, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    result_df.to_csv(save_path, index=False)
    return save_path


# ============== CT SCAN MODEL ==============

class CTPredictor:
    """Class to handle CT Scan cancer detection using Keras/TensorFlow model."""
    
    def __init__(self, model_path=None):
        """
        Initialize CT Predictor.
        
        Args:
            model_path: Path to the .keras model file
        """
        self.model = None
        self.model_path = model_path
        self.input_size = (224, 224)  # EfficientNetB2 default size
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load the Keras model.
        
        Args:
            model_path: Path to .keras model file
            
        Returns:
            tuple: (success, message)
        """
        try:
            if keras is None:
                return False, "Keras/TensorFlow not installed. Please install: pip install keras tensorflow"
            
            self.model = keras.models.load_model(model_path)
            self.model_path = model_path
            return True, f"CT Model loaded successfully from {os.path.basename(model_path)}"
        except Exception as e:
            self.model = None
            return False, f"Failed to load CT model: {str(e)}"
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for model input.
        
        Args:
            image_path: Path to image file (PNG, JPG, JPEG, DICOM)
            
        Returns:
            numpy array: Preprocessed image ready for prediction
        """
        try:
            # Check if DICOM file
            if image_path.lower().endswith('.dcm'):
                if pydicom is None:
                    raise ValueError("pydicom not installed. Install with: pip install pydicom")
                
                # Read DICOM file
                ds = pydicom.dcmread(image_path)
                img_array = ds.pixel_array
                
                # Normalize to 0-255
                img_array = img_array.astype(np.float32)
                img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255
                img_array = img_array.astype(np.uint8)
                
                # Convert to PIL Image and then to RGB
                img = Image.fromarray(img_array).convert('RGB')
            else:
                # Regular image (PNG, JPG, etc.)
                img = Image.open(image_path).convert('RGB')
            
            # Resize to model input size
            img = img.resize(self.input_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(img, dtype=np.float32)
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
        except Exception as e:
            raise ValueError(f"Failed to preprocess image: {str(e)}")
    
    def predict(self, image_path):
        """
        Make prediction for a CT scan image.
        
        Args:
            image_path: Path to CT scan image
            
        Returns:
            tuple: (is_cancer, confidence_percentage, raw_probability)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Preprocess image
            img_array = self.preprocess_image(image_path)
            
            # Make prediction
            prediction = self.model.predict(img_array, verbose=0)[0][0]
            
            # Convert to percentage
            confidence = float(prediction * 100)
            
            # Binary classification: >0.5 = Cancer, <=0.5 = Normal
            is_cancer = prediction > 0.5
            
            return is_cancer, confidence, float(prediction)
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")
    
    def predict_batch(self, image_paths):
        """
        Make predictions for multiple CT scan images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            list: List of dicts with results for each image
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        results = []
        for i, img_path in enumerate(image_paths):
            try:
                is_cancer, confidence, prob = self.predict(img_path)
                results.append({
                    "Image": os.path.basename(img_path),
                    "Diagnosis": "CANCER" if is_cancer else "Normal",
                    "Confidence": f"{confidence:.2f}%",
                    "Probability": f"{prob:.4f}"
                })
            except Exception as e:
                results.append({
                    "Image": os.path.basename(img_path),
                    "Diagnosis": "ERROR",
                    "Confidence": "N/A",
                    "Probability": str(e)
                })
        
        return results