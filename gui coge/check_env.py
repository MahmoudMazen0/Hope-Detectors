import sys
import os

print(f"Python Executable: {sys.executable}")
print("Attempting imports...")

try:
    import customtkinter
    print("customtkinter imported successfully")
    import pandas
    print("pandas imported successfully")
    import joblib
    print("joblib imported successfully")
    import sklearn
    print("sklearn imported successfully")
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    sys.exit(1)

print("Attempting to load models...")
base_path = os.getcwd()
model_path = os.path.join(base_path, "FINAL_MODEL_SVM.pkl")
scaler_path = os.path.join(base_path, "FINAL_SCALER.pkl")

if not os.path.exists(model_path):
    print(f"ERROR: Model file not found at {model_path}")
elif not os.path.exists(scaler_path):
    print(f"ERROR: Scaler file not found at {scaler_path}")
else:
    try:
        joblib.load(model_path)
        print("Model loaded successfully")
        joblib.load(scaler_path)
        print("Scaler loaded successfully")
    except Exception as e:
        print(f"MODEL LOAD ERROR: {e}")
        sys.exit(1)

print("VERIFICATION COMPLETE: SYSTEM READY")
