"""
Medical Diagnostics Dashboard - GUI Module
Uses backend.py for ML logic and predictions.
"""

import os
import sys
import subprocess

# =========================================================================
# AUTO-RELAUNCH WITH CORRECT PYTHON ENVIRONMENT
# This ensures binary compatibility (numpy, etc.) by enforcing venv usage.
# =========================================================================
def ensure_correct_environment():
    """Relaunch script with venv python if running with incorrect interpreter."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Locate venv python - check in project root (parent of src)
    project_root = os.path.dirname(current_dir)  # Go up from src/ to project root
    venv_python = os.path.join(project_root, 'venv311', 'Scripts', 'python.exe')
    if not os.path.exists(venv_python):
        # Check current directory
        venv_python = os.path.join(current_dir, 'venv311', 'Scripts', 'python.exe')
    
    # Use venv if found
    if os.path.exists(venv_python):
        # Normalize for case-insensitive comparison
        running_python = os.path.abspath(sys.executable).lower()
        target_python = os.path.abspath(venv_python).lower()
        
        # If mismatch, relaunch!
        if running_python != target_python:
            print(f"[!] Environment Mismatch Detected!")
            print(f"Running: {running_python}")
            print(f"Target:  {target_python}")
            print(f"[*] Relaunching with correct environment...")
            
            # Execute and wait
            ret_code = subprocess.call([target_python, __file__] + sys.argv[1:])
            sys.exit(ret_code)

# Run check immediately
ensure_correct_environment()

import customtkinter as ctk
from tkinter import messagebox
from PIL import Image, ImageTk
from datetime import datetime

# Import backend (same directory)
from backend import MedicalPredictor, AVAILABLE_MODELS, load_patient_file, save_results, CTPredictor, AnalysisHistory

# Configuration
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# ============== FUTURISTIC COLOR PALETTE ==============
COLORS = {
    # Backgrounds
    "bg_main": "#0B1120",
    "bg_card": "#111827",
    "bg_header": "#0F172A",
    
    # Accent/Glow Colors
    "glow_cyan": "#06B6D4",
    "glow_blue": "#3B82F6",
    "primary": "#3B82F6",
    "primary_hover": "#2563EB",
    
    # Text
    "text_primary": "#F1F5F9",
    "text_secondary": "#94A3B8",
    "text_title": "#38BDF8",
    
    # Status
    "success": "#10B981",
    "danger": "#EF4444",
    "warning": "#F59E0B",
}


class MedicalDashboardApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Medical Diagnostics Dashboard")
        self.geometry("1400x800")
        self.configure(fg_color=COLORS["bg_main"])
        
        # Get script directory and project root for paths
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            # Running in PyInstaller bundle
            self.project_root = sys._MEIPASS
        else:
            # Running in normal Python environment
            self.script_dir = os.path.dirname(os.path.abspath(__file__))
            self.project_root = os.path.dirname(self.script_dir)  # Go up from src/ to project root
        self.assets_dir = os.path.join(self.project_root, "assets")
        
        # Backend predictor
        self.predictor = MedicalPredictor()
        
        # CT Predictor - use model from models/ct_scans folder
        ct_model_path = os.path.join(self.project_root, "models", "ct_scans", "final_model.keras")
        
        
        self.ct_predictor = CTPredictor(ct_model_path)
        

        
        # UI Variables
        self.selected_model_name = ctk.StringVar(value="SVM (Best)")
        self.sex_var = ctk.StringVar(value="Male")
        self.result_text = ctk.StringVar(value="Awaiting Analysis...")
        
        # CT Variables
        self.ct_image_path = None
        self.ct_result_text = ctk.StringVar(value="Upload CT Scan Image...")
        self.ct_confidence_text = ctk.StringVar(value="—")
        
        # Data storage
        self.loaded_patients_df = None
        self.patient_names = []
        
        # Animation state
        self.animation_id = None
        
        # History manager
        self.history_manager = AnalysisHistory()
        
        # Show selection page directly (no splash screen)
        self.create_selection_page()
    
    # ============== SELECTION PAGE ==============
    def create_selection_page(self):
        """Create the main selection page with CT Scans and Lab Tests options."""
        self.selection_frame = ctk.CTkFrame(self, fg_color=COLORS["bg_main"])
        self.selection_frame.pack(fill="both", expand=True)
        
        # Add background image - covers entire frame
        bg_path = os.path.join(self.assets_dir, "page_background.png")
        if os.path.exists(bg_path):
            self.update()
            frame_width = self.winfo_width() or 1400
            frame_height = self.winfo_height() or 800
            
            bg_image = Image.open(bg_path).convert("RGBA")
            bg_image = bg_image.resize((frame_width, frame_height), Image.Resampling.LANCZOS)
            
            # Reduce opacity to 40% for better card visibility
            dark_overlay = Image.new("RGBA", bg_image.size, (11, 17, 32, 153))  # #0B1120 with 60% opacity overlay
            bg_image = Image.alpha_composite(bg_image, dark_overlay)
            
            self.page_bg_photo = ImageTk.PhotoImage(bg_image)
            
            bg_label = ctk.CTkLabel(
                self.selection_frame,
                image=self.page_bg_photo,
                text=""
            )
            bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        # Title frame with logo at top - using place to overlay on background
        title_frame = ctk.CTkFrame(self.selection_frame, fg_color=COLORS["bg_header"], height=90, corner_radius=0)
        title_frame.place(x=0, y=0, relwidth=1)
        
        # Add logo to header
        logo_path = os.path.join(self.assets_dir, "logo.png")
        if os.path.exists(logo_path):
            header_logo = Image.open(logo_path)
            header_logo = header_logo.resize((65, 65), Image.Resampling.LANCZOS)
            self.header_logo_photo = ImageTk.PhotoImage(header_logo)
            
            logo_label = ctk.CTkLabel(
                title_frame,
                image=self.header_logo_photo,
                text=""
            )
            logo_label.pack(side="left", padx=20, pady=12)
        
        title = ctk.CTkLabel(
            title_frame,
            text="Hope Detectors - Medical Diagnostics",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=COLORS["text_title"]
        )
        title.pack(side="left", padx=10, pady=25)
        
        # Subtitle - placed directly on background
        subtitle = ctk.CTkLabel(
            self.selection_frame,
            text="Choose your diagnostic module",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#FFFFFF",
            fg_color="transparent"
        )
        subtitle.place(relx=0.5, y=140, anchor="center")
        
        # Cards Container - placed directly on background (no frame wrapper)

        # CT Scans Card - placed directly on background
        ct_card = ctk.CTkFrame(
            self.selection_frame,
            fg_color=("#0a0f1a", "#0a0f1a"),  # Dark semi-transparent look
            corner_radius=20,
            border_width=0,
            width=340,
            height=400
        )
        ct_card.place(relx=0.32, rely=0.55, anchor="center")
        ct_card.pack_propagate(False)
        
        # CT Icon - Large
        ct_icon_label = ctk.CTkLabel(
            ct_card, 
            text="🔬",  # CT Scanner icon
            font=ctk.CTkFont(size=80)
        )
        ct_icon_label.pack(pady=(40, 15))
        
        ctk.CTkLabel(
            ct_card, 
            text="CT Scans", 
            font=ctk.CTkFont(size=28, weight="bold"), 
            text_color=COLORS["glow_cyan"]
        ).pack(pady=8)
        
        ctk.CTkLabel(
            ct_card, 
            text="Cancer Detection from\nCT Scan Images", 
            text_color=COLORS["text_secondary"], 
            font=ctk.CTkFont(size=14),
            justify="center"
        ).pack(pady=10)
        
        ctk.CTkButton(
            ct_card,
            text="Open CT Module",
            command=self.open_ct_scans,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=COLORS["glow_cyan"],
            hover_color=COLORS["glow_blue"],
            text_color="#0B1120",
            height=50,
            width=250,
            corner_radius=12
        ).pack(pady=(25, 30))
        
        # Lab Tests Card - placed directly on background
        lab_card = ctk.CTkFrame(
            self.selection_frame,
            fg_color=("#0a0f1a", "#0a0f1a"),  # Dark semi-transparent look
            corner_radius=20,
            border_width=0,
            width=340,
            height=400
        )
        lab_card.place(relx=0.68, rely=0.55, anchor="center")
        lab_card.pack_propagate(False)
        
        # Lab Icon - Large
        lab_icon_label = ctk.CTkLabel(
            lab_card, 
            text="�",  # DNA/Lab icon
            font=ctk.CTkFont(size=80)
        )
        lab_icon_label.pack(pady=(40, 15))
        
        ctk.CTkLabel(
            lab_card, 
            text="Lab Tests", 
            font=ctk.CTkFont(size=28, weight="bold"), 
            text_color=COLORS["glow_blue"]
        ).pack(pady=8)
        
        ctk.CTkLabel(
            lab_card, 
            text="Pancreatic Cancer Detection\nfrom Blood & Urine Tests", 
            text_color=COLORS["text_secondary"], 
            font=ctk.CTkFont(size=14),
            justify="center"
        ).pack(pady=10)
        
        ctk.CTkButton(
            lab_card,
            text="Open Lab Module",
            command=self.open_lab_tests,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=COLORS["glow_blue"],
            hover_color=COLORS["primary_hover"],
            text_color="#FFFFFF",
            height=50,
            width=250,
            corner_radius=12
        ).pack(pady=(25, 30))
    
    def open_ct_scans(self):
        """Navigate to CT Scans dashboard."""
        self.selection_frame.destroy()
        self.create_ct_header()
        self.create_ct_dashboard()
    
    def open_lab_tests(self):
        """Navigate to Lab Tests dashboard."""
        self.selection_frame.destroy()
        
        # Load model using backend
        success, message = self.predictor.load_model(self.selected_model_name.get())
        if not success:
            messagebox.showerror("Error", message)
        
        self.create_header()
        self.create_dashboard()

    # ============== CT SCANS MODULE ==============
    def create_ct_header(self):
        self.ct_header = ctk.CTkFrame(self, fg_color=COLORS["bg_header"], height=80, corner_radius=0)
        self.ct_header.pack(fill="x")
        self.ct_header.pack_propagate(False)
        
        # Add logo
        logo_path = os.path.join(self.assets_dir, "logo.png")
        if os.path.exists(logo_path):
            header_logo = Image.open(logo_path)
            header_logo = header_logo.resize((60, 60), Image.Resampling.LANCZOS)
            self.ct_header_logo_photo = ImageTk.PhotoImage(header_logo)
            
            logo_label = ctk.CTkLabel(
                self.ct_header,
                image=self.ct_header_logo_photo,
                text=""
            )
            logo_label.pack(side="left", padx=15, pady=10)
        
        back_btn = ctk.CTkButton(
            self.ct_header, text="← Back", command=self.go_back_from_ct,
            width=100, height=40, font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="transparent", border_width=2, border_color=COLORS["glow_cyan"],
            text_color=COLORS["text_primary"], hover_color=COLORS["bg_card"], corner_radius=10
        )
        back_btn.pack(side="left", padx=10, pady=20)
        
        title = ctk.CTkLabel(self.ct_header, text="CT Scan Cancer Detection", 
                             font=ctk.CTkFont(size=24, weight="bold"), text_color=COLORS["text_title"])
        title.pack(side="left", padx=10, pady=20)
    
    def go_back_from_ct(self):
        if hasattr(self, 'ct_header'):
            self.ct_header.destroy()
        if hasattr(self, 'ct_dashboard'):
            self.ct_dashboard.destroy()
        self.create_selection_page()
    
    def create_ct_dashboard(self):
        """Create complete CT Scan dashboard with image upload and analysis."""
        self.ct_dashboard = ctk.CTkFrame(self, fg_color=COLORS["bg_main"])
        self.ct_dashboard.pack(fill="both", expand=True)
        
        # Add background image
        bg_path = os.path.join(self.assets_dir, "page_background.png")
        if os.path.exists(bg_path):
            self.update()
            frame_width = self.winfo_width() or 1400
            frame_height = self.winfo_height() or 800
            
            bg_image = Image.open(bg_path).convert("RGBA")
            bg_image = bg_image.resize((frame_width, frame_height), Image.Resampling.LANCZOS)
            
            # Reduce opacity to 40% for better card visibility
            dark_overlay = Image.new("RGBA", bg_image.size, (11, 17, 32, 153))  # #0B1120 with 60% opacity overlay
            bg_image = Image.alpha_composite(bg_image, dark_overlay)
            
            self.ct_bg_photo = ImageTk.PhotoImage(bg_image)
            
            bg_label = ctk.CTkLabel(
                self.ct_dashboard,
                image=self.ct_bg_photo,
                text=""
            )
            bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        # Card 1: Image Upload & Preview (top-left)
        upload_card = ctk.CTkFrame(
            self.ct_dashboard, 
            fg_color=("#0a1628", "#0a1628"),
            corner_radius=20, border_width=0,
            width=400, height=280
        )
        upload_card.place(relx=0.26, rely=0.32, anchor="center")
        upload_card.pack_propagate(False)
        
        ctk.CTkLabel(upload_card, text="📤 Upload CT Scan", font=ctk.CTkFont(size=18, weight="bold"),
                     text_color="#40C4E0").pack(anchor="w", padx=20, pady=(18, 12))
        
        upload_content = ctk.CTkFrame(upload_card, fg_color="transparent")
        upload_content.pack(fill="both", expand=True, padx=20, pady=10)
        
        ctk.CTkButton(upload_content, text="📁 Select CT Scan Image", command=self.upload_ct_image,
                      height=45, font=ctk.CTkFont(size=14, weight="bold"),
                      fg_color="#40C4E0", hover_color=COLORS["glow_blue"],
                      text_color=COLORS["bg_main"], corner_radius=10).pack(pady=8, fill="x")
        
        self.ct_image_label = ctk.CTkLabel(upload_content, text="No image loaded",
                                            font=ctk.CTkFont(size=11), text_color=COLORS["text_secondary"])
        self.ct_image_label.pack(pady=3)
        
        self.ct_preview_frame = ctk.CTkFrame(upload_content, fg_color=COLORS["bg_main"],
                                              corner_radius=10, height=100)
        self.ct_preview_frame.pack(pady=5, fill="x")
        
        # Card 2: Analysis Controls (top-right)
        control_card = ctk.CTkFrame(
            self.ct_dashboard, fg_color=("#0a1628", "#0a1628"),
            corner_radius=20, border_width=0,
            width=400, height=280
        )
        control_card.place(relx=0.74, rely=0.32, anchor="center")
        control_card.pack_propagate(False)
        
        ctk.CTkLabel(control_card, text="⚙️ Analysis Controls", font=ctk.CTkFont(size=18, weight="bold"),
                     text_color="#40C4E0").pack(anchor="w", padx=20, pady=(18, 12))
        
        control_content = ctk.CTkFrame(control_card, fg_color="transparent")
        control_content.pack(fill="both", expand=True, padx=20, pady=10)
        
        model_status = "✅ Model Loaded" if self.ct_predictor.model else "❌ Model Not Loaded"
        status_color = COLORS["success"] if self.ct_predictor.model else COLORS["danger"]
        
        ctk.CTkLabel(control_content, text="Model Status:", text_color=COLORS["text_secondary"],
                     font=ctk.CTkFont(size=11)).pack(anchor="w", pady=(5, 2))
        ctk.CTkLabel(control_content, text=model_status, text_color=status_color,
                     font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", pady=(0, 15))
        
        ctk.CTkButton(control_content, text="🔬 Analyze CT Scan", command=self.analyze_ct_image,
                      height=45, font=ctk.CTkFont(size=14, weight="bold"),
                      fg_color=COLORS["primary"], hover_color=COLORS["primary_hover"],
                      corner_radius=10).pack(fill="x", pady=8)
        
        ctk.CTkButton(control_content, text="🗑️ Clear Image", command=self.clear_ct_image,
                      height=35, fg_color="transparent", border_width=2,
                      border_color="#40C4E0", text_color=COLORS["text_primary"],
                      corner_radius=10).pack(fill="x", pady=5)
        
        # Card 3: Results (bottom-left)
        result_card = ctk.CTkFrame(
            self.ct_dashboard, fg_color=("#0a1628", "#0a1628"),
            corner_radius=20, border_width=0,
            width=400, height=260
        )
        result_card.place(relx=0.26, rely=0.74, anchor="center")
        result_card.pack_propagate(False)
        
        ctk.CTkLabel(result_card, text="📊 Analysis Results", font=ctk.CTkFont(size=18, weight="bold"),
                     text_color="#40C4E0").pack(anchor="w", padx=20, pady=(18, 12))
        
        result_content = ctk.CTkFrame(result_card, fg_color="transparent")
        result_content.pack(fill="both", expand=True, padx=20, pady=10)
        
        ctk.CTkLabel(result_content, text="Diagnosis:", text_color=COLORS["text_secondary"],
                     font=ctk.CTkFont(size=12)).pack(pady=(10, 3))
        
        self.ct_result_label = ctk.CTkLabel(result_content, textvariable=self.ct_result_text,
                                            font=ctk.CTkFont(size=28, weight="bold"),
                                            text_color=COLORS["text_primary"])
        self.ct_result_label.pack(pady=5)
        
        ctk.CTkLabel(result_content, text="Confidence:", text_color=COLORS["text_secondary"],
                     font=ctk.CTkFont(size=12)).pack(pady=(8, 3))
        
        self.ct_confidence_label = ctk.CTkLabel(result_content, textvariable=self.ct_confidence_text,
                                                 font=ctk.CTkFont(size=22, weight="bold"),
                                                 text_color="#40C4E0")
        self.ct_confidence_label.pack(pady=3)
        
        # Card 4: Batch Processing (bottom-right)
        batch_card = ctk.CTkFrame(
            self.ct_dashboard, fg_color=("#0a1628", "#0a1628"),
            corner_radius=20, border_width=0,
            width=400, height=260
        )
        batch_card.place(relx=0.74, rely=0.74, anchor="center")
        batch_card.pack_propagate(False)
        
        ctk.CTkLabel(batch_card, text="📂 Batch Processing", font=ctk.CTkFont(size=18, weight="bold"),
                     text_color="#40C4E0").pack(anchor="w", padx=20, pady=(18, 12))
        
        batch_content = ctk.CTkScrollableFrame(batch_card, fg_color="transparent")
        batch_content.pack(fill="both", expand=True, padx=20, pady=10)
        
        ctk.CTkLabel(batch_content, text="Upload multiple CT scans", text_color=COLORS["text_secondary"],
                     font=ctk.CTkFont(size=11)).pack(anchor="w", pady=3)
        
        ctk.CTkButton(batch_content, text="📁 Select Multiple Images", command=self.upload_ct_batch,
                      height=35, font=ctk.CTkFont(size=12, weight="bold"),
                      fg_color=COLORS["glow_blue"], hover_color=COLORS["primary"],
                      corner_radius=10).pack(fill="x", pady=4)
        
        ctk.CTkButton(batch_content, text="📂 Select DICOM Folder", command=self.upload_ct_folder,
                      height=35, font=ctk.CTkFont(size=12, weight="bold"),
                      fg_color="#40C4E0", hover_color=COLORS["glow_blue"],
                      text_color=COLORS["bg_main"], corner_radius=10).pack(fill="x", pady=4)
        
        self.ct_batch_status = ctk.CTkLabel(batch_content, text="No files selected",
                                            text_color=COLORS["text_secondary"], wraplength=200)
        self.ct_batch_status.pack(pady=5)

    # ============== LAB TESTS MODULE ==============
    def create_header(self):
        self.header = ctk.CTkFrame(self, fg_color=COLORS["bg_header"], height=80, corner_radius=0)
        self.header.pack(fill="x")
        self.header.pack_propagate(False)
        
        # Add logo
        logo_path = os.path.join(self.assets_dir, "logo.png")
        if os.path.exists(logo_path):
            header_logo = Image.open(logo_path)
            header_logo = header_logo.resize((60, 60), Image.Resampling.LANCZOS)
            self.lab_header_logo_photo = ImageTk.PhotoImage(header_logo)
            
            logo_label = ctk.CTkLabel(
                self.header,
                image=self.lab_header_logo_photo,
                text=""
            )
            logo_label.pack(side="left", padx=15, pady=10)
        
        back_btn = ctk.CTkButton(
            self.header, text="← Back", command=self.go_back_to_selection,
            width=100, height=40, font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="transparent", border_width=2, border_color=COLORS["glow_cyan"],
            text_color=COLORS["text_primary"], hover_color=COLORS["bg_card"], corner_radius=10
        )
        back_btn.pack(side="left", padx=10, pady=20)
        
        title = ctk.CTkLabel(self.header, text="Medical Diagnostics Dashboard", 
                             font=ctk.CTkFont(size=24, weight="bold"), text_color=COLORS["text_title"])
        title.pack(side="left", padx=10, pady=20)
        
        # History button
        history_btn = ctk.CTkButton(
            self.header, text="📋 History", command=self.show_history_popup,
            width=120, height=40, font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=COLORS["glow_blue"], hover_color=COLORS["primary_hover"],
            text_color="white", corner_radius=10
        )
        history_btn.pack(side="right", padx=15, pady=20)
    
    def go_back_to_selection(self):
        if hasattr(self, 'header'):
            self.header.destroy()
        if hasattr(self, 'dashboard'):
            self.dashboard.destroy()
        self.create_selection_page()

    def create_dashboard(self):
        self.dashboard = ctk.CTkFrame(self, fg_color="transparent")
        self.dashboard.pack(fill="both", expand=True)
        
        # Add background image
        bg_path = os.path.join(self.assets_dir, "page_background.png")
        if os.path.exists(bg_path):
            self.update()
            frame_width = self.winfo_width() or 1400
            frame_height = self.winfo_height() or 800
            
            bg_image = Image.open(bg_path).convert("RGBA")
            bg_image = bg_image.resize((frame_width, frame_height), Image.Resampling.LANCZOS)
            
            # Reduce opacity to 40% for better card visibility
            dark_overlay = Image.new("RGBA", bg_image.size, (11, 17, 32, 153))  # #0B1120 with 60% opacity overlay
            bg_image = Image.alpha_composite(bg_image, dark_overlay)
            
            self.lab_bg_photo = ImageTk.PhotoImage(bg_image)
            
            bg_label = ctk.CTkLabel(
                self.dashboard,
                image=self.lab_bg_photo,
                text=""
            )
            bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        # Cards placed directly on dashboard (no intermediate frame)
        self.create_patient_input_card()
        self.create_model_settings_card()
        self.create_result_card()
        self.create_batch_card()

    def create_lab_card(self, title, relx, rely, width=400, height=280, border_color=None):
        """Create a professional glassmorphism card."""
        # No borders - background shows through
        
        card = ctk.CTkFrame(
            self.dashboard, 
            fg_color=("#0a1628", "#0a1628"),  # Dark blue transparent
            corner_radius=20,
            border_width=0,  # No border
            width=width,
            height=height
        )
        card.place(relx=relx, rely=rely, anchor="center")
        card.pack_propagate(False)
        
        # Title with icon
        title_label = ctk.CTkLabel(
            card, 
            text=title, 
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#40C4E0"  # Cyan title
        )
        title_label.pack(anchor="w", padx=20, pady=(18, 12))
        return card

    # ============== CARD 1: PATIENT INPUT ==============
    def create_patient_input_card(self):
        card = self.create_lab_card("📋 Patient Data Input", relx=0.26, rely=0.32, height=280)
        
        input_frame = ctk.CTkScrollableFrame(card, fg_color="transparent")
        input_frame.pack(fill="both", expand=True, padx=15, pady=10)
        
        self.entries = {}
        self.entry_list = []
        
        fields = [
            ("Age (years)", "age"),
            ("Serum Creatinine", "creatinine"),
            ("Total Bilirubin", "bilirubin"),
            ("Serum Glucose", "glucose"),
            ("Urine Volume (mL)", "urine_volume"),
            ("Urine pH", "urine_pH"),
        ]
        
        # Sex Dropdown
        ctk.CTkLabel(input_frame, text="Sex", text_color=COLORS["text_secondary"]).pack(anchor="w", pady=(5, 2))
        ctk.CTkComboBox(input_frame, values=["Male", "Female"], variable=self.sex_var, width=200).pack(anchor="w", pady=(0, 10))
        
        for label, key in fields:
            ctk.CTkLabel(input_frame, text=label, text_color=COLORS["text_secondary"]).pack(anchor="w", pady=(5, 2))
            entry = ctk.CTkEntry(input_frame, width=200, fg_color=COLORS["bg_main"], border_color=COLORS["glow_cyan"])
            entry.pack(anchor="w", pady=(0, 5))
            self.entries[key] = entry
            self.entry_list.append(entry)
        
        # Bind Enter key
        for i, entry in enumerate(self.entry_list):
            if i < len(self.entry_list) - 1:
                entry.bind("<Return>", lambda e, next_entry=self.entry_list[i+1]: next_entry.focus_set())
            else:
                entry.bind("<Return>", lambda e: self.predict())

    # ============== CARD 2: MODEL SETTINGS ==============
    def create_model_settings_card(self):
        card = self.create_lab_card("⚙️ Model Settings", relx=0.74, rely=0.32, height=280)
        content = ctk.CTkFrame(card, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=10)
        
        ctk.CTkLabel(content, text="Select Model:", text_color=COLORS["text_secondary"]).pack(anchor="w", pady=(10, 5))
        model_combo = ctk.CTkComboBox(
            content, values=list(AVAILABLE_MODELS.keys()), variable=self.selected_model_name,
            command=self.on_model_change, width=250, fg_color=COLORS["bg_main"], border_color=COLORS["glow_blue"]
        )
        model_combo.pack(anchor="w", pady=(0, 20))
        
        ctk.CTkButton(
            content, text="🔬 Analyze Patient", command=self.predict,
            height=50, font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=COLORS["primary"], hover_color=COLORS["primary_hover"], corner_radius=10
        ).pack(fill="x", pady=10)
        
        ctk.CTkButton(
            content, text="🗑️ Clear Form", command=self.clear_form,
            height=40, fg_color="transparent", border_width=2, border_color=COLORS["glow_cyan"],
            text_color=COLORS["text_primary"], corner_radius=10
        ).pack(fill="x", pady=10)

    # ============== CARD 3: RESULT DISPLAY ==============
    def create_result_card(self):
        card = self.create_lab_card("📊 Analysis Result", relx=0.26, rely=0.74, height=260)
        content = ctk.CTkFrame(card, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=10)
        
        ctk.CTkLabel(content, text="Status:", text_color=COLORS["text_secondary"], 
                     font=ctk.CTkFont(size=13)).pack(pady=(10, 3))
        
        self.result_label = ctk.CTkLabel(
            content, textvariable=self.result_text,
            font=ctk.CTkFont(size=26, weight="bold"), text_color=COLORS["text_primary"]
        )
        self.result_label.pack(pady=5)
        
        # Confidence display
        ctk.CTkLabel(content, text="Confidence:", text_color=COLORS["text_secondary"], 
                     font=ctk.CTkFont(size=13)).pack(pady=(8, 3))
        
        self.confidence_text = ctk.StringVar(value="—")
        self.confidence_label = ctk.CTkLabel(
            content, textvariable=self.confidence_text,
            font=ctk.CTkFont(size=22, weight="bold"), text_color=COLORS["glow_cyan"]
        )
        self.confidence_label.pack(pady=3)

    # ============== CARD 4: BATCH PROCESSING ==============
    def create_batch_card(self):
        card = self.create_lab_card("📂 Batch Processing", relx=0.74, rely=0.74, height=260)
        content = ctk.CTkScrollableFrame(card, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=10)
        
        # UPLOAD PATIENT DATA
        ctk.CTkLabel(content, text="📊 Upload Patient Data (CSV/Excel):", 
                     text_color=COLORS["text_title"], font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", pady=(5, 3))
        ctk.CTkLabel(content, text="Raw values: Age=50, Glucose=100, etc.", 
                     text_color=COLORS["text_secondary"], font=ctk.CTkFont(size=11)).pack(anchor="w")
        
        ctk.CTkButton(
            content, text="📤 Upload Patient Data", command=self.upload_patients,
            height=40, font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=COLORS["glow_cyan"], hover_color=COLORS["glow_blue"],
            text_color=COLORS["bg_main"], corner_radius=10
        ).pack(fill="x", pady=(5, 15))
        
        # PATIENT SELECTION - opens popup
        ctk.CTkLabel(content, text="Patient:", text_color=COLORS["text_secondary"]).pack(anchor="w", pady=(10, 5))
        
        self.patient_var = ctk.StringVar(value="No patient selected")
        self.selected_patient_idx = 0
        self.patient_buttons = []
        
        # Show selected patient name
        self.selected_patient_label = ctk.CTkLabel(
            content,
            textvariable=self.patient_var,
            text_color=COLORS["text_title"],
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.selected_patient_label.pack(anchor="w", pady=(0, 5))
        
        ctk.CTkButton(
            content, text="👤 Select Patient", command=self.open_patient_selection,
            height=35, font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=COLORS["glow_blue"], hover_color=COLORS["primary"],
            text_color=COLORS["text_primary"], corner_radius=10
        ).pack(fill="x", pady=5)
        
        ctk.CTkButton(
            content, text="📋 Fill Form with Patient", command=self.fill_form_with_patient,
            height=35, fg_color=COLORS["primary"], hover_color=COLORS["primary_hover"], corner_radius=10
        ).pack(fill="x", pady=5)
        
        ctk.CTkButton(
            content, text="🔬 Analyze All Patients", command=self.analyze_all_patients,
            height=35, fg_color=COLORS["success"], hover_color=COLORS["glow_blue"], corner_radius=10
        ).pack(fill="x", pady=5)
        
        self.batch_status = ctk.CTkLabel(content, text="No file loaded.", 
                                          text_color=COLORS["text_secondary"], wraplength=250)
        self.batch_status.pack(pady=10)

    # ============== LOGIC ==============
    def on_model_change(self, choice):
        success, message = self.predictor.load_model(choice)
        if not success:
            messagebox.showerror("Error", message)
    
    def clear_form(self):
        for entry in self.entries.values():
            entry.delete(0, 'end')
        self.sex_var.set("Male")
        self.result_text.set("Awaiting Analysis...")
        self.result_label.configure(text_color=COLORS["text_primary"])

    def get_float(self, key):
        val = self.entries[key].get()
        if not val:
            raise ValueError(f"Missing: {key}")
        return float(val)

    def predict(self):
        try:
            age = self.get_float("age")
            sex = 1 if self.sex_var.get() == "Male" else 0
            creat = self.get_float("creatinine")
            bili = self.get_float("bilirubin")
            glucose = self.get_float("glucose")
            u_vol = self.get_float("urine_volume")
            u_ph = self.get_float("urine_pH")

            prediction, is_cancer, confidence = self.predictor.predict(age, sex, creat, bili, glucose, u_vol, u_ph)

            if is_cancer:
                self.result_text.set("CANCER DETECTED")
                self.result_label.configure(text_color=COLORS["danger"])
                diagnosis = "CANCER"
            else:
                self.result_text.set("NORMAL")
                self.result_label.configure(text_color=COLORS["success"])
                diagnosis = "Healthy"
            
            # Display confidence
            self.confidence_text.set(f"{confidence:.1f}%")
            
            # Save to history
            patient_name = self.patient_var.get() if hasattr(self, 'patient_var') and self.patient_var.get() else "Manual Entry"
            self.history_manager.add_record(
                patient_name=patient_name,
                analysis_type="Lab Test",
                model_used=self.selected_model_name.get(),
                diagnosis=diagnosis,
                confidence=confidence,
                input_data={"age": age, "sex": sex, "creatinine": creat, "bilirubin": bili, 
                           "glucose": glucose, "urine_volume": u_vol, "urine_pH": u_ph}
            )

        except ValueError as ve:
            messagebox.showwarning("Input Error", str(ve))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def upload_patients(self):
        """Upload patient file (raw data only)."""
        file_path = ctk.filedialog.askopenfilename(filetypes=[("Excel/CSV", "*.xlsx *.xls *.csv")])
        if not file_path:
            return
        
        success, result, patient_names = load_patient_file(file_path)
        if not success:
            messagebox.showerror("Error", result)
            return
        
        self.loaded_patients_df = result
        self.patient_names = patient_names
        self.selected_patient_idx = 0
        
        if patient_names:
            self.patient_var.set(patient_names[0])
        else:
            self.patient_var.set("No patients")
        
        self.batch_status.configure(text=f"✅ Loaded {len(result)} patients")
    
    def open_patient_selection(self):
        """Open popup window to select patient."""
        if not self.patient_names:
            messagebox.showwarning("No Data", "Please upload patient data first.")
            return
        
        # Create popup window
        popup = ctk.CTkToplevel(self)
        popup.title("Select Patient")
        popup.geometry("400x500")
        popup.configure(fg_color=COLORS["bg_main"])
        popup.transient(self)
        popup.grab_set()
        
        # Center popup
        popup.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - 400) // 2
        y = self.winfo_y() + (self.winfo_height() - 500) // 2
        popup.geometry(f"400x500+{x}+{y}")
        
        # Title
        ctk.CTkLabel(
            popup,
            text="👤 Select Patient",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=COLORS["text_title"]
        ).pack(pady=(20, 10))
        
        # Scrollable patient list
        list_frame = ctk.CTkScrollableFrame(
            popup,
            fg_color=COLORS["bg_card"],
            corner_radius=10,
            border_width=2,
            border_color=COLORS["glow_blue"]
        )
        list_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Create button for each patient
        for i, name in enumerate(self.patient_names):
            btn = ctk.CTkButton(
                list_frame,
                text=name,
                command=lambda idx=i, p=popup: self.select_patient_from_popup(idx, p),
                height=35,
                font=ctk.CTkFont(size=13),
                fg_color=COLORS["glow_blue"] if i == self.selected_patient_idx else "transparent",
                hover_color=COLORS["glow_cyan"],
                text_color=COLORS["text_primary"],
                corner_radius=8,
                anchor="w"
            )
            btn.pack(fill="x", pady=2, padx=5)
        
        # Cancel button
        ctk.CTkButton(
            popup,
            text="Cancel",
            command=popup.destroy,
            height=40,
            fg_color="transparent",
            border_width=2,
            border_color=COLORS["glow_cyan"],
            text_color=COLORS["text_primary"],
            corner_radius=10
        ).pack(pady=15, padx=20, fill="x")
    
    def select_patient_from_popup(self, idx, popup):
        """Select patient and close popup."""
        self.selected_patient_idx = idx
        self.patient_var.set(self.patient_names[idx])
        popup.destroy()
        self.batch_status.configure(text=f"👤 Selected: {self.patient_names[idx]}")
    
    def select_patient(self, idx):
        """Select a patient (legacy method)."""
        self.selected_patient_idx = idx
        self.patient_var.set(self.patient_names[idx])
    


    def fill_form_with_patient(self):
        """Fill form with selected patient."""
        if self.loaded_patients_df is None:
            messagebox.showwarning("No Data", "Please upload a file first.")
            return
        
        if not self.patient_names:
            return
        
        try:
            idx = self.selected_patient_idx
            row = self.loaded_patients_df.iloc[idx]
            
            self.clear_form()
            self.entries["age"].insert(0, str(row['age']))
            self.entries["creatinine"].insert(0, str(row['creatinine']))
            self.entries["bilirubin"].insert(0, str(row['bilirubin']))
            self.entries["glucose"].insert(0, str(row['glucose']))
            self.entries["urine_volume"].insert(0, str(row['urine_volume']))
            self.entries["urine_pH"].insert(0, str(row['urine_ph']))
            
            sex_val = str(row['sex']).strip()
            self.sex_var.set("Male" if sex_val in ['1', '1.0', 'male', 'm', 'Male'] else "Female")
            self.batch_status.configure(text=f"📋 Loaded: {self.patient_names[idx]}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def analyze_all_patients(self):
        """Analyze all loaded patients."""
        if self.loaded_patients_df is None:
            messagebox.showwarning("No Data", "Please upload a file first.")
            return
        
        try:
            results = self.predictor.predict_batch(self.loaded_patients_df, skip_normalization=False)
            save_path = save_results(self.loaded_patients_df, results, self.predictor.base_path)
            
            cancer_count = sum(1 for r in results if r["Diagnosis"] == "CANCER")
            self.batch_status.configure(text=f"✅ Analyzed {len(results)}\nCancer: {cancer_count}\nSaved!")
            messagebox.showinfo("Done", f"Results saved to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Batch Error", str(e))

    # ============== CT SCAN METHODS ==============
    
    def upload_ct_image(self):
        """Upload a single CT scan image."""
        file_path = ctk.filedialog.askopenfilename(
            filetypes=[
                ("Medical Images", "*.dcm *.png *.jpg *.jpeg *.bmp"),
                ("DICOM Files", "*.dcm"),
                ("Image Files", "*.png *.jpg *.jpeg *.bmp"),
                ("All Files", "*.*")
            ]
        )
        if not file_path:
            return
        
        try:
            # Store image path
            self.ct_image_path = file_path
            
            # Update label
            filename = os.path.basename(file_path)
            self.ct_image_label.configure(text=f"📄 {filename}")
            
            # Show preview
            from PIL import Image, ImageTk
            import numpy as np
            
            # Handle DICOM files
            if file_path.lower().endswith('.dcm'):
                try:
                    import pydicom
                    ds = pydicom.dcmread(file_path)
                    img_array = ds.pixel_array
                    
                    # Normalize to 0-255
                    img_array = img_array.astype(np.float32)
                    img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255
                    img_array = img_array.astype(np.uint8)
                    
                    img = Image.fromarray(img_array).convert('RGB')
                except Exception as e:
                    messagebox.showerror("DICOM Error", f"Failed to read DICOM: {str(e)}")
                    return
            else:
                img = Image.open(file_path)
            
            # Resize for preview (maintain aspect ratio)
            max_size = (300, 300)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Clear previous preview
            for widget in self.ct_preview_frame.winfo_children():
                widget.destroy()
            
            # Show image
            preview_label = ctk.CTkLabel(
                self.ct_preview_frame,
                image=photo,
                text=""
            )
            preview_label.image = photo  # Keep reference
            preview_label.pack(expand=True, pady=10)
            
            # Reset results
            self.ct_result_text.set("Ready to Analyze")
            self.ct_confidence_text.set("—")
            self.ct_result_label.configure(text_color=COLORS["text_primary"])
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def analyze_ct_image(self):
        """Analyze the uploaded CT scan image."""
        if not self.ct_image_path:
            messagebox.showwarning("No Image", "Please upload a CT scan image first.")
            return
        
        if not self.ct_predictor.model:
            messagebox.showerror("Model Error", "CT Model is not loaded. Please check the model file.")
            return
        
        try:
            # Make prediction
            is_cancer, confidence, prob = self.ct_predictor.predict(self.ct_image_path)
            
            # Update results
            if is_cancer:
                self.ct_result_text.set("⚠️ CANCER DETECTED")
                self.ct_result_label.configure(text_color=COLORS["danger"])
                # Confidence = probability of cancer (already correct)
                display_confidence = confidence
            else:
                self.ct_result_text.set("✓ NORMAL")
                self.ct_result_label.configure(text_color=COLORS["success"])
                # Confidence = probability of normal = 1 - probability of cancer
                display_confidence = 100 - confidence
            
            self.ct_confidence_text.set(f"{display_confidence:.1f}%")
            
            # Save to history
            image_name = os.path.basename(self.ct_image_path) if self.ct_image_path else "Unknown"
            diagnosis = "CANCER" if is_cancer else "Normal"
            self.history_manager.add_record(
                patient_name=image_name,
                analysis_type="CT Scan",
                model_used="EfficientNetB2",
                diagnosis=diagnosis,
                confidence=display_confidence,
                input_data={"image_path": self.ct_image_path}
            )
            
        except Exception as e:
            messagebox.showerror("Analysis Error", str(e))
    
    def clear_ct_image(self):
        """Clear the current CT image and results."""
        self.ct_image_path = None
        self.ct_image_label.configure(text="No image loaded")
        
        # Clear preview
        for widget in self.ct_preview_frame.winfo_children():
            widget.destroy()
        
        # Reset results
        self.ct_result_text.set("Upload CT Scan Image...")
        self.ct_confidence_text.set("—")
        self.ct_result_label.configure(text_color=COLORS["text_primary"])
    
    def upload_ct_batch(self):
        """Upload multiple CT images for batch processing."""
        file_paths = ctk.filedialog.askopenfilenames(
            filetypes=[
                ("Medical Images", "*.dcm *.png *.jpg *.jpeg *.bmp"),
                ("DICOM Files", "*.dcm"),
                ("Image Files", "*.png *.jpg *.jpeg *.bmp"),
                ("All Files", "*.*")
            ]
        )
        if not file_paths:
            return
        
        if not self.ct_predictor.model:
            messagebox.showerror("Model Error", "CT Model is not loaded.")
            return
        
        try:
            # Analyze batch
            results = self.ct_predictor.predict_batch(file_paths)
            
            # Save results
            import pandas as pd
            results_df = pd.DataFrame(results)
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            save_path = os.path.join(base_path, f"ct_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            results_df.to_csv(save_path, index=False)
            
            # Count results
            cancer_count = sum(1 for r in results if r["Diagnosis"] == "CANCER")
            total = len(results)
            
            self.ct_batch_status.configure(
                text=f"✅ Analyzed {total} images\n🔴 Cancer: {cancer_count}\n✅ Normal: {total - cancer_count}\n📁 Saved!"
            )
            
            messagebox.showinfo("Batch Complete", f"Results saved to:\n{save_path}")
            
        except Exception as e:
            messagebox.showerror("Batch Error", str(e))
    
    def upload_ct_folder(self):
        """Upload a DICOM folder for batch processing."""
        folder_path = ctk.filedialog.askdirectory(
            title="Select DICOM Folder (e.g., Pancreas-XXXXX)"
        )
        if not folder_path:
            return
        
        if not self.ct_predictor.model:
            messagebox.showerror("Model Error", "CT Model is not loaded.")
            return
        
        try:
            # Find all DICOM files in the folder (and subfolders)
            dcm_files = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith('.dcm'):
                        dcm_files.append(os.path.join(root, file))
            
            if not dcm_files:
                messagebox.showwarning("No DICOM Files", "No .dcm files found in the selected folder.")
                return
            
            self.ct_batch_status.configure(text=f"🔄 Processing {len(dcm_files)} DICOM files...")
            self.update()
            
            # Analyze batch
            results = self.ct_predictor.predict_batch(dcm_files)
            
            # Save results
            import pandas as pd
            results_df = pd.DataFrame(results)
            folder_name = os.path.basename(folder_path)
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            save_path = os.path.join(base_path, f"ct_{folder_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            results_df.to_csv(save_path, index=False)
            
            # Count results
            cancer_count = sum(1 for r in results if r["Diagnosis"] == "CANCER")
            total = len(results)
            
            self.ct_batch_status.configure(
                text=f"✅ Folder: {folder_name}\n📊 Analyzed: {total} images\n🔴 Cancer: {cancer_count}\n✅ Normal: {total - cancer_count}\n📁 Saved!"
            )
            
            messagebox.showinfo("Folder Analysis Complete", f"Folder: {folder_name}\nTotal: {total} images\nCancer: {cancer_count}\nNormal: {total - cancer_count}\n\nResults saved to:\n{save_path}")
            
        except Exception as e:
            messagebox.showerror("Folder Error", str(e))
    
    # ============== HISTORY POPUP ==============
    def show_history_popup(self):
        """Show history popup window with analysis records."""
        popup = ctk.CTkToplevel(self)
        popup.title("Analysis History")
        popup.geometry("900x600")
        popup.configure(fg_color=COLORS["bg_main"])
        popup.transient(self)
        popup.grab_set()
        
        # Header
        header_frame = ctk.CTkFrame(popup, fg_color=COLORS["bg_header"], height=60)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)
        
        ctk.CTkLabel(header_frame, text="📋 Analysis History", 
                     font=ctk.CTkFont(size=20, weight="bold"),
                     text_color=COLORS["text_title"]).pack(side="left", padx=20, pady=15)
        
        count = self.history_manager.get_record_count()
        ctk.CTkLabel(header_frame, text=f"Total Records: {count}", 
                     font=ctk.CTkFont(size=14),
                     text_color=COLORS["text_secondary"]).pack(side="right", padx=20, pady=15)
        
        # Content frame with scrollable area
        content_frame = ctk.CTkScrollableFrame(popup, fg_color=COLORS["bg_card"])
        content_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Load history
        history = self.history_manager.load_history()
        
        if not history:
            ctk.CTkLabel(content_frame, text="No analysis history yet.\nStart analyzing patients to build history!",
                        font=ctk.CTkFont(size=16), text_color=COLORS["text_secondary"]).pack(pady=50)
        else:
            # Table header
            header_row = ctk.CTkFrame(content_frame, fg_color=COLORS["bg_header"])
            header_row.pack(fill="x", pady=(0, 5))
            
            cols = ["Time", "Patient", "Type", "Model", "Diagnosis", "Confidence"]
            widths = [150, 120, 80, 150, 100, 80]
            for col, w in zip(cols, widths):
                ctk.CTkLabel(header_row, text=col, width=w, font=ctk.CTkFont(size=12, weight="bold"),
                            text_color=COLORS["glow_cyan"]).pack(side="left", padx=5, pady=8)
            
            # Data rows (newest first)
            for record in reversed(history[-50:]):  # Show last 50 records
                row = ctk.CTkFrame(content_frame, fg_color="transparent")
                row.pack(fill="x", pady=2)
                
                diagnosis = record.get("diagnosis", "N/A")
                row_color = COLORS["danger"] if diagnosis == "CANCER" else COLORS["success"] if diagnosis == "Healthy" else COLORS["text_secondary"]
                
                values = [
                    record.get("timestamp", "N/A")[:16],
                    record.get("patient_name", "Unknown")[:15],
                    record.get("analysis_type", "N/A")[:10],
                    record.get("model_used", "N/A")[:20],
                    diagnosis,
                    f"{record.get('confidence', 0):.1f}%"
                ]
                for val, w in zip(values, widths):
                    lbl = ctk.CTkLabel(row, text=str(val), width=w, font=ctk.CTkFont(size=11),
                                      text_color=row_color if val == diagnosis else COLORS["text_primary"])
                    lbl.pack(side="left", padx=5, pady=5)
        
        # Button frame
        btn_frame = ctk.CTkFrame(popup, fg_color=COLORS["bg_header"], height=60)
        btn_frame.pack(fill="x", side="bottom")
        btn_frame.pack_propagate(False)
        
        ctk.CTkButton(btn_frame, text="📤 Export CSV", command=self.export_history,
                     fg_color=COLORS["glow_blue"], width=120).pack(side="left", padx=20, pady=12)
        
        ctk.CTkButton(btn_frame, text="🗑️ Clear All", command=lambda: self.clear_history(popup),
                     fg_color=COLORS["danger"], width=120).pack(side="left", padx=10, pady=12)
        
        ctk.CTkButton(btn_frame, text="Close", command=popup.destroy,
                     fg_color="transparent", border_width=2, border_color=COLORS["text_secondary"],
                     width=100).pack(side="right", padx=20, pady=12)
    
    def export_history(self):
        """Export history to CSV."""
        path, msg = self.history_manager.export_to_csv()
        if path:
            messagebox.showinfo("Export Success", f"History exported to:\n{path}")
        else:
            messagebox.showwarning("Export", msg)
    
    def clear_history(self, popup=None):
        """Clear all history."""
        if messagebox.askyesno("Clear History", "Are you sure you want to delete all history records?"):
            self.history_manager.clear_history()
            messagebox.showinfo("Cleared", "History has been cleared.")
            if popup:
                popup.destroy()
                self.show_history_popup()  # Refresh

if __name__ == "__main__":
    app = MedicalDashboardApp()
    app.mainloop()