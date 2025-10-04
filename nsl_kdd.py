import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox 

# --- 1. IDS MACHINE LEARNING MODEL BACKEND CLASS ---
class IDSModel:
    """Handles all data loading, preprocessing, training, and evaluation for the IDS."""
    
    COLUMNS = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
        'attack_type', 'difficulty_level'
    ]
    
    TRAIN_FILE = 'KDDTrain+.txt'
    TEST_FILE = 'KDDTest+.txt'
    CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']
    
    # Mapping for multiclass classification (DoS, Probe, R2L, U2R, Normal)
    ATTACK_MAPPING = {
        'normal': 'Normal',
        'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 'smurf': 'DoS',
        'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS', 'processtable': 'DoS', 'udpstorm': 'DoS',
        'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
        'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R', 'xterm': 'U2R', 
        'ps': 'U2R', 'sqlattack': 'U2R', 'httptunnel': 'U2R',
        'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L', 'phf': 'R2L',
        'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L'
    }
    CLASS_NAMES_MULTICLASS = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']

    def __init__(self):
        self.X_train_final = None
        self.Y_train_bin = None
        self.Y_train_multi = None
        self.X_test_final = None
        self.Y_test_bin = None
        self.Y_test_multi = None
        self.model = None
        self.scaler = None

    def load_and_preprocess_data(self):
        """Loads and preprocesses both training and testing datasets."""
        try:
            # Load Raw Data - returns X, Y_bin, Y_multi
            X_train_raw, self.Y_train_bin, self.Y_train_multi = self._load_data(self.TRAIN_FILE)
            X_test_raw, self.Y_test_bin, self.Y_test_multi = self._load_data(self.TEST_FILE)

            if X_train_raw is None or X_test_raw is None:
                return "Error: Data files not found. Check file paths."

            # Feature Engineering and Alignment
            X_train_enc, X_test_enc = self._feature_engineering(X_train_raw, X_test_raw, self.CATEGORICAL_COLS)

            # Feature Scaling
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train_enc)
            X_test_scaled = self.scaler.transform(X_test_enc)

            self.X_train_final = pd.DataFrame(X_train_scaled, columns=X_train_enc.columns)
            self.X_test_final = pd.DataFrame(X_test_scaled, columns=X_test_enc.columns)
            
            return f"✅ Data Loading Complete.\n- Train Samples: {len(self.Y_train_bin)}\n- Test Samples: {len(self.Y_test_bin)}\n- Features: {self.X_train_final.shape[1]}"

        except Exception as e:
            return f"Error during data loading/preprocessing: {e}"

    def _load_data(self, filepath):
        """Helper function to load raw data and map labels."""
        try:
            df = pd.read_csv(filepath, names=self.COLUMNS)
        except FileNotFoundError:
            return None, None, None

        df = df.drop('difficulty_level', axis=1)
        
        # 1. Binary Label
        Y_binary = df['attack_type'].apply(lambda x: 0 if x == 'normal' else 1)
        
        # 2. Multiclass Label (using the defined mapping, 'Unknown' if not found)
        Y_multiclass = df['attack_type'].apply(lambda x: self.ATTACK_MAPPING.get(x, 'Unknown'))
        
        # Drop the original attack column
        X = df.drop(['attack_type'], axis=1)
        return X, Y_binary, Y_multiclass

    def _feature_engineering(self, X_train, X_test, categorical_cols):
        """Applies One-Hot Encoding and aligns the feature sets (CRITICAL)."""
        combined_df = pd.concat([X_train, X_test], ignore_index=True)
        combined_encoded = pd.get_dummies(combined_df, columns=categorical_cols, drop_first=True)

        X_train_processed = combined_encoded.iloc[:len(X_train)]
        X_test_processed = combined_encoded.iloc[len(X_train):]

        # 1. Ensure X_test has all columns present in X_train (fill missing with 0)
        missing_cols_in_test = set(X_train_processed.columns) - set(X_test_processed.columns)
        for c in missing_cols_in_test:
            X_test_processed[c] = 0

        # 2. Align the column order strictly
        X_test_processed = X_test_processed[X_train_processed.columns]
        
        # 3. Drop zero-variance columns based on the training set
        cols_to_keep = X_train_processed.columns[(X_train_processed != 0).any(axis=0)]
        
        X_train_processed = X_train_processed[cols_to_keep]
        X_test_processed = X_test_processed[cols_to_keep]
        
        return X_train_processed, X_test_processed

    def train_model(self, classification_type):
        """Trains the Random Forest model based on the selected classification type."""
        if self.X_train_final is None:
            return "❌ Error: Data not loaded. Click 'Load Data' first."

        start_time = time.time()
        
        # Determine the target labels and class weights based on selection
        if classification_type == 'multiclass':
            Y_train_target = self.Y_train_multi
            class_weights = None  # Using default weights for simplicity in multiclass
            label_info = "Multiclass (Normal, DoS, Probe, R2L, U2R)"
        else: # binary
            Y_train_target = self.Y_train_bin
            class_weights = 'balanced' # Use balanced weights to fight False Negatives
            label_info = "Binary (Normal vs. Attack) - Balanced Weights"

        self.model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1, 
            class_weight=class_weights, 
            verbose=0
        )
        self.model.fit(self.X_train_final, Y_train_target)
        
        end_time = time.time()
        return f"✅ Model Training Complete.\n- Algorithm: Random Forest (100 Trees)\n- Time: {end_time - start_time:.2f} seconds\n- Classification Mode: {label_info}"

    def evaluate_model(self, classification_type):
        """Evaluates the trained model and returns the report string."""
        if self.model is None:
            return "❌ Error: Model not trained. Click 'Train Model' first."
        if self.X_test_final is None:
            return "❌ Error: Test data not ready."

        # Determine the target labels and class names for evaluation
        if classification_type == 'multiclass':
            Y_test_target = self.Y_test_multi
            target_names = self.CLASS_NAMES_MULTICLASS
            # For multiclass, the labels are strings (Normal, DoS, Probe, R2L, U2R)
            labels_to_use = target_names
        else: # binary
            Y_test_target = self.Y_test_bin
            target_names = ['Normal (0)', 'Attack (1)']
            # For binary, the labels are integers 0 and 1
            labels_to_use = [0, 1] 

        Y_pred = self.model.predict(self.X_test_final)
        
        # 1. Filter targets and predictions to include only the classes we care about.
        valid_indices = Y_test_target[Y_test_target.isin(labels_to_use)].index
        
        Y_test_filtered = Y_test_target.loc[valid_indices]
        Y_pred_filtered = pd.Series(Y_pred).loc[valid_indices]
        
        # FIX: Ensure Y_pred_filtered is of the same type as Y_test_filtered to avoid "Mix of label input types"
        if classification_type == 'binary':
            # Convert predictions to integer type for consistency with Y_test_filtered (which is int)
            Y_pred_filtered = Y_pred_filtered.astype(int)
        else:
            # Convert predictions to string type for consistency with Y_test_filtered (which is string)
            Y_pred_filtered = Y_pred_filtered.astype(str)
        
        # Check if the filtering resulted in empty data
        if Y_test_filtered.empty:
             return "❌ Error: Evaluation data is empty after filtering. Check data loading or classification type selection."

        # 2. Pass explicit labels to the scikit-learn functions to link int labels (0/1) 
        # or string labels (DoS/Probe/...) to the target_names list.
        accuracy = accuracy_score(Y_test_filtered, Y_pred_filtered)
        report = classification_report(
            Y_test_filtered, 
            Y_pred_filtered, 
            target_names=target_names, 
            labels=labels_to_use, 
            output_dict=True, 
            zero_division=0
        )
        cm = confusion_matrix(
            Y_test_filtered, 
            Y_pred_filtered, 
            labels=labels_to_use 
        )

        # Build the formatted report string
        report_str = f"--- Evaluation Results ({classification_type.upper()})---\n"
        report_str += f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n"
        
        report_str += "Classification Report (Filtered to Known Classes):\n"
        report_str += f"{'':<12}{'Precision':>10}{'Recall':>10}{'F1-Score':>10}{'Support':>10}\n"
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                report_str += f"{label:<12}{metrics['precision']:>10.2f}{metrics['recall']:>10.2f}{metrics['f1-score']:>10.2f}{metrics['support']:>10.0f}\n"

        report_str += "\nConfusion Matrix:\n"
        # Print confusion matrix dynamically based on target names
        cm_labels = target_names
        
        # Calculate the required width for the header string dynamically
        PREDICTED_HEADER_WIDTH = 18 * len(cm_labels)
        
        # Use the calculated width for the header line
        report_str += f"{'':<14}|{'Predicted: ':<{PREDICTED_HEADER_WIDTH}}|\n"
        report_str += f"{'Actual':<14}|" + "".join([f"{l:<18}" for l in cm_labels]) + "|\n"
        
        for i, row in enumerate(cm):
            report_str += f"{cm_labels[i]:<14}|" + "".join([f"{val:>18}" for val in row]) + "|\n"

        if classification_type == 'binary':
            report_str += "\nInterpretation (Cybersecurity Focus):\n"
            report_str += f"- True Positives (Attacks Detected): {cm[1, 1]} (Good)\n"
            report_str += f"- False Negatives (Attacks Missed): {cm[1, 0]} (Critical Error)\n"
            report_str += f"- False Positives (False Alarms): {cm[0, 1]} (Nuisance)\n"
        else:
            report_str += "\nInterpretation:\n"
            report_str += "The main diagonal (e.g., Actual DoS -> Predicted DoS) shows correct classification.\n"
            report_str += "Off-diagonal values represent misclassifications. Focus on 'R2L' and 'U2R' for rare attack detection performance."
            
        return report_str

# --- 2. TKINTER GUI APPLICATION FRONTEND ---

class IDSApp:
    def __init__(self, master):
        self.master = master
        master.title("ML Intrusion Detection System (NSL-KDD)")
        master.geometry("850x700") # Increased size for new elements
        
        # Initialize the ML backend
        self.model_backend = IDSModel()
        # Removed export functionality, no need for self.last_report
        self.classification_type = tk.StringVar(value='binary') # Default classification type
        
        self._setup_styles()
        self._create_widgets()
        self._show_initial_message()

    def _setup_styles(self):
        """Configures basic visual styles for the application."""
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#f0f4f8')
        style.configure('TButton', font=('Inter', 10, 'bold'), padding=10, background='#3498db', foreground='white') # Changed button color
        style.configure('Header.TLabel', font=('Inter', 16, 'bold'), foreground='#2c3e50', background='#f0f4f8')
        style.configure('Status.TLabel', font=('Inter', 10, 'italic'), foreground='#2980b9', background='#f0f4f8')
        style.configure('TLabel', background='#f0f4f8')
        style.configure('TLabelframe', background='#ecf0f1')
        style.configure('TLabelframe.Label', background='#ecf0f1', font=('Inter', 10, 'bold'))


    def _create_widgets(self):
        """Lays out the buttons, status, and output text area."""
        
        main_frame = ttk.Frame(self.master, padding="15 15 15 15", style='TFrame')
        main_frame.pack(fill='both', expand=True)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(5, weight=1) # Moved output to row 5

        # 1. Header
        ttk.Label(main_frame, text="Network Intrusion Detection System", style='Header.TLabel').grid(row=0, column=0, pady=(0, 15), sticky='ew')
        
        # 2. Control Buttons Frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, pady=(0, 15), sticky='ew')
        
        # Configure button weights for uniform resizing (Now 3 columns instead of 4)
        button_frame.columnconfigure((0, 1, 2), weight=1)

        self.load_btn = ttk.Button(button_frame, text="1. Load & Preprocess Data", command=self.action_load_data)
        self.load_btn.grid(row=0, column=0, padx=5, sticky='ew')

        self.train_btn = ttk.Button(button_frame, text="2. Train Model", command=self.action_train_model, state=tk.DISABLED)
        self.train_btn.grid(row=0, column=1, padx=5, sticky='ew')

        self.eval_btn = ttk.Button(button_frame, text="3. Evaluate Model", command=self.action_evaluate_model, state=tk.DISABLED)
        self.eval_btn.grid(row=0, column=2, padx=5, sticky='ew')
        
        # Export button was removed

        # 3. Model Type Selection Frame (New)
        type_frame = ttk.LabelFrame(main_frame, text="Classification Task Selection", padding="10 5 10 5")
        type_frame.grid(row=2, column=0, pady=(5, 10), sticky='ew')
        type_frame.columnconfigure((0, 1), weight=1)
        
        radio_bin = ttk.Radiobutton(type_frame, text="Binary (Normal/Attack) - High Recall Focus", variable=self.classification_type, value='binary', command=self.update_status_message)
        radio_multi = ttk.Radiobutton(type_frame, text="Multiclass (DoS, Probe, R2L, U2R, Normal)", variable=self.classification_type, value='multiclass', command=self.update_status_message)
        
        radio_bin.grid(row=0, column=0, padx=15, pady=5, sticky='w')
        radio_multi.grid(row=0, column=1, padx=15, pady=5, sticky='w')

        # 4. Status Message
        self.status_var = tk.StringVar(value="Ready. Click 'Load Data' to begin.")
        ttk.Label(main_frame, textvariable=self.status_var, style='Status.TLabel').grid(row=3, column=0, pady=(5, 10), sticky='ew')
        
        # 5. Output Text Area
        ttk.Label(main_frame, text="Output & Results:", font=('Inter', 10, 'bold'), foreground='#2c3e50').grid(row=4, column=0, sticky='nw')

        self.output_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=20, font=('Consolas', 9), bg='#ffffff', fg='#1c2833', relief=tk.FLAT, bd=2, padx=10, pady=10)
        self.output_text.grid(row=5, column=0, sticky='nsew', pady=(5, 0))
        
    def _show_initial_message(self):
        """Displays startup instructions in the output area."""
        initial_message = (
            "Welcome to the ML-based Intrusion Detection System Demo.\n"
            "----------------------------------------------------------\n"
            "INSTRUCTIONS:\n"
            "1. Prerequisites: Ensure 'pandas', 'numpy', and 'scikit-learn' are installed.\n"
            "2. Data Files: Ensure 'KDDTrain+.txt' and 'KDDTest+.txt' are in this script's directory.\n"
            "3. Workflow: Click the buttons in sequence (1 -> 2 -> 3).\n\n"
            "Classification Type Details:\n"
            "- Binary: Treats all attacks as one group (Focus: Detect ANY threat).\n"
            "- Multiclass: Classifies threats into DoS, Probe, R2L, U2R, and Normal (Focus: Identify threat type).\n"
        )
        self.output_text.insert(tk.END, initial_message)
        self.output_text.tag_config('success', foreground='#27ae60')
        self.output_text.tag_config('error', foreground='#c0392b')

    def update_status_message(self):
        """Updates the status message based on the current classification selection."""
        if self.train_btn['state'] == tk.NORMAL or self.eval_btn['state'] == tk.NORMAL:
            self.status_var.set(f"Classification type set to: {self.classification_type.get().upper()}. Rerun training.")
        else:
            self.status_var.set(f"Classification type set to: {self.classification_type.get().upper()}.")

    def update_output(self, message, tag=None):
        """Appends a message to the text area and scrolls to the end."""
        self.output_text.insert(tk.END, message + "\n\n", tag)
        self.output_text.see(tk.END)
        self.status_var.set(message.split('\n')[0])

    def action_load_data(self):
        """Action for Load Data button."""
        self.load_btn.config(state=tk.DISABLED)
        self.train_btn.config(state=tk.DISABLED)
        self.eval_btn.config(state=tk.DISABLED)
        self.output_text.delete('1.0', tk.END)
        self.update_output("Processing Data (This may take a moment)...")
        self.master.update()
        
        result = self.model_backend.load_and_preprocess_data()
        
        if result.startswith("✅"):
            self.update_output(result, 'success')
            self.train_btn.config(state=tk.NORMAL)
        else:
            self.update_output(result, 'error')
            self.load_btn.config(state=tk.NORMAL) 

    def action_train_model(self):
        """Action for Train Model button."""
        self.train_btn.config(state=tk.DISABLED)
        self.eval_btn.config(state=tk.DISABLED)
        self.update_output(f"Starting Model Training in {self.classification_type.get().upper()} mode...")
        self.master.update()

        classification_type = self.classification_type.get()
        result = self.model_backend.train_model(classification_type)
        
        if result.startswith("✅"):
            self.update_output(result, 'success')
            self.eval_btn.config(state=tk.NORMAL)
        else:
            self.update_output(result, 'error')
            self.train_btn.config(state=tk.NORMAL)

    def action_evaluate_model(self):
        """Action for Evaluate Model button."""
        self.eval_btn.config(state=tk.DISABLED)
        self.update_output("Evaluating Model on Test Data...")
        self.master.update()

        classification_type = self.classification_type.get()
        result = self.model_backend.evaluate_model(classification_type)
        
        if result.startswith("--- Evaluation Results ---"):
            self.update_output(result)
        else:
            self.update_output(result, 'error')
            self.eval_btn.config(state=tk.NORMAL)

    # Removed action_export_results

if __name__ == '__main__':
    # Initialize the main tkinter window
    root = tk.Tk()
    app = IDSApp(root)
    # Start the GUI event loop
    root.mainloop()
