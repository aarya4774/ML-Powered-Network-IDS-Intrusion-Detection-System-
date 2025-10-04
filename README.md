ML-Powered Network IDS (Intrusion Detection System)
This project is a standalone, Python-based Intrusion Detection System (IDS) that uses the Random Forest machine learning algorithm to analyze network traffic and classify threats. It features a complete data pipeline and an interactive Tkinter GUI for easy experimentation and performance evaluation.

The system is trained and evaluated using the industry-standard NSL-KDD dataset.

✨ Key Features
Interactive GUI: A user-friendly graphical interface (Tkinter) guides the user through the entire ML workflow: Data Load → Model Train → Evaluation.

Dual Classification Modes: Users can select between two critical security tasks before training the model:

Binary Classification: Distinguishes traffic as either Normal (0) or Attack (1). This mode uses balanced class weights to prioritize minimizing missed attacks (False Negatives).

Multiclass Classification: Identifies specific attack categories: DoS, Probe, R2L, U2R, and Normal traffic.

Robust Preprocessing: Handles complex tabular data pipeline steps automatically, including feature scaling, one-hot encoding of categorical variables, and crucial feature alignment between training and testing sets.

Comprehensive Metrics: Displays detailed evaluation results, including Overall Accuracy, Classification Report (Precision, Recall, F1-Score), and an interpreted Confusion Matrix.

🛠️ Getting Started
Prerequisites
To run this application, you need Python and three core data science libraries installed.

Install Libraries:

pip install pandas numpy scikit-learn

Acquire Data: Download the following two files from the NSL-KDD dataset source and place them in the same directory as the ids_gui_app.py script:

KDDTrain+.txt (Training Data)

KDDTest+.txt (Testing Data)

Execution
Save the project code as ids_gui_app.py.

Run the application from your terminal:

python ids_gui_app.py

🚀 Workflow
Once the application launches, follow the numbered buttons in sequence:

Load & Preprocess Data: Loads the two .txt files, handles one-hot encoding for categorical features (like protocol type and service), scales numerical data, and aligns the 118 resulting features.

Train Model: Trains the Random Forest Classifier based on the currently selected classification task (Binary or Multiclass).

Evaluate Model: Predicts labels on the separate test dataset and displays a detailed performance report.

Example Binary Output Interpretation
When running the Binary task, the system focuses on minimizing False Negatives (missed attacks). The output will detail:

True Positives: Number of attacks correctly flagged.

False Negatives: Number of actual attacks that were missed (the most dangerous type of error).

False Positives: Number of normal connections mistakenly flagged as attacks (false alarms).

💡 Future Enhancements
Integrate additional classifiers (e.g., XGBoost, Deep Neural Networks) for comparative analysis.

Add a live feature importance visualization tool.

Implement hyperparameter tuning options within the GUI.
