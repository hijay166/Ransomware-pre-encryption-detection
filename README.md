# Ransomware-pre-encryption-detection
ransomware-pre-encryption-detection on a network level based framework

🛡️ Ransomware Pre-Encryption Detection System
📌 Overview

This project implements a machine learning-based ransomware detection system that identifies malicious activity during the pre-encryption stage of ransomware execution.

Instead of relying on traditional signature-based antivirus methods, the system analyzes system API calls, registry operations, and behavioral indicators to detect ransomware before file encryption begins.

🎯 Objectives
Detect ransomware activity before encryption occurs
Reduce false negatives from signature-based antivirus tools
Use behavioral system-level features for classification
Compare multiple machine learning models for detection accuracy
⚙️ System Architecture

The system follows a standard ML pipeline:

Data Collection
Cuckoo Sandbox reports (API calls, registry, file behavior)
Data Preprocessing
Handling missing values
Encoding categorical features
Feature selection (removing irrelevant system calls)
Feature Engineering
Label encoding of target (Sample_Type)
Scaling numeric features
Model Training
Logistic Regression
Random Forest Classifier
Evaluation
Accuracy score
Confusion matrix
Classification report (Precision, Recall, F1-score)
🧠 Machine Learning Models Used
Logistic Regression
Random Forest Classifier

Future improvements:

XGBoost / LightGBM
Neural Networks (LSTM for sequential API calls)
Ensemble methods
📊 Dataset

The dataset consists of behavioral logs extracted from Windows execution traces, including:

API calls (e.g., CreateFile, WriteProcessMemory)
Registry operations
File system interactions
System-level indicators of compromise (IOCs)

⚠️ Note: The dataset is not included in this repository due to size. Place it under Data/Raw/ locally.
Dataset was gotten from kaggle : https://www.kaggle.com/datasets/zirarikhalid/ransomware-detection-dataset .

🗂️ Project Structure
Ransomware-pre-encryption-detection/
│
├── Data/
│   ├── Raw/                # Original dataset (not tracked by Git)
│   └── Processed/         # Cleaned dataset (ignored)
│
├── Doc/
│   ├── Methodology of the Architecture.pdf
│   └── figure/            # Generated plots (ignored)
│
├── Notebook/              # Jupyter experiments
├── src/
│   └── main.py            # ML pipeline script
├── tests/                 # Pytest unit tests
├── requirements.txt
├── README.md
└── .gitignore


🚀 How to Run
1. Clone the repository
git clone https://github.com/hijay166/Ransomware-pre-encryption-detection.git
cd Ransomware-pre-encryption-detection

3. Install dependencies
pip install -r requirements.txt

5. Run the pipeline
python src/main.py

7. Run tests (optional)
python -m pytest tests/


📈 Results
Logistic Regression: baseline performance
Random Forest: improved accuracy on non-linear patterns
Evaluation metrics:
Accuracy
Precision / Recall
Confusion Matrix

📌 Key Features
Pre-encryption ransomware detection
Behavioral feature engineering
Comparative ML model evaluation
Visualization of classification performance

⚠️ Limitations
Dataset imbalance (ransomware samples are limited)
Heavy reliance on sandbox-generated logs
Requires further real-time streaming integration

🔮 Future Work
Real-time detection pipeline
Deep learning models (LSTM/Transformers for API sequences)
Cloud-based deployment (AWS/Azure)
Improved feature selection and explainability (SHAP)

🧑‍💻 Tech Stack
Python
Pandas, NumPy, Scikit-learn
Matplotlib, Seaborn
Jupyter Notebook

📄 License
This project is for academic and research purposes.

👤 Author

Bolaji Tobi 
GitHub: https://github.com/hijay166
LinkedIn: linkedin.com/in/tobi-bolaji-0861b218b
