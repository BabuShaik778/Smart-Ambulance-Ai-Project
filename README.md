
# Smart Ambulance AI Monitoring System #

Real-time patient monitoring for ambulances using ML-based anomaly detection and risk scoring. The system detects early warning signs, calculates a risk score, and provides a confidence level to assist clinical decision-making.


# Key Features #

# 1.Anomaly Detection #

Detects abnormal vitals beyond simple thresholds

Uses statistical preprocessing and machine learning (Isolation Forest)

Rolling window smoothing reduces false positives from noisy sensor data

# 2.Risk Scoring #

Combines multiple vital signs (HR, SpO2, BP) and trends

Generates a triage score for immediate assessment

Confidence score indicates model certainty

 # 3.Dashboard & API #

Modern HTML/CSS/JS dashboard for real-time input and visualization

Flask REST API for integration with other systems

Returns anomaly status, risk score, and confidence

# 3.Evaluation & Metrics #

Precision, Recall, F1 Score, False Alert Rate, Alert Latency

Designed for safety-critical scenarios in emergency medical services


SmartAmbulanceAI/
├─ data/                  # Raw and processed vital sign data
│   └─ vitals.csv
├─ models/                # Trained ML models
│   └─ anomaly_model.pkl
├─ templates/             # Dashboard HTML/CSS
│   └─ index.html
├─ app.py                 # Flask API for inference
├─ train_model.py         # Training and preprocessing scripts
├─ requirements.txt       # Python dependencies
└─ README.md


# Installation : #

# 1.Clone the repository: #

git clone https://github.com/BabuShaik778/SmartAmbulanceAI.git
cd SmartAmbulanceAI


  # 2.Create a virtual environment: #

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows


# 3.Install dependencies: #

pip install -r requirements.txt


# 4.Run the API: #

python app.py


 # 5.Open the dashboard: #

http://127.0.0.1:5000/



 # Usage Dashboard #

Enter patient vitals: Heart Rate, SpO2, BP SYS, BP DIA, Motion

Click Analyze Patient

# View outputs: #

Anomaly Status

Risk Score

Confidence


# API Example : #

POST /predict

 # Request Body: #

 {
  "hr": 120,
  "spo2": 90,
  "bp_sys": 85,
  "bp_dia": 60,
  "motion": 0.3
}

 # Response:#

{
  "anomaly": 1,
  "risk_score": 7,
  "confidence": 0.9
}


 # Model Details #

# Algorithm: Isolation Forest #

Features: Rolling means and trends of HR, SpO2, BP, Motion

Windowing: Rolling windows smooth sensor noise and short-term spikes

 # Risk Logic: #

HR > 110 → +2

SpO2 < 92 → +3

BP SYS < 90 → +2

Confidence Score: 0.7–0.9 based on anomaly probability


# Metrics & Evaluation #

Precision: ~0.97–1.0

Recall: ~0.90–0.95

F1 Score: ~0.92–0.96

False Alert Rate: (false alerts / total alerts)

Alert Latency: Time difference between true event and alert trigger

Designed for ambulance safety, where missing a critical deterioration is unacceptable. Minor false positives are acceptable if they improve recall.



 # Safety Considerations #

Most Dangerous Failure: Missing critical deterioration → life-threatening

Reducing False Alerts: Rolling median smoothing, multi-vital scoring, confidence thresholding

Human Oversight: Alerts assist clinicians; decisions are never fully automated


# Reproducibility #

Training: train_model.py

Inference/API: app.py + templates/index.html

Dependencies: requirements.txt


# Requirements : #
 
Flask==2.3.2
numpy==1.24.4
pandas==2.1.2
scikit-learn==1.3.2
joblib==1.3.2
matplotlib==3.8.0
