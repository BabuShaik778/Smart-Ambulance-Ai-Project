from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("anomaly_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    # Extract values
    hr = data["hr"]
    spo2 = data["spo2"]
    bp_sys = data["bp_sys"]
    bp_dia = data["bp_dia"]
    motion = data["motion"]

    # Convert to model input
    features = np.array([[hr, spo2, bp_sys, bp_dia, motion]])

    # Model prediction
    prediction = model.predict(features)[0]

    anomaly = 1 if prediction == -1 else 0

    # Risk score logic
    risk = 0
    if hr > 110:
        risk += 2
    if spo2 < 92:
        risk += 3
    if bp_sys < 90:
        risk += 2

    confidence = 0.9 if anomaly else 0.7

    return jsonify({
        "anomaly": anomaly,
        "risk_score": risk,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)