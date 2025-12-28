import numpy as np
import pandas as pd
import sqlite3
import joblib
from flask import Flask, request, jsonify, render_template

# -------------------------------
# App Initialization
# -------------------------------
app = Flask(__name__)

# -------------------------------
# Load Model & Scaler
# -------------------------------
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------
# Database Connection Function
# -------------------------------
def get_db_connection():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    return conn

# -------------------------------
# Expected Feature Order
# -------------------------------
expected_columns = [
    "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
    "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
    "V21","V22","V23","V24","V25","V26","V27","V28","Amount"
]

# -------------------------------
# Home Page (Frontend)
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------------------
# Real-Time Fraud Prediction API
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Simulate real-time transaction features (hidden PCA values)
        features = {
            "Time": data["Time"],
            "Amount": data["Amount"]
        }

        # Generate PCA-like features
        for i in range(1, 29):
            features[f"V{i}"] = np.random.normal(0, 1)

        # Create DataFrame
        df = pd.DataFrame([features])[expected_columns]

        # Scale features
        scaled_data = scaler.transform(df)

        # Predict fraud probability
        fraud_prob = model.predict_proba(scaled_data)[0][1]

        # Risk-based decision logic
        if fraud_prob > 0.8:
            risk = "high"
            decision = "🚫 Transaction Blocked"
        elif fraud_prob > 0.4:
            risk = "medium"
            decision = "⚠️ Manual Verification Required"
        else:
            risk = "low"
            decision = "✅ Transaction Approved"

        # -------------------------------
        # Save Transaction to Database
        # -------------------------------
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO transactions (
                amount, time, fraud_probability, risk_level, decision
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            data["Amount"],
            data["Time"],
            float(fraud_prob),
            risk,
            decision
        ))

        conn.commit()
        conn.close()

        # Return response
        return jsonify({
            "fraud_probability": round(float(fraud_prob), 4),
            "risk_level": risk,
            "decision": decision
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# -------------------------------
# View All Transactions (API)
# -------------------------------
@app.route("/transactions")
def view_transactions():
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT * FROM transactions ORDER BY created_at DESC"
    ).fetchall()
    conn.close()

    return jsonify({
        "transactions": [dict(row) for row in rows]
    })

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
