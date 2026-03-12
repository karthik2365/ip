"""
Smart Disposal - Prediction API Server
Loads the trained LSTM model and serves predictions via Flask REST API.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import os
import time
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from Next.js

# ── Load model & scaler params on startup ──
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'waste_level_lstm_model.h5')
model = load_model(MODEL_PATH)

# The scaler was fit on the training data. We need the min/max to inverse transform.
# From the synthetic data generation, fill_level ranges roughly 3-85 (resets at 85).
# We'll accept raw fill levels (0-100) and normalize internally.
FILL_MIN = 0.0
FILL_MAX = 100.0

SEQ_LENGTH = 14  # Model expects 14 days of history


def normalize(data):
    """Normalize fill levels to 0-1 range."""
    return (np.array(data) - FILL_MIN) / (FILL_MAX - FILL_MIN)


def inverse_normalize(data):
    """Convert normalized values back to 0-100% range."""
    return np.array(data) * (FILL_MAX - FILL_MIN) + FILL_MIN


MODEL_INFO = {
    "model_type": "LSTM Neural Network",
    "architecture": "LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16) → Dense(1)",
    "total_parameters": int(model.count_params()),
    "sequence_length": SEQ_LENGTH,
    "optimizer": "Adam (lr=0.001)",
    "loss_function": "Mean Squared Error",
}


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None, "model_info": MODEL_INFO})


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict the next N days of fill levels.

    Request JSON:
    {
      "fill_levels": [12.5, 18.3, 25.1, ...],   // At least 14 values (0-100%)
      "forecast_days": 7                          // How many days to predict (default 7, max 30)
    }

    Response JSON:
    {
      "predictions": [
        {"day": 1, "fill_level": 45.2},
        {"day": 2, "fill_level": 52.8},
        ...
      ],
      "alerts": [
        {"day": 5, "fill_level": 82.3, "message": "Collection recommended"}
      ],
      "input_summary": {
        "days_provided": 14,
        "avg_fill": 35.2,
        "max_fill": 68.1,
        "min_fill": 12.5
      }
    }
    """
    try:
        start_time = time.time()
        prediction_time = datetime.now()
        data = request.get_json()

        if not data or 'fill_levels' not in data:
            return jsonify({"error": "Missing 'fill_levels' in request body"}), 400

        fill_levels = data['fill_levels']
        forecast_days = min(data.get('forecast_days', 7), 30)

        if not isinstance(fill_levels, list) or len(fill_levels) < SEQ_LENGTH:
            return jsonify({
                "error": f"Need at least {SEQ_LENGTH} fill level values. Got {len(fill_levels) if isinstance(fill_levels, list) else 0}."
            }), 400

        # Validate values
        for i, v in enumerate(fill_levels):
            if not isinstance(v, (int, float)):
                return jsonify({"error": f"Value at index {i} is not a number: {v}"}), 400
            if v < 0 or v > 100:
                return jsonify({"error": f"Value at index {i} out of range (0-100): {v}"}), 400

        # Use the last 14 values
        last_14 = fill_levels[-SEQ_LENGTH:]
        scaled = normalize(last_14)
        current_seq = scaled.reshape((1, SEQ_LENGTH, 1))

        predictions = []
        alerts = []
        today = prediction_time.date()

        for day in range(1, forecast_days + 1):
            pred_scaled = model.predict(current_seq, verbose=0)
            pred_value = float(inverse_normalize(pred_scaled[0, 0]))
            pred_value = max(0.0, min(100.0, pred_value))  # Clamp

            forecast_date = today + timedelta(days=day)
            predictions.append({
                "day": day,
                "date": forecast_date.isoformat(),
                "day_name": forecast_date.strftime("%A"),
                "fill_level": round(pred_value, 2)
            })

            if pred_value >= 80:
                alerts.append({
                    "day": day,
                    "date": forecast_date.isoformat(),
                    "day_name": forecast_date.strftime("%A"),
                    "fill_level": round(pred_value, 2),
                    "message": "Collection recommended — fill level exceeds 80%"
                })

            # Slide window
            new_val = pred_scaled[0, 0]
            current_seq = np.append(current_seq[0, 1:], [[new_val]], axis=0).reshape(1, SEQ_LENGTH, 1)

        elapsed_ms = round((time.time() - start_time) * 1000, 1)

        input_summary = {
            "days_provided": len(fill_levels),
            "avg_fill": round(float(np.mean(fill_levels)), 2),
            "max_fill": round(float(np.max(fill_levels)), 2),
            "min_fill": round(float(np.min(fill_levels)), 2),
        }

        return jsonify({
            "predictions": predictions,
            "alerts": alerts,
            "input_summary": input_summary,
            "prediction_time": prediction_time.isoformat(),
            "inference_ms": elapsed_ms,
            "model_info": MODEL_INFO,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict-csv', methods=['POST'])
def predict_csv():
    """
    Accept a CSV file with a 'fill_level' column.
    Uses last 14 rows to predict future fill levels.
    """
    try:
        import pandas as pd
        import io

        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded. Send as 'file' in multipart form."}), 400

        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "Only CSV files are supported."}), 400

        content = file.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(content))

        # Find the fill_level column (case-insensitive)
        fill_col = None
        for col in df.columns:
            if 'fill' in col.lower():
                fill_col = col
                break

        if fill_col is None:
            return jsonify({
                "error": f"No fill level column found. Columns: {list(df.columns)}. Expected a column containing 'fill' in the name."
            }), 400

        fill_levels = df[fill_col].dropna().tolist()
        forecast_days = min(int(request.form.get('forecast_days', 7)), 30)

        # Reuse the predict logic
        start_time = time.time()
        prediction_time = datetime.now()

        if len(fill_levels) < SEQ_LENGTH:
            return jsonify({"error": f"CSV needs at least {SEQ_LENGTH} rows. Got {len(fill_levels)}."}), 400

        last_14 = fill_levels[-SEQ_LENGTH:]
        scaled = normalize(last_14)
        current_seq = scaled.reshape((1, SEQ_LENGTH, 1))

        predictions = []
        alerts = []
        today = prediction_time.date()

        for day in range(1, forecast_days + 1):
            pred_scaled = model.predict(current_seq, verbose=0)
            pred_value = float(inverse_normalize(pred_scaled[0, 0]))
            pred_value = max(0.0, min(100.0, pred_value))

            forecast_date = today + timedelta(days=day)
            predictions.append({
                "day": day,
                "date": forecast_date.isoformat(),
                "day_name": forecast_date.strftime("%A"),
                "fill_level": round(pred_value, 2)
            })
            if pred_value >= 80:
                alerts.append({
                    "day": day,
                    "date": forecast_date.isoformat(),
                    "day_name": forecast_date.strftime("%A"),
                    "fill_level": round(pred_value, 2),
                    "message": "Collection recommended"
                })

            new_val = pred_scaled[0, 0]
            current_seq = np.append(current_seq[0, 1:], [[new_val]], axis=0).reshape(1, SEQ_LENGTH, 1)

        elapsed_ms = round((time.time() - start_time) * 1000, 1)

        return jsonify({
            "predictions": predictions,
            "alerts": alerts,
            "input_summary": {
                "days_provided": len(fill_levels),
                "avg_fill": round(float(np.mean(fill_levels)), 2),
                "max_fill": round(float(np.max(fill_levels)), 2),
                "min_fill": round(float(np.min(fill_levels)), 2),
            },
            "prediction_time": prediction_time.isoformat(),
            "inference_ms": elapsed_ms,
            "model_info": MODEL_INFO,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("Smart Disposal Prediction API running on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
