"""
Inference REST API server for the Federated Learning Traffic Prediction system.

Endpoints:
  GET  /api/inference/available-models   - list trained client models
  POST /api/inference/predict            - upload CSV and run predictions
  GET  /api/inference/metrics/<pred_id>  - return accuracy metrics for a prediction
  GET  /api/inference/export/<pred_id>   - download predictions as CSV
"""

import os
import io
import uuid
import csv
import traceback

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import load_model as _keras_load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from PythonUtils import load_final_model_path, save_inference_results, space_path

# Serve the inference dashboard UI from the simulator/ subdirectory.
SIMULATOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulator')

app = Flask(__name__, static_folder=SIMULATOR_DIR, static_url_path='/ui')
CORS(app)

# In-memory cache for completed predictions (keyed by prediction_id).
PREDICTIONS_CACHE = {}

CLIENT_NAMES = {
    '1': 'LOSAng (Los Angeles)',
    '2': 'NYCMng (New York City)',
    '3': 'SNVAng (Sunnyvale)',
    '4': 'STTLng (Seattle)',
    '5': 'WASHng (Washington)',
}

LOOKBACK = 70


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_exists(client_id):
    path = load_final_model_path(client_id)
    return os.path.exists(path)


def _run_prediction(data_values, client_id):
    """Scale data, create sequences, run model, inverse-transform results."""
    data_array = np.array(data_values, dtype=np.float64).reshape(-1, 1)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data_array)

    # Build (x, y) sequences with the same lookback used during training.
    x_list, y_list = [], []
    for i in range(len(scaled) - LOOKBACK - 1):
        x_list.append(scaled[i: i + LOOKBACK, 0])
        y_list.append(scaled[i + LOOKBACK, 0])

    if not x_list:
        raise ValueError(
            f"Not enough data points. Need at least {LOOKBACK + 2} rows, "
            f"got {len(data_values)}."
        )

    x_arr = np.array(x_list)
    y_arr = np.array(y_list)
    x_input = x_arr.reshape(x_arr.shape[0], 1, x_arr.shape[1])

    model_path = load_final_model_path(client_id)
    model = _keras_load_model(model_path)

    # Use direct model call instead of model.predict() to avoid a TF
    # compatibility bug with tensorflow.python.keras in TF >= 2.x.
    x_tensor = tf.constant(x_input.astype(np.float32))
    y_pred_scaled = model(x_tensor, training=False).numpy().flatten()

    # Inverse-transform
    y_actual = scaler.inverse_transform(y_arr.reshape(-1, 1)).flatten()
    y_predicted = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    rows = [
        {
            'index': int(i),
            'actual': round(float(y_actual[i]), 4),
            'predicted': round(float(y_predicted[i]), 4),
            'error': round(float(y_actual[i] - y_predicted[i]), 4),
        }
        for i in range(len(y_actual))
    ]

    mse = float(mean_squared_error(y_actual, y_predicted))
    mae = float(mean_absolute_error(y_actual, y_predicted))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((y_actual - y_predicted) ** 2))
    ss_tot = float(np.sum((y_actual - np.mean(y_actual)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    metrics = {
        'mse': round(mse, 6),
        'mae': round(mae, 6),
        'rmse': round(rmse, 6),
        'r2': round(r2, 6),
        'num_predictions': len(rows),
    }

    return rows, metrics


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
@app.route('/dashboard')
def dashboard():
    """Serve the inference dashboard HTML UI."""
    from flask import send_from_directory
    return send_from_directory(SIMULATOR_DIR, 'inference_dashboard.html')


@app.route('/api/inference/available-models', methods=['GET'])
def available_models():
    """Return list of client IDs for which a trained model file exists."""
    models = []
    for cid, name in CLIENT_NAMES.items():
        path = load_final_model_path(cid)
        models.append({
            'client_id': cid,
            'name': name,
            'available': os.path.exists(path),
            'model_path': path,
        })
    return jsonify({'models': models})


@app.route('/api/inference/predict', methods=['POST'])
def predict():
    """
    Accept a multipart/form-data POST with:
      - file  : CSV file (required)
      - client_id : '1'–'5' (required)

    Returns JSON with prediction_id, rows, and metrics.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    client_id = request.form.get('client_id', '1')
    if client_id not in CLIENT_NAMES:
        return jsonify({'error': f'Invalid client_id "{client_id}". Choose 1–5.'}), 400
    if not _model_exists(client_id):
        return jsonify({
            'error': (
                f'No trained model found for client {client_id}. '
                'Please run the training first (python run.py).'
            )
        }), 404

    file = request.files['file']
    if not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Only CSV files are supported'}), 400

    try:
        df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
    except Exception as exc:
        return jsonify({'error': f'Failed to read CSV: {exc}'}), 400

    # Accept any single numeric column or auto-detect the traffic column.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return jsonify({'error': 'No numeric columns found in the uploaded CSV'}), 400
    values = df[numeric_cols[0]].dropna().tolist()
    if len(values) < LOOKBACK + 2:
        return jsonify({
            'error': (
                f'Not enough data. Need at least {LOOKBACK + 2} numeric rows, '
                f'got {len(values)}.'
            )
        }), 400

    try:
        rows, metrics = _run_prediction(values, client_id)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500

    prediction_id = str(uuid.uuid4())
    PREDICTIONS_CACHE[prediction_id] = {
        'client_id': client_id,
        'rows': rows,
        'metrics': metrics,
    }

    return jsonify({
        'prediction_id': prediction_id,
        'client_id': client_id,
        'client_name': CLIENT_NAMES[client_id],
        'rows': rows,
        'metrics': metrics,
    })


@app.route('/api/inference/metrics/<prediction_id>', methods=['GET'])
def get_metrics(prediction_id):
    """Return accuracy metrics for a completed prediction."""
    entry = PREDICTIONS_CACHE.get(prediction_id)
    if not entry:
        return jsonify({'error': 'Prediction not found'}), 404
    return jsonify({
        'prediction_id': prediction_id,
        'client_id': entry['client_id'],
        'metrics': entry['metrics'],
    })


@app.route('/api/inference/export/<prediction_id>', methods=['GET'])
def export_predictions(prediction_id):
    """Download prediction results as a CSV file."""
    entry = PREDICTIONS_CACHE.get(prediction_id)
    if not entry:
        return jsonify({'error': 'Prediction not found'}), 404

    rows = entry['rows']
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=['index', 'actual', 'predicted', 'error'])
    writer.writeheader()
    writer.writerows(rows)
    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'predictions_{prediction_id[:8]}.csv',
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Starting Inference API server on http://localhost:8002")
    print("Endpoints:")
    print("  GET  /api/inference/available-models")
    print("  POST /api/inference/predict  (form fields: file, client_id)")
    print("  GET  /api/inference/metrics/<prediction_id>")
    print("  GET  /api/inference/export/<prediction_id>")
    # threaded=False avoids TensorFlow/Keras conflicts inside Flask worker threads.
    app.run(host='0.0.0.0', port=8002, debug=False, threaded=False)
