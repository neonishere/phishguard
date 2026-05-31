"""
app.py  —  PhishGuard Flask backend
Serves the frontend and exposes a /analyse endpoint.
"""

import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

from feature_extractor import (
    extract_features, normalise_url,
    FEATURE_NAMES, FEATURE_LABELS
)

app = Flask(__name__)

# ── Load model at startup (once, not per-request) ─────────────────────
print("Loading PhishGuard model...")
MODEL_PATH  = os.path.join(os.path.dirname(__file__), 'phishguard_model.pkl')
model = joblib.load(MODEL_PATH)
print("Model loaded.")

# Initialise SHAP explainer once
try:
    import shap
    explainer = shap.TreeExplainer(model)
    SHAP_AVAILABLE = True
    print("SHAP explainer ready.")
except Exception as e:
    SHAP_AVAILABLE = False
    print(f"SHAP not available: {e}")


# ── Routes ────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyse', methods=['POST'])
def analyse():
    data = request.get_json(silent=True)
    if not data or 'url' not in data:
        return jsonify({'error': 'No URL provided.'}), 400

    raw_url   = str(data['url']).strip()
    clean_url = normalise_url(raw_url)

    features = extract_features(clean_url)
    if features is None:
        return jsonify({'error': 'Could not parse URL. Please check the format.'}), 400

    X = pd.DataFrame([features])

    # Prediction
    prob     = float(model.predict_proba(X)[0][1])   # probability of phishing
    pred     = int(model.predict(X)[0])
    verdict  = 'Phishing' if pred == 1 else 'Legitimate'
    confidence = round(prob * 100, 1)

    # Feature explanations via SHAP
    top_features = []
    if SHAP_AVAILABLE:
        try:
            shap_vals = explainer.shap_values(X)[0]
            impacts = [
                {
                    'name':      name,
                    'label':     FEATURE_LABELS.get(name, name),
                    'value':     float(X[name].iloc[0]),
                    'impact':    round(float(shap_vals[i]), 4),
                    'direction': 'phishing' if shap_vals[i] > 0 else 'legitimate',
                }
                for i, name in enumerate(FEATURE_NAMES)
            ]
            impacts.sort(key=lambda x: abs(x['impact']), reverse=True)
            top_features = impacts[:8]
        except Exception:
            pass

    # Fallback: if SHAP unavailable, show non-zero feature values
    if not top_features:
        flagged = [
            {'label': FEATURE_LABELS.get(k, k), 'value': v, 'direction': 'phishing'}
            for k, v in features.items()
            if isinstance(v, (int, float)) and v > 0
            and k in ('has_ip', 'has_at_symbol', 'suspicious_tld',
                      'keyword_count', 'has_port', 'has_double_slash')
        ]
        top_features = flagged[:8]

    return jsonify({
        'url':          raw_url,
        'verdict':      verdict,
        'is_phishing':  pred == 1,
        'confidence':   confidence,
        'top_features': top_features,
    })


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5000)
