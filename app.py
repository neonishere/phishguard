"""
app.py  —  PhishGuard Flask backend
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OMP_THREAD_LIMIT'] = '1'
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from urllib.parse import urlparse

from feature_extractor import (
    extract_features, normalise_url,
    FEATURE_NAMES, FEATURE_LABELS
)

app = Flask(__name__)
CORS(app, resources={r"/analyse*": {"origins": "*"}})

# ── Load model ────────────────────────────────────────────────────────
print("Loading PhishGuard model...")
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'phishguard_model.pkl')
model      = joblib.load(MODEL_PATH)
model.set_params(n_jobs=1)

_raw_imp  = model.feature_importances_
_imp_norm = _raw_imp / (_raw_imp.sum() + 1e-9)
IMPORTANCES = dict(zip(FEATURE_NAMES, _imp_norm.tolist()))

PHISHING_POSITIVE = {
    'keyword_count', 'suspicious_tld', 'has_ip', 'has_at_symbol',
    'has_port', 'has_double_slash', 'num_hyphens', 'num_percent',
    'num_at', 'digit_ratio', 'special_char_ratio', 'num_subdomains',
    'url_length', 'domain_entropy',
}
print("Model ready.")

# ── Known-legitimate domain whitelist ─────────────────────────────────
# Lexical models have no domain-reputation knowledge; well-known sites
# with unusual structural features (e.g. short domains without 'www') can
# be misclassified. This whitelist is a standard production mitigation.
KNOWN_LEGITIMATE = {
    'google.com','youtube.com','facebook.com','twitter.com','x.com',
    'instagram.com','linkedin.com','reddit.com','wikipedia.org',
    'github.com','gitlab.com','stackoverflow.com','amazon.com',
    'microsoft.com','apple.com','netflix.com','spotify.com',
    'bbc.co.uk','bbc.com','theguardian.com','nytimes.com',
    'cnn.com','reuters.com','bloomberg.com','techcrunch.com',
    'cloudflare.com','wordpress.com','medium.com','substack.com',
    'notion.so','figma.com','dropbox.com','onedrive.live.com',
    'drive.google.com','docs.google.com','maps.google.com',
    'mail.google.com','accounts.google.com',
    'twitch.tv','discord.com','slack.com','zoom.us',
    'whatsapp.com','telegram.org','signal.org',
    'paypal.com','stripe.com','ebay.com','etsy.com',
    'dmuedu.ac.uk','dmu.ac.uk','le.ac.uk','ox.ac.uk','cam.ac.uk',
    'gov.uk','nhs.uk','hmrc.gov.uk','police.uk',
    'onrender.com','pythonanywhere.com','heroku.com','netlify.app',
    'vercel.app','railway.app',
}

def _is_whitelisted(url):
    """
    Return True if the hostname or any parent domain is in the whitelist.
    e.g. neonishere.pythonanywhere.com matches pythonanywhere.com
    """
    try:
        host = urlparse(url).hostname or ''
        bare = host.removeprefix('www.')
        if bare in KNOWN_LEGITIMATE or host in KNOWN_LEGITIMATE:
            return True
        parts = bare.split('.')
        for i in range(1, len(parts) - 1):
            parent = '.'.join(parts[i:])
            if parent in KNOWN_LEGITIMATE:
                return True
        return False
    except Exception:
        return False


# ── Feature explanation (fast, no SHAP) ──────────────────────────────
def explain_features(features_dict, top_n=8):
    results = []
    for name in FEATURE_NAMES:
        val = float(features_dict.get(name, 0))   # explicit float() cast
        imp = float(IMPORTANCES.get(name, 0))
        if imp == 0:
            continue
        score     = imp * (abs(val) if val != 0 else 0.1)
        direction = 'phishing' if (name in PHISHING_POSITIVE and val > 0) else 'legitimate'
        if name == 'uses_https' and val == 1:
            direction = 'legitimate'
        if name == 'has_www' and val == 1:
            direction = 'legitimate'
        results.append({
            'name':      name,
            'label':     FEATURE_LABELS.get(name, name),
            'value':     val,
            'impact':    round(score, 5),
            'direction': direction,
        })
    results.sort(key=lambda x: x['impact'], reverse=True)
    return results[:top_n]


# ── Core analysis function ────────────────────────────────────────────
def _analyse_url(raw_url):
    clean_url = normalise_url(str(raw_url).strip())
    features  = extract_features(clean_url)
    if features is None:
        return None, 'Could not parse URL.'

    # Whitelist check — bypass model for known-legitimate domains
    if _is_whitelisted(clean_url):
        return {
            'url':          raw_url,
            'verdict':      'Legitimate',
            'is_phishing':  False,
            'confidence':   2.0,
            'top_features': [],
            'whitelisted':  True,
        }, None

    X          = pd.DataFrame([features])
    prob       = float(model.predict_proba(X)[0][1])
    pred       = int(model.predict(X)[0])
    confidence = round(prob * 100, 1)

    return {
        'url':          raw_url,
        'verdict':      'Phishing' if pred == 1 else 'Legitimate',
        'is_phishing':  pred == 1,
        'confidence':   confidence,
        'top_features': explain_features(features),
        'whitelisted':  False,
    }, None


# ── Routes ────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyse', methods=['POST'])
def analyse():
    data = request.get_json(silent=True)
    if not data or 'url' not in data:
        return jsonify({'error': 'No URL provided.'}), 400
    result, err = _analyse_url(data['url'])
    if err:
        return jsonify({'error': err}), 400
    return jsonify(result)

@app.route('/analyse/batch', methods=['POST'])
def analyse_batch():
    data = request.get_json(silent=True)
    if not data or 'urls' not in data:
        return jsonify({'error': 'No URLs provided.'}), 400
    results = {}
    for url in data['urls'][:50]:
        result, err = _analyse_url(url)
        if result:
            results[url] = {
                'verdict':     result['verdict'],
                'is_phishing': result['is_phishing'],
                'confidence':  result['confidence'],
            }
        else:
            results[url] = {'error': err}
    return jsonify(results)

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=False, port=5000)
