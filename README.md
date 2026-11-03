# Phishing URL Detector

A production-style, three-tier phishing URL classifier:

| Tier | File | Role |
|------|------|------|
| Features | `extractor.py` | Extracts **30 structural/lexical features** from a raw URL (`urllib` + `re`) |
| Training | `train.py` | Builds a **balanced** SAFE/PHISHING dataset, trains Scikit-Learn models, saves `model.pkl` via `joblib` |
| API | `app.py` | FastAPI service with `POST /predict` → `{url, classification, confidence}` |
| UI | `ui.py` | Streamlit front-end with colour-coded verdicts (🟢 Safe / 🟡 Suspicious / 🔴 Phishing) |

The three classifications are **SAFE**, **SUSPICIOUS** and **PHISHING**, each
returned with a confidence score in `[0, 1]`.

## 1. Install requirements

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Train the model

```bash
python train.py
```

This generates a balanced `dataset.csv` (1500 SAFE + 1500 PHISHING URLs),
extracts the 30 features, trains a **Random Forest** and a **Gradient Boosting**
baseline, prints accuracy / macro-F1 / classification report / confusion matrix
/ top feature importances, and writes the winning estimator to **`model.pkl`**.

Use `python train.py --retrain` to regenerate the dataset from scratch.

## 3. Start the FastAPI backend

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

Interactive docs: <http://127.0.0.1:8000/docs>

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "https://paypal.com@secure-login.evil.tk/verify?user=1"}'
```

Response:

```json
{
  "url": "https://paypal.com@secure-login.evil.tk/verify?user=1",
  "classification": "PHISHING",
  "confidence": 0.94,
  "phishing_probability": 0.94,
  "features": { "url_length": 55, "...": "..." }
}
```

## 4. Launch the Streamlit UI

In a second terminal (same virtualenv), while the API from step 3 is running:

```bash
streamlit run ui.py
```

Then open <http://localhost:8501>, paste a URL and press **Analyze URL**.

## 5. Run everything with one command each (summary)

```bash
source .venv/bin/activate
pip install -r requirements.txt   # once
python train.py                   # once  -> model.pkl
uvicorn app:app --port 8000       # terminal 1
streamlit run ui.py               # terminal 2
```

## Notes

- **Feature set (30):** `url_length`, `hostname_length`, `path_length`,
  `query_length`, `is_ip_address`, `count_at`, `count_dash`, `count_underscore`,
  `count_question`, `count_equals`, `count_ampersand`, `count_percent`,
  `count_dot`, `count_slash`, `count_digits`, `count_uppercase`,
  `subdomain_depth`, `has_https`, `has_port`, `has_fragment`,
  `double_slash_redirect`, `digit_letter_ratio`, `suspicious_keyword_count`,
  `brand_keyword_count`, `uses_shortener`, `path_depth`, `tld_length`,
  `is_localhost`, `count_hex_escapes`, `query_param_count`.
- **Decision rule:** `P(phishing) ≥ 0.70` → PHISHING, `≤ 0.35` → SAFE,
  otherwise SUSPICIOUS (thresholds live in `app.py`).
- **Honesty disclaimer:** the training corpus is synthetic. This project
  demonstrates a complete ML pipeline; for real-world blocking, fine-tune on a
  labelled live feed (e.g. PhishTank / OpenPhish) before trusting its verdicts.
- Quick sanity check of the extractor alone: `python extractor.py`.
