"""Streamlit front-end for the Phishing URL Detector.

Launches a browser UI that POSTs URLs to the FastAPI service:

    streamlit run ui.py

Prerequisite: the backend must be running, e.g.
    uvicorn app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"
TIMEOUT_SECONDS = 15

st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="centered",
)

st.title("🛡️ Phishing URL Detector")
st.caption("Paste a URL and click **Analyze URL**. The backend must be running "
           "on `http://127.0.0.1:8000`.")

url_input = st.text_input(
    "URL to analyse",
    placeholder="https://paypal.com@secure-login.evil.tk/verify?user=1",
    label_visibility="collapsed",
)

analyze = st.button("Analyze URL", type="primary", width="stretch")


def render_result(payload: dict) -> None:
    """Show the colour-coded verdict plus the confidence score."""
    classification = payload.get("classification", "UNKNOWN")
    confidence = float(payload.get("confidence", 0.0))
    probability = float(payload.get("phishing_probability", 0.0))
    label = payload.get("url", "")

    if classification == "SAFE":
        st.success(f"✅ **SAFE** — confidence {confidence:.1%}")
    elif classification == "SUSPICIOUS":
        st.warning(f"⚠️ **SUSPICIOUS** — confidence {confidence:.1%}")
    elif classification == "PHISHING":
        st.error(f"🚨 **PHISHING** — confidence {confidence:.1%}")
    else:
        st.info(f"ℹ️ **{classification}** — confidence {confidence:.1%}")

    st.progress(min(max(confidence, 0.0), 1.0),
                text=f"Confidence: {confidence:.1%}")

    with st.expander("Details"):
        st.write(f"**Analysed URL:** `{label}`")
        st.write(f"**P(phishing):** {probability:.1%}")
        st.write(f"**Verdict:** {classification}")

        features = payload.get("features")
        if isinstance(features, dict) and features:
            st.dataframe(
                {"feature": list(features.keys()), "value": list(features.values())},
                width="stretch",
                height=320,
            )


if analyze:
    url = url_input.strip()
    if not url:
        st.error("Please enter a URL first.")
    else:
        try:
            response = requests.post(API_URL, json={"url": url}, timeout=TIMEOUT_SECONDS)
        except requests.exceptions.ConnectionError:
            st.error(
                "Could not reach the API at "
                f"`{API_URL}`. Start it with:\n\n"
                "`uvicorn app:app --host 127.0.0.1 --port 8000`"
            )
        except requests.exceptions.Timeout:
            st.error("The API request timed out. Please try again.")
        except requests.exceptions.RequestException as exc:
            st.error(f"Request failed: {exc}")
        else:
            if response.status_code == 200:
                render_result(response.json())
            else:
                try:
                    detail = response.json().get("detail", response.text)
                except ValueError:
                    detail = response.text
                st.error(f"API returned {response.status_code}: {detail}")

st.markdown(
    """
---
**Legend** — 🟢 Green: **Safe** · 🟡 Yellow: **Suspicious** · 🔴 Red: **Phishing**
"""
)
