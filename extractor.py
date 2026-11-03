"""Feature extraction for phishing URL detection.

Extracts 30 structural and lexical features from a raw URL using only the
standard library (urllib, re, ipaddress). The feature order is defined by
``FEATURE_NAMES`` and must stay stable: training (train.py) and inference
(app.py) both rely on it.
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit

import numpy as np

# --- Vocabulary used by the lexical features -------------------------------

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "sign-in", "verify", "verification", "secure",
    "account", "update", "confirm", "password", "passwd", "banking",
    "wallet", "billing", "invoice", "refund", "gift", "prize", "winner",
    "bonus", "lucky", "click", "urgent", "alert", "validate", "recovery",
    "credential", "ssn", "oauth", "webscr", "ebayisapi", "suspend",
    "restricted", "unusual", "activity", "unlock", "validate", "session",
    "authenticate", "authorization", "wallet", "bitcoin", "crypto",
    "airdrop", "claim", "lottery", "kyc", "confirm", "unlock",
]

BRAND_KEYWORDS = [
    "paypal", "apple", "google", "microsoft", "amazon", "facebook",
    "instagram", "netflix", "whatsapp", "binance", "coinbase", "dropbox",
    "linkedin", "github", "yahoo", "outlook", "steam", "spotify", "icloud",
    "chase", "wellsfargo", "alibaba", "tiktok", "uber", "dhl", "fedex",
    "usps", "office", "onedrive", "telegram", "discord", "revolut",
    "kraken", "metamask", "aol", "proton", "signin",
]

SHORTENERS = [
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "cutt.ly", "rebrand.ly", "shorturl.at", "lnkd.in",
    "rb.gy", "tiny.cc", "url.ms", "v.gd", "s.id", "shorte.st",
]

SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "work", "click",
    "link", "live", "info", "biz", "rest", "loan", "country", "stream",
    "download", "review", "science", "date", "racing", "win",
}

HEX_ESCAPE_RE = re.compile(r"%[0-9a-fA-F]{2}")
DIGIT_RE = re.compile(r"\d")

# Ordered list of the 30 extracted features. DO NOT reorder.
FEATURE_NAMES: list[str] = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "is_ip_address",
    "count_at",
    "count_dash",
    "count_underscore",
    "count_question",
    "count_equals",
    "count_ampersand",
    "count_percent",
    "count_dot",
    "count_slash",
    "count_digits",
    "count_uppercase",
    "subdomain_depth",
    "has_https",
    "has_port",
    "has_fragment",
    "double_slash_redirect",
    "digit_letter_ratio",
    "suspicious_keyword_count",
    "brand_keyword_count",
    "uses_shortener",
    "path_depth",
    "tld_length",
    "is_localhost",
    "count_hex_escapes",
    "query_param_count",
]

assert len(FEATURE_NAMES) == 30, "FEATURE_NAMES must contain exactly 30 features"


def _contains_any(haystack: str, needles: list[str]) -> int:
    """Count how many vocabulary entries appear as substrings."""
    return sum(1 for word in needles if word in haystack)


def extract_features(url: str) -> dict[str, float]:
    """Extract the 30 phishing-indicative features from a raw URL string.

    Returns a dict keyed by ``FEATURE_NAMES`` with numeric values, ready to be
    fed into a scikit-learn estimator as a single-row feature vector.
    """
    raw = (url or "").strip()
    lowered = raw.lower()

    try:
        parts = urlsplit(raw)
    except ValueError:
        parts = None

    scheme = (parts.scheme or "").lower() if parts else ""
    hostname = (parts.hostname or "") if parts else ""
    path = (parts.path or "") if parts else ""
    query = (parts.query or "") if parts else ""
    fragment = (parts.fragment or "") if parts else ""
    netloc = (parts.netloc or "") if parts else ""

    host_lower = hostname.lower()

    # --- host / domain structure -----------------------------------------
    host_labels = [label for label in host_lower.split(".") if label]
    # "a.b.c.example.com" -> 2 subdomains (a, b) under example.com
    subdomain_depth = max(len(host_labels) - 2, 0) if len(host_labels) >= 2 else 0

    tld = host_labels[-1] if host_labels else ""

    try:
        ipaddress.ip_address(hostname)
        is_ip = 1
    except ValueError:
        is_ip = 0

    is_localhost = 1 if host_lower in {"localhost", "127.0.0.1", "::1"} or host_lower.endswith(".local") else 0

    # --- counts ------------------------------------------------------------
    letters = sum(ch.isalpha() for ch in raw)
    digits = len(DIGIT_RE.findall(raw))
    denom = letters + digits
    digit_letter_ratio = round(digits / denom, 4) if denom else 0.0

    query_param_count = len([p for p in query.split("&") if p]) if query else 0

    features: dict[str, float] = {
        "url_length": len(raw),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),
        "is_ip_address": is_ip,
        "count_at": raw.count("@"),
        "count_dash": raw.count("-"),
        "count_underscore": raw.count("_"),
        "count_question": raw.count("?"),
        "count_equals": raw.count("="),
        "count_ampersand": raw.count("&"),
        "count_percent": raw.count("%"),
        "count_dot": raw.count("."),
        "count_slash": raw.count("/"),
        "count_digits": digits,
        "count_uppercase": sum(ch.isupper() for ch in raw),
        "subdomain_depth": subdomain_depth,
        "has_https": 1 if scheme == "https" else 0,
        "has_port": 1 if (parts is not None and parts.port is not None) else 0,
        "has_fragment": 1 if fragment else 0,
        # "http://evil.com//path" or "http:/" style obfuscation
        "double_slash_redirect": 1 if "//" in path else 0,
        "digit_letter_ratio": digit_letter_ratio,
        "suspicious_keyword_count": _contains_any(lowered, SUSPICIOUS_KEYWORDS),
        "brand_keyword_count": _contains_any(lowered, BRAND_KEYWORDS),
        "uses_shortener": 1 if any(s in host_lower for s in SHORTENERS) else 0,
        "path_depth": len([seg for seg in path.split("/") if seg]),
        "tld_length": len(tld),
        "is_localhost": is_localhost,
        "count_hex_escapes": len(HEX_ESCAPE_RE.findall(raw)),
        "query_param_count": query_param_count,
    }

    # Defensive guard: a typo in the dict above must never reach the model.
    if set(features) != set(FEATURE_NAMES):
        missing = set(FEATURE_NAMES) - set(features)
        extra = set(features) - set(FEATURE_NAMES)
        raise RuntimeError(f"Feature mismatch. missing={missing} extra={extra}")

    return features


def extract_vector(url: str) -> np.ndarray:
    """Return the feature dict as a 1-D float array in ``FEATURE_NAMES`` order."""
    feats = extract_features(url)
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=float)


def extract_matrix(urls: list[str]) -> np.ndarray:
    """Return a 2-D float matrix of shape (len(urls), 30)."""
    return np.vstack([extract_vector(u) for u in urls])


if __name__ == "__main__":
    demo = "https://login.paypal.com@secure-update-123.evil.tk/verify?user=a&token=bb%2F"
    for name, value in extract_features(demo).items():
        print(f"{name:26s} {value}")
