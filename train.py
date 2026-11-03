"""Train a phishing-URL classifier and persist it to ``model.pkl``.

Usage:
    python train.py            # build dataset (or load dataset.csv) -> train
    python train.py --retrain  # ignore dataset.csv and regenerate it

The dataset is balanced 50/50 between SAFE and PHISHING URLs. If a
``dataset.csv`` produced by this script already exists it is loaded instead of
being regenerated, so repeated runs are fast and reproducible.
"""

from __future__ import annotations

import argparse
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from extractor import FEATURE_NAMES, extract_matrix

MODEL_PATH = "model.pkl"
DATASET_PATH = "dataset.csv"
RANDOM_STATE = 42

# --- Safe URL corpus -------------------------------------------------------
SAFE_DOMAINS = [
    "google.com", "www.google.co.uk", "github.com", "docs.python.org",
    "en.wikipedia.org", "stackoverflow.com", "microsoft.com", "apple.com",
    "www.amazon.com", "nytimes.com", "www.bbc.co.uk", "nasa.gov",
    "medium.com", "www.reddit.com", "www.linkedin.com", "netflix.com",
    "arxiv.org", "www.weather.com", "www.twitch.tv", "openai.com",
    "www.cloudflare.com", "huggingface.co", "gitlab.com", "mozilla.org",
    "www.python.org", "www.kernel.org", "statista.com", "www.imdb.com",
    "www.dropbox.com", "zoom.us", "www.spotify.com", "www.notion.so",
    "stripe.com", "www.fedex.com", "www.wikipedia.org", "coursera.org",
    "www.nasa.gov", "bankofamerica.com", "www.chase.com", "example.org",
    "www.whitehouse.gov", "health.nih.gov", "www.gov.uk", "debian.org",
    "ubuntu.com", "www.jetbrains.com", "www.npmjs.com", "pypi.org",
    "www.docker.com", "www.salesforce.com", "www.adobe.com", "www.ibm.com",
    "www.oracle.com", "www.intel.com", "www.nvidia.com", "www.samsung.com",
    "www.bmw.com", "www.airbnb.com", "www.booking.com", "www.spotify.com",
    "www.espn.com", "www.theguardian.com", "www.reuters.com", "www.forbes.com",
    "www.w3.org", "www.ietf.org", "www.gnu.org", "www.kernel.org",
    "scholar.google.com", "www.nature.com", "www.sciencedirect.com",
    "www.senate.gov", "www.fda.gov", "www.irs.gov", "www.cdc.gov",
]

SAFE_PATHS = [
    "/", "/search", "/search?q=python+tutorial", "/products", "/products/list?page=2",
    "/news/2026/03/technology", "/docs/getting-started", "/about", "/contact",
    "/account/login?next=/home", "/users/sign_in", "/blog/fastapi-guide",
    "/api/v1/items?limit=25&offset=0", "/images/logo.png", "/category/books",
    "/watch?v=dQw4w9WgXcQ", "/downloads/release-1.4.2.zip", "/help/faq",
    "/settings/profile", "/posts/2026/why-urls-matter", "/pricing?plan=pro",
    "/repository/main/README.md", "/maps/@51.5,-0.12,13z", "/trending",
]

SAFE_SCHEMES = ["https", "https", "https", "http"]  # mostly https, some http


# --- Phishing URL patterns -------------------------------------------------
PHISH_TLDS = ["tk", "ml", "xyz", "ga", "cf", "top", "click", "info", "biz", "live"]
BRANDS = ["paypal", "apple", "google", "microsoft", "amazon", "netflix",
          "facebook", "whatsapp", "binance", "coinbase", "instagram", "github"]
ACTIONS = ["login", "signin", "verify", "secure", "update", "confirm",
           "unlock", "validate", "recover", "activate"]
DECOY_TLDS = ["com", "net", "org", "co", "io"]
EVIL_WORDS = ["account-secure", "verify-now", "auth-check", "session-restore",
              "update-billing", "confirm-identity", "safe-login", "member-area"]
IP_OCTETS = ["192.168", "10.0", "172.16", "8.8.8", "45.33", "185.212"]


def build_dataset(n_per_class: int = 1500, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate a balanced dataset of SAFE and PHISHING URLs with labels."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, str]] = []

    # --- SAFE URLs: real-looking hosts with ordinary paths ----------------
    for _ in range(n_per_class):
        scheme = rng.choice(SAFE_SCHEMES)
        host = str(rng.choice(SAFE_DOMAINS))
        path = str(rng.choice(SAFE_PATHS))
        url = f"{scheme}://{host}{path}"
        # Occasionally add a benign fragment or trailing slash variation.
        if rng.random() < 0.15:
            url += f"#section-{int(rng.integers(1, 99))}"
        # Occasionally append ordinary analytics parameters for realism.
        if rng.random() < 0.35:
            sep = "&" if "?" in url else "?"
            url += f"{sep}utm_source={str(rng.choice(['newsletter', 'blog', 'search', 'social']))}&ref={int(rng.integers(1000, 9999))}"
        rows.append({"url": url, "label": "SAFE"})

    # --- PHISHING URLs: several classic lure families ---------------------
    for i in range(n_per_class):
        brand = str(rng.choice(BRANDS))
        action = str(rng.choice(ACTIONS))
        evil = str(rng.choice(EVIL_WORDS))
        tld = str(rng.choice(PHISH_TLDS))
        decoy = str(rng.choice(DECOY_TLDS))
        family = i % 10

        if family == 0:
            # Brand + keyword hyphenated subdomain
            url = (f"http://{brand}-{evil}.{decoy}/"
                   f"{action}.php?user={brand}&session={int(rng.integers(10**6, 10**7))}")
        elif family == 1:
            # '@' credential confusion: real brand, attacker host
            url = (f"http://{brand}.com@{evil}.{tld}/{action}"
                   f"?redirect=https%3A%2F%2F{brand}.{tld}%2F{action}")
        elif family == 2:
            # Raw IP host
            ip = f"{rng.choice(IP_OCTETS)}.{int(rng.integers(2, 254))}"
            url = f"http://{ip}/{brand}/{action}?acct={int(rng.integers(10**4, 10**5))}&otp=0000"
        elif family == 3:
            # Deep subdomain stack hiding the real domain
            layers = ".".join(f"{rng.choice(ACTIONS)}-{int(rng.integers(10, 99))}"
                              for _ in range(int(rng.integers(3, 6))))
            url = f"https://{layers}.{brand}.{tld}/{action}?id={int(rng.integers(10**5, 10**6))}"
        elif family == 4:
            # Homoglyph / typosquat brand
            typo = brand.replace("a", "4").replace("e", "3").replace("o", "0")
            if typo == brand:
                typo = brand + str(int(rng.integers(10, 99)))
            url = f"https://{typo}-{evil}.{tld}/{action}/index.php?ref=bonus"
        elif family == 5:
            # URL shortener wrapping a phishing lure
            short = str(rng.choice(["bit.ly", "tinyurl.com", "t.co", "rb.gy", "cutt.ly"]))
            url = f"https://{short}/{int(rng.integers(10**6, 10**7))}"
        elif family == 6:
            # Long obfuscated query string with many parameters
            params = "&".join(
                f"{rng.choice(ACTIONS)}{k}={int(rng.integers(10**3, 10**4))}"
                for k in range(int(rng.integers(6, 12)))
            )
            url = f"http://secure-{brand}-portal.{tld}/{action}?{params}"
        elif family == 7:
            # Percent-encoding and mixed case to dodge filters
            url = (f"http://WWW.{brand.upper()}-{evil}.{decoy}/"
                   f"%2F{action}%2Fverify.php%3Fuser%3D{brand}")
        elif family == 8:
            # Underscore-heavy host + many dashes + port
            port = int(rng.choice([8080, 8443, 8081, 4444]))
            url = (f"http://login_{brand}_secure-{int(rng.integers(100, 999))}.{tld}:"
                   f"{port}/{action}_account?user={brand}&pass=1")
        else:
            # Brand as path segment on an unrelated decoy host
            url = (f"https://www.{evil}.{decoy}/{brand}/{action}"
                   f"?email={brand}%40victim.com&token={int(rng.integers(10**7, 10**8))}")

        rows.append({"url": url, "label": "PHISHING"})

    df = pd.DataFrame(rows, columns=["url", "label"])
    df = df.drop_duplicates(subset="url").reset_index(drop=True)

    # De-duplication can skew the ratio: downsample every class to the size of
    # the smallest one so the training set is exactly balanced.
    min_count = df["label"].value_counts().min()
    parts = [g.sample(n=min_count, random_state=seed) for _, g in df.groupby("label")]
    df = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def load_or_build_dataset(rebuild: bool = False) -> pd.DataFrame:
    """Load ``dataset.csv`` when present, otherwise generate a fresh one."""
    if not rebuild:
        try:
            df = pd.read_csv(DATASET_PATH)
            if {"url", "label"}.issubset(df.columns) and len(df) > 0:
                print(f"[dataset] Loaded {len(df)} rows from {DATASET_PATH}")
                return df
        except FileNotFoundError:
            pass

    print("[dataset] Generating balanced synthetic dataset ...")
    df = build_dataset()
    df.to_csv(DATASET_PATH, index=False)
    print(f"[dataset] Wrote {len(df)} rows to {DATASET_PATH}")
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the phishing URL detector.")
    parser.add_argument("--retrain", action="store_true",
                        help="Regenerate dataset.csv even if it already exists.")
    parser.add_argument("--model", default=MODEL_PATH, help="Output path for the model.")
    args = parser.parse_args()

    df = load_or_build_dataset(rebuild=args.retrain)

    class_counts = df["label"].value_counts()
    print(f"[dataset] Class balance:\n{class_counts.to_string()}")
    if class_counts.min() < 100:
        print("[dataset] WARNING: fewer than 100 samples in a class.", file=sys.stderr)

    print(f"[features] Extracting {len(FEATURE_NAMES)} features ...")
    X = extract_matrix(df["url"].tolist())
    y = df["label"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("[model] Training RandomForest (class_weight=balanced) ...")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # A Gradient Boosting baseline is trained too; we keep whichever generalises
    # better on the hold-out set.
    print("[model] Training GradientBoosting baseline ...")
    gb = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=3,
        random_state=RANDOM_STATE,
    )
    gb.fit(X_train, y_train)

    candidates = {"RandomForest": clf, "GradientBoosting": gb}
    scores = {
        name: f1_score(y_test, model.predict(X_test), average="macro")
        for name, model in candidates.items()
    }
    best_name = max(scores, key=scores.get)
    best_model = candidates[best_name]
    print(f"[model] Hold-out macro-F1: "
          + ", ".join(f"{k}={v:.4f}" for k, v in scores.items()))
    print(f"[model] Selected estimator: {best_name}")

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS (hold-out test set)")
    print("=" * 60)
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro-F1 : {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    labels = ["PHISHING", "SAFE"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print(f"{'':14s} {'pred PHISHING':>14s} {'pred SAFE':>10s}")
    for label, row in zip(labels, cm):
        print(f"actual {label:9s} {row[0]:14d} {row[1]:10d}")

    # Feature importances (top 10) for interpretability.
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=FEATURE_NAMES)
        print("\nTop 10 features:")
        for name, imp in importances.sort_values(ascending=False).head(10).items():
            print(f"  {name:26s} {imp:.4f}")

    bundle = {
        "model": best_model,
        "feature_names": FEATURE_NAMES,
        "estimator_name": best_name,
        "metrics": {"accuracy": float(acc), "macro_f1": float(f1)},
    }
    joblib.dump(bundle, args.model)
    print(f"\n[model] Saved trained model to {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
