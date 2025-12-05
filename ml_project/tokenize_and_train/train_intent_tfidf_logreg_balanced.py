from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

# ORIGINAL (non-oversampled) intent dataset
INTENT_SOURCE_PATH = DATA_PROCESSED / "intent_final_dataset.csv"
INTENT_LABEL_MAPPING_PATH = DATA_PROCESSED / "intent_label_mapping.json"

MODEL_PATH = MODELS_DIR / "intent_logreg_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "intent_tfidf_vectorizer.pkl"
REPORT_PATH = METRICS_DIR / "intent_logreg_report.txt"

# where we save test split for evaluate.py
INTENT_X_TEST_PATH = DATA_PROCESSED / "intent_X_test.npy"
INTENT_Y_TEST_PATH = DATA_PROCESSED / "intent_y_test.npy"


def oversample_train(texts, labels):
    """Oversample TRAIN ONLY to balance classes."""
    df_train = pd.DataFrame({"text": texts, "y": labels})
    groups = df_train.groupby("y")

    target_per_class = max(len(g) for _, g in groups)
    parts = []

    print("[INFO] Train class distribution BEFORE oversampling:")
    print(df_train["y"].value_counts())

    for y_val, g in groups:
        n = len(g)
        if n < target_per_class:
            sampled = g.sample(target_per_class, replace=True, random_state=42)
            print(f"  class {y_val}: {n} → {target_per_class} (oversample)")
        else:
            sampled = g
            print(f"  class {y_val}: {n} (no change)")
        parts.append(sampled)

    balanced = (
        pd.concat(parts, axis=0)
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )

    print("[INFO] Train class distribution AFTER oversampling:")
    print(balanced["y"].value_counts())

    return balanced["text"].tolist(), balanced["y"].values


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1] Loading ORIGINAL intent data (no oversampling)...")
    df = pd.read_csv(INTENT_SOURCE_PATH)
    df = df.dropna(subset=["intent_label"]).copy()

    texts = df["text"].astype(str).tolist()
    labels_str = df["intent_label"].astype(str).tolist()

    # mapping label -> index (consistent with earlier)
    with open(INTENT_LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
        label_to_idx = json.load(f)

    y = np.array([label_to_idx[label] for label in labels_str], dtype="int64")

    print("[2] Train/test split (70/30, stratified) on ORIGINAL data...")
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    # SAVE TEST SPLIT FOR EVALUATION (untouched!)
    np.save(INTENT_X_TEST_PATH, np.array(X_test_texts, dtype=object))
    np.save(INTENT_Y_TEST_PATH, y_test)
    print(f"[OK] Saved test split → {INTENT_X_TEST_PATH.name}, {INTENT_Y_TEST_PATH.name}")

    # 3. Oversample TRAIN ONLY
    print("[3] Oversampling TRAIN set to balance classes...")
    X_train_balanced, y_train_balanced = oversample_train(X_train_texts, y_train)

    # 4. TF-IDF vectorization (fit on balanced TRAIN)
    print("[4] TF-IDF vectorization (fit on train)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        min_df=2,
    )
    X_train_vec = vectorizer.fit_transform(X_train_balanced)
    X_test_vec = vectorizer.transform(X_test_texts)

    # 5. Train Logistic Regression
    print("[5] Training Logistic Regression...")
    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        multi_class="auto",
    )
    clf.fit(X_train_vec, y_train_balanced)

    # 6. Evaluate on ORIGINAL TEST (no oversampling)
    print("[6] Evaluating on held-out test set...")
    y_pred = clf.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    target_items = sorted(label_to_idx.items(), key=lambda kv: kv[1])
    target_names = [label for label, _ in target_items]
    labels_int = [idx for _, idx in target_items]

    report = classification_report(
        y_test,
        y_pred,
        labels=labels_int,
        target_names=target_names,
        digits=4,
    )

    print(f"Test accuracy: {acc:.4f}")
    print(report)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Test accuracy: {acc:.4f}\n\n")
        f.write(report)

    print(f"[OK] Saved report to {REPORT_PATH}")

    # 7. Save model + vectorizer
    print("[7] Saving model + vectorizer...")
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"[OK] Saved model to {MODEL_PATH}")
    print(f"[OK] Saved vectorizer to {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
