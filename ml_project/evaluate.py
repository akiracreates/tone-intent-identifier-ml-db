import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

# tone data
INTENT_X_TEST_PATH = DATA_PROCESSED / "intent_X_test.npy"
INTENT_Y_TEST_PATH = DATA_PROCESSED / "intent_y_test.npy"
TONE_LABEL_MAPPING_PATH = DATA_PROCESSED / "tone_label_mapping.json"

# intent data
INTENT_LABEL_MAPPING_PATH = DATA_PROCESSED / "intent_label_mapping.json"

# models
TONE_BILSTM_MODEL_PATH = MODELS_DIR / "tone_bilstm_model_final.h5"
TONE_CNN_MODEL_PATH = MODELS_DIR / "tone_cnn_model_final.h5"
INTENT_MODEL_PATH = MODELS_DIR / "intent_logreg_model.pkl"
INTENT_VECTORIZER_PATH = MODELS_DIR / "intent_tfidf_vectorizer.pkl"


def _ensure_metrics_dir():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def _load_idx_to_label(mapping_path: Path):
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    # expect label -> index; invert
    idx_to_label = {}
    for label, idx in mapping.items():
        idx_to_label[int(idx)] = label
    # make sure order is stable
    return [idx_to_label[i] for i in sorted(idx_to_label.keys())]


def _plot_and_save_confusion_matrix(cm, labels, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# =========================
#   TONE EVALUATION
# =========================

def evaluate_tone_model(model_path: Path, report_name: str, cm_name: str):
    print(f"\n[Eval] Tone model: {model_path.name}")
    _ensure_metrics_dir()

    X_test = np.load(TONE_X_TEST_PATH)
    y_test = np.load(TONE_Y_TEST_PATH)

    labels = _load_idx_to_label(TONE_LABEL_MAPPING_PATH)
    num_classes = len(labels)

    model = load_model(model_path)
    y_probs = model.predict(X_test, batch_size=512, verbose=0)
    y_pred = y_probs.argmax(axis=1)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(num_classes)),
        target_names=labels,
        digits=4,
    )

    report_path = METRICS_DIR / report_name
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_path.name}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n\n")
        f.write(report)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=list(range(num_classes)),
    )
    cm_path = METRICS_DIR / cm_name
    _plot_and_save_confusion_matrix(
        cm,
        labels,
        cm_path,
        title=f"Confusion matrix — {model_path.name}",
    )

    print(f"  → Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}")
    print(f"  → report: {report_path}")
    print(f"  → confusion matrix: {cm_path}")


# =========================
#   INTENT EVALUATION
# =========================

def evaluate_intent_model(report_name: str, cm_name: str):
    print("\n[Eval] Intent TF-IDF + Logistic Regression")
    _ensure_metrics_dir()

    # 1. Load saved test split (created in the training script)
    X_test_texts = np.load(INTENT_X_TEST_PATH, allow_pickle=True)
    y_test = np.load(INTENT_Y_TEST_PATH)

    # make sure we work with plain Python strings
    X_test_texts = [str(t) for t in X_test_texts.tolist()]

    # 2. Load label mapping: label(str) -> index(int)
    with open(INTENT_LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
        label_to_idx = json.load(f)

    # build reverse mapping: index(int) -> label(str)
    idx_to_label = {int(v): k for k, v in label_to_idx.items()}

    # 3. Load vectorizer & model
    vectorizer = joblib.load(INTENT_VECTORIZER_PATH)
    model = joblib.load(INTENT_MODEL_PATH)

    # 4. Transform texts and predict
    X_test_vec = vectorizer.transform(X_test_texts)
    y_pred = model.predict(X_test_vec)  # ints, same space as y_test

    # 5. Labels as ints, sorted; names as strings
    labels = sorted(set(model.classes_))
    target_names = [idx_to_label[int(i)] for i in labels]

    # 6. Metrics
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=4,
    )

    report_path = METRICS_DIR / report_name
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Model: intent_logreg_model.pkl\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n\n")
        f.write(report)

    # 7. Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_path = METRICS_DIR / cm_name
    _plot_and_save_confusion_matrix(
        cm,
        target_names,
        cm_path,
        title="Confusion matrix — intent LogReg",
    )

    print(f"  → Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}")
    print(f"  → report: {report_path}")
    print(f"  → confusion matrix: {cm_path}")



# =========================
#   CLI ENTRY POINT
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate tone and intent models."
    )
    parser.add_argument(
        "--task",
        choices=["tone_bilstm", "tone_cnn", "intent", "all"],
        default="all",
        help="Which model(s) to evaluate.",
    )
    args = parser.parse_args()

    if args.task in ("tone_bilstm", "all"):
        evaluate_tone_model(
            TONE_BILSTM_MODEL_PATH,
            report_name="tone_bilstm_report.txt",
            cm_name="tone_bilstm_confusion_matrix.png",
        )

    if args.task in ("tone_cnn", "all"):
        evaluate_tone_model(
            TONE_CNN_MODEL_PATH,
            report_name="tone_cnn_report.txt",
            cm_name="tone_cnn_confusion_matrix.png",
        )

    if args.task in ("intent", "all"):
        evaluate_intent_model(
            report_name="intent_logreg_eval_report.txt",
            cm_name="intent_logreg_confusion_matrix.png",
        )


if __name__ == "__main__":
    main()
