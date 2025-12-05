import argparse
import json
from pathlib import Path

from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"

TOKENIZER_PATH = DATA_PROCESSED / "tokenizer.pkl"
TONE_LABEL_MAPPING_PATH = DATA_PROCESSED / "tone_label_mapping.json"
INTENT_LABEL_MAPPING_PATH = DATA_PROCESSED / "intent_label_mapping.json"

TONE_BILSTM_MODEL_PATH = MODELS_DIR / "tone_bilstm_model_final.h5"
TONE_CNN_MODEL_PATH = MODELS_DIR / "tone_cnn_model_final.h5"

INTENT_MODEL_PATH = MODELS_DIR / "intent_logreg_model.pkl"
INTENT_VECTORIZER_PATH = MODELS_DIR / "intent_tfidf_vectorizer.pkl"

MAX_SEQ_LEN = 40


# lazy-loaded globals
_TOKENIZER = None
_TONE_BILSTM = None
_TONE_CNN = None
_INTENT_VECTORIZER = None
_INTENT_MODEL = None
_TONE_LABELS = None
_INTENT_LABELS = None


def _ensure_predictions_dir():
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


def _load_idx_to_label(mapping_path: Path):
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    idx_to_label = {}
    for label, idx in mapping.items():
        idx_to_label[int(idx)] = label
    return [idx_to_label[i] for i in sorted(idx_to_label.keys())]


def _load_artifacts():
    global _TOKENIZER, _TONE_BILSTM, _TONE_CNN
    global _INTENT_VECTORIZER, _INTENT_MODEL
    global _TONE_LABELS, _INTENT_LABELS

    if _TOKENIZER is None:
        _TOKENIZER = joblib.load(TOKENIZER_PATH)

    if _TONE_BILSTM is None:
        _TONE_BILSTM = load_model(TONE_BILSTM_MODEL_PATH)

    if _TONE_CNN is None:
        _TONE_CNN = load_model(TONE_CNN_MODEL_PATH)

    if _INTENT_VECTORIZER is None:
        _INTENT_VECTORIZER = joblib.load(INTENT_VECTORIZER_PATH)

    if _INTENT_MODEL is None:
        _INTENT_MODEL = joblib.load(INTENT_MODEL_PATH)

    if _TONE_LABELS is None:
        _TONE_LABELS = _load_idx_to_label(TONE_LABEL_MAPPING_PATH)

    if _INTENT_LABELS is None:
        _INTENT_LABELS = _load_idx_to_label(INTENT_LABEL_MAPPING_PATH)


def _preprocess_texts(texts):
    _load_artifacts()
    seqs = _TOKENIZER.texts_to_sequences(texts)
    X = pad_sequences(
        seqs,
        maxlen=MAX_SEQ_LEN,
        padding="post",
        truncating="post",
    )
    return X


def _probs_to_dict(labels, probs_row):
    return {label: float(p) for label, p in zip(labels, probs_row)}


def predict_single(text: str) -> dict:
    _load_artifacts()

    # --- Tone ---
    X_tone = _preprocess_texts([text])

    probs_bilstm = _TONE_BILSTM.predict(X_tone, verbose=0)[0]
    probs_cnn = _TONE_CNN.predict(X_tone, verbose=0)[0]

    idx_bilstm = int(np.argmax(probs_bilstm))
    idx_cnn = int(np.argmax(probs_cnn))

    tone_bilstm_label = _TONE_LABELS[idx_bilstm]
    tone_cnn_label = _TONE_LABELS[idx_cnn]

    # --- Intent ---
    X_intent = _INTENT_VECTORIZER.transform([text])
    intent_probs = _INTENT_MODEL.predict_proba(X_intent)[0]

    intent_idx = int(np.argmax(intent_probs))
    intent_label = _INTENT_LABELS[intent_idx]  # ← decode to string label

    return {
        "text": text,
        "tone_bilstm_pred": tone_bilstm_label,
        "tone_bilstm_probs": _probs_to_dict(_TONE_LABELS, probs_bilstm),
        "tone_cnn_pred": tone_cnn_label,
        "tone_cnn_probs": _probs_to_dict(_TONE_LABELS, probs_cnn),
        "intent_pred": intent_label,
        "intent_probs": _probs_to_dict(_INTENT_LABELS, intent_probs),
        "predicted_at": datetime.utcnow().isoformat(),
    }


def predict_file(input_path: Path, output_path: Path):
    _load_artifacts()
    _ensure_predictions_dir()

    df = pd.read_csv(input_path)
    if "text" not in df.columns:
        raise ValueError(
            f"Input file {input_path} must contain a 'text' column."
        )

    texts = df["text"].astype(str).tolist()

    # --- Tone ---
    X_tone = _preprocess_texts(texts)
    probs_bilstm = _TONE_BILSTM.predict(X_tone, verbose=0)
    probs_cnn = _TONE_CNN.predict(X_tone, verbose=0)

    idx_bilstm = probs_bilstm.argmax(axis=1)
    idx_cnn = probs_cnn.argmax(axis=1)

    df["tone_bilstm_pred"] = [
        _TONE_LABELS[i] for i in idx_bilstm
    ]
    df["tone_cnn_pred"] = [
        _TONE_LABELS[i] for i in idx_cnn
    ]

    for class_idx, label in enumerate(_TONE_LABELS):
        df[f"tone_bilstm_prob_{label}"] = probs_bilstm[:, class_idx]
        df[f"tone_cnn_prob_{label}"] = probs_cnn[:, class_idx]

    # --- Intent ---
    X_intent = _INTENT_VECTORIZER.transform(texts)
    intent_probs = _INTENT_MODEL.predict_proba(X_intent)

    # argmax over columns → integer indices / classes
    intent_idx = intent_probs.argmax(axis=1)
    df["intent_pred"] = [
        _INTENT_LABELS[i] for i in intent_idx
    ]

    for class_idx, label in enumerate(_INTENT_LABELS):
        df[f"intent_prob_{label}"] = intent_probs[:, class_idx]

    df["predicted_at"] = datetime.utcnow().isoformat()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run tone + intent predictions."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--text",
        type=str,
        help="Single text string to classify.",
    )
    group.add_argument(
        "--input",
        type=str,
        help="Path to CSV file with a 'text' column.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(PREDICTIONS_DIR / "predictions.csv"),
        help="Where to save predictions for CSV mode.",
    )

    args = parser.parse_args()

    if args.text:
        result = predict_single(args.text)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        input_path = Path(args.input)
        output_path = Path(args.output)
        predict_file(input_path, output_path)


if __name__ == "__main__":
    main()
