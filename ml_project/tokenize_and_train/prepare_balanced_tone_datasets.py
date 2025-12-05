from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

TONE_BALANCED_PATH = DATA_PROCESSED / "tone_balanced_for_training.csv"
TOKENIZER_PATH = DATA_PROCESSED / "tokenizer.pkl"
TONE_LABEL_MAPPING_PATH = DATA_PROCESSED / "tone_label_mapping.json"

# where we will overwrite the old splits
TONE_X_TRAIN = DATA_PROCESSED / "tone_X_train.npy"
TONE_X_VAL = DATA_PROCESSED / "tone_X_val.npy"
TONE_X_TEST = DATA_PROCESSED / "tone_X_test.npy"

TONE_Y_TRAIN = DATA_PROCESSED / "tone_y_train.npy"
TONE_Y_VAL = DATA_PROCESSED / "tone_y_val.npy"
TONE_Y_TEST = DATA_PROCESSED / "tone_y_test.npy"

MAX_SEQ_LEN = 40  # same as in other scripts


def main():
    print("[1] Loading balanced tone data...")
    df = pd.read_csv(TONE_BALANCED_PATH)

    texts = df["text"].astype(str).tolist()
    tone_labels = df["tone_label"].astype(str).tolist()

    # load tokenizer
    print("[2] Loading tokenizer...")
    tokenizer = joblib.load(TOKENIZER_PATH)

    print("[3] Converting texts to padded sequences...")
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(
        sequences,
        maxlen=MAX_SEQ_LEN,
        padding="post",
        truncating="post",
    )

    # load label mapping: label -> index
    print("[4] Encoding labels with tone_label_mapping.json...")
    with open(TONE_LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
        label_to_idx = json.load(f)

    y = np.array([label_to_idx[label] for label in tone_labels], dtype="int64")

    # 70 / 15 / 15 split
    print("[5] Splitting into train/val/test (70/15/15, stratified)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,  # 0.15 of total
        random_state=42,
        stratify=y_temp,
    )

    print("[6] Saving .npy files (overwriting old tone splits)...")
    np.save(TONE_X_TRAIN, X_train)
    np.save(TONE_X_VAL, X_val)
    np.save(TONE_X_TEST, X_test)

    np.save(TONE_Y_TRAIN, y_train)
    np.save(TONE_Y_VAL, y_val)
    np.save(TONE_Y_TEST, y_test)

    print("[OK] Done.")
    print(f"  Train size: {len(y_train)}")
    print(f"  Val size:   {len(y_val)}")
    print(f"  Test size:  {len(y_test)}")


if __name__ == "__main__":
    main()