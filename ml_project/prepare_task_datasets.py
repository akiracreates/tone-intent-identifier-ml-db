import os
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --- CONFIG ---
PROCESSED_DIR = "data/processed"
SEQUENCES_PATH = os.path.join(PROCESSED_DIR, "X_all_sequences.npy")
LABELS_PATH = os.path.join(PROCESSED_DIR, "labels_raw.csv")

# train/val/test split proportions
VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_STATE = 42


def encode_labels(column: pd.Series):
    """
    Take a pandas Series of string labels and:
    - drop NaNs
    - build mapping {label_str: index}
    - return (encoded_array, mapping_dict)
    """
    # ensure string type and drop NaNs
    labels_str = column.dropna().astype(str)

    # unique labels, sorted for stability
    unique_labels = sorted(labels_str.unique())
    mapping = {label: idx for idx, label in enumerate(unique_labels)}

    encoded = labels_str.map(mapping).values
    return encoded, mapping


def split_dataset(X, y, val_size=VAL_SIZE, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Split X, y into train / val / test with given proportions.
    """
    # first split off test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # now split train_val into train and val
    val_ratio = val_size / (1.0 - test_size)  # proportion of train_val that should go to val

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. Load sequences and labels
    print(f"Loading sequences from: {SEQUENCES_PATH}")
    X_all = np.load(SEQUENCES_PATH)

    print(f"Loading labels from: {LABELS_PATH}")
    labels_df = pd.read_csv(LABELS_PATH)

    print("X_all shape:", X_all.shape)
    print("Labels head:")
    print(labels_df.head())

    # --- TONE TASK ---
    print("\n=== Preparing TONE dataset ===")
    tone_mask = labels_df["tone_label"].notna()

    X_tone = X_all[tone_mask.values]
    y_tone_encoded, tone_mapping = encode_labels(labels_df.loc[tone_mask, "tone_label"])

    print("Tone classes mapping:", tone_mapping)
    print("X_tone shape:", X_tone.shape)
    print("y_tone shape:", y_tone_encoded.shape)

    X_tone_train, X_tone_val, X_tone_test, y_tone_train, y_tone_val, y_tone_test = split_dataset(
        X_tone, y_tone_encoded
    )

    # save tone splits
    np.save(os.path.join(PROCESSED_DIR, "tone_X_train.npy"), X_tone_train)
    np.save(os.path.join(PROCESSED_DIR, "tone_X_val.npy"),   X_tone_val)
    np.save(os.path.join(PROCESSED_DIR, "tone_X_test.npy"),  X_tone_test)

    np.save(os.path.join(PROCESSED_DIR, "tone_y_train.npy"), y_tone_train)
    np.save(os.path.join(PROCESSED_DIR, "tone_y_val.npy"),   y_tone_val)
    np.save(os.path.join(PROCESSED_DIR, "tone_y_test.npy"),  y_tone_test)

    with open(os.path.join(PROCESSED_DIR, "tone_label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(tone_mapping, f, ensure_ascii=False, indent=2)

    # --- INTENT TASK ---
    print("\n=== Preparing INTENT dataset ===")
    intent_mask = labels_df["intent_label"].notna()

    X_intent = X_all[intent_mask.values]
    y_intent_encoded, intent_mapping = encode_labels(labels_df.loc[intent_mask, "intent_label"])

    print("Intent classes mapping:", intent_mapping)
    print("X_intent shape:", X_intent.shape)
    print("y_intent shape:", y_intent_encoded.shape)

    X_intent_train, X_intent_val, X_intent_test, y_intent_train, y_intent_val, y_intent_test = split_dataset(
        X_intent, y_intent_encoded
    )

    # save intent splits
    np.save(os.path.join(PROCESSED_DIR, "intent_X_train.npy"), X_intent_train)
    np.save(os.path.join(PROCESSED_DIR, "intent_X_val.npy"),   X_intent_val)
    np.save(os.path.join(PROCESSED_DIR, "intent_X_test.npy"),  X_intent_test)

    np.save(os.path.join(PROCESSED_DIR, "intent_y_train.npy"), y_intent_train)
    np.save(os.path.join(PROCESSED_DIR, "intent_y_val.npy"),   y_intent_val)
    np.save(os.path.join(PROCESSED_DIR, "intent_y_test.npy"),  y_intent_test)

    with open(os.path.join(PROCESSED_DIR, "intent_label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(intent_mapping, f, ensure_ascii=False, indent=2)

    print("\nDone. Tone and intent datasets are prepared and saved in data/processed/.")


if __name__ == "__main__":
    main()
