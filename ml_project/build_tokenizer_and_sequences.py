import os
import pickle

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# --- CONFIG: adjust if needed ---
DATA_PATH = "data/processed/combined_clean.csv" # final dataset
OUTPUT_DIR = "data/processed" # where to save outputs

MAX_VOCAB_SIZE = 20000
MAX_SEQ_LEN = 40 #(messages are short)


def main():
    # Load dataset
    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    #Safety: make sure 'text' column exists
    if "text" not in df.columns:
        raise ValueError("Column 'text' not found in the dataset")

    #Convert to string and fill NaN
    texts = df["text"].astype(str).fillna("")

    print(f"Total rows: {len(texts)}")

    # Create and fit tokenizer
    print("Fitting tokenizer...")
    tokenizer = Tokenizer(
        num_words=MAX_VOCAB_SIZE,
        oov_token="<OOV>"   # token for unknown words
    )
    tokenizer.fit_on_texts(texts.tolist())

    #Convert texts to sequences of integers
    print("Converting texts to sequences...")
    sequences = tokenizer.texts_to_sequences(texts.tolist())

    #Pad sequences to fixed length
    print(f"Padding sequences to length = {MAX_SEQ_LEN}...")
    X_all = pad_sequences(
        sequences,
        maxlen=MAX_SEQ_LEN,
        padding="post",
        truncating="post"
    )

    print("X_all shape:", X_all.shape)  # (num_samples, MAX_SEQ_LEN)

    # Create output dir if not exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #  Save tokenizer
    tokenizer_path = os.path.join(OUTPUT_DIR, "tokenizer.pkl")
    print(f"Saving tokenizer to: {tokenizer_path}")
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)

    #Save sequences as .npy
    sequences_path = os.path.join(OUTPUT_DIR, "X_all_sequences.npy")
    print(f"Saving sequences to: {sequences_path}")
    np.save(sequences_path, X_all)

    # save raw labels to a separate CSV
    labels_path = os.path.join(OUTPUT_DIR, "labels_raw.csv")
    print(f"Saving labels snapshot to: {labels_path}")
    df[["id", "tone_label", "intent_label"]].to_csv(labels_path, index=False)

    print("Done. Shared tokenizer + sequences are ready.")


if __name__ == "__main__":
    main()
