import os
import pickle
import json

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- CONFIG ---
PROCESSED_DIR = "data/processed"
OUTPUT_MODEL_DIR = "outputs/models"
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

TOKENIZER_PATH = os.path.join(PROCESSED_DIR, "tokenizer.pkl")

TONE_X_TRAIN = os.path.join(PROCESSED_DIR, "tone_X_train.npy")
TONE_X_VAL   = os.path.join(PROCESSED_DIR, "tone_X_val.npy")
TONE_X_TEST  = os.path.join(PROCESSED_DIR, "tone_X_test.npy")

TONE_Y_TRAIN = os.path.join(PROCESSED_DIR, "tone_y_train.npy")
TONE_Y_VAL   = os.path.join(PROCESSED_DIR, "tone_y_val.npy")
TONE_Y_TEST  = os.path.join(PROCESSED_DIR, "tone_y_test.npy")

TONE_LABEL_MAPPING_PATH = os.path.join(PROCESSED_DIR, "tone_label_mapping.json")

MAX_SEQ_LEN = 40

EMBED_DIM = 128
FILTERS_1 = 64
FILTERS_2 = 128
KERNEL_SIZE = 3
POOL_SIZE = 2
DENSE_UNITS = 128
DROPOUT_RATE = 0.5

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3


def build_tone_cnn_model(vocab_size: int, max_seq_len: int, num_classes: int):
    model = Sequential([
        Input(shape=(max_seq_len,)),
        Embedding(input_dim=vocab_size, output_dim=EMBED_DIM, input_length=max_seq_len),

        Conv1D(filters=FILTERS_1, kernel_size=KERNEL_SIZE, padding="same", activation="relu"),
        MaxPooling1D(pool_size=POOL_SIZE),

        Conv1D(filters=FILTERS_2, kernel_size=KERNEL_SIZE, padding="same", activation="relu"),
        MaxPooling1D(pool_size=POOL_SIZE),

        Flatten(),
        Dense(DENSE_UNITS, activation="relu"),
        Dropout(DROPOUT_RATE),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=["accuracy"],
    )
    return model


def main():
    print(f"Loading tokenizer from: {TOKENIZER_PATH}")
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    vocab_size = tokenizer.num_words or (len(tokenizer.word_index) + 1)
    print("Vocab size:", vocab_size)

    print("Loading tone datasets...")
    X_train = np.load(TONE_X_TRAIN)
    X_val   = np.load(TONE_X_VAL)
    X_test  = np.load(TONE_X_TEST)

    y_train = np.load(TONE_Y_TRAIN)
    y_val   = np.load(TONE_Y_VAL)
    y_test  = np.load(TONE_Y_TEST)

    with open(TONE_LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
        tone_mapping = json.load(f)
    num_classes = len(tone_mapping)
    print("Tone classes:", tone_mapping)

    model = build_tone_cnn_model(vocab_size, MAX_SEQ_LEN, num_classes)
    model.summary()

    model_path = os.path.join(OUTPUT_MODEL_DIR, "tone_cnn_model.h5")
    checkpoint = ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True, mode="max")
    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    print("Training CNN model...")
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stop],
        verbose=1
    )

    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    final_model_path = os.path.join(OUTPUT_MODEL_DIR, "tone_cnn_model_final.h5")
    model.save(final_model_path)

    print("Saved best checkpoint to:", model_path)
    print("Saved final model to:", final_model_path)


if __name__ == "__main__":
    main()
