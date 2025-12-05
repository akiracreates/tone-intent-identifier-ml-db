import os
import json
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIG ---
DATA_PATH = "data/processed/combined_clean.csv"
PROCESSED_DIR = "data/processed"
OUTPUT_MODEL_DIR = "outputs/models"
OUTPUT_METRICS_DIR = "outputs/metrics"

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_METRICS_DIR, exist_ok=True)

INTENT_LABEL_MAPPING_PATH = os.path.join(PROCESSED_DIR, "intent_label_mapping.json")

TEST_SIZE = 0.2
RANDOM_STATE = 42

MAX_FEATURES = 10000
NGRAM_RANGE = (1, 2)


def main():
    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    df_intent = df[df["intent_label"].notna()].copy()
    df_intent["text"] = df_intent["text"].astype(str).fillna("")
    print("Total intent samples:", len(df_intent))

    with open(INTENT_LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
        intent_mapping = json.load(f)
    print("Mapping:", intent_mapping)

    df_intent["intent_id"] = df_intent["intent_label"].map(intent_mapping)

    X = df_intent["text"].values
    y = df_intent["intent_id"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Train size:", len(X_train))
    print("Test size:", len(X_test))

    # TF-IDF
    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
    )
    X_train_tf = vectorizer.fit_transform(X_train)
    X_test_tf = vectorizer.transform(X_test)

    # Logistic Regression
    print("Training Logistic Regression...")
    clf = LogisticRegression(max_iter=1000, n_jobs=-1, multi_class="multinomial")
    clf.fit(X_train_tf, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_tf)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")

    target_names = [label for label, _ in sorted(intent_mapping.items(), key=lambda x: x[1])]
    report = classification_report(y_test, y_pred, target_names=target_names)
    print("\nClassification report:\n", report)

    # Save model + vectorizer
    with open(os.path.join(OUTPUT_MODEL_DIR, "intent_tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(OUTPUT_MODEL_DIR, "intent_logreg_model.pkl"), "wb") as f:
        pickle.dump(clf, f)

    # Save metrics
    with open(os.path.join(OUTPUT_METRICS_DIR, "intent_logreg_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Test accuracy: {acc:.4f}\n\n")
        f.write(report)

    print("Done. Model and metrics saved.")


if __name__ == "__main__":
    main()
