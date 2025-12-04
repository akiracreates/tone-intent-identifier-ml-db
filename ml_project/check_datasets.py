import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

def check_sentiment140():
    path = RAW_DIR / "sentiment140.csv"
    print(f"Reading: {path}")
    df = pd.read_csv(path, nrows=5, header=None, encoding="latin-1")
    print("Sentiment140 - first 5 rows:")
    print(df.head())
    print("Shape:", df.shape)
    print()

def check_chatbot_intents():
    path = RAW_DIR / "chatbot_intents.csv"
    print(f"Reading: {path}")
    df = pd.read_csv(path, nrows=5)
    print("Chatbot intents - columns:", df.columns.tolist())
    print("First 5 rows:")
    print(df.head())
    print("Shape:", df.shape)
    print()

if __name__ == "__main__":
    check_sentiment140()
    check_chatbot_intents()
