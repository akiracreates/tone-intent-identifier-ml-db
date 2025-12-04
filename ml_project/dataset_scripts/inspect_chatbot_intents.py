import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

def inspect():
    path = RAW_DIR / "chatbot_intents.csv"
    df = pd.read_csv(path)

    print("Columns:", df.columns.tolist())
    print("\nUnique intents and counts:\n")
    print(df["intent"].value_counts().sort_index())

if __name__ == "__main__":
    inspect()