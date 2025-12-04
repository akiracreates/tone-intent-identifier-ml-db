import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

def inspect_file(filename):
    path = RAW_DIR / filename
    print(f"\n=== {filename} ===")
    df = pd.read_csv(path, nrows=5)
    print("Columns:", df.columns.tolist())
    print(df.head())
    print("Shape (first 5 shown):", df.shape)

if __name__ == "__main__":
    inspect_file("sentiment_analysis.csv")
    inspect_file("Reddit_Data.csv")
    inspect_file("Twitter_Data.csv")