import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def build_tone_from_sentiment140():
    path = RAW_DIR / "sentiment140.csv"
    # no header, 6 columns
    df = pd.read_csv(path, header=None, encoding="latin-1")

    df.columns = ["target", "id", "date", "flag", "user", "text"]

    # Map target 0/4 to tone labels
    def map_target(t):
        if t == 4:
            return "positive_friendly"
        elif t == 0:
            return "negative_rude"
        else:
            return None

    df["tone_label"] = df["target"].apply(map_target)

    # we don't have intent labels here
    df["intent_label"] = None

    # keep only what we need
    result = df[["text", "tone_label", "intent_label"]].copy()

    #sample down so it's not huge (e.g. 20k rows)
    result = result.sample(n=20000, random_state=42)

    # add id
    result.insert(0, "id", range(1, len(result) + 1))

    out_path = PROCESSED_DIR / "tone_from_sentiment140.csv"
    result.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved tone dataset to: {out_path}")
    print(result.head())

if __name__ == "__main__":
    build_tone_from_sentiment140()
