import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Helper to map generic sentiment labels to your tone labels
TONE_MAP = {
    "positive": "positive_friendly",
    "negative": "negative_rude",
    "neutral": "neutral",
    # if you see other labels like "Positive", "Neg", etc,
    # just add them here:
    # "Positive": "positive_friendly",
    # "Negative": "negative_rude",
}


def build_from_sentiment_analysis():
    """
    For sentiment_analysis.csv

    After running inspect_extra_tone_datasets.py, replace TEXT_COL and LABEL_COL
    with the actual column names, for example:
        TEXT_COL = "text"
        LABEL_COL = "sentiment"
    """
    TEXT_COL = "text"       # TODO: adjust after inspection
    LABEL_COL = "sentiment" # TODO: adjust after inspection

    path = RAW_DIR / "sentiment_analysis.csv"
    df = pd.read_csv(path)

    labels_str = df[LABEL_COL].astype(str).str.lower()
    df["tone_label"] = labels_str.map(TONE_MAP)

    df["intent_label"] = None

    result = df[[TEXT_COL, "tone_label", "intent_label"]].rename(
        columns={TEXT_COL: "text"}
    )
    result = result.dropna(subset=["tone_label"])

    result.insert(0, "id", range(1, len(result) + 1))

    out = PROCESSED_DIR / "tone_from_sentiment_analysis.csv"
    result.to_csv(out, index=False, encoding="utf-8")
    print(f"Saved: {out}")


def build_from_reddit():
    """
    For Reddit_Data.csv

    Again, set TEXT_COL and LABEL_COL from inspection, e.g.:
        TEXT_COL = "clean_comment"
        LABEL_COL = "category"
    """
    TEXT_COL = "clean_comment"  # TODO: adjust
    LABEL_COL = "category"      # TODO: adjust

    path = RAW_DIR / "Reddit_Data.csv"
    df = pd.read_csv(path)

    labels_str = df[LABEL_COL].astype(str).str.lower()
    df["tone_label"] = labels_str.map(TONE_MAP)

    df["intent_label"] = None

    result = df[[TEXT_COL, "tone_label", "intent_label"]].rename(
        columns={TEXT_COL: "text"}
    )
    result = result.dropna(subset=["tone_label"])

    result.insert(0, "id", range(1, len(result) + 1))

    out = PROCESSED_DIR / "tone_from_reddit.csv"
    result.to_csv(out, index=False, encoding="utf-8")
    print(f"Saved: {out}")


def build_from_twitter():
    """
    For Twitter_Data.csv

    Set TEXT_COL and LABEL_COL from inspection, e.g.:
        TEXT_COL = "clean_text"
        LABEL_COL = "category"
    """
    TEXT_COL = "clean_text"  # TODO: adjust
    LABEL_COL = "category"   # TODO: adjust

    path = RAW_DIR / "Twitter_Data.csv"
    df = pd.read_csv(path)

    labels_str = df[LABEL_COL].astype(str).str.lower()
    df["tone_label"] = labels_str.map(TONE_MAP)

    df["intent_label"] = None

    result = df[[TEXT_COL, "tone_label", "intent_label"]].rename(
        columns={TEXT_COL: "text"}
    )
    result = result.dropna(subset=["tone_label"])

    result.insert(0, "id", range(1, len(result) + 1))

    out = PROCESSED_DIR / "tone_from_twitter.csv"
    result.to_csv(out, index=False, encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    build_from_sentiment_analysis()
    build_from_reddit()
    build_from_twitter()
