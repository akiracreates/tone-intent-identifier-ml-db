import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_processed(filename):
    """Load processed tone or intent dataset_scripts from data/processed."""
    path = PROCESSED_DIR / filename
    df = pd.read_csv(path)

    # Normalize column names
    expected = {"text", "tone_label", "intent_label"}
    available = set(df.columns)

    # Keep only expected columns
    keep_cols = [c for c in df.columns if c in expected]

    # Add any missing columns (for tone-only or intent-only datasets)
    if "tone_label" not in df.columns:
        df["tone_label"] = None
        keep_cols.append("tone_label")
    if "intent_label" not in df.columns:
        df["intent_label"] = None
        keep_cols.append("intent_label")

    df = df[["text", "tone_label", "intent_label"]]
    return df


def load_custom():
    """Load your synthetic custom data from data/raw/custom_data.csv."""
    path = RAW_DIR / "custom_data.csv"
    df = pd.read_csv(path)

    # Basic sanity check
    required = {"text", "tone_label", "intent_label"}
    if not required.issubset(df.columns):
        raise ValueError("custom_data.csv must contain text, tone_label, intent_label")

    return df[["text", "tone_label", "intent_label"]]

def load_custom_intent_only():
    """Load extra intent-only custom data from data/raw/custom_intent_only.csv."""
    path = RAW_DIR / "custom_intent_only.csv"
    df = pd.read_csv(path)

    if "text" not in df.columns or "intent_label" not in df.columns:
        raise ValueError("custom_intent_only.csv must contain text and intent_label columns")

    # ensure tone_label exists and is empty
    if "tone_label" not in df.columns:
        df["tone_label"] = None

    return df[["text", "tone_label", "intent_label"]]

def main():
    frames = []

    tone_files = [
        "tone_from_sentiment140.csv",
        "tone_from_sentiment_analysis.csv",
        "tone_from_reddit.csv",
        "tone_from_twitter.csv",
    ]
    # Load intent dataset_scripts
    try:
        df_intent = load_processed("intent_from_chatbot.csv")
        print(f"Loaded intent_from_chatbot.csv → {len(df_intent)} rows")
        frames.append(df_intent)
    except FileNotFoundError:
        print("WARNING: intent_from_chatbot.csv missing!")

    # Load tone datasets
    for name in tone_files:
        try:
            df = load_processed(name)
            print(f"Loaded {name} → {len(df)} rows")
            frames.append(df)
        except FileNotFoundError:
            print(f"WARNING: {name} not found, skipping.")

    # Load custom
    try:
        df_custom = load_custom()
        print(f"Loaded custom_data.csv → {len(df_custom)} rows")
        frames.append(df_custom)
    except FileNotFoundError:
        print("WARNING: custom_data.csv missing!")

    #extra custom intent-only data
    try:
        df_custom_intent = load_custom_intent_only()
        print(f"Loaded custom_intent_only.csv → {len(df_custom_intent)} rows")
        frames.append(df_custom_intent)
    except FileNotFoundError:
        print("WARNING: custom_intent_only.csv missing!")

    #Combine all datasets
    combined = pd.concat(frames, ignore_index=True)

    #Drop duplicates for cleanliness
    before = len(combined)
    combined = combined.drop_duplicates(subset=["text", "tone_label", "intent_label"])
    after = len(combined)
    print(f"Removed {before - after} duplicate rows")

    #Add final ID column
    combined.insert(0, "id", range(1, len(combined) + 1))

    #Save combined dataset_scripts
    out_path = PROCESSED_DIR / "combined_clean.csv"
    combined.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\nSUCCESS — Saved final dataset_scripts to:\n{out_path}")
    print("Preview:")
    print(combined.head())


if __name__ == "__main__":
    main()