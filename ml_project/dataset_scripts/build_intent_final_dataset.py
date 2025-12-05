from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

COMBINED_PATH = DATA_PROCESSED / "combined_clean.csv"
INTENT_FINAL_PATH = DATA_PROCESSED / "intent_final_dataset.csv"


def main():
    print(f"[1] Loading {COMBINED_PATH} ...")
    df = pd.read_csv(COMBINED_PATH)

    print("[2] Filtering rows with non-null intent_label ...")
    intent_df = df.dropna(subset=["intent_label"]).copy()

    print("[3] Intent label distribution:")
    print(intent_df["intent_label"].value_counts())

    print(f"[4] Saving to {INTENT_FINAL_PATH} ...")
    intent_df.to_csv(INTENT_FINAL_PATH, index=False)
    print("[OK] Saved intent_final_dataset.csv")


if __name__ == "__main__":
    main()
