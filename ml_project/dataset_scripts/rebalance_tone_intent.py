from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

COMBINED_PATH = DATA_PROCESSED / "combined_clean.csv"

TONE_OUT_PATH = DATA_PROCESSED / "tone_balanced_for_training.csv"
INTENT_OUT_PATH = DATA_PROCESSED / "intent_balanced_for_training.csv"


def make_balanced(df: pd.DataFrame, label_col: str, target_per_class: int) -> pd.DataFrame:
    """
    Build a class-balanced dataset by downsampling large classes
    and oversampling (with replacement) smaller ones up to target_per_class.
    """
    groups = df.groupby(label_col)
    parts = []

    print(f"\n[INFO] Original distribution for {label_col}:")
    print(df[label_col].value_counts())

    for label, group in groups:
        n = len(group)
        if n == 0:
            continue

        if n > target_per_class:
            # downsample
            sampled = group.sample(target_per_class, random_state=42)
            print(f"  {label}: {n} → {target_per_class} (downsample)")
        elif n < target_per_class:
            # oversample with replacement
            sampled = group.sample(target_per_class, replace=True, random_state=42)
            print(f"  {label}: {n} → {target_per_class} (oversample)")
        else:
            sampled = group
            print(f"  {label}: {n} (no change)")

        parts.append(sampled)

    balanced = pd.concat(parts, axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"\n[INFO] Balanced distribution for {label_col}:")
    print(balanced[label_col].value_counts())

    return balanced


def main():
    df = pd.read_csv(COMBINED_PATH)

    # --- Tone ---
    tone_df = df.dropna(subset=["tone_label"]).copy()
    # pick a reasonable target per class; you can tweak this
    TONE_TARGET = 1500
    tone_balanced = make_balanced(tone_df, "tone_label", TONE_TARGET)
    tone_balanced.to_csv(TONE_OUT_PATH, index=False)
    print(f"\n[OK] Saved balanced tone dataset to {TONE_OUT_PATH}")

    # --- Intent ---
    intent_df = df.dropna(subset=["intent_label"]).copy()
    INTENT_TARGET = 400  # oversample each intent class up to 400 samples
    intent_balanced = make_balanced(intent_df, "intent_label", INTENT_TARGET)
    intent_balanced.to_csv(INTENT_OUT_PATH, index=False)
    print(f"[OK] Saved balanced intent dataset to {INTENT_OUT_PATH}")


if __name__ == "__main__":
    main()
