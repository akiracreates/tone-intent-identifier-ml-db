import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def build_intent_from_chatbot():
    path = RAW_DIR / "chatbot_intents.csv"
    df = pd.read_csv(path)

    df = df.rename(columns={"user_input": "text"})

    # Final mapping:
    intent_map = {
        "account_help": "request",
        "business_hours": "clarification_question",
        "cancellation": "request",
        "order_status": "clarification_question",
        "password_reset": "request",
        "payment_update": "request",
        "return_request": "request",
        "service_info": "clarification_question",
        "technical_support": "request",
    }

    df["intent_label"] = df["intent"].map(intent_map)
    df["tone_label"] = None

    # Drop unknowns (should be zero)
    df = df.dropna(subset=["intent_label"])

    # Keep only the needed columns
    result = df[["text", "tone_label", "intent_label"]].copy()

    # Add numeric ID
    result.insert(0, "id", range(1, len(result) + 1))

    out_path = PROCESSED_DIR / "intent_from_chatbot.csv"
    result.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Saved intent dataset to: {out_path}")
    print(result.head())


if __name__ == "__main__":
    build_intent_from_chatbot()