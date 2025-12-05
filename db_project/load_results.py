import os
import argparse
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "predictions" / "predictions.csv"


def get_db_connection():
    """Create a PostgreSQL connection using environment variables."""
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "tone_intent_db")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "888")

    if not password:
        print("WARNING: DB_PASSWORD is empty. "
              "Set it via environment variable if your DB requires a password.")

    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=name,
        user=user,
        password=password,
    )
    return conn


def load_predictions(csv_path: Path):
    """Load predictions CSV and insert into messages + predictions tables."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"Reading predictions from: {csv_path}")
    df = pd.read_csv(csv_path)

    required_cols = [
        "text",
        "tone_bilstm_pred",
        "tone_cnn_pred",
        "intent_pred",
        "predicted_at",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                # 1) Insert messages, collect (message_id, row index)
                message_rows = [(str(text),) for text in df["text"].astype(str)]
                insert_messages_sql = """
                    INSERT INTO messages (text)
                    VALUES (%s)
                    RETURNING id;
                """

                print(f"Inserting {len(message_rows)} messages...")
                message_ids = []
                for row in message_rows:
                    cur.execute(insert_messages_sql, row)
                    new_id = cur.fetchone()[0]
                    message_ids.append(new_id)

                # 2) Insert predictions referencing messages
                prediction_rows = []
                for idx, msg_id in enumerate(message_ids):
                    row = df.iloc[idx]
                    prediction_rows.append(
                        (
                            msg_id,
                            row["tone_bilstm_pred"],
                            row["tone_cnn_pred"],
                            row["intent_pred"],
                            row["predicted_at"],
                        )
                    )

                insert_predictions_sql = """
                    INSERT INTO predictions (
                        message_id,
                        tone_bilstm_pred,
                        tone_cnn_pred,
                        intent_pred,
                        predicted_at
                    )
                    VALUES (%s, %s, %s, %s, %s);
                """

                print(f"Inserting {len(prediction_rows)} predictions...")
                execute_batch(cur, insert_predictions_sql, prediction_rows, page_size=100)

        print("Done. Data committed to the database.")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Load ML predictions CSV into PostgreSQL tables (messages, predictions)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_PREDICTIONS_PATH),
        help="Path to predictions CSV. "
             "Defaults to outputs/predictions/predictions.csv",
    )
    args = parser.parse_args()

    csv_path = Path(args.input)
    load_predictions(csv_path)


if __name__ == "__main__":
    main()
