import random
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------
# LABELS
# ---------------------------------------------
tones = [
    "neutral",
    "positive_friendly",
    "negative_rude",
    "sarcastic_ironic",
    "formal",
    "informal_casual",
]

intents = [
    "request",
    "command_instruction",
    "complaint",
    "praise_appreciation",
    "clarification_question",
    "statement_information",
]


# ---------------------------------------------
# BASE PHRASES — 100% English only
# ---------------------------------------------

request_actions = [
    "send me the report",
    "review my application",
    "check the server status",
    "share the project files",
    "approve my access request",
    "reset my account password",
    "confirm my registration",
    "forward the latest update",
    "look into this issue",
    "recheck the deadline",
]

command_actions = [
    "restart the server",
    "update the documentation",
    "push your changes",
    "merge the pull request",
    "run the deployment script",
    "clean the cache",
    "delete outdated backups",
    "optimize the database",
    "apply the new configuration",
    "install the latest package",
]

complaint_phrases = [
    "the app keeps crashing",
    "no one is replying to my messages",
    "the website loads too slowly",
    "my ticket has been ignored",
    "the instructions are unclear",
    "my account randomly logs out",
    "the UI is extremely buggy",
    "the service has frequent downtime",
    "the chat support never responds",
    "my data didn't save properly",
]

praise_phrases = [
    "your explanation was very helpful",
    "the update made everything smoother",
    "your support was excellent",
    "the new feature works brilliantly",
    "you handled the issue professionally",
    "the system feels much faster now",
    "your documentation is clear and concise",
    "you resolved the bug quickly",
    "the design looks fantastic",
    "your code review was very insightful",
]

clarification_phrases = [
    "what exactly do you mean by that",
    "when is the final deadline",
    "which version should we use",
    "who is responsible for this task",
    "how should we format the report",
    "where should I upload the files",
    "why was my request rejected",
    "how do I access the logs",
    "where can I find the documentation",
    "which branch should we commit to",
]

statement_phrases = [
    "the meeting starts at ten",
    "the server will restart tonight",
    "the deadline was moved to Friday",
    "the update rolls out tomorrow",
    "the system maintenance begins soon",
    "the new policy takes effect next week",
    "the package arrived this morning",
    "the build completed successfully",
    "the logs were archived today",
    "the service will be unavailable tonight",
]

# tie intents to core phrases
intent_map = {
    "request": request_actions,
    "command_instruction": command_actions,
    "complaint": complaint_phrases,
    "praise_appreciation": praise_phrases,
    "clarification_question": clarification_phrases,
    "statement_information": statement_phrases,
}

# ---------------------------------------------
# TONE STYLE TEMPLATES
# ---------------------------------------------
def apply_tone(core: str, tone: str, intent: str) -> str:
    c = core.rstrip(".?!")
    if tone == "neutral":
        if intent in ["request", "clarification_question"]:
            return c.capitalize() + "?"
        return c.capitalize() + "."

    if tone == "positive_friendly":
        if intent in ["request", "clarification_question"]:
            return "Could you please " + c + "? That would be great!"
        return c.capitalize() + ", thank you so much!"

    if tone == "negative_rude":
        return c.capitalize() + ", this is getting ridiculous."

    if tone == "sarcastic_ironic":
        return "Wow, " + c + ", incredible."

    if tone == "formal":
        if intent in ["request", "command_instruction"]:
            return "Please " + c + " at your earliest convenience."
        if intent == "clarification_question":
            return "Could you please clarify: " + c + "?"
        return "Kindly note that " + c + "."

    if tone == "informal_casual":
        if intent in ["request", "clarification_question"]:
            return "hey, can u " + c + "?"
        return "btw, " + c + "."

    return c.capitalize() + "."

# ---------------------------------------------
# MAIN GENERATOR
# ---------------------------------------------
def generate_custom_data(n_rows=600):
    rows = []
    random.seed(42)

    while len(rows) < n_rows:
        intent = random.choice(intents)
        tone = random.choice(tones)
        core = random.choice(intent_map[intent])
        sentence = apply_tone(core, tone, intent)

        rows.append({
            "text": sentence,
            "tone_label": tone,
            "intent_label": intent,
        })

    return pd.DataFrame(rows)


def main():
    df = generate_custom_data(600)

    out_path = RAW_DIR / "custom_data.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Generated {len(df)} rows → saved to {out_path}")
    print(df.head())


if __name__ == "__main__":
    main()