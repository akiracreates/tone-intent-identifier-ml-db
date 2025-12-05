import csv
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "examples"
OUTPUT_PATH = OUTPUT_DIR / "eval_messages_1000.csv"

random.seed(42)


FORMAL_REQUESTS = [
    "Could you please provide the updated report?",
    "Could you send me the latest version of the document?",
    "Please review the attached file and share your feedback.",
    "Would you mind checking the figures in section three?",
    "Could you clarify which template we should use for this task?",
    "Please upload the final draft to the shared folder.",
    "Could you confirm the deadline for this assignment?",
    "Please let me know if any changes are required.",
    "Could you verify that the data is up to date?",
    "Please resend the presentation when you have a moment.",
]

FORMAL_INFO = [
    "The updated policy will take effect next Monday.",
    "The meeting is scheduled for 10 AM tomorrow.",
    "The report was submitted to the supervisor yesterday.",
    "The new template is now available in the shared folder.",
    "The current version of the application is stable.",
    "The data set has been successfully uploaded to the server.",
    "The training session will start at 3 PM.",
    "The deadline for submissions has been extended by two days.",
    "The new guidelines apply to all future projects.",
    "The documentation has been updated according to the latest changes.",
]

PRAISE = [
    "You did an excellent job on this project.",
    "Your attention to detail is highly appreciated.",
    "The quality of your work is impressive.",
    "Thank you for the quick and professional response.",
    "This solution is very well designed and implemented.",
    "Your presentation was clear and informative.",
    "I really appreciate the effort you put into this.",
    "The refactoring you did made the code much cleaner.",
    "You handled this task in a very efficient way.",
    "The final result looks great, thank you.",
]

COMPLAINTS = [
    "The current solution does not meet the stated requirements.",
    "This feature still does not work as expected.",
    "The application is significantly slower after the last update.",
    "The report contains several inconsistencies and errors.",
    "The instructions were not followed correctly.",
    "The layout is confusing and hard to read.",
    "The changes introduced new bugs into the system.",
    "The test results are incomplete and unclear.",
    "The documentation is missing important details.",
    "The response time to this request has been unreasonably long.",
]

INFORMAL_REQUESTS = [
    "Hey, can you send me that file when you get a chance?",
    "Can you drop the link in the chat?",
    "Mind sharing the latest version with me?",
    "Can you double-check the numbers real quick?",
    "Could you help me fix this bug?",
    "Can you upload the slides to the drive?",
    "Can you ping me when you're done with the draft?",
    "Can you check if this looks okay?",
    "Can you send me the screenshots you took?",
    "Can you remind me what the deadline is?",
]

INFORMAL_COMPLAINTS = [
    "This app is super buggy again lol.",
    "Nothing works the way it should right now.",
    "This update broke everything, seriously.",
    "The layout looks weird on my screen.",
    "The page keeps freezing whenever I try to save.",
    "The results look totally off again.",
    "This thing is painfully slow today.",
    "The UI is really confusing to use.",
    "The form keeps throwing random errors.",
    "The export function is acting up again.",
]

SARCASM = [
    "Amazing, another update that breaks everything, great job.",
    "Wow, it actually worked on the first try, shocking.",
    "Perfect, exactly what we needed, another mysterious error.",
    "Fantastic, now it crashes even faster than before.",
    "Great, now nothing loads but at least the button is blue.",
    "Wonderful, the bug is gone but so is the data.",
    "Nice, the feature works as long as you never touch it.",
    "Incredible, the fix fixed everything except the actual problem.",
    "Brilliant, we solved one issue by creating three new ones.",
    "Awesome, the application finally runs, sort of.",
]

NEUTRAL_STATEMENTS = [
    "The file has been saved in the shared folder.",
    "The form has been submitted successfully.",
    "The results are ready for review.",
    "The system is running as expected.",
    "The changes have been deployed to the test environment.",
    "The logs have been archived for future reference.",
    "The backup completed without errors.",
    "The configuration has been updated.",
    "The dataset has been cleaned and validated.",
    "The script finished executing without issues.",
]

CLARIFICATION_QUESTIONS = [
    "Which version of the document should we use?",
    "Are we following the old guideline or the new one?",
    "Do we need to include these metrics in the report?",
    "Should this task be finished by the end of the day?",
    "Are we supposed to use the test server or the production server?",
    "Do we need approval before merging these changes?",
    "Should this be done in Python or another language?",
    "Is this feature required for the first release?",
    "Do we have any examples we can follow?",
    "Are there any constraints we should know about?",
]

INTENT_DECORATORS = [
    "",
    " Thanks in advance.",
    " Let me know if anything changes.",
    " If anything is unclear, please tell me.",
    " I would really appreciate your help with this.",
    " When you have a moment, please take a look.",
]

INFORMAL_DECORATORS = [
    "",
    " Thanks!",
    " Let me know what you think.",
    " Just wanted to check in about this.",
    " If it's not too much trouble.",
    " Feel free to poke me if needed.",
]


def build_base_messages():
    """Create a pool of base messages with different tones and intents."""
    messages = []

    # Formal requests
    for base in FORMAL_REQUESTS:
        for dec in INTENT_DECORATORS:
            messages.append(base + dec)

    # Formal information
    for base in FORMAL_INFO:
        for dec in INTENT_DECORATORS:
            messages.append(base + dec)

    # Praise / appreciation
    for base in PRAISE:
        for dec in INTENT_DECORATORS:
            messages.append(base + " " + dec if dec else base)

    # Complaints (more formal)
    for base in COMPLAINTS:
        for dec in INTENT_DECORATORS:
            messages.append(base + " " + dec if dec else base)

    # Informal requests
    for base in INFORMAL_REQUESTS:
        for dec in INFORMAL_DECORATORS:
            messages.append(base + dec)

    # Informal complaints
    for base in INFORMAL_COMPLAINTS:
        for dec in INFORMAL_DECORATORS:
            messages.append(base + dec)

    # Sarcastic lines
    for base in SARCASM:
        for dec in INFORMAL_DECORATORS:
            messages.append(base + " " + dec if dec else base)

    # Neutral statements
    for base in NEUTRAL_STATEMENTS:
        for dec in INTENT_DECORATORS:
            messages.append(base + " " + dec if dec else base)

    # Clarification questions
    for base in CLARIFICATION_QUESTIONS:
        for dec in INTENT_DECORATORS:
            messages.append(base + " " + dec if dec else base)

    # Small clean-up: strip double spaces
    messages = [m.replace("  ", " ").strip() for m in messages]

    # Remove duplicates while preserving order
    seen = set()
    unique_messages = []
    for m in messages:
        if m not in seen:
            seen.add(m)
            unique_messages.append(m)

    return unique_messages


def generate_eval_messages(n_samples: int = 1000):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_messages = build_base_messages()
    print(f"Base pool size: {len(base_messages)}")

    # If we have fewer base messages than needed, sample with replacement
    if len(base_messages) >= n_samples:
        chosen = random.sample(base_messages, n_samples)
    else:
        chosen = []
        # use all base messages at least once
        full_cycles = n_samples // len(base_messages)
        remainder = n_samples % len(base_messages)

        for _ in range(full_cycles):
            shuffled = base_messages[:]
            random.shuffle(shuffled)
            chosen.extend(shuffled)
        if remainder > 0:
            chosen.extend(random.sample(base_messages, remainder))

    # Write to CSV
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text"])
        for msg in chosen:
            writer.writerow([msg])

    print(f"Saved {len(chosen)} synthetic evaluation messages to {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_eval_messages(n_samples=1000)
