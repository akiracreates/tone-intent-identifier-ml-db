# Tone & Intent Identifier: ML + DB Suite

> 🇷🇺 Educational project for the modules\
> **МДК 01.01** -- "Разработка модулей программного обеспечения для
> компьютерных систем"\
> **МДК 11.01** -- "Технология разработки и защиты баз данных"\
> Group: **ИСП9-Kh32**

A combined course project that integrates:

-   a **machine learning pipeline** for determining the **tone** and
    **communicative intent** of short text messages, and\
-   a **relational database** for storing messages, annotations, model
    predictions, and analytics.

Both course works (ML + DB) are implemented as one unified system.

------------------------------------------------------------------------

## 1. Project Overview

### 1.1. What the project does

The **Tone & Intent Identifier: ML + DB Suite**:

-   Takes short text messages from a dataset (CSV or similar).\
-   Predicts **two characteristics** per message:
    -   **Tone** --- how the message sounds\
    -   **Intent** --- what the message tries to achieve\
-   Saves all data, labels, and predictions into a **PostgreSQL**
    database.\
-   Includes **analytical SQL queries** to evaluate results and explore
    trends.

### 1.2. Who this project is for

-   Students and instructors of ML and database development.\
-   Beginners looking for an approachable example of:
    -   a classic ML text-classification workflow, and\
    -   a structured database designed to support ML experiments.

------------------------------------------------------------------------

## 2. Tone & Intent Label System

The project uses **two separate classifiers**, producing **two
independent predictions**:

### 2.1. Tone labels

1.  `neutral`\
2.  `positive_friendly`\
3.  `negative_rude`\
4.  `sarcastic_ironic`\
5.  `formal`\
6.  `informal_casual`

### 2.2. Intent labels

7.  `request`\
8.  `command_instruction`\
9.  `complaint`\
10. `praise_appreciation`\
11. `clarification_question`\
12. `statement_information`

> The label set is intentionally balanced and can be expanded later.

------------------------------------------------------------------------

## 3. Installation & Setup

> Developed and tested on **Windows 10**.\
> All steps assume a Windows environment.

### 3.1. Requirements

-   Python **3.10+**\
-   Git\
-   PostgreSQL **14 or 15**\
-   (Optional) pgAdmin UI

------------------------------------------------------------------------

### 3.2. Clone the repository

``` bash
git clone https://github.com/<username>/tone-intent-identifier-ml-db.git
cd tone-intent-identifier-ml-db
```

------------------------------------------------------------------------

### 3.3. Create a virtual environment

``` bash
python -m venv .venv
.venv\Scriptsctivate
```

------------------------------------------------------------------------

### 3.4. Install dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

### 3.5. Set up the PostgreSQL database

1.  Create the database:

    ``` sql
    CREATE DATABASE tone_intent_db;
    ```

2.  Set environment variables (PowerShell):

    ``` powershell
    setx DB_HOST "localhost"
    setx DB_PORT "5432"
    setx DB_NAME "tone_intent_db"
    setx DB_USER "<your_pg_user>"
    setx DB_PASSWORD "<your_pg_password>"
    ```

3.  Apply the database schema:

    ``` bash
    psql -U <your_pg_user> -d tone_intent_db -f db_project/schema.sql
    ```

------------------------------------------------------------------------

## 4. Usage Examples

### 4.1. Preprocess raw messages

``` bash
python -m ml_project.preprocess ^
  --input data/raw/messages.csv ^
  --output data/processed/messages_clean.csv
```

------------------------------------------------------------------------

### 4.2. Train tone and intent models

``` bash
python -m ml_project.train ^
  --config configs/tone_tfidf_logreg.yml
```

``` bash
python -m ml_project.train ^
  --config configs/intent_tfidf_logreg.yml
```

------------------------------------------------------------------------

### 4.3. Evaluate models

``` bash
python -m ml_project.evaluate ^
  --model outputs/models/tone_model.pkl ^
  --task tone ^
  --data data/processed/messages_clean.csv
```

``` bash
python -m ml_project.evaluate ^
  --model outputs/models/intent_model.pkl ^
  --task intent ^
  --data data/processed/messages_clean.csv
```

------------------------------------------------------------------------

### 4.4. Predict on new messages

``` bash
python -m ml_project.predict ^
  --tone-model outputs/models/tone_model.pkl ^
  --intent-model outputs/models/intent_model.pkl ^
  --data data/processed/messages_clean.csv ^
  --output outputs/predictions/predictions.csv
```

------------------------------------------------------------------------

### 4.5. Load predictions into the database

``` bash
python -m db_project.load_results ^
  --messages data/processed/messages_clean.csv ^
  --predictions outputs/predictions/predictions.csv
```

------------------------------------------------------------------------

### 4.6. Run analytical SQL queries

Example:

``` bash
psql -d tone_intent_db -f db_project/queries/tone_accuracy_by_label.sql
```

------------------------------------------------------------------------

## 5. Repository Structure

```text
tone-intent-identifier-ml-db/
├─ ml_project/
│  ├─ __init__.py
│  ├─ preprocess.py
│  ├─ train_tone.py
│  ├─ train_intent.py
│  ├─ evaluate.py
│  └─ predict.py
│
├─ db_project/
│  ├─ schema.sql
│  ├─ load_results.py
│  └─ queries/
│      ├─ tone_accuracy_by_label.sql
│      └─ intent_accuracy_by_label.sql
│
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ examples/
│
├─ outputs/
│  ├─ models/
│  ├─ metrics/
│  └─ predictions/
│
├─ configs/
├─ docs/
├─ tests/
├─ README_RU.md
├─ .gitignore
└─ requirements.txt


------------------------------------------------------------------------

## 6. Technical Requirements

-   **Language:** Python 3.10+\
-   **ML libraries:**
    -   pandas\
    -   numpy\
    -   scikit-learn\
    -   matplotlib / seaborn (optional)\
-   **Database:** PostgreSQL 14 or 15\
-   **OS:** Windows 10\
-   **Hardware:** Any standard laptop (no GPU required)

------------------------------------------------------------------------

## 7. Deployment & Hosting (Planned)

The system is designed to run locally for coursework.\
In the future, it can be hosted using:

-   A friend's machine with PostgreSQL\
-   A low-cost VPS (Hetzner, Contabo, DigitalOcean)\
-   Free-tier PostgreSQL hosting (Render, Railway, Supabase)

Instructions for hosting will be added later.

------------------------------------------------------------------------

## 8. Authors & Roles

**Author:**\
**Amira Haggag**\
Group **ИСП9-Kh32**

**Roles:**\
- ML module development (preprocessing, model training, evaluation)\
- Database module development (schema design, queries, integrations)\
- Documentation and technical writing (EN + RU)

------------------------------------------------------------------------

## 9. Contact

-   **Email:** akiracreates.comms@gmail.com\
-   **Telegram:** https://t.me/akiracreates\
-   **GitHub:** https://github.com/https://github.com/akiracreates\