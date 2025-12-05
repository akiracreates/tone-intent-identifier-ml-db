# Tone & Intent Identifier: ML + DB Suite

# System Description (English Version)

> Course project for the disciplines:\
> **MDK 01.01** --- Development of Software Modules for Computer
> Systems\
> **MDK 11.01** --- Database Development and Security Technologies\
> Group: **ИСП9-Kh32**

This repository contains:

-   a complete **machine learning subsystem** for classifying the
    **tone** and **intent** of short text messages,\
-   a **PostgreSQL database subsystem** for storing messages, labels,
    predictions, and running analytical SQL queries.

Both projects (ML + DB) are implemented as a unified system.

------------------------------------------------------------------------

## 1. Project Description

### 1.1. Functionality

The **Tone & Intent Identifier: ML + DB Suite** allows you to:

-   Input short text messages\
-   Identify **two independent properties**:
    -   **Tone** of the message (neural models)\
    -   **Intent** of the message (classical ML model)\
-   Store results inside a **PostgreSQL database**\
-   Perform analytical SQL queries to understand model performance

------------------------------------------------------------------------

### 1.2. Target Audience

-   Students working with ML or databases\
-   Beginners wanting to see:
    -   A clean, correct ML pipeline (data → processing → training →
        evaluation → inference)\
    -   A properly structured relational database for ML experiment
        tracking

------------------------------------------------------------------------

## 2. Label System

The system uses two independent classifiers, each producing one label.

### 2.1 Tone Classes (6)

1.  `neutral`\
2.  `positive_friendly`\
3.  `negative_rude`\
4.  `sarcastic_ironic`\
5.  `formal`\
6.  `informal_casual`

### 2.2 Intent Classes (6)

7.  `request`\
8.  `command_instruction`\
9.  `complaint`\
10. `praise_appreciation`\
11. `clarification_question`\
12. `statement_information`

------------------------------------------------------------------------

## 3. Installation & Setup

> Developed and tested on **Windows 10**.

### 3.1 Requirements

-   Python **3.10+**\
-   Git\
-   PostgreSQL **14/15**\
-   Optional: pgAdmin

------------------------------------------------------------------------

### 3.2 Clone the Repository

``` bash
git clone https://github.com/<username>/tone-intent-identifier-ml-db.git
cd tone-intent-identifier-ml-db
```

------------------------------------------------------------------------

### 3.3 Create Virtual Environment

``` bash
python -m venv .venv
.venv\Scripts\activate
```

------------------------------------------------------------------------

### 3.4 Install Dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

### 3.5 PostgreSQL Setup

1.  Create database:

``` sql
CREATE DATABASE tone_intent_db;
```

2.  Set environment variables:

``` powershell
setx DB_HOST "localhost"
setx DB_PORT "5432"
setx DB_NAME "tone_intent_db"
setx DB_USER "<your_user>"
setx DB_PASSWORD "<your_password>"
```

3.  Apply schema:

``` bash
psql -U <your_user> -d tone_intent_db -f db_project/schema.sql
```

------------------------------------------------------------------------

## 4. Usage Examples

### 4.1 Data Preparation Pipeline

#### (A) Rebalance tone & intent datasets

``` bash
python -m ml_project.dataset_scripts.rebalance_tone_intent
```

Creates:

-   `tone_balanced_for_training.csv` (final tone training dataset)
-   Legacy `intent_balanced_for_training.csv` (not used further)

#### (B) Build final intent dataset

``` bash
python -m ml_project.dataset_scripts.build_intent_final_dataset
```

Creates:

-   `intent_final_dataset.csv` --- used for intent model training

------------------------------------------------------------------------

## 4.2 Training Tone Models

#### 1. Build tokenizer & sequences (first time only)

``` bash
python -m ml_project.tokenize_and_train.build_tokenizer_and_sequences
```

#### 2. Create balanced tone train/val/test splits

``` bash
python -m ml_project.tokenize_and_train.prepare_balanced_tone_datasets
```

#### 3. Train neural models

BiLSTM:

``` bash
python -m ml_project.tokenize_and_train.train_tone_lstm
```

CNN:

``` bash
python -m ml_project.tokenize_and_train.train_tone_cnn
```

------------------------------------------------------------------------

## 4.3 Training Intent Model (Balanced)

``` bash
python -m ml_project.tokenize_and_train.train_intent_tfidf_logreg_balanced
```

This script:

-   loads `intent_final_dataset.csv`\
-   performs **70/30 stratified split**\
-   **saves the test split** → `intent_X_test.npy`, `intent_y_test.npy`\
-   oversamples **training set only**\
-   trains **TF-IDF + Logistic Regression**\
-   writes evaluation report and saves the model + vectorizer

------------------------------------------------------------------------

## 4.4 Model Evaluation

Evaluate all models:

``` bash
python -m ml_project.evaluate --task all
```

Or individually:

``` bash
python -m ml_project.evaluate --task tone_bilstm
python -m ml_project.evaluate --task tone_cnn
python -m ml_project.evaluate --task intent
```

Outputs:

-   classification reports (`.txt`)
-   confusion matrices (`.png`)

Stored in:

    outputs/metrics/

------------------------------------------------------------------------

## 4.5 Predicting on New Data

Single message:

``` bash
python -m ml_project.predict --text "I need help resetting my password"
```

Batch mode:

``` bash
python -m ml_project.predict ^
  --input data/processed/messages_clean.csv ^
  --output outputs/predictions/predictions.csv
```

------------------------------------------------------------------------

## 4.6 Upload Results to PostgreSQL

``` bash
python -m db_project.load_results ^
  --messages data/processed/messages_clean.csv ^
  --predictions outputs/predictions/predictions.csv
```

------------------------------------------------------------------------

## 4.7 Run SQL Analytics

``` bash
psql -d tone_intent_db -f db_project/queries/tone_accuracy_by_label.sql
```

------------------------------------------------------------------------

## 5. Repository Structure

    tone-intent-identifier-ml-db/
    ├─ ml_project/
    │  ├─ dataset_scripts/
    │  │   ├─ build_combined_dataset.py
    │  │   ├─ rebalance_tone_intent.py
    │  │   ├─ build_intent_final_dataset.py
    │  │   └─ __init__.py
    │  │
    │  ├─ tokenize_and_train/
    │  │   ├─ build_tokenizer_and_sequences.py
    │  │   ├─ prepare_task_datasets.py
    │  │   ├─ prepare_balanced_tone_datasets.py
    │  │   ├─ train_tone_lstm.py
    │  │   ├─ train_tone_cnn.py
    │  │   └─ train_intent_tfidf_logreg_balanced.py
    │  │
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
    │  ├─ raw/                    (ignored)
    │  ├─ processed/
    │  │   ├─ combined_clean.csv
    │  │   ├─ intent_final_dataset.csv
    │  │   ├─ tokenizer.pkl
    │  │   ├─ intent_label_mapping.json
    │  │   ├─ tone_label_mapping.json
    │  │   ├─ tone_balanced_for_training.csv
    │  │   ├─ tone_X_*.npy
    │  │   └─ intent_X_test.npy, intent_y_test.npy
    │
    ├─ outputs/                   (ignored)
    │  ├─ models/
    │  ├─ metrics/
    │  └─ predictions/
    │
    ├─ README.md
    ├─ README_RU.md
    ├─ .gitignore
    └─ requirements.txt

------------------------------------------------------------------------

## 6. Technical Requirements

-   Python 3.10+\
-   pandas, numpy, scikit-learn, tensorflow, matplotlib, joblib\
-   PostgreSQL 14/15\
-   Windows 10 or similar environment

------------------------------------------------------------------------

## 7. Deployment & Hosting (Planned)

Future hosting options include:

-   Local server / home PC\
-   VPS (Hetzner, Contabo, DigitalOcean)\
-   Free PostgreSQL hosting (Railway, Render, Supabase)

------------------------------------------------------------------------

## 8. Author

**Amira Haggag / Хаггаг Амира**\
Group: **ИСП9-Kh32**

Roles:

-   Machine learning module development\
-   Database schema development\
-   Full system integration\
-   Technical documentation (EN + RU)

------------------------------------------------------------------------

## 9. Contacts

-   Email: **akiracreates.comms@gmail.com**\
-   Telegram: **https://t.me/akiracreates**\
-   GitHub: **https://github.com/akiracreates**
