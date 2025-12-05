# 📘 Tone & Intent Identifier: ML + DB Suite  
**Course Project – Amira Haggag (ИСП9-Kh32)**

> Course project for the disciplines:  
> **МДК 01.01 «Разработка модулей программного обеспечения для компьютерных систем»**  
> **МДК 11.01 «Технология разработки и защиты баз данных»**  
> Group: **ИСП9-Kh32**

This repository contains:

- a complete **machine learning subsystem** for classifying the **tone** and **intent** of short text messages;  
- a **PostgreSQL database subsystem** for storing messages, predictions, and running analytical SQL queries.

Both course projects (ML + DB) are implemented as a unified integrated system.

---

## 1. Project Description

### 1.1 System Functionality

The **Tone & Intent Identifier: ML + DB Suite** provides:

- Input of arbitrary short text messages  
- Independent classification into:
  - **Tone** (BiLSTM + CNN neural models)
  - **Intent** (TF-IDF + Logistic Regression)
- Storage of predictions in a **PostgreSQL** database  
- Execution of SQL analytics for evaluation and exploration  

---

### 1.2 Target Audience

- Students studying ML or databases  
- Beginners who want to see:
  - A clean ML pipeline (dataset → training → evaluation → inference)  
  - A properly structured relational DB for storing ML results  
  - A full integrated system (ML + DB)

---

## 2. Label System

### 2.1 Tone (6 classes)

1. neutral  
2. positive_friendly  
3. negative_rude  
4. sarcastic_ironic  
5. formal  
6. informal_casual  

### 2.2 Intent (6 classes)

1. request  
2. command_instruction  
3. complaint  
4. praise_appreciation  
5. clarification_question  
6. statement_information  

---

## 3. Installation & Setup

> Developed and tested on **Windows 10**.

### 3.1 Requirements

- Python 3.10+  
- Git  
- PostgreSQL 14/15  
- Optional: pgAdmin  

---

### 3.2 Clone the Repository

```bash
git clone https://github.com/<username>/tone-intent-identifier-ml-db.git
cd tone-intent-identifier-ml-db
```

---

### 3.3 Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

### 3.4 Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3.5 PostgreSQL Setup

1. Create database:

```sql
CREATE DATABASE tone_intent_db;
```

2. Set environment variables:

```powershell
setx DB_HOST "localhost"
setx DB_PORT "5432"
setx DB_NAME "tone_intent_db"
setx DB_USER "<your_user>"
setx DB_PASSWORD "<your_password>"
```

3. Apply schema:

```bash
psql -U <your_user> -d tone_intent_db -f db_project/schema.sql
```

---

## 4. Machine Learning Pipeline

### 4.1 Training Tone Models (BiLSTM + CNN)

```bash
python -m ml_project.tokenize_and_train.build_tokenizer_and_sequences
python -m ml_project.tokenize_and_train.prepare_balanced_tone_datasets
python -m ml_project.tokenize_and_train.train_tone_lstm
python -m ml_project.tokenize_and_train.train_tone_cnn
```

---

### 4.2 Training Intent Classifier (TF-IDF + Logistic Regression)

```bash
python -m ml_project.tokenize_and_train.train_intent_tfidf_logreg_balanced
```

---

### 4.3 Model Evaluation

```bash
python -m ml_project.evaluate --task all
```

Outputs stored in:

```
outputs/metrics/
```

---

## 5. Prediction Pipeline

Single text:

```bash
python -m ml_project.predict --text "Hello, I need help"
```

Batch mode:

```bash
python -m ml_project.predict ^
  --input data/examples/eval_messages_1000.csv ^
  --output outputs/predictions/eval_predictions_1000.csv
```

Each prediction includes tone, intent, probabilities, and timestamp.

---

## 6. Synthetic Evaluation Dataset

Generate 1000 evaluation messages:

```bash
python -m ml_project.generate_eval_messages_1000
```

Output:

```
data/examples/eval_messages_1000.csv
```

---

## 7. Database Integration

### 7.1 Schema

```
db_project/schema.sql
```

Creates:

- messages  
- predictions  

---

### 7.2 Load Predictions into PostgreSQL

```bash
python -m db_project.load_results ^
  --input outputs/predictions/eval_predictions_1000.csv
```

---

## 8. SQL Analytics

Queries located in:

```
db_project/queries/
```

Run via pgAdmin or:

```bash
psql -d tone_intent_db -f db_project/queries/<query>.sql
```

---

## 9. Repository Structure

```
tone-intent-identifier-ml-db/
├─ ml_project/
│  ├─ tokenize_and_train/
│  ├─ dataset_scripts/
│  ├─ generate_eval_messages_1000.py
│  ├─ evaluate.py
│  └─ predict.py
│
├─ db_project/
│  ├─ schema.sql
│  ├─ load_results.py
│  └─ queries/
│
├─ data/
│  ├─ raw/ (ignored)
│  ├─ processed/ (ignored)
│  └─ examples/ (tracked)
│
├─ outputs/ (ignored except .gitkeep)
├─ README.md
├─ README_RU.md
└─ requirements.txt
```

---

## 10. Author

**Amira Haggag / Хаггаг Амира**  
Group: **ИСП9-Kh32**

Roles:

- ML module development  
- DB schema & integration  
- System architecture  
- Technical documentation (EN + RU)
