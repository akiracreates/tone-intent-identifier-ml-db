# Tone & Intent Identifier: ML + DB Suite

# System Description (English Version)

> Course project for the disciplines:  
> **MDK 01.01** — Development of Software Modules for Computer Systems  
> **MDK 11.01** — Database Development and Security Technologies  
> Group: **ISP9-Kh32**

This repository combines:

- a **machine learning subsystem** for classifying the **tone** and **intent** of short text messages;  
- a **PostgreSQL relational database** for storing messages, labels, model predictions, and analytical SQL queries.

Both course projects (ML + DB) are implemented as a single integrated system.

---

## 1. Project Description

### 1.1. Functionality

**Tone & Intent Identifier: ML + DB Suite** allows you to:

- accept text messages (short messages, comments, emails);  
- determine two properties:  
  - **Tone** of the message  
  - **Intent** of the message  
- store messages, true labels, and predictions inside **PostgreSQL**;  
- run analytical SQL queries to evaluate model performance.

---

### 1.2. Target Audience

- Students learning ML or databases  
- Beginners who want to see:  
  - a clean ML pipeline using classical and neural methods  
  - a properly designed database supporting ML experiments  

---

## 2. Label System (Tone & Intent)

The project uses **two separate classifiers**, producing **two independent predictions**.

### 2.1. Tone Labels

1. `neutral`  
2. `positive_friendly`  
3. `negative_rude`  
4. `sarcastic_ironic`  
5. `formal`  
6. `informal_casual`  

### 2.2. Intent Labels

7. `request`  
8. `command_instruction`  
9. `complaint`  
10. `praise_appreciation`  
11. `clarification_question`  
12. `statement_information`  

---

## 3. Installation & Setup

> Development and testing were performed on **Windows 10**.

### 3.1. Requirements

- Python **3.10+**  
- Git  
- PostgreSQL **14 or 15**  
- (Optional) pgAdmin  

---

### 3.2. Clone the Repository

```bash
git clone https://github.com/<username>/tone-intent-identifier-ml-db.git
cd tone-intent-identifier-ml-db
```

---

### 3.3. Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

### 3.4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3.5. PostgreSQL Configuration

1. **Create the database:**

```sql
CREATE DATABASE tone_intent_db;
```

2. **Set environment variables:**

```powershell
setx DB_HOST "localhost"
setx DB_PORT "5432"
setx DB_NAME "tone_intent_db"
setx DB_USER "<your_user>"
setx DB_PASSWORD "<your_password>"
```

3. **Apply database schema:**

```bash
psql -U <your_user> -d tone_intent_db -f db_project/schema.sql
```

---

## 4. Usage Examples

### 4.1. Data Preprocessing

```bash
python -m ml_project.preprocess ^
  --input data/raw/messages.csv ^
  --output data/processed/messages_clean.csv
```

---

### 4.2. Training Tone & Intent Models

```bash
python -m ml_project.train ^
  --config configs/tone_tfidf_logreg.yml
```

```bash
python -m ml_project.train ^
  --config configs/intent_tfidf_logreg.yml
```

---

### 4.3. Model Evaluation

```bash
python -m ml_project.evaluate ^
  --model outputs/models/tone_model.pkl ^
  --task tone ^
  --data data/processed/messages_clean.csv
```

```bash
python -m ml_project.evaluate ^
  --model outputs/models/intent_model.pkl ^
  --task intent ^
  --data data/processed/messages_clean.csv
```

---

### 4.4. Predicting on New Data

```bash
python -m ml_project.predict ^
  --tone-model outputs/models/tone_model.pkl ^
  --intent-model outputs/models/intent_model.pkl ^
  --data data/processed/messages_clean.csv ^
  --output outputs/predictions/predictions.csv
```

---

### 4.5. Uploading Results to the Database

```bash
python -m db_project.load_results ^
  --messages data/processed/messages_clean.csv ^
  --predictions outputs/predictions/predictions.csv
```

---

### 4.6. Running SQL Analytics

```bash
psql -d tone_intent_db -f db_project/queries/tone_accuracy_by_label.sql
```

---

## 5. Repository Structure

```text
tone-intent-identifier-ml-db/
├─ ml_project/
│  ├─ dataset_scripts/
│  ├─ build_tokenizer_and_sequences.py
│  ├─ prepare_task_datasets.py
│  ├─ train_tone_cnn.py
│  ├─ train_tone_lstm.py
│  ├─ train_intent_tfidf_logreg.py
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
```

---

## 6. Technical Requirements

- **Programming language:** Python 3.10+  
- **ML libraries:** pandas, numpy, scikit-learn, tensorflow/keras  
- **Database:** PostgreSQL 14/15  
- **OS:** Windows 10  
- **Hardware:** Any standard laptop  

---

## 7. Deployment & Hosting (Planned)

The educational version works locally.  
For hosting, you may use:

- Personal server  
- Budget VPS (Hetzner, Contabo, DigitalOcean)  
- Free PostgreSQL hosting services (Render, Railway, Supabase)

---

## 8. Author

**Amira Haggag / Хаггаг Амира**  
Group: **ИСП9-Kh32**

**Roles:**
- ML module development  
- Database design and development  
- Technical documentation (EN + RU)

---

## 9. Contacts

- Email: akiracreates.comms@gmail.com  
- Telegram: https://t.me/akiracreates  
- GitHub: https://github.com/akiracreates  
