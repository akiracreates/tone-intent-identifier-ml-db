# Идентификатор тона и намерения сообщения (ML + БД)

> Учебный проект по дисциплинам  
> **МДК 01.01** — «Разработка модулей программного обеспечения для компьютерных систем»  
> **МДК 11.01** — «Технология разработки и защиты баз данных»  
> Группа: **ИСП9-Kh32**

Данный репозиторий объединяет:

- **машинное обучение** для классификации **тона** и **коммуникативного намерения** коротких текстовых сообщений;  
- **реляционную базу данных PostgreSQL** для хранения сообщений, разметки, предсказаний моделей и аналитических SQL‑запросов.

Оба курсовых проекта (ML и БД) реализованы как единая система.

---

# 1. Описание проекта

## 1.1. Функционал

**Tone & Intent Identifier: ML + DB Suite** позволяет:

- принимать текстовые сообщения (небольшие сообщения, комментарии, письма);  
- определять два параметра:
  - **Tone (тон сообщения)**
  - **Intent (коммуникативное намерение)**
- сохранять сообщения, истинные метки и предсказания в **PostgreSQL**;  
- выполнять аналитические SQL‑запросы для оценки качества моделей.

---

## 1.2. Для кого проект

- Студенты дисциплин по машинному обучению и базам данных.  
- Новички, которые хотят увидеть:
  - понятный ML‑пайплайн на классических и нейронных методах;
  - правильно структурированную БД, поддерживающую ML‑эксперименты.

---

# 2. Система меток (тон и намерение)

Проект использует **два отдельных классификатора**, формируя **два независимых предсказания**: тон и намерение.

## 2.1. Метки тона

1. `neutral` — нейтральный  
2. `positive_friendly` — дружелюбный  
3. `negative_rude` — грубый  
4. `sarcastic_ironic` — саркастичный  
5. `formal` — формальный  
6. `informal_casual` — неформальный, разговорный  

## 2.2. Метки намерения

7. `request` — запрос информации или действия  
8. `command_instruction` — указание / инструкция  
9. `complaint` — жалоба  
10. `praise_appreciation` — благодарность / похвала  
11. `clarification_question` — уточняющий вопрос  
12. `statement_information` — утверждение / сообщение информации  

---

# 3. Установка и запуск

> Разработка и тестирование выполнялись на **Windows 10**.

## 3.1. Требования

- Python **3.10+**  
- Git  
- PostgreSQL **14 или 15**  
- (Необязательно) pgAdmin  

---

## 3.2. Клонирование репозитория

```bash
git clone https://github.com/<username>/tone-intent-identifier-ml-db.git
cd tone-intent-identifier-ml-db
```

---

## 3.3. Создание виртуального окружения

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

## 3.4. Установка зависимостей

```bash
pip install -r requirements.txt
```

---

## 3.5. Настройка PostgreSQL

1. Создайте базу данных:

```sql
CREATE DATABASE tone_intent_db;
```

2. Установите переменные окружения:

```powershell
setx DB_HOST "localhost"
setx DB_PORT "5432"
setx DB_NAME "tone_intent_db"
setx DB_USER "<ваш_логин>"
setx DB_PASSWORD "<ваш_пароль>"
```

3. Примените схему БД:

```bash
psql -U <ваш_логин> -d tone_intent_db -f db_project/schema.sql
```

---

# 4. Примеры использования

## 4.1. Предобработка данных

```bash
python -m ml_project.preprocess ^
  --input data/raw/messages.csv ^
  --output data/processed/messages_clean.csv
```

---

## 4.2. Обучение моделей тона и намерения

```bash
python -m ml_project.train ^
  --config configs/tone_tfidf_logreg.yml
```

```bash
python -m ml_project.train ^
  --config configs/intent_tfidf_logreg.yml
```

---

## 4.3. Оценка моделей

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

## 4.4. Предсказание на новых данных

```bash
python -m ml_project.predict ^
  --tone-model outputs/models/tone_model.pkl ^
  --intent-model outputs/models/intent_model.pkl ^
  --data data/processed/messages_clean.csv ^
  --output outputs/predictions/predictions.csv
```

---

## 4.5. Загрузка данных в БД

```bash
python -m db_project.load_results ^
  --messages data/processed/messages_clean.csv ^
  --predictions outputs/predictions/predictions.csv
```

---

## 4.6. Выполнение SQL‑запросов

```bash
psql -d tone_intent_db -f db_project/queries/tone_accuracy_by_label.sql
```

---

# 5. Структура репозитория

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
├─ README_EN.md
├─ .gitignore
└─ requirements.txt
```

---

# 6. Технические требования

- **Язык программирования:** Python 3.10+  
- **ML‑библиотеки:** pandas, numpy, scikit-learn, tensorflow/keras  
- **База данных:** PostgreSQL 14/15  
- **ОС:** Windows 10  
- **Оборудование:** любой стандартный ноутбук  

---

# 7. Развёртывание и хостинг (планируется)

В учебной версии проект работает локально.  
Для хостинга возможно использовать:

- VPS (Hetzner, Contabo, DigitalOcean)  
- Бесплатные облачные PostgreSQL‑сервисы (Render, Railway, Supabase)  

---

# 8. Автор и роли

**Автор:** Хаггаг Амира  
Группа **ИСП9-Kh32**

**Роли:**
- Разработка ML‑модуля  
- Проектирование и разработка БД  
- Подготовка технической документации (RU + EN)

---

# 9. Контакты

- Email: akiracreates.comms@gmail.com  
- Telegram: https://t.me/akiracreates  
- GitHub: https://github.com/akiracreates  
