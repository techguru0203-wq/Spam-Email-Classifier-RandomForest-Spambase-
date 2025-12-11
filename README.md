# 📘 **Spam Email Classifier (spam_email_classifier)**

# Spam Email Classifier (Spambase Dataset)

This project builds a machine-learning classifier that identifies whether an email is spam or not using the **UCI Spambase dataset** via `fetch_openml`.

---

## 🚀 Features
- Downloads dataset automatically from OpenML  
- Trains a RandomForest classifier  
- Outputs accuracy + classification report  
- Includes prediction script for quick demos  

---

## 📂 Project Structure
```text
spam_email_classifier/
├── src/
│ ├── train.py
│ └── predict.py
├── requirements.txt
└── README.md
```

---

## 🔧 Installation
```bash
python -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
pip install -r requirements.txt
```

---

## 🧠 Train Model
```bash
python src/train.py
```

---

## 🔍 Predict Sample Emails
```bash
python src/predict.py
```