# 💳 Credit Card Fraud Detection System
A complete end-to-end Machine Learning + ANN project with Streamlit deployment.

---

## 📁 Project Structure

```
credit_card_fraud_detection/
├── venv/                   ← Virtual environment (created by you)
├── creditcard.csv          ← (Download from Kaggle - see step below)
├── requirements.txt
├── README.md
├── preprocess.py           ← Reusable preprocessing utilities
├── train_models.py         ← Standalone model training script
├── app.py                  ← Streamlit web app for deployment
└── fraud_detection.ipynb   ← Full analysis notebook
```

---

## 🚀 Quick Start

### Step 1: Download the Dataset
1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Download `creditcard.csv`
3. Place it inside the `credit_card_fraud_detection/` folder

### Step 2: Create & Activate Virtual Environment
```bash
# From inside credit_card_fraud_detection/
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Train the Models
```bash
python train_models.py
```
This saves:
- `best_model.pkl` — Best ML model (Random Forest)
- `scaler.pkl` — Fitted StandardScaler
- `ann_model.h5` — Trained ANN model

### Step 5: Run the Streamlit App
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### Step 6: Explore the Notebook
```bash
jupyter notebook fraud_detection.ipynb
```

> 💡 Always activate the `venv` before every session.

---

## 📊 Models Included
| Model | Type |
|-------|------|
| Logistic Regression | ML |
| Decision Tree | ML |
| Random Forest | ML |
| XGBoost | ML |
| ANN (Keras) | Deep Learning |

---

## 📈 Key Features
- EDA with class distribution, heatmap, and feature visualizations
- SMOTE for class imbalance handling
- GridSearchCV hyperparameter tuning
- ROC-AUC comparison across all models
- Confusion matrices for every model
- Overfitting analysis (train vs validation accuracy)
- Production-ready Streamlit app

---

## ⚠️ Note
The dataset is highly imbalanced (~0.17% fraud). SMOTE is applied to balance the training set.
All dependencies are isolated inside `venv/` — nothing installs globally.
