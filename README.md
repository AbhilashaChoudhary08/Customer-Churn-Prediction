# Customer Churn Prediction – Capstone Project

This project predicts whether a telecom customer is likely to churn based on their account features and service usage. It is built using Python (pandas, scikit-learn, xgboost) and deployed via an interactive Streamlit app.

---

## 📁 Project Structure

customer-churn-prediction/
│
├── data/
│ ├── WA*Fn-UseC*-Telco-Customer-Churn.csv # Original dataset
│ └── cleaned_telco.csv # Cleaned dataset after preprocessing
│
├── models/
│ ├── logisticregression_model.pkl # Trained Logistic Regression model
│ ├── randomforest_model.pkl # Trained Random Forest model
│ ├── xgboost_model.pkl # Trained XGBoost model
│ ├── scaler.pkl # StandardScaler object for normalization
│ └── feature_columns.pkl # Ordered list of model input features
│
├── src/
│ ├── preprocessing.py # Cleans and encodes dataset
│ ├── model_training.py # Trains models and saves artifacts
│ └── eda.py # Generates visualizations (EDA)
│
├── app/
│ └── app.py # Streamlit web app for predictions
│
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## 🔍 Features

- End-to-end customer churn prediction pipeline
- Cleaned and encoded dataset using pandas and sklearn
- Trains 3 machine learning models:
  - Logistic Regression
  - Random Forest Classifier
  - XGBoost Classifier
- All models saved as `.pkl` files
- Scaler and feature list saved for consistent deployment
- Streamlit UI for real-time predictions with user-friendly inputs

---

## 📊 Model Performance Snapshot (Logistic Regression)

- Accuracy: ~78.7%
- Precision: ~62%
- Recall: ~51%
- AUC Score: ~0.70

---

## 🚀 How to Run the Project

1. Clone this repo or download the folder
2. Install all dependencies:

```bash
pip install -r requirements.txt
```

---

### 🔧 Launch the Web App

streamlit run app/app.py

---
