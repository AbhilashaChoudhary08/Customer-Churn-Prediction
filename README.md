# Customer Churn Prediction Capstone Project
This project implements a comprehensive machine learning pipeline to predict customer attrition within the telecommunications industry. The system analyzes demographic data, account information, and service usage patterns to identify customers at risk of leaving the service.

## Project Structure
* **customer-churn-prediction/**
* **data/**: Contains the raw and preprocessed datasets.
* WAFn-UseC-Telco-Customer-Churn.csv: The original raw dataset.
* cleaned_telco.csv: Dataset after feature engineering and cleaning.


* **models/**: Serialized model artifacts and preprocessing objects.
* logisticregression_model.pkl: Trained Logistic Regression model.
* randomforest_model.pkl: Trained Random Forest classifier.
* xgboost_model.pkl: Trained XGBoost classifier.
* scaler.pkl: StandardScaler object used for data normalization.
* feature_columns.pkl: Ordered list of input features for the model.


* **src/**: Source code for the data science pipeline.
* preprocessing.py: Scripts for data cleaning and categorical encoding.
* model_training.py: Logic for model training and evaluation.
* eda.py: Scripts for generating Exploratory Data Analysis visualizations.


* **app/**: Web deployment files.
* app.py: Streamlit application for real-time model predictions.


* **requirements.txt**: List of Python dependencies for environment reproduction.
* **README.md**: Project documentation.



## Features

* End-to-end customer churn prediction pipeline from raw data to deployment.
* Automated data preprocessing and feature scaling using Scikit-learn.
* Comparison of multiple classification algorithms including Logistic Regression, Random Forest, and XGBoost.
* Model persistence using Pickle for consistent inference across different environments.
* Interactive user interface built with Streamlit for real-time risk assessment.

## Model Performance Snapshot (Baseline: Logistic Regression)

The primary model yields the following performance metrics on the test set:

* Accuracy: Approximately 78.7%
* Precision: Approximately 62%
* Recall: Approximately 51%
* AUC Score: Approximately 0.70

## How to Run the Project

Follow these instructions to set up the project locally:

1. Clone the repository to your local environment.
2. Install the necessary Python libraries via the terminal:

```bash
pip install -r requirements.txt

```

3. Launch the Streamlit application:

```bash
streamlit run app/app.py

```

---

**Next Step:** Since your code is now successfully pushed to GitHub, would you like me to generate a specific `requirements.txt` file based on the libraries used in your churn prediction scripts?
