import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

df.drop('customerID', axis=1, inplace=True)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df.dropna(inplace=True)

le = LabelEncoder()
binary_cols = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtype == 'object']
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

df = pd.get_dummies(df, drop_first=True)

df.to_csv('data/cleaned_telco.csv', index=False)

print("Preprocessing complete. Cleaned data saved to 'data/cleaned_telco.csv'")
