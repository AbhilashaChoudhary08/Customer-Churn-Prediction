import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/cleaned_telco.csv')

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Target Distribution:\n", df['Churn'].value_counts())

sns.countplot(x='Churn', data=df)
plt.title('Churn Count')
plt.savefig('eda/churn_count.png')
plt.clf()

sns.histplot(df['tenure'], kde=True)
plt.title('Tenure Distribution')
plt.savefig('eda/tenure_distribution.png')
plt.clf()

sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Monthly Charges by Churn')
plt.savefig('eda/monthly_charges_by_churn.png')
plt.clf()

correlation = df.corr(numeric_only=True)
plt.figure(figsize=(12, 10))
sns.heatmap(correlation, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('eda/correlation_heatmap.png')
plt.clf()
