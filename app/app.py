import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title='Churn Predictor', layout='wide')

with open('models/logisticregression_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/feature_columns.pkl', 'rb') as f:
    feature_order = pickle.load(f)

df = pd.read_csv('data/cleaned_telco.csv')
columns = feature_order

custom_binary_labels = {
    'gender': ['Female', 'Male'],
    'SeniorCitizen': ['No', 'Yes'],
    'Partner': ['No', 'Yes'],
    'Dependents': ['No', 'Yes'],
    'PhoneService': ['No', 'Yes'],
    'PaperlessBilling': ['No', 'Yes']
}

custom_display_names = {
    'gender': 'Gender',
    'SeniorCitizen': 'Senior Citizen',
    'Partner': 'Has Partner',
    'Dependents': 'Has Dependents',
    'PhoneService': 'Phone Service',
    'PaperlessBilling': 'Paperless Billing'
}

st.title('Customer Churn Prediction')
st.markdown("Fill in the details below to check if a customer is likely to churn.")

user_input = {}

with st.form("churn_form"):
    st.header("Demographic Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        user_input['tenure'] = st.number_input("Tenure (months)", min_value=0, max_value=100, value=5)
    with col2:
        user_input['MonthlyCharges'] = st.number_input("Monthly Charges", min_value=0.0, max_value=150.0, value=70.0)
    with col3:
        user_input['TotalCharges'] = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=300.0)

    st.header("Binary & Service Features")
    for col in custom_binary_labels:
        if col not in ['tenure', 'MonthlyCharges', 'TotalCharges']:
            display_name = custom_display_names.get(col, col.replace('_', ' '))
            user_input[col] = st.selectbox(display_name, options=custom_binary_labels[col])

    st.header("Other Encoded Features")
    for col in columns:
        if col not in custom_binary_labels and col not in ['tenure', 'MonthlyCharges', 'TotalCharges']:
            if df[col].nunique() == 2:
                user_input[col] = st.selectbox(col.replace('_', ' '), options=['No', 'Yes'])
            else:
                user_input[col] = st.number_input(col.replace('_', ' '), value=float(df[col].mean()))

    submit = st.form_submit_button("Predict")

if submit:
    for k in user_input:
        if isinstance(user_input[k], str):
            if user_input[k] in ['Yes', 'Male']:
                user_input[k] = 1
            elif user_input[k] in ['No', 'Female']:
                user_input[k] = 0

    df_input = pd.DataFrame([user_input])
    df_input = df_input[feature_order]
    df_scaled = scaler.transform(df_input)
    prediction = model.predict(df_scaled)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("Customer is likely to churn")
    else:
        st.success("Customer is likely to stay")
