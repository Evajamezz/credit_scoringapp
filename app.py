import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('credit_score_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define numeric features (used during training)
numeric_features = [
    'Age', 'Annual_Income', 'Monthly_Inhand_Salary',
    'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
    'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
    'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio',
    'Total_EMI_per_month', 'Monthly_Balance', 'Credit_History_Age_Months'
]

# Set page config
st.set_page_config(page_title="Credit Scoring App", layout="wide")
st.sidebar.title("📄 Upload CSV")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

st.title("📊 Credit Scoring Prediction")
st.write("Upload a CSV with the required numeric features. We'll predict credit score categories.")

# App logic
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")

        # Fill missing numeric features
        for feature in numeric_features:
            if feature not in df.columns:
                df[feature] = 0
        df[numeric_features] = df[numeric_features].fillna(0)

        input_df = df[numeric_features]
        scaled_input = scaler.transform(input_df)

        preds = model.predict(scaled_input)
        probas = model.predict_proba(scaled_input)

        df['Predicted_Class'] = preds
        df['Probability_Poor'] = probas[:, 0]
        df['Probability_Standard'] = probas[:, 1]
        df['Probability_Good'] = probas[:, 2]

        st.subheader("🔍 Prediction Results")
        st.dataframe(df)

        # Download results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="credit_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
else:
    st.info("📌 Please upload a CSV file to begin.")
