import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load Model and Scaler
model = joblib.load('credit_score_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit Setup
st.set_page_config(page_title="Credit Scoring App", layout="wide")
st.sidebar.title("📄 Upload your CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

st.title("📊 Credit Scoring Batch Prediction App")
st.write("Upload **any dataset**. We'll automatically handle missing features!")

# ========================
# FULL list of features used when training your model
# (Fill this list carefully based on your training phase)
# ========================
model_features = [
    'Age', 'Annual_Income', 'Changed_Credit_Limit', 'Credit_Utilization_Ratio', 
    'Delay_from_due_date', 'Outstanding_Debt', 'Monthly_Inhand_Salary',
    'Total_EMI_per_month', 'Amount_invested_monthly', 'Num_Bank_Accounts',
    'Num_Credit_Card', 'Num_Credit_Inquiries', 'Num_of_Loan',
    'Num_of_Delayed_Payment', 'Payment_of_Min_Amount', 'Payment_Behaviour',
    'Credit_Mix', 'Interest_Rate', 'Credit_History_Age_Months'
    # add ALL features you used in training here
]

if uploaded_file is not None:
    try:
        with st.spinner('Reading file... ⏳'):
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")

            # Fill missing features with 0
            for feature in model_features:
                if feature not in df.columns:
                    df[feature] = 0  # or np.nan

            # Select columns in exact order
            input_df = df[model_features]

            # Scale and Predict
            input_scaled = scaler.transform(input_df)
            preds = model.predict(input_scaled)
            probas = model.predict_proba(input_scaled)

            df['Predicted_Class'] = preds
            df['Probability_Poor'] = probas[:, 0]
            df['Probability_Standard'] = probas[:, 1]
            df['Probability_Good'] = probas[:, 2]

            st.subheader("🔍 Prediction Results")
            st.dataframe(df)

            # Downloadable CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name='predictions_with_results.csv',
                mime='text/csv',
            )

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")

else:
    st.info("📌 Please upload a CSV file to start.")
