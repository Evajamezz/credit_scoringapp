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

# Model Expected Features
model_features = [
    'Credit_Mix',
    'Payment_Behaviour',
    'Credit_History_Age_Months',
    'Payment_of_Min_Amount',
    'Num_of_Delayed_Payment',
    'Interest_Rate'
]

if uploaded_file is not None:
    try:
        with st.spinner('Reading file... ⏳'):
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")

            # Ensure all required features are present
            for feature in model_features:
                if feature not in df.columns:
                    df[feature] = 0  # Fill missing features with default value (e.g., 0)

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
