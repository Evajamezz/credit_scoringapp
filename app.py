import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests

# ========================
# CONFIG - Update this section!
# ========================
MODEL_URL = "https://huggingface.co/your-username/your-model-repo/resolve/main/credit_score_model.pkl"
SCALER_URL = "https://huggingface.co/your-username/your-model-repo/resolve/main/scaler.pkl"

# ========================
# Function to Download Model & Scaler
# ========================
def download_file(url, filename):
    if not os.path.exists(filename):
        with st.spinner(f"📥 Downloading {filename}..."):
            r = requests.get(url)
            with open(filename, 'wb') as f:
                f.write(r.content)
            st.success(f"✅ {filename} downloaded successfully!")

# ========================
# Download Model and Scaler First
# ========================
download_file(MODEL_URL, "credit_score_model.pkl")
download_file(SCALER_URL, "scaler.pkl")

# ========================
# Load Model and Scaler
# ========================
model = joblib.load('credit_score_model.pkl')
scaler = joblib.load('scaler.pkl')

# ========================
# Streamlit App Setup
# ========================
st.set_page_config(page_title="Credit Scoring App", layout="wide")
st.sidebar.title("📄 Upload your CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

st.title("📊 Credit Scoring Batch Prediction App")
st.write("Upload **any dataset**. We'll find the needed features and generate credit predictions.")

# ========================
# Model Required Features
# ========================
model_features = [
    'Credit_Mix',
    'Payment_Behaviour',
    'Credit_History_Age_Months',
    'Payment_of_Min_Amount',
    'Num_of_Delayed_Payment',
    'Interest_Rate'
]

# ========================
# Main App Logic
# ========================
if uploaded_file is not None:
    try:
        with st.spinner('Processing your file... ⏳'):
            df = pd.read_csv(uploaded_file)

            available_features = [col for col in model_features if col in df.columns]

            if len(available_features) < len(model_features):
                missing = list(set(model_features) - set(available_features))
                st.warning(f"⚠️ Missing columns: {missing}")

            if len(available_features) == 0:
                st.error("❌ No required features found. Cannot predict.")
            else:
                input_df = df[model_features]
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
