import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests

# --- Config: External URLs for model files ---
MODEL_URL = "https://huggingface.co/your-username/credit-model/resolve/main/credit_score_model.pkl"
SCALER_URL = "https://huggingface.co/your-username/credit-model/resolve/main/scaler.pkl"

# --- Download model files if not already present ---
def download_file(url, filename):
    if not os.path.exists(filename):
        with st.spinner(f"Downloading {filename}... ⏳"):
            r = requests.get(url)
            with open(filename, 'wb') as f:
                f.write(r.content)
            st.success(f"✅ {filename} downloaded successfully!")

download_file(MODEL_URL, 'credit_score_model.pkl')
download_file(SCALER_URL, 'scaler.pkl')

# --- Load Model and Scaler ---
model = joblib.load('credit_score_model.pkl')
scaler = joblib.load('scaler.pkl')  # comment this if not used

# --- Define Features your model needs ---
model_features = [
    'Credit_Mix',
    'Payment_Behaviour',
    'Credit_History_Age_Months',
    'Payment_of_Min_Amount',
    'Num_of_Delayed_Payment',
    'Interest_Rate'
]

# --- Streamlit Page Settings ---
st.set_page_config(page_title="Credit Scoring Batch Prediction", layout="wide")

# --- Sidebar for file upload ---
st.sidebar.title("📄 Upload your CSV data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# --- Main App Interface ---
st.title("📊 Credit Scoring Model")
st.write("Upload **any dataset**. We'll automatically find needed features and generate credit predictions.")

if uploaded_file is not None:
    try:
        with st.spinner('Processing uploaded data... Please wait ⏳'):
            # Load CSV
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded and processed successfully!")

            # Check available features
            available_features = [col for col in model_features if col in df.columns]

            if len(available_features) < len(model_features):
                missing = list(set(model_features) - set(available_features))
                st.warning(f"⚠️ Missing required columns for prediction: {missing}")

            if len(available_features) == 0:
                st.error("❌ No required features found in uploaded file. Cannot predict.")
            else:
                # Proceed with available data
                input_df = df[model_features]  # Model expects full feature order
                input_scaled = scaler.transform(input_df)

                # Make predictions
                preds = model.predict(input_scaled)
                probas = model.predict_proba(input_scaled)

                # Attach predictions to original data
                df['Predicted_Class'] = preds
                df['Probability_Poor'] = probas[:, 0]
                df['Probability_Standard'] = probas[:, 1]
                df['Probability_Good'] = probas[:, 2]

                # Display predictions
                st.subheader("🔍 Prediction Results")
                st.dataframe(df)

                # Allow download
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
