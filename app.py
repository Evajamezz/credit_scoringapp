import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ========================
# Load Model and Scaler
# ========================
model = joblib.load('credit_score_model.pkl')
scaler = joblib.load('scaler.pkl')

# ========================
# Streamlit App Setup
# ========================
st.set_page_config(page_title="Credit Scoring App", layout="wide")
st.sidebar.title("📄 Upload your CSV data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

st.title("📊 Credit Scoring Batch Prediction App")
st.write("Upload **any dataset**. We'll handle missing features and predict safely!")

# ========================
# FULL List of Features Used During Training
# ========================
numeric_features = [
    'Age', 'Annual_Income', 'Changed_Credit_Limit', 'Credit_Utilization_Ratio',
    'Delay_from_due_date', 'Outstanding_Debt', 'Monthly_Inhand_Salary',
    'Monthly_Balance', 'Num_Bank_Accounts', 'Num_Credit_Card', 
    'Num_Credit_Inquiries', 'Num_of_Loan', 'Num_of_Delayed_Payment', 
    'Interest_Rate', 'Credit_History_Age_Months', 'Total_EMI_per_month'
]

categorical_features = [
    'Payment_of_Min_Amount', 'Payment_Behaviour', 'Credit_Mix', 
    'Occupation', 'Type_of_Loan'
]

# Combine all expected features
model_features = numeric_features + categorical_features

# ========================
# Main App Logic
# ========================
if uploaded_file is not None:
    try:
        with st.spinner('🔄 Processing uploaded file...'):
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")

            # Step 1: Auto-fill Missing Columns
            for feature in numeric_features:
                if feature not in df.columns:
                    df[feature] = 0  # Fill missing numeric with 0

            for feature in categorical_features:
                if feature not in df.columns:
                    df[feature] = 'Unknown'  # Fill missing categorical with 'Unknown'

            # Step 2: Ensure Correct Column Order
            input_df = df[model_features]

            # Step 3: Encoding (if necessary)
            # Assuming model is already trained to handle raw categorical text
            # (if LabelEncoder used, you need to encode here)

            # Step 4: Scale Numeric Features Only
            numeric_df = input_df[numeric_features]
            scaled_numeric = scaler.transform(numeric_df)

            # Combine scaled numeric + categorical (unchanged)
            final_input = pd.DataFrame(scaled_numeric, columns=numeric_features)
            final_input = pd.concat([final_input, input_df[categorical_features].reset_index(drop=True)], axis=1)

            # Step 5: Predict
            preds = model.predict(final_input)
            probas = model.predict_proba(final_input)

            df['Predicted_Class'] = preds
            df['Probability_Poor'] = probas[:, 0]
            df['Probability_Standard'] = probas[:, 1]
            df['Probability_Good'] = probas[:, 2]

            st.subheader("🔍 Prediction Results")
            st.dataframe(df)

            # Allow CSV Download
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
