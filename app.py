import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('credit_score_model.pkl')
scaler = joblib.load('scaler.pkl')

# Page settings
st.set_page_config(page_title="Credit Scoring App", layout="wide")
st.sidebar.title("📄 Upload your CSV data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

st.title("📊 Credit Scoring Batch Prediction App")
st.write("Upload **any dataset**. Missing features will be filled, and categorical features will be encoded safely.")

# ===== Training-time features =====
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

model_features = numeric_features + categorical_features

# ===== LabelEncoder mappings used during training =====
label_encoders = {
    'Payment_of_Min_Amount': {
        'No': 0, 'NM': 1, 'Yes': 2, 'Unknown': 1
    },
    'Payment_Behaviour': {
        'Low_spent_Small_value_payments': 0,
        'Low_spent_Medium_value_payments': 1,
        'High_spent_Small_value_payments': 2,
        'High_spent_Large_value_payments': 3,
        'Unknown': 1
    },
    'Credit_Mix': {
        'Bad': 0, 'Good': 1, 'Standard': 2, 'Unknown': 1
    },
    'Occupation': {
        'Teacher': 0, 'Lawyer': 1, 'Engineer': 2, 'Doctor': 3,
        'Entrepreneur': 4, 'Unknown': 2
    },
    'Type_of_Loan': {
        'Auto Loan': 0, 'Personal Loan': 1, 'Home Loan': 2,
        'Credit-Builder Loan': 3, 'Unknown': 1
    }
}

# ===== Main App Logic =====
if uploaded_file is not None:
    try:
        with st.spinner('🔄 Processing your file...'):
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")

            # Fill missing numeric features with 0
            for feature in numeric_features:
                if feature not in df.columns:
                    df[feature] = 0
            df[numeric_features] = df[numeric_features].fillna(0)

            # Fill and encode categorical features
            for col in categorical_features:
                if col not in df.columns:
                    df[col] = 'Unknown'
                df[col] = df[col].fillna('Unknown').map(label_encoders[col])
                df[col] = df[col].fillna(label_encoders[col]['Unknown'])

            # Arrange columns in exact order
            input_df = df[model_features]

            # Scale only numeric columns
            scaled_numeric = scaler.transform(input_df[numeric_features])
            scaled_numeric_df = pd.DataFrame(scaled_numeric, columns=numeric_features)

            # Combine numeric + encoded categorical
            final_input = pd.concat(
                [scaled_numeric_df.reset_index(drop=True), input_df[categorical_features].reset_index(drop=True)],
                axis=1
            )

            # Predict
            preds = model.predict(final_input)
            probas = model.predict_proba(final_input)

            df['Predicted_Class'] = preds
            df['Probability_Poor'] = probas[:, 0]
            df['Probability_Standard'] = probas[:, 1]
            df['Probability_Good'] = probas[:, 2]

            # Results
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
