import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('credit_score_model.pkl')
scaler = joblib.load('scaler.pkl')

# Page setup
st.set_page_config(page_title="Credit Scoring App", layout="wide")
st.sidebar.title("📄 Upload Your CSV")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

st.title("📊 Credit Scoring Batch Prediction")
st.write("Upload **any dataset** — we’ll fill missing features, encode categories, and predict safely.")

# === Final model features (from training)
model_features = [
    'Age', 'Occupation', 'Annual_Income', 'Monthly_Inhand_Salary',
    'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
    'Type_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment',
    'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Credit_Mix',
    'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Payment_of_Min_Amount',
    'Total_EMI_per_month', 'Payment_Behaviour', 'Monthly_Balance',
    'Credit_History_Age_Months'
]

categorical_features = [
    'Occupation', 'Type_of_Loan', 'Credit_Mix',
    'Payment_of_Min_Amount', 'Payment_Behaviour'
]

numeric_features = [f for f in model_features if f not in categorical_features]

# === LabelEncoder mappings (must match training!)
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

# === Main logic
if uploaded_file is not None:
    try:
        with st.spinner("Processing..."):
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")

            # Fill missing numeric features
            for feature in numeric_features:
                if feature not in df.columns:
                    df[feature] = 0
            df[numeric_features] = df[numeric_features].fillna(0)

            # Encode categorical features
            for col in categorical_features:
                if col not in df.columns:
                    df[col] = 'Unknown'
                df[col] = df[col].fillna('Unknown').map(label_encoders[col])
                df[col] = df[col].fillna(label_encoders[col]['Unknown']).astype(int)

            # Ensure column order
            input_df = df[model_features]

            # Scale numeric
            scaled_numeric = scaler.transform(input_df[numeric_features])
            scaled_numeric_df = pd.DataFrame(scaled_numeric, columns=numeric_features)

            # Combine
            final_input = pd.concat([
                scaled_numeric_df.reset_index(drop=True),
                input_df[categorical_features].reset_index(drop=True)
            ], axis=1)

            # ✅ Predict using .values to skip feature name check
            preds = model.predict(final_input.values)
            probas = model.predict_proba(final_input.values)

            df['Predicted_Class'] = preds
            df['Probability_Poor'] = probas[:, 0]
            df['Probability_Standard'] = probas[:, 1]
            df['Probability_Good'] = probas[:, 2]

            st.subheader("🔍 Prediction Results")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV with Predictions",
                data=csv,
                file_name='predictions_with_results.csv',
                mime='text/csv',
            )

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
else:
    st.info("📌 Please upload a CSV file to get started.")
