import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('credit_score_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit page config
st.set_page_config(page_title="Credit Scoring App", layout="wide")
st.sidebar.title("📄 Upload Your CSV")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

st.title("📊 Credit Scoring Prediction App")
st.write("Upload a dataset — we'll clean the data, encode categories, and predict safely!")

# Model features
model_features = [
    'Age', 'Occupation', 'Annual_Income', 'Monthly_Inhand_Salary',
    'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
    'Type_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment',
    'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Credit_Mix',
    'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Payment_of_Min_Amount',
    'Total_EMI_per_month', 'Payment_Behaviour', 'Monthly_Balance',
    'Credit_History_Age_Months'
]

numeric_features = [
    'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
    'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
    'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
    'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month',
    'Monthly_Balance', 'Credit_History_Age_Months'
]

categorical_features = [
    'Occupation', 'Type_of_Loan', 'Credit_Mix',
    'Payment_of_Min_Amount', 'Payment_Behaviour'
]

# Label encoder maps
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
        'Entrepreneur': 4, 'Scientist': 5, 'Developer': 6,
        'Healthcare': 7, 'Media_Manager': 8, 'Government': 9,
        'Accountant': 10, 'Unknown': 2
    },
    'Type_of_Loan': {
        'Auto Loan': 0, 'Personal Loan': 1, 'Home Loan': 2,
        'Credit-Builder Loan': 3, 'Debt Consolidation Loan': 4,
        'Payday Loan': 5, 'Student Loan': 6, 'Mortgage Loan': 7,
        'Unknown': 1
    }
}

# App logic
if uploaded_file is not None:
    try:
        with st.spinner("Processing your file..."):
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")

            # Fill missing numeric features
            for feature in numeric_features:
                if feature not in df.columns:
                    df[feature] = 0

            # Safely convert numeric columns (fixes '28_', '40_' errors)
            for feature in numeric_features:
                df[feature] = pd.to_numeric(df[feature], errors='coerce')

            # Fill any NaN after numeric cleaning
            df[numeric_features] = df[numeric_features].fillna(0)

            # Encode categorical features
            for feature in categorical_features:
                if feature not in df.columns:
                    df[feature] = 'Unknown'
                df[feature] = df[feature].fillna('Unknown').map(label_encoders[feature])
                df[feature] = df[feature].fillna(label_encoders[feature]['Unknown']).astype(int)

            # Arrange columns
            input_df = df[model_features]

            # Scale all features
            scaled_input = scaler.transform(input_df)

            # Predict
            preds = model.predict(scaled_input)
            probas = model.predict_proba(scaled_input)

            # Attach predictions
            df['Predicted_Class'] = preds
            df['Probability_Poor'] = probas[:, 0]
            df['Probability_Standard'] = probas[:, 1]
            df['Probability_Good'] = probas[:, 2]

            # Display
            st.subheader("🔍 Prediction Results")
            st.dataframe(df)

            # Download results
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV with Predictions",
                data=csv,
                file_name="predictions_with_results.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

else:
    st.info("📌 Please upload a CSV file to start.")

import matplotlib.pyplot as plt

# ===============================
# 📊 Prediction Class Distribution Chart
# ===============================
st.subheader("📊 Credit Score Prediction Distribution")

# Count the number of Poor / Standard / Good
pred_counts = df['Predicted_Class'].value_counts().sort_index()

# Map class numbers to readable names
class_labels = {0: 'Poor', 1: 'Standard', 2: 'Good'}
pred_counts.index = pred_counts.index.map(class_labels)

# Bar Plot
fig1, ax1 = plt.subplots()
pred_counts.plot(kind='bar', ax=ax1)
ax1.set_ylabel('Number of Customers')
ax1.set_xlabel('Predicted Credit Score')
ax1.set_title('Distribution of Credit Score Predictions')
st.pyplot(fig1)

# ===============================
# 🥧 Pie Chart (optional, nice)
# ===============================
st.subheader("🥧 Credit Score Prediction Percentages")

fig2, ax2 = plt.subplots()
ax2.pie(pred_counts, labels=pred_counts.index, autopct='%1.1f%%', startangle=90, shadow=True)
ax2.axis('equal')  # Equal aspect ratio ensures pie is circular.
st.pyplot(fig2)

# ===============================
# 📈 Average Probabilities (optional)
# ===============================
st.subheader("📈 Average Prediction Probabilities")

# Show mean probabilities across all records
avg_probs = df[['Probability_Poor', 'Probability_Standard', 'Probability_Good']].mean()

fig3, ax3 = plt.subplots()
avg_probs.plot(kind='bar', ax=ax3)
ax3.set_ylabel('Average Probability')
ax3.set_title('Average Confidence per Credit Score Class')
st.pyplot(fig3)
