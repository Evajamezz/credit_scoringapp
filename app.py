import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load('credit_score_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit page config
st.set_page_config(page_title="Credit Scoring App", layout="wide")
st.sidebar.title("📄 Upload Your CSV")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

st.title("📊 Credit Scoring Prediction App")
st.write("Upload your dataset — we'll clean the data, predict Credit Scores, and visualize the results!")

# Define model features
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

# Categorical encoders
label_encoders = {
    'Payment_of_Min_Amount': {'No': 0, 'NM': 1, 'Yes': 2, 'Unknown': 1},
    'Payment_Behaviour': {
        'Low_spent_Small_value_payments': 0, 'Low_spent_Medium_value_payments': 1,
        'High_spent_Small_value_payments': 2, 'High_spent_Large_value_payments': 3,
        'Unknown': 1
    },
    'Credit_Mix': {'Bad': 0, 'Good': 1, 'Standard': 2, 'Unknown': 1},
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

# Main app
if uploaded_file is not None:
    try:
        with st.spinner("Processing your file..."):
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")

            # Clean numeric features
            for feature in numeric_features:
                if feature not in df.columns:
                    df[feature] = 0
            for feature in numeric_features:
                df[feature] = pd.to_numeric(df[feature], errors='coerce')
            df[numeric_features] = df[numeric_features].fillna(0)

            # Encode categorical features
            for feature in categorical_features:
                if feature not in df.columns:
                    df[feature] = 'Unknown'
                df[feature] = df[feature].fillna('Unknown').map(label_encoders[feature])
                df[feature] = df[feature].fillna(label_encoders[feature]['Unknown']).astype(int)

            # Arrange columns
            input_df = df[model_features]

            # Scale
            scaled_input = scaler.transform(input_df)

            # Predict
            preds = model.predict(scaled_input)
            probas = model.predict_proba(scaled_input)

            # Attach predictions
            df['Predicted_Class'] = preds
            df['Probability_Poor'] = probas[:, 0]
            df['Probability_Standard'] = probas[:, 1]
            df['Probability_Good'] = probas[:, 2]

            # Show results
            st.subheader("🔍 Prediction Results")
            st.dataframe(df)

            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV with Predictions",
                data=csv,
                file_name="predictions_with_results.csv",
                mime="text/csv"
            )

            # ✅ Show Visualizations if Predictions are ready
            if 'Predicted_Class' in df.columns:
                st.subheader("📊 Credit Score Prediction Distribution")

                pred_counts = df['Predicted_Class'].value_counts().sort_index()
                class_labels = {0: 'Poor', 1: 'Standard', 2: 'Good'}
                pred_counts.index = pred_counts.index.map(class_labels)

                fig1, ax1 = plt.subplots()
                pred_counts.plot(kind='bar', ax=ax1)
                ax1.set_ylabel('Number of Predictions')
                ax1.set_xlabel('Credit Score Category')
                ax1.set_title('Distribution of Predicted Credit Scores')
                st.pyplot(fig1)

                st.subheader("🥧 Prediction Percentages")
                fig2, ax2 = plt.subplots()
                ax2.pie(pred_counts, labels=pred_counts.index, autopct='%1.1f%%', startangle=90, shadow=True)
                ax2.axis('equal')
                st.pyplot(fig2)

                st.subheader("📈 Average Confidence per Class")
                avg_probs = df[['Probability_Poor', 'Probability_Standard', 'Probability_Good']].mean()
                fig3, ax3 = plt.subplots()
                avg_probs.plot(kind='bar', ax=ax3)
                ax3.set_ylabel('Average Probability')
                ax3.set_title('Model Confidence by Class')
                st.pyplot(fig3)

    except Exception as e:
        st.error(f"⚠️ Error during processing: {e}")

else:
    st.info("📌 Please upload a CSV file to get started.")
