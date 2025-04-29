import streamlit as st
import pandas as pd
import joblib

# ========================
# Load Model and Scaler Directly
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
st.write("Upload **any dataset**. We'll detect required features and generate credit predictions!")

# ========================
# Model Expected Features
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
# Main Logic
# ========================
if uploaded_file is not None:
    try:
        with st.spinner('Reading file... ⏳'):
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")

            available_features = [col for col in model_features if col in df.columns]
            missing_features = list(set(model_features) - set(available_features))

            if missing_features:
                st.warning(f"⚠️ Missing columns: {missing_features}")

            if len(available_features) == 0:
                st.error("❌ No usable features found for prediction. Please check your file.")
            else:
                # Proceed with features that exist
                input_df = df[available_features]

                # Since model expects fixed order and full features, handle missing ones carefully
                if len(available_features) < len(model_features):
                    st.error("❌ Cannot predict because some required features are missing.")
                else:
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
