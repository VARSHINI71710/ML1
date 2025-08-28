import streamlit as st
import pickle
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

with open("model.pkl", "rb") as file:
    scaler, model = pickle.load(file)

st.title("📱 Telecom Customer Churn Prediction")
st.write("Enter customer details below to predict churn:")

# -------------------------
# Input fields
# -------------------------
monthly_charges = st.number_input("Monthly Charges ($):", min_value=0, max_value=500, value=70)
tenure_months = st.number_input("Tenure (Months):", min_value=0, max_value=100, value=8)
support_calls = st.number_input("Number of Support Calls:", min_value=0, max_value=50, value=4)

# Contract type dropdown with mapping
contract_options = {"Monthly": 1, "Yearly": 2}
contract_type_label = st.selectbox(
    "Contract Type:",
    options=list(contract_options.keys())
)
contract_type = contract_options[contract_type_label]

# -------------------------
# Predict button
# -------------------------
if st.button("🔮 Predict Churn"):
    # Prepare input as DataFrame with correct column names
    input_df = pd.DataFrame(
        [[monthly_charges, tenure_months, support_calls, contract_type]],
        columns=['monthly_charges', 'tenure_months', 'support_calls', 'contract_type']
    )

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]  # Probability of churn

    # -------------------------
    # Display results
    # -------------------------
    st.write("### Scaled Input Features:")
    st.write(input_scaled)

    st.write("### Prediction Probability of Churn:")
    st.write(f"{probability:.2f}")

    if prediction == 1:
        st.error("❌ This customer is likely to churn.")
    else:
        st.success("✅ This customer is not likely to churn.")
