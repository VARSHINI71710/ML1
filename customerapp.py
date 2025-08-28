import streamlit as st
import pickle
import numpy as np



rf, scaler = pickle.load(open('trained_model.pkl', 'rb'))


st.title("Churn Prediction App")

st.write("Enter the details below to predict the churn.")



monthly_charges = st.number_input("Monthly Charges ($):",  min_value=0.0, max_value=500.0,  value=70.0, step=1.0)

tenure = st.number_input("Tenure (in months):",min_value=0, max_value=72, value=12,step=1)

total_charges = st.number_input("Total Charges ($):",  min_value=0.0,max_value=10000.0,value=800.0,step=10.0)

contract = st.radio("Contract Type:",options=[0, 1],format_func=lambda x: "Month-to-month" if x == 0 else "One year")


features = np.array([[monthly_charges, tenure, total_charges, contract]])


if st.button("Predict churn"):
    prediction = rf.predict(features)
    probability = rf.predict_proba(features)[0][1]
    st.write("### Prediction:", "Churn" if prediction[0] == 1 else "No Churn")
    st.write("### Probability of churn:", round(probability, 2))
    