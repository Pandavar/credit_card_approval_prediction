import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
# Title and description
st.title('Credit Card Approval Prediction')
st.write('Enter your details to predict credit card approval.')

# load pre-trained model
rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
# Form to collect user inputs
with st.form("user_input_form"):
    # Collecting inputs from the user
    gender = st.selectbox("Gender", ["Male", "Female"])
    income = st.number_input("Total Income", min_value=1000, max_value=1000000, step=1000)
    education = st.selectbox("Education Level", ["Secondary", "Higher", "Incomplete Higher", "Lower Secondary"])
    family_status = st.selectbox("Family Status", ["Single", "Married", "Divorced", "Widow"])
    housing_type = st.selectbox("Housing Type", ["Owned Apartment", "House", "Municipal Apartment", "Co-op Apartment"])
    age = st.slider("Age (in years)", min_value=18, max_value=100, value=30)
    years_employed = st.slider("Years Employed", min_value=0, max_value=40, value=5)

    # Submit button
    submit_button = st.form_submit_button("Predict Approval")

# Preprocessing user inputs if form is submitted
if submit_button:
    # Encoding user inputs (you would ideally use the same encoding used during model training)
    gender_encoded = 1 if gender == "Male" else 0
    education_encoded = {
        "Secondary": 2,
        "Higher": 0,
        "Incomplete Higher": 1,
        "Lower Secondary": 3
    }[education]
    family_encoded = {
        "Single": 0,
        "Married": 1,
        "Divorced": 2,
        "Widow": 3
    }[family_status]
    housing_encoded = {
        "Owned Apartment": 0,
        "House": 1,
        "Municipal Apartment": 2,
        "Co-op Apartment": 3
    }[housing_type]

    # Creating input array for the model
    user_input = np.array([[gender_encoded, income, education_encoded, family_encoded, housing_encoded, age, years_employed]])

    # Standardizing the inputs
    user_input_scaled = scaler.fit_transform(user_input)

    
    # Make prediction
    prediction = rf_model.predict(user_input_scaled)

    # Display the result
    if prediction[0] == 1:
        st.success("Credit Card Application Approved!")
    else:
        st.error("Credit Card Application Denied!")
