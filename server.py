import streamlit as st
import pickle
import pandas as pd
import numpy as np
model = pickle.load(open('xgb_tuned.pkl', 'rb'))

def predict_promotion(employee_id, department, region, education, gender, recruitment_channel, no_of_trainings, previous_year_rating, length_of_service, awards_won, avg_training_score,  age_group):

    prediction = model.predict(pd.DataFrame([[employee_id, department, region, education, gender, recruitment_channel, no_of_trainings, previous_year_rating, length_of_service, awards_won, avg_training_score,  age_group]], columns=['employee_id', 'department', 'region', 'education', 'gender', 'recruitment_channel', 'no_of_trainings', 'previous_year_rating', 'length_of_service', 'awards_won?', 'avg_training_score', 'age_group']))
    return prediction

st.title("HR Analytics: Employee Promotion Prediction")
html_temp = """ <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">HR Analytics: Employee Promotion Prediction </h2>

    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)

employee_id = st.number_input("Employee ID")
department = st.number_input("Department")
gender = st.number_input("Gender")
region = st.number_input("Region")
education = st.number_input("Education")
recruitment_channel = st.number_input("Recruitment channel")
no_of_trainings = st.number_input("No. of trainings")
age = st.number_input("Age")
previous_year_rating = st.number_input("Previous year rating")
length_of_service = st.number_input("Length of service")
awards_won = st.number_input("Awards won")
avg_training_score = st.number_input("Average training score")

result=""
if st.button("Predict"):
    result=predict_promotion(employee_id, department, region, education, gender, recruitment_channel, no_of_trainings, age, previous_year_rating, length_of_service, awards_won, avg_training_score)
    st.success('The output is {}'.format(result))



