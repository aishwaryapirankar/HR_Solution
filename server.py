import streamlit as st
import pickle
import pandas as pd
import numpy as np
model = pickle.load(open('xgb_tuned.pkl', 'rb'))

def predict_promotion(employee_id, department, region, education, gender, recruitment_channel, no_of_trainings, previous_year_rating, length_of_service, awards_won, avg_training_score,  age_group):
    if department == 'Technology':
        department = 0
    elif department == 'Analytics':
        department = 1
    elif department == 'Operations':
        department = 2
    elif department == 'Sales':
        department = 3
    elif department == 'Finance':
        department = 4
    elif department == 'HR':
        department = 5
    elif department == 'Legal':
        department = 6
    elif department == 'Procurement':
        department = 7
    elif department == 'Marketing':
        department = 8

    if gender == 'Male':
        gender = 0
    elif gender == 'Female':
        gender = 1

    if education == "Bachelor's":
        education = 0
    elif education == "Master's & above":
        education = 1
    elif education == "Below Secondary":
        education = 2

    if recruitment_channel == 'Sourcing':
        recruitment_channel = 0
    elif recruitment_channel == 'Other':
        recruitment_channel = 1
    elif recruitment_channel == 'Referred':
        recruitment_channel = 2

    prediction = model.predict(pd.DataFrame([[employee_id, department, region, education, gender, recruitment_channel, no_of_trainings, previous_year_rating, length_of_service, awards_won, avg_training_score,  age_group]], columns=['employee_id', 'department', 'region', 'education', 'gender', 'recruitment_channel', 'no_of_trainings', 'previous_year_rating', 'length_of_service', 'awards_won?', 'avg_training_score', 'age_group']))
    return prediction

st.title("HR Analytics: Employee Promotion Prediction")
html_temp = """ <div style="background-color:#480F17;padding:10px">
    <h2 style="color:white;text-align:center;">HR Analytics: Employee Promotion Prediction </h2>

    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)

employee_id = st.number_input("Employee ID")
department = st.selectbox('Department', ['Technology', 'Analytics', 'Operations', 'Sales', 'Finance', 'HR', 'Legal', 'Procurement', 'Marketing'])
gender = st.selectbox('Gender', ['Male', 'Female'])
region = st.number_input("Region No.")
education = st.selectbox('Education Level', ['Bachelor\'s', 'Master\'s & above', 'Below Secondary'])
recruitment_channel = st.selectbox('Recruitment Channel', ['Sourcing', 'Other', 'Referred'])
no_of_trainings = st.slider('Number of Trainings Completed', 1, 10)
age = st.slider('Age', 20, 60)
previous_year_rating = st.selectbox('Previous Year Rating', [1.0, 2.0, 3.0, 4.0, 5.0])
length_of_service = st.slider('Length of Service in Years', 1, 35)
awards_won = st.number_input("Awards won")
avg_training_score = st.number_input("Average training score")

result=""
if st.button("Predict"):
    result=predict_promotion(employee_id, department, region, education, gender, recruitment_channel, no_of_trainings, age, previous_year_rating, length_of_service, awards_won, avg_training_score)
    if result == 1:
        st.success('Congrats! Employee is promoted!')
    else:
        st.success('Sorry, Employee is not promoted.')



