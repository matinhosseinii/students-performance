import streamlit as st
import pandas as pd
from StudentsPerformance.pipelines.prediction_pipeline import PredictPipeline, CustomData

# Page title and header
st.set_page_config(page_title="Student Performance Predictor", layout="wide")
st.title("Student Math Score Predictor")
st.write("Enter the student's details to predict their math score.")

# --- Create the input form ---
with st.form("prediction_form"):
    st.header("Student Details")
    
    # Input fields arranged in columns for a cleaner layout
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", options=['male', 'female'])
        race_ethnicity = st.selectbox("Race/Ethnicity", options=['group A', 'group B', 'group C', 'group D', 'group E'])
    
    with col2:
        parental_level_of_education = st.selectbox(
            "Parental Level of Education",
            options=["bachelor's degree", 'some college', "master's degree", "associate's degree", 'high school', 'some high school']
        )
        lunch = st.selectbox("Lunch", options=['standard', 'free/reduced'])

    with col3:
        test_preparation_course = st.selectbox("Test Preparation Course", options=['none', 'completed'])
        reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=50)
        writing_score = st.number_input("Writing Score", min_value=0, max_value=100, value=50)

    # Submit button for the form
    submit_button = st.form_submit_button(label='Predict Score')


# --- Prediction Logic ---
if submit_button:
    with st.spinner("Calculating..."):
        # 1. Create a CustomData instance from the form inputs
        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )
        
        # 2. Convert the data to a DataFrame
        pred_df = data.get_data_as_dataframe()
        
        # 3. Initialize the prediction pipeline and make a prediction
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(pred_df)
        
        # 4. Display the result
        st.success(f"The predicted math score is: **{prediction:.2f}**")
