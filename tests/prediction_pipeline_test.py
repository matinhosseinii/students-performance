import pytest
import pandas as pd
from StudentsPerformance.pipelines.prediction_pipeline import PredictPipeline, CustomData

# --- Define the different test cases ---
# Each tuple represents a different student profile to test.
test_cases = [
    # Test Case 1: Normal, average student (your original test)
    ('male', 'group A', "master's degree", 'standard', 'none', 100, 100),
    
    # Test Case 2: High-performing student (might predict > 100)
    ('female', 'group C', "bachelor's degree", 'standard', 'completed', 80, 50),
    
    # Test Case 3: Low-performing student (might predict < 0)
    ('female', 'group C', 'some high school', 'free/reduced', 'none', 0, 0)
]

@pytest.mark.parametrize(
    "gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score",
    test_cases
)
def test_prediction_pipeline(gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score):
    """
    Tests the end-to-end prediction pipeline with multiple data samples,
    including normal and edge cases.
    """
    # 1. Simulate new, raw data using the parameters for this test run
    new_student_data = CustomData(
        gender=gender,
        race_ethnicity=race_ethnicity,
        parental_level_of_education=parental_level_of_education,
        lunch=lunch,
        test_preparation_course=test_preparation_course,
        reading_score=reading_score,
        writing_score=writing_score
    )

    # 2. Convert the new data into a DataFrame
    new_data_df = new_student_data.get_data_as_dataframe()

    # 3. Create an instance of the prediction pipeline
    prediction_pipeline = PredictPipeline()

    # 4. Make a prediction
    predicted_score = prediction_pipeline.predict(new_data_df)

    # 5. Assert the results
    # These assertions now run for EVERY test case, ensuring the clipping
    # logic in your pipeline works correctly for the edge cases.
    assert isinstance(predicted_score, float), "Prediction should be a float"
    assert 0 <= predicted_score <= 100, "Prediction is outside the valid score range"