import pytest
from StudentsPerformance.pipelines.prediction_pipeline import PredictPipeline, CustomData

def test_prediction_pipeline():
    """
    Tests the end-to-end prediction pipeline.

    This test simulates creating new data, running it through the prediction
    pipeline, and asserts that the output is of the expected type and within
    a logical range.
    """
    # 1. Simulate new, raw data
    new_student_data = CustomData(
        gender='female',
        race_ethnicity='group C',
        parental_level_of_education="bachelor's degree",
        lunch='standard',
        test_preparation_course='completed',
        reading_score=85,
        writing_score=88
    )

    # 2. Convert the new data into a DataFrame
    new_data_df = new_student_data.get_data_as_dataframe()

    # 3. Create an instance of the prediction pipeline
    prediction_pipeline = PredictPipeline()

    # 4. Make a prediction
    predicted_score = prediction_pipeline.predict(new_data_df)

    # 5. Assert the results
    # Check that the prediction is a float
    assert isinstance(predicted_score, float), "Prediction should be a float"
    
    # Check that the prediction is within a logical range (e.g., 0-100 for a score)
    assert 0 <= predicted_score <= 100, "Prediction is outside the valid score range"
