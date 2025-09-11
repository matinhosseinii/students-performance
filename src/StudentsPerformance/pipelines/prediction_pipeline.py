import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from StudentsPerformance.exception.custom_exception import CustomException
from StudentsPerformance.logginig.logger import setup_logger
from StudentsPerformance.utils.common import load_object

logger = setup_logger()

@dataclass
class PredictPipelineConfig:
    # Define the paths to the pre-trained model and preprocessor objects
    model_path = os.path.join('models', 'ridge_model.pkl')
    preprocessor_path = os.path.join('models', 'preprocessor.pkl')
    
class PredictPipeline:
    """
    This class defines the pipeline for making predictions on new data.
    """
    def __init__(self):
        self.predict_pipeline_config = PredictPipelineConfig()

    def predict(self, features: pd.DataFrame) -> float:
        """
        Takes new data as a DataFrame, processes it, and returns a prediction.

        Args:
            features (pd.DataFrame): A DataFrame containing the new, raw input data
                                     with the same columns as the original training data.

        Returns:
            float: The predicted value (e.g., the math score).
        """
        try:
            logger.info("Loading preprocessor and model artifacts.")
            # Load the saved model and preprocessor objects
            model = load_object(file_path=self.predict_pipeline_config.model_path)
            preprocessor = load_object(file_path=self.predict_pipeline_config.preprocessor_path)
            logger.info("Artifacts loaded successfully.")

            logger.info("Applying preprocessor to the new data.")
            # Apply the preprocessor to transform the new input data
            data_scaled = preprocessor.transform(features)
            
            logger.info("Making prediction.")
            # Make a prediction using the loaded model
            prediction = model.predict(data_scaled)
            
            clipped_prediction = np.clip(prediction[0], 0, 100)

            # Since we expect a single prediction, we return the first element
            return clipped_prediction

        except Exception as e:
            logger.error("An error occurred during the prediction process.")
            raise CustomException(e, sys)

class CustomData:
    """
    This class is responsible for mapping data from a web form or API
    to a pandas DataFrame that can be used by the prediction pipeline.
    """
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self) -> pd.DataFrame:
        """
        Returns the custom data as a pandas DataFrame.
        """
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)