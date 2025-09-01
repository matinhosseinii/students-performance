import os
import sys
import joblib  # Or you can use 'import pickle'
from StudentsPerformance.exception.custom_exception import CustomException
from StudentsPerformance.logginig.logger import setup_logger

logger = setup_logger()

def save_object(file_path: str, obj):
    """
    Saves a Python object to a file using joblib.
    """
    try:
        # Get the directory path from the full file path
        dir_path = os.path.dirname(file_path)

        # Create the directory if it does not already exist
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in "write binary" mode and save the object
        with open(file_path, "wb") as file_obj:
            joblib.dump(obj, file_obj)  # writes the object into the file 
        
        logger.info(f"Object saved successfully at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)