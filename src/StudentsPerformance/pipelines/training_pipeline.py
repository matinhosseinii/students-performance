import sys
from StudentsPerformance.exception.custom_exception import CustomException
from StudentsPerformance.logginig.logger import setup_logger
from StudentsPerformance.components.data_ingestion import DataIngestion
from StudentsPerformance.components.data_preprocessing import DataPreprocessor
from StudentsPerformance.components.model_trainer import ModelTrainer

logger = setup_logger()

class TrainingPipeline:
    """
    This class defines and runs the complete model training pipeline.
    """
    def __init__(self):
        # Initialize each component of the pipeline
        self.data_ingestion = DataIngestion()
        self.data_preprocessor = DataPreprocessor()
        self.model_trainer = ModelTrainer()

    def run(self):
        """
        Executes all the steps of the training pipeline in sequence.
        """
        try:
            logger.info("========== Starting the Training Pipeline ==========")
            
            # --- Step 1: Data Ingestion ---
            # This step downloads the data (if needed), removes duplicates,
            # performs the train-test split, and saves the results.
            # It returns the paths to the intermediate train and test CSV files.
            logger.info("Executing Data Ingestion component...")
            train_path, test_path = self.data_ingestion.initiate_data_ingestion()
            logger.info("Data Ingestion completed successfully.")

            # --- Step 2: Data Preprocessing ---
            # This step takes the paths from ingestion, applies the full
            # preprocessing pipeline, and returns the processed arrays and
            # the path to the saved preprocessor object.
            logger.info("Executing Data Preprocessing component...")
            train_arr, test_arr, _ = self.data_preprocessor.run_preprocessing(train_path, test_path)
            logger.info("Data Preprocessing completed successfully.")

            # --- Step 3: Model Training ---
            # This step takes the processed arrays, trains the final model
            # with the best hyperparameters, and saves the model object.
            logger.info("Executing Model Training component...")
            r2_score = self.model_trainer.initiate_model_training(
                train_array=train_arr,
                test_array=test_arr
            )
            logger.info(f"Model Training completed successfully. Final R2 Score: {r2_score:.4f}")
            
            logger.info("========== Training Pipeline Finished ==========")

        except Exception as e:
            logger.error("An error occurred during the training pipeline execution.")
            raise CustomException(e, sys)


if __name__ == '__main__':
    pipeline = TrainingPipeline()
    pipeline.run()

