from StudentsPerformance.components.data_ingestion import DataIngestion
from StudentsPerformance.components.data_preprocessing import DataPreprocessor
from StudentsPerformance.components.model_trainer import ModelTrainer
from StudentsPerformance.logginig.logger import setup_logger

logger = setup_logger()

# 1. Initialize the components
ingestion = DataIngestion()
preprocessor = DataPreprocessor()
trainer = ModelTrainer()

# 2. Run the pipeline in sequence
try:
    logger.info("Starting the training pipeline.")
    
    # Run data ingestion and get the paths for train and test data
    train_path, test_path = ingestion.initiate_data_ingestion()
    
    # Run data preprocessing using the paths from the ingestion step
    train_arr, test_arr, _ = preprocessor.run_preprocessing(train_path, test_path)
    
    # Run model training using the processed arrays
    score = trainer.initiate_model_training(train_array=train_arr, test_array=test_arr)
    
    logger.info(f'Finished executing training pipeline. Final R2 score: {score:.4f}')

except Exception as e:
    logger.error(f"An error occurred in the training pipeline: {e}")
    # You would raise your custom exception here
    # raise CustomException(e, sys)
