import os
import sys
import kaggle
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Assuming these are in your project structure
from StudentsPerformance.exception.custom_exception import CustomException
from StudentsPerformance.logginig.logger import setup_logger

logger = setup_logger()

@dataclass
class DataIngestionConfig:
    # Defines the final paths for train/test sets
    train_data_path: str = os.path.join('data', '02_Intermediate', 'train.csv')
    test_data_path: str = os.path.join('data', '02_Intermediate', 'test.csv')
    # Defines the location of the raw data file
    raw_data_path: str = os.path.join('data', '01_Raw', 'StudentsPerformance.csv')
    # Defines the reference of dataset on kaggle
    dataset_reference: str = 'spscientist/students-performance-in-exams'

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Orchestrates the data ingestion process: download, clean duplicates, and split.
        """
        logger.info("Starting the data ingestion process.")
        
        # --- KAGGLE DOWNLOAD LOGIC ---
        # =================================================================
        try:
            
            # Check if the directory exists and has files
            if os.path.exists(self.ingestion_config.raw_data_path):
                logger.info(f"Raw data already exists in {self.ingestion_config.raw_data_path}. Skipping download.")
            else:
                logger.info("Raw data not found. Downloading from Kaggle...")
                dataset_reference = self.ingestion_config.dataset_reference # The dataset name on Kaggle
                
                # Get the directory where the raw data should be saved
                raw_data_dir = os.path.dirname(self.ingestion_config.raw_data_path)

                # Create the directory
                os.makedirs(raw_data_dir, exist_ok=True)
                
                # Download and unzip the files into the 'data/01_Raw' directory
                kaggle.api.dataset_download_files(
                    dataset=dataset_reference,
                    path=raw_data_dir,
                    unzip=True
                )
                logger.info(f"Dataset successfully downloaded to: {raw_data_dir}")

        except Exception as e:
            raise CustomException(e, sys)
        # =================================================================

        try:
            # Now, read the downloaded CSV from the path defined in the config
            df = pd.read_csv(self.ingestion_config.raw_data_path)
            logger.info("Raw dataset read successfully.")
            
            # ... (rest of the logic for removing duplicates and splitting remains the same) ...

            df.drop_duplicates(inplace=True)
            
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logger.info("Data ingestion completed successfully.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)