import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

# Assuming these are in your project structure as discussed
from StudentsPerformance.exception.custom_exception import CustomException
from StudentsPerformance.logginig.logger import setup_logger
from StudentsPerformance.utils.common import save_object 

logger = setup_logger()

def custom_data_cleaner(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles custom, rule-based cleaning. Per the data preprocessing notebook, no specific
    typo cleaning or date formatting is needed for this dataset, so it acts as a placeholder.
    """
    logger.info("Applying custom data cleaning rules (no custom rules for now).")
    return df

def remove_boundary_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes invalid outliers based on domain knowledge (scores outside [0, 100]).
    """
    logger.info("Checking for boundary outliers in score columns...")
    
    # Using a copy is a good habit to avoid unexpected side effects
    df_out = df.copy() 
    
    indices_to_drop = set()
    
    # Define the columns you want to check
    score_columns = ['reading score', 'writing score']
    
    for column in score_columns:
        # Check if the column exists BEFORE you try to use it
        if column in df_out.columns:
            # Find the indices of invalid rows
            outlier_indices = df_out[(df_out[column] < 0) | (df_out[column] > 100)].index
            
            if not outlier_indices.empty:
                logger.info(f"Found {len(outlier_indices)} outliers in column '{column}'.")
                indices_to_drop.update(outlier_indices)
            else:
                logger.info(f"No boundary outliers found in column '{column}'.")
        else:
            # Log a warning that a required column was not found
            logger.warning(f"Required column '{column}' not found in DataFrame. Skipping.")
    
    # Perform a single drop operation after the loop if any outliers were found
    if len(indices_to_drop) > 0:
        logger.info(f"Removing a total of {len(indices_to_drop)} unique rows with boundary outliers.")
        df_out = df_out.drop(index=list(indices_to_drop))
    else:
        logger.info(f"No boundary outliers found in the dataset.")

    return df_out

@dataclass
class DataPreprocessingConfig:
    """
    Configuration class for the Data Preprocessing component.
    """
    train_path = os.path.join('data', '02_Intermediate', 'Train.csv')
    test_path = os.path.join('data', '02_Intermediate', 'Test.csv')
    preprocessor_obj_file_path: str = os.path.join('models', 'preprocessor.pkl')

class DataPreprocessor:
    """
    This class handles the entire data preprocessing workflow.
    """
    def __init__(self):
        self.preprocessing_config = DataPreprocessingConfig()

    def get_preprocessor_pipeline(self, numerical_features: list, categorical_features: list) -> Pipeline:
        """
        Builds and returns a scikit-learn pipeline that handles all
        data transformation steps in the correct order based on EDA findings.
        """
        try:
            logger.info("Building the full preprocessing pipeline.")

            # Pipeline for numerical features based on EDA
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')), # <-- UPDATED based on EDA
                ('scaler', StandardScaler())
            ])

            # Pipeline for categorical features
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse_output=False)) # <-- UPDATED
            ])

            # Combine pipelines
            data_transformer = ColumnTransformer(
                transformers=[
                    ('num_pipeline', numerical_pipeline, numerical_features),
                    ('cat_pipeline', categorical_pipeline, categorical_features)
                ]
            )

            # Create the master pipeline
            full_preprocessing_pipeline = Pipeline(steps=[
                ('custom_cleaner', FunctionTransformer(custom_data_cleaner)),
                ('outlier_handler', FunctionTransformer(remove_boundary_outliers)),
                ('data_transformer', data_transformer)
            ])
            
            logger.info("Preprocessing pipeline created successfully.")
            return full_preprocessing_pipeline

        except Exception as e:
            raise CustomException(e, sys)

    def run_preprocessing(self):
        """
        Orchestrates the data preprocessing process.
        """
        try:
            train_df = pd.read_csv(self.preprocessing_config.train_path)
            test_df = pd.read_csv(self.preprocessing_config.test_path)
            logger.info("Read train and test data completed.")

            # Define features and target based on the notebook
            target_column = "math score"
            numerical_features = ['reading score', 'writing score']
            categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
            
            preprocessor_pipeline = self.get_preprocessor_pipeline(
                numerical_features=numerical_features,
                categorical_features=categorical_features
            )
            
            # Separate features and target
            X_train = train_df.drop(columns=[target_column], axis=1)
            y_train = train_df[target_column]
            
            X_test = test_df.drop(columns=[target_column], axis=1)
            y_test = test_df[target_column]
            
            logger.info("Applying preprocessing pipeline on training and testing dataframes.")

            # Fit the pipeline on training data and transform it
            X_train_processed = preprocessor_pipeline.fit_transform(X_train)
            # Transform the test data
            X_test_processed = preprocessor_pipeline.transform(X_test)
            
            # Combine features and target back into single arrays
            train_arr = np.c_[X_train_processed, np.array(y_train)]
            test_arr = np.c_[X_test_processed, np.array(y_test)]
            
            logger.info("Saving preprocessing object.")
            
            save_object(
                file_path=self.preprocessing_config.preprocessor_obj_file_path,
                obj=preprocessor_pipeline
            )

            logger.info(f"Preprocessing object is saved in {self.preprocessing_config.preprocessor_obj_file_path}")

            return (
                train_arr,
                test_arr,
                self.preprocessing_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

