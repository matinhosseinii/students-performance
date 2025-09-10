import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from StudentsPerformance.exception.custom_exception import CustomException
from StudentsPerformance.logginig.logger import setup_logger
from StudentsPerformance.utils.common import save_object

logger = setup_logger()

@dataclass
class ModelTrainerConfig:
    """
    Configuration class for the Model Trainer component.
    Defines the path where the final trained model will be saved.
    """
    trained_model_file_path: str = os.path.join("models", "ridge_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        """
        This method orchestrates the final model training process.
        It trains the best model with the best hyperparameters found during
        experimentation and saves the resulting model object.
        
        Args:
            train_array (np.array): Processed training data (features and target).
            test_array (np.array): Processed test data (features and target).
        """
        try:
            logger.info("Splitting training and testing data into features and target.")
            # The last column is the target, the rest are features
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            logger.info("Initializing the best model with optimal hyperparameters.")
            # Based on the notebook, Ridge was the best model.
            # We initialize it directly with the best parameters found by GridSearchCV.
            best_model = Ridge(alpha=1.0, solver='sparse_cg')

            logger.info("Training the model...")
            best_model.fit(X_train, y_train)
            logger.info("Model training completed.")

            logger.info("Evaluating the model on the test set.")
            y_pred = best_model.predict(X_test)
            r2_square = r2_score(y_test, y_pred)
            logger.info(f"Final R2 score on the test data: {r2_square:.4f}")

            logger.info("Saving the trained model object.")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
