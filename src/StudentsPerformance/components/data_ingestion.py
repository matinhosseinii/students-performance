import kaggle
from StudentsPerformance.logginig.logger import setup_logger

logger = setup_logger()

# This function correctly uses the 'path' argument as a destination folder
def download_kaggle_dataset(dataset_reference, path_to_save):
    kaggle.api.dataset_download_files(
        dataset = dataset_reference,
        path = path_to_save,
        unzip = True  # Automatically unzips the files for you
    )

    logger.info(f"Dataset successfully downloaded and unzipped to: src\\data\\01_Raw")