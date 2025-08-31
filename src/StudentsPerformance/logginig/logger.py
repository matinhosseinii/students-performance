import logging
import sys
from logging.handlers import RotatingFileHandler

# Define the format for the log messages
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Function to set up a configured logger.
def setup_logger(name="student-performance", log_file="logs\\students-performance.log", level=logging.INFO):
    
    
    logger = logging.getLogger(name)
    
    # 🛡️ Prevents adding handlers multiple times
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)

    # ⚙️ Handler for rotating log files (e.g., max 5MB, keep 5 old files)
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)
    
    # Handler for console output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger