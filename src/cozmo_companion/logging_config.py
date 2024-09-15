# logging_config.py

import logging
import os


# Function to set up logging configuration
def setup_logging(log_file="logs/chatbot_log.txt"):
    """
    Set up logging configuration.

    Args:
    log_file (str): The file path to log to. Defaults to 'chatbot_log.txt'.
    """
    # Ensure the log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set up basic logging configuration
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,  # Adjust the level depending on verbosity needed (INFO, DEBUG, etc.)
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Create a logger for Marvin or third-party libraries
    marvin_logger = logging.getLogger("marvin")
    marvin_logger.setLevel(logging.INFO)

    # Ensure Marvin or other third-party logs are captured
    root_logger = logging.getLogger()  # This is the root logger
    root_logger.setLevel(logging.INFO)

    logging.info("Logging setup complete.")
