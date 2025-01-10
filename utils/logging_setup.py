import logging
from logging.handlers import RotatingFileHandler
import os
import datetime

def setup_logging(app_name="VDEOFCK"):
    # Set up a logs folder in the user's home directory
    logs_folder = os.path.join(os.path.expanduser("~"), f"{app_name}")
    os.makedirs(logs_folder, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define the log file path
    log_file = os.path.join(logs_folder, f"process_{current_time}.log")

    try:
        # Use a rotating file handler for process.log
        rotating_handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        handlers = [rotating_handler, logging.StreamHandler()]  # File and terminal
    except Exception as e:
        print(f"Failed to initialize file logging: {e}")
        handlers = [logging.StreamHandler()]  # Terminal fallback

    # Configure logging globally
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers
    )

    logging.info(f"Logging initialized. Logs will be written to {log_file}")

    return log_file
