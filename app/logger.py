import logging
import os.path

from app.definitions import LOG_DIR

logger = logging.getLogger("mini-rag")


def set_logger(log_file: str):
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(os.path.join(LOG_DIR, log_file))
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
