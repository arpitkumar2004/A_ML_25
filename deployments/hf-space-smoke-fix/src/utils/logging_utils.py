# src/utils/logging_utils.py
from typing import Optional
import logging
import os
import sys


class LoggerFactory:
    """
    Create and configure loggers with file and console handlers.
    Usage:
        logger = LoggerFactory.get('train', log_dir='experiments/logs')
        logger.info("message")
    """
    @staticmethod
    def get(name: str, log_dir: Optional[str] = "experiments/logs", level: int = logging.INFO):
        logger = logging.getLogger(name)
        if logger.handlers:
            return logger  # already configured

        logger.setLevel(level)
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Console handler
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # File handler
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
            fh.setLevel(level)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

        # Prevent propagation to root logger twice
        logger.propagate = False
        return logger

def get_logger(name: str, log_dir: str = "experiments/logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
