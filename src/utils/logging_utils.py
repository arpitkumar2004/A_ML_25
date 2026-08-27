# src/utils/logging_utils.py
from typing import Optional
import logging
import os
import sys
import io

if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

if hasattr(sys.stderr, "buffer") and getattr(sys.stderr, "encoding", "").lower() != "utf-8":
    try:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass


class UnbufferedStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes stream after every log emit and safely handles unicode characters."""
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            try:
                stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                encoding = getattr(stream, "encoding", "utf-8") or "utf-8"
                safe_msg = msg.encode(encoding, errors="replace").decode(encoding, errors="replace")
                stream.write(safe_msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


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

        # Console handler with real-time flushing
        ch = UnbufferedStreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # File handler
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"), encoding="utf-8", errors="replace")
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
    fh = logging.FileHandler(log_file, encoding="utf-8", errors="replace")
    fh.setLevel(logging.INFO)

    # console handler with real-time flushing
    ch = UnbufferedStreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    logger.propagate = False
    return logger


