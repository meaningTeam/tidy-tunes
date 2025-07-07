import logging
from typing import Optional

# Global logger instance
_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    """
    Get the shared logger instance. Creates it if it doesn't exist.

    Returns:
        logging.Logger: The shared logger instance.
    """
    global _logger
    if _logger is None:
        _logger = setup_shared_logger()
    return _logger


def setup_shared_logger(
    name: str = "tidytunes", level: int = logging.INFO, log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup the shared logger that logs to both the console and an optional file.

    Args:
        name (str): Name of the logger (default: "tidytunes").
        level (int): Logging level (default: logging.INFO).
        log_file (str, optional): Path to a log file (default: None, meaning no file logging).

    Returns:
        logging.Logger: Configured logger instance.
    """
    global _logger

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _logger = logger
    return logger


def configure_logger(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """
    Configure the shared logger with specific settings.

    Args:
        log_file (str, optional): Path to a log file.
        level (int): Logging level.
    """
    setup_shared_logger(log_file=log_file, level=level)
