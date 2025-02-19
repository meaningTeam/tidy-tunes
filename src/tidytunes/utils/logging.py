import logging


def setup_logger(
    name: str, level: int = logging.INFO, log_file: str | None = None
) -> logging.Logger:
    """
    Setup a logger that logs to both the console and an optional file.

    Args:
        name (str): Name of the logger.
        level (int): Logging level (default: logging.INFO).
        log_file (str, optional): Path to a log file (default: None, meaning no file logging).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        return logger

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

    return logger
