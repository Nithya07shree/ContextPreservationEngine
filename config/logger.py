# src/logger.py
"""
Centralized logger setup
"""

import logging
import sys
from typing import Optional

def get_logger(
    name: str, 
    level: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Returns a logger instance
    
    Args:
        name: Module name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR). Defaults to INFO
        log_file: Optional file path to write logs to
        
    Returns:
        Configured logger instance
    """
    
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    
    # Set level
    log_level = getattr(logging, level.upper(), logging.INFO) if level else logging.INFO
    logger.setLevel(log_level)
    
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not create file handler: {e}")
    
    logger.propagate = False
    
    return logger


def set_level(logger: logging.Logger, level: str):
    """
    Change logger level at runtime
    
    Args:
        logger: Logger instance
        level: New level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    for handler in logger.handlers:
        handler.setLevel(log_level)