"""
Logging Configuration for PolyHedgeRL

Professional logging setup with file rotation, console output, and structured formatting.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logger(
    name: str = "polyhedgerl",
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_file: Optional[str] = None,
    console: bool = True,
    file_logging: bool = True
) -> logging.Logger:
    """
    Set up a professional logger with console and file handlers.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: results/logs)
        log_file: Name of log file (default: polyhedgerl.log)
        console: Enable console logging
        file_logging: Enable file logging
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logger("training", level="DEBUG")
        >>> logger.info("Training started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers = []  # Clear existing handlers
    
    # Console handler with colors
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if file_logging:
        if log_dir is None:
            log_dir = Path("results/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if log_file is None:
            log_file = f"{name}.log"
        
        file_path = log_dir / log_file
        
        # Rotating file handler (10MB per file, keep 5 backups)
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    logger.propagate = False
    return logger


def get_logger(name: str = "polyhedgerl") -> logging.Logger:
    """
    Get or create a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        setup_logger(name)
    return logger


# Create default logger
logger = setup_logger()
