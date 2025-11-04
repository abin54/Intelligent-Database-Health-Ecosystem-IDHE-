"""
Logger Configuration for IDHE (Intelligent Database Health Ecosystem)
====================================================================

Advanced logging setup with structured logging, log rotation, and multiple outputs.
"""

import logging
import logging.handlers
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json
from pythonjsonlogger import jsonlogger

# Create logs directory
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

def setup_logging(
    level: str = "INFO",
    log_file: str = "idhe.log",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    enable_json_logging: bool = True,
    enable_console_output: bool = True
) -> logging.Logger:
    """
    Setup comprehensive logging for IDHE system
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file name
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
        enable_json_logging: Enable structured JSON logging
        enable_console_output: Enable console logging
    
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger("IDHE")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Formatter
    if enable_json_logging:
        # JSON formatter for structured logging
        json_formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s %(module)s %(funcName)s %(lineno)d',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        # Standard formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # File handler with rotation
    log_file_path = LOGS_DIR / log_file
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(json_formatter if enable_json_logging else formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    if enable_console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        
        # Simpler format for console
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Error file handler (only errors and critical)
    error_log_file = LOGS_DIR / f"error_{log_file}"
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(json_formatter if enable_json_logging else formatter)
    logger.addHandler(error_handler)
    
    # Performance log handler
    performance_log_file = LOGS_DIR / f"performance_{log_file}"
    performance_handler = logging.handlers.RotatingFileHandler(
        performance_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    performance_handler.setLevel(logging.INFO)
    
    # Performance formatter with custom fields
    perf_formatter = logging.Formatter(
        '%(asctime)s PERFORMANCE %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    performance_handler.setFormatter(perf_formatter)
    
    # Create performance logger
    perf_logger = logging.getLogger("IDHE.PERFORMANCE")
    perf_logger.setLevel(logging.INFO)
    perf_logger.addHandler(performance_handler)
    perf_logger.propagate = False  # Don't propagate to root logger
    
    # Security log handler
    security_log_file = LOGS_DIR / f"security_{log_file}"
    security_handler = logging.handlers.RotatingFileHandler(
        security_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    security_handler.setLevel(logging.INFO)
    
    # Security formatter
    security_formatter = logging.Formatter(
        '%(asctime)s SECURITY %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    security_handler.setFormatter(security_formatter)
    
    # Create security logger
    security_logger = logging.getLogger("IDHE.SECURITY")
    security_logger.setLevel(logging.INFO)
    security_logger.addHandler(security_handler)
    security_logger.propagate = False
    
    logger.info("IDHE logging system initialized", extra={
        'log_level': level,
        'log_file': str(log_file_path),
        'json_logging': enable_json_logging
    })
    
    return logger

def get_performance_logger() -> logging.Logger:
    """Get performance-specific logger"""
    return logging.getLogger("IDHE.PERFORMANCE")

def get_security_logger() -> logging.Logger:
    """Get security-specific logger"""
    return logging.getLogger("IDHE.SECURITY")

def log_function_performance(func):
    """Decorator to log function performance"""
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            
            # Log successful execution
            execution_time = time.time() - start_time
            get_performance_logger().info(
                f"Function {func.__name__} completed successfully",
                extra={
                    'function_name': func.__name__,
                    'execution_time': execution_time,
                    'status': 'success'
                }
            )
            
            return result
            
        except Exception as e:
            # Log failed execution
            execution_time = time.time() - start_time
            logger = logging.getLogger(func.__module__)
            logger.error(
                f"Function {func.__name__} failed with error: {str(e)}",
                extra={
                    'function_name': func.__name__,
                    'execution_time': execution_time,
                    'status': 'error',
                    'error_type': type(e).__name__
                },
                exc_info=True
            )
            raise
    
    return wrapper

def log_security_event(event_type: str, details: Dict[str, Any], level: str = "INFO"):
    """Log security-related events"""
    security_logger = get_security_logger()
    
    log_method = getattr(security_logger, level.lower(), security_logger.info)
    
    log_method(
        f"Security event: {event_type}",
        extra={
            'event_type': event_type,
            'security_event': True,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
    )

def setup_structured_logging_for_component(component_name: str) -> logging.Logger:
    """Setup logging for a specific component"""
    logger = logging.getLogger(f"IDHE.{component_name.upper()}")
    
    # Component-specific file handler
    component_log_file = LOGS_DIR / f"{component_name.lower()}.log"
    component_handler = logging.handlers.RotatingFileHandler(
        component_log_file,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    
    # JSON formatter for component logs
    json_formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(levelname)s %(message)s %(module)s %(funcName)s %(lineno)d',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    component_handler.setFormatter(json_formatter)
    logger.addHandler(component_handler)
    
    return logger

# Initialize default logger
logger = setup_logging()