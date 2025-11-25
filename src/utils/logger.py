import logging
import functools
import time
from typing import Optional, Callable, Any
from pathlib import Path
import pandas as pd

def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """
    Set up a logger with specified name and configuration
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter('%(levelname)s - %(message)s')
    
    # Always add console handler with simple format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log_file specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_data_stats(logger: logging.Logger, data: Any) -> None:
    """Log detailed statistics about data"""
    if isinstance(data, pd.DataFrame):
        # Log basic info
        logger.info(f"DataFrame shape: {data.shape}")
        
        # Log missing value statistics
        missing_stats = data.isnull().sum()
        if missing_stats.any():
            logger.info("Missing value statistics:")
            for col, count in missing_stats[missing_stats > 0].items():
                logger.info(f"  {col}: {count} missing values")
        
        # Log data types
        logger.info("Column data types:")
        for col, dtype in data.dtypes.items():
            logger.info(f"  {col}: {dtype}")

def log_data_operation(description: str):
    """Decorator to log data operations with detailed statistics"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = args[0]._logger if hasattr(args[0], '_logger') else setup_logger(__name__)
            
            start_time = time.time()
            logger.info(f"Starting {description}")
            
            try:
                result = func(*args, **kwargs)
                
                # Log execution time
                duration = time.time() - start_time
                logger.info(f"Completed {description} in {duration:.2f}s")
                
                # Log detailed statistics about the result
                if isinstance(result, (pd.DataFrame, pd.Series)):
                    log_data_stats(logger, result)
                elif isinstance(result, tuple) and any(isinstance(x, (pd.DataFrame, pd.Series)) for x in result):
                    for i, item in enumerate(result):
                        if isinstance(item, (pd.DataFrame, pd.Series)):
                            logger.info(f"Output {i+1} statistics:")
                            log_data_stats(logger, item)
                
                return result
                
            except Exception as e:
                logger.error(f"Error in {description}: {str(e)}")
                raise
                
        return wrapper
    return decorator

def log_step(step_name: str):
    """
    Decorator for logging workflow steps
    
    Args:
        step_name: Name of the workflow step
        
    Example:
        @log_step("Data Processing")
        def process_data(self):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get logger from class instance (self)
            logger = args[0]._logger if args else logging.getLogger(func.__module__)
            
            logger.info(f"=== Starting {step_name} ===")
            
            try:
                result = func(*args, **kwargs)
                logger.info(f"=== Completed {step_name} ===")
                return result
                
            except Exception as e:
                logger.error(f"=== Failed {step_name}: {str(e)} ===")
                raise
                
        return wrapper
    return decorator

def log_info(logger: logging.Logger, message: str, data: Optional[Any] = None):
    """
    Log information about data state
    
    Args:
        logger: Logger instance
        message: Base message
        data: Optional data to analyze and include in log
        
    Example:
        log_info(logger, "Found chemical species columns", species_data)
    """
    if data is None:
        logger.info(message)
        return
        
    # Add data-specific information
    if hasattr(data, 'columns'):  # DataFrame
        logger.info(f"{message}: {len(data.columns)} columns")
    elif hasattr(data, 'shape'):  # NumPy array
        logger.info(f"{message}: shape {data.shape}")
    elif isinstance(data, (list, tuple, set)):
        logger.info(f"{message}: {len(data)} items")
    else:
        logger.info(f"{message}: {data}")