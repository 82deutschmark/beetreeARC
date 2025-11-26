import logging
import sys

def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Configures the root logger.
    
    - INFO level goes to stdout (standard application output).
    - ERROR/WARNING goes to stderr.
    - DEBUG level goes to stderr if verbose is True.
    """
    logger = logging.getLogger("arc_agi")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()

    # Handler for standard info/results (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    # Simple format for CLI output, or just message for cleanliness if we are printing tables manually
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    
    # Handler for debug/errors (stderr)
    # We filter this so it only takes ERROR (always) and DEBUG (if verbose)
    # But standard StreamHandler(sys.stderr) takes everything >= level.
    
    # If verbose, we want detailed logs to stderr
    if verbose:
        debug_handler = logging.StreamHandler(sys.stderr)
        debug_handler.setLevel(logging.DEBUG)
        debug_formatter = logging.Formatter("[%(levelname)s] %(message)s")
        debug_handler.setFormatter(debug_formatter)
        logger.addHandler(debug_handler)
    
    # We generally don't attach the console_handler to the root logger if we are 
    # manually printing tables with print(). 
    # However, for this refactor, we will use logger.info() for standard messages.
    # But carefully: print_table_header/row in reporting.py uses print() for strict formatting.
    # We should probably leave "presentation" prints as prints, and usage logs as logs.
    
    return logger

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"arc_agi.{name}")
