"""
Utils package for common utilities and helpers.

This package contains various utility modules for the art-agents project,
including colored logging, fuzzy matching, input processing, and other helpers.
"""

# Import commonly used utilities for easier access
from .colored_logger import (
    ColoredLogger,
    get_colored_logger,
    init_default_logger,
    log_router,
    log_agent,
    log_tool,
    log_flow,
    log_state,
    log_warning,
    log_error,
    log_success,
    log_debug,
)

from .fuzzy_match import (
    find_best_match,
    find_best_match_from_dict,
    get_match_suggestions,
)

__all__ = [
    # Colored logging
    "ColoredLogger",
    "get_colored_logger",
    "init_default_logger",
    "log_router",
    "log_agent",
    "log_tool",
    "log_flow",
    "log_state",
    "log_warning",
    "log_error",
    "log_success",
    "log_debug",
    # Fuzzy matching
    "find_best_match",
    "find_best_match_from_dict",
    "get_match_suggestions",
]
