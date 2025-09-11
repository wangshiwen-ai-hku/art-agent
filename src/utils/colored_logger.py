"""
Colored logging utilities for enhanced visual debugging and monitoring.

This module provides a comprehensive colored logging system designed for
multi-agent systems and complex workflows. It includes color-coded output
for different types of operations, making it easier to track system behavior
and debug issues.

Features:
- Color-coded log messages for different component types
- Flow transition tracking
- State change monitoring  
- Agent operation logging
- Tool usage tracking
- Error and warning highlighting
"""

import logging
from typing import Optional


# ANSI Color codes for terminal output
class Colors:
    """ANSI color codes for terminal output styling."""

    # Primary colors for different component types
    BLUE = "\033[94m"  # Router operations
    GREEN = "\033[92m"  # Success/tool calls
    YELLOW = "\033[93m"  # Warnings/important info
    RED = "\033[91m"  # Errors
    PURPLE = "\033[95m"  # Specialized agents
    CYAN = "\033[96m"  # State changes/flow
    WHITE = "\033[97m"  # General info

    # Text formatting
    BOLD = "\033[1m"  # Bold text
    UNDERLINE = "\033[4m"  # Underlined text

    # Reset
    END = "\033[0m"  # Reset color


class ColoredLogger:
    """
    A colored logging utility for multi-agent systems.

    This class provides various logging methods with color-coded output
    to make it easier to visually distinguish between different types
    of operations and components in complex multi-agent workflows.
    """

    def __init__(self, logger_name: str):
        """
        Initialize the colored logger.

        Args:
            logger_name: Name of the logger to use (typically __name__)
        """
        self.logger = logging.getLogger(logger_name)

    def log_router(self, message: str, level: str = "info") -> None:
        """
        Log router operations in blue.

        Args:
            message: The message to log
            level: Logging level (info, debug, warning, error)
        """
        colored_msg = f"{Colors.BLUE}{Colors.BOLD}[ROUTER]{Colors.END} {Colors.BLUE}{message}{Colors.END}"
        getattr(self.logger, level)(colored_msg)

    def log_agent(self, agent_name: str, message: str, level: str = "info") -> None:
        """
        Log specialized agent operations in purple.

        Args:
            agent_name: Name of the agent performing the operation
            message: The message to log
            level: Logging level (info, debug, warning, error)
        """
        colored_msg = f"{Colors.PURPLE}{Colors.BOLD}[{agent_name.upper()}]{Colors.END} {Colors.PURPLE}{message}{Colors.END}"
        getattr(self.logger, level)(colored_msg)

    def log_tool(self, tool_name: str, message: str, level: str = "info") -> None:
        """
        Log tool usage in green.

        Args:
            tool_name: Name of the tool being used
            message: The message to log
            level: Logging level (info, debug, warning, error)
        """
        colored_msg = f"{Colors.GREEN}{Colors.BOLD}[TOOL:{tool_name}]{Colors.END} {Colors.GREEN}{message}{Colors.END}"
        getattr(self.logger, level)(colored_msg)

    def log_flow(
        self, from_node: str, to_node: str, reason: str = "", level: str = "info"
    ) -> None:
        """
        Log flow transitions in cyan.

        Args:
            from_node: Source node name
            to_node: Destination node name
            reason: Optional reason for the transition
            level: Logging level (info, debug, warning, error)
        """
        arrow = f"{Colors.CYAN}→{Colors.END}"
        colored_msg = f"{Colors.CYAN}{Colors.BOLD}[FLOW]{Colors.END} {Colors.CYAN}{from_node} {arrow} {to_node}"
        if reason:
            colored_msg += f" ({reason})"
        colored_msg += f"{Colors.END}"
        getattr(self.logger, level)(colored_msg)

    def log_state(self, message: str, level: str = "info") -> None:
        """
        Log state changes in cyan.

        Args:
            message: The state change message to log
            level: Logging level (info, debug, warning, error)
        """
        colored_msg = f"{Colors.CYAN}{Colors.BOLD}[STATE]{Colors.END} {Colors.CYAN}{message}{Colors.END}"
        getattr(self.logger, level)(colored_msg)

    def log_warning(self, message: str) -> None:
        """
        Log warnings in yellow.

        Args:
            message: The warning message to log
        """
        colored_msg = f"{Colors.YELLOW}{Colors.BOLD}[WARNING]{Colors.END} {Colors.YELLOW}{message}{Colors.END}"
        self.logger.warning(colored_msg)

    def log_error(self, message: str) -> None:
        """
        Log errors in red.

        Args:
            message: The error message to log
        """
        colored_msg = f"{Colors.RED}{Colors.BOLD}[ERROR]{Colors.END} {Colors.RED}{message}{Colors.END}"
        self.logger.error(colored_msg)

    def log_success(self, message: str, level: str = "info") -> None:
        """
        Log success messages in green.

        Args:
            message: The success message to log
            level: Logging level (info, debug, warning, error)
        """
        colored_msg = f"{Colors.GREEN}{Colors.BOLD}[SUCCESS]{Colors.END} {Colors.GREEN}{message}{Colors.END}"
        getattr(self.logger, level)(colored_msg)

    def log_debug(self, message: str) -> None:
        """
        Log debug messages in white.

        Args:
            message: The debug message to log
        """
        colored_msg = f"{Colors.WHITE}[DEBUG] {message}{Colors.END}"
        self.logger.debug(colored_msg)


# Convenience functions for backward compatibility and ease of use
_default_logger: Optional[ColoredLogger] = None


def get_colored_logger(logger_name: str) -> ColoredLogger:
    """
    Get a colored logger instance.

    Args:
        logger_name: Name of the logger to create

    Returns:
        ColoredLogger instance
    """
    return ColoredLogger(logger_name)


def init_default_logger(logger_name: str) -> None:
    """
    Initialize the default logger for use with convenience functions.

    Args:
        logger_name: Name of the logger to initialize
    """
    global _default_logger
    _default_logger = ColoredLogger(logger_name)


def log_router(message: str, level: str = "info") -> None:
    """Convenience function for logging router operations."""
    if _default_logger is None:
        raise RuntimeError(
            "Default logger not initialized. Call init_default_logger() first."
        )
    _default_logger.log_router(message, level)


def log_agent(agent_name: str, message: str, level: str = "info") -> None:
    """Convenience function for logging agent operations."""
    if _default_logger is None:
        raise RuntimeError(
            "Default logger not initialized. Call init_default_logger() first."
        )
    _default_logger.log_agent(agent_name, message, level)


def log_tool(tool_name: str, message: str, level: str = "info") -> None:
    """Convenience function for logging tool usage."""
    if _default_logger is None:
        raise RuntimeError(
            "Default logger not initialized. Call init_default_logger() first."
        )
    _default_logger.log_tool(tool_name, message, level)


def log_flow(
    from_node: str, to_node: str, reason: str = "", level: str = "info"
) -> None:
    """Convenience function for logging flow transitions."""
    if _default_logger is None:
        raise RuntimeError(
            "Default logger not initialized. Call init_default_logger() first."
        )
    _default_logger.log_flow(from_node, to_node, reason, level)


def log_state(message: str, level: str = "info") -> None:
    """Convenience function for logging state changes."""
    if _default_logger is None:
        raise RuntimeError(
            "Default logger not initialized. Call init_default_logger() first."
        )
    _default_logger.log_state(message, level)


def log_warning(message: str) -> None:
    """Convenience function for logging warnings."""
    if _default_logger is None:
        raise RuntimeError(
            "Default logger not initialized. Call init_default_logger() first."
        )
    _default_logger.log_warning(message)


def log_error(message: str) -> None:
    """Convenience function for logging errors."""
    if _default_logger is None:
        raise RuntimeError(
            "Default logger not initialized. Call init_default_logger() first."
        )
    _default_logger.log_error(message)


def log_success(message: str, level: str = "info") -> None:
    """Convenience function for logging success messages."""
    if _default_logger is None:
        raise RuntimeError(
            "Default logger not initialized. Call init_default_logger() first."
        )
    _default_logger.log_success(message, level)


def log_debug(message: str) -> None:
    """Convenience function for logging debug messages."""
    if _default_logger is None:
        raise RuntimeError(
            "Default logger not initialized. Call init_default_logger() first."
        )
    _default_logger.log_debug(message)
