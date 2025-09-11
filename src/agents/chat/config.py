"""
Chat agent configuration.

This module defines the configuration schema for the chat agent.
"""

from dataclasses import dataclass

from src.config.manager import BaseThreadConfiguration


@dataclass(kw_only=True)
class ThreadConfiguration(BaseThreadConfiguration):
    """Chat agent configuration."""
