"""
Reasoning agent configuration.

This module defines the configuration schema for the reasoning agent.
"""

from dataclasses import dataclass

from src.config.manager import BaseThreadConfiguration


@dataclass(kw_only=True)
class ThreadConfiguration(BaseThreadConfiguration):
    """Reasoning agent configuration."""
