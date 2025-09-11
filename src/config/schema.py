"""Configuration schema for LangGraph agents.

This module defines the configuration schemas used across all agents,
following LangGraph best practices for handling configurable parameters.
"""

from typing import Optional, Dict, Any
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig


class BaseConfigSchema(TypedDict, total=False):
    """Base configuration schema for all agents."""

    # User context - moved from state to config
    user_id: str
    user_name: Optional[str]

    # Agent behavior configuration
    debug: bool
    locale: str

    # Timeout settings
    default_timeout_seconds: int


class DeepResearcherConfigSchema(BaseConfigSchema, total=False):
    """Configuration schema for deep researcher agents."""

    # Plan and execution configuration
    max_step_num: int
    max_search_results: int
    auto_accepted_plan: bool


class StudentConfigSchema(BaseConfigSchema, total=False):
    """Configuration schema for student domain agents."""

    # Student-specific configuration
    grade_level: Optional[str]
    subject: Optional[str]
    learning_style: Optional[str]


class CoordinatorConfigSchema(BaseConfigSchema, total=False):
    """Configuration schema for coordinator agents."""

    # Coordinator-specific configuration
    max_handoff_depth: int
    enable_sub_agents: bool


def extract_config_from_runnable_config(
    config: Optional[RunnableConfig] = None, config_class: type = BaseConfigSchema
) -> Dict[str, Any]:
    """Extract configuration values from RunnableConfig.

    Args:
        config: The RunnableConfig object containing configurable values
        config_class: The configuration schema class to use for extraction

    Returns:
        Dictionary containing the extracted configuration values
    """
    if not config or "configurable" not in config:
        return {}

    configurable = config["configurable"]
    extracted = {}

    # Get the TypedDict annotations to know which fields are expected
    if hasattr(config_class, "__annotations__"):
        for field_name, field_type in config_class.__annotations__.items():
            if field_name in configurable:
                extracted[field_name] = configurable[field_name]

    return extracted


def create_default_config(
    user_id: str, user_name: Optional[str] = None, **kwargs
) -> Dict[str, Any]:
    """Create a default configuration dictionary.

    Args:
        user_id: The user identifier
        user_name: Optional user name
        **kwargs: Additional configuration parameters

    Returns:
        Configuration dictionary suitable for RunnableConfig.configurable
    """
    config = {
        "user_id": user_id,
        "user_name": user_name,
        "debug": False,
        "locale": "zh-CN",
        "default_timeout_seconds": 600,
        **kwargs,
    }

    # Remove None values
    return {k: v for k, v in config.items() if v is not None}
