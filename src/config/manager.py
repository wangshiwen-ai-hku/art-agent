"""Unified configuration management"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, TypedDict, Literal
import uuid
from abc import ABC

from dataclasses import dataclass, field, fields
import yaml
from dotenv import load_dotenv

from langchain_core.runnables import RunnableConfig
from .model import ModelConfig, MCPServerConfig, AgentConfig, MultiAgentConfig


load_dotenv()


class ConfigManager:
    """Centralized AGENT configuration manager."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration manager."""
        self._config_path = (
            config_path or Path(__file__).parent.parent.parent / "config.yaml"
        )
        self._config = self._load_config()
        self._is_production = os.getenv("PRODUCTION", "false").lower() == "true"

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self._config_path, "r") as f:
            config_content = f.read()

        # Expand environment variables in the config content
        config_content = self._expand_env_vars(config_content)

        config = yaml.safe_load(config_content)
        return config

    def _expand_env_vars(self, content: str) -> str:
        """Expand environment variables in configuration content."""
        import re

        def replace_var(match):
            var_name = match.group(1)
            return os.getenv(var_name, "")

        # Replace ${VAR_NAME} with environment variable values
        return re.sub(r"\$\{([^}]+)\}", replace_var, content)

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self._is_production

    def get_model_config(
        self, override: Optional[Dict[str, Any]] = None
    ) -> ModelConfig:
        """Get model configuration."""
        config = self._config.get("default_model", {})
        if override:
            config.update(override)
        return ModelConfig(**config)

    def get_mcp_server_config(self, server_name: str) -> Optional[MCPServerConfig]:
        """Get MCP server configuration by name."""
        servers = self._config.get("mcp_servers", {})
        if server_name not in servers:
            return None
        return MCPServerConfig(**servers[server_name])

    def get_all_mcp_servers(self) -> Dict[str, MCPServerConfig]:
        """Get all MCP server configurations."""
        servers = self._config.get("mcp_servers", {})
        return {name: MCPServerConfig(**config) for name, config in servers.items()}

    def get_agent_config(self, agent_type: str, category: str = "core") -> AgentConfig:
        """Get agent configuration by type and category."""
        # Navigate nested structure: agents -> category -> agent_type
        agents = self._config.get("agents", {})
        category_agents = agents.get(category, {})
        agent_config = category_agents.get(agent_type, {})

        if not agent_config:
            raise ValueError(f"Agent '{agent_type}' not found in category '{category}'")

        # Get model config (use default if not specified)
        model_config = self.get_model_config(agent_config.get("model"))

        tool_names = agent_config.get("tools", [])

        return AgentConfig(
            agent_type=agent_type,
            agent_category=category,
            agent_name=f"{category}_{agent_type}_agent",
            model=model_config,
            tools=tool_names,
        )

    def get_multi_agent_config(
        self, agent_type: str, category: str = "core"
    ) -> MultiAgentConfig:
        """Get multi-agent configuration by type and category."""
        # Navigate nested structure: agents -> category -> agent_type
        agents = self._config.get("agents", {})
        category_agents = agents.get(category, {})
        agent_config = category_agents.get(agent_type, {})

        if not agent_config:
            raise ValueError(f"Agent '{agent_type}' not found in category '{category}'")

        # Check if it's a multi-agent configuration
        if "agents" not in agent_config:
            raise ValueError(
                f"Agent '{agent_type}' is not configured as a multi-agent system"
            )

        # Get model config (use default if not specified)
        default_model = self.get_model_config(agent_config.get("model"))

        # Create AgentConfig for each sub-agent
        sub_agents = []
        for sub_agent_config in agent_config["agents"]:
            agent_name = sub_agent_config["agent_name"]

            # Use sub-agent specific model or default
            sub_agent_model = self.get_model_config(
                sub_agent_config.get("model", default_model.__dict__)
            )

            # Get tools for this sub-agent
            tool_names = sub_agent_config.get("tools", [])

            sub_agent = AgentConfig(
                agent_name=agent_name,
                model=sub_agent_model,
                tools=tool_names,
            )
            sub_agents.append(sub_agent)

        # Extract agent-specific configuration if available
        agent_specific_config = agent_config.get("config", {})

        return MultiAgentConfig(
            agents=sub_agents, agent_specific_config=agent_specific_config
        )

    def get_mem0_config(self) -> Dict[str, Any]:
        """Get mem0-compatible configuration from mem.py."""
        from .mem import MEM_CONFIG

        return MEM_CONFIG


@dataclass(kw_only=True)
class BaseThreadConfiguration(ABC):
    """Base THREAD configuration."""

    user_id: str = ""
    user_name: str = ""

    # Random thread id if not specified
    thread_id: str = str(uuid.uuid4())

    # Common configuration fields that all threads might need
    recursion_limit: int = 20  # Default recursion limit for LangGraph execution

    @classmethod
    def from_runnable_config(
        cls, config_obj: Optional[RunnableConfig] = None
    ) -> "BaseThreadConfiguration":
        """Create a Configuration instance from a RunnableConfig.

        Args:
            config_obj: Optional RunnableConfig containing configurable parameters

        Returns:
            Configuration instance with values from config_obj or defaults
        """
        configurable = (
            config_obj["configurable"]
            if config_obj and "configurable" in config_obj
            else {}
        )

        # Get agent type and category for centralized config lookup
        # These should be provided in the configurable section when needed
        agent_type = configurable.get("agent_type", "default")
        agent_category = configurable.get("agent_category", "core")

        # Get default values from centralized configuration
        try:
            agent_specific_config = config.get_agent_specific_config(
                agent_type, agent_category
            )
        except (ValueError, AttributeError):
            # If agent-specific config is not found, use empty dict
            agent_specific_config = {}

        values: dict[str, Any] = {}

        for f in fields(cls):
            if f.init:
                # Priority order: environment variables > configurable > centralized config > defaults

                # Check environment variable first (uppercase field name)
                env_value = os.environ.get(f.name.upper())
                if env_value is not None:
                    # Convert string values to appropriate types
                    if f.type == bool:
                        values[f.name] = env_value.lower() in ("true", "1", "yes", "on")
                    elif f.type == int:
                        values[f.name] = int(env_value)
                    else:
                        values[f.name] = env_value
                # Then check configurable values
                elif f.name in configurable:
                    values[f.name] = configurable[f.name]
                # Then check centralized config
                elif f.name in agent_specific_config:
                    values[f.name] = agent_specific_config[f.name]

        return cls(**{k: v for k, v in values.items() if v is not None})

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of all configuration fields
        """
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init}


# Global configuration instance
config = ConfigManager()
