from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class ModelConfig:
    """Model configuration."""

    model_provider: str = "openai"
    model: str = "gpt-4.1-mini"
    temperature: float = 0
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class MCPServerConfig:
    """MCP server configuration."""

    transport: str
    url: Optional[str] = None
    command: Optional[str] = None
    args: Optional[List[str]] = None
    enabled_tools: List[str] = field(default_factory=list)


@dataclass
class AgentConfig:
    """Agent configuration."""

    model: ModelConfig
    agent_name: Optional[str] = None
    agent_type: Optional[str] = None
    agent_category: Optional[str] = None
    tools: List[str] = field(default_factory=list)


@dataclass
class MultiAgentConfig:
    """Multi-agent configuration."""

    agents: List[AgentConfig]
    agent_specific_config: Optional[Dict[str, Any]] = None
