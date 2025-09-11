import logging
from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import asdict
from datetime import datetime

from langchain.chat_models import init_chat_model
from langchain_core.tools import Tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.config.manager import MultiAgentConfig, ModelConfig
from src.infra.tools import get_tools


from .schema import BaseState

logger = logging.getLogger(__name__)


class MultiAgentBase(ABC):
    """
    Base class for multi-agent systems that provides multi-agent registry for prompts and tools
    """

    name = "multi_agent_base"
    description = "Base multi-agent system that coordinates multiple specialized agents"

    def __init__(self, multi_agent_config: MultiAgentConfig):
        """Initialize the multi-agent system."""
        self._multi_agent_config = multi_agent_config
        self._agent_tools_registry = {}
        self._agent_llm_registry = {}
        for agent_config in self._multi_agent_config.agents:
            self.register_agent_llm(agent_config.agent_name, agent_config.model)
            self.register_agent_tools(agent_config.agent_name, agent_config.tools)

        self._graph = self.build_graph()

        self.compiled_graph = self.compile_graph()  # entry_point

    def register_agent_llm(self, agent_name: str, llm: ModelConfig):
        """Register agent LLM."""
        _llm = init_chat_model(**asdict(llm))
        self._agent_llm_registry[agent_name] = _llm

    def register_agent_tools(self, agent_name: str, tools: List[Tool]):
        """Register agent tools."""
        _tools = get_tools(tools)
        self._agent_tools_registry[agent_name] = _tools

    @abstractmethod
    def build_graph(self) -> StateGraph:
        """Build the agent graph.

        Returns:
            StateGraph: The constructed graph for the agent.
        """
        pass

    def compile_graph(self) -> CompiledStateGraph:
        """Compile the agent graph with memory checkpointer.

        Returns:
            The compiled graph ready for execution.
        """
        memory = MemorySaver()
        compiled_graph = self._graph.compile(checkpointer=memory)
        logger.info("Agent graph built and compiled successfully")
        return compiled_graph
