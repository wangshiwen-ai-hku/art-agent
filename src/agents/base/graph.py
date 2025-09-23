import logging
from abc import ABC, abstractmethod
from dataclasses import asdict


from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from src.config.manager import AgentConfig, BaseThreadConfiguration
from src.infra.tools import get_tools

from .prompt import BASE_SYSTEM_PROMPT
from .schema import BaseState

from typing import Optional
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig
from datetime import datetime

logger = logging.getLogger(__name__)

# set PYTHONUTF8=1
class BaseAgent(ABC):
    """Base class for LLM-based agents."""

    name = "base_agent"
    description = "Base agent that can be used to build other agents."

    def __init__(self, agent_config: AgentConfig):
        """Initialize the agent.

        Args:
            agent_config: Configuration object containing model parameters and other settings.
        """
        self._agent_config = agent_config
        self._agent_type = agent_config.agent_type
        self._agent_category = agent_config.agent_category

        self._model_config = agent_config.model
        self._tool_names: list[str] = agent_config.tools

        # Load tools
        self._tools = get_tools(self._tool_names)
        # Load system prompt
        self._load_system_prompt()
        # Init context
        self._context = None

        logger.info(f"Loaded tools: {[tool.name for tool in self._tools]}")
        # logger.info(f"Loaded system prompt: {self._system_prompt}")
        self._llm = self._init_llm()
        
        if self._tools:
            self._llm = self._llm.bind_tools(self._tools)
            
        self._graph = self.build_graph()
        self.compiled_graph = self.compile_graph()
        
    # @abstractmethod
    def _init_llm(self):
        return init_chat_model(**asdict(self._model_config))
        
    @abstractmethod
    def _load_system_prompt(self):
        """Load the system prompt for the agent."""
        self._system_prompt = BASE_SYSTEM_PROMPT

    def init_context_node(
        self, state: BaseState, config: Optional[RunnableConfig] = None
    ):
        """Initialize the context for the agent."""
        configuration = BaseThreadConfiguration.from_runnable_config(config)
        self._context = "\n".join(
            [
                "当前时间是：" + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "当前用户是：" + configuration.user_name,
            ]
        )

    @abstractmethod
    def build_graph(self) -> StateGraph:
        """Build the agent graph.

        Returns:
            StateGraph: The constructed graph for the agent.
        """
        graph = StateGraph(state_schema=BaseState)
        graph.add_node("dummy_node", self.dummy_node)
        graph.add_edge(START, "dummy_node")
        return graph

    def compile_graph(self) -> CompiledStateGraph:
        """Compile the agent graph with memory checkpointer."""
        # This is now optional and can be used for non-service-based testing
        memory = MemorySaver()
        compiled_graph = self._graph.compile(checkpointer=memory)
        logger.info("Agent graph built and compiled successfully with MemorySaver")
        return compiled_graph

    async def dummy_node(
        self, state: BaseState, config: Optional[RunnableConfig] = None
    ):
        """Mock node for base implementation.

        Args:
            state: The current state of the agent.
            config: Optional configuration for the node.

        Returns:
            dict: Updated state with a mock message.
        """
        messages = [
            SystemMessage(content=self._system_prompt + "\n" + (self._context or "")),
            HumanMessage(content=state.get("task", "No task specified")),
        ]
        response = await self._llm.ainvoke(messages)
        return {"messages": [response]}
