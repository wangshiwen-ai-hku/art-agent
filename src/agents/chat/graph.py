"""
Base chat agent implementation that provides common chat functionality
for agents that need interactive conversation capabilities.
"""

import logging
from abc import abstractmethod
from typing import Optional
from langchain_core.runnables import RunnableConfig

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.prebuilt import create_react_agent

from ..base import BaseAgent, BaseState
from .prompt import CHAT_SYSTEM_PROMPT
from .config import ThreadConfiguration
from src.config.manager import AgentConfig
from src.utils.input_processor import create_multimodal_message

logger = logging.getLogger(__name__)


class BaseChatAgent(BaseAgent):
    """
    Base chat agent that provides common interactive conversation functionality.
    This class can be inherited by specific agents like tutor, solver, etc.
    """

    name = "base_chat_agent"
    description = "Base chat agent that provides interactive conversation capabilities."

    def __init__(self, agent_config: AgentConfig):
        """Initialize the chat agent.

        Args:
            agent_config: Configuration object containing model parameters and tools.
        """
        super().__init__(agent_config)
        logger.info(f"{self.name} chat agent initialized successfully")

    def _load_system_prompt(self):
        """Load the system prompt for the chat agent."""
        self._system_prompt = CHAT_SYSTEM_PROMPT

    def build_graph(self) -> StateGraph:
        """Build the chat agent graph with common conversation flow.

        Returns:
            StateGraph: The constructed graph for the chat agent.
        """
        # Use the state schema defined by the concrete agent
        graph = StateGraph(state_schema=BaseState)

        # Add nodes
        graph.add_node("init_context_node", self.init_context_node)
        graph.add_node("chat_node", self.chat_node)
        graph.add_node("human_chat_node", self.human_chat_node)

        # Add edges
        graph.add_edge(START, "init_context_node")
        graph.add_edge("init_context_node", "chat_node")
        graph.add_edge("chat_node", "human_chat_node")
        graph.add_edge("human_chat_node", "chat_node")

        return graph

    async def chat_node(
        self, state: BaseState, config: Optional[RunnableConfig] = None
    ) -> Command:
        """Handle chat conversation with tools support.

        Args:
            state: The current agent state.
            config: Optional configuration for the node.

        Returns:
            Command: Command to continue to human_chat_node.
        """
        # Create react agent with tools for interactive conversation
        if self._tools:
            react_agent = create_react_agent(
                name=self.name,
                model=self._llm,
                tools=self._tools,
            )

            # Prepare messages with system context
            system_message = SystemMessage(
                content=self._system_prompt + "\n" + self._context
            )
            messages = [system_message] + state["messages"]

            # Get response from react agent
            response = await react_agent.ainvoke({"messages": messages})
            last_message = response["messages"][-1]

            return Command(goto="human_chat_node", update={"messages": [last_message]})
        else:
            # Fallback to simple LLM without tools
            system_message = SystemMessage(
                content=self._system_prompt + "\n" + self._context
            )
            messages = [system_message] + state["messages"]

            response = await self._llm.ainvoke(messages)
            logger.info(f"{self.name} agent response: {response.content}")

            return Command(goto="human_chat_node", update={"messages": [response]})

    async def human_chat_node(
        self, state: BaseState, config: Optional[RunnableConfig] = None
    ) -> Command:
        """Handle human interaction with multimodal support.

        Args:
            state: The current agent state.
            config: Optional configuration for the node.

        Returns:
            Command: Command to continue chat conversation.
        """
        user_id = state.get("user_id")
        logger.info(f"Waiting for user response from user {user_id}")

        # Interrupt for user input
        feedback = interrupt("你的回答：")

        # Create multimodal message from feedback
        human_message = create_multimodal_message(**feedback)

        return Command(goto="chat_node", update={"messages": [human_message]})
