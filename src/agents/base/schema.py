from typing import Optional
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages


class BaseState(TypedDict):
    """Base state schema for agents."""

    messages: Annotated[list, add_messages]  # Messages in the graph
