from typing import Optional
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages, MessagesState
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage

class BaseState(TypedDict):
    """Base state schema for agents."""

    messages: Annotated[list, add_messages]  # Messages in the graph
    
class ArtistState(TypedDict):
    """Base state schema for artists."""
    messages: Annotated[list[AnyMessage], add_messages]
    # input fields
    topic: str = Field(default="", description="设计主题")
    technique: Optional[str] = "设计技法"
    requirement: Optional[str] = Field(default="", description="用户对设计主题的特殊需求")
    
    # result fields
    design_draft: Optional[str] = ""
    final_image: Optional[bytes] = None
    project_dir: Optional[str] = None
    
    