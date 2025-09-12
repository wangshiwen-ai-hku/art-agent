from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage, HumanMessage
from enum import Enum
from typing import (
    Annotated,
    List,
    Optional,
    Any,
    Callable,
    Literal,
    Union,
    cast,
)

from pydantic import BaseModel, Field

def _add_list_str(current: List[bytes], new: List[bytes]) -> List[bytes]:
    return current + new

def _add_str(current: str, new: str) -> str:
    return current + new

class ArtistState(BaseModel):
    project_dir: str = Field(default="", description="项目保存目录")
    topic: str = Field(default="", description="设计主题")
    requirement: str = Field(default="", description="用户对设计主题的特殊需求")
    design_draft: str = ""
    
    user_example_images: List[str] = []
    
    messages: Annotated[list[AnyMessage], add_messages] = []
    history_images: Annotated[list[bytes], _add_list_str] = []

    history_drafts: Annotated[list[str], _add_list_str] = []
    human_feedback: Annotated[str, _add_str] = ""
  
