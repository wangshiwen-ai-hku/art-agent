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

class DesignModel(BaseModel):
    positive_figure_gen_prompt: str = Field(default="", description="正形图像生成prompt，直接用于下游的图像生成模型，全英文")
    negative_figure_gen_prompt: str = Field(default="", description="负形图像生成prompt，直接用于下游的图像生成模型，全英文")
    mutual_border_gen_prompt: str = Field(default="", description="互借边框图像生成prompt，直接用于下游的图像生成模型，全英文")
    analysis_and_think: str = Field(default="", description="设计分析与构思")
    aspect_ratio: str = Field(default="1:1", description="图像比例，如3:4,1:1等")
    
    
class EditModel(BaseModel):
    edit_prompt: str = Field(default="", description="图像编辑prompt，直接用于下游的图像编辑模型，全英文")
    analysis_and_think: str = Field(default="", description="设计分析与构思")
    
class ArtistState(BaseModel):
    project_dir: str = Field(default="", description="项目保存目录")
    topic: str = Field(default="", description="设计主题")
    requirement: str = Field(default="", description="用户对设计主题的特殊需求")
    design_draft: str = ""
    
    user_example_image_paths: List[str] = []
    user_example_images: List[bytes] = []
    
    messages: Annotated[list[AnyMessage], add_messages] = []
    history_images: Annotated[list[bytes], _add_list_str] = []

    history_drafts: Annotated[list[str], _add_list_str] = []
    human_feedback: Annotated[str, _add_str] = ""
    aspect_ratio: str = Field(default="1:1", description="图像比例，如3:4,1:1等")
  
