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
import svgwrite
from src.agents.base.schema import ArtistState
from pydantic import BaseModel, Field


class SketchState(ArtistState):
    sketch_draft: str = ""
    sketch_image: List[bytes] = []
 
  
class SketchDraft(BaseModel):
    sketch_prompt: str = ""
    sketch_elements: str = "几条点、线、面组成"
    sketch_style: str = "几何风格/极简风/抽象风格/儿童画风格"
    design_description: str = "设计意图描述"
    
class SketchImaginary(ArtistState):
    sketch_imaginary: List[SketchDraft] = []
    design_analysis: str = "设计意图分析"

class CanvasState(BaseModel):
    canvas: svgwrite.Drawing = Field(default_factory=svgwrite.Drawing)
    svg_elements: List[str] = []
    
