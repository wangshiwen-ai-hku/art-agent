# src/infra/tools/design_agent/schemas.py
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel
from src.agents.base.schema import ArtistState

# Registry for runtime-only Pydantic models
_MODEL_REGISTRY: dict = {}

class StageEnum(Enum):
    PLAN = "plan"
    BRAINSTORM = "brainstorm"
    COMPOSE = "compose"
    SUMMARIZE = "summarize"
    REFLECT = "reflect"
    PREVIEW_IMAGE = "preview_image"
    PREVIEW_SVG = "preview_svg"
    HUMAN = "human"
    END = "end"

class ComposeMap(BaseModel):
    picked_elements: str
    elements_composition: str
    imagen_style_prefix: str
    design_prompt: str
    next_step: StageEnum

class PlanMap(BaseModel):
    plan_steps: list[StageEnum]

class ReflectMap(BaseModel):
    reflection: str
    old_design_prompt: str | List[str]
    improved_design_prompt: str
    next_step: StageEnum

class MultiModalMap(BaseModel):
    image_generation_prompt: str
    image_url: str
    next_step: StageEnum

class SummarizeMap(BaseModel):
    draw_description: str
    design_prompt: str
    image_preview_url: str

class DesignState(ArtistState):
    plan_map: Optional[PlanMap] = None
    stage: Optional[StageEnum] = None
    generated_image_paths: List[str] = []
    generated_svg_paths: List[str] = []
    elements_map_schema_key: Optional[str] = None
    element_map_messages: List[BaseModel] = []
    compose_map_messages: List[ComposeMap] = []
    reflect_map_messages: List[ReflectMap] = []
    multi_modal_messages: List[MultiModalMap] = []
    notes: str = ""
    stage_iteration: int = 0
    summarize_map: Optional[SummarizeMap] = None
