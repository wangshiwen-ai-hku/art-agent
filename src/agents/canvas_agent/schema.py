from typing import TypedDict, List, Optional
from src.agents.base.schema import ArtistState
from pydantic import BaseModel, Field


class SketchDraft(BaseModel):
    """A model for a single sketch idea, including its description and drawing instructions."""
    design_description: str = Field(..., description="A concise explanation of the core design concept.")
    sketch_description: str = Field(..., description="A detailed description of the visual elements for the sketch.")
    drawing_prompt: Optional[str] = Field(None, description="A detailed, step-by-step prompt for an AI drawing agent to follow.")


class SketchOutput(TypedDict):
    """The final output for a single sketch, including the original idea and the SVG elements."""
    idea: SketchDraft
    svg_elements: List[str]


class CanvasState(ArtistState):
    """The state of the CanvasAgent, including all sketch ideas and final outputs."""
    topic: str
    technique: str
    requirement: str
    sketch_ideas: List[SketchDraft]
    sketches: List[SketchOutput]
    generated_image_paths: Optional[List[str]] = None
    
    ## for single node test
    stage: Optional[str] = None # draw_only | describe_only | generate
    draw_prompt: Optional[str] = None
