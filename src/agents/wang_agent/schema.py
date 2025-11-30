"""State schema for Wang Agent."""

from typing import TypedDict, List, Optional, Dict, Any
from src.agents.base.schema import ArtistState
from pydantic import BaseModel, Field


class Instruction(BaseModel):
    """Instruction containing intention and criteria."""
    intention: str = Field(..., description="The precise intention, e.g., 'a tree'")
    criteria: List[str] = Field(..., description="List of criteria for evaluation")


class DesignPrompt(BaseModel):
    """Output from THINK node: complex design prompt."""
    design_prompt: str = Field(..., description="Detailed design prompt for image generation")
    image_prompt: str = Field(..., description="Prompt specifically for image generation")
    instruction: Instruction = Field(..., description="Instruction with intention and criteria")


class WeightedComponent(BaseModel):
    """A component of the image with its weight and description."""
    weight: float = Field(..., description="Weight/priority of this component (higher = more important)")
    description: str = Field(..., description="Description of what this component represents")
    svg_elements: Optional[str] = Field(None, description="Generated SVG elements for this component")
    structure_info: Optional[Dict[str, Any]] = Field(None, description="Structural information extracted from PNG (keypoints, contours, colors, etc.)")


class WeightedComponentsList(BaseModel):
    """Wrapper for a list of weighted components."""
    components: List[WeightedComponent] = Field(..., description="List of weighted components")


class StageResult(BaseModel):
    """Result of a single generation stage."""
    stage_number: int = Field(..., description="Stage number (1, 2, 3, ...)")
    component: WeightedComponent = Field(..., description="The component generated in this stage")
    svg_code: str = Field(..., description="Complete SVG code after this stage")
    passed_reflection: bool = Field(False, description="Whether this stage passed reflection")


class ReflectionResult(BaseModel):
    """Result of reflection/evaluation."""
    passed: bool = Field(..., description="Whether the current SVG meets quality standards")
    feedback: str = Field(..., description="Detailed feedback on the SVG quality")
    suggestions: Optional[List[str]] = Field(None, description="Suggestions for improvement if not passed")
    metrics: Optional[Dict[str, float]] = Field(None, description="Quantitative metrics (SSIM, MSE, PSNR, LPIPS)")
    difference_regions: Optional[List[Dict]] = Field(None, description="Regions with high differences")


class WangState(ArtistState):
    """State schema for Wang Agent - iterative SVG generation with reflection."""
    
    # Input
    user_intention: Optional[str] = None  # User's complex intention
    user_image_path: Optional[str] = None  # Optional user-provided image
    
    # THINK output
    design_prompt: Optional[str] = None  # Complex design prompt
    image_prompt: Optional[str] = None  # Image generation prompt
    instruction: Optional[Instruction] = None  # Instruction with intention and criteria
    
    # Generated resources
    initial_image_path: Optional[str] = None  # Generated or user-provided initial image
    
    # Weighted components analysis
    weighted_components: Optional[List[WeightedComponent]] = None  # Components sorted by weight
    
    # Generation stages
    current_stage: int = 0  # Current stage number (0 = not started)
    stages: Optional[List[StageResult]] = None  # History of all stages
    current_svg: Optional[str] = None  # Current SVG code
    
    # Reflection
    last_reflection: Optional[ReflectionResult] = None
    
    # Control
    max_iterations: int = 7  # Maximum number of iterations
    iteration_count: int = 0  # Current iteration count
    is_complete: bool = False  # Whether generation is complete (quality passed)
    stop_reason: Optional[str] = None  # Reason for stopping: "quality_passed", "max_iterations", "error"
    fine_tune_attempts: int = 0  # Number of fine-tuning attempts for current stage
    max_fine_tune_attempts: int = 2  # Maximum fine-tuning attempts before rollback

