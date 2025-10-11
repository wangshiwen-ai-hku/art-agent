from typing import TypedDict, List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

# === 定义清晰的枚举类型 ===
class AgentStage(str, Enum):
    CHAT = "chat"
    PLAN = "plan"
    DRAW = "draw"
    EDIT = "edit"
    CRITIQUE = "critique"
    DESCRIBE = "describe"
    PICK_PATH = "pick"
    GENERATE_IMAGE = "generate"

class UserIntent(str, Enum):
    CREATE = "create"      # 创建新图形
    MODIFY = "modify"      # 修改现有图形
    ANALYZE = "analyze"    # 分析描述图形
    SELECT = "select"      # 选择路径
    CHAT = "chat"          # 普通对话
    DESIGN = "design_image"      # routed to design general agent (design branch)

class Router(BaseModel):
    """Represents the routing decision for a user's request."""
    intent: UserIntent = Field(
        description="The user's intent. This should be one of 'create', 'modify', 'analyze', 'select', or 'chat'."
    )
    query: str = Field(
        description="The user's query, extracted from the message, that can be used as a prompt for other agents. For example, for 'create', this would be the description of the SVG to draw."
    )

ROUTER_PROMPT = """You are a router for a multi-agent system that designs SVG images. Your task is to analyze the user's message and determine their intent. Based on the intent, you will route the request to the appropriate agent.

The available intents are:
- `create`: The user wants to create a new SVG image from a description.
- `modify`: The user wants to modify an existing SVG image.
- `analyze`: The user wants to get a description of an image (e.g., to turn a raster image into an SVG).
- `select`: The user wants to select a specific part (path) of an SVG.
- `chat`: The user is having a general conversation, asking questions, or providing feedback that doesn't fall into the other categories.
- `design_image`: The user wants to run the design-general image-generation branch (preview images or multimodal design exploration).

Given the user's message, determine the `intent` and extract the core `query` that the next agent will use. For example, if the user says "Can you draw a smiling sun for me?", the intent is `create` and the query is "a smiling sun". If the user says "make the circle bigger", the intent is `modify` and the query is "make the circle bigger". If it's a simple greeting like "hello", the intent is `chat` and the query can be the original message.
"""

# === 核心数据模型 ===
class SketchDesign(BaseModel):
    """草图设计概念"""
    concept: str = Field(..., description="设计概念描述")
    visual_description: str = Field(..., description="视觉元素详细描述")
    drawing_instructions: Optional[str] = Field(None, description="绘制指令")

class SvgArtwork(BaseModel):
    """SVG艺术作品"""
    svg_code: str = Field(..., description="完整的SVG代码")
    elements: List[str] = Field(default_factory=list, description="SVG元素列表")
    metadata: dict = Field(default_factory=dict, description="元数据")


class DesignPlan(BaseModel):
    """Structured design plan captured during planning stage."""

    concept_summary: str = Field(..., description="整体概念与愿景")
    visual_elements: List[str] = Field(default_factory=list, description="关键视觉元素清单")
    layout_strategy: str = Field(..., description="布局与构图策略")
    execution_steps: List[str] = Field(..., description="逐步执行指令，便于绘制与编辑阶段参考")


class CritiqueFeedback(BaseModel):
    """Feedback record for generated SVG output."""

    approved: bool = Field(..., description="是否接受当前设计")
    overall_feedback: str = Field(..., description="总体评价")
    revision_instructions: List[str] = Field(default_factory=list, description="需要改动的要点")
    svg_targets: List[str] = Field(default_factory=list, description="涉及的SVG路径或元素标识")

# === 会话状态 ===
class ConversationState(TypedDict):
    """对话状态"""
    messages: List  # LangChain消息列表
    current_topic: Optional[str]

# === 工作状态 ===
class WorkflowState(TypedDict):
    """工作流状态"""
    current_stage: AgentStage
    current_intent: UserIntent
    is_completed: bool
    manual_override: bool

# === 画布内容状态 ===
class CanvasContentState(TypedDict):
    """画布内容状态"""
    current_svg: Optional[SvgArtwork]
    svg_history: List[SvgArtwork]  # 版本历史
    reference_images: List[str]    # 参考图片路径
    tool_events: List[str]         # 最近一次工具事件的概要
    design_plan: Optional[DesignPlan]
    critique_history: List[CritiqueFeedback]

# === 项目状态 ===
class ProjectState(TypedDict):
    """项目状态"""
    project_dir: Optional[str]
    saved_files: List[str]
    metadata: Dict[str, Any]

# === 主状态 ===
class CanvasState(TypedDict):
    """CanvasAgent的主状态 - 清晰分离关注点"""
    
    # 会话相关
    conversation: ConversationState
    
    # 工作流相关
    workflow: WorkflowState
    
    # 内容相关
    content: CanvasContentState
    
    # 项目相关
    project: ProjectState
    
    # 当前用户输入
    user_input: str

# === 辅助函数 ===
def create_initial_state(project_dir: Optional[str] = None) -> CanvasState:
    """创建初始状态"""
    return {
        "conversation": {
            "messages": [],
            "current_topic": None
        },
        "workflow": {
            "current_stage": AgentStage.CHAT,
            "current_intent": UserIntent.CHAT,
            "is_completed": False,
            "manual_override": False,
        },
        "content": {
            "current_svg": None,
            "svg_history": [],
            "reference_images": [],
            "tool_events": [],
            "design_plan": None,
            "critique_history": [],
        },
        "project": {
            "project_dir": project_dir,
            "saved_files": [],
            "metadata": {},
        },
        "user_input": ""
    }

def update_workflow_state(state: CanvasState, stage: AgentStage, intent: UserIntent) -> CanvasState:
    """更新工作流状态"""
    state["workflow"]["current_stage"] = stage
    state["workflow"]["current_intent"] = intent
    state["workflow"]["manual_override"] = False
    return state

def add_svg_to_history(state: CanvasState, svg_code: str, elements: List[str] = None) -> CanvasState:
    """添加SVG到历史记录"""
    svg_artwork = SvgArtwork(
        svg_code=svg_code,
        elements=elements or extract_svg_elements(svg_code)
    )
    
    state["content"]["current_svg"] = svg_artwork
    state["content"]["svg_history"].append(svg_artwork)
    return state

def extract_svg_elements(svg_code: str) -> List[str]:
    """从SVG代码中提取元素（简化实现）"""
    import re
    return re.findall(r'<[^>]+>', svg_code)

# === 状态映射（简化版）===
STATE_MAP = {
    UserIntent.CHAT: "chat_node",
    UserIntent.CREATE: "plan_node", 
    UserIntent.DESIGN: "design_image_node",
    UserIntent.MODIFY: "edit_node",
    UserIntent.ANALYZE: "describe_node",
    UserIntent.SELECT: "pick_path_node"
}

AGENT_STAGE_MAP = {
    AgentStage.CHAT: "chat_node",
    AgentStage.PLAN: "plan_node",
    AgentStage.DRAW: "draw_node",
    AgentStage.EDIT: "edit_node", 
    AgentStage.CRITIQUE: "critique_node",
    AgentStage.DESCRIBE: "describe_node",
    AgentStage.PICK_PATH: "pick_path_node",
    AgentStage.GENERATE_IMAGE: "design_image_node"
}

INTENT_TO_STAGE = {
    UserIntent.CHAT: AgentStage.CHAT,
    UserIntent.CREATE: AgentStage.PLAN,
    UserIntent.MODIFY: AgentStage.EDIT,
    UserIntent.ANALYZE: AgentStage.DESCRIBE,
    UserIntent.SELECT: AgentStage.PICK_PATH,
    UserIntent.DESIGN: AgentStage.GENERATE_IMAGE,
}
