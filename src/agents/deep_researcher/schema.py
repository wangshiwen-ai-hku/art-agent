from langgraph.graph import MessagesState
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class StepType(str, Enum):
    GENERATION = "generation"
    RESEARCH = "research"


class Step(BaseModel):
    title: str
    description: str = Field(..., description="明确说明该步骤的执行内容")
    step_type: StepType = Field(..., description="表示该步骤的性质")
    execution_res: Optional[str] = Field(default=None, description="该步骤的执行结果")


class Plan(BaseModel):
    # locale: str = Field(
    #     ..., description="例如：'zh-CN' 或 'en-US'，根据用户要求的语言或用户的语言"
    # )
    thought: str
    title: str
    steps: List[Step] = Field(
        default_factory=list,
        description="Research steps to add more content and get more context",
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "thought": (
                        "为了系统梳理牛顿第一定律课程设计，需要收集教学目标、重点难点、教具准备、教学过程及课后作业等相关信息，形成清晰的教学大纲。"
                    ),
                    "title": "牛顿第一定律课程设计",
                    "steps": [
                        {
                            "title": "教学目标",
                            "description": "知识与技能：理解牛顿第一定律的内容及其物理意义，能用牛顿第一定律解释生活和自然界中的相关现象，掌握惯性的概念及其表现。过程与方法：通过实验探究，培养观察、分析和归纳能力，学会用科学的方法解决实际问题。情感、态度与价值观：培养尊重事实、勇于探究的科学精神，认识到科学理论源于实验和观察，体会科学方法的重要性。该步骤无需检索数据，直接生成，总共约300字。",
                            "step_type": "generation",
                        },
                        {
                            "title": "教学重点与难点",
                            "description": "重点：牛顿第一定律的内容与惯性的概念。难点：理解力与运动状态的关系，区分惯性和力的关系。该步骤检索数据，总共约300字。",
                            "step_type": "research",
                        },
                        {
                            "title": "教具与准备",
                            "description": "滚动小车、斜面、润滑剂、弹簧测力计、木板、砂纸、乒乓球、细线，多媒体课件（动画、视频）、板书设计。该步骤无需检索数据，直接生成，总共约100字。",
                            "step_type": "generation",
                        },
                        {
                            "title": "导入新课",
                            "description": "情景导入，提问生活中的运动现象，播放小车运动视频，引发思考。学生带着问题进入新课。该步骤检索数据，撰写讲稿约1000字。",
                            "step_type": "research",
                        },
                        {
                            "title": "新知探究",
                            "description": "实验探究小车在不同表面上的运动，讨论小车停下的原因，引导无摩擦时的运动状态，讲述伽利略实验和牛顿第一定律，解释惯性概念与表现。该步骤检索数据，撰写讲稿约2000字。",
                            "step_type": "research",
                        },
                        {
                            "title": "知识深化与拓展",
                            "description": "力与运动状态的关系，动画演示，惯性大小与质量关系，历史故事拓展。该步骤检索数据，撰写讲稿约2000字。",
                            "step_type": "research",
                        },
                        {
                            "title": "课堂练习与小结",
                            "description": "随堂练习、简答题、小组讨论惯性现象，课堂小结回顾核心内容。该步骤检索数据，撰写讲稿约1000字。",
                            "step_type": "research",
                        },
                        {
                            "title": "课后作业与拓展",
                            "description": "课本练习题，探究性作业：观察生活中物体运动的惯性现象并写说明，设计简单实验演示惯性。该步骤无需检索数据，直接生成，总共约200字。",
                            "step_type": "generation",
                        },
                    ],
                }
            ]
        }


class State(MessagesState):
    """State for the agent system, extends MessagesState with next field."""

    # Runtime Variables
    locale: str = "zh-CN"
    subject: str = None
    query: str = None
    observations: list[str] = []
    task_description: str = None
    plan_iterations: int = 0
    current_plan: Plan | str = None
    final_report: str = ""
    auto_accepted_plan: bool = False
    background_investigation_results: str = None
    retry_count: int = 0
    filtered_resources: str = ""
