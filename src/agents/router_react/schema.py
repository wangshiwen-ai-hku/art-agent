"""
Router React agent state schema.
"""

from typing import Optional, List
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages

from ..base.schema import BaseState


class RouterReactState(BaseState):
    """State schema for router react multi-agent system with handoff mechanism."""

    current_plan: Optional[
        dict
    ]  # Current plan from plan_tool for step-by-step execution
    current_step: Optional[int]  # Current step index in the plan (0-based)
    step_results: Optional[List[str]]  # Results from each completed step
    agent_mode: Optional[str]  # Current agent mode: 'router', 'research', 'analysis'
    sub_agent: Optional[str]  # Current sub-agent name
    task: Optional[str]  # Current task description for sub-agent
    is_task_completed: Optional[bool]  # Whether the overall task is completed
