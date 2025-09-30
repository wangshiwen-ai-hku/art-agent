# canvas_tools.py
import logging
from typing import List, Tuple, Optional
from langchain_core.tools import tool
import svgwrite
import math
from src.config.manager import config
# from src.agents.base import BaseAgent
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from dataclasses import asdict
from langchain.chat_models import init_chat_model
from .math_tools import math_agent, _init_math_agent, MATH_SYSTEM_PROMPT, MATH_PROMPT, parse_math_agent_response
from src.utils.print_utils import show_messages
import json
import re, glob
import os
from pydantic import BaseModel
from langchain_core.tools import tool
import json


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DesignPlan(BaseModel):
    conceptual_analysis: str
    visual_elements: str
    layout_strategy: str
    complexity_roadmap: str
    execution_steps: str
    design_draft: str


@tool
def design_create_plan(task_description: str, width: int = 400, height: int = 400, constraints: str = "") -> str:
    """
    Create a structured design plan for a given drawing task.
    Returns JSON with `type: design_plan` and a `value` object containing fields similar to DesignPlan.
    Use this tool to produce a human-readable plan the draw agent can follow or present to the user.
    """
    try:
        # Basic heuristic-driven plan generator. Keep concise to save tokens.
        conceptual = f"Goal: {task_description}. Canvas {width}x{height}. Constraints: {constraints}" if constraints else f"Goal: {task_description}. Canvas {width}x{height}."

        visual_elements = (
            "Identify 1-3 main elements (e.g., letterform, mascot, decorative curve). "
            "Prioritize simple shapes: circles/ellipses for heads, bezier for organic curves, text as outlined paths."
        )

        layout_strategy = (
            "Center primary element; use rule-of-thirds for supporting elements. "
            "Reserve padding equal to 5-10% of the smaller canvas dimension."
        )

        complexity_roadmap = (
            "1) Sketch coarse silhouettes using primitives; 2) refine critical curves with math agent; "
            "3) apply stylization (stroke/weight)."
        )

        execution_steps = (
            "1. Block out shapes (circles, rectangles, text paths). "
            "2. For any high-precision organic curve, call math agent with a focused subtask (e.g., 'compute cubic bezier for mouse back between (x1,y1) and (x2,y2)'). "
            "3. Assemble SVG and evaluate balance."
        )

        design_draft = (
            "Draft: center main motif at canvas center; scale to 60-70% width; place secondary ornament to the lower-right; use 2-3 colors max."
        )

        plan = {
            "conceptual_analysis": conceptual,
            "visual_elements": visual_elements,
            "layout_strategy": layout_strategy,
            "complexity_roadmap": complexity_roadmap,
            "execution_steps": execution_steps,
            "design_draft": design_draft,
        }

        return json.dumps({"type": "design_plan", "value": plan, "explanation": "Structured design plan generated.", "error": ""})
    except Exception as e:
        logger.exception("design_create_plan failed")
        return json.dumps({"type": "design_plan", "value": {}, "explanation": "", "error": str(e)})


@tool
def design_reflect(svg_snippet: str, design_goal: str = "", limit_suggestions: int = 5) -> str:
    """
    Critique a proposed SVG (or partial SVG) and return a reflection report with concrete suggestions.
    The tool returns a JSON with `type: reflection` and `value` containing `critique`, `suggestions`, and optional `math_subtasks` for the math agent.
    """
    try:
        critique_parts = []
        suggestions = []
        math_subtasks = []

        if not svg_snippet or not svg_snippet.strip():
            critique_parts.append("No SVG content provided.")
            suggestions.append("Provide a coarse path or shape for the area that needs precision (e.g., mouse back curve).")
            return json.dumps({"type": "reflection", "value": {"critique": " ".join(critique_parts), "suggestions": suggestions, "math_subtasks": math_subtasks}, "explanation": "", "error": ""})

        # Heuristics: look for common issues
        lower = svg_snippet.lower()
        # Too many nodes / very long paths
        if len(svg_snippet) > 2000:
            critique_parts.append("SVG appears very long/complex; consider decomposing into smaller segments.")
            suggestions.append("Split complex path into logical subpaths (head, body, tail) and request math agent per subpath.")

        # Lack of viewBox or root svg
        if '<svg' not in lower:
            critique_parts.append("SVG snippet is a fragment (no <svg> root). Ensure elements are well-formed.")
            suggestions.append("Wrap fragments in an <svg> with correct width/height or provide coordinates relative to canvas.")

        # Balance heuristic: check for presence of transform/translate hints
        if 'translate' not in lower and 'transform' not in lower:
            suggestions.append("Consider aligning or translating primary motif to canvas center to improve balance.")

        # Suggest math subtasks by finding long path commands
        import re
        path_cmds = re.findall(r'([MmLlHhVvCcQqTtSsAaZz])', svg_snippet)
        if path_cmds and len(path_cmds) > 50:
            math_subtasks.append("Fit or reparameterize long paths: 'fit cubic bezier to points segment X-Y' for smoother control.'")

        # Produce up to limit suggestions
        suggestions = suggestions[:limit_suggestions]

        critique = " ".join(critique_parts) if critique_parts else "No major structural issues detected by heuristic checks."

        # Example of suggested focused math subtasks (helpful hints for draw_agent to send to math agent)
        if not math_subtasks:
            math_subtasks = [
                "Compute cubic bezier for a smooth mouse-back between (x1,y1) and (x2,y2) with symmetry axis x=...",
                "Sample 20 points along a sigmoid-like curve and fit a cubic Bezier to them."
            ]

        value = {"critique": critique, "suggestions": suggestions, "math_subtasks": math_subtasks}

        return json.dumps({"type": "reflection", "value": value, "explanation": "Reflection completed.", "error": ""})
    except Exception as e:
        logger.exception("design_reflect failed")
        return json.dumps({"type": "reflection", "value": {}, "explanation": "", "error": str(e)})

