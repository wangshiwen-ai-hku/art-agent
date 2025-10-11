# canvas_tools.py
import logging
from typing import List, Tuple, Optional, Dict, Any
from langchain_core.tools import tool
import svgwrite
import re
import json
import os
import uuid
from xml.etree import ElementTree as ET

from src.config.manager import config
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage, BaseMessage
from dataclasses import asdict
from langchain.chat_models import init_chat_model
from src.utils.print_utils import show_messages
from .math_tools import (
    math_agent,
    _init_math_agent,
    MATH_SYSTEM_PROMPT,
    MATH_PROMPT,
    parse_math_agent_response,
)  # Assuming parse_math_agent_response is defined in math_tools.py

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------- 编辑工具 ----------
@tool
def add_line(svg_code: str, x1: int, y1: int, x2: int, y2: int,
                  color: str = "black", width: int = 2) -> str:
    """
    Add a straight line to an existing SVG code and return the updated SVG string.
    Args:
        svg_code: str, the existing SVG code
        x1: int, the x-coordinate of the start point of the line
        y1: int, the y-coordinate of the start point of the line
        x2: int, the x-coordinate of the end point of the line
        y2: int, the y-coordinate of the end point of the line
        color: str, the color of the line
        width: int, the width of the line
    """
    try:
        line = svgwrite.shapes.Line(start=(x1, y1), end=(x2, y2),
                                    stroke=color, stroke_width=width)
        element_str = line.tostring()
        updated_svg = re.sub(r'</svg>', f'{element_str}</svg>', svg_code, flags=re.IGNORECASE)
        explanation = f"Added line from ({x1},{y1}) to ({x2},{y2}) with color '{color}' and width {width}."
        return json.dumps({"type": "svg_string", "value": updated_svg, "explanation": explanation, "error": ""})
    except Exception as e:
        return json.dumps({"error": f"Failed to add line: {str(e)}"})

@tool
def add_circle(svg_code: str, x: int, y: int, radius: int,
                    color: str = "black", fill: str = "none") -> str:
    """
    Add a circle to an existing SVG code and return the updated SVG string. `fill` can be a color name or 'none'.
    Args:
        svg_code: str, the existing SVG code
        x: int, the x-coordinate of the center of the circle
        y: int, the y-coordinate of the center of the circle
        radius: int, the radius of the circle
        color: str, the color of the circle
        fill: str, the fill color of the circle
    """
    try:
        circ = svgwrite.shapes.Circle(center=(x, y), r=radius,
                                      stroke=color, fill=fill)
        element_str = circ.tostring()
        updated_svg = re.sub(r'</svg>', f'{element_str}</svg>', svg_code, flags=re.IGNORECASE)
        explanation = f"Added circle at ({x},{y}) with radius {radius}, color '{color}', fill '{fill}'."
        return json.dumps({"type": "svg_string", "value": updated_svg, "explanation": explanation, "error": ""})
    except Exception as e:
        return json.dumps({"error": f"Failed to add circle: {str(e)}"})

@tool
def modify_path(svg_code: str, path_index: int, new_path_data: str,
                     color: Optional[str] = None, fill: Optional[str] = None, width: Optional[int] = None) -> str:
    """
    Modify a specific <path> element in the SVG by index (0-based) with new data/attributes and return the updated SVG string.
    Args:
        svg_code: str, the existing SVG code
        path_index: int, the 0-based index of the path to modify
        new_path_data: str, the new 'd' attribute string
        color: str, optional, new stroke color
        fill: str, optional, new fill color
        width: int, optional, new stroke width
    """
    try:
        paths = re.findall(r'<path[^>]*>', svg_code, re.IGNORECASE | re.DOTALL)
        if path_index >= len(paths) or path_index < 0:
            return json.dumps({"error": "Path index out of range"})
        
        path_tag = paths[path_index]
        updated_tag = re.sub(r'd="[^"]*"', f'd="{new_path_data}"', path_tag)
        if color:
            updated_tag = re.sub(r'stroke="[^"]*"', f'stroke="{color}"', updated_tag) if 'stroke="' in updated_tag else updated_tag.replace('>', f' stroke="{color}">')
        if fill:
            updated_tag = re.sub(r'fill="[^"]*"', f'fill="{fill}"', updated_tag) if 'fill="' in updated_tag else updated_tag.replace('>', f' fill="{fill}">')
        if width is not None:
            updated_tag = re.sub(r'stroke-width="[^"]*"', f'stroke-width="{width}"', updated_tag) if 'stroke-width="' in updated_tag else updated_tag.replace('>', f' stroke-width="{width}">')
        
        updated_svg = svg_code.replace(path_tag, updated_tag, 1)
        explanation = f"Modified path at index {path_index} with new data '{new_path_data[:50]}...'."
        return json.dumps({"type": "svg_string", "value": updated_svg, "explanation": explanation, "error": ""})
    except Exception as e:
        return json.dumps({"error": f"Failed to modify path: {str(e)}"})

@tool
def add_gradient(svg_code: str, gradient_id: str, colors: List[Tuple[float, str]],
                      gradient_type: str = "linear", x1: float = 0, y1: float = 0,
                      x2: float = 1, y2: float = 1) -> str:
    """
    Add a linear or radial gradient definition to <defs> in SVG and return the updated SVG string.
    Args:
        svg_code: str, the existing SVG code
        gradient_id: str, unique ID for the gradient
        colors: List[Tuple[float, str]], list of (offset, color) e.g., [(0, "#ff0000"), (1, "#0000ff")]
        gradient_type: str, "linear" or "radial"
        x1,y1,x2,y2: floats, for linear gradients
    """
    try:
        if '<defs>' not in svg_code.lower():
            svg_code = re.sub(r'<svg[^>]*>', r'\g<0><defs></defs>', svg_code, flags=re.IGNORECASE)
        
        if gradient_type == "linear":
            grad = svgwrite.gradients.LinearGradient(start=(x1, y1), end=(x2, y2), id=gradient_id)
        else:
            grad = svgwrite.gradients.RadialGradient(center=(x1, y1), r=1, id=gradient_id)
        
        for offset, color in colors:
            grad.add_stop_color(offset=offset, color=color)
        
        element_str = grad.tostring()
        updated_svg = re.sub(r'</defs>', f'{element_str}</defs>', svg_code, flags=re.IGNORECASE)
        explanation = f"Added {gradient_type} gradient with ID '{gradient_id}'."
        return json.dumps({"type": "svg_string", "value": updated_svg, "explanation": explanation, "error": ""})
    except Exception as e:
        return json.dumps({"error": f"Failed to add gradient: {str(e)}"})

@tool
def remove_element(svg_code: str, element_index: int, element_type: str = "path") -> str:
    """
    Remove a specific element (e.g., <path>, <circle>) by index (0-based) and return the updated SVG string.
    Args:
        svg_code: str, the existing SVG code
        element_index: int, the 0-based index of the element to remove
        element_type: str, the type of element to remove (e.g., "path", "circle", "line")
    """
    try:
        pattern = fr'<{element_type}[^>]*>(?:.*?</{element_type}>)?' if element_type in ["g", "defs"] else fr'<{element_type}[^>]*>'
        elements = re.findall(pattern, svg_code, re.IGNORECASE | re.DOTALL)
        if element_index >= len(elements) or element_index < 0:
            return json.dumps({"error": "Element index out of range"})
        
        element_tag = elements[element_index]
        updated_svg = svg_code.replace(element_tag, "", 1)
        explanation = f"Removed {element_type} element at index {element_index}."
        return json.dumps({"type": "svg_string", "value": updated_svg, "explanation": explanation, "error": ""})
    except Exception as e:
        return json.dumps({"error": f"Failed to remove element: {str(e)}"})

@tool
def add_path(svg_code: str, path_data: str,
                  color: str = "black", fill: str = "none", width: int = 2) -> str:
    """
    Add a path to an existing SVG code and return the updated SVG string.
    """
    path = svgwrite.path.Path(d=path_data, stroke=color, fill=fill, stroke_width=width)
    element_str = path.tostring()
    updated_svg = re.sub(r'</svg>', f'{element_str}</svg>', svg_code, flags=re.IGNORECASE)
    explanation = f"Added path with data '{path_data[:50]}...'."
    return json.dumps({"type": "svg_string", "value": updated_svg, "explanation": explanation, "error": ""})

@tool
async def calculate_path_data_with_math_agent(task_description: str, coarse_svg_path_data: Optional[str] = None, other_info: Optional[str] = None, width: Optional[int] = None, height: Optional[int] = None) -> str:
    """
    Use this tool to calculate or modify the complex data of SVG path. It can calculate float datapoints using powerful symmetries/rotation/translation/scaling/etc math tools. It will return the path data.
    You must use `modify_path` tool to apply the path data.
    Args:
        task_description: str, the task description (e.g., "Modify the path to add a symmetric curve")
        coarse_svg_path_data: str, recommended, the coarse SVG path data to modify
        other_info: str, optional, possible settings of position, center, size, area, etc.
        width: int, optional, the width of the canvas
        height: int, optional, the height of the canvas
    Returns:
        JSON {"type": "path_data", "value": "...", "math_agent_explanation": "...", "explanation": "...", "error": ""}
    """
    # Defensive checks similar to draw_canvas_tools.py
    def _is_broad_design_text(text: str) -> bool:
        if not text:
            return False
        lowered = text.lower()
        broad_keywords = [
            'design', 'logo', 'compose', 'layout', 'full', 'whole', 'combine', 'arrange', 'complete', 'entire',
            'create a', 'make a', 'draw a', 'sketch a'
        ]
        if len(text) > 300:
            return True
        for kw in broad_keywords:
            if kw in lowered:
                return True
        return False

    # if _is_broad_design_text(task_description) and not coarse_svg_path_data:
    #     guidance = (
    #         "Task appears to be a full design. The math agent only accepts focused micro-tasks (e.g., "
    #         "'compute cubic bezier for mouse-back between (x1,y1) and (x2,y2) with symmetry axis x=50'). "
    #         "Please provide a concise computational subtask and, if possible, a coarse_svg_path_data sample."
    #     )
    #     return json.dumps({
    #         "type": "error",
    #         "value": "task_too_broad",
    #         "math_agent_explanation": "",
    #         "explanation": guidance,
    #         "usage": "Refactor your request into a single numeric geometric computation and retry."
    #     })

    messages = [
        SystemMessage(content=MATH_SYSTEM_PROMPT),
        HumanMessage(content=MATH_PROMPT.format(user_instruction=task_description, coarse_svg_path_data=coarse_svg_path_data, other_info=other_info, width=width, height=height))
    ]
    state = await math_agent.ainvoke({"messages": messages})

    path_data = parse_math_agent_response(state)

    return json.dumps({
        "type": "path_data",
        "value": path_data.data,
        "math_agent_explanation": path_data.explanation,
        "explanation": "This is SVG path data generated by the math agent; use it with modify_path() or add_path() to apply.",
        "usage": "Use modify_path(svg_code=..., path_index=..., new_path_data=this_value, ...) to apply or add_path(svg_code=..., path_data=this_value, ...) to add."
    })

EditCanvasAgentTools = [
    add_line,
    add_circle,
    add_path,
    add_gradient,

    remove_element,

    modify_path,
    calculate_path_data_with_math_agent,
]

SYSTEM_PROMPT = """
# Role
You are an expert SVG editor. Interpret semantic edit requests, reference the existing design plan and critique notes when provided, and return a cohesive, production-ready SVG. Maintain geometry fidelity, respect canvas bounds, and document the operations you perform.

# Tools
- `add_` tools: introduce new shapes or text blocks.
- `modify_` tools: adjust attributes/path data of existing elements.
- `remove_` tools: delete elements by index/type.
- `calculate_path_data_with_math_agent`: derive precise path instructions (e.g., bezier adjustments). Supply clear targets and reuse results via `modify_path` or `add_path`.

# Expectations
- Produce deterministic edits—no randomness.
- Reference targeted element ids or indices when possible.
- Return a complete `<svg>` document and a concise change log.
"""

EDIT_PROMPT = """
user_instruction: {user_instruction}
original_svg_code: {original_svg_code}
revision_notes: {revision_notes}
target_elements: {target_elements}
plan_context: {plan_context}
"""

edit_agent = None

def _init_edit_agent():
    global agent
    if edit_agent is None:
        agent_config = config.get_agent_config("edit_agent", "core")
        llm = init_chat_model(**asdict(agent_config.model))
        agent = create_react_agent(llm, name='agent', tools=EditCanvasAgentTools)
        logger.info(f"🖊 [agent_with_tool] agent initialized")
    return agent

edit_agent = _init_edit_agent()

@tool
async def edit_agent_with_tool(
    task_description: str,
    original_svg_code: str,
    revision_notes: str = "",
    target_elements: str = "",
    plan_context: str = "",
):
    """
    Edit agent with tool. You can use this tool to edit the SVG code of the canvas. You will be given the task description and original SVG code, and your task is to edit the SVG code.
    Args:
        task_description: str, the task description
        original_svg_code: str, the original SVG code
    Returns:
        new_svg: str, the entire updated SVG code
    """
    logger.info(
        "🖊 [edit_agent_with_tool] task_description=%s revision_notes=%s plan=%s",
        task_description,
        bool(revision_notes),
        bool(plan_context),
    )
    prompt = EDIT_PROMPT.format(
        user_instruction=task_description,
        original_svg_code=original_svg_code,
        revision_notes=revision_notes or "",
        target_elements=target_elements or "",
        plan_context=plan_context or "",
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    state = await agent.ainvoke({"messages": messages}, {"recursion_limit": 100})
    show_messages(state.get("messages", []))

    # Fallback to assembling from tool calls if the final message is not a complete SVG.
    svg_segments = []
    for message in state.get("messages", []):
        if isinstance(message, ToolMessage):
            data = json.loads(message.content)
            if data.get("type") == "svg_string":
                svg_segments.append(data.get("value"))  # But actually, tools return full updated SVG, so last one is final

    final_messages = state.get("messages", [])
    new_svg = svg_segments[-1] if svg_segments else ""
    if not new_svg:
        for message in reversed(final_messages):
            if isinstance(message, (AIMessage, ToolMessage)):
                content = message.content
                if isinstance(content, str) and content.strip().startswith("<svg"):
                    new_svg = content
                    break
                if isinstance(content, dict) and content.get("value", "").startswith("<svg"):
                    new_svg = content["value"]
                    break

    explanation = "[edit agent]"
    if final_messages and isinstance(final_messages[-1], AIMessage):
        explanation = final_messages[-1].content

    tool_events = _collect_tool_events(final_messages)
    summary = _summarize_svg(new_svg) if new_svg else {"elements": []}

    result_payload = {
        "type": "edit_result",
        "svg": new_svg,
        "value": new_svg,
        "explanation": explanation,
        "tool_events": tool_events,
        "summary": summary,
        "revision_notes": revision_notes[:300] if revision_notes else "",
        "target_elements": target_elements,
        "plan_used": bool(plan_context),
    }

    return json.dumps(result_payload)

async def run_agent_with_tool(task_description: str, original_svg_code: str):
    """
    Run edit agent with tool for testing. Edits the SVG code based on the task description.
    """
    logger.info(f"🖊 [run_agent_with_tool] task_description: {task_description}")
    messages = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=
                EDIT_PROMPT.format(
                    user_instruction=task_description,
                    original_svg_code=original_svg_code,
                    revision_notes="make improvements for highlights",
                    target_elements="",
                    plan_context="",
                )
                + "After your execution, please give me some feedback on your tools, are they useful?"
            ),
        ]
    }
    state = await agent.ainvoke(messages)
    show_messages(state.get("messages", []), limit=-1)
    messages_list = state.get("messages", [])
    # Extract the tool message with type "svg_string" (last one is the final updated SVG)
    svg_segments = []
    for message in messages_list:
        if isinstance(message, ToolMessage):
            data = json.loads(message.content)
            if data.get("type") == "svg_string":
                svg_segments.append(data.get("value"))
    
    new_svg = svg_segments[-1] if svg_segments else messages_list[-1].content
    import uuid
    output_dir = "output/test_edit/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir + uuid.uuid4().hex + ".svg"
    open(output_path, "w", encoding="utf-8").write(new_svg)
    logger.info(f"🖊 [run_agent_with_tool] saved to {output_path}")
    return new_svg

if __name__ == "__main__":
    import asyncio
    original_svg = '<svg width="400" height="400" xmlns="http://www.w3.org/2000/svg"><rect fill="lightgray" height="100" stroke="black" width="100" x="165" y="180" /><rect fill="lightgray" height="60" stroke="black" width="20" x="170" y="120" /><rect fill="lightgray" height="60" stroke="black" width="20" x="195" y="120" /><rect fill="lightgray" height="60" stroke="black" width="20" x="220" y="120" /><rect fill="lightgray" height="60" stroke="black" width="20" x="245" y="120" /><rect fill="lightgray" height="50" stroke="black" width="30" x="135" y="205" /></svg>'
    task = "Make this hand more vivid and colorful."
    asyncio.run(run_agent_with_tool(task, original_svg))
def _summarize_svg(svg_code: str, limit: int = 10) -> Dict[str, Any]:
    try:
        root = ET.fromstring(svg_code)
    except ET.ParseError:
        return {"elements": [], "error": "invalid_svg"}

    elements: List[Dict[str, Any]] = []
    for idx, elem in enumerate(root.iter()):
        if idx >= limit:
            break
        tag = elem.tag.split('}')[-1]
        if tag == "svg":
            continue
        record: Dict[str, Any] = {"tag": tag}
        if "id" in elem.attrib:
            record["id"] = elem.attrib["id"]
        if "d" in elem.attrib:
            record["d"] = elem.attrib["d"][:120]
        elements.append(record)
    return {"elements": elements, "element_count": len(elements)}


def _collect_tool_events(messages: List[BaseMessage], limit: int = 6) -> List[str]:
    events: List[str] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            content = message.content
            if isinstance(content, str):
                events.append(content[:240])
            elif isinstance(content, dict):
                summary = content.get("explanation") or content.get("value") or str(content)
                events.append(str(summary)[:240])
        if len(events) >= limit:
            break
    return events
