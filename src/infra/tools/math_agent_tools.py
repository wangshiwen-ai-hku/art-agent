"""math_agent_tools.py

This file implements a revolutionary, graph-based SVG design agent inspired by
the workflows of expert designers using tools like Adobe Illustrator. It moves
beyond a simple ReAct loop to a structured, multi-node graph that can plan,
execute, summarize, reflect, and learn.

Core architectural concepts:
- **Graph-Based Workflow:** Uses LangGraph to orchestrate a flow between specialized nodes (Planner, Executor, Summarizer, Reflector).
- **Abstracted Geometric State:** Maintains a concise, textual summary of the design's state to avoid context window overload and improve reasoning.
- **Tool Proficiency Playbook:** A persistent memory (`memory.md`) stores successful tool combinations and design patterns, allowing the agent to develop "tool intuition."
- **Predict-Validate & Staged Decomposition:** The Planner breaks tasks into phases and can incorporate predict-validate loops for error correction.
- **Iterative Refinement:** A Reflector node critiques the design, enabling the agent to loop back and improve its work.
"""
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import subprocess
from langchain_core.tools import tool
from src.config.manager import config
from langgraph.prebuilt import create_react_agent
import logging
from langchain.chat_models import init_chat_model
from dataclasses import asdict
# from langchain_google_vertexai import ChatVertexAI
import json
import math
import numpy as np
import sympy as sp
import re
from typing import Tuple, List, Dict, Any, Optional, TypedDict

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, HumanMessage, SystemMessage
from src.utils.print_utils import show_messages
import sys
import time
import os
from langgraph.graph import StateGraph, END

# ==================================================================================================
# 1. LOGGING AND CONFIGURATION
# ==================================================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================================================================================================
# 2. CORE MATH & DESIGN TOOLS (The "Toolbox")
# ==================================================================================================
# These are the fundamental building blocks for geometric calculations and path creation.

@tool
def create_path_rectangle(x: float, y: float, width: float, height: float) -> str:
    """
    [CREATE] Creates the SVG path data 'd' for a rectangle.
    Args:
        x: The x-coordinate of the top-left corner.
        y: The y-coordinate of the top-left corner.
        width: The width of the rectangle.
        height: The height of the rectangle.
    Returns: JSON {"result": "<path_data>", "explanation": "..."}.
    """
    try:
        path_data = f"M {x} {y} H {x + width} V {y + height} H {x} Z"
        explanation = f"Created rectangle path at ({x},{y}) with width {width} and height {height}."
        return json.dumps({"result": path_data, "explanation": explanation, "error": ""})
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def create_path_circle(cx: float, cy: float, r: float) -> str:
    """
    [CREATE] Creates the SVG path data 'd' for a circle.
    Args:
        cx: The x-coordinate of the circle's center.
        cy: The y-coordinate of the circle's center.
        r: The radius of the circle.
    Returns: JSON {"result": "<path_data>", "explanation": "..."}.
    """
    try:
        path_data = f"M {cx - r},{cy} a {r},{r} 0 1,0 {2*r},0 a {r},{r} 0 1,0 {-2*r},0"
        explanation = f"Created circle path centered at ({cx},{cy}) with radius {r}."
        return json.dumps({"result": path_data, "explanation": explanation, "error": ""})
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def create_path_from_bezier_fit(points: List[Tuple[float, float]], degree: int = 3) -> str:
    """
    [CREATE] Fits a Bezier curve to points and returns the full SVG path data string (M, C/Q).
    Args:
        points: [[x1,y1],...] to fit (at least degree+1). The first point is the start point (M).
        degree: 2 (Q quadratic) or 3 (C cubic; default).
    Returns: JSON {"result": "<path_data>", "explanation": "..."}.
    """
    try:
        if len(points) < 2:
            return json.dumps({"error": "Need at least 2 points to create a path."})
        if len(points) < degree + 1:
            degree = len(points) - 1 # Downgrade degree if not enough points
            if degree < 2: # Not enough for a curve, create a line
                start_point = points[0]
                end_point = points[-1]
                path_data = f"M {start_point[0]} {start_point[1]} L {end_point[0]} {end_point[1]}"
                return json.dumps({"result": path_data, "explanation": f"Created line as fallback.", "error": ""})

        ts = np.linspace(0, 1, len(points))
        def bezier_basis(t, deg):
            return [math.comb(deg, i) * t**i * (1-t)**(deg-i) for i in range(deg+1)]
        
        A = np.array([bezier_basis(t, degree) for t in ts])
        fit_points = np.linalg.lstsq(A, np.array(points), rcond=None)[0]
        
        start_point = fit_points[0].tolist()
        bez_type = "Q" if degree == 2 else "C"
        
        path_data = f"M {start_point[0]} {start_point[1]}"
        if degree == 2:
            path_data += f" Q {fit_points[1][0]},{fit_points[1][1]} {fit_points[2][0]},{fit_points[2][1]}"
        elif degree == 3:
            path_data += f" C {fit_points[1][0]},{fit_points[1][1]} {fit_points[2][0]},{fit_points[2][1]} {fit_points[3][0]},{fit_points[3][1]}"

        explanation = f"Fitted {bez_type} Bezier and created path data."
        return json.dumps({"result": path_data, "explanation": explanation, "error": ""})
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def calculate_point_symmetric(point: Tuple[float, float], axis_index: int, axis_value: float) -> str:
    """
    [CALCULATE] Calculates a symmetric point across a vertical (x) or horizontal (y) axis.
    Args:
        point: (x, y) to reflect.
        axis_index: 0 for reflection across a vertical line (x=axis_value), 1 for a horizontal line (y=axis_value).
        axis_value: The x or y position of the axis.
    Returns: JSON {"result": [new_x, new_y], "explanation": "..."}.
    """
    try:
        x, y = point
        if axis_index == 1:
            result = [x, 2 * axis_value - y]
        elif axis_index == 0:
            result = [2 * axis_value - x, y]
        else:
            return json.dumps({"error": "Axis must be 0 or 1"})
        explanation = f"Reflected {point} to {result}."
        return json.dumps({"result": result, "explanation": explanation, "error": ""})
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def calculate_point_rotated(point: Tuple[float, float], center: Tuple[float, float], angle_deg: float) -> str:
    """
    [CALCULATE] Rotates a point around a center by a given number of degrees (positive is counter-clockwise).
    Args:
        point: (x, y) to rotate.
        center: Rotation center (x, y).
        angle_deg: Rotation angle in degrees.
    Returns: JSON {"result": [new_x, new_y], "explanation": "..."}.
    """
    try:
        cx, cy = center
        px, py = point
        rad = math.radians(angle_deg)
        new_x = cx + (px - cx) * math.cos(rad) - (py - cy) * math.sin(rad)
        new_y = cy + (px - cx) * math.sin(rad) + (py - cy) * math.cos(rad)
        result = [new_x, new_y]
        explanation = f"Rotated {point} around {center} by {angle_deg}° to {result}."
        return json.dumps({"result": result, "explanation": explanation, "error": ""})
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def calculate_intersection_line_line(line1: Tuple[Tuple[float, float], Tuple[float, float]], line2: Tuple[Tuple[float, float], Tuple[float, float]]) -> str:
    """
    [CALCULATE] Calculates the intersection point of two lines.
    Args:
        line1, line2: Defined by two points each, e.g., ((x1,y1), (x2,y2)).
    Returns: JSON {"result": [ix, iy], "explanation": "..."} or error if parallel.
    """
    try:
        (x1, y1), (x2, y2) = line1
        (x3, y3), (x4, y4) = line2
        det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(det) < 1e-6:
            return json.dumps({"error": "Lines are parallel"})
        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / det
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / det
        result = [px, py]
        explanation = f"Intersection of lines at {result}."
        return json.dumps({"result": result, "explanation": explanation, "error": ""})
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def calculate_tangent_angle(path_data: str, point_index: int = -1) -> str:
    """
    [CALCULATE] Infers the tangent angle (in degrees) at a specific command-point in an SVG path.
    Args:
        path_data: The SVG path data string (e.g., "M 10 10 C 20 20, 30 20, 40 10").
        point_index: The index of the command to find the tangent at. Defaults to -1 (the last command).
    Returns: JSON {"result": angle_deg, "explanation": "..."}.
    """
    try:
        commands = re.findall(r'([MmLlHhVvCcSsQqTtAaZz])([^MmLlHhVvCcSsQqTtAaZz]*)', path_data)
        if not commands: return json.dumps({"error": "Invalid path"})
        points = []
        last_point = [0,0]
        for cmd, params_str in commands:
            params = [float(p) for p in re.findall(r'-?[\d.]+', params_str)]
            if cmd.upper() in ['M', 'L'] and len(params) >= 2:
                last_point = params[-2:]
                points.append(last_point)
            elif cmd.upper() in ['C'] and len(params) >= 6:
                points.append(params[-4:-2])
                last_point = params[-2:]
                points.append(last_point)
            elif cmd.upper() in ['Q'] and len(params) >= 4:
                points.append(params[-4:-2])
                last_point = params[-2:]
                points.append(last_point)
        if len(points) < 2: return json.dumps({"error": "Not enough points to determine tangent"})
        p2, p1 = points[-1], points[-2]
        angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
        return json.dumps({"result": angle, "explanation": f"Tangent from {p1} to {p2} is {angle}°", "error": ""})
    except Exception as e:
        return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})

@tool
def calculate_point_on_line(line_start: Tuple[float, float], line_end: Tuple[float, float], percentage: float) -> str:
    """
    [CALCULATE] Calculates the coordinates of a point on a straight line at a given percentage of its length.
    Args:
        line_start: (x1, y1) of the start of the line.
        line_end: (x2, y2) of the end of the line.
        percentage: Position on the line, where 0.0 is the start and 1.0 is the end.
    Returns: JSON {"result": [x, y], "explanation": "..."}.
    """
    try:
        x1, y1 = line_start
        x2, y2 = line_end
        if not (0.0 <= percentage <= 1.0): return json.dumps({"error": "Percentage must be between 0.0 and 1.0."})
        result = [x1 + percentage * (x2 - x1), y1 + percentage * (y2 - y1)]
        explanation = f"Calculated point at {percentage*100}% on the line, resulting in {result}."
        return json.dumps({"result": result, "explanation": explanation, "error": ""})
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def data_sample_points_from_function(function_expr: str, x_range: Tuple[float, float], num_points: int = 20) -> str:
    """
    [DATA] Samples points from a mathematical function string.
    Args:
        function_expr: Expression with 'x' (e.g., "1 / (1 + math.exp(-x))").
        x_range: (min_x, max_x).
        num_points: Number of samples.
    Returns: JSON {"result": [[x1,y1],...], "explanation": "..."}.
    """
    try:
        safe_ns = {"math": math, "np": np, "__builtins__": {}}
        xs = np.linspace(x_range[0], x_range[1], num_points)
        points = [[float(x), eval(function_expr, dict(safe_ns, x=x))] for x in xs]
        explanation = f"Sampled {num_points} points from '{function_expr}'."
        return json.dumps({"result": points, "explanation": explanation, "error": ""})
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def path_find_intersections(path_a_data: str, path_b_data: str) -> str:
    """
    [PATH] Finds all intersection points between two SVG paths. Requires 'svgpathtools'.
    Args:
        path_a_data: The SVG 'd' attribute string for the first path.
        path_b_data: The SVG 'd' attribute string for the second path.
    Returns: JSON {"result": [[x, y], ...], "explanation": "..."}.
    """
    try:
        from svgpathtools import parse_path
        path_a = parse_path(path_a_data)
        path_b = parse_path(path_b_data)
        intersections = []
        for T1, seg1, T2, seg2 in path_a.intersect(path_b):
            p = path_a.point(T1)
            intersections.append([p.real, p.imag])
        explanation = f"Found {len(intersections)} intersection(s)."
        return json.dumps({"result": intersections, "explanation": explanation, "error": ""})
    except ImportError:
        return json.dumps({"error": "This tool requires the 'svgpathtools' library. Please install it."})
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def path_split_at_points(path_data: str, points: List[Tuple[float, float]]) -> str:
    """
    [PATH] Splits a single SVG path into multiple segments at specified (x, y) points. Requires 'svgpathtools'.
    Args:
        path_data: The SVG 'd' attribute string for the path to be split.
        points: A list of [x, y] coordinates where the path should be split.
    Returns: JSON {"result": ["<segment1_d>", "<segment2_d>", ...], "explanation": "..."}.
    """
    try:
        from svgpathtools import parse_path
        path = parse_path(path_data)
        split_params = [path.index(complex(p[0], p[1])) for p in points]
        t_points = sorted(list(set([0.0] + split_params + [1.0])))
        segments_data = [path.subpath(t_points[i], t_points[i+1]).d() for i in range(len(t_points) - 1) if abs(t_points[i] - t_points[i+1]) > 1e-9]
        explanation = f"Split path into {len(segments_data)} segments."
        return json.dumps({"result": segments_data, "explanation": explanation, "error": ""})
    except ImportError:
        return json.dumps({"error": "This tool requires the 'svgpathtools' library. Please install it."})
    except Exception as e:
        return json.dumps({"error": str(e)})

# ==================================================================================================
# 3. META & MEMORY TOOLS (The "Learning" Tools)
# ==================================================================================================

@tool
def agent_save_example(title: str, description: str, code_example: str) -> str:
    """
    [AGENT] Saves a successful tool usage example or design pattern to the agent's long-term memory (`memory.md`).
    Args:
        title: A concise title for the memory (e.g., "Creating a Smooth S-Curve").
        description: A short explanation of why this pattern is useful.
        code_example: A markdown-formatted code block showing the sequence of tool calls or the resulting path.
    Returns: A confirmation message.
    """
    try:
        memory_path = os.path.join(os.path.dirname(__file__), "memory.md")
        entry = f"\n\n## {title}\n\n**Description:** {description}\n\n**Example:**\n```markdown\n{code_example}\n```\n---\n"
        with open(memory_path, "a") as f:
            f.write(entry)
        return json.dumps({"result": f"Successfully saved learning: '{title}' to memory."})
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def agent_execute_python_code(code: str, output_variable_name: str) -> str:
    """
    [AGENT] Executes a snippet of Python code and returns the value of a specified output variable.
    This is a powerful, general-purpose calculator for any geometric or mathematical task.
    Use it when a specific tool for a calculation does not exist.
    The code runs in a restricted environment with 'math', 'numpy as np', and 'sympy as sp' available.
    Args:
        code: A string containing the Python code to execute. Must not contain imports.
        output_variable_name: The name of the variable in the code whose value should be returned.
    Returns: JSON {"result": <value>, "explanation": "Executed Python code."}.
    Example: To find the point (25, 75), use code='x=25; y=75; point=[x,y]' and output_variable_name='point'.
    """
    try:
        allowed_modules = {"math": math, "np": np, "sp": sp}
        local_scope = {}
        exec(code, allowed_modules, local_scope)
        if output_variable_name in local_scope:
            result = local_scope[output_variable_name]
            return json.dumps({"result": result, "explanation": "Successfully executed Python code.", "error": ""})
        else:
            return json.dumps({"error": f"Output variable '{output_variable_name}' not found after executing code."})
    except Exception as e:
        return json.dumps({"error": str(e)})

# List of all tools available to the agent
ALL_TOOLS = [
    create_path_rectangle,
    create_path_circle,
    create_path_from_bezier_fit,
    calculate_point_symmetric,
    calculate_point_rotated,
    calculate_intersection_line_line,
    calculate_tangent_angle,
    calculate_point_on_line,
    data_sample_points_from_function,
    path_find_intersections,
    path_split_at_points,
    agent_save_example,
    agent_execute_python_code,
]

# ==================================================================================================
# 4. GRAPH STATE (The "Memory" or "Workspace")
# ==================================================================================================

class GraphState(TypedDict):
    """Represents the state of our design process."""
    task: str
    plan: List[str]
    memory: str
    past_steps: List[Tuple[str, str]]
    current_step: int
    paths: Dict[str, str]
    next_path_id: int
    abstract_state: str
    critique: str
    width: int
    height: int

# ==================================================================================================
# 5. PROMPTS (The "Mind" of Each Node)
# ==================================================================================================

PLANNER_PROMPT = """You are an expert SVG designer serving as the 'Planner' in a design agency. Your role is to break down a complex design task into a concrete, step-by-step plan for your 'Executor' colleague.

**Your Goal:** Create a phased, high-level plan to achieve the user's design task.

**Your Tools & Process:**
1.  **Consult Memory:** Review the `memory` field for useful patterns from past work.
2.  **Decompose Task:** Break the `task` into logical stages (e.g., Stage 1: Create main structure, Stage 2: Add details).
3.  **Formulate Steps:** For each stage, define a sequence of clear, actionable steps using the available tools. Refer to paths by their ID (e.g., "split path 'path_0'").
4.  **Tool Categories:** The Executor has a categorized set of tools. Use the prefixes to guide your choice:
    - `create_path_*`: Tools that create final SVG path data for shapes (e.g., `create_path_rectangle`, `create_path_from_bezier_fit`).
    - `calculate_*`: Tools for geometric calculations that return points, angles, or numbers (e.g., `calculate_point_symmetric`).
    - `path_*`: Tools that modify or analyze existing SVG paths (e.g., `path_split_at_points`).
    - `data_*`: Tools that generate lists of points from functions (e.g., `data_sample_points_from_function`).
    - `agent_*`: Meta-tools. This includes `agent_execute_python_code`, a powerful tool that can execute arbitrary Python code for any calculation you might need. **If a specific calculation tool doesn't exist, use this one to compute the result.**
5.  **Refine from Critique:** If you receive a `critique`, use it to create a new, improved plan that addresses the feedback.

**Input:**
- `task`: The user's design request.
- `memory`: Relevant examples from the 'Tool Proficiency Playbook'.
- `critique` (optional): Feedback on a previous version of the design.
- `abstract_state`: A summary of the current geometric state of all paths.

**Output:**
You must return ONLY a JSON object with a single key, "plan", which is a list of strings. Each string is a step in the plan.

**Example Output:**
{{
  "plan": [
    "// Stage 1: Create the left wing",
    "Use data_sample_points_from_function to get points for a curve.",
    "Use create_path_from_bezier_fit with those points to create 'path_1'.",
    "// Stage 2: Create the right wing",
    "Use agent_execute_python_code to get the control points and endpoint of 'path_1'.",
    "Use calculate_point_symmetric to reflect the control points across a vertical axis at x=200.",
    "Construct the right wing path, 'path_2', from the symmetric points."
  ]
}}
"""

SUMMARIZER_PROMPT = """You are a geometric analyst. Your job is to look at a collection of SVG paths and provide a concise, abstract, human-readable summary of their key geometric properties.

**Your Goal:** Describe the provided `paths` in terms of their fundamental shapes, symmetries, and relationships.

**Your Process:**
1.  Analyze the SVG `paths` provided as a dictionary of ID-to-path_data.
2.  Identify major components and their relationships (e.g., "'path_0' is a circle, 'path_1' is a line tangent to it.").
3.  Synthesize these observations into a short, descriptive text.

**Input:**
- `paths`: A dictionary where keys are path IDs and values are the raw SVG path data strings (e.g., {{"path_0": "M 10 10...", "path_1": "M 50 50..."}}).

**Output:**
You must return ONLY a JSON object with a single key, "abstract_state", containing the textual summary.

**Example Output:**
{{
  "abstract_state": "The design consists of two paths. 'path_0' is a single, closed cubic Bezier curve forming the left half of a heart shape. 'path_1' is a straight vertical line serving as a central axis."
}}
"""

REFLECTOR_PROMPT = """You are a discerning Art Director, responsible for quality control. Your job is to critique a finished design based on the original task.

**Your Goal:** Provide constructive feedback on the final set of `paths` to guide refinement, or approve it if it meets the criteria.

**Your Process:**
1.  **Compare to Task:** Does the collection of `paths` successfully fulfill the original `task`?
2.  **Assess Quality:** Evaluate the design's aesthetic and technical qualities. Is it smooth? Is it symmetrical where it should be? Are there any obvious flaws?
3.  **Formulate Critique:**
    -   If the design is good, provide a short, positive confirmation (e.g., "Approved. The design is clean and meets all requirements.").
    -   If the design has flaws, provide clear, actionable feedback for the Planner (e.g., "The connection between 'path_0' and 'path_1' is not smooth. The plan should be revised to ensure tangent continuity.").

**Input:**
- `task`: The original user request.
- `paths`: The final dictionary of SVG paths generated by the Executor.

**Output:**
You must return ONLY a JSON object with a single key, "critique", containing your feedback.

**Example Output (for refinement):**
{{
  "critique": "The overall shape is correct, but 'path_1' is not a perfect mirror of 'path_0'. The Planner should re-calculate the symmetric control points to ensure perfect symmetry."
}}

**Example Output (for approval):**
{{
  "critique": "Approved."
}}
"""

# ==================================================================================================
# 6. GRAPH NODES (The "Specialists" in the Agency)
# ==================================================================================================

def get_llm():
    agent_config = config.get_agent_config("math_agent", "core")
    model_args = asdict(agent_config.model)
    return init_chat_model(**model_args)

def invoke_llm(prompt_template, system_message, **kwargs):
    llm = get_llm()
    prompt = prompt_template.format(**kwargs)
    messages = [SystemMessage(content=system_message), HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    
    response_text = response.content
    
    # Regex to find JSON content within a markdown block, making the 'json' keyword optional
    match = re.search(r'```(json)?\s*(\{[\s\S]*?\})\s*```', response_text, re.DOTALL)
    
    json_str = ""
    if match:
        json_str = match.group(2)
    else:
        # If no markdown block, find the first '{' and last '}' to extract the JSON object.
        start = response_text.find('{')
        end = response_text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = response_text[start:end+1]
        else:
            json_str = response_text # Fallback to original content

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        logger.error(f"Failed to decode LLM response. Original content: '{response.content}'. Attempted to parse: '{json_str}'")
        return {"error": "Invalid JSON response from LLM."}

def _get_relevant_memories(task: str, all_memories: str, top_k: int = 3) -> str:
    """
    Retrieves the most relevant memories based on keyword overlap.
    """
    if not all_memories.strip() or "No memories found" in all_memories:
        return "No relevant memories found."

    # Simple keyword extraction from task (split by space, remove common words)
    stopwords = {'a', 'an', 'the', 'is', 'in', 'it', 'to', 'for', 'of', 'with', 'on', 'and', 'design', 'create', 'a', 'stylized'}
    task_keywords = {word.lower() for word in re.findall(r'\w+', task) if word.lower() not in stopwords}

    if not task_keywords:
        return "No specific keywords in task to search memory."

    # Split memories into entries
    memories = all_memories.split('---')
    
    scored_memories = []
    for memory in memories:
        if not memory.strip():
            continue
        
        # Score based on keyword overlap
        memory_keywords = {word.lower() for word in re.findall(r'\w+', memory)}
        score = len(task_keywords.intersection(memory_keywords))
        
        if score > 0:
            scored_memories.append((score, memory))

    # Sort by score and get top_k
    scored_memories.sort(key=lambda x: x[0], reverse=True)
    
    top_memories = [mem for score, mem in scored_memories[:top_k]]

    if not top_memories:
        return "No relevant memories found."

    return "\n---\n".join(top_memories)


def memory_retriever_node(state: GraphState) -> Dict[str, Any]:
    """Loads the entire memory and retrieves the most relevant examples for the planner."""
    logger.info("NODE: Memory Retriever")
    try:
        memory_path = os.path.join(os.path.dirname(__file__), "memory.md")
        with open(memory_path, "r") as f:
            all_memories = f.read()
        
        relevant_memories = _get_relevant_memories(state['task'], all_memories)
        logger.info(f"Retrieved {len(relevant_memories.split('---'))} relevant memories.")
        return {"memory": relevant_memories}

    except FileNotFoundError:
        logger.warning("memory.md not found. Starting with empty memory.")
        return {"memory": "No memories found."}

def planner_node(state: GraphState) -> Dict[str, Any]:
    """Generates a plan to accomplish the task."""
    logger.info("NODE: Planner")
    response = invoke_llm(
        PLANNER_PROMPT,
        "You are the Planner. Your output must be a JSON object with a 'plan' key.",
        task=state['task'],
        memory=state['memory'],
        critique=state.get('critique', 'N/A'),
        abstract_state=state.get('abstract_state', 'N/A')
    )
    logger.info(f"Plan generated: {response.get('plan', [])}")
    return {"plan": response.get('plan', []), "current_step": 0}

def executor_node(state: GraphState) -> Dict[str, Any]:
    """Executes a single step of the plan using the ReAct agent."""
    step = state['plan'][state['current_step']]
    logger.info(f"NODE: Executor (Step {state['current_step'] + 1}/{len(state['plan'])}): {step}")

    if step.strip().startswith("//"): # Skip comments
        return {"past_steps": state["past_steps"] + [(step, "Skipped comment.")]}

    agent_executor = create_react_agent(get_llm(), name='math_executor', tools=ALL_TOOLS)
    
    # Create a focused prompt for the executor
    executor_prompt = f"""You are a focused math tool executor. Your ONLY job is to execute the following single task by calling the appropriate tool(s). Do not try to plan or do anything else.

    Current Task: "{step}"
    
    Current Abstract State of the design:
    {state.get('abstract_state', 'N/A')}
    
    All SVG Paths in the current design (referenced by ID):
    {json.dumps(state.get('paths', {}), indent=2)}
    """
    
    messages = [HumanMessage(content=executor_prompt)]
    response = agent_executor.invoke({"messages": messages})
    
    # Extract the tool output from the last ToolMessage
    last_message = response['messages'][-1]
    result = last_message.content if isinstance(last_message, ToolMessage) else str(last_message.content)

    logger.info(f"Executor result: {result}")
    return {"past_steps": state["past_steps"] + [(step, result)]}


def summarizer_node(state: GraphState) -> Dict[str, Any]:
    """Generates a new abstract summary of the current design."""
    logger.info("NODE: Summarizer")
    if not state['paths']:
        return {"abstract_state": "The canvas is empty."}
        
    response = invoke_llm(
        SUMMARIZER_PROMPT,
        "You are the Summarizer. Your output must be a JSON object with an 'abstract_state' key.",
        paths=state['paths']
    )
    logger.info(f"New abstract state: {response.get('abstract_state', '')}")
    return {"abstract_state": response.get('abstract_state', '')}

def reflector_node(state: GraphState) -> Dict[str, Any]:
    """Critiques the final design."""
    logger.info("NODE: Reflector")
    response = invoke_llm(
        REFLECTOR_PROMPT,
        "You are the Reflector. Your output must be a JSON object with a 'critique' key.",
        task=state['task'],
        paths=state['paths']
    )
    logger.info(f"Critique: {response.get('critique', '')}")
    return {"critique": response.get('critique', '')}

def updater_node(state: GraphState) -> Dict[str, Any]:
    """
    Parses the result from the last execution step and updates the main state.
    This is the critical link between tool execution and state modification.
    It uses heuristics to interpret tool outputs and apply them to the 'paths' dictionary.
    """
    logger.info("NODE: Updater")

    if not state['past_steps']:
        return {"current_step": state.get("current_step", 0) + 1}

    current_step_str, current_result_str = state['past_steps'][-1]

    # Default return values, to be modified if updates occur
    paths = state.get('paths', {}).copy()
    next_path_id = state.get('next_path_id', 1)
    current_step = state.get("current_step", 0)
    
    updated_state = {
        "paths": paths,
        "next_path_id": next_path_id,
        "current_step": current_step + 1
    }

    try:
        current_result_data = json.loads(current_result_str)
        if not isinstance(current_result_data, dict) or (current_result_data.get("error") and current_result_data.get("error") != ""):
            logger.warning(f"Tool execution resulted in an error or invalid data: {current_result_str}")
            return updated_state
    except (json.JSONDecodeError, TypeError):
        logger.info(f"Executor result was not a JSON object, likely an informational message. Result: {current_result_str}")
        return updated_state

    tool_result = current_result_data.get("result")
    if not tool_result:
        logger.info("Tool result JSON has no 'result' key. Nothing to update.")
        return updated_state

    path_ids_in_step = re.findall(r"'(path_\d+)'", current_step_str)

    # Heuristic 1: Handle path creation from bezier fitting or direct path data
    if isinstance(tool_result, str) and tool_result.strip().startswith('M'):
        if path_ids_in_step:
            new_path_id = path_ids_in_step[0]
            paths[new_path_id] = tool_result
            new_id_num = int(new_path_id.split('_')[1])
            if new_id_num >= next_path_id:
                next_path_id = new_id_num + 1
            logger.info(f"Created or updated path '{new_path_id}'.")
            updated_state.update({"paths": paths, "next_path_id": next_path_id})

    # Heuristic 2: Handle path splitting
    elif isinstance(tool_result, list) and len(tool_result) > 0 and isinstance(tool_result[0], str) and tool_result[0].strip().startswith('M'):
        if path_ids_in_step:
            source_path_id = path_ids_in_step[0]
            if source_path_id in paths:
                del paths[source_path_id]
                logger.info(f"Removed original path '{source_path_id}' to replace with segments.")
                for segment_d in tool_result:
                    new_path_id = f"path_{next_path_id}"
                    paths[new_path_id] = segment_d
                    logger.info(f"Created new path segment '{new_path_id}'.")
                    next_path_id += 1
                updated_state.update({"paths": paths, "next_path_id": next_path_id})
    
    # Heuristic 3: Handle results from agent_execute_python_code or other calculations
    # This is more complex as the result can be anything. We assume the next step in the plan will use it.
    # For now, we just log it. The result is already in past_steps for context.
    else:
        logger.info(f"Received a non-path result from a tool. It will be available in context for subsequent steps. Result: {tool_result}")


    return updated_state


# ==================================================================================================
# 7. GRAPH EDGES (The "Decision-Making" Logic)
# ==================================================================================================

def should_continue(state: GraphState) -> str:
    """Determines if there are more steps in the plan."""
    if state['current_step'] < len(state['plan']):
        return "continue"
    else:
        return "end"

def should_refine(state: GraphState) -> str:
    """Determines if the design needs refinement based on the critique."""
    critique = state.get('critique', '').strip().lower()
    if critique.startswith("approved"):
        logger.info("Critique approved. Ending workflow.")
        return "end"
    else:
        logger.info("Critique requires refinement. Returning to Planner.")
        return "refine"

# ==================================================================================================
# 8. GRAPH ASSEMBLY (The "Agency Workflow")
# ==================================================================================================

def build_graph():
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("memory_retriever", memory_retriever_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("updater", updater_node)
    workflow.add_node("summarizer", summarizer_node)
    workflow.add_node("reflector", reflector_node)

    # Define edges
    workflow.set_entry_point("memory_retriever")
    workflow.add_edge("memory_retriever", "planner")
    workflow.add_edge("planner", "executor")
    
    workflow.add_edge("executor", "updater")
    workflow.add_edge("updater", "summarizer")

    workflow.add_conditional_edges(
        "summarizer",
        should_continue,
        {
            "continue": "executor",
            "end": "reflector",
        },
    )
    
    workflow.add_conditional_edges(
        "reflector",
        should_refine,
        {
            "refine": "planner",
            "end": END,
        },
    )

    return workflow.compile()

# ==================================================================================================
# 9. MAIN ENTRY POINT & TEST
# ==================================================================================================

async def run_math_agent_graph(task: str, width: int, height: int) -> str:
    """Runs the full design graph to generate SVG path data."""
    app = build_graph()
    
    initial_state = {
        "task": task,
        "width": width,
        "height": height,
        "past_steps": [],
        "paths": {f"path_0": f"M {width/2} {height/2}"}, # Start with one path
        "next_path_id": 1,
        "abstract_state": "A single point named 'path_0' in the center of the canvas.",
    }
    
    final_state = app.invoke(initial_state, {"recursion_limit": 25}) # Increased recursion limit

    logger.info("\n\n----- FINAL DESIGN -----")
    all_paths_data = " ".join(final_state['paths'].values())
    logger.info(f"Final Paths: {json.dumps(final_state['paths'], indent=2)}")
    logger.info(f"Final Critique: {final_state['critique']}")
    
    # Combine all path elements for the preview
    path_elements = "".join([f'<path d="{d}" fill="none" stroke="black"/>' for d in final_state['paths'].values()])
    svg_preview = (
        f"<svg width='{width}' height='{height}' xmlns='http://www.w3.org/2000/svg'>"
        f"{path_elements}</svg>"
    )

    result_payload = {
        "type": "paths_data",
        "paths": final_state['paths'],
        "explanation": final_state.get('critique', 'Completed.'),
        "svg_preview": svg_preview,
        "full_trace": final_state.get('past_steps', [])
    }
    
    return json.dumps(result_payload, indent=2)


if __name__ == "__main__":
    import asyncio
    # A complex task that requires planning, symmetry, and combining shapes.
    design_task = "Design a stylized butterfly logo. It should be perfectly symmetrical. Start with a central body, then create elegant, curved wings on one side, and mirror them to the other. The wings should be composed of at least two distinct, smooth Bezier curve segments."
    
    # This is a placeholder for running the async function.
    # In a real scenario, you would use an asyncio event loop.
    print("Starting math agent graph...")
    result_json = asyncio.run(run_math_agent_graph(design_task, width=400, height=300))
    print(result_json)
    # print("To run this, you would need to uncomment the lines in the `if __name__ == '__main__` block and run within an async context.")