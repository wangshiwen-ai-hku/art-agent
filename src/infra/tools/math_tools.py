# Completely rewritten math_tools.py
# Changes:
# - Rewrote the entire tool system following Claude Code golden rules: Focused on a few accurate, powerful tools with distinct purposes to handle ALL calculation tasks for SVG paths (e.g., deciding Bezier segments, endpoints, types C/Q/T/S, control points).
#   - Chosen right tools: Only 5 tools total (2 design_, 2 function_, 1 auxiliary_), non-overlapping, high-impact for Bezier decisions and path complexity. Avoided redundancy (e.g., no separate line/arc tools; consolidated into intersections and transformations).
#     - design_: For symmetry and transformations to infer endpoints/control points (e.g., symmetric controls for balanced curves).
#     - function_: For curve sampling and fitting to generate/optimize points (e.g., sigmoid for smooth transitions).
#     - auxiliary_: Single tool for intersections to aid positioning without overlap.
#   - Namespacing: Used prefixes (design_, function_, auxiliary_) to group and delineate boundaries.
#   - Meaningful context: Tools return JSON with "result" (e.g., points or params), "explanation" (natural language summary), "error". Added response_format enum ("concise" or "detailed") for flexibility.
#   - Token efficiency: Limited outputs (e.g., default num_points=20); concise mode omits explanation. Helpful errors guide retries.
#   - Prompt-engineering: Descriptions explain like to a new hire, with examples, strict inputs/outputs. Tools enable human-like subdivision (e.g., intersect to find endpoint, then fit Bezier).
# - Updated MATH_SYSTEM_PROMPT: Guides agent to use tools for Bezier decisions (e.g., infer type from points, choose controls via symmetry/functions).
# - Kept agent structure but emphasized ReAct for iteration if invalid.

import subprocess
from langchain_core.tools import tool
from src.config.manager import config
from langgraph.prebuilt import create_react_agent
import logging
from dataclasses import asdict
from langchain.chat_models import init_chat_model
import json
import math
import numpy as np
import sympy as sp
import re
from typing import Tuple, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langchain_core.messages import HumanMessage, SystemMessage
from src.utils.print_utils import show_messages
import sys
import time
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# New Tools: Minimal set, promised useful for Bezier/path calc (e.g., intersect for endpoint, symmetry for controls, fit for type decision).

@tool
def design_symmetric_point(point: Tuple[float, float], axis_index: int, axis_value: float,
                           response_format: str = "concise") -> str:
    """
    Calculates a symmetric point across a line axis (e.g., for balanced Bezier controls in symmetric designs).
    Args:
        point: (x, y) to reflect.
        axis_index: 0 for x, 1 for y.
        axis_value: Position of the axis (e.g., x=50 for vertical line at x=50).
        response_format: "concise" (just result) or "detailed" (with explanation).
    Returns: JSON {"result": [new_x, new_y], "explanation": "...", "error": ""}. Use for inferring symmetric endpoints/controls in paths.
    Example: Reflect (10,20) over y=50 → symmetric point for curve balance.
    """
    try:
        x, y = point
        if axis_index == 1:
            new_y = 2 * axis_value - y
            result = [x, new_y]
        elif axis_index == 0:
            new_x = 2 * axis_value - x
            result = [new_x, y]
        else:
            return json.dumps({"error": "Axis must be 0 or 1"})
        axis_name = "x" if axis_index == 0 else "y"
        explanation = f"Reflected {point} over {axis_name}={axis_value} to {result} for symmetry."
        if response_format == "concise":
            return json.dumps({"result": result, "error": ""})
        return json.dumps({"result": result, "explanation": explanation, "error": ""})
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def design_rotate_point(point: Tuple[float, float], center: Tuple[float, float],
                        angle_deg: float, response_format: str = "concise") -> str:
    """
    Rotates a point around a center by degrees (positive CCW), useful for directional adjustments in path segments.
    Args:
        point: (x, y) to rotate.
        center: Rotation center (x, y).
        angle_deg: Rotation angle in degrees.
        response_format: "concise" or "detailed".
    Returns: JSON {"result": [new_x, new_y], "explanation": "...", "error": ""}. Use to infer next endpoint direction in complex paths.
    Example: Rotate (100,0) around (0,0) by 90° → (0,100) for curve orientation.
    """
    try:
        cx, cy = center
        px, py = point
        rad = math.radians(angle_deg)
        new_x = cx + (px - cx) * math.cos(rad) - (py - cy) * math.sin(rad)
        new_y = cy + (px - cx) * math.sin(rad) + (py - cy) * math.cos(rad)
        result = [new_x, new_y]
        
        explanation = f"Rotated {point} around {center} by {angle_deg}° to {result}."
        if response_format == "concise":
            return json.dumps({"result": result, "error": ""})
        return json.dumps({"result": result, "explanation": explanation, "error": ""})
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def function_sample_points(function_expr: str, x_range: Tuple[float, float],
                           num_points: int = 20, complexity: str = "medium", response_format: str = "concise") -> str:
    """
    Samples points from a math function (e.g., sigmoid for smooth curves), for key points in Bezier segments.
    Args:
        function_expr: Expression with 'x' (e.g., "1 / (1 + math.exp(-x))" for sigmoid).
        x_range: (min_x, max_x).
        num_points: Number of samples (default 20 for efficiency).
        complexity: "low", "medium", "high" for number of points.
        response_format: "concise" or "detailed".
    Returns: JSON {"result": [[x1,y1],...], "explanation": "...", "error": ""}. Use to generate points for fitting into Bezier.
    Example: Sigmoid from -5 to 5 → points for smooth transition curve.
    """
    num_points = {"low": 10, "medium": 20, "high": 50}.get(complexity.lower(), num_points)
    try:
        safe_ns = {"math": math, "np": np, "__builtins__": {}}
        xs = np.linspace(x_range[0], x_range[1], num_points)
        points = []
        for x in xs:
            safe_ns["x"] = x
            y = eval(function_expr, safe_ns)
            points.append([float(x), float(y)])
        
        explanation = f"Sampled {num_points} points from '{function_expr}' over {x_range}."
        if response_format == "concise":
            return json.dumps({"result": points, "error": ""})
        return json.dumps({"result": points, "explanation": explanation, "error": ""})
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def function_fit_bezier(points: List[Tuple[float, float]], degree: int = 3,
                        response_format: str = "concise") -> str:
    """
    Fits a Bezier curve to points, inferring type (Q=2, C=3) and controls; core for deciding segment type/symbol.
    Args:
        points: [[x1,y1],...] to fit (at least degree+1).
        degree: 2 (Q quadratic) or 3 (C cubic; default).
        response_format: "concise" or "detailed".
    Returns: JSON {"result": {"type": "C" or "Q", "controls": [[cx1,cy1],...], "endpoint": [ex,ey]}, "explanation": "...", "error": ""}.
    Use to choose C/Q/T/S based on fit (e.g., smooth T/S if controls align).
    Example: Fit to sigmoid points → cubic controls for complex path.
    """
    
    try:
        if len(points) < degree + 1:
            return json.dumps({"error": f"Need at least {degree+1} points"})
        
        ts = np.linspace(0, 1, len(points))
        def bezier_basis(t, deg):
            return [math.comb(deg, i) * t**i * (1-t)**(deg-i) for i in range(deg+1)]
        
        A = np.array([bezier_basis(t, degree) for t in ts])
        controls = np.linalg.lstsq(A, np.array(points), rcond=None)[0].tolist()
        bez_type = "Q" if degree == 2 else "C"
        endpoint = controls[-1]
        controls = controls[1:-1]  # Exclude start/end
        
        explanation = f"Fitted {bez_type} Bezier (degree {degree}) to {len(points)} points; controls: {controls}, endpoint: {endpoint}."
        if response_format == "concise":
            return json.dumps({"result": {"type": bez_type, "controls": controls, "endpoint": endpoint}, "error": ""})
        return json.dumps({"result": {"type": bez_type, "controls": controls, "endpoint": endpoint}, "explanation": explanation, "error": ""})
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def auxiliary_line_circle_intersection(line: Tuple[Tuple[float, float], Tuple[float, float]],
                                      circle_center: Tuple[float, float], radius: float,
                                      response_format: str = "concise") -> str:
    """
    Finds intersection of a line and circle for precise endpoint positioning.
    Returns: JSON {"result": [[x1,y1], [x2,y2]], "explanation": "...", "error": ""}.
    """
    try:
        (x1, y1), (x2, y2) = line
        cx, cy = circle_center
        dx, dy = x2 - x1, y2 - y1
        fx, fy = x1 - cx, y1 - cy
        a = dx**2 + dy**2
        b = 2 * (fx * dx + fy * dy)
        c = fx**2 + fy**2 - radius**2
        disc = b**2 - 4 * a * c
        if disc < 0:
            return json.dumps({"result": [], "error": "No intersection"})
        t1 = (-b + math.sqrt(disc)) / (2 * a)
        t2 = (-b - math.sqrt(disc)) / (2 * a)
        points = [[x1 + t1 * dx, y1 + t1 * dy], [x1 + t2 * dx, y1 + t2 * dy]] if disc > 0 else [[x1 + t1 * dx, y1 + t1 * dy]]
        explanation = f"Intersected line {line} with circle at {circle_center}, radius {radius}."
        return json.dumps({"result": points, "explanation": explanation if response_format == "detailed" else "", "error": ""})
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def auxiliary_line_line_intersection(line1: Tuple[Tuple[float, float], Tuple[float, float]],
                                     line2: Tuple[Tuple[float, float], Tuple[float, float]],
                                     response_format: str = "concise") -> str:
    """
    Calculates intersection of two lines (for auxiliary positioning of endpoints/controls).
    Args:
        line1, line2: ((x1,y1), (x2,y2)) each.
        response_format: "concise" or "detailed".
    Returns: JSON {"result": [ix, iy], "explanation": "...", "error": ""} or error if parallel.
    Use as helper for locating next segment endpoints without grids.
    Example: Intersect guidelines → precise endpoint for Bezier start.
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
        
        explanation = f"Intersection of lines {line1} and {line2} at {result}."
        if response_format == "concise":
            return json.dumps({"result": result, "error": ""})
        return json.dumps({"result": result, "explanation": explanation, "error": ""})
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def design_infer_tangent(coarse_path: str, point_index: int = -1,
                         response_format: str = "concise") -> str:
    """
    Infers tangent angle (degrees, CCW) at a point in a coarse SVG path (last point if index=-1).
    Args:
        coarse_path: The SVG path data string (e.g., "M 10 10 C 20 20, 30 20, 40 10").
        point_index: The index of the command to find the tangent at. Defaults to -1 (the last command).
        response_format: "concise" or "detailed".
    Returns: JSON {"result": angle_deg, "explanation": "...", "error": ""}.
    Use to guide the next segment's direction.
    Example: Infer tangent on "M0,0 L10,10" -> 45 degrees.
    """
    try:
        # Regex to find commands and their parameters, allowing for spaces and commas
        commands = re.findall(r'([MmLlHhVvCcSsQqTtAaZz])([^MmLlHhVvCcSsQqTtAaZz]*)', coarse_path)
        if not commands:
            return json.dumps({"error": "Invalid or empty coarse path"})

        if point_index < 0:
            point_index += len(commands)

        if not (0 <= point_index < len(commands)):
            return json.dumps({"error": f"Point index {point_index} is out of bounds for {len(commands)} commands."})

        current_point = [0.0, 0.0]
        path_points = []

        for i, (cmd, params_str) in enumerate(commands):
            # Regex to extract numbers (integers or floats)
            params = [float(p) for p in re.findall(r'-?[\d.]+', params_str)]
            
            cmd_upper = cmd.upper()
            is_relative = cmd.islower()
            
            if cmd_upper == 'M' and params:
                current_point = params[:2]
                if is_relative and i > 0:
                    prev_point = path_points[-1] if path_points else [0,0]
                    current_point = [current_point[0] + prev_point[0], current_point[1] + prev_point[1]]
                path_points.append(current_point)
            elif cmd_upper == 'L' and params:
                prev_point = current_point
                end_point = [params[0] + prev_point[0], params[1] + prev_point[1]] if is_relative else params[:2]
                current_point = end_point
                path_points.append(current_point)
            elif cmd_upper == 'C' and len(params) >= 6:
                prev_point = current_point
                cp2 = [params[2] + prev_point[0], params[3] + prev_point[1]] if is_relative else params[2:4]
                end_point = [params[4] + prev_point[0], params[5] + prev_point[1]] if is_relative else params[4:6]
                current_point = end_point
                path_points.append(current_point)
            elif cmd_upper == 'Q' and len(params) >= 4:
                prev_point = current_point
                cp1 = [params[0] + prev_point[0], params[1] + prev_point[1]] if is_relative else params[0:2]
                end_point = [params[2] + prev_point[0], params[3] + prev_point[1]] if is_relative else params[2:4]
                current_point = end_point
                path_points.append(current_point)

            if i == point_index:
                break
        
        if not path_points or len(path_points) < 2 and cmd.upper() not in ['C', 'Q']:
             return json.dumps({"error": "Not enough points to determine tangent. Need at least two points or a curve command."})

        # Determine points for tangent calculation
        cmd, params_str = commands[point_index]
        params = [float(p) for p in re.findall(r'-?[\d.]+', params_str)]
        cmd_upper = cmd.upper()
        
        p2 = path_points[-1]
        p1 = None

        if cmd_upper == 'C' and len(params) >= 6:
            p1 = params[2:4] # Second control point
        elif cmd_upper == 'Q' and len(params) >= 4:
            p1 = params[0:2] # The only control point
        elif len(path_points) >= 2:
            p1 = path_points[-2]
        else:
            return json.dumps({"error": f"Cannot determine tangent for command '{cmd}' at index {point_index}."})
            
        if cmd.islower() and p1 is not None and len(path_points) > 1:
             p1 = [p1[0] + path_points[-2][0], p1[1] + path_points[-2][1]]

        if p1 is None:
            return json.dumps({"error": "Could not determine prior point for tangent calculation."})

        angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
        explanation = f"Tangent at point {p2} from {p1}: {angle}°."
        return json.dumps({"result": angle, "explanation": explanation if response_format == "detailed" else "", "error": ""})

    except Exception as e:
        return json.dumps({"error": f"An unexpected error occurred in design_infer_tangent: {str(e)}"})

@tool
def get_point_on_line(line_start: Tuple[float, float], line_end: Tuple[float, float],
                             percentage: float, response_format: str = "concise") -> str:
    """
    Calculates the coordinates of a point on a straight line segment at a given percentage of its length.
    Args:
        line_start: (x1, y1) coordinates of the start of the line.
        line_end: (x2, y2) coordinates of the end of the line.
        percentage: The position of the point on the line, where 0.0 is the start point and 1.0 is the end point.
        response_format: "concise" (just result) or "detailed" (with explanation).
    Returns: JSON {"result": [x, y], "explanation": "...", "error": ""}. Use for precise positioning on a line.
    Example: Find the midpoint of a line from (0,0) to (100,100) by using percentage=0.5.
    """
    try:
        x1, y1 = line_start
        x2, y2 = line_end
        
        if not (0.0 <= percentage <= 1.0):
            return json.dumps({"error": "Percentage must be between 0.0 and 1.0."})

        new_x = x1 + percentage * (x2 - x1)
        new_y = y1 + percentage * (y2 - y1)
        result = [new_x, new_y]
        
        explanation = f"Calculated point at {percentage*100}% on the line from {line_start} to {line_end}, resulting in {result}."
        if response_format == "concise":
            return json.dumps({"result": result, "error": ""})
        return json.dumps({"result": result, "explanation": explanation, "error": ""})
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def admin_critique_math_tool_for_me(tool_name: str, critique: str) -> str:
    """
        WHAT'S NEW: ! Create a math tool for yourself with PYTHON CODE. And exec it.
        Args:
            tool_name: str, the name of the tool
            critique: str, the critique of the tool
        Returns:
            result: str, the result of the tool
    """
    return json.dumps({"result": tool_name, "critique": critique})

@tool
def admin_create_math_tool_for_yourself(tool_name: str, tool_code: str, tool_description: str) -> str:
    """
    WHAT'S NEW: ! Create a math tool for yourself with PYTHON CODE. And exec it.
    Args:
        tool_name: str, the name of the tool
        tool_code: str, the code of the tool
    Returns:
        result: str, the result of the tool
    """
    return json.dumps({"result": tool_name, "tool_code": tool_code, "tool_description": tool_description})

@tool
def admin_exec_math_tool_for_yourself(tool_name: str, tool_code: str, input_params_json_str: str, output_variable_name: Optional[str] = None) -> str:
    """
    Executes Python code and optionally captures a result from a specified output variable.
    Args:
        tool_name: str, the name of the tool (for logging/identification).
        tool_code: str, the Python code to execute. include all imports and functions you need.
        input_params_json_str: json_str, a JSON string representing a dictionary of initial global variables
                                or context for the executed `tool_code`. Must be loaded by json.loads.
        output_variable_name: Optional[str], the name of a variable in the executed `tool_code`'s global scope
                                            whose value should be captured as the result. If None, no explicit result is captured.
    Returns:
        JSON {"tool_name": ..., "tool_code": ..., "input_params_dict": ..., "result": ..., "error": ""}.
        The "result" key will contain the value of `output_variable_name` if specified and found, otherwise a success message.
    """
    try:
        global_scope = json.loads(input_params_json_str)
        exec(tool_code, global_scope)

        captured_result = None
        if output_variable_name and output_variable_name in global_scope:
            captured_result = global_scope[output_variable_name]
        elif output_variable_name:
            return json.dumps({ "tool_name": tool_name, "tool_code": tool_code, "error": f"Output variable '{output_variable_name}' not found in executed code's scope."})

        return json.dumps({"tool_name": tool_name, "tool_code": tool_code, "input_params_dict": global_scope, "result": captured_result, "error": ""})
    except Exception as e:
        return json.dumps({ "tool_name": tool_name, "tool_code": tool_code, "error": str(e)})

import importlib
@tool
def admin_list_libiaries() -> str:
    """
    WHAT'S NEW: ! List the libiaries you have.
    Returns:
        result: str, the result of the tool
    """
    list_libiaries = subprocess.check_output("pip list", shell=True).decode("utf-8")
    return json.dumps({"result": list_libiaries})

@tool
def admin_install_libiaries(libiaries: str) -> str:
    """
    WHAT'S NEW: ! Install the libiaries you want.
    Returns:
        result: str, the result of the tool
    """
    subprocess.check_output(f"pip install {libiaries}", shell=True)
    return json.dumps({"result": "Installed libiaries: " + libiaries})

TASK_INTEPRETATION_PROMPT = """
First, intepret the task description to direct you how to use tools. return precise and short step by step instructions.
task_description: {task_description}
coarse_svg_path_data: {coarse_svg_path_data}
other_info: {other_info}
width: {width}
height: {height}
"""

@tool
def inteprete_and_plan_task(task_description: str, coarse_svg_path_data: Optional[str] = None, other_info: Optional[str] = None, width: Optional[int] = None, height: Optional[int] = None) -> str:
    """
    WHAT'S NEW: Inteprete and plan the task to use tools. Don't decline the task, just plan it.
    Returns:
        result: str, the substeps of the task with your tools.
    """
    
    task_intepretion = math_agent.ainvoke({"messages": [HumanMessage(content=TASK_INTEPRETATION_PROMPT.format(task_description=task_description, coarse_svg_path_data=coarse_svg_path_data, other_info=other_info, width=width, height=height))]})
    
    return json.dumps({"task_intepretion": task_intepretion})

MathAgentTools = [
    design_symmetric_point,
    design_rotate_point,
    design_infer_tangent,

    get_point_on_line,

    function_sample_points,
    function_fit_bezier,
    auxiliary_line_circle_intersection,
    auxiliary_line_line_intersection,

    admin_critique_math_tool_for_me,
    admin_create_math_tool_for_yourself,
    admin_exec_math_tool_for_yourself,
    admin_list_libiaries,
    admin_install_libiaries,

    inteprete_and_plan_task,
]

class PathData(BaseModel):
    """
    data: pure svg path data string, example: "M 10 10 C 20 20, 30 20, 40 10"
    explanation: the explanation of the path data
    """
    data: str = Field(description="")
    explanation: str = Field(description="The explanation of the path data")

MATH_SYSTEM_PROMPT = """
You are a precise math computation agent for SVG paths, acting as a specialized calculator. Your purpose is to solve focused, geometric sub-problems posed by another agent. You do not handle broad design tasks.

Your Profile:
- Identity: A computational geometry expert.
- Behavior: You are precise, tool-driven, and focused. You do not engage in creative design. You only solve the mathematical sub-problem you are given.
- Input: You receive specific computational tasks (e.g., "calculate a symmetric point", "find the intersection of two lines", "generate a bezier curve through these points").
- Output: You must ONLY return a JSON object with "data" (the SVG path data string) and "explanation" (a summary of how you calculated it).

Workflow:
1.  **Analyze Request**: Understand the specific, isolated geometric calculation being asked of you.
2.  **Decompose and Chain Tools**: If the calculation is complex, break it down into a sequence of tool calls. For example, to place a smooth curve between two symmetric points, you might first use `design_symmetric_point` to find an endpoint, then `function_sample_points` to generate points for a shape, and finally `function_fit_bezier` to create the curve.
3.  **Handle Ambiguity**: If a task is too broad or lacks necessary numerical details (e.g., "make a pretty curve"), you must not try to invent parameters. Instead, return an error in your final JSON output explaining what specific information is missing (e.g., "Error: Cannot create curve. Missing start and end points.").
4.  **Construct Final Path**: After using tools to find all necessary points and control points, assemble the final SVG path data string.
5.  **Format Output**: Return a single JSON object as your final answer, containing the `data` string and your `explanation`.
"""

MATH_PROMPT = """
user_instruction: {user_instruction}
coarse_svg_path_data: {coarse_svg_path_data}
other_info: {other_info}
width: {width}
height: {height}
"""

math_agent = None

def parse_math_agent_response(state) -> PathData:
    """
    Robustly parse the math agent final response into a `PathData`.
    Tries multiple fallbacks because the agent may return:
      - a JSON string with keys like {"data":..., "explanation":...}
      - a JSON object with nested keys (e.g. {"result": {"data":...}})
      - a raw SVG path string starting with 'M'/'m'
      - plain text explanation containing the path or a code block

    Returns a PathData instance (may contain empty `data` and a helpful `explanation` when parsing fails).
    """
    messages = state.get("messages", [])
    # Walk messages backwards to find the last AIMessage that is not immediately followed by a ToolMessage
    for i in range(len(messages) - 1, -1, -1):
        if not isinstance(messages[i], AIMessage):
            continue
        # If next message is a tool call, this AIMessage likely initiated a tool use, skip it
        if (i < len(messages) - 1) and isinstance(messages[i+1], ToolMessage):
            continue

        content = messages[i].content
        # 1) If it's already a dict-like object
        if isinstance(content, dict):
            # Accept either top-level `data` or nested `result` with `data`
            candidate = None
            if 'data' in content:
                candidate = content
            elif 'result' in content and isinstance(content['result'], dict) and 'data' in content['result']:
                candidate = {'data': content['result']['data'], 'explanation': content.get('explanation', '')}
            if candidate:
                try:
                    return PathData(**candidate)
                except Exception:
                    # Fall through to other parsing strategies
                    pass

        # 2) If it's a string, attempt to extract JSON first
        if isinstance(content, str):
            text = content.strip()
            # Try direct JSON parse
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    if 'data' in parsed and 'explanation' in parsed:
                        return PathData(**parsed)
                    # nested result
                    if 'result' in parsed and isinstance(parsed['result'], dict) and 'data' in parsed['result']:
                        cand = {'data': parsed['result']['data'], 'explanation': parsed.get('explanation', '')}
                        return PathData(**cand)
            except Exception:
                pass

            # Try to find a JSON blob inside code fences or anywhere in the text
            json_blob_match = None
            # content may contain ```json { ... } ``` or ``` { ... } ``` blocks
            import re as _re
            m = _re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', text)
            if m:
                json_blob_match = m.group(1)
            else:
                # fallback: find first {...} that looks like a JSON object
                m2 = _re.search(r'({\s*"?data"?[\s\S]*})', text)
                if m2:
                    json_blob_match = m2.group(1)

            if json_blob_match:
                try:
                    parsed = json.loads(json_blob_match)
                    if 'data' in parsed:
                        return PathData(**{'data': parsed.get('data', ''), 'explanation': parsed.get('explanation', '')})
                except Exception:
                    pass

            # 3) If text looks like raw path data (starts with M/m or contains move + curve commands), accept it
            if text and (_re.match(r'^[Mm]\s*[-0-9.,\s]+', text) or _re.search(r'[CcQqTtSsAaLlHhVv]', text)):
                # Use the whole text as `data` if it appears path-like, and include a short explanation
                explanation = ''
                # if there is surrounding explanation, keep it as explanation
                if len(text) > 512:
                    explanation = text[:512]
                else:
                    # If text is short (likely the path itself), leave explanation empty
                    explanation = ''
                return PathData(data=text, explanation=explanation)

            # 4) Final fallback: treat the entire content as explanation and leave data empty
            short_expl = text if len(text) < 1000 else text[:1000]
            return PathData(data="", explanation=short_expl)

    # If we got here, nothing parseable was found
    logger.error("Failed to parse a valid PathData response from the math agent.")
    return PathData(data="", explanation="Error: Could not parse a valid final response from the math agent.")

def _init_math_agent():
    global math_agent
    if math_agent is None:
        agent_config = config.get_agent_config("math_agent", "core")
        llm = init_chat_model(**asdict(agent_config.model))
        math_agent = create_react_agent(llm, name='math_agent', tools=MathAgentTools)
        logger.info(f" [math_agent] math_agent initialized")
    return math_agent

math_agent = _init_math_agent()

async def calculate_path_data_with_math_agent(task_description: str, coarse_svg_path_data: Optional[str] = None, other_info: Optional[str] = None, width: Optional[int] = None, height: Optional[int] = None) -> str:
    """
    Calculate the path data with math agent.
    """
    math_agent = _init_math_agent()
    messages = [
        SystemMessage(content=MATH_SYSTEM_PROMPT),
        HumanMessage(content=MATH_PROMPT.format(user_instruction=task_description, coarse_svg_path_data=coarse_svg_path_data, other_info=other_info, width=width, height=height))
    ]
    state = await math_agent.ainvoke({"messages": messages})
    show_messages(state.get("messages", []), -1)
    data = parse_math_agent_response(state)
    svg_code = f"<svg width='{width}' height='{height}' xmlns='http://www.w3.org/2000/svg'><path d='{data.data}' /></svg>"
    output_dir = 'output/test_math'
    os.makedirs(output_dir, exist_ok=True)
    output_name = f"math_agent_output_{time.strftime('%Y%m%d_%H%M%S')}.svg"
    with open(os.path.join(output_dir, output_name), "w", encoding="utf-8") as f:
        f.write(svg_code)
    logger.info(f"Math agent output saved to {os.path.join(output_dir, output_name)}")
    return svg_code

MATH_AGENT_PROFILE = """

"""

if __name__ == "__main__":
    import asyncio
    asyncio.run(calculate_path_data_with_math_agent("Create a heart.", width=400, height=400))