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
from .math_tools import math_agent, _init_math_agent, MATH_SYSTEM_PROMPT, MATH_PROMPT, parse_math_agent_response, MATH_AGENT_PROFILE
from .edit_canvas_tools import edit_agent, _init_edit_agent, edit_agent_with_tool
# from .design_tools import design_agent, DESIGN_SYSTEM_PROMPT, DESIGN_IN_LOOP_PROMPT
from src.utils.print_utils import show_messages
import json
import re, glob
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TTF_FILES = glob.glob('resource/*.ttf')

@tool
def draw_text_paths(width: int=400, height: int=400, padding: int=20, text: str = "ICSML", ttf_file: str = "") -> str:
    """
    Draws text as a group of SVG paths using a .ttf font file, returning an SVG string for the group.
    Each character is converted to a <path> element. The paths are wrapped in a <g> element.
    Args:
        font_size: Size of the font in pixels (default: 16).
        text: Text to draw (default: "ICSML").
        ttf_file: Path to the .ttf font file. If empty, a default font is used.
    Returns: JSON {"result": "<g>...</g>", "explanation": "...", "error": ""}.
    Use for rendering text as paths in logos (e.g., ICSML letters as dragon tail curves).
    Example: Draw "ICSML" with a .ttf file → SVG string with a <g> containing five <path> elements.
    """
    try:
        from fontTools.ttLib import TTFont
        from fontTools.pens.svgPathPen import SVGPathPen

        if not ttf_file:
            if not TTF_FILES:
                return json.dumps({"error": "No .ttf files found in resource/ directory."})
            ttf_file = TTF_FILES[0]
            logger.info(f"ttf_file is empty, using default ttf file: {ttf_file}")

        try:
            font = TTFont(ttf_file)
        except Exception as e:
            return json.dumps({"error": f"Failed to load .ttf file '{ttf_file}': {str(e)}"})

        glyph_set = font.getGlyphSet()
        units_per_em = font['head'].unitsPerEm
        if units_per_em == 0:
            return json.dumps({"error": "Invalid unitsPerEm value in font."})
        
        font_size = int(min((width - padding * 2) / len(text), height-padding*2)) 
        scale = font_size / units_per_em
        
        # Robustly get ascender value for baseline calculation
        if 'OS/2' in font and hasattr(font['OS/2'], 'sTypoAscender'):
            ascender = font['OS/2'].sTypoAscender
        elif 'hhea' in font and hasattr(font['hhea'], 'ascent'):
            ascender = font['hhea'].ascent
        else:
            ascender = font['head'].yMax # Fallback to bounding box yMax

        y_baseline = ascender * scale + (height - font_size) / 2 + padding

        x_offset = (width - font_size * len(text) * 0.7) / 2 + padding
        paths = []

        for char in text:
            glyph_name = font.getBestCmap().get(ord(char))
            if not glyph_name:
                logger.warning(f"Glyph for character '{char}' not found in font '{ttf_file}'. Skipping.")
                continue

            glyph = glyph_set[glyph_name]
            pen = SVGPathPen(glyph_set)
            glyph.draw(pen)
            path_data = pen.getCommands()

            if path_data:
                transform = f"translate({x_offset}, {y_baseline}) scale({scale}, {-scale})"
                paths.append(f'<path transform="{transform}" d="{path_data}" fill="black" stroke="black"/>')
            
            advance_width = font['hmtx'].metrics.get(glyph_name, (0, 0))[0] * scale
            x_offset += advance_width

        if not paths:
            return json.dumps({"error": "Could not generate any paths for the given text."})

        grouped_paths = f"<g>{''.join(paths)}</g>"
        explanation = f"Rendered '{text}' as {len(paths)} SVG paths using font '{ttf_file}', scaled to size {font_size}."

        return json.dumps({"type": "svg_string", "value": grouped_paths, "explanation": explanation, "error": ""})
    except Exception as e:
        return json.dumps({"error": f"An unexpected error occurred in draw_font: {str(e)}"})
 

# ---------- 几何工具 ----------
@tool
def draw_line(x1: int, y1: int, x2: int, y2: int,
              color: str = "black", width: int = 2) -> str:
    """
    Draw a straight line on the canvas and return the SVG string.
    Args:
        x1: int, the x-coordinate of the start point of the line
        y1: int, the y-coordinate of the start point of the line
        x2: int, the x-coordinate of the end point of the line
        y2: int, the y-coordinate of the end point of the line
        color: str, the color of the line
        width: int, the width of the line
    """
    line = svgwrite.shapes.Line(start=(x1, y1), end=(x2, y2),
                                stroke=color, stroke_width=width)
    return json.dumps({"type": "svg_string", "value": line.tostring()})


@tool
def draw_circle(x: int, y: int, radius: int,
                color: str = "black", fill: str = "none") -> str:
    """
    Draw a circle on the canvas and return the SVG string. `fill` can be a color name or 'none'.
    Args:
        x: int, the x-coordinate of the center of the circle
        y: int, the y-coordinate of the center of the circle
        radius: int, the radius of the circle
        color: str, the color of the circle
        fill: str, the fill color of the circle
    """
    circ = svgwrite.shapes.Circle(center=(x, y), r=radius,
                                  stroke=color, fill=fill)
    return json.dumps({"type": "svg_string", "value": circ.tostring()})


@tool
def draw_rectangle(x: int, y: int, width: int, height: int,
                   color: str = "black", fill: str = "none") -> str:
    """
    Draw a rectangle on the canvas and return the SVG string. `fill` can be a color name or 'none'.
    Args:
        x: int, the x-coordinate of the top-left corner of the rectangle
        y: int, the y-coordinate of the top-left corner of the rectangle
        width: int, the width of the rectangle
        height: int, the height of the rectangle
        color: str, the color of the rectangle
        fill: str, the fill color of the rectangle
    """
    rect = svgwrite.shapes.Rect(insert=(x, y), size=(width, height),
                                 stroke=color, fill=fill)
    return json.dumps({"type": "svg_string", "value": rect.tostring()})


@tool
def draw_polygon(points: List[Tuple[int, int]],
                 color: str = "black", fill: str = "none") -> str:
    """
    Draw a closed polygon on the canvas and return the SVG string. `fill` can be a color name or 'none'.
    Args:
        points: List[Tuple[int, int]], the points of the polygon
        color: str, the color of the polygon
        fill: str, the fill color of the polygon
    """
    poly = svgwrite.shapes.Polygon(points=points,
                                   stroke=color, fill=fill)
    return json.dumps({"type": "svg_string", "value": poly.tostring()})


@tool
def draw_arc(x: int, y: int, radius: int,
             start_angle: int, end_angle: int,
             color: str = "black", width: int = 2) -> str:
    """
    Draw an arc (in degrees, 0°=east, CCW) and return the SVG path string.
    Args:
        x: int, the x-coordinate of the center of the arc
        y: int, the y-coordinate of the center of the arc
        radius: int, the radius of the arc
        start_angle: int, the start angle of the arc
        end_angle: int, the end angle of the arc
        color: str, the color of the arc
        width: int, the width of the arc
    """
    start_rad = math.radians(start_angle)
    end_rad = math.radians(end_angle)
    large = 1 if abs(end_angle - start_angle) > 180 else 0
    sweep = 1 if end_angle > start_angle else 0

    x1 = x + radius * math.cos(start_rad)
    y1 = y + radius * math.sin(start_rad)
    x2 = x + radius * math.cos(end_rad)
    y2 = y + radius * math.sin(end_rad)

    path_data = f"M {x1},{y1} A {radius},{radius} 0 {large},{sweep} {x2},{y2}"
    path = svgwrite.path.Path(d=path_data, stroke=color, fill="none", stroke_width=width)
    return json.dumps({"type": "svg_string", "value": path.tostring()})


@tool
def draw_bezier_curve(x1: int, y1: int,
                      cx1: int, cy1: int,
                      cx2: int, cy2: int,
                      x2: int, y2: int,
                      color: str = "black", width: int = 2) -> str:
    """
    Draw a Bezier curve and return the SVG path string.
    Args:
        x1: int, the x-coordinate of the start point of the Bezier curve
        y1: int, the y-coordinate of the start point of the Bezier curve
        cx1: int, the x-coordinate of the first control point of the Bezier curve
        cy1: int, the y-coordinate of the first control point of the Bezier curve
        cx2: int, the x-coordinate of the second control point of the Bezier curve
        cy2: int, the y-coordinate of the second control point of the Bezier curve
        x2: int, the x-coordinate of the end point of the Bezier curve
        y2: int, the y-coordinate of the end point of the Bezier curve
        color: str, the color of the Bezier curve
        width: int, the width of the Bezier curve
    """
    path_data = f"M {x1},{y1} C {cx1},{cy1} {cx2},{cy2} {x2},{y2}"
    path = svgwrite.path.Path(d=path_data, stroke=color,
                               fill="none", stroke_width=width)
    return json.dumps({"type": "svg_string", "value": path.tostring()})



@tool
def draw_text(x: int, y: int, text: str,
              color: str = "black", font_size: int = 16) -> str:
    """Draw text on the canvas and return the SVG string."""
    txt = svgwrite.text.Text(text, insert=(x, y), fill=color,
                              font_size=font_size, font_family="sans-serif")
    return json.dumps({"type": "svg_string", "value": txt.tostring()})


@tool
def draw_ellipse(cx: int, cy: int, rx: int, ry: int,
                 color: str = "black", fill: str = "none") -> str:
    """
    Draw an ellipse on the canvas and return the SVG string. `fill` can be a color name or 'none'.
    Args:
        cx: int, the x-coordinate of the center of the ellipse
        cy: int, the y-coordinate of the center of the ellipse
        rx: int, the radius of the ellipse
        ry: int, the radius of the ellipse
        color: str, the color of the ellipse
        fill: str, the fill color of the ellipse
    """
    elli = svgwrite.shapes.Ellipse(center=(cx, cy), r=(rx, ry),
                                   stroke=color, fill=fill)
    return json.dumps({"type": "svg_string", "value": elli.tostring()})

@tool
def draw_path(path_data: str,
              color: str = "black", fill: str = "none", width: int = 2) -> str:
    """
    Draw a complex shape using an SVG path data string and return the SVG string.
    The path_data string uses commands like:
    M x,y (moveto)
    L x,y (lineto)
    C c1x,c1y c2x,c2y x,y (curveto)
    Q c1x,c1y x,y (quadratic Bézier curve)
    A rx,ry rot large_arc_flag,sweep_flag x,y (elliptical arc)
    Z (closepath)
    Example: "M 100,100 L 900,900 C 500,900 500,100 900,100 Z"
    """
    path = svgwrite.path.Path(d=path_data, stroke=color, fill=fill, stroke_width=width)
    return json.dumps({"type": "svg_string", "value": path.tostring()})


@tool
async def calculate_path_data_with_math_agent(task_description: str, coarse_svg_path_data: Optional[str] = None, other_info: Optional[str] = None, width: Optional[int] = None, height: Optional[int] = None) -> str:
    """
    You can use this tool to calculate the complex data of svg path. It can calculate the float datapoint using powerful symmerics/rotation/translation/scaling/etc math tools. It will return the path data 
    You must use `draw_path` tool to draw the path data.
    Args:
        task_description: str, the task description
        coarse_svg_path_data: str, recommended, the coarse svg path data
        other_info: str, optional, possible settings of position, center, size, area, etc.
        width: int, optional, the width of the canvas
        height: int, optional, the height of the canvas
    Returns:
        warpped svg string: str, the complex data of svg path wrapped by <svg> tags
    """
    messages = [
        SystemMessage(content=MATH_SYSTEM_PROMPT),
        HumanMessage(content=MATH_PROMPT.format(user_instruction=task_description, coarse_svg_path_data=coarse_svg_path_data, other_info=other_info, width=width, height=height))
    ]
    # Ensure we call the math agent implementation from math_tools.py
    # The math agent is initialized in that module and exposed as `math_agent`.
    # Call its `ainvoke` method rather than attempting to call this function recursively.
    from .math_tools import math_agent as _math_agent
    state = await _math_agent.ainvoke({"messages": messages})

    path_data = parse_math_agent_response(state)

    return json.dumps({
        "type": "path_data",
        "value": path_data.data,
        "math_agent_explanation": path_data.explanation,
        "explanation": "This is SVG path data generated by the math agent; use it with draw_path().",
        "usage": "Use draw_path(path_data=this_value, color=your_color, fill=your_fill, width=your_width) to create the SVG element"
    })


# ---------- Transformation Tools ----------
def _add_transform(svg_string: str, transform: str) -> str:
    """Helper to add or append a transform to an SVG element string."""
    # Find if a transform attribute already exists
    match = re.search(r' transform="([^"]*)"', svg_string)
    if match:
        # If it exists, append the new transform
        existing_transform = match.group(1)
        new_svg_string = svg_string.replace(
            f' transform="{existing_transform}"',
            f' transform="{existing_transform} {transform}"'
        )
    else:
        # If not, add the transform attribute to the first tag
        new_svg_string = re.sub(r'(<[a-zA-Z0-9]+)', r'\1 transform="' + transform + r'"', svg_string, 1)
    return new_svg_string

@tool
def transform_translate(svg_string: str, dx: int, dy: int) -> str:
    """
    Translates (moves) an existing SVG element or group by a delta x and y.
    Args:
        svg_string: The SVG element string to transform (e.g., '<circle.../>' or '<g>...</g>').
        dx: The distance to move along the x-axis.
        dy: The distance to move along the y-axis.
    Returns: The transformed SVG element string.
    """
    transformed = _add_transform(svg_string, f"translate({dx}, {dy})")
    return json.dumps({"type": "svg_string", "value": transformed, "explanation": "translated SVG element string."})

@tool
def transform_scale(svg_string: str, sx: float, sy: Optional[float] = None) -> str:
    """
    Scales an existing SVG element or group by a factor.
    Args:
        svg_string: The SVG element string to transform.
        sx: The scaling factor for the x-axis.
        sy: The scaling factor for the y-axis. If None, it's the same as sx.
    Returns: The transformed SVG element string.
    """
    if sy is None:
        sy = sx
    transformed = _add_transform(svg_string, f"scale({sx}, {sy})")
    return json.dumps({"type": "svg_string", "value": transformed, "explanation": "scaled SVG element string."})

@tool
def transform_rotate(svg_string: str, angle: float, cx: Optional[int] = None, cy: Optional[int] = None) -> str:
    """
    Rotates an existing SVG element or group.
    Args:
        svg_string: The SVG element string to transform.
        angle: The rotation angle in degrees.
        cx: The x-coordinate of the rotation center. If None, rotates around the element's origin.
        cy: The y-coordinate of the rotation center. If None, rotates around the element's origin.
    Returns: The transformed SVG element string.
    """
    if cx is not None and cy is not None:
        transform_str = f"rotate({angle}, {cx}, {cy})"
    else:
        transform_str = f"rotate({angle})"
    transformed = _add_transform(svg_string, transform_str)
    return json.dumps({"type": "svg_string", "value": transformed, "explanation": "rotated SVG element string."})


DrawCanvasAgentTools = [
    draw_line,
    draw_circle,
    draw_rectangle,
    draw_polygon,
    draw_arc,
    draw_bezier_curve,
    draw_text_paths,
    draw_ellipse,
    draw_path,
    # edit_agent_with_tool,
    # Design-level planning and critique tools
    # design_create_plan,
    # design_reflect,
    calculate_path_data_with_math_agent,  
    transform_translate,
    transform_scale,
    transform_rotate,
    # design_agent,
    # draw_bezier_segment,
]

SYSTEM_PROMPT = """
# Role
You are an excellent SVG Drawer. Your role is to DESIGN and DRAW SVG content as `user instruction`, within the provided canvas dimensions. If the task is about `design`, you can have to breakdown the semantic meaning and draw step by step.
# Tools: 
- agent as tools:
    - `calculate_path_data_with_math_agent`: If you need to draw VIVIDLY or draw COMPLEX path/elements, Use your `Math agent` and Give it the `coarse path` or `positions and size`.  Its profile is: MATH_AGENT_PROFILE: 
    ---------------------------------
    {MATH_SYSTEM_PROMPT}
    ---------------------------------
- `draw_`: Draw base shapes or text with `draw` tools. 
- `transform_`: Scale, translate and rotate different elements with your `transform` tools. 
""".format(MATH_SYSTEM_PROMPT=MATH_SYSTEM_PROMPT[100:-100]+'...')


DRAW_PROMPT = """
user_instruction: {user_instruction}
canvas_width: {width}
canvas_height: {height}
"""

draw_agent = None

def _init_draw_agent():
    global draw_agent
    if draw_agent is None:
        agent_config = config.get_agent_config("draw_agent", "core")
        # agent = BaseAgent(agent_config)
        llm = init_chat_model(**asdict(agent_config.model))
        draw_agent = create_react_agent(llm, name='draw_agent', tools=DrawCanvasAgentTools)
        logger.info(f"🖊 [draw_agent_with_tool] draw_agent initialized")
    return draw_agent

draw_agent = _init_draw_agent()

def is_valid_svg_element(element: str) -> bool:
    return element.startswith('<') and element.endswith('>')

@tool
async def draw_agent_with_tool(task_description: str, width: int, height: int):
    """
    Draw agent with tool. You can use this tool to draw the svg code of the canvas. You will be given the task description, and your task is the draw the svg code of the canvas.
    Args:
        task_description: str, the task description
        width: int, the width of the canvas
        height: int, the height of the canvas
    Returns:
        new_svg: str, the entire svg code of the canvas with width and height
    """
    logger.info(f"🖊 [draw_agent_with_tool] task_description: {task_description}, width: {width}, height: {height}")
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=DRAW_PROMPT.format(user_instruction=task_description, width=width, height=height))
    ]
    state = await draw_agent.ainvoke({"messages": messages})
    show_messages(state.get("messages", []))

    # Fallback to assembling from tool calls if the final message is not a complete SVG.
    svg_segments = []
    for message in state.get("messages", []):
        if isinstance(message, ToolMessage):
            data = json.loads(message.content)
            if data.get("type") == "svg_string":
                svg_segments.append(data.get("value"))
    
    # new_svg = "".join(svg_segments) if svg_segments else messages_list[-1].content
    new_svg = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">{"".join(svg_segments)}</svg>'
    if isinstance(state.get("messages", [])[-1], AIMessage):
        explanation = "[draw agent] " + state.get("messages", [])[-1].content
    else:
        explanation = "[draw agent] draw the svg code of the canvas."

    # Final assembly: ensure the output is always a complete SVG document.
    return json.dumps({"type": "draw_result", "value": new_svg, "explanation": explanation})

async def run_draw_agent_with_tool(task_description: str, width: int, height: int):
    """
    Draw agent with tool. You can use this tool to draw the svg code of the canvas. You will be given the task description, and your task is the draw the svg code of the canvas.
    """
    logger.info(f"🖊 [draw_agent_with_tool] task_description: {task_description}, width: {width}, height: {height}")
    messages = {"messages": [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=DRAW_PROMPT.format(user_instruction=task_description, width=width, height=height) + "After your execution, please give me some feedback on your tools, are there useful? What other tools do you want? And specify the math tools you want to use, comment on the current math tools you used, is it useful?")
    ]}
    state = await draw_agent.ainvoke(messages)
    show_messages(state.get("messages", []), limit=-1)
    messages_list = state.get("messages", [])
    # extract the tool message with type "svg_string"
    svg_segments = []
    for message in messages_list:
        if isinstance(message, ToolMessage):
            data = json.loads(message.content)
            if data.get("type") == "svg_string":
                svg_segments.append(data.get("value"))
    
    # new_svg = "".join(svg_segments) if svg_segments else messages_list[-1].content
    new_svg = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">{"".join(svg_segments)}</svg>'
    import uuid
    output_dir = "output/test_draw/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir + uuid.uuid4().hex + ".svg"
    open(output_path, "w", encoding="utf-8").write(new_svg)
    logger.info(f"🖊 [draw_agent_with_tool] saved to {output_path}")
    return new_svg

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_draw_agent_with_tool("""A vibrant, interconnected wireframe globe with glowing nodes and lines, representing a global network. The acronym 'ICSML' is centrally placed within the globe, accompanied by the year '2024'. Below the globe, a stylized architectural gate, potentially referencing the host university, forms a base.""", 400, 400))