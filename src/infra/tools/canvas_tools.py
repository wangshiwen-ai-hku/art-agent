# canvas_tools.py
import logging
from typing import List, Tuple
from langchain_core.tools import tool
import svgwrite
import math

logger = logging.getLogger(__name__)


# ---------- 几何工具 ----------
@tool
def draw_line(x1: int, y1: int, x2: int, y2: int,
              color: str = "black", width: int = 2) -> str:
    """Draw a straight line on the canvas and return the SVG string."""
    line = svgwrite.shapes.Line(start=(x1, y1), end=(x2, y2),
                                stroke=color, stroke_width=width)
    return line.tostring()


@tool
def draw_circle(x: int, y: int, radius: int,
                color: str = "black", fill: str = "none") -> str:
    """Draw a circle on the canvas and return the SVG string. `fill` can be a color name or 'none'."""
    circ = svgwrite.shapes.Circle(center=(x, y), r=radius,
                                  stroke=color, fill=fill)
    return circ.tostring()


@tool
def draw_rectangle(x: int, y: int, width: int, height: int,
                   color: str = "black", fill: str = "none") -> str:
    """Draw a rectangle on the canvas and return the SVG string. `fill` can be a color name or 'none'."""
    rect = svgwrite.shapes.Rect(insert=(x, y), size=(width, height),
                                 stroke=color, fill=fill)
    return rect.tostring()


@tool
def draw_polygon(points: List[Tuple[int, int]],
                 color: str = "black", fill: str = "none") -> str:
    """Draw a closed polygon on the canvas and return the SVG string. `fill` can be a color name or 'none'."""
    poly = svgwrite.shapes.Polygon(points=points,
                                   stroke=color, fill=fill)
    return poly.tostring()


@tool
def draw_arc(x: int, y: int, radius: int,
             start_angle: int, end_angle: int,
             color: str = "black", width: int = 2) -> str:
    """
    Draw an arc (in degrees, 0°=east, CCW) and return the SVG path string.
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
    return path.tostring()


@tool
def draw_bezier_curve(x1: int, y1: int,
                      cx1: int, cy1: int,
                      cx2: int, cy2: int,
                      x2: int, y2: int,
                      color: str = "black", width: int = 2) -> str:
    """
    Draw a cubic Bézier curve and return the SVG path string:
    start=(x1,y1), control1=(cx1,cy1), control2=(cx2,cy2), end=(x2,y2)
    """
    path_data = f"M {x1},{y1} C {cx1},{cy1} {cx2},{cy2} {x2},{y2}"
    path = svgwrite.path.Path(d=path_data, stroke=color,
                               fill="none", stroke_width=width)
    return path.tostring()


@tool
def draw_text(x: int, y: int, text: str,
              color: str = "black", font_size: int = 16) -> str:
    """Draw text on the canvas and return the SVG string."""
    txt = svgwrite.text.Text(text, insert=(x, y), fill=color,
                              font_size=font_size, font_family="sans-serif")
    return txt.tostring()


@tool
def draw_ellipse(cx: int, cy: int, rx: int, ry: int,
                 color: str = "black", fill: str = "none") -> str:
    """Draw an ellipse on the canvas and return the SVG string. `fill` can be a color name or 'none'."""
    elli = svgwrite.shapes.Ellipse(center=(cx, cy), r=(rx, ry),
                                   stroke=color, fill=fill)
    return elli.tostring()

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
    return path.tostring()


CanvasAgentTools = [
    draw_line,
    draw_circle,
    draw_rectangle,
    draw_polygon,
    draw_arc,
    draw_bezier_curve,
    draw_text,
    draw_ellipse,
]