"""
增强的SVG绘制工具 - 专注于贝塞尔曲线和高级路径生成技术
"""

import json
import math
from typing import List, Tuple, Optional
from langchain_core.tools import tool
import svgwrite


@tool
def draw_quadratic_bezier(
    x1: int, y1: int,
    cx: int, cy: int,
    x2: int, y2: int,
    color: str = "black",
    width: int = 2,
    fill: str = "none"
) -> str:
    """
    绘制二次贝塞尔曲线（Q命令）。
    二次贝塞尔曲线由起点、一个控制点和终点定义，适合绘制简单弧线、对称曲线。
    
    Args:
        x1, y1: 起点坐标
        cx, cy: 控制点坐标
        x2, y2: 终点坐标
        color: 描边颜色
        width: 描边宽度
        fill: 填充颜色（通常为"none"）
    
    Returns:
        SVG路径字符串
    
    Example:
        draw_quadratic_bezier(100, 100, 150, 50, 200, 100)
        → 创建从(100,100)到(200,100)的平滑弧线，控制点为(150,50)
    """
    path_data = f"M {x1},{y1} Q {cx},{cy} {x2},{y2}"
    path = svgwrite.path.Path(d=path_data, stroke=color, fill=fill, stroke_width=width)
    return json.dumps({"type": "svg_string", "value": path.tostring()})


@tool
def draw_smooth_cubic_bezier(
    x1: int, y1: int,
    cx2: int, cy2: int,
    x2: int, y2: int,
    color: str = "black",
    width: int = 2,
    fill: str = "none"
) -> str:
    """
    绘制平滑的三次贝塞尔曲线（S命令）。
    S命令会自动计算第一个控制点，使曲线与前一条曲线平滑连接。
    
    Args:
        x1, y1: 起点坐标（前一条曲线的终点）
        cx2, cy2: 第二个控制点坐标
        x2, y2: 终点坐标
        color: 描边颜色
        width: 描边宽度
        fill: 填充颜色
    
    Returns:
        SVG路径字符串
    
    Note:
        这个工具假设前一条曲线存在，会自动计算平滑的控制点。
        如果这是第一条曲线，请使用 draw_bezier_curve。
    """
    path_data = f"M {x1},{y1} S {cx2},{cy2} {x2},{y2}"
    path = svgwrite.path.Path(d=path_data, stroke=color, fill=fill, stroke_width=width)
    return json.dumps({"type": "svg_string", "value": path.tostring()})


@tool
def draw_smooth_quadratic_bezier(
    x1: int, y1: int,
    x2: int, y2: int,
    color: str = "black",
    width: int = 2,
    fill: str = "none"
) -> str:
    """
    绘制平滑的二次贝塞尔曲线（T命令）。
    T命令会自动计算控制点，使曲线与前一条曲线平滑连接。
    
    Args:
        x1, y1: 起点坐标（前一条曲线的终点）
        x2, y2: 终点坐标
        color: 描边颜色
        width: 描边宽度
        fill: 填充颜色
    
    Returns:
        SVG路径字符串
    """
    path_data = f"M {x1},{y1} T {x2},{y2}"
    path = svgwrite.path.Path(d=path_data, stroke=color, fill=fill, stroke_width=width)
    return json.dumps({"type": "svg_string", "value": path.tostring()})


@tool
def draw_bezier_path_from_points(
    points: List[Tuple[int, int]],
    curve_type: str = "cubic",
    color: str = "black",
    width: int = 2,
    fill: str = "none",
    closed: bool = False
) -> str:
    """
    从一系列点生成平滑的贝塞尔曲线路径。
    自动计算控制点以创建平滑的曲线。
    
    Args:
        points: 点的列表，例如 [(100,100), (150,120), (200,100), (250,130)]
        curve_type: "cubic" (三次) 或 "quadratic" (二次)
        color: 描边颜色
        width: 描边宽度
        fill: 填充颜色
        closed: 是否闭合路径（添加Z命令）
    
    Returns:
        SVG路径字符串
    
    Algorithm:
        使用Catmull-Rom样条或简单的控制点计算来创建平滑曲线。
    """
    if len(points) < 2:
        return json.dumps({"error": "至少需要2个点"})
    
    if len(points) == 2:
        # 只有两个点，使用直线
        path_data = f"M {points[0][0]},{points[0][1]} L {points[1][0]},{points[1][1]}"
    else:
        path_parts = [f"M {points[0][0]},{points[0][1]}"]
        
        if curve_type == "cubic":
            # 三次贝塞尔曲线：为每对点计算控制点
            for i in range(len(points) - 1):
                p0 = points[max(0, i - 1)]
                p1 = points[i]
                p2 = points[i + 1]
                p3 = points[min(len(points) - 1, i + 2)]
                
                # 计算控制点（使用Catmull-Rom样条的简化版本）
                if i == 0:
                    # 第一个段：使用简单的控制点
                    cx1 = p1[0] + (p2[0] - p1[0]) * 0.3
                    cy1 = p1[1] + (p2[1] - p1[1]) * 0.3
                else:
                    # 后续段：使用S命令（平滑连接）
                    cx1 = None
                    cy1 = None
                
                cx2 = p2[0] - (p3[0] - p1[0]) * 0.3
                cy2 = p2[1] - (p3[1] - p1[1]) * 0.3
                
                if cx1 is not None:
                    path_parts.append(f"C {cx1},{cy1} {cx2},{cy2} {p2[0]},{p2[1]}")
                else:
                    path_parts.append(f"S {cx2},{cy2} {p2[0]},{p2[1]}")
        else:
            # 二次贝塞尔曲线
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i + 1]
                
                # 计算控制点（中点上方）
                cx = (p1[0] + p2[0]) / 2
                cy = (p1[1] + p2[1]) / 2 - abs(p2[0] - p1[0]) * 0.3
                
                if i == 0:
                    path_parts.append(f"Q {cx},{cy} {p2[0]},{p2[1]}")
                else:
                    path_parts.append(f"T {p2[0]},{p2[1]}")
        
        path_data = " ".join(path_parts)
    
    if closed:
        path_data += " Z"
    
    path = svgwrite.path.Path(d=path_data, stroke=color, fill=fill, stroke_width=width)
    return json.dumps({"type": "svg_string", "value": path.tostring()})


@tool
def draw_organic_shape(
    center_x: int, center_y: int,
    radius: int,
    num_points: int = 8,
    irregularity: float = 0.3,
    color: str = "black",
    fill: str = "none",
    width: int = 2
) -> str:
    """
    绘制有机形状（不规则但平滑的形状），使用贝塞尔曲线。
    适合绘制自然、有机的形状，如叶子、云朵等。
    
    Args:
        center_x, center_y: 中心点坐标
        radius: 基础半径
        num_points: 控制点数量（越多越平滑，建议6-12）
        irregularity: 不规则程度（0.0-1.0，0.3为中等）
        color: 描边颜色
        fill: 填充颜色
        width: 描边宽度
    
    Returns:
        SVG路径字符串（闭合的平滑曲线）
    """
    import random
    import numpy as np
    
    # 使用numpy的random以获得更好的控制
    rng = np.random.RandomState(42)  # 固定seed以获得可重复结果
    
    # 生成控制点
    points = []
    angle_step = 2 * math.pi / num_points
    
    for i in range(num_points):
        angle = i * angle_step
        # 添加随机变化
        r = radius * (1 + rng.uniform(-irregularity, irregularity))
        x = center_x + r * math.cos(angle)
        y = center_y + r * math.sin(angle)
        points.append((int(x), int(y)))
    
    # 使用平滑的贝塞尔曲线连接
    path_data = f"M {points[0][0]},{points[0][1]}"
    
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        p3 = points[(i + 2) % len(points)]
        
        # 计算控制点
        cx1 = p1[0] + (p2[0] - p1[0]) * 0.5
        cy1 = p1[1] + (p2[1] - p1[1]) * 0.5
        cx2 = p2[0] - (p3[0] - p1[0]) * 0.2
        cy2 = p2[1] - (p3[1] - p1[1]) * 0.2
        
        if i == 0:
            path_data += f" C {cx1},{cy1} {cx2},{cy2} {p2[0]},{p2[1]}"
        else:
            path_data += f" S {cx2},{cy2} {p2[0]},{p2[1]}"
    
    path_data += " Z"
    
    path = svgwrite.path.Path(d=path_data, stroke=color, fill=fill, stroke_width=width)
    return json.dumps({"type": "svg_string", "value": path.tostring()})


@tool
def draw_smooth_curve_through_points(
    points: List[Tuple[int, int]],
    tension: float = 0.5,
    color: str = "black",
    width: int = 2,
    fill: str = "none"
) -> str:
    """
    使用张力参数绘制通过所有点的平滑曲线。
    使用Catmull-Rom样条算法，可以控制曲线的"张力"。
    
    Args:
        points: 点列表，曲线必须通过这些点
        tension: 张力参数（0.0-1.0），0.5为标准，越小越紧，越大越松
        color: 描边颜色
        width: 描边宽度
        fill: 填充颜色
    
    Returns:
        SVG路径字符串
    
    Algorithm:
        使用Catmull-Rom样条转换为贝塞尔曲线，确保曲线精确通过所有给定点。
    """
    if len(points) < 2:
        return json.dumps({"error": "至少需要2个点"})
    
    if len(points) == 2:
        path_data = f"M {points[0][0]},{points[0][1]} L {points[1][0]},{points[1][1]}"
    else:
        path_parts = [f"M {points[0][0]},{points[0][1]}"]
        
        for i in range(len(points) - 1):
            p0 = points[max(0, i - 1)]
            p1 = points[i]
            p2 = points[i + 1]
            p3 = points[min(len(points) - 1, i + 2)]
            
            # Catmull-Rom到贝塞尔转换
            t = tension
            cx1 = p1[0] + (p2[0] - p0[0]) * t / 6
            cy1 = p1[1] + (p2[1] - p0[1]) * t / 6
            cx2 = p2[0] - (p3[0] - p1[0]) * t / 6
            cy2 = p2[1] - (p3[1] - p1[1]) * t / 6
            
            path_parts.append(f"C {cx1},{cy1} {cx2},{cy2} {p2[0]},{p2[1]}")
        
        path_data = " ".join(path_parts)
    
    path = svgwrite.path.Path(d=path_data, stroke=color, fill=fill, stroke_width=width)
    return json.dumps({"type": "svg_string", "value": path.tostring()})


# 导出所有增强工具
ENHANCED_SVG_TOOLS = [
    draw_quadratic_bezier,
    draw_smooth_cubic_bezier,
    draw_smooth_quadratic_bezier,
    draw_bezier_path_from_points,
    draw_organic_shape,
    draw_smooth_curve_through_points,
]

