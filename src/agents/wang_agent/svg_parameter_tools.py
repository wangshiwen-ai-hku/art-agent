"""
SVG参数提取和调整工具 - 从DiffVG启发的参数化操作
允许提取和微调SVG元素的参数，而不是完全重新生成
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)


def extract_svg_parameters(svg_code: str) -> Dict[str, Any]:
    """
    从SVG代码中提取关键参数。
    
    Args:
        svg_code: SVG代码字符串
    
    Returns:
        包含所有元素参数的字典：
        {
            "elements": [
                {
                    "id": "element_0",
                    "type": "path",
                    "parameters": {
                        "d": "M 100,100 L 200,200",
                        "fill": "red",
                        "stroke": "black",
                        ...
                    },
                    "control_points": [(100,100), (200,200)],  # 如果是路径
                    "position": (100, 100),  # 元素位置
                    "color": "red"  # 主要颜色
                },
                ...
            ]
        }
    """
    try:
        # 解析SVG
        root = ET.fromstring(svg_code)
        
        elements = []
        element_id = 0
        
        # 遍历所有子元素
        for child in root:
            element_info = {
                "id": f"element_{element_id}",
                "type": child.tag.split('}')[-1] if '}' in child.tag else child.tag,
                "parameters": dict(child.attrib),
                "control_points": [],
                "position": None,
                "color": None
            }
            
            # 根据元素类型提取特定参数
            tag_name = element_info["type"]
            
            if tag_name == "path":
                # 提取路径数据
                d = child.attrib.get("d", "")
                element_info["control_points"] = extract_path_points(d)
                element_info["position"] = extract_path_start_point(d)
                
            elif tag_name in ["circle", "ellipse"]:
                # 提取圆心和半径
                cx = float(child.attrib.get("cx", 0))
                cy = float(child.attrib.get("cy", 0))
                element_info["position"] = (cx, cy)
                if tag_name == "circle":
                    element_info["parameters"]["r"] = child.attrib.get("r", "0")
                else:
                    element_info["parameters"]["rx"] = child.attrib.get("rx", "0")
                    element_info["parameters"]["ry"] = child.attrib.get("ry", "0")
                    
            elif tag_name in ["rect", "polygon", "polyline"]:
                # 提取位置
                if tag_name == "rect":
                    x = float(child.attrib.get("x", 0))
                    y = float(child.attrib.get("y", 0))
                    element_info["position"] = (x, y)
                elif tag_name in ["polygon", "polyline"]:
                    points = child.attrib.get("points", "")
                    if points:
                        point_list = parse_points_string(points)
                        if point_list:
                            element_info["position"] = point_list[0]
                            element_info["control_points"] = point_list
            
            # 提取颜色
            fill = child.attrib.get("fill", "none")
            stroke = child.attrib.get("stroke", "none")
            element_info["color"] = fill if fill != "none" else stroke
            
            elements.append(element_info)
            element_id += 1
        
        return {
            "elements": elements,
            "total_elements": len(elements)
        }
    
    except Exception as e:
        logger.error(f"Error extracting SVG parameters: {e}", exc_info=True)
        return {"error": str(e), "elements": []}


def extract_path_points(path_data: str) -> List[Tuple[float, float]]:
    """从路径数据中提取所有点（包括控制点）"""
    points = []
    
    # 匹配所有数字对
    pattern = r'([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)'
    matches = re.findall(pattern, path_data)
    
    for match in matches:
        x = float(match[0])
        y = float(match[1])
        points.append((x, y))
    
    return points


def extract_path_start_point(path_data: str) -> Optional[Tuple[float, float]]:
    """提取路径的起始点（M命令后的第一个点）"""
    # 查找M命令后的第一个坐标
    match = re.search(r'M\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)', path_data)
    if match:
        return (float(match.group(1)), float(match.group(2)))
    return None


def parse_points_string(points_str: str) -> List[Tuple[float, float]]:
    """解析points属性字符串，如 "100,100 200,200 300,100" """
    points = []
    # 匹配所有坐标对
    pattern = r'([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)'
    matches = re.findall(pattern, points_str)
    for match in matches:
        points.append((float(match[0]), float(match[1])))
    return points


def adjust_path_control_points(
    svg_code: str,
    path_index: int,
    new_control_points: List[Tuple[float, float]]
) -> str:
    """
    调整路径的控制点。
    
    Args:
        svg_code: 原始SVG代码
        path_index: 路径的索引（从0开始）
        new_control_points: 新的控制点列表 [(x1,y1), (x2,y2), ...]
    
    Returns:
        调整后的SVG代码
    """
    try:
        root = ET.fromstring(svg_code)
        
        # 找到所有path元素
        paths = [elem for elem in root if elem.tag.split('}')[-1] == 'path']
        
        if path_index >= len(paths):
            logger.warning(f"Path index {path_index} out of range (total: {len(paths)})")
            return svg_code
        
        path_elem = paths[path_index]
        old_d = path_elem.attrib.get("d", "")
        
        # 重建路径数据
        if len(new_control_points) < 2:
            logger.warning("Need at least 2 points for a path")
            return svg_code
        
        # 使用简单的直线连接（可以改进为贝塞尔曲线）
        new_d = f"M {new_control_points[0][0]},{new_control_points[0][1]}"
        for point in new_control_points[1:]:
            new_d += f" L {point[0]},{point[1]}"
        
        # 如果原路径是闭合的，添加Z
        if old_d.strip().endswith('Z') or old_d.strip().endswith('z'):
            new_d += " Z"
        
        path_elem.attrib["d"] = new_d
        
        # 转换回字符串
        return ET.tostring(root, encoding='unicode')
    
    except Exception as e:
        logger.error(f"Error adjusting path control points: {e}", exc_info=True)
        return svg_code


def adjust_element_position(
    svg_code: str,
    element_index: int,
    new_x: float,
    new_y: float
) -> str:
    """
    调整元素的位置。
    
    Args:
        svg_code: 原始SVG代码
        element_index: 元素的索引
        new_x: 新的x坐标
        new_y: 新的y坐标
    
    Returns:
        调整后的SVG代码
    """
    try:
        root = ET.fromstring(svg_code)
        
        # 获取所有子元素
        elements = list(root)
        
        if element_index >= len(elements):
            logger.warning(f"Element index {element_index} out of range (total: {len(elements)})")
            return svg_code
        
        elem = elements[element_index]
        tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        
        if tag_name == "path":
            # 调整路径的起始点
            old_d = elem.attrib.get("d", "")
            start_point = extract_path_start_point(old_d)
            if start_point:
                # 计算偏移量
                offset_x = new_x - start_point[0]
                offset_y = new_y - start_point[1]
                # 调整所有点
                new_d = offset_path_points(old_d, offset_x, offset_y)
                elem.attrib["d"] = new_d
                
        elif tag_name == "circle":
            elem.attrib["cx"] = str(new_x)
            elem.attrib["cy"] = str(new_y)
            
        elif tag_name == "ellipse":
            elem.attrib["cx"] = str(new_x)
            elem.attrib["cy"] = str(new_y)
            
        elif tag_name == "rect":
            elem.attrib["x"] = str(new_x)
            elem.attrib["y"] = str(new_y)
            
        elif tag_name in ["polygon", "polyline"]:
            # 调整第一个点的位置
            points = elem.attrib.get("points", "")
            if points:
                point_list = parse_points_string(points)
                if point_list:
                    offset_x = new_x - point_list[0][0]
                    offset_y = new_y - point_list[0][1]
                    # 调整所有点
                    new_points = [(p[0] + offset_x, p[1] + offset_y) for p in point_list]
                    elem.attrib["points"] = " ".join([f"{p[0]},{p[1]}" for p in new_points])
        
        # 转换回字符串
        return ET.tostring(root, encoding='unicode')
    
    except Exception as e:
        logger.error(f"Error adjusting element position: {e}", exc_info=True)
        return svg_code


def adjust_element_color(
    svg_code: str,
    element_index: int,
    new_color: str
) -> str:
    """
    调整元素的颜色。
    
    Args:
        svg_code: 原始SVG代码
        element_index: 元素的索引
        new_color: 新的颜色值（如 "red", "#FF0000", "rgb(255,0,0)"）
    
    Returns:
        调整后的SVG代码
    """
    try:
        root = ET.fromstring(svg_code)
        
        # 获取所有子元素
        elements = list(root)
        
        if element_index >= len(elements):
            logger.warning(f"Element index {element_index} out of range (total: {len(elements)})")
            return svg_code
        
        elem = elements[element_index]
        
        # 优先设置fill，如果fill是none则设置stroke
        current_fill = elem.attrib.get("fill", "none")
        if current_fill != "none":
            elem.attrib["fill"] = new_color
        else:
            elem.attrib["stroke"] = new_color
        
        # 转换回字符串
        return ET.tostring(root, encoding='unicode')
    
    except Exception as e:
        logger.error(f"Error adjusting element color: {e}", exc_info=True)
        return svg_code


def offset_path_points(path_data: str, offset_x: float, offset_y: float) -> str:
    """偏移路径中的所有点"""
    def offset_coord(match):
        x = float(match.group(1))
        y = float(match.group(2))
        return f"{x + offset_x},{y + offset_y}"
    
    # 匹配所有坐标对并偏移
    pattern = r'([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)'
    new_path = re.sub(pattern, offset_coord, path_data)
    return new_path


# 导出为LangChain工具
from langchain_core.tools import tool


@tool
def extract_svg_parameters_tool(svg_code: str) -> str:
    """
    从SVG代码中提取关键参数（位置、颜色、控制点等）。
    返回JSON格式的参数信息。
    """
    result = extract_svg_parameters(svg_code)
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def adjust_path_control_points_tool(
    svg_code: str,
    path_index: int,
    new_control_points: str  # JSON格式的坐标列表，如 "[[100,100], [200,200]]"
) -> str:
    """
    调整SVG路径的控制点。
    
    Args:
        svg_code: SVG代码
        path_index: 路径索引（从0开始）
        new_control_points: 新的控制点，JSON格式，如 "[[100,100], [200,200]]"
    """
    try:
        points_list = json.loads(new_control_points)
        points = [tuple(p) for p in points_list]
        result = adjust_path_control_points(svg_code, path_index, points)
        return result
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def adjust_element_position_tool(
    svg_code: str,
    element_index: int,
    new_x: float,
    new_y: float
) -> str:
    """
    调整SVG元素的位置。
    
    Args:
        svg_code: SVG代码
        element_index: 元素索引（从0开始）
        new_x: 新的x坐标
        new_y: 新的y坐标
    """
    return adjust_element_position(svg_code, element_index, new_x, new_y)


@tool
def adjust_element_color_tool(
    svg_code: str,
    element_index: int,
    new_color: str
) -> str:
    """
    调整SVG元素的颜色。
    
    Args:
        svg_code: SVG代码
        element_index: 元素索引（从0开始）
        new_color: 新的颜色值（如 "red", "#FF0000"）
    """
    return adjust_element_color(svg_code, element_index, new_color)


# 导出所有工具
SVG_PARAMETER_TOOLS = [
    extract_svg_parameters_tool,
    adjust_path_control_points_tool,
    adjust_element_position_tool,
    adjust_element_color_tool,
]

