from langchain_core.tools import tool
from typing import List, Dict, Any
import base64
from io import BytesIO
from xml.etree import ElementTree as ET
import re

@tool
def convert_svg_to_structured_data_description(svg_code: str) -> Dict[str, Any]:
    """将SVG代码解析为结构化数据"""
    try:
        root = ET.fromstring(svg_code)
        structured_data = {
            "width": root.get("width", "未知"),
            "height": root.get("height", "未知"), 
            "viewBox": root.get("viewBox", "未知"),
            "elements": []
        }
        
        # 提取所有图形元素
        for elem in root.iter():
            if elem.tag.endswith(('path', 'rect', 'circle', 'ellipse', 'line', 'polygon')):
                element_data = {
                    "type": elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag,
                    "attributes": dict(elem.attrib)
                }
                structured_data["elements"].append(element_data)
                
        """根据结构化数据生成视觉描述"""
        description = f"SVG图像，尺寸: {structured_data.get('width', '未知')}x{structured_data.get('height', '未知')}\n"
        description += "包含以下元素:\n"
        
        for i, elem in enumerate(structured_data.get("elements", [])):
            desc = f"{i+1}. {elem['type']}元素: "
            if elem['type'] == 'path':
                desc += "路径图形"
            elif elem['type'] == 'rect':
                desc += f"矩形"
            elif elem['type'] == 'circle':
                desc += "圆形"
            
            description += desc + "\n"
        
        return description
    except Exception as e:
        return {"error": f"解析失败: {str(e)}"}
 
@tool 
def convert_svg_to_paths(svg_code: str) -> List[Dict]:
    """extract the paths from the svg code.
    example: <svg width="100" height="100">
                    <path d="M10,10 L50,50 Q70,30 90,90 Z" fill="red"/>
                </svg>
    return: the paths of the svg.
    example:
    [
        {
            "path_data": "M10,10 L50,50 Q70,30 90,90 Z",
            "commands": [{"type": "M", "parameters": [10, 10]}, {"type": "L", "parameters": [50, 50]}, {"type": "Q", "parameters": [70, 30, 90, 90]}, {"type": "Z"}],
            "style": {"fill": "red", "stroke": "none", "stroke_width": "1"}
        }
    ]
    """
    paths = []
    try:
        root = ET.fromstring(svg_code)
        
        for path_elem in root.findall(".//{*}path"):
            d_attr = path_elem.get("d", "")
            commands = _parse_path_commands(d_attr)
            
            path_info = {
                "path_data": d_attr,
                "commands": commands,
                "style": {
                    "fill": path_elem.get("fill", "none"),
                    "stroke": path_elem.get("stroke", "none"),
                    "stroke_width": path_elem.get("stroke-width", "1")
                }
            }
            paths.append(path_info)
            
    except Exception as e:
        return [{"error": f"路径解析失败: {str(e)}"}]
    
    return paths
    
def _parse_path_commands(d_attr: str) -> List[Dict]:
    """解析路径命令"""
    commands = []
    # 匹配SVG路径命令：M、L、C、Q、A等
    pattern = r'([MLCQTAZ])([^MLCQTAZ]*)'
    matches = re.findall(pattern, d_attr)
    
    for command, params in matches:
        cmd_info = {
            "type": command,
            "parameters": [float(p) for p in re.findall(r'[-+]?[0-9]*\.?[0-9]+', params) if p]
        }
        commands.append(cmd_info)
    
    return commands
    
@tool
def convert_path_data_to_bezier_segments(path_data: str) -> List[Dict]:
    """analyze the path data and return the bezier segments.
    example: <path d="M10,10 L50,50 Q70,30 90,90 Z" fill="red"/>
    return: the bezier segments of the path.
    example:
    [
        {
            "segment_id": 0,
            "type": "M",
            "start_point": (0, 0),
            "end_point": (10, 10),
            "curve_type": "linear"
        }
    ]
    hints: you can use the `convert_svg_to_paths` tool to get the path data.
    """
    segments = []
    commands = _parse_path_commands(path_data)["commands"]
    
    current_point = (0, 0)
    for i, cmd in enumerate(commands):
        segment = {
            "segment_id": i,
            "type": cmd["type"],
            "start_point": current_point
        }
        
        if cmd["type"] == "C":  # 三次贝塞尔曲线
            params = cmd["parameters"]
            if len(params) >= 6:
                segment.update({
                    "control_point1": (params[0], params[1]),
                    "control_point2": (params[2], params[3]), 
                    "end_point": (params[4], params[5]),
                    "curve_type": "cubic_bezier"
                })
                current_point = (params[4], params[5])
        
        elif cmd["type"] == "Q":  # 二次贝塞尔曲线
            params = cmd["parameters"]
            if len(params) >= 4:
                segment.update({
                    "control_point": (params[0], params[1]),
                    "end_point": (params[2], params[3]),
                    "curve_type": "quadratic_bezier"
                })
                current_point = (params[2], params[3])
        
        elif cmd["type"] in ["M", "L"]:  # 移动或直线
            params = cmd["parameters"]
            if len(params) >= 2:
                segment.update({
                    "end_point": (params[0], params[1]),
                    "curve_type": "linear"
                })
                current_point = (params[0], params[1])
        
        segments.append(segment)
    
    return segments
    


# 补充的关键工具
@tool
def convert_svg_to_png_base64(svg_code: str) -> str:
    """svg_code: the svg code of the path you want to draw.
        example: <svg width="100" height="100">
                    <path d="M10,10 L50,50 Q70,30 90,90 Z" fill="red"/>
                </svg>
    return: the base64 str of the png image of the picked paths.
    """
    try:
        # 使用cairosvg或其他SVG渲染库
        import cairosvg
        png_data = cairosvg.svg2png(bytestring=svg_code.encode())
        base64_str = base64.b64encode(png_data).decode('utf-8')
        return f"data:image/png;base64,{base64_str}"
    except ImportError:
        # 备选方案：返回SVG的base64
        return f"data:image/svg+xml;base64,{base64.b64encode(svg_code.encode()).decode()}"



PickPathTools = [
    # convert_svg_to_structured_data_description,
    convert_svg_to_paths,
    convert_path_data_to_bezier_segments,
    convert_svg_to_png_base64,
]