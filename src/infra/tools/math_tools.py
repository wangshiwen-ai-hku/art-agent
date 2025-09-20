from langchain_core.tools import tool
import numexpr
import numpy as np

@tool
def calculator(expression: str) -> str:
    """
    Calculates the result of a mathematical expression using a safe evaluator.
    Use this to perform any mathematical calculations needed to determine coordinates, angles, or dimensions for SVG paths.
    The expression can contain numbers, standard operators, and functions from the math module (e.g., 'math.sin', 'math.pi').
    Example: "200 + 150 * math.cos(math.pi / 3)"
    """
    try:
        # Using numexpr for safe evaluation of mathematical expressions.
        # We provide a limited context with the 'math' module.
        local_dict = {"math": __import__("math")}
        result = numexpr.evaluate(expression, global_dict={}, local_dict=local_dict)
        return f"The result of '{expression}' is {result}."
    except Exception as e:
        return f"Error evaluating expression '{expression}': {e}" 
    
import json
import math
import numpy as np
from typing import Any, Dict, List, Tuple

# 可选：再封一层沙箱，限制 built-ins
SAFE_BUILTINS = {
    "len": len, "range": range, "enumerate": enumerate,
    "zip": zip, "map": map, "filter": filter, "sum": sum,
    "min": min, "max": max, "abs": abs, "round": round,
    "math": math, "np": np, "numpy": np,
}


@tool
def find_points_with_shape(shapes: List[Dict[str, Any]], func: str) -> str:
    """
    让 LLM 用 Python 代码任意计算 shapes 中的点坐标。
    shapes: 标准结构，见 docstring。
    func:   一段 Python 代码字符串，要求把最终结果放到变量 result 中；
            result 必须是 list[tuple(float, float)] 或 list[list[float]]。
    返回:   JSON 字符串，{"points": [[x1,y1],...], "error": ""}
    """
    try:
        # 1. 构造沙箱环境
        local_dict = {
            "shapes": shapes,
            "result": None,
            "__builtins__": SAFE_BUILTINS,
        }

        # 2. 执行代码
        exec(func, local_dict, local_dict)

        # 3. 取出结果
        result = local_dict.get("result")
        if result is None:
            raise ValueError("变量 result 未赋值")

        # 4. 统一转成 list[list[float]]
        points = []
        for p in result:
            if isinstance(p, (tuple, list)) and len(p) >= 2:
                points.append([float(p[0]), float(p[1])])
            else:
                raise ValueError("result 元素必须是 (x,y) 或 [x,y]")

        return json.dumps({"points": points, "error": ""}, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"points": [], "error": str(e)}, ensure_ascii=False)

import math
import numpy as np
from typing import Any, Dict, Tuple

import math
import numpy as np
from typing import List, Tuple, Optional

import math
import json
from typing import Tuple

@tool
def line_circle_intersection(
    line: Tuple[Tuple[float, float], Tuple[float, float]],  # ((x1,y1), (x2,y2))
    circle_center: Tuple[float, float],
    radius: float
) -> str:
    """
    直线由两点确定，圆由圆心和半径确定。
    返回 JSON：
      {"points": [[x1,y1], [x2,y2]]}  （0/1/2 个交点）
    或
      {"error": "..."}
    """
    try:
        (x1, y1), (x2, y2) = line
        cx, cy = circle_center
        r = radius

        # 向量
        dx = x2 - x1
        dy = y2 - y1

        # 起点到圆心向量
        fx = x1 - cx
        fy = y1 - cy

        a = dx*dx + dy*dy
        b = 2*(fx*dx + fy*dy)
        c = fx*fx + fy*fy - r*r

        discriminant = b*b - 4*a*c
        pts = []
        if discriminant < 0:
            # 无交点
            pass
        elif discriminant == 0:
            # 1 个交点
            t = -b / (2*a)
            pts.append([x1 + t*dx, y1 + t*dy])
        else:
            # 2 个交点
            sqrt_d = math.sqrt(discriminant)
            t1 = (-b - sqrt_d) / (2*a)
            t2 = (-b + sqrt_d) / (2*a)
            pts.append([x1 + t1*dx, y1 + t1*dy])
            pts.append([x1 + t2*dx, y1 + t2*dy])

        return json.dumps({"points": pts}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    
# --------------------------------------------------
# 1. 给定起点、终点、圆心、弧度，生成圆弧上的离散点
# --------------------------------------------------
@tool
def arc_points_from_center(
    start_point: Tuple[float, float],
    end_point: Tuple[float, float],
    center: Tuple[float, float],
    angle_rad: float,
    num: int = 50
) -> str:
    """
    以 center 为圆心，从 start_point 到 end_point 扫过 angle_rad 弧度，
    均匀采样 num 个点，返回 JSON {"points": [[x,y],...]}。
    angle_rad >0 逆时针，<0 顺时针。
    """
    try:
        cx, cy = center
        x0, y0 = start_point
        r = math.hypot(x0 - cx, y0 - cy)

        # 起点角度
        theta0 = math.atan2(y0 - cy, x0 - cx)
        dtheta = angle_rad / (num - 1)

        pts = []
        for i in range(num):
            theta = theta0 + i * dtheta
            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)
            pts.append([x, y])

        return json.dumps({"points": pts}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"points": [], "error": str(e)})

# --------------------------------------------------
# 2. 两条直线交点（一般式 ax+by+c=0）
# --------------------------------------------------
from typing import Tuple, List
import math
import json

@tool
def line_line_intersection(
    line1: Tuple[Tuple[float, float], Tuple[float, float]],  # ((x1,y1), (x2,y2))
    line2: Tuple[Tuple[float, float], Tuple[float, float]]
) -> str:
    """
    每条直线由两点确定 ((x1,y1), (x2,y2))。
    返回交点 JSON {"point": [x,y]} 或 {"error": "平行"}。
    """
    try:
        (x1, y1), (x2, y2) = line1
        (x3, y3), (x4, y4) = line2

        # 一般式系数  ax+by+c=0
        a1, b1, c1 = y2 - y1, x1 - x2, x2*y1 - x1*y2
        a2, b2, c2 = y4 - y3, x3 - x4, x4*y3 - x3*y4

        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-12:
            return json.dumps({"error": "Lines are parallel"}, ensure_ascii=False)

        x = (b1 * c2 - b2 * c1) / det
        y = (a2 * c1 - a1 * c2) / det
        return json.dumps({"point": [x, y]}, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
# --------------------------------------------------
# 3. 已知圆心、半径、极角（水平 0°，逆时针为正）求圆弧上一点
# --------------------------------------------------
@tool
def point_on_arc(
    center: Tuple[float, float],
    radius: float,
    angle_deg: float
) -> str:
    """
    极坐标定义：水平向右为 0°，逆时针为正。
    返回 JSON {"point": [x,y]}。
    """
    try:
        cx, cy = center
        rad = math.radians(angle_deg)
        x = cx + radius * math.cos(rad)
        y = cy + radius * math.sin(rad)
        return json.dumps({"point": [x, y]}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    
@tool
def calculate_points(expression: str, x_range: Tuple[float, float], x_num: int) -> str:
    """
    将数学表达式在指定 x 区间均匀采样，返回离散点坐标列表。
    表达式里可用：
        - 常数、运算符、括号
        - 变量 x
        - math 模块所有函数（math.sin / math.cos / math.log / math.pi …）
    入参：
        expression: str  →  例  "200 + 150 * math.sin(0.02 * x)"
        x_range   : tuple → 例  (0, 400)
        x_num     : int   → 采样点数
    返回：
        JSON 字符串
        {
          "points": [[x0,y0], [x1,y1], ...],
          "expr": "原始表达式",
          "error": ""
        }
    失败时 points 为空，error 带详情。
    """
    try:
        # 1. 构造安全命名空间：只给 math 与内置少量白名单
        safe_ns = {"math": math, "__builtins__": {}}

        # 2. 生成 x 序列
        x0, x1 = x_range
        xs = np.linspace(x0, x1, x_num)

        # 3. 逐点计算 y
        points = []
        for x in xs:
            # 把当前 x 注入命名空间
            safe_ns["x"] = float(x)
            y = float(eval(expression, safe_ns))   # 仅允许 math 与 x
            points.append([x, y])

        return json.dumps({"points": points, "expr": expression, "error": ""},
                          ensure_ascii=False)

    except Exception as e:
        return json.dumps({"points": [], "expr": expression, "error": str(e)},
                          ensure_ascii=False)
