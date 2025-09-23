# canvas_tools.py  追加内容（放在文件末尾即可）
import ast
import operator
import textwrap
from typing import Callable, List, Tuple
import math
from langchain_core.tools import tool
import svgwrite

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------- 内部公共绘图引擎，不暴露给 LLM --------------
def _plot_fn(f: Callable[[float], float],
             x_range: Tuple[float, float] = (-6.0, 6.0),
             samples: int = 400,
             width: int = 400,
             height: int = 200,
             padding: int = 10,
             position: Tuple[float, float] = (0, 0)) -> str:
    """
    把任意一元函数 f(x) 绘制成 SVG 路径数据字符串（不含 <svg> 标签，仅路径）
    返回可直接喂给 draw_path 的 path_data
    """
    x_min, x_max = x_range
    if x_min >= x_max:
        raise ValueError("x_range 必须是 (min, max) 且 min < max")

    graph_w, graph_h = width - 2 * padding + position[0], height - 2 * padding + position[1]

    def map_x(x: float) -> float:
        t = (x - x_min) / (x_max - x_min)
        return padding + t * graph_w

    def map_y(y: float) -> float:
        # 假设 y 归一化到 [0,1]；若超出会被线性拉伸
        return height - padding - y * graph_h

    dx = (x_max - x_min) / (samples - 1)
    pts: List[str] = []
    for i in range(samples):
        x = x_min + i * dx
        try:
            y = float(f(x))
        except Exception as exc:
            logger.warning("f(%s) 计算失败: %s", x, exc)
            y = 0.0
        pts.append(f"{map_x(x):.3f},{map_y(y):.3f}")

    return "M " + " L ".join(pts)


# -------------- 可安全执行的单表达式解析器 --------------
# 仅允许数学运算符与常用函数，防止任意代码执行
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}
_SAFE_FN = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "sqrt": math.sqrt,
    "log": math.log,
    "abs": abs,
}

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

def _eval_expr(expr: str) -> Callable[[float], float]:
    """把单变量表达式字符串编译成 f(x)。"""
    expr = textwrap.dedent(expr).strip()
    node = ast.parse(expr, mode="eval").body

    def _eval(node, x_val):
        if isinstance(node, ast.BinOp):
            op = _SAFE_OPS.get(type(node.op))
            if not op:
                raise ValueError(f"不支持的二元运算符: {type(node.op)}")
            return op(_eval(node.left, x_val), _eval(node.right, x_val))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_eval(node.operand, x_val)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("仅支持简单函数调用")
            fn = _SAFE_FN.get(node.func.id)
            if fn is None:
                raise ValueError(f"不支持的函数: {node.func.id}")
            if len(node.args) != 1:
                raise ValueError("仅支持单参数函数")
            return fn(_eval(node.args[0], x_val))
        if isinstance(node, ast.Name) and node.id == "x":
            return x_val
        if isinstance(node, ast.Num):          # py<3.8
            return node.n
        if isinstance(node, ast.Constant):     # py>=3.8
            return node.value
        raise ValueError(f"不支持的语法: {ast.dump(node)}")

    # 预编译检查
    try:
        _eval(node, 0.0)
    except Exception as exc:
        raise RuntimeError(f"表达式编译失败: {exc}")

    return lambda x_val: _eval(node, x_val)
# -------------- 正式暴露给 langchain 的两个工具 --------------
@tool
def draw_function(expr: str,
                  x_range: Tuple[float, float] = (-6.0, 6.0),
                  samples: int = 400,
                  width: int = 400,
                  height: int = 200,
                  position: Tuple[float, float] = (0, 0),
                  stroke_color: str = "#1f77b4",
                  stroke_width: int = 2) -> str:
    """
    绘制任意一元函数曲线并返回完整 SVG 字符串。
    :param expr:          函数表达式，仅含变量 x 与常用数学函数，例如 "1/(1+exp(-x))" 或 "sin(x)*exp(-x**2)"
    :param x_range:       自变量区间 (min, max)
    :param samples:       采样点数
    :param width/height:  画布尺寸
    :param stroke_color:  曲线颜色
    :param stroke_width:  线宽
    :param position:      原点/中心位置, 默认在原点
    """
    f = _eval_expr(expr)
    path_data = _plot_fn(f, x_range, samples, width, height, position)
    # 用已有工具画路径，白底矩形自己加
    return _draw_path(path_data, color=stroke_color, width=stroke_width, fill="none")


@tool
def draw_sigmoid(samples: int = 400,
                 width: int = 400,
                 height: int = 200,
                 stroke_color: str = "#1f77b4",
                 stroke_width: int = 2) -> str:
    """
    专用 sigmoid 曲线，内部直接调用 draw_function，保持与老接口兼容。
    """
    return draw_function(
        expr="1/(1+exp(-x))",
        x_range=(-6, 6),
        samples=samples,
        width=width,
        height=height,
        stroke_color=stroke_color,
        stroke_width=stroke_width
    )