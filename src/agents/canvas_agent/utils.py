import os
from pathlib import Path
from langchain_core.messages import ToolMessage, BaseMessage, HumanMessage

try:
    import cairosvg
except ImportError:
    cairosvg = None

def show_messages(update: list[BaseMessage]):
    for m in update:
        if isinstance(m, HumanMessage):
            continue
        print(f"  [{m.type}] {m.name or ''}: {m.content[:200]}")
        if isinstance(m, ToolMessage):
            print(f"  [tool-result] {m.content[:200]}")     
        if hasattr(m, "tool_calls") and m.tool_calls:
            for tc in m.tool_calls:
                print(f"  [tool-call] {tc['name']}({tc['args']})")
        if isinstance(m, ToolMessage):
            print(f"  [tool-result] {m.content[:200]}")     

def svg_to_png(svg_path: str) -> str:
    """
    将 svg 文件转换成同名 png 文件，保存在同一目录下。

    :param svg_path: 输入的 svg 文件路径
    :return: 生成的 png 文件绝对路径；若失败则返回空字符串
    """
    if cairosvg is None:
        print("-> `cairosvg` is not installed. Skipping PNG conversion. "
              "To install, run: pip install cairosvg")
        return ""

    svg_file = Path(svg_path).expanduser().resolve()
    if not svg_file.is_file():
        print(f"-> SVG file not found: {svg_file}")
        return ""

    png_file = svg_file.with_suffix('.png')

    try:
        # 读取 SVG 内容
        svg_content = svg_file.read_text(encoding='utf-8')
        # 生成 PNG
        cairosvg.svg2png(
            bytestring=svg_content.encode('utf-8'),
            write_to=str(png_file),
            background_color="white"
        )
        print(f"-> PNG saved to: {png_file}")
        return str(png_file)
    except Exception as e:
        print(f"-> Failed to convert SVG to PNG: {e}")
        return ""


# ----------------- 使用示例 -----------------
if __name__ == "__main__":
    svg_path = input("请输入 SVG 文件路径：").strip()
    png_path = svg_to_png(svg_path)
    if png_path:
        print("转换成功，PNG 路径：", png_path)