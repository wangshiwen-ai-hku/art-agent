"""PPT Agent 提示词定义，参考 deep_researcher 架构。"""

from pathlib import Path

# 提示词文件路径
PROMPTS_DIR = Path(__file__).parent / "prompt"

def _read_prompt_file(filename: str) -> str:
    """读取提示词文件内容。如果文件不存在返回占位文本，方便 CI 检测。"""
    file_path = PROMPTS_DIR / filename
    if file_path.exists():
        return file_path.read_text(encoding="utf-8")
    return f"# 提示词文件未找到: {filename}"

# PPT 生成相关提示词
PLANNER_SYSTEM_PROMPT   = _read_prompt_file("planner_system_prompt.txt")
SECTION_SYSTEM_PROMPT   = _read_prompt_file("section_system_prompt.txt")
PPT_SYSTEM_PROMPT       = _read_prompt_file("ppt_system_prompt.txt")
IMAGE_SYSTEM_PROMPT     = _read_prompt_file("image_system_prompt.txt")
DIGITAL_HUMAN_SYSTEM_PROMPT = _read_prompt_file("digital_human_system_prompt.txt")
FLASHCARD_SYSTEM_PROMPT = _read_prompt_file("flashcard_system_prompt.txt")