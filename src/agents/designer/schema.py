from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage
from enum import Enum
from typing import (
    Annotated,
    List,
    Optional,
    Any,
    Callable,
    Literal,
    Union,
    cast,
)

from pydantic import BaseModel, Field

class UserSpecifiedNoteHelperFormat(BaseModel):
    """
    user指定的课堂笔记格式
    """
    language: str = Field(default="中文", description="课堂笔记的语言")
    format: str = Field(default="markdown", description="课堂笔记的格式")
    requirements: str = Field(default="", description="用户对课堂笔记的特殊需求")
    
def _add_list_str(current: List[str], new: List[str]) -> List[str]:
    return current + new

def _add_str(current: str, new: str) -> str:
    return current + new

class NoteHelperState(BaseModel):
    final_notes: str = ""
    topic: str = ""
    user_specified_format: UserSpecifiedNoteHelperFormat = UserSpecifiedNoteHelperFormat()
    # 新增：从 Refiner Agent 传入的行业领域（默认空）
    industry_domain: str = ""

    # 历史分段摘要
    history_summaries: Annotated[List[str], _add_list_str] = []
    # 和助教的互动历史
    interaction_history: Annotated[List[str], _add_list_str] = []
    # 已累积的原始 ASR
    currimulate_asr_result: Annotated[str, _add_str] = []

    # 下一段要追加的 ASR（由外部调用者写入）
    current_asr_chunk: str = ""
    current_message: str = ""
    
    messages: Annotated[list[AnyMessage], add_messages] = []
    asr_end: bool = False
  
    # messages: Annotated[list[AnyMessage], add_messages]
