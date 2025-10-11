import os
from pathlib import Path
from langchain_core.messages import ToolMessage, BaseMessage, HumanMessage, AIMessage
import base64

from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

def show_messages(update: List[BaseMessage], limit: int = 2000, num=2):
    # 颜色定义
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'human': '\033[94m',      # 蓝色
        'ai': '\033[92m',         # 绿色
        'tool': '\033[93m',       # 黄色
        'metadata': '\033[90m',   # 灰色
        'separator': '\033[95m',  # 紫色
    }
    update = update[-num:]

    def color_text(text, color_key):
        return f"{COLORS[color_key]}{text}{COLORS['reset']}"
    
    print("\n\n" + color_text("="*60, 'separator'))
    print(color_text("💬 对话历史", 'bold'))
    print(color_text("="*60, 'separator'))
    
    for i, m in enumerate(update):
        # 消息头
        message_type = m.type.capitalize() if hasattr(m, 'type') else 'Unknown'
        sender_name = f" ({m.name})" if hasattr(m, 'name') and m.name else ""
        
        print(f"\n{color_text(f'#{i+1}', 'metadata')} ", end="")
        
        if isinstance(m, HumanMessage):
            print(color_text(f"👤 Human{sender_name}:", 'human'))
            # Handle multi-modal content
            if isinstance(m.content, list):
                for part in m.content:
                    if part.get("type") == "text":
                        content = _format_content(part.get("text", ""), limit)
                        print(f"   {content}")
                    elif part.get("type") == "image_url":
                        print(f"   {color_text('[🖼️ Image included]', 'metadata')}")
            else:
                content = _format_content(m.content, limit)
                print(f"   {content}")
            
        elif isinstance(m, AIMessage):
            print(color_text(f"🤖 AI{sender_name}:", 'ai'))
            content = _format_content(m.content, limit)
            print(f"   {content}")
            
            # 工具调用
            if hasattr(m, "tool_calls") and m.tool_calls:
                for j, tc in enumerate(m.tool_calls):
                    print(f"   {color_text('🛠️ 工具调用:', 'tool')}")
                    print(f"     {color_text('名称:', 'metadata')} {tc['name']}")
                    print(f"     {color_text('参数:', 'metadata')} {tc['args']}")
                    
        elif isinstance(m, ToolMessage):
            print(color_text(f"🔧 工具结果:", 'tool'))
            content = _format_content(m.content, limit)
            print(f"   {content}")
            
    print(color_text("\n" + "="*60, 'separator'))

def _format_content(content: str, limit: int) -> str:
    """格式化内容，尝试美化JSON输出"""
    if isinstance(content, str):
        # 尝试解析JSON
        if content.strip().startswith('{') or content.strip().startswith('['):
            try:
                import json
                parsed = json.loads(content)
                # 美化JSON输出，限制长度
                formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
                if len(formatted) > limit:
                    formatted = formatted[:limit] + "..."
                return formatted
            except:
                pass
        
        # 普通文本处理
        if len(content) > limit:
            return content[:limit] + "..."
        return content
    
    # 处理非字符串内容
    content_str = str(content)
    if len(content_str) > limit:
        return content_str[:limit] + "..."
    return content_str

# 如果不支持颜色，可以使用这个简化版本
def show_messages_simple(update: List[BaseMessage], limit: int = 800):
    print("\n\n" + "="*60)
    print("💬 对话历史")
    print("="*60)
    
    for i, m in enumerate(update):
        print(f"\n#{i+1} ", end="")
        
        if isinstance(m, HumanMessage):
            if 'base64' in m.content:
                continue
            print(f"👤 Human:")
            content = _format_content(m.content, limit)
            print(f"   {content}")
            
        elif isinstance(m, AIMessage):
            print(f"🤖 AI:")
            content = _format_content(m.content, limit)
            print(f"   {content}")
            
            if hasattr(m, "tool_calls") and m.tool_calls:
                for tc in m.tool_calls:
                    print(f"   🛠️ 工具调用:")
                    print(f"     名称: {tc['name']}")
                    print(f"     参数: {tc['args']}")
                    
        elif isinstance(m, ToolMessage):
            print(f"🔧 工具结果:")
            content = _format_content(m.content, limit)
            print(f"   {content}")
            
    print("="*60)
    
# def show_messages(update: list[BaseMessage], limit: int = 800):
#     print("\n\n" + "="*50 )
#     for m in update:
#         if isinstance(m, HumanMessage):
#             # print only text
#             if 'base64' in m.content:
#                 continue
#             print(f"  [{m.type}] {m.name or ''}: {m.content[:limit]}")
#             continue
#         if isinstance(m, AIMessage):
#             print(f"  [{m.type}] {m.name or ''}: {m.content[:limit]}")
#         if hasattr(m, "tool_calls") and m.tool_calls:
#             for tc in m.tool_calls:
#                 print(f"  [tool-call] {tc['name']}({tc['args']})")
#         if isinstance(m, ToolMessage):
#             print(f"  [tool-result] {m.content[:limit]}")     
   