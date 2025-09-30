import os
from pathlib import Path
from langchain_core.messages import ToolMessage, BaseMessage, HumanMessage, AIMessage
import base64


def show_messages(update: list[BaseMessage], limit: int = 800):
    print("\n\n" + "="*100 )
    for m in update:
        if isinstance(m, HumanMessage):
            # print only text
            if 'base64' in m.content:
                continue
            print(f"  [{m.type}] {m.name or ''}: {m.content[:limit]}")
            continue
        if isinstance(m, AIMessage):
            print(f"  [{m.type}] {m.name or ''}: {m.content[:limit]}")
        if hasattr(m, "tool_calls") and m.tool_calls:
            for tc in m.tool_calls:
                print(f"  [tool-call] {tc['name']}({tc['args']})")
        if isinstance(m, ToolMessage):
            print(f"  [tool-result] {m.content[:limit]}")     
   