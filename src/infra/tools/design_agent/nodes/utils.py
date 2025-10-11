# src/infra/tools/design_agent/nodes/utils.py
import json
from enum import Enum
from typing import Dict, List
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from pydantic import BaseModel
from dataclasses import asdict

from src.config.manager import config
from src.infra.tools.design_agent.schemas import _MODEL_REGISTRY

design_llm = None

def _init_design_agent():
    agent_config = config.get_agent_config("design_agent", "core")
    model_config = agent_config.model
    global design_llm
    design_llm = init_chat_model(**asdict(model_config))
    return design_llm

design_llm = _init_design_agent()

DESIGN_SYSTEM_PROMPT = """
# ROLE
You are an award-winning brand designer. Read the **TASK** and produce the most valuable design artwork.
each `next_step` must refer the current notes and situation. Reflect the *TASK* and be coherent with it.

# Task
%s
-----------------

"""
SYSTEM_CONTENT = DESIGN_SYSTEM_PROMPT

NODE_MAP = {
    "brainstorm": "brainstorm_node",
    "compose": "compose_elements_node",
    "human": "human_in_loop_node",
    "summarize": "summarize_node",
    "reflect": "reflect_node",
    "preview_image": "preview_image_node",
    "preview_svg": "preview_svg_node",
    "end": "end",
}

NODE_MM_HINTS = {
    "plan": "Use these images (if any) as supplementary context: only for determining which high-level stages are needed and why.",
    "brainstorm": "These attached images are for reference or rough previews; use them to expand diverse visual ideas, do not finalize the composition immediately.",
    "compose": "These attached images show candidate elements or previews—select and combine elements to form a coherent composition, while using the images as example references.",
    "refine": "These attached images are intermediate compositions; refine them for clarity, balance, and implementability, and refer to these previews.",
    "reflect": "These images are generated previews: analyze whether they meet requirements, point out deficiencies, and determine if it's necessary to return to a specific step for reshaping (if so, provide clear instructions).",
    "preview_image": "Generate or explain preview images to verify if they satisfy the task; subsequent nodes will then evaluate or adjust based on these images.",
    "preview_svg": "Generate or explain preview svg to verify if they satisfy the task; subsequent nodes will then evaluate or adjust based on these svg."
}

def _resolve_model_from_state(state: dict):
    """Return a Pydantic model class referenced by state."""
    try:
        key = state.get('elements_map_schema_key') if isinstance(state, dict) else getattr(state, 'elements_map_schema_key', None)
        if key:
            model = _MODEL_REGISTRY.get(key)
            if model:
                return model
        direct = state.get('elements_map_schema') if isinstance(state, dict) else getattr(state, 'elements_map_schema', None)
        if direct and hasattr(direct, '__mro__'):
            return direct
    except Exception:
        pass
    return BaseModel

def _safe_model_dump(model) -> Dict:
    """convert pydanic model to safe dict that can be json dumped, the value can be Enum or another pydantic model"""
    try:
        if hasattr(model, 'dict') and callable(getattr(model, 'dict')):
            try:
                raw = model.dict()
            except Exception:
                try:
                    return json.loads(json.dumps(model, default=lambda o: getattr(o, 'value', str(o))))
                except Exception:
                    return {}
            if isinstance(raw, dict):
                return {k: _safe_model_dump(v) for k, v in raw.items()}
            if isinstance(raw, list):
                return [_safe_model_dump(v) for v in raw]
            return raw
        if isinstance(model, Enum):
            return model.value
        if model is None or isinstance(model, (str, int, float, bool)):
            return model
        if isinstance(model, dict):
            return {k: _safe_model_dump(v) for k, v in model.items()}
        if isinstance(model, list):
            return [_safe_model_dump(v) for v in model]
        return json.loads(json.dumps(model, default=lambda o: getattr(o, 'value', str(o))))
    except Exception:
        return {}

def preprocess_llm_input_with_memory(messages: List[AnyMessage], state: dict):
    from src.infra.tools.design_agent.nodes.memory import note_helper
    try:
        if note_helper is not None:
            note_msgs = note_helper.get_notes_messages()
            if note_msgs:
                return note_msgs
            else:
                return messages
    except Exception:
        return state.get('messages', [])
