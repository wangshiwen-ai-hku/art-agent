# src/infra/tools/design_agent/nodes/create_elements_map.py
import logging
import json
import re
from typing import Optional
from pydantic import create_model, BaseModel
from langchain_core.messages import AIMessage, SystemMessage
from src.utils.print_utils import show_messages
from src.utils.multi_modal_utils import create_multimodal_message
from src.infra.tools.design_agent.schemas import StageEnum, _MODEL_REGISTRY
from src.infra.tools.design_agent.nodes.utils import (
    design_llm,
    SYSTEM_CONTENT,
    NODE_MM_HINTS,
)

logger = logging.getLogger(__name__)

async def create_elements_map_node(state, config=None):
    logger.info("-> create_elements_map_node entered")
    messages = state.get('messages', [])
    task_desc = state.get('task_description', '') or (config and config.get('task_description', '')) or ''

    prompt = (
        "Inspect the design task and produce a JSON schema describing the elements map. "
        "Return a JSON object with keys: 'fields' (list of objects with 'name','type','default','description') and 'next_step' (one of brainstorm/compose/preview/refect). "
        "Only output valid JSON. Task: " + task_desc
    )

    human_message = create_multimodal_message(text=prompt, image_data=state.get('generated_image_paths', []), mm_hint=NODE_MM_HINTS.get('plan'))
    llm_input = [SystemMessage(content=SYSTEM_CONTENT)] + messages + [human_message]
    show_messages(llm_input)

    try:
        llm_response = await design_llm.ainvoke(llm_input)
        raw = getattr(llm_response, 'content', None) or str(llm_response)
        m = re.search(r"\{[\s\S]*\}", raw)
        json_text = m.group(0) if m else raw
        parsed = json.loads(json_text)
        fields = parsed.get('fields', [])
        next_step = parsed.get('next_step', StageEnum.BRAINSTORM.value)
    except Exception as e:
        logger.warning(f"create_elements_map_node: failed to parse LLM schema ({e}), falling back to default schema")
        fields = [
            {"name": "directly_relevant_but_common", "type": "str", "default": "", "description": "Direct elements"},
            {"name": "indirectly_relevant_but_creative", "type": "str", "default": "", "description": "Creative elements"},
            {"name": "imagen_style_prefix", "type": "str", "default": None, "description": "Style hint (vector, abstract...)"},
            {"name": "next_step", "type": "str", "default": "compose", "description": "Next stage"},
        ]
        next_step = 'brainstorm'

    simple_type_map = {"str": str, "int": int, "float": float, "bool": bool}
    model_kwargs = {}
    for f in fields:
        fname = f.get('name')
        ftype = f.get('type', 'str')
        default = f.get('default', None)
        pytype = simple_type_map.get(ftype, str)
        if default is None:
            model_kwargs[fname] = (Optional[pytype], None)
        else:
            model_kwargs[fname] = (pytype, default)

    try:
        ElementsMap = create_model('ElementsMap', __base__=BaseModel, **model_kwargs)
    except Exception as e:
        logger.error(f"Failed to create ElementsMap model: {e}")
        ElementsMap = create_model('ElementsMap', directly_relevant_but_common=(str, ''), indirectly_relevant_but_creative=(str, ''), __base__=BaseModel)

    ai_msg = AIMessage(content=json.dumps({"generated_fields": [f.get('name') for f in fields], "next_step": next_step}, ensure_ascii=False))
    _MODEL_REGISTRY[state.get('project_dir')] = ElementsMap

    update = {
        "elements_map_schema_key": state.get('project_dir'),
        "messages": [ai_msg],
        "stage": StageEnum(next_step),
        "stage_iteration": state.get('stage_iteration', 0) + 1,
    }
    return update
