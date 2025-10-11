# src/infra/tools/design_agent/nodes/brainstorm.py
import logging
import json
from pydantic import BaseModel
from langchain_core.messages import AIMessage, SystemMessage
from src.utils.print_utils import show_messages
from src.utils.multi_modal_utils import create_multimodal_message
from src.infra.tools.design_agent.schemas import StageEnum
from src.infra.tools.design_agent.nodes.utils import (
    design_llm,
    preprocess_llm_input_with_memory,
    _resolve_model_from_state,
    _safe_model_dump,
    SYSTEM_CONTENT,
    NODE_MM_HINTS,
)

logger = logging.getLogger(__name__)

def brainstorm_node():
    async def _inner(state, config=None):
        logger.info(f"-> Brainstorm node entered")
        if state.get('stage_iteration', 0) > 10:
            return {
                "messages": AIMessage(content="Task stop due to stage iteration limit. Summarize the current design."),
                "stage": StageEnum.SUMMARIZE,
            }
        messages = state.get('messages', [])
        prompt = "Brainstorm several visual elements and ideas for the design given the history. Return short items and a next_step."
        human_message = create_multimodal_message(text=prompt, image_data=state.get('generated_image_paths', []), mm_hint=NODE_MM_HINTS.get('brainstorm'))
        llm_input = [SystemMessage(content=SYSTEM_CONTENT)] + preprocess_llm_input_with_memory(messages, state) + [human_message]
        show_messages(llm_input)
        
        resolved_model = _resolve_model_from_state(state)
        try:
            if resolved_model is BaseModel:
                llm_output = await design_llm.ainvoke(llm_input)
            else:
                structured_llm = design_llm.with_structured_output(resolved_model)
                llm_output = await structured_llm.ainvoke(llm_input)
        except Exception:
            llm_output = await design_llm.ainvoke(llm_input)
        logger.info(f"-> Brainstorm node llm_output: {llm_output}")
        
        try:
            element_map = _safe_model_dump(llm_output)
            raw_next_step = getattr(llm_output, 'next_step', None)

            if raw_next_step and isinstance(raw_next_step, str):
                try:
                    next_step = StageEnum(raw_next_step)
                except ValueError:
                    next_step = StageEnum.COMPOSE
            else:
                next_step = StageEnum.COMPOSE
        except Exception:
            ai_content = getattr(llm_output, "__repr__", lambda: str(llm_output))()
            element_map = ai_content
            next_step = StageEnum.COMPOSE

        ai_content = json.dumps(element_map, ensure_ascii=False)
        ai_msg = AIMessage(content=ai_content)
        update = {
            "element_map_messages": [json.dumps(element_map, ensure_ascii=False)],
            "messages": [ai_msg],
            "stage": next_step,
            "stage_iteration": state.get('stage_iteration', 0) + 1,
        }
        return update
    return _inner
