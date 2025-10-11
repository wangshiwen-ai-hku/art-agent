# src/infra/tools/design_agent/nodes/summarize.py
import logging
import json
from langchain_core.messages import AIMessage, SystemMessage
from src.utils.print_utils import show_messages
from src.utils.multi_modal_utils import create_multimodal_message
from src.infra.tools.design_agent.schemas import StageEnum, SummarizeMap
from src.infra.tools.design_agent.nodes.utils import (
    design_llm,
    preprocess_llm_input_with_memory,
    _safe_model_dump,
    SYSTEM_CONTENT,
    NODE_MM_HINTS,
)
from src.infra.tools.design_agent.nodes.memory import note_helper

logger = logging.getLogger(__name__)

async def summarize_node(state, config=None):
    logger.info(f"-> Summarize node entered")
    try:
        messages = state.get('messages', [])
        prompt = (
            "Produce a concise `draw_description` suitable for SVG LOGO generation and "
            "a short `design_prompt` summarizing the concept. Return both fields."
        )
        llm_input = [SystemMessage(content=SYSTEM_CONTENT)] + note_helper.get_notes_messages()
        human_message = create_multimodal_message(text=prompt, image_data=state.get('generated_image_paths', []), mm_hint=NODE_MM_HINTS.get('preview'))
        llm_input = [SystemMessage(content=SYSTEM_CONTENT)] + preprocess_llm_input_with_memory(messages, state) + [human_message]
        show_messages(llm_input)

        structured_llm = design_llm.with_structured_output(SummarizeMap)
        
        llm_output = await structured_llm.ainvoke(llm_input)

        try:
            llm_output_dict = _safe_model_dump(llm_output)
            ai_content = json.dumps(llm_output_dict, ensure_ascii=False)
            ai_msg = AIMessage(content=ai_content)
            next_step = StageEnum.END
            logger.info(f"-> Summarize node to: {next_step.value}")
        except Exception as e:
            ai_content = getattr(llm_output, "__repr__", lambda: str(llm_output))()
            llm_output_dict = {}
            next_step = StageEnum.SUMMARIZE
            ai_content = json.dumps(llm_output_dict, ensure_ascii=False)
            new_messages = [AIMessage(content="Error in summarize_node: " + str(e))]
            return {"messages": new_messages, "stage": next_step, "stage_iteration": state.get('stage_iteration', 0) + 1}

        new_messages = [ai_msg]

        update = {
            "summarize_map": llm_output_dict,
            "messages": new_messages,
            "stage": next_step,
            "stage_iteration": state.get('stage_iteration', 0) + 1,
        }
        try:
            if note_helper is not None:
                _ = await note_helper.do_notes(state.get('messages', []) + new_messages)
                update['notes'] = note_helper.get_notes()
        except Exception:
            pass
        return update
    except Exception as e:
        logger.error(f"Error in summarize_node: {e}")
        return {"messages": state.get('messages', [])}
