# src/infra/tools/design_agent/nodes/compose.py
import logging
import json
from langchain_core.messages import AIMessage, SystemMessage
from src.utils.print_utils import show_messages
from src.utils.multi_modal_utils import create_multimodal_message
from src.infra.tools.design_agent.schemas import StageEnum, ComposeMap
from src.infra.tools.design_agent.nodes.utils import (
    design_llm,
    preprocess_llm_input_with_memory,
    _safe_model_dump,
    SYSTEM_CONTENT,
    NODE_MM_HINTS,
)
from src.infra.tools.design_agent.nodes.memory import note_helper

logger = logging.getLogger(__name__)

async def compose_elements_node(state, config=None):
    logger.info(f"-> Compose elements node entered")
    try:
        messages = state.get('messages', [])
        prompt = (
            "Compose the visual elements for the design based on the history.\n"
            "Please return picked elements, a short composition description, a design idea, and the next step."
        )
        
        human_message = create_multimodal_message(text=prompt, image_data=state.get('generated_image_paths', []), mm_hint=NODE_MM_HINTS.get('compose'))
        llm_input = [SystemMessage(content=SYSTEM_CONTENT)] + preprocess_llm_input_with_memory(messages, state) + [human_message]
        logger.info(f"-> Compose elements node")
        show_messages(llm_input)

        structured_llm = design_llm.with_structured_output(ComposeMap)
        llm_output = await structured_llm.ainvoke(llm_input)
        
        try:
            llm_output_dict = _safe_model_dump(llm_output)
            ai_content = json.dumps(llm_output_dict, ensure_ascii=False)
            next_step = llm_output.next_step
            logger.info(f"-> Compose elements node to: {next_step.value}")
        except Exception as e:
            logger.info(f"-> Compose elements node error: {e}")
            ai_content = getattr(llm_output, "__repr__", lambda: str(llm_output))()
            llm_output_dict = {}
            next_step = StageEnum.COMPOSE
            new_messages = [AIMessage(content="Error in compose_elements_node: " + str(e))]

        ai_msg = AIMessage(content=ai_content)
        new_messages = [ai_msg]
        update = {
            "compose_map_messages": state.get('compose_map_messages', []) + [
                llm_output_dict
            ],
            "messages": new_messages,
            "stage": next_step,
            "stage_iteration": state.get('stage_iteration', 0) + 1,
        }
        try:
            if note_helper is not None:
                _ = await note_helper.do_notes(new_messages)
                update['notes'] = note_helper.get_notes()
        except Exception:
            pass
        return update

    except Exception as e:
        logger.error(f"Error in compose_elements_node: {e}")
        return {"messages": state.get('messages', [])}
