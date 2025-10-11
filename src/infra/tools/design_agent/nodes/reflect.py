# src/infra/tools/design_agent/nodes/reflect.py
import logging
import json
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from src.utils.print_utils import show_messages
from src.utils.multi_modal_utils import create_multimodal_message
from src.infra.tools.design_agent.schemas import StageEnum, ReflectMap
from src.infra.tools.design_agent.nodes.utils import (
    design_llm,
    _safe_model_dump,
    SYSTEM_CONTENT,
    NODE_MM_HINTS,
)
from src.infra.tools.design_agent.nodes.memory import note_helper

logger = logging.getLogger(__name__)

async def reflect_node(state, config=None):
    if state.get('stage_iteration', 0) > 10:
        return {
            "messages": [AIMessage(content="Task stop due to stage iteration limit. Summarize the current design.")],
            "stage": StageEnum.SUMMARIZE,
            "stage_iteration": state.get('stage_iteration', 0) + 1,
        }
    logger.info(f"-> Reflect node entered")

    def preprocess_llm_input_with_memory(messages, state):
        msgs = []
        compose_msgs = state.get('compose_map_messages', [])
        if compose_msgs:
            recent = compose_msgs[-2:]
            for cm in recent:
                try:
                    text = cm.get('improved_design_prompt') or cm.get('design_prompt') or str(cm)
                    msgs.append(HumanMessage(content=f"Recent composition: {text}"))
                except Exception:
                    msgs.append(HumanMessage(content=str(cm)))

        g_images = state.get('generated_image_paths', [])[-2:]
        for p in g_images:
            msgs.append(create_multimodal_message(text="", image_data=[p], mm_hint=NODE_MM_HINTS.get('reflect')))
        g_svgs = state.get('generated_svg_paths', [])[-2:]
        for p in g_svgs:
            msgs.append(AIMessage(content=f"SVG_PREVIEW:{p}"))

        try:
            if note_helper is not None:
                note_msgs = note_helper.get_notes_messages()
                if note_msgs:
                    msgs = note_msgs + msgs
        except Exception:
            pass

        if not msgs:
            msgs = list(messages) if messages else []

        return msgs

    messages = state.get('messages', [])
    prompt = "Based on the design history above, write a short reflection. to make the design more consistent with the `TASK`"
    human_and_mm_message = preprocess_llm_input_with_memory(messages, state)
    llm_input = [SystemMessage(content=SYSTEM_CONTENT + prompt)] + human_and_mm_message
    show_messages(llm_input)
    structured_llm = design_llm.with_structured_output(ReflectMap)
    llm_output = await structured_llm.ainvoke(llm_input)
    logger.info(f"-> Reflect node llm_output: {llm_output}")
    try:
        llm_output_dict = _safe_model_dump(llm_output)
        ai_content = json.dumps(llm_output_dict, ensure_ascii=False)
        ai_msg = AIMessage(content=ai_content)
        nstep = llm_output.next_step
        logger.info(f"-> Reflect node to: {nstep.value}")

    except Exception as e:
        new_messages = [AIMessage(content="Error in reflect_node: " + str(e))]
        nstep = StageEnum.END
        return {"messages": new_messages, "stage": nstep, "stage_iteration": state.get('stage_iteration', 0) + 1}

    ai_msg = AIMessage(content=ai_content)
    new_messages = [ai_msg]
    
    update = {
        "reflect_map_messages": state.get('reflect_map_messages', []) + [
            llm_output_dict
        ],
        "compose_map_messages": state.get('compose_map_messages', []) + [
            llm_output_dict
        ],
        "messages": new_messages,
        "stage_iteration": state.get('stage_iteration', 0) + 1,
        "stage": nstep,
    }
    try:
        if note_helper is not None:
            _ = await note_helper.do_notes(state.get('messages', []) + new_messages)
            update['notes'] = note_helper.get_notes()
    except Exception:
        pass
    return update
