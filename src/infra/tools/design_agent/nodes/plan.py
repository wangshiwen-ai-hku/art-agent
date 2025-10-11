# src/infra/tools/design_agent/nodes/plan.py
import logging
import json
import re
from langchain_core.messages import AIMessage, SystemMessage
from src.utils.print_utils import show_messages
from src.utils.multi_modal_utils import create_multimodal_message
from src.infra.tools.design_agent.schemas import StageEnum, PlanMap
from src.infra.tools.design_agent.nodes.utils import (
    design_llm,
    preprocess_llm_input_with_memory,
    SYSTEM_CONTENT,
    NODE_MM_HINTS,
    DESIGN_SYSTEM_PROMPT,
)
from src.infra.tools.design_agent.nodes.memory import note_helper

logger = logging.getLogger(__name__)

async def plan_node(state, config=None):
    if state.get('stage_iteration', 0) > 10:
        return {
            "messages": [AIMessage(content="Task stop due to stage iteration limit. Summarize the current design.")],
            "stage": StageEnum.SUMMARIZE,
            "stage_iteration": state.get('stage_iteration', 0) + 1,
        }
    
    logger.info(f"-> Plan node entered")
    try:
        messages = state.get('messages', [])
        prompt = (
            DESIGN_SYSTEM_PROMPT % state.get('task_description', '') + "\n\n"
            + "You are responsible for creating a plan for the design process. "
            + "Use reflect and summarize in the end. Make full use of other nodes: brainstorm, compose, preview_image "
        )
        human_message = create_multimodal_message(text=prompt, image_data=state.get('generated_image_paths', []), mm_hint=NODE_MM_HINTS.get('plan'))
        llm_input = [SystemMessage(content=SYSTEM_CONTENT)] + preprocess_llm_input_with_memory(messages, state) + [human_message]
        show_messages(llm_input)
        structured_llm = design_llm.with_structured_output(PlanMap)
        llm_output = await structured_llm.ainvoke(llm_input)
        logger.info(f"-> Plan node llm_output: {llm_output}")
        try:
            steps = llm_output.plan_steps
            logger.info(f"-> Plan steps: {steps}")
            ai_content = json.dumps({"plan_steps": [s.value if isinstance(s, StageEnum) else s for s in steps]}, ensure_ascii=False)
        except Exception:
            ai_content = getattr(llm_output, "__repr__", lambda: str(llm_output))()
            steps = re.findall(r"(brainstorm|compose|refine|reflect|preview)", ai_content, flags=re.IGNORECASE)
        ai_msg = AIMessage(content=ai_content)
        new_messages = [ai_msg]

        found = [s if isinstance(s, str) else (s.value if hasattr(s, 'value') else str(s)) for s in (steps or [])]
        plan_steps = []
        for s in found:
            key = s.lower()
            try:
                plan_steps.append(StageEnum(key))
            except Exception:
                continue

        if not plan_steps:
            plan_steps = [StageEnum.BRAINSTORM, StageEnum.COMPOSE, StageEnum.PREVIEW_IMAGE]
        logger.info(f"- plan node to: {plan_steps[0].value}")
        update = {
            "plan_map": {"plan_steps": plan_steps},
            "stage": plan_steps[0],
            "messages": new_messages,
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
        logger.error(f"Error in plan_node: {e}")
        return {"messages": state.get('messages', [])}
