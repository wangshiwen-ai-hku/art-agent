# src/infra/tools/design_agent/nodes/human_in_loop.py
import logging
from langgraph.types import interrupt
from langchain_core.messages import HumanMessage, AIMessage
from src.infra.tools.design_agent.schemas import StageEnum
from src.infra.tools.design_agent.nodes.utils import NODE_MAP

logger = logging.getLogger(__name__)

async def human_in_loop_node(state, config=None):
    logger.info(f"-> Human in loop node entered")
    payload = interrupt("Wait for user input")
    user_text = payload.get("user_message", "")
    requested_stage = payload.get("stage", StageEnum.REFLECT.value)

    try:
        stage_enum = requested_stage if isinstance(requested_stage, StageEnum) else StageEnum(requested_stage)
    except Exception:
        stage_enum = StageEnum.REFLECT

    next_node = NODE_MAP.get(stage_enum.value, "reflect_node")

    human_msg = HumanMessage(content=user_text)
    ai_msg = AIMessage(content=f"Received human input; routing to {stage_enum.value}.")

    return {
        "messages": [ai_msg, human_msg],
        "stage": stage_enum,
        "stage_iteration": state.get('stage_iteration', 0) + 1,
    }
