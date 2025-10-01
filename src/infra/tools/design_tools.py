# canvas_tools.py
import logging
from typing import List, Tuple, Optional, TypedDict, Any
from langchain_core.tools import tool
from langgraph.types import Command
import svgwrite
from langchain_core.messages import HumanMessage, AIMessage
import math
from src.agents.base.schema import ArtistState
from src.config.manager import config
# from src.agents.base import BaseAgent
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage, AnyMessage
from langgraph.graph.message import add_messages, MessagesState
from dataclasses import asdict
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from src.utils.print_utils import show_messages
import json
import re, glob
import os
from pydantic import BaseModel
from langchain_core.tools import tool
import json
from enum import Enum
import uuid
from .image_tools import generate_image
from langchain_core.runnables import RunnableConfig
from src.utils.multi_modal_utils import create_multimodal_message

from src.utils.visualize_utils import draw_graph
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

NODE_MAP = {
    "brainstorm": "brainstorm_node",
    "compose": "compose_elements_node",
    # "refine": "refine_node",
    "summarize": "summarize_node",
    "reflect": "reflect_node",
    "preview": "multi_modal_node",
    "end": END,
}

# Multimodal hints per logical node. These describe the ROLE of attached images
# at each step (e.g. reflect should tell the model the images are previews to
# analyze and possibly return to an earlier step).
NODE_MM_HINTS = {
    "plan": "Use these images (if any) as supplementary context: only for determining which high-level stages are needed and why.",
    "brainstorm": "These attached images are for reference or rough previews; use them to expand diverse visual ideas, do not finalize the composition immediately.",
    "compose": "These attached images show candidate elements or previews—select and combine elements to form a coherent composition, while using the images as example references.",
    "refine": "These attached images are intermediate compositions; refine them for clarity, balance, and implementability, and refer to these previews.",
    "reflect": "These images are generated previews: analyze whether they meet requirements, point out deficiencies, and determine if it's necessary to return to a specific step for reshaping (if so, provide clear instructions).",
    "preview": "Generate or explain preview images to verify if they satisfy the task; subsequent nodes will then evaluate or adjust based on these images."
}

class DesignPlan(BaseModel):
    """
    design_idea: The description of the design. unique and creative ideas, 
    """
    design_description: str
    design_ideas: str
    visual_elements: str
    

DESIGN_SYSTEM_PROMPT = """
# ROLE
Your MBTI is INFP. F is Feeling, P is Perceiving. 
Act as a thoughtful, professional, cool and unique designer. You will be given a `task description` and you need to design a logo for the topic. 

You will be assgined a plan, BUT you can CHANGE the plan by the `next_step` of each node, based on the current history and situation.
"""


design_llm = None

def _init_design_agent():
    agent_config = config.get_agent_config("design_agent", "core")
    model_config = agent_config.model
    global design_llm
    design_llm = init_chat_model(**asdict(model_config))
    return design_llm

design_llm = _init_design_agent() 

async def compose_elements_node(state, config=None):
    """
    Compose the elements of the design. Nodes receive the full message
    history through `state['messages']` and should append their outputs
    so later nodes can see the full design process.

    Returns a ComposeMap-like update and appends an AIMessage to messages.
    """
    logger.info(f"-> Compose elements node entered")
    try:
        messages = state.get('messages', [])
        prompt = (
            "Compose the visual elements for the design based on the history.\n"
            "Please return picked elements, a short composition description, a design idea, and the next step."
        )
        
        # Add user prompt into messages so LLM sees history + instruction
        human_message = create_multimodal_message(text=prompt, image_data=state.get('generated_image_paths', []), mm_hint=NODE_MM_HINTS.get('compose'))
        llm_input = messages + [human_message]
        logger.info(f"-> Compose elements node")
        show_messages(llm_input)

        # Use structured output to get reliable fields
        structured_llm = design_llm.with_structured_output(ComposeMap)
        llm_output = await structured_llm.ainvoke(llm_input)
        # llm_output is expected to be a ComposeMap-like object
        try:
            picked = llm_output.picked_elements
            comp_desc = llm_output.composition_description
            design_idea_text = llm_output.design_idea
            next_step_text = llm_output.next_step.value or "preview"
            ai_content = json.dumps({
                "picked_elements": picked,
                "composition_description": comp_desc,
                "design_idea": design_idea_text,
                "next_step": next_step_text,
            }, ensure_ascii=False)
            logger.info(f"-> Compose elements node to: {next_step_text}")
        except Exception:
            # fallback to raw text
            ai_content = getattr(llm_output, "__repr__", lambda: str(llm_output))()
            picked = ai_content
            comp_desc = ""
            design_idea_text = ""
            next_step_text = "preview"

        ai_msg = AIMessage(content=ai_content)
        new_messages = [ai_msg]
        return Command(goto=NODE_MAP[next_step_text], update={
            "compose_map_messages": [
                {
                    "picked_elements": picked,
                    "composition_description": comp_desc,
                    "design_idea": design_idea_text,
                    "next_step": next_step_text,
                }
            ],
            "messages": new_messages,
        })
    except Exception as e:
        logger.error(f"Error in compose_elements_node: {e}")
        return {"messages": state.get('messages', [])}

async def plan_node(state, config=None):
    """
    Produce an initial plan for the design process. The plan node reads the
    full message history and returns a PlanMap containing an ordered list of
    `StageEnum` values. It appends the AI decision to `state['messages']` so
    subsequent nodes can observe the decision-making trace.
    """
    if state.get('stage_iteration', 0) > 10:
        return Command(goto=NODE_MAP["summarize"], update={
            "messages": AIMessage(content="Task stop due to stage iteration limit. Summarize the current design."),
            "stage": StageEnum.SUMMARIZE,
        })
    
    logger.info(f"-> Plan node entered")
    try:
        messages = state.get('messages', [])
        task_desc = state.get('task_description', '') or config and config.get('task_description', '') or ''
        prompt = (
            DESIGN_SYSTEM_PROMPT + "\n\n"
            + "You are responsible for creating a plan for the design process. "
            + "Use reflect and summarize in the end. Make full use of other nodes: brainstorm, compose, preview "
            + f"\n# TASK: \n{task_desc}\n\n"
        )
        human_message = create_multimodal_message(text=prompt, image_data=state.get('generated_image_paths', []), mm_hint=NODE_MM_HINTS.get('plan'))
        llm_input = messages + [human_message]
        # logger.info(f"-> Plan node llm_input: {llm_input}")
        show_messages(llm_input)
        structured_llm = design_llm.with_structured_output(PlanMap)
        llm_output = await structured_llm.ainvoke(llm_input)
        logger.info(f"-> Plan node llm_output: {llm_output}")
        # Append decision to history
        try:
            steps = llm_output.plan_steps
            logger.info(f"-> Plan steps: {steps}")
            ai_content = json.dumps({"plan_steps": [s.value if isinstance(s, StageEnum) else s for s in steps]}, ensure_ascii=False)
        except Exception:
            # fallback: try to parse raw text from llm_output
            ai_content = getattr(llm_output, "__repr__", lambda: str(llm_output))()
            steps = re.findall(r"(brainstorm|compose|refine|reflect|preview)", ai_content, flags=re.IGNORECASE)
        ai_msg = AIMessage(content=ai_content)
        new_messages = [human_message, ai_msg]

        # Parse the response for stage keywords
        found = [s if isinstance(s, str) else (s.value if hasattr(s, 'value') else str(s)) for s in (steps or [])]
        plan_steps = []
        for s in found:
            key = s.lower()
            try:
                plan_steps.append(StageEnum(key))
            except Exception:
                continue

        if not plan_steps:
            # fallback to a reasonable default
            plan_steps = [StageEnum.BRAINSTORM, StageEnum.COMPOSE, StageEnum.PREVIEW]
        logger.info(f"- plan node to: {plan_steps[0].value}")
        return Command(goto=NODE_MAP[plan_steps[0].value], update={
            "plan_map": {"plan_steps": plan_steps},
            "stage": plan_steps[0],
            "messages": new_messages,
            "stage_iteration": state.get('stage_iteration', 0) + 1,
        })
        

    except Exception as e:
        logger.error(f"Error in plan_node: {e}")
        return {"messages": state.get('messages', [])}

async def summarize_node(state, config=None):
    """
    Summarize the current design into a concise drawing description and a
    short design idea that's suitable for generating a preview image.

    Returns a Command that advances to the `preview` node, attaches a
    `summarize_map` and appends an AIMessage to the history. If the LLM does
    not provide an image URL, this node will attempt to generate a preview
    image using `generate_image` and save it into `state['project_dir']`.
    """
    
    logger.info(f"-> Summarize node entered")
    try:
        messages = state.get('messages', [])
        prompt = (
            "Produce a concise `draw_description` suitable for SVG LOGO generation and "
            "a short `design_idea` summarizing the concept. Return both fields."
        )

        human_message = create_multimodal_message(text=prompt, image_data=state.get('generated_image_paths', []), mm_hint=NODE_MM_HINTS.get('preview'))
        llm_input = messages + [human_message]
        show_messages(llm_input)

        structured_llm = design_llm.with_structured_output(SummarizeMap)
        llm_output = await structured_llm.ainvoke(llm_input)

        try:
            draw_desc = llm_output.draw_description
            design_idea = llm_output.design_idea
            image_preview_url = getattr(llm_output, 'image_preview_url', None)
            ai_content = json.dumps({
                "draw_description": draw_desc,
                "design_idea": design_idea,
                "image_preview_url": image_preview_url,
            }, ensure_ascii=False)
        except Exception:
            # fallback to raw text
            ai_content = getattr(llm_output, "__repr__", lambda: str(llm_output))()
            draw_desc = ai_content
            design_idea = ""
            image_preview_url = None

        # If the LLM didn't return an image URL, try to generate one locally
        png_path = image_preview_url
        if not png_path:
            try:
                prompt_for_image = draw_desc or design_idea or "Generate a preview image for the design."
                image_pil = generate_image(prompt=prompt_for_image, aspect_ratio="1:1")
                if state.get('project_dir'):
                    import time
                    timestamp = time.strftime('%Y%m%d_%H%M%S')
                    save_path = os.path.join(state['project_dir'], f'summarize_preview_{timestamp}.png')
                    image_pil.save(save_path)
                    png_path = save_path
                else:
                    png_path = None
            except Exception as e:
                logger.error(f"Error generating preview image in summarize_node: {e}")
                png_path = None

        image_paths = state.get('generated_image_paths', [])
        if png_path:
            image_paths.append(png_path)

        ai_msg = AIMessage(content=ai_content)
        new_messages = [human_message, ai_msg]

        return {
            "summarize_map": {"draw_description": draw_desc, "design_idea": design_idea, "image_preview_url": png_path},
            "generated_image_paths": image_paths,
            "messages": new_messages,
            "stage": StageEnum.PREVIEW,
            "stage_iteration": state.get('stage_iteration', 0) + 1,
        }
    except Exception as e:
        logger.error(f"Error in summarize_node: {e}")
        return {"messages": state.get('messages', [])}

async def reflect_node(state, config=None):
    """
    Reflect on the current designs and decide whether to continue or finish.
    This node reads `state['messages']` to summarise and append its reflection.
    """
    
    if state.get('stage_iteration', 0) > 10:
        return Command(goto=NODE_MAP["summarize"], update={
            "messages": AIMessage(content="Task stop due to stage iteration limit. Summarize the current design."),
            "stage": StageEnum.SUMMARIZE,
        })
    logger.info(f"-> Reflect node entered")
    # try:
    if True:
        messages = state.get('messages', [])
        prompt = "Based on the design history above, write a short reflection including design_description, design_idea and suggested next_step. Your task is to make the design more consistent with the `TASK`"
        human_message = create_multimodal_message(text=prompt, image_data=state.get('generated_image_paths', [])[-2:], mm_hint=NODE_MM_HINTS.get('reflect'))
        llm_input = messages + [human_message]
        # logger.info(f"-> Reflect node llm_input: {llm_input}")
        show_messages(llm_input)
        structured_llm = design_llm.with_structured_output(ReflectMap)
        llm_output = await structured_llm.ainvoke(llm_input)
        logger.info(f"-> Reflect node llm_output: {llm_output}")
        try:
            ddesc = llm_output.reflection
            didea = llm_output.renewed_design_idea
            nstep = llm_output.next_step
            ai_content = json.dumps({"design_description": ddesc, "renewed_design_idea": didea, "next_step": nstep.value}, ensure_ascii=False)
            logger.info(f"-> Reflect node to: {nstep.value}")

        except Exception:
            ai_content = getattr(llm_output, "__repr__", lambda: str(llm_output))()
            ddesc = ai_content
            didea = ""
            nstep = StageEnum.END

        ai_msg = AIMessage(content=ai_content)
        new_messages = [human_message, ai_msg]
        

        return Command(
            goto=NODE_MAP[nstep.value], 
            # goto=, 
            update={
            "reflect_map_messages": [
                {
                    "reflection": ddesc,
                    "renewed_design_idea": didea,
                    "next_step": nstep,
                }
            ],
            "compose_map_messages": [
                {
                    "design_idea": didea,
                }
            ],
            "messages": new_messages,
            "stage_iteration": state.get('stage_iteration', 0) + 1,
            "stage": nstep,
        })
    # except Exception as e:
    #     logger.error(f"Error in reflect_node: {e}")
    #     return {"messages": state.get('messages', [])}

async def multi_modal_node(state, config=None):
    """
    Generate preview images from the current design idea and feed the
    resulting images back into the state so the next node can inspect them.
    """
    if state.get('stage_iteration', 0) > 10:
        return Command(goto=NODE_MAP["summarize"], update={
            "messages": AIMessage(content="Task stop due to stage iteration limit. Summarize the current design."),
            "stage": StageEnum.SUMMARIZE,
        })
    
    logger.info(f"-> Multi modal node entered")
    try:
        messages = state.get('messages', [])
        # Attempt to construct a prompt from the last design idea or compose messages
        show_messages(messages)
        last_text = ""
        if state.get('compose_map_messages'):
            last_text = state['compose_map_messages'][-1].get('design_idea', '')
        if not last_text:
            last_text = "Please generate a preview image for the current design based on the history."

        logger.info(f"-> Multi modal image generation prompt: {last_text}")
        prompt = f"{last_text}"

        # Call the image generation tool in a thread to avoid blocking
        # Run the blocking image generation in a thread so we don't await a
        # non-coroutine object (the image return is a blocking call).
        import asyncio as _asyncio
        async def generate_image_and_save(prompt, aspect_ratio):
            image_pil = generate_image(prompt=prompt, aspect_ratio=aspect_ratio)
            if state.get('project_dir'):
                import time
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                save_path = os.path.join(state['project_dir'], f'preview_{timestamp}.png')
                image_pil.save(save_path)
                png_path = save_path
            else:
                png_path = None
            return png_path

        png_path = await generate_image_and_save(prompt=prompt, aspect_ratio="1:1")
        image_paths = state.get('generated_image_paths', [])

        if png_path:
            image_paths.append(png_path)
        logger.info(f"-> Multi modal node image paths: {image_paths}")

        ai_msg = AIMessage(content=f"Generated preview image: {png_path or 'in-memory image'}")
        new_messages = [ai_msg]

        return Command(goto=NODE_MAP["reflect"], update={
            "multi_modal_messages": [{"image_generation_prompt": prompt, "image_url": png_path, "next_step": StageEnum.REFLECT}],
            "generated_image_paths": image_paths,
            "stage": StageEnum.REFLECT,
            "messages": new_messages,
            "stage_iteration": state.get('stage_iteration', 0) + 1,
        })
    except Exception as e:
        logger.error(f"Error in multi_modal_node: {e}")
        return {"messages": state.get('messages', [])}

class StageEnum(Enum):
    BRAINSTORM = "brainstorm"
    COMPOSE = "compose"
    SUMMARIZE = "summarize"
    # REFINE = "refine"
    REFLECT = "reflect"
    PREVIEW = "preview"
    END = "end"


class ElementMap(BaseModel):
    directly_relevant_but_common: str
    indirectly_relevant_but_creative: str
    next_step: StageEnum

class ComposeMap(BaseModel):
    picked_elements: str
    composition_description: str
    design_idea: str
    next_step: StageEnum

class PlanMap(BaseModel):
    plan_steps: list[StageEnum]


class ReflectMap(BaseModel):
    reflection: str
    renewed_design_idea: Optional[str]
    next_step: StageEnum

class MultiModalMap(BaseModel):
    image_generation_prompt: str
    image_url: str
    next_step: StageEnum

class SummarizeMap(BaseModel):
    draw_description: str
    design_idea: str
    image_preview_url: str

def brainstorm_node():
    """
    Brainstorm the design.
    """
    # Keep a minimal interface compatible with StateGraph nodes: accept state
    async def _inner(state, config=None):
        if state.get('stage_iteration', 0) > 10:
            return Command(goto=NODE_MAP["summarize"], update={
                "messages": AIMessage(content="Task stop due to stage iteration limit. Summarize the current design."),
                "stage": StageEnum.SUMMARIZE,
            })
        messages = state.get('messages', [])
        prompt = "Brainstorm several visual elements and ideas for the design given the history. Return short items and a next_step."
        human_message = create_multimodal_message(text=prompt, image_data=state.get('generated_image_paths', []), mm_hint=NODE_MM_HINTS.get('brainstorm'))
        llm_input = messages + [human_message]
        show_messages(llm_input)
        structured_llm = design_llm.with_structured_output(ElementMap)
        llm_output = await structured_llm.ainvoke(llm_input)
        logger.info(f"-> Brainstorm node llm_output: {llm_output}")
        try:
            directly = llm_output.directly_relevant_but_common
            indirectly = llm_output.indirectly_relevant_but_creative
            next_step = llm_output.next_step or StageEnum.COMPOSE
            ai_content = json.dumps({"directly_relevant_but_common": directly, "indirectly_relevant_but_creative": indirectly, "next_step": next_step}, ensure_ascii=False)
        except Exception:
            ai_content = getattr(llm_output, "__repr__", lambda: str(llm_output))()
            directly = ai_content
            indirectly = ""
            next_step = StageEnum.COMPOSE
        ai_msg = AIMessage(content=ai_content)
        new_messages = [human_message, ai_msg]
        return Command(goto=NODE_MAP[next_step.value], update={
            "element_map_messages": [{"directly_relevant_but_common": directly, "indirectly_relevant_but_creative": indirectly, "next_step": next_step}],
            "messages": new_messages,
            "stage": next_step,
            "stage_iteration": state.get('stage_iteration', 0) + 1,
        })
    logger.info(f"-> Brainstorm node entered")
    return _inner

class DesignState(ArtistState):
    plan_map: PlanMap
    stage: StageEnum
    generated_image_paths: List[str]
    element_map_messages: List[ElementMap]
    compose_map_messages: List[ComposeMap]
    # refine_map_messages: List[RefineMap]
    reflect_map_messages: List[ReflectMap]
    multi_modal_messages: List[MultiModalMap]

    stage_iteration: int


graph = StateGraph(DesignState)


def _register_graph_nodes(graph: StateGraph):
    """Register nodes and edges for the autonomous design agent graph."""
    graph.add_node("plan_node", plan_node)
    graph.add_node("brainstorm_node", brainstorm_node())
    graph.add_node("compose_elements_node", compose_elements_node)
    graph.add_node("multi_modal_node", multi_modal_node)
    graph.add_node("reflect_node", reflect_node)
    graph.add_node("summarize_node", summarize_node)

    graph.add_edge(START, "plan_node")
    graph.add_edge("summarize_node", END)
    # graph.add_edge("plan_node", "brainstorm_node")
    # graph.add_edge("brainstorm_node", "compose_elements_node")
    # graph.add_edge("compose_elements_node", "multi_modal_node")
    # graph.add_edge("multi_modal_node", "reflect_node")
    # graph.add_edge("reflect_node", END)


async def run_agent_with_tool(task_description: str, width: int, height: int):
    """
    Build and run the autonomous design agent graph for a single pass.
    Generates preview image and writes to `output/design_agent_preview/`.
    Returns the final state dictionary.
    """
    _register_graph_nodes(graph)
    graph.set_entry_point("plan_node")

    project_dir = "output/design_agent_preview/" + str(uuid.uuid4())
    os.makedirs(project_dir, exist_ok=True)

    init_state = {
        "messages": [SystemMessage(content=DESIGN_SYSTEM_PROMPT)],
        "task_description": task_description,
        "generated_image_paths": [],
        "project_dir": project_dir,
    }
    from langgraph.checkpoint.memory import MemorySaver
    draw_graph(graph, "design_agent_graph")
    compiled_graph = graph.compile(checkpointer=MemorySaver())
    # graph = graph.compile(checkpointer=MemorySaver())    # Run the graph from the entry point until END
    config = RunnableConfig({"configurable": {"thread_id": "user_design_agent"}})
    try:
        final_state = await compiled_graph.ainvoke(init_state, config=config)
        show_messages(final_state.get("messages", []))
    except Exception as e:
        logger.error(f"Error running design graph: {e}")
        raise

    return final_state

if __name__ == "__main__":
    import asyncio

    asyncio.run(run_agent_with_tool("""International Conference on Scientific Computing (ICSML),再加上年份，2024， 2025, 加上港大logo的元素在里面.""", 400, 400))



