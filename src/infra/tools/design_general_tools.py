# canvas_tools.py
import logging
from statistics import mean
from typing import List, Tuple, Optional, TypedDict, Any, Dict, Union
from langchain_core.tools import tool
from langgraph.types import Command, interrupt
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
from pydantic import BaseModel, create_model
from langchain_core.tools import tool
import json
from enum import Enum
import uuid
from src.infra.tools.image_tools import generate_image
from src.infra.tools.draw_canvas_tools import draw_agent_with_tool
from langchain_core.runnables import RunnableConfig
from src.utils.multi_modal_utils import create_multimodal_message
from src.utils.schema_utils import safe_model_dump

from src.utils.visualize_utils import draw_graph
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# TODO omit the short and long memory timely, and improve the notehelper as global memory manager, make it intelligent in handling different messages (format), and do notes with `use_llm` intervally, usually it will simply concantanae directly. 
# Registry for runtime-only Pydantic models that must not be placed into the
# persisted state (they are not msgpack-serializable). Models are stored by
# a string key (we use `project_dir` when available).
_MODEL_REGISTRY: dict = {}

class NoteHelper():
    def __init__(self):
        self.notes = ""
        self.note_file = os.path.join(os.path.dirname(__file__), 'data/notes.md')
        os.makedirs(os.path.dirname(self.note_file), exist_ok=True)
        self.write = True
        if self.write:
            self.f = open(self.note_file, 'w')
        # TODO 
        self.pending_messages = []
        self.pending_notes = ""
        # TODO define the system message, as notehelper must take and compose the notes(maybe, the image_preview_url must be append to the nearest generated idea. )
        # self.system_message = SystemMessage(content="""You are a helpful note taking agent, You role is the """)
        try:
            self.llm = init_chat_model(**asdict(config.get_agent_config("notehelper_agent", "core").model))
        except Exception:
            # Fallback to the design llm if a dedicated notehelper config is not present
            try:
                self.llm = design_llm
            except Exception:
                self.llm = None

    def get_notes_messages(self, key='') -> List[HumanMessage]:
        # Return notes as a list containing a single HumanMessage for LLM input.
        try:
            note_text = self.get_notes()
            if not note_text:
                return []
            if key:
                note_text = f"[{key}]\n" + note_text
            return [HumanMessage(content=f"NOTES_SUMMARY:\n{note_text}")]
        except Exception:
            return []
    
    def get_notes(self):
        """Return the current notes string."""
        return self.notes + self.pending_notes

    async def do_notes(self, messages, use_llm=True):
        """
        Form short markdown notes from a list of messages. If an LLM is
        available and use_llm is True, ask it to summarise; otherwise fall
        back to a simple concatenation.
        Returns the notes string and stores into `self.notes`.
        """
        try:
            # Normalize messages to text with lightweight prefixes
            parts = []
            for m in messages or []:
                try:
                    prefix = ""
                    clsname = m.__class__.__name__ if hasattr(m, '__class__') else ''
                    if 'AIMessage' in clsname:
                        prefix = '[AI] '
                    elif 'HumanMessage' in clsname:
                        prefix = '[H] '
                    elif 'ToolMessage' in clsname or 'AnyMessage' in clsname:
                        prefix = '[TOOL] '
                    parts.append(prefix + getattr(m, 'content', str(m)))
                except Exception:
                    parts.append(str(m))

            # accumulate pending messages and text
            for p in parts:
                self.pending_messages.append(p)
            joined = "\n\n".join(parts) if parts else ""
            if joined:
                self.pending_notes = (self.pending_notes + "\n\n" + joined) if self.pending_notes else joined

            # Only call the LLM when we've accumulated enough messages
            if use_llm and self.llm is not None and len(self.pending_messages) >= getattr(self, 'llm_interval', 3):
                human = HumanMessage(content=("Existing notes:\n\n % s\n\n Summarise the following new messages to short markdown notes chunk, increasingly based on exsiting notes:\n\n % s" % (self.notes[-200:], self.pending_notes)))
                llm_input = [SystemMessage(content=(getattr(self, 'system_message', 'You are a concise note-taking assistant.'))), human]
                try:
                    llm_resp = await self.llm.ainvoke(llm_input)
                    note_text = getattr(llm_resp, 'content', str(llm_resp))
                    # commit summary and clear pending
                    self.notes = (self.notes + "\n\n" + note_text).strip() if self.notes else note_text
                    self.pending_notes = ""
                    self.pending_messages = []
                    if self.write:
                        self.f.write(note_text)
                        self.f.flush()
                        logger.info(f"-> NoteHelper.do_notes: {note_text}")
                except Exception:
                    # on failure, commit naive pending_notes
                    if self.pending_notes:
                        self.notes = (self.notes + "\n\n" + self.pending_notes).strip() if self.notes else self.pending_notes
                        self.pending_notes = ""
                        self.pending_messages = []
            else:
                # do not call LLM yet; keep pending notes and optionally commit short text
                if not use_llm:
                    # if user intentionally disabled LLM, commit the joined text to pending only
                    pass

            # return committed notes (not pending)
            return self.notes
        except Exception as e:
            logger.warning(f"NoteHelper.do_notes failed: {e}")
            return self.notes

    

def _resolve_model_from_state(state: dict):
    """Return a Pydantic model class referenced by state.

    This accepts either:
    - a direct model class stored under `elements_map_schema` (legacy)
    - a registry key stored under `elements_map_schema_key` that points into
      `_MODEL_REGISTRY` (preferred, serializable state)
    Falls back to `BaseModel` if nothing found.
    """
    try:
        # prefer explicit registry key
        key = state.get('elements_map_schema_key') if isinstance(state, dict) else getattr(state, 'elements_map_schema_key', None)
        if key:
            model = _MODEL_REGISTRY.get(key)
            if model:
                return model

        # legacy: model class stored directly
        direct = state.get('elements_map_schema') if isinstance(state, dict) else getattr(state, 'elements_map_schema', None)
        if direct and hasattr(direct, '__mro__'):
            return direct
    except Exception:
        pass
    return BaseModel

NODE_MAP = {
    "brainstorm": "brainstorm_node",
    "compose": "compose_elements_node",
    # "refine": "refine_node",
    "human": "human_in_loop_node",
    "summarize": "summarize_node",
    "reflect": "reflect_node",
    "preview_image": "preview_image_node",
    "preview_svg": "preview_svg_node",
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
    "preview_image": "Generate or explain preview images to verify if they satisfy the task; subsequent nodes will then evaluate or adjust based on these images.",
    "preview_svg": "Generate or explain preview svg to verify if they satisfy the task; subsequent nodes will then evaluate or adjust based on these svg."
}


DESIGN_SYSTEM_PROMPT = """
# ROLE
You are an award-winning brand designer. Read the **TASK** and produce the most valuable design artwork.
each `next_step` must refer the current notes and situation. Reflect the *TASK* and be coherent with it. 

# Router rule
1. `compose` and `preview_image` can help revise the design after `reflect`.
2. `brainstorm` can generate totally different design elements, had better use once.
3. use `summarize` in the end.

# Task
%s
-----------------

"""

design_llm = None

def preprocess_llm_input_with_memory(messages, state):
    # Compose an LLM input list by injecting compact memory and recent notes.
    # use case: llm_input = [SystemMessage(content=SYSTEM_CONTENT)] + preprocess_llm_input_with_memory(messages, state) + [human_message]
    # This function keeps the returned messages as a list suitable for passing
    # into `design_llm.ainvoke`.
 
    try:
        if note_helper is not None:
            note_msgs = note_helper.get_notes_messages()
            if note_msgs:
                return note_msgs
            else:
                return messages
    except Exception:
        return state.get('messages', [])


def _safe_model_dump(model) -> Dict:
    """
    convert pydanic model to safe dict that can be json dumped, the value can be Enum or another pydantic model
    """
    # Robust recursive conversion for BaseModel, Enum and nested containers.
    try:
        # Pydantic/BaseModel-like objects: call .dict() and recurse
        if hasattr(model, 'dict') and callable(getattr(model, 'dict')):
            try:
                raw = model.dict()
            except Exception:
                # Fallback to best-effort serialization
                try:
                    return json.loads(json.dumps(model, default=lambda o: getattr(o, 'value', str(o))))
                except Exception:
                    return {}

            if isinstance(raw, dict):
                return {k: _safe_model_dump(v) for k, v in raw.items()}
            if isinstance(raw, list):
                return [_safe_model_dump(v) for v in raw]
            return raw

        # Enum instances should be converted to their .value
        if isinstance(model, Enum):
            return model.value

        # Simple builtins and None
        if model is None or isinstance(model, (str, int, float, bool)):
            return model

        # dict/list: recurse
        if isinstance(model, dict):
            return {k: _safe_model_dump(v) for k, v in model.items()}
        if isinstance(model, list):
            return [_safe_model_dump(v) for v in model]

        # Last resort: try JSON encode with fallback to object's `value` attr
        return json.loads(json.dumps(model, default=lambda o: getattr(o, 'value', str(o))))
    except Exception:
        return {}


def _init_design_agent():
    agent_config = config.get_agent_config("design_agent", "core")
    model_config = agent_config.model
    global design_llm
    design_llm = init_chat_model(**asdict(model_config))
    return design_llm

design_llm = _init_design_agent() 

# Global note helper used to collect and provide short shared notes between nodes
try:
    note_helper = NoteHelper()
except Exception:
    note_helper = None

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
        llm_input = [SystemMessage(content=SYSTEM_CONTENT)] + preprocess_llm_input_with_memory(messages, state) + [human_message]
        logger.info(f"-> Compose elements node")
        show_messages(llm_input)

        # Use structured output to get reliable fields
        structured_llm = design_llm.with_structured_output(ComposeMap)
        llm_output = await structured_llm.ainvoke(llm_input)
        
        # llm_output is expected to be a ComposeMap-like object
        try:
            llm_output_dict = _safe_model_dump(llm_output)  # 保证每个组分都能被json dump
            ai_content = json.dumps(llm_output_dict, ensure_ascii=False)
            next_step = llm_output.next_step
            logger.info(f"-> Compose elements node to: {next_step.value}")
        except Exception as e:
            # fallback to raw text
            logger.info(f"-> Compose elements node error: {e}")
            ai_content = getattr(llm_output, "__repr__", lambda: str(llm_output))()
            llm_output_dict = {}
            next_step = StageEnum.COMPOSE
            new_messages = [AIMessage(content="Error in compose_elements_node: " + str(e))]

        ai_msg = AIMessage(content=ai_content)
        new_messages = [ai_msg]
        update = {
            "compose_map_messages": [
                llm_output_dict
            ],
            "messages": new_messages,
            "stage": next_step,
            "stage_iteration": state.get('stage_iteration', 0) + 1,
        }
        # attach notehelper-generated notes (best-effort)
        try:
            if note_helper is not None:
                # TODO create notes from the ai message, not include history, do the additional notes
                _ = await note_helper.do_notes(new_messages)
                update['notes'] = note_helper.get_notes()
        except Exception:
            pass
        # attach a short memory snapshot for this composition step
        # update = _attach_short_memory_to_update(update, state, idea=design_prompt_text)
        # logger.info(f"-> Compose elements node update: {update}")
        return update

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
        return {
            "messages": [AIMessage(content="Task stop due to stage iteration limit. Summarize the current design.")],
            "stage": StageEnum.SUMMARIZE,
            "stage_iteration": state.get('stage_iteration', 0) + 1,
        }
    
    logger.info(f"-> Plan node entered")
    try:
        messages = state.get('messages', [])
        task_desc = state.get('task_description', '') or config and config.get('task_description', '') or ''
        prompt = (
            DESIGN_SYSTEM_PROMPT + "\n\n"
            + "You are responsible for creating a plan for the design process. "
            + "Use reflect and summarize in the end. Make full use of other nodes: brainstorm, compose, preview_image "
        )
        human_message = create_multimodal_message(text=prompt, image_data=state.get('generated_image_paths', []), mm_hint=NODE_MM_HINTS.get('plan'))
        llm_input = [SystemMessage(content=SYSTEM_CONTENT)] + preprocess_llm_input_with_memory(messages, state) + [human_message]
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
        new_messages = [ai_msg]

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
        # attach a short memory snapshot for planning
        # update = _attach_short_memory_to_update(update, state, idea=task_desc)
        # logger.info(f"-> Plan node update: {update}")
        return update
        

    except Exception as e:
        logger.error(f"Error in plan_node: {e}")
        return {"messages": state.get('messages', [])}

async def human_in_loop_node(state, config=None):
    """
    add human in loop node to the graph
    """
    logger.info(f"-> Human in loop node entered")
    # Ask for an interrupt payload and route based on the provided stage
    payload = interrupt("Wait for user input")
    # payload is expected to have keys: 'user_message' and optional 'stage'
    user_text = payload.get("user_message", "")
    requested_stage = payload.get("stage", StageEnum.REFLECT.value)

    # Normalise into a StageEnum
    try:
        stage_enum = requested_stage if isinstance(requested_stage, StageEnum) else StageEnum(requested_stage)
    except Exception:
        stage_enum = StageEnum.REFLECT

    # Map to node (fallback to reflect_node)
    next_node = NODE_MAP.get(stage_enum.value, "reflect_node")

    human_msg = HumanMessage(content=user_text)
    ai_msg = AIMessage(content=f"Received human input; routing to {stage_enum.value}.")

    return {
        "messages": [ai_msg, human_msg],
        "stage": stage_enum,
        "stage_iteration": state.get('stage_iteration', 0) + 1,
    }

async def create_elements_map_node(state, config=None):
    """
    Ask the LLM to analyze the task and produce a JSON schema describing the
    elements map. The LLM should return JSON like:
    {
      "fields": [{"name": "directly_relevant_but_common", "type": "str", "default": "" , "description":"..."}, ...],
      "next_step": "brainstorm"
    }

    We parse that JSON and construct a Pydantic model with `create_model`, then
    store it in state['elements_map_schema'] for downstream nodes to use.
    """
    logger.info("-> create_elements_map_node entered")
    messages = state.get('messages', [])
    task_desc = state.get('task_description', '') or (config and config.get('task_description', '')) or ''

    prompt = (
        "Inspect the design task and produce a JSON schema describing the **core aspects** of the design, for example, `relevant_elements`, `imagen_style_prefix`......  "
        "Return a JSON object with keys: 'fields' (list of objects with 'name','type','default','description') and 'next_step' (one of brainstorm/compose/preview/refect). "
        "Only output valid JSON. Task: " + task_desc
    )

    human_message = create_multimodal_message(text=prompt, image_data=state.get('generated_image_paths', []), mm_hint=NODE_MM_HINTS.get('plan'))
    llm_input = [SystemMessage(content=SYSTEM_CONTENT)] + messages + [human_message]
    show_messages(llm_input)

    try:
        llm_response = await design_llm.ainvoke(llm_input)
        raw = getattr(llm_response, 'content', None) or str(llm_response)
        # Try to extract the first JSON object found in the LLM output
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

    # Map simple type strings to Python types
    simple_type_map = {"str": str, "int": int, "float": float, "bool": bool}
    model_kwargs = {}
    for f in fields:
        fname = f.get('name')
        ftype = f.get('type', 'str')
        default = f.get('default', None)
        pytype = simple_type_map.get(ftype, str)
        # If default is explicitly null/None, allow Optional
        if default is None:
            model_kwargs[fname] = (Optional[pytype], None)
        else:
            model_kwargs[fname] = (pytype, default)

    try:
        ElementsMap = create_model('ElementsMap', __base__=BaseModel, **model_kwargs)
    except Exception as e:
        logger.error(f"Failed to create ElementsMap model: {e}")
        # fallback to simple base model
        ElementsMap = create_model('ElementsMap', directly_relevant_but_common=(str, ''), indirectly_relevant_but_creative=(str, ''), __base__=BaseModel)

    ai_msg = AIMessage(content=json.dumps({"generated_fields": [f.get('name') for f in fields], "next_step": next_step}, ensure_ascii=False))
    _MODEL_REGISTRY[state.get('project_dir')] = ElementsMap

    update = {
        "elements_map_schema_key": state.get('project_dir'),
        "messages": [ai_msg],
        "stage": StageEnum(next_step),
        "stage_iteration": state.get('stage_iteration', 0) + 1,
    }
    # attach short memory for created elements map (use the first generated field as idea)
    # first_field = (fields[0].get('name') if fields else '')
    # update = _attach_short_memory_to_update(update, state, idea=first_field)
    
    # Register the dynamically-created model so it can be resolved by key
    # if state.get('project_dir'):
    #     _MODEL_REGISTRY[state['project_dir']] = ElementsMap
    
    return update

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
            "Produce a  `draw_description` detailed, path level, concise, suitable for SVG LOGO generation and "
            "a short `design_idea` summarizing the innovation, concept. Return both fields."
            "collect the latest image url as `image_url`"
        )
        if not state.get('generated_image_paths'):
            return {
                "messages": [AIMessage(content="No image generated. Go to generated image node.")],
                "stage": StageEnum.PREVIEW_IMAGE,
                "stage_iteration": state.get('stage_iteration', 0) + 1,
            }
        llm_input = [SystemMessage(content=SYSTEM_CONTENT)] + note_helper.get_notes_messages()
        human_message = create_multimodal_message(text=prompt, image_data=state.get('generated_image_paths', []), mm_hint=NODE_MM_HINTS.get('preview'))
        llm_input = [SystemMessage(content=SYSTEM_CONTENT)] + preprocess_llm_input_with_memory(messages, state) + [human_message]
        show_messages(llm_input)

        structured_llm = design_llm.with_structured_output(SummarizeMap)
        
        llm_output = await structured_llm.ainvoke(llm_input)

        try:
            # draw_desc = llm_output.draw_description
            # design_prompt = llm_output.design_prompt
            # image_preview_url = getattr(llm_output, 'image_preview_url', None)
            # ai_content = json.dumps({
            #     "draw_description": draw_desc,
            #     "design_prompt": design_prompt,
            #     "image_preview_url": image_preview_url,
            # }, ensure_ascii=False)
            llm_output_dict = _safe_model_dump(llm_output)
            ai_content = json.dumps(llm_output_dict, ensure_ascii=False)
            ai_msg = AIMessage(content=ai_content)
            next_step = StageEnum.END
            logger.info(f"-> Summarize node to: {next_step.value}")
        except Exception as e:
            # fallback to raw text
            ai_content = getattr(llm_output, "__repr__", lambda: str(llm_output))()
            llm_output_dict = {}
            next_step = StageEnum.SUMMARIZE
            ai_content = json.dumps(llm_output_dict, ensure_ascii=False)
            new_messages = [AIMessage(content="Error in summarize_node: " + str(e))]
            return {"messages": new_messages, "stage": next_step, "stage_iteration": state.get('stage_iteration', 0) + 1}

        # If the LLM didn't return an image URL, try to generate one locally
        new_messages = [ai_msg]

        update = {
            "summarize_map": llm_output_dict,
            "messages": new_messages,
            "stage": StageEnum.END,
            "stage_iteration": state.get('stage_iteration', 0) + 1,
        }
        
        try:
            if note_helper is not None:
                _ = await note_helper.do_notes(state.get('messages', []) + new_messages)
                update['notes'] = note_helper.get_notes()
        except Exception:
            pass
        # attach short memory for summarize step
        # update = _attach_short_memory_to_update(update, state, idea=draw_desc, image_url=png_path)
        # router will handle routing based on `stage` so return plain update
        return update
    except Exception as e:
        logger.error(f"Error in summarize_node: {e}")
        return {"messages": state.get('messages', [])}

async def reflect_node(state, config=None):
    """
    Reflect on the current designs and decide whether to continue or finish.
    Evaluates design on user-defined or default dimensions, computes an overall score,
    and provides recommendations for improvement.
    """
    if state.get('stage_iteration', 0) > 10:
        return {
            "messages": [AIMessage(content="Task stop due to stage iteration limit. Summarize the current design.")],
            "stage": StageEnum.SUMMARIZE,
            "stage_iteration": state.get('stage_iteration', 0) + 1,
        }
    logger.info(f"-> Reflect node entered")
    messages = state.get('messages', [])

    def preprocess_llm_input_with_memory(messages, state):
        """Reflect-specific preprocessing: prefer recent compose outputs and image previews.
        Injects previous dimension scores for trend analysis."""
        msgs = []
        # Prefer including last two compose_map_messages
        compose_msgs = state.get('compose_map_messages', [])
        if compose_msgs:
            recent = compose_msgs[-2:]
            for cm in recent:
                try:
                    text = cm.get('improved_design_prompt') or cm.get('design_prompt') or str(cm)
                    msgs.append(HumanMessage(content=f"Recent composition: {text}"))
                except Exception:
                    msgs.append(HumanMessage(content=str(cm)))

        # Include last generated image/svg paths
        g_images = state.get('generated_image_paths', [])[-2:]
        for p in g_images:
            msgs.append(create_multimodal_message(text="", image_data=[p], mm_hint=NODE_MM_HINTS.get('reflect')))
        g_svgs = state.get('generated_svg_paths', [])[-2:]
        for p in g_svgs:
            msgs.append(AIMessage(content=f"SVG_PREVIEW:{p}"))

        # Inject previous reflection dimensions
        reflect_msgs = state.get('reflect_map_messages', [])
        if reflect_msgs:
            recent = reflect_msgs[-2:]
            for rm in recent:
                try:
                    dim_text = json.dumps(rm.get('dimensions', {}), ensure_ascii=False)
                    msgs.append(HumanMessage(content=f"Previous reflection dimensions: {dim_text}"))
                except Exception:
                    pass

        # Prepend global notes
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

    # Get dimensions from config or use defaults
    dimensions = state.get('reflect_dimensions', {
        "consistency": "How well the design aligns with the task description",
        "creativity": "Uniqueness and originality of the design",
        "aesthetics": "Visual appeal and balance",
        "simplicity": "Clarity and ease of implementation",
        "usability": "Practicality for intended use (e.g., logo scalability)"
    })

    # Construct dynamic prompt
    dimension_prompt = "\n".join([f"- {dim}: {_safe_model_dump(desc)}" for dim, desc in dimensions.items()])
    prompt = (
        f"Based on the design history above, write a short reflection to make the design more consistent with the `TASK`. "
        f"Evaluate on these dimensions:\n{dimension_prompt}\n"
        f"For each dimension, provide a score (0-10) and a brief explanation. "
        f"Compute overall_score as the average of dimension scores. "
        f"Provide recommendations based on low-scoring dimensions (<6). "
        f"Suggest next_step: if overall_score < 6, suggest COMPOSE or BRAINSTORM; if 6-8, PREVIEW_IMAGE; if >8, SUMMARIZE."
    )
    human_and_mm_message = preprocess_llm_input_with_memory(messages, state)
    llm_input = [SystemMessage(content=SYSTEM_CONTENT + prompt)] + human_and_mm_message
    show_messages(llm_input)
    
    structured_llm = design_llm.with_structured_output(ReflectMapWithDimension)
    llm_output = await structured_llm.ainvoke(llm_input)
    logger.info(f"-> Reflect node llm_output: {llm_output}")
    
    try:
        llm_output_dict = _safe_model_dump(llm_output)
        # Ensure all requested dimensions are present
        ai_content = json.dumps(llm_output_dict, ensure_ascii=False)
        ai_msg = AIMessage(content=ai_content)
        nstep = StageEnum(llm_output_dict['next_step'])
        logger.info(f"-> Reflect node to: {nstep.value}")

    except Exception as e:
        logger.error(f"Error in reflect_node: {e}")
        new_messages = [AIMessage(content="Error in reflect_node: " + str(e))]
        nstep = StageEnum.END
        return {
            "messages": new_messages,
            "stage": nstep,
            "stage_iteration": state.get('stage_iteration', 0) + 1,
        }

    ai_msg = AIMessage(content=ai_content)
    new_messages = [ai_msg]
    
    update = {
        "reflect_map_messages": state.get('reflect_map_messages', []) + [llm_output_dict],
        "compose_map_messages": [llm_output_dict],
        "messages": new_messages,
        "stage_iteration": state.get('stage_iteration', 0) + 1,
        "stage": nstep,
    }
    try:
        if note_helper is not None:
            # `dimensions_dict` was a typo — use `dimensions` available in scope
            note_text = f"Reflection dimensions: {json.dumps(dimensions, ensure_ascii=False)}"
            _ = await note_helper.do_notes([HumanMessage(content=note_text)] + new_messages)
            update['notes'] = note_helper.get_notes()
    except Exception:
        pass
    return update

async def intention_recognition_node(state, config=None):
    """
    Recognize the intention of the user from the messages.

    Use the raw LLM output (no structured_output) to avoid Pydantic
    validation issues with message types. Attempt to parse a JSON object
    from the model response and return a safe, serializable dict.
    """
    logger.info(f"-> Intention recognition node entered")
    try:
        messages = state.get('messages', [])
        last_content = ""
        is_human = False
        for msg in reversed(messages):
            try:
                if isinstance(msg, AIMessage):
                    last_content += '[AI]' + getattr(msg, 'content', str(msg)) + '\n'
                elif isinstance(msg, HumanMessage):
                    is_human = True
                    last_content += '[Human]' + getattr(msg, 'content', str(msg)) + '\n'
            except Exception:
                last_content += str(msg) + '\n'
            if is_human:
                break

        prompt = """You are a helpful intention analyzer for a design agent.
Please analyze the following context and provide an `IntentionRecognitionMap` JSON object with fields:
- `task_description` (string, required)
- `width` (int, required)
- `height` (int, required)
- `critique_dimensions` (object, required): a map from metric name -> {expected_score: int 0-10, rationale: str}.
For `dimensions` return at least 4 named metrics that are directly motivated by the task or context. THINK thoroughly before you choose the metrics.
Each metric MUST include: `expected_score` (0-10 integer) and a short `rationale` (one sentence) explaining why that score was chosen.
Return ONLY a single JSON object (no extra commentary). Example:
{"task_description": "Design a compact logo combining a tiger and letter H", "width": 400, "height": 400, "critique_dimensions": {"consistency": {"expected_score": 8, "rationale": "Matches brief and uses tiger/H motif"}, "simplicity": {"expected_score": 6, "rationale": "Use high deformation to simplify the tiger"}, "artistic_value": {"expected_score": 10, "rationale": "Highly harmonious and balanced, with the tiger and H well integrated."}}}
The extracted context is provided below.

"""

        llm_input = [SystemMessage(content=prompt), HumanMessage(content=last_content)]

        structured_llm = design_llm.with_structured_output(IntentionRecognitionMap)
        llm_response = await structured_llm.ainvoke(llm_input)
        output = _safe_model_dump(llm_response)
        logger.info(f"-> Intention recognition raw LLM response: {llm_response}")

        ai_msg = AIMessage(content=json.dumps(output, ensure_ascii=False))

        return {
            "messages": [ai_msg],
            "task_description": llm_response.task_description,
            "critique_dimensions": llm_response.critique_dimensions,
            "width": llm_response.width,
            "height": llm_response.height,
            "stage": StageEnum.PLAN,
            "stage_iteration": state.get('stage_iteration', 0) + 1,
        }
    except Exception as e:
        logger.error(f"Error in intention recognition node: {e}")
        return {
            "messages": state.get('messages', []) + [AIMessage(content="Error in intention recognition node: " + str(e))],
            "stage": StageEnum.END,
            "stage_iteration": state.get('stage_iteration', 0) + 1,
        }


SYSTEM_CONTENT = DESIGN_SYSTEM_PROMPT 


async def _init_context_node(state, config=None):
    """
    init the context of the design process.
    """
    global SYSTEM_CONTENT
    SYSTEM_CONTENT  = DESIGN_SYSTEM_PROMPT % state.get('task_description', '')
    return {"messages": [SystemMessage(content=SYSTEM_CONTENT)]}
  
async def preview_image_node(state, config=None):
    """
    Generate preview images from the current design idea and feed the
    resulting images back into the state so the next node can inspect them.
    """
    
    logger.info(f"-> Preview image node entered")
    try:
        messages = state.get('messages', [])
        # Attempt to construct a prompt from the last design idea or compose messages
        show_messages(messages)
        last_text = ""
        if state.get('compose_map_messages'):
            last_text = state['compose_map_messages'][-1].get('improved_design_prompt', '') or state['compose_map_messages'][-1].get('design_prompt', '')
            last_style_hint = state['compose_map_messages'][-1].get('imagen_style_prefix', '')
        if not last_text:
            last_text = "Please generate a preview image for the current design based on the history."

        logger.info(f"-> Preview image generation prompt: {last_text}")
        prompt = f"{last_style_hint}. {last_text}"

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
        logger.info(f"-> Preview image node image paths: {image_paths}")

        ai_msg = AIMessage(content=json.dumps({"image_preview_url": png_path, "image_generation_prompt": prompt}, ensure_ascii=False))
        state.get('compose_map_messages', [])[-1].update({"image_preview_url": png_path})
        new_messages = [ai_msg]

        update = {
            "multi_modal_messages": [{"image_generation_prompt": prompt, "image_url": png_path, "next_step": StageEnum.REFLECT}],
            "generated_image_paths": image_paths,
            "stage": StageEnum.REFLECT,
            "messages": new_messages,
            "stage_iteration": state.get('stage_iteration', 0) + 1,
        }
        try:
            if note_helper is not None:
                note_text = f"Generated image: {png_path or 'in-memory'}; prompt: {prompt}"
                _ = await note_helper.do_notes([HumanMessage(content=note_text)])
                update['notes'] = note_helper.get_notes()
        except Exception:
            pass
        # attach short memory for the generated preview image
        # update = _attach_short_memory_to_update(update, state, idea=last_text, image_url=png_path)
        return update
    except Exception as e:
        logger.error(f"Error in preview_image_node: {e}")
        return {"messages": state.get('messages', [])}

async def preview_svg_node(state, config=None):
    """
    the agent can generate the svg code from the design idea as preview, not just the image generated by imagen-4
    with another agent that with just math agent. 
    can take in the preview image as multi-modal input
    """
    if state.get('stage_iteration', 0) > 10:
        return {
            "messages": AIMessage(content="Task stop due to stage iteration limit. Summarize the current design."),
            "stage": StageEnum.SUMMARIZE,
        }

    logger.info("-> Preview SVG node entered (delegating to draw_canvas_tools.draw_agent_with_tool)")
    try:
        # Prefer explicit summarize_map draw_description, then compose_map design_prompt
        draw_desc = None
        if state.get('summarize_map'):
            draw_desc = state['summarize_map'].get('draw_description')
        if not draw_desc and state.get('compose_map_messages'):
            draw_desc = state['compose_map_messages'][-1].get('design_prompt')

        if not draw_desc:
            draw_desc = "Preview SVG for current design"

        width = 400
        height = 400

        # Pass plan/critique context if available
        plan_context = None
        critique_notes = None
        if state.get('plan_map'):
            try:
                plan_context = json.dumps(state['plan_map'], ensure_ascii=False)
            except Exception:
                plan_context = str(state['plan_map'])
        if state.get('reflect_map_messages'):
            try:
                critique_notes = json.dumps(state['reflect_map_messages'][-1], ensure_ascii=False)
            except Exception:
                critique_notes = str(state['reflect_map_messages'][-1])

        # Call centralized draw agent tool to build a proper SVG
        draw_result_json = await draw_agent_with_tool(task_description=draw_desc, width=width, height=height, plan_context=plan_context or "", critique_notes=critique_notes or "")
        try:
            draw_result = json.loads(draw_result_json)
        except Exception:
            draw_result = {"svg": draw_result_json}

        new_svg = draw_result.get('svg') or draw_result.get('value') or (draw_result_json if isinstance(draw_result_json, str) else '')

        svg_path = None
        if new_svg and state.get('project_dir'):
            import time as _time
            timestamp = _time.strftime('%Y%m%d_%H%M%S')
            filename = f'preview_svg_{timestamp}.svg'
            svg_path = os.path.join(state['project_dir'], filename)
            open(svg_path, 'w', encoding='utf-8').write(new_svg)

        svg_paths = state.get('generated_svg_paths', [])
        if svg_path:
            svg_paths.append(svg_path)

        ai_msg = AIMessage(content=f"Draw agent produced svg: {svg_path or 'in-memory'}")

        update = {
            "multi_modal_messages": [{"image_generation_prompt": draw_desc, "image_url": svg_path or '', "next_step": StageEnum.REFLECT}],
            "generated_svg_paths": svg_paths,
            "messages": [ai_msg],
            "stage": StageEnum.REFLECT,
            "stage_iteration": state.get('stage_iteration', 0) + 1,
        }
        try:
            if note_helper is not None:
                note_text = f"Generated svg: {svg_path or 'in-memory'}; draw_desc: {draw_desc}"
                _ = await note_helper.do_notes([HumanMessage(content=note_text)])
                update['notes'] = note_helper.get_notes()
        except Exception:
            pass
        # attach short memory for generated svg preview
        # update = _attach_short_memory_to_update(update, state, idea=draw_desc, image_url=svg_path)
        return update
    except Exception as e:
        logger.error(f"Error in preview_svg_node: {e}")
        return {"messages": state.get('messages', [])}

async def update_memory_node(state, config=None):
    """
    update the memory of the design process.
    """
    logger.info("-> update_memory_node entered")
    try:
        # Collect short memories and condense into long memory summary
        short_list = state.get('short_memorys', [])
        if not short_list:
            return {
                "messages": [AIMessage(content="No short memories to update.")],
                "stage": StageEnum.REFLECT,
            }

        # Simple condensation: take top N fields and concatenate
        summaries = []
        for s in short_list[-10:]:
            try:
                if isinstance(s, dict):
                    summaries.append(s.get('current_idea', str(s)))
                else:
                    summaries.append(str(s))
            except Exception:
                summaries.append(str(s))

        long_summary = " | ".join([str(x) for x in summaries if x])[:2000]

        long_mem = state.get('long_memory', {}) or {}
        long_mem['most_consistent_with_user_instructions_ideas'] = long_summary

        ai_msg = AIMessage(content=json.dumps({"updated_long_memory": True, "summary_len": len(long_summary)}))

        return {
            "long_memory": long_mem,
            "short_memorys": [],
            "messages": [ai_msg],
            "stage": StageEnum.REFLECT,
            "stage_iteration": state.get('stage_iteration', 0) + 1,
        }
    except Exception as e:
        logger.error(f"Error in update_memory_node: {e}")
        return {"messages": state.get('messages', [])}

def router_node(state, config=None):
    """Simple pass-through node.

    The router node is a synchronous hub that all other nodes connect to. It
    does not modify state; it simply acts as the source for the conditional
    edge that follows, which executes the actual routing logic.
    """
    return {}

def route_logic(state):
    """Synchronous routing logic to determine the next node."""
    logger.info(f"-> Route logic: {state.get('stage')}")
    if state.get('stage') == StageEnum.END:
        return END
        
    if state.get('stage_iteration', 0) > 10 or len(state.get('generated_image_paths', [])) >= 4:
        return "summarize"
    
    if (state.get('stage') == StageEnum.REFLECT or state.get('stage') == StageEnum.SUMMARIZE) and not state.get('generated_image_paths', []):
        return "preview_image"
    
    if len(state.get('short_memorys', [])) >= 3:
        return "update_memory_node"
    
    if state.get('reflect_map_messages'):
        latest_reflect = state['reflect_map_messages'][-1]
        if latest_reflect.get('overall_score', 10) < 5:
            dimensions = latest_reflect.get('dimensions', [])
            low_scores = [dim for dim in dimensions if dim.get('score', 0) < 6]
            if "consistency" in low_scores:
                return "compose"
            return "brainstorm"
        if latest_reflect.get('overall_score', 10) >= 8:
            return "summarize"

    stage = state.get('stage', StageEnum.PLAN)

    if isinstance(stage, StageEnum):
        stage = stage.value
    
    # Route to END if the stage is 'end'. The conditional edges mapping expects
    # `route_logic` to return the *stage key* (e.g. 'compose', 'brainstorm'),
    # not the node id. Return the stage key when known, otherwise fall back to
    # the 'reflect' stage or END.
    if stage == StageEnum.END:
        return END
    if stage in NODE_MAP:
        return stage
    return "reflect"

def brainstorm_node():
    """
    Brainstorm the design.
    """
    # Keep a minimal interface compatible with StateGraph nodes: accept state
    async def _inner(state, config=None):
        logger.info(f"-> Brainstorm node entered")
        if state.get('stage_iteration', 0) > 10:
            return {
                "messages": AIMessage(content="Task stop due to stage iteration limit. Summarize the current design."),
                "stage": StageEnum.SUMMARIZE,
            }
        messages = state.get('messages', [])
        prompt = "Brainstorm and fill each key with design elements and ideas."
        human_message = create_multimodal_message(text=prompt, image_data=state.get('generated_image_paths', []), mm_hint=NODE_MM_HINTS.get('brainstorm'))
        llm_input = [SystemMessage(content=SYSTEM_CONTENT)] + preprocess_llm_input_with_memory(messages, state) + [human_message]
        show_messages(llm_input)
        # Resolve a runtime model for structured output. If resolver returns
        # the generic BaseModel (meaning no specific schema available), avoid
        # calling `with_structured_output` because some providers call
        # `model_json_schema()` which fails on BaseModel itself.
        resolved_model = _resolve_model_from_state(state)
        try:
            if resolved_model is BaseModel:
                llm_output = await design_llm.ainvoke(llm_input)
            else:
                structured_llm = design_llm.with_structured_output(resolved_model)
                llm_output = await structured_llm.ainvoke(llm_input)
        except Exception:
            # Fallback to raw LLM call if structured invocation fails for any reason
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
            next_step_text = next_step.value

        except Exception:
            ai_content = getattr(llm_output, "__repr__", lambda: str(llm_output))()
            element_map = ai_content
            next_step = StageEnum.COMPOSE
            next_step_text = next_step.value

        ai_content = json.dumps(element_map, ensure_ascii=False)
        # ai_content = json.dumps({"directly_relevant_but_common": directly, "indirectly_relevant_but_creative": indirectly, "next_step": next_step_text}, ensure_ascii=False)
        ai_msg = AIMessage(content=ai_content)
        update = {
            "element_map_messages": [json.dumps(element_map, ensure_ascii=False)],
            "messages": [ai_msg],
            "stage": next_step,
            "stage_iteration": state.get('stage_iteration', 0) + 1,
        }
        # update = _attach_short_memory_to_update(update, state, idea=element_map)
        return update
    # 
    return _inner

class StageEnum(Enum):
    PLAN = "plan"
    BRAINSTORM = "brainstorm"
    COMPOSE = "compose"
    SUMMARIZE = "summarize"
    # HUMAN = "human"
    REFLECT = "reflect"
    PREVIEW_IMAGE = "preview_image"
    # PREVIEW_SVG = "preview_svg"
    END = "end"

class ComposeMap(BaseModel):
    picked_elements: str
    elements_composition: str
    imagen_style_prefix: str
    design_prompt: str
    next_step: StageEnum

class PlanMap(BaseModel):
    plan_steps: list[StageEnum]

class ReflectMap(BaseModel):
    reflection: str
    old_design_prompt: str | List[str]
    improved_design_prompt: str
    next_step: StageEnum

class DimensionEval(BaseModel):
    dimension: str
    score: int  # 0-10
    explanation: str  # Brief explanation for the score

class ReflectMapWithDimension(BaseModel):
    reflection: str
    # old_design_prompt: Union[str, List[str]]
    improved_design_prompt: str
    dimensions: List[DimensionEval]  # Dynamic dimensions
    overall_score: float
    recommendations: List[str]
    next_step: StageEnum

class MultiModalMap(BaseModel):
    image_generation_prompt: str
    image_url: str
    next_step: StageEnum

class SummarizeMap(BaseModel):
    draw_description: str
    design_idea: str
    image_preview_url: str

class DesignState(ArtistState):
    plan_map: PlanMap
    stage: StageEnum
    generated_image_paths: List[str]
    generated_svg_paths: List[str]

    # elements_map_schema is not serializable. Use elements_map_schema_key
    # to store a reference to the runtime-only Pydantic model instead.
    elements_map_schema_key: Optional[str]
    element_map_messages: List[BaseModel]

    compose_map_messages: List[ComposeMap]
    # refine_map_messages: List[RefineMap]
    reflect_map_messages: List[ReflectMapWithDimension]
    multi_modal_messages: List[MultiModalMap]
    critique_dimensions: Dict[str, str]
    # short_memorys: List[ShortMemory]
    # long_memory: LongMemory
    notes: str
    stage_iteration: int

class DimensionScore(BaseModel):
    expected_score: int
    rationale: str


class IntentionRecognitionMap(BaseModel):
    task_description: str
    width: int = 400
    height: int = 400
    critique_dimensions: Dict[str, DimensionScore] = {"consistency": DimensionScore(expected_score=8, rationale="Aligns with xxxxxx")}


graph = StateGraph(DesignState)


def _register_graph_nodes(graph: StateGraph):
    """Register nodes and edges for the autonomous design agent graph."""
    graph.add_node("plan_node", plan_node)
    graph.add_node("router_node", router_node)
    graph.add_node("init_context_node", _init_context_node)
    graph.add_node("human_in_loop_node", human_in_loop_node)
    graph.add_node("create_elements_map_node", create_elements_map_node)
    graph.add_node("brainstorm_node", brainstorm_node())
    graph.add_node("compose_elements_node", compose_elements_node)
    graph.add_node("preview_image_node", preview_image_node)
    graph.add_node("preview_svg_node", preview_svg_node)
    graph.add_node("reflect_node", reflect_node)
    graph.add_node("summarize_node", summarize_node)
    graph.add_node("update_memory_node", update_memory_node)
    graph.add_node("intention_recognition_node", intention_recognition_node)

    graph.add_edge(START,  "init_context_node")
    graph.add_edge("init_context_node", "intention_recognition_node")
    graph.add_edge('intention_recognition_node', "create_elements_map_node")
    # graph.add_edge('create_elements_map_node', "plan_node")
    graph.add_edge('create_elements_map_node', "router_node")
    graph.add_edge('plan_node', "router_node")
    # graph.add_edge('compose_elements_node', "router_node")
    # All logical nodes route back to the router to decide the next step
    # graph.add_edge("plan_node", "router_node")
    graph.add_edge("human_in_loop_node", "router_node")
    graph.add_edge("brainstorm_node", "router_node")
    graph.add_edge("compose_elements_node", "router_node")
    graph.add_edge("preview_image_node", "router_node")
    graph.add_edge("preview_svg_node", "router_node")
    graph.add_edge("reflect_node", "router_node")
    graph.add_edge("update_memory_node", "router_node")
    graph.add_edge("summarize_node", "router_node")
    
    # The router dispatches to the next logical node based on 'stage'
    graph.add_conditional_edges(
        "router_node",
        route_logic,
        {
            "plan": "plan_node",
            "brainstorm": "brainstorm_node",
            "compose": "compose_elements_node",
            "preview_image": "preview_image_node",
            "preview_svg": "preview_svg_node",
            "reflect": "reflect_node",
            "summarize": "summarize_node",
            "human": "human_in_loop_node",
            "update_memory_node": "update_memory_node",
            END: END,
        },
    )


async def run_agent_with_tool(task_description: str, width: int, height: int, reflect_dimensions: Optional[Dict[str, str]] = None):
    """
    Build and run the autonomous design agent graph for a single pass.
    Generates preview image and writes to `output/design_agent_preview/`.
    Allows custom reflection dimensions via reflect_dimensions.
    Returns the final state dictionary.
    """
    _register_graph_nodes(graph)
    graph.set_entry_point("init_context_node")

    project_dir = "output/design_agent_preview/" + str(uuid.uuid4())
    os.makedirs(project_dir, exist_ok=True)

    init_state = {
        "messages": [HumanMessage(content=task_description)],
        # "task_description": task_description,
        # "generated_image_paths": [],
        "project_dir": project_dir,
        # "critique_dimensions": reflect_dimensions,
    }
    from langgraph.checkpoint.memory import MemorySaver
    draw_graph(graph, "design_agent_graph")
    compiled_graph = graph.compile(checkpointer=MemorySaver())
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
    # 示例：自定义维度
    custom_dimensions = {
        "brand_alignment": "Alignment with brand identity",
        "innovation": "Novelty and forward-thinking design",
        "clarity": "Visual clarity and readability",
        "versatility": "Adaptability across different media"
    }
    asyncio.run(run_agent_with_tool("我希望建立一个app，为这个app设计一个logo。名字为”psycho“。关注于具有心理疾病或者精神问题的人，给他们提供一个新的社交平台，, 目的是凸显这些小众群体的个性，给我设计一个简洁的，令人印象深刻的app logo。\n 简约线条，使用字母变形为核心", 400, 400, reflect_dimensions=custom_dimensions))


@tool
async def design_agent_with_tool(
    task_description: str,
    width: int = 400,
    height: int = 400,
    max_iterations: int = 3,
    reference_notes: str = "",
    initial_svg: str = "",
) -> str:
    """Expose the internal `run_agent_with_tool` as a reusable tool for the
    main CanvasAgent. Returns a JSON string with keys similar to the
    existing `design_tools.design_agent_with_tool` output so callers can
    consume a consistent payload.
    """
    try:
        final_state = await run_agent_with_tool(task_description, width, height)
    except Exception as e:
        # return a minimal failure payload
        return json.dumps({"type": "design_result", "svg": "", "value": "", "plan": {}, "critique_history": [], "iterations": 0, "tool_events": [], "iteration_logs": [], "memory": {"error": str(e)}})

    # Attempt to read the last generated svg if present
    svg_content = ""
    try:
        svg_paths = final_state.get("generated_svg_paths") or []
        if svg_paths:
            last_path = svg_paths[-1]
            if last_path and os.path.exists(last_path):
                with open(last_path, "r", encoding="utf-8") as f:
                    svg_content = f.read()
    except Exception:
        svg_content = ""

    result = {
        "type": "design_result",
        "svg": svg_content,
        "value": svg_content,
        "plan": final_state.get("plan_map") or {},
        "critique_history": final_state.get("reflect_map_messages", []),
        "iterations": len(final_state.get("generated_svg_paths", []) or final_state.get("generated_image_paths", [])),
        "tool_events": final_state.get("multi_modal_messages", []),
        "iteration_logs": final_state.get("iteration_logs", []),
        "memory": {"notes": final_state.get("notes", "")},
    }

    return json.dumps(result, ensure_ascii=False)



