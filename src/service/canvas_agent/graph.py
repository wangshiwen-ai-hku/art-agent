"""Canvas agent graph implementation with stage-aware routing and tooling."""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, interrupt

from src.agents.base import BaseAgent
from src.config.manager import AgentConfig
from src.infra.tools.design_tools import design_agent_with_tool
from src.infra.tools.design_general_tools import design_agent_with_tool as design_general_agent_with_tool
from src.infra.tools.design_general_tools import DesignState
from src.infra.tools.svg_tools import PickPathTools

from .schema import (
    AGENT_STAGE_MAP,
    INTENT_TO_STAGE,
    ROUTER_PROMPT,
    AgentStage,
    CanvasState,
    CritiqueFeedback,
    DesignPlan,
    Router,
    SvgArtwork,
    UserIntent,
)
from .utils import show_messages, svg_to_png

logger = logging.getLogger(__name__)


class CanvasAgent(BaseAgent):
    """Stage-aware canvas agent orchestrating draw/edit/describe workflows."""

    name = "canvas_agent"
    description = "An agent that generates and edits SVG sketches with tool support."

    def __init__(self, agent_config: AgentConfig):
        super().__init__(agent_config)
        self._load_system_prompt()
        self.config = agent_config
        self._tools_by_name: Dict[str, BaseTool] = {tool.name: tool for tool in self._tools}
        self._router_llm = self._build_router_llm()
        self._planner_llm = self._build_planner_llm()
        self._critique_llm = self._build_critique_llm()
        self._stage_agents: Dict[AgentStage, Any] = self._build_stage_agents()

    # ---------------------------------------------------------------------
    # Initialization helpers
    # ---------------------------------------------------------------------
    def init_llm(self, temperature: float = 0.7):
        """Instantiate an LLM with a specific temperature without mutating the base config."""
        model_config = replace(self._model_config, temperature=temperature)
        return init_chat_model(**asdict(model_config))

    def _load_system_prompt(self):
        """Loads reusable prompt snippets."""
        prompt_dir = Path(__file__).parent / "prompt"
        self._system_prompts = {
            "system": (prompt_dir / "system_prompt.txt").read_text(encoding="utf-8"),
            "chat": (prompt_dir / "chat_prompt.txt").read_text(encoding="utf-8"),
            "draw": (prompt_dir / "drawer_prompt.txt").read_text(encoding="utf-8"),
            "edit": (prompt_dir / "edit_prompt.txt").read_text(encoding="utf-8"),
            "describe": (prompt_dir / "describe_prompt.txt").read_text(encoding="utf-8"),
            "pick": (prompt_dir / "pick_path_prompt.txt").read_text(encoding="utf-8"),
            "generate_image": (prompt_dir / "image_generator_prompt.txt").read_text(encoding="utf-8"),
            "plan": (prompt_dir / "sketch_planner_prompt.txt").read_text(encoding="utf-8"),
            "critique": (prompt_dir / "critique_prompt.txt").read_text(encoding="utf-8"),
        }

    def _build_router_llm(self):
        """Create a structured-output router model."""
        return self.init_llm(temperature=0.0).with_structured_output(Router)

    def _build_planner_llm(self):
        """LLM used to draft structured design plans."""
        return self.init_llm(temperature=0.2).with_structured_output(DesignPlan)

    def _build_critique_llm(self):
        """LLM used to critique and gate SVG outputs."""
        return self.init_llm(temperature=0.1).with_structured_output(CritiqueFeedback)

    def _build_stage_agents(self) -> Dict[AgentStage, Any]:
        """Create stage-specific ReAct agents with curated tools and prompts."""
        agents: Dict[AgentStage, Any] = {}

        def get_tools(*names: str, extra: Optional[Iterable[BaseTool]] = None) -> List[BaseTool]:
            tools: List[BaseTool] = []
            for name in names:
                tool = self._tools_by_name.get(name)
                if tool:
                    tools.append(tool)
            if extra:
                tools.extend(extra)
            return tools

        agents[AgentStage.CHAT] = create_react_agent(
            model=self.init_llm(temperature=0.4),
            tools=[],
            prompt=self._system_prompts["chat"],
        )
        agents[AgentStage.DRAW] = create_react_agent(
            model=self.init_llm(temperature=0.6),
            tools=get_tools("draw_agent_with_tool"),
            prompt=self._system_prompts["draw"],
        )
        agents[AgentStage.EDIT] = create_react_agent(
            model=self.init_llm(temperature=0.5),
            tools=get_tools("edit_agent_with_tool", "draw_agent_with_tool"),
            prompt=self._system_prompts["edit"],
        )
        agents[AgentStage.DESCRIBE] = create_react_agent(
            model=self.init_llm(temperature=0.4),
            tools=list(PickPathTools),
            prompt=self._system_prompts["describe"],
        )
        agents[AgentStage.PICK_PATH] = create_react_agent(
            model=self.init_llm(temperature=0.4),
            tools=list(PickPathTools),
            prompt=self._system_prompts["pick"],
        )
        agents[AgentStage.GENERATE_IMAGE] = design_general_agent_with_tool
        return agents

    # ---------------------------------------------------------------------
    # Graph nodes
    # ---------------------------------------------------------------------
    async def _init_context_node(self, state: CanvasState, config=None):
        logger.info("---CANVAS AGENT: INIT CONTEXT NODE---")
        messages = state["conversation"]["messages"]
        if not messages:
            messages.append(SystemMessage(content=self._system_prompts["system"]))
        return state

    async def wait_for_user_input(self, state: CanvasState, config=None):
        logger.info("---CANVAS AGENT: WAIT FOR USER INPUT---")
        payload = interrupt("wait_for_user_input")
        user_input = payload.get("user_input", "").strip()
        if not user_input:
            logger.info("-> Empty user input received; staying in wait state.")
            return state

        state["user_input"] = user_input
        stage_override = payload.get("stage")
        if stage_override:
            try:
                override_stage = AgentStage(stage_override)
                state["workflow"]["current_stage"] = override_stage
                state["workflow"]["manual_override"] = True
            except ValueError:
                logger.warning("-> Invalid stage override '%s' ignored.", stage_override)
        else:
            state["workflow"]["manual_override"] = False

        if payload.get("reference_images"):
            state["content"]["reference_images"] = payload["reference_images"]

        conversation = state["conversation"]["messages"]
        conversation = [msg for msg in conversation if not (isinstance(msg, HumanMessage) and getattr(msg, "name", "") == "canvas_svg_context")]
        conversation.append(HumanMessage(content=user_input))
        state["conversation"]["messages"] = conversation
        return state

    async def router_node(self, state: CanvasState, config=None):
        logger.info("---CANVAS AGENT: ROUTER NODE---")
        user_input = state.get("user_input", "")
        workflow = state["workflow"]

        if not user_input:
            logger.info("-> No user input available; returning to wait node.")
            return Command(goto="wait_for_user_input")

        if workflow.get("manual_override"):
            target_stage = workflow["current_stage"]
            workflow["manual_override"] = False
            logger.info("-> Respecting manual stage override: %s", target_stage)
        else:
            try:
                router_result = await self._router_llm.ainvoke(
                    [
                        SystemMessage(content=ROUTER_PROMPT),
                        HumanMessage(content=user_input),
                    ]
                )
                # Allow the router to emit either legacy intents or the new
                # `DESIGN` intent. Map DESIGN to the generate-image branch so
                # `preview_image_node` / `generate_image_node` handles image
                # generation while `plan_node` continues handling SVG plan+draw.
                intent_value = getattr(router_result, "intent", None)
                try:
                    mapped_intent = UserIntent(intent_value)
                except Exception:
                    # Some routers may return raw strings like 'design'
                    if intent_value == "design":
                        mapped_intent = UserIntent.DESIGN
                    else:
                        mapped_intent = UserIntent.CHAT

                target_stage = INTENT_TO_STAGE.get(mapped_intent, AgentStage.CHAT)
                workflow["current_intent"] = router_result.intent
                workflow["current_stage"] = target_stage
                state["conversation"]["current_topic"] = (
                    router_result.query if router_result.intent == UserIntent.CREATE else state["conversation"]["current_topic"]
                )
                if router_result.query:
                    state["user_input"] = router_result.query
                logger.info("-> Routed intent %s to stage %s", router_result.intent, target_stage)
            except Exception as exc:
                logger.exception("Router failed; defaulting to chat. %s", exc)
                target_stage = AgentStage.CHAT
                workflow["current_stage"] = target_stage
                workflow["current_intent"] = UserIntent.CHAT

        return Command(
            goto=AGENT_STAGE_MAP[target_stage],
            update={
                "workflow": workflow,
                "user_input": state["user_input"],
                "conversation": state["conversation"],
            },
        )

    async def chat_node(self, state: CanvasState, config=None):
        return await self._invoke_stage_agent(AgentStage.CHAT, state)

    async def plan_node(self, state: CanvasState, config=None):
        logger.info("---CANVAS AGENT: PLAN NODE---")
        task = state.get("user_input", "") or state["conversation"].get("current_topic") or ""
        width = 512
        height = 512

        try:
            # Use the design agent exposed as a tool. Tools follow a BaseTool
            # interface; the `design_tools.design_agent_with_tool` is decorated
            # with `@tool` and supports async `.ainvoke` with a dict payload.
            design_tool = design_agent_with_tool
            design_result = await design_tool.ainvoke(
                {
                    "task_description": task or "Create a new SVG concept",
                    "width": width,
                    "height": height,
                    "max_iterations": 3,
                    "reference_notes": state.get("conversation", {}).get("current_topic") or "",
                    "initial_svg": "",
                }
            )
            # The tool returns a JSON string; parse into dict
            design_payload = json.loads(design_result)
        except Exception as exc:
            logger.exception("Design orchestrator failed: %s", exc)
            state["conversation"]["messages"].append(
                AIMessage(content=f"Design orchestration failed: {exc}")
            )
            state["workflow"]["current_stage"] = AgentStage.CHAT
            state["workflow"]["current_intent"] = UserIntent.CHAT
            return Command(goto="wait_for_user_input", update=state)

        plan_data = design_payload.get("plan") or {}
        try:
            plan = DesignPlan(**plan_data)
        except Exception:
            plan = None

        if plan:
            state["content"]["design_plan"] = plan
        plan_summary = self._format_plan_for_conversation(plan) if plan else ""

        svg_code = design_payload.get("svg", "")
        if svg_code:
            self._persist_svg_result(AgentStage.DRAW, svg_code, state)

        critique_payloads = design_payload.get("critique_history", []) or []
        critique_objects = []
        for item in critique_payloads:
            try:
                critique_objects.append(CritiqueFeedback(**item))
            except Exception:
                continue
        state["content"]["critique_history"] = critique_objects

        tool_events = design_payload.get("tool_events", []) or []
        existing_events = state["content"].get("tool_events", [])
        state["content"]["tool_events"] = (existing_events + tool_events)[-12:]

        iterations = design_payload.get("iterations")
        iteration_logs = design_payload.get("iteration_logs", [])

        summary_lines = [
            "Design workflow completed.",
            f"Iterations: {iterations}",
            f"Plan summary:\n{plan_summary}" if plan_summary else "",
            "Latest critique approved." if critique_objects and critique_objects[-1].approved else "Critique pending further refinement.",
        ]
        summary_text = "\n".join(filter(None, summary_lines))
        state["conversation"]["messages"].append(AIMessage(content=summary_text))

        # Store iteration logs in project metadata for retrieval
        state["project"]["metadata"]["design_iteration_logs"] = iteration_logs

        state["workflow"]["current_stage"] = AgentStage.CHAT
        state["workflow"]["current_intent"] = UserIntent.CHAT
        state["workflow"]["is_completed"] = True
        state["user_input"] = ""
        return Command(goto="wait_for_user_input", update=state)

    async def draw_node(self, state: CanvasState, config=None):
        updated_state = await self._invoke_stage_agent(
            AgentStage.DRAW,
            state,
            default_next_stage=AgentStage.CRITIQUE,
        )
        return Command(goto="critique_node", update=updated_state)

    async def edit_node(self, state: CanvasState, config=None):
        if not state["content"]["current_svg"]:
            logger.info("-> No current SVG to edit; falling back to draw stage.")
            state["workflow"]["current_stage"] = AgentStage.DRAW
            return Command(goto=AGENT_STAGE_MAP[AgentStage.DRAW])
        updated_state = await self._invoke_stage_agent(
            AgentStage.EDIT,
            state,
            default_next_stage=AgentStage.CRITIQUE,
        )
        return Command(goto="critique_node", update=updated_state)

    async def critique_node(self, state: CanvasState, config=None):
        logger.info("---CANVAS AGENT: CRITIQUE NODE---")

        current_svg = state["content"].get("current_svg")
        design_plan = state["content"].get("design_plan")

        if not current_svg or not current_svg.get("svg_code"):
            logger.info("-> No SVG to critique. Returning to chat.")
            state["workflow"]["current_stage"] = AgentStage.CHAT
            return Command(goto="wait_for_user_input", update=state)

        critique_messages = [
            SystemMessage(content=self._system_prompts["critique"]),
            HumanMessage(
                content=json.dumps(
                    {
                        "plan": design_plan.model_dump() if isinstance(design_plan, DesignPlan) else design_plan,
                        "svg_code": self._coerce_svg_code(current_svg),
                    },
                    ensure_ascii=False,
                )
            ),
        ]

        try:
            feedback = await self._critique_llm.ainvoke(critique_messages)
        except Exception as exc:
            logger.exception("Critique failed: %s", exc)
            state["conversation"]["messages"].append(
                AIMessage(content=f"Critique failed: {exc}."),
            )
            state["workflow"]["current_stage"] = AgentStage.CHAT
            return Command(goto="wait_for_user_input", update=state)

        state["content"]["critique_history"].append(feedback)

        critique_summary = self._format_critique_for_conversation(feedback)
        state["conversation"]["messages"].append(AIMessage(content=critique_summary))

        if feedback.approved:
            state["workflow"]["is_completed"] = True
            state["workflow"]["current_stage"] = AgentStage.CHAT
            state["workflow"]["current_intent"] = UserIntent.CHAT
            return Command(goto="wait_for_user_input", update=state)

        # Critique requests revisions; push instructions into conversation and loop to edit
        revision_prompt = "\n".join(f"- {item}" for item in feedback.revision_instructions)
        if revision_prompt:
            state["conversation"]["messages"].append(
                HumanMessage(content=f"Revision targets:\n{revision_prompt}", name="critique_notes")
            )
        state["workflow"]["current_stage"] = AgentStage.EDIT
        state["workflow"]["current_intent"] = UserIntent.MODIFY
        state["workflow"]["is_completed"] = False
        return Command(goto="edit_node", update=state)

    async def describe_node(self, state: CanvasState, config=None):
        return await self._invoke_stage_agent(AgentStage.DESCRIBE, state)

    async def pick_path_node(self, state: CanvasState, config=None):
        return await self._invoke_stage_agent(AgentStage.PICK_PATH, state)

    async def generate_image_node(self, state: CanvasState, config=None):
        return await self._invoke_stage_agent(AgentStage.GENERATE_IMAGE, state)

    def _extract_image_urls_from_messages(self, messages: List[BaseMessage]) -> List[str]:
        image_urls = []
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.content:
                if isinstance(msg.content, dict):
                    image_urls.append(msg.content.get("image_preview_url"))
                elif isinstance(msg.content, str):
                    msg_dict = json.loads(msg.content)
                    image_urls.append(msg_dict.get("image_preview_url"))
                else:
                    continue
        return image_urls
    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    async def _invoke_stage_agent(
        self,
        stage: AgentStage,
        state: CanvasState,
        *,
        default_next_stage: AgentStage = AgentStage.CHAT,
    ) -> CanvasState:
        agent = self._stage_agents.get(stage)
        if not agent:
            logger.warning("No agent configured for stage %s", stage)
            return state

        messages = self._prepare_stage_messages(stage, state)
        logger.debug("-> Invoking stage %s with %d messages", stage, len(messages))
        try:
            final_state = await agent.ainvoke({"messages": messages})
        except Exception as exc:
            logger.exception("Stage %s failed: %s", stage, exc)
            state["conversation"]["messages"].append(
                AIMessage(content=f"Stage {stage.value} failed: {exc}")
            )
            return state

        final_messages = final_state.get("messages", [])
        if final_messages:
            state["conversation"]["messages"] = final_messages
            show_messages(final_messages)

        tool_events = self._summarize_tool_events(final_messages)
        state["content"]["tool_events"] = tool_events

        svg_code = self._extract_svg_from_messages(final_messages)
        if svg_code:
            self._persist_svg_result(stage, svg_code, state)
        image_urls = self._extract_image_urls_from_messages(final_messages)
        if image_urls:
            state["content"]["image_urls"] = image_urls

        state["workflow"]["is_completed"] = False
        state["workflow"]["current_stage"] = default_next_stage
        state["workflow"]["current_intent"] = (
            UserIntent.CHAT if default_next_stage == AgentStage.CHAT else state["workflow"].get("current_intent", UserIntent.CHAT)
        )
        return state

    def _prepare_stage_messages(self, stage: AgentStage, state: CanvasState) -> List[BaseMessage]:
        messages = list(state["conversation"]["messages"])
        current_svg = state["content"].get("current_svg")
        if stage in {AgentStage.DRAW, AgentStage.EDIT, AgentStage.DESCRIBE, AgentStage.PICK_PATH, AgentStage.CRITIQUE} and current_svg:
            svg_code = self._coerce_svg_code(current_svg)
            context_message = HumanMessage(
                content=f"[CURRENT_SVG]\n{svg_code}",
                name="canvas_svg_context",
            )
            # Remove existing context to avoid duplication
            messages = [m for m in messages if not (isinstance(m, HumanMessage) and getattr(m, "name", "") == "canvas_svg_context")]
            if messages and isinstance(messages[-1], HumanMessage):
                messages.insert(-1, context_message)
            else:
                messages.append(context_message)
        if stage in {AgentStage.DRAW, AgentStage.EDIT}:
            design_plan = state["content"].get("design_plan")
            if design_plan:
                plan_text = self._format_plan_for_conversation(design_plan)
                plan_context = HumanMessage(content=f"[PLAN]\n{plan_text}", name="design_plan_context")
                messages = [m for m in messages if not (isinstance(m, HumanMessage) and getattr(m, "name", "") == "design_plan_context")]
                messages.insert(-1, plan_context) if messages else messages.append(plan_context)    
        return messages

    def _summarize_tool_events(self, messages: List[BaseMessage]) -> List[str]:
        events: List[str] = []
        for message in messages:
            if isinstance(message, ToolMessage):
                content = message.content
                if isinstance(content, str):
                    events.append(content[:300])
                elif isinstance(content, list):
                    text = " ".join(
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in content
                    )
                    events.append(text[:300])
                else:
                    events.append(str(content)[:300])
        return events[-5:]

    def _extract_svg_from_messages(self, messages: List[BaseMessage]) -> Optional[str]:
        for message in reversed(messages):
            candidate = None
            if isinstance(message, ToolMessage):
                candidate = self._extract_svg_from_tool_message(message.content)
            elif isinstance(message, AIMessage) and isinstance(message.content, str):
                text = message.content.strip()
                if text.startswith("<svg"):
                    candidate = text
            if candidate:
                return candidate
        return None

    @staticmethod
    def _extract_svg_from_tool_message(content: Any) -> Optional[str]:
        if isinstance(content, str):
            text = content.strip()
            if text.startswith("<svg"):
                return text
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return None
        elif isinstance(content, dict):
            parsed = content
        else:
            return None

        if isinstance(parsed, dict):
            value = parsed.get("value") or parsed.get("svg") or parsed.get("result")
            if isinstance(value, str) and value.strip().startswith("<svg"):
                return value
        return None

    @staticmethod
    def _coerce_svg_code(svg: Any) -> str:
        if isinstance(svg, SvgArtwork):
            return svg.svg_code
        if isinstance(svg, dict):
            return svg.get("svg_code", "")
        return str(svg)

    @staticmethod
    def _format_plan_for_conversation(plan: DesignPlan | dict | None) -> str:
        if plan is None:
            return ""
        if isinstance(plan, DesignPlan):
            plan_dict = plan.model_dump()
        else:
            plan_dict = plan
        sections = [
            f"Concept: {plan_dict.get('concept_summary', '')}",
            "Key Elements:\n" + "\n".join(f"- {el}" for el in plan_dict.get("visual_elements", []) or []),
            f"Layout strategy: {plan_dict.get('layout_strategy', '')}",
            "Execution steps:\n" + "\n".join(
                f"{idx+1}. {step}" for idx, step in enumerate(plan_dict.get("execution_steps", []) or [])
            ),
        ]
        return "\n".join(section for section in sections if section.strip())

    @staticmethod
    def _format_critique_for_conversation(feedback: CritiqueFeedback) -> str:
        bullet_revision = "\n".join(f"- {item}" for item in feedback.revision_instructions) or "None"
        target_info = ", ".join(feedback.svg_targets) or "General"
        status = "✅ Approved" if feedback.approved else "⚠️ Needs revisions"
        return (
            f"{status}\n"
            f"Summary: {feedback.overall_feedback}\n"
            f"Targets: {target_info}\n"
            f"Next actions:\n{bullet_revision}"
        )

    def _persist_svg_result(self, stage: AgentStage, svg_code: str, state: CanvasState) -> None:
        svg_artwork = SvgArtwork(svg_code=svg_code, elements=[], metadata={"stage": stage.value})
        state["content"]["current_svg"] = svg_artwork
        state["content"]["svg_history"].append(svg_artwork)

        project_dir = state["project"].get("project_dir")
        if not project_dir:
            return

        os.makedirs(project_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        svg_path = os.path.join(project_dir, f"artwork_{timestamp}.svg")
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_code)
        png_path = svg_to_png(svg_path)
        if png_path:
            state["project"]["saved_files"].append(png_path)
        logger.info("-> Saved SVG to %s", svg_path)

    # ---------------------------------------------------------------------
    # Graph wiring
    # ---------------------------------------------------------------------
    def build_graph(self):
        graph = StateGraph(CanvasState)

        graph.add_node("init_context_node", self._init_context_node)
        graph.add_node("wait_for_user_input", self.wait_for_user_input)
        graph.add_node("router_node", self.router_node)
        graph.add_node("plan_node", self.plan_node)
        graph.add_node("chat_node", self.chat_node)
        graph.add_node("draw_node", self.draw_node)
        graph.add_node("edit_node", self.edit_node)
        graph.add_node("critique_node", self.critique_node)
        graph.add_node("describe_node", self.describe_node)
        graph.add_node("pick_path_node", self.pick_path_node)
        graph.add_node("generate_image_node", self.generate_image_node)

        graph.set_entry_point("init_context_node")
        graph.add_edge("init_context_node", "wait_for_user_input")
        graph.add_edge("wait_for_user_input", "router_node")
        graph.add_edge("router_node", "chat_node")  # default fallback
        graph.add_edge("chat_node", "wait_for_user_input")
        graph.add_edge("plan_node", "draw_node")
        graph.add_edge("draw_node", "critique_node")
        graph.add_edge("critique_node", "wait_for_user_input")
        graph.add_edge("critique_node", "edit_node")
        graph.add_edge("edit_node", "critique_node")
        graph.add_edge("describe_node", "wait_for_user_input")
        graph.add_edge("pick_path_node", "wait_for_user_input")
        graph.add_edge("generate_image_node", "wait_for_user_input")

        return graph
