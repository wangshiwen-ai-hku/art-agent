"""Design orchestration tools for the canvas agent platform."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from typing import TYPE_CHECKING

from src.config.manager import config
from src.infra.tools.draw_canvas_tools import draw_agent_with_tool
from src.infra.tools.edit_canvas_tools import edit_agent_with_tool
from src.utils.print_utils import show_messages

if TYPE_CHECKING:
    # Imported only for type checkers to avoid circular imports at runtime.
    from src.service.canvas_agent.schema import CritiqueFeedback, DesignPlan

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
_PROMPT_DIR = Path(__file__).resolve().parents[3] / "service" / "canvas_agent" / "prompt"

PLANNER_PROMPT = (
    "You are an award-winning brand designer. Read the task description and "
    "produce a structured design plan. The plan must be JSON with keys: "
    "concept_summary (string), visual_elements (array of short strings), "
    "layout_strategy (string), execution_steps (array of imperative strings). "
    "Focus on geometric hints useful for SVG work."
)

CRITIC_PROMPT = (
    "You are a meticulous creative director. Evaluate the provided SVG against "
    "the design plan and the latest instructions. Return JSON with keys: "
    "approved (bool), overall_feedback, revision_instructions (array of strings), "
    "svg_targets (array of ids/index hints). Base your decision on coherence, "
    "balance, and whether the brief is satisfied."
)

# ---------------------------------------------------------------------------
# LLM initialisation helpers
# ---------------------------------------------------------------------------
_planner_llm = None
_critic_llm = None


def _get_planner_llm():
    global _planner_llm
    if _planner_llm is None:
        agent_cfg = config.get_agent_config("design_loop_agent", "core")
        _planner_llm = init_chat_model(**agent_cfg.model.__dict__)
    # Import here to avoid triggering package-level imports when this module
    # is imported at runtime (breaks circular import with src.service).
    from src.service.canvas_agent.schema import DesignPlan

    return _planner_llm.with_structured_output(DesignPlan)


def _get_critic_llm():
    global _critic_llm
    if _critic_llm is None:
        agent_cfg = config.get_agent_config("design_agent", "core")
        _critic_llm = init_chat_model(**agent_cfg.model.__dict__)
    # Defer import for the same reason as planner LLM.
    from src.service.canvas_agent.schema import CritiqueFeedback

    return _critic_llm.with_structured_output(CritiqueFeedback)


# ---------------------------------------------------------------------------
# Memory container
# ---------------------------------------------------------------------------
@dataclass
class DesignMemory:
    task: str
    plan: Optional[DesignPlan] = None
    svg_versions: List[str] = field(default_factory=list)
    critiques: List[CritiqueFeedback] = field(default_factory=list)
    tool_events: List[str] = field(default_factory=list)

    def record_svg(self, svg: str):
        if svg:
            self.svg_versions.append(svg)

    def extend_tool_events(self, events: List[str]):
        for event in events:
            if event and event not in self.tool_events:
                self.tool_events.append(event)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "plan": self.plan.model_dump() if self.plan else None,
            "svg_versions": self.svg_versions,
            "critiques": [c.model_dump() for c in self.critiques],
            "tool_events": self.tool_events[-12:],
        }


# ---------------------------------------------------------------------------
# Helper formatting utilities
# ---------------------------------------------------------------------------

def _plan_to_text(plan: Optional[DesignPlan]) -> str:
    if not plan:
        return ""
    elements = "\n".join(f"- {item}" for item in plan.visual_elements)
    steps = "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(plan.execution_steps))
    return (
        f"Concept: {plan.concept_summary}\n"
        f"Visual elements:\n{elements}\n"
        f"Layout strategy: {plan.layout_strategy}\n"
        f"Execution steps:\n{steps}"
    )


def _critique_to_revision(critique: CritiqueFeedback) -> str:
    notes = "\n".join(f"- {item}" for item in critique.revision_instructions)
    return f"Feedback: {critique.overall_feedback}\nRequired changes:\n{notes}"


async def _generate_plan(task_description: str, reference_notes: str = "") -> DesignPlan:
    planner = _get_planner_llm()
    messages = [
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=f"TASK:\n{task_description}\n\nREFERENCES:\n{reference_notes}"),
    ]
    plan = await planner.ainvoke(messages)
    logger.info("[design_tools] Generated plan: %s", plan)
    return plan


async def _run_critique(
    plan: DesignPlan,
    svg_code: str,
    iteration: int,
    revision_history: List[CritiqueFeedback],
    extra_notes: str = "",
) -> CritiqueFeedback:
    critic = _get_critic_llm()
    memory_excerpt = "\n\n".join(
        f"Iteration {idx + 1}: {crit.overall_feedback}"
        for idx, crit in enumerate(revision_history[-3:])
    )
    payload = {
        "iteration": iteration,
        "plan": plan.model_dump(),
        "svg": svg_code,
        "prior_critiques": memory_excerpt,
        "designer_notes": extra_notes,
    }
    messages = [
        SystemMessage(content=CRITIC_PROMPT),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ]
    critique = await critic.ainvoke(messages)
    logger.info("[design_tools] Critique (iteration %s): %s", iteration, critique)
    return critique


def _merge_tool_events(memory: DesignMemory, payload: Dict[str, Any]):
    events = payload.get("tool_events") or []
    if isinstance(events, list):
        memory.extend_tool_events([str(e) for e in events])


def _extract_svg_from_payload(payload: Dict[str, Any]) -> str:
    for key in ("svg", "value"):
        if key in payload and isinstance(payload[key], str) and payload[key].strip().startswith("<svg"):
            return payload[key]
    return ""


# ---------------------------------------------------------------------------
# Public tool: orchestrated design workflow
# ---------------------------------------------------------------------------
@tool
async def design_agent_with_tool(
    task_description: str,
    width: int = 512,
    height: int = 512,
    max_iterations: int = 3,
    reference_notes: str = "",
    initial_svg: str = "",
) -> str:
    """End-to-end SVG design loop combining plan, draw, critique, and edit."""

    logger.info(
        "[design_agent_with_tool] task=%s width=%s height=%s iterations=%s",
        task_description,
        width,
        height,
        max_iterations,
    )

    memory = DesignMemory(task=task_description)
    plan = await _generate_plan(task_description, reference_notes)
    memory.plan = plan
    plan_text = _plan_to_text(plan)

    current_svg = initial_svg if initial_svg.strip().startswith("<svg") else ""
    critique_notes = ""
    targets = ""
    iteration_logs: List[Dict[str, Any]] = []

    for iteration in range(1, max_iterations + 1):
        if not current_svg:
            # draw_agent_with_tool is a tool (BaseTool); call its async invoke interface
            draw_result = await draw_agent_with_tool.ainvoke(
                {
                    "task_description": task_description,
                    "width": width,
                    "height": height,
                    "plan_context": plan_text,
                    "critique_notes": critique_notes,
                }
            )
            draw_payload = json.loads(draw_result)
            _merge_tool_events(memory, draw_payload)
            current_svg = _extract_svg_from_payload(draw_payload)
            memory.record_svg(current_svg)
            explanation = draw_payload.get("explanation", "")
            iteration_logs.append(
                {
                    "iteration": iteration,
                    "mode": "draw",
                    "explanation": explanation,
                    "summary": draw_payload.get("summary", {}),
                }
            )
        else:
            # edit_agent_with_tool is a tool (BaseTool); call via its async invoke API
            edit_result = await edit_agent_with_tool.ainvoke(
                {
                    "task_description": task_description,
                    "original_svg_code": current_svg,
                    "revision_notes": critique_notes,
                    "target_elements": targets,
                    "plan_context": plan_text,
                }
            )
            edit_payload = json.loads(edit_result)
            _merge_tool_events(memory, edit_payload)
            current_svg = _extract_svg_from_payload(edit_payload) or current_svg
            memory.record_svg(current_svg)
            explanation = edit_payload.get("explanation", "")
            iteration_logs.append(
                {
                    "iteration": iteration,
                    "mode": "edit",
                    "explanation": explanation,
                    "summary": edit_payload.get("summary", {}),
                }
            )

        show_messages([AIMessage(content=explanation)])

        critique = await _run_critique(
            plan=plan,
            svg_code=current_svg,
            iteration=iteration,
            revision_history=memory.critiques,
            extra_notes=critique_notes,
        )
        memory.critiques.append(critique)

        iteration_logs[-1]["critique"] = critique.model_dump()

        if critique.approved:
            logger.info("[design_tools] Design approved on iteration %s", iteration)
            break

        critique_notes = _critique_to_revision(critique)
        targets = ", ".join(critique.svg_targets)

    result = {
        "type": "design_result",
        "svg": current_svg,
        "value": current_svg,
        "plan": plan.model_dump(),
        "critique_history": [c.model_dump() for c in memory.critiques],
        "iterations": len(memory.svg_versions),
        "tool_events": memory.tool_events[-12:],
        "iteration_logs": iteration_logs,
        "memory": memory.to_dict(),
    }

    return json.dumps(result)


if __name__ == "__main__":
    async def _demo():
        result = await design_agent_with_tool.ainvoke({
            "task_description": "Design a modern monoline skyline for a smart-city startup with a sunrise motif.",
            "width": 512,
            "height": 512,
            "max_iterations": 2,
        })
        print(result)

    asyncio.run(_demo())
