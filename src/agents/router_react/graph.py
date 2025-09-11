"""
Router React agent implementation - A multi-agent system with router and specialized react agents.
"""

import logging
from typing import Optional
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START
from langgraph.types import Command, interrupt
from langgraph.prebuilt import create_react_agent

from ..base.mas_graph import MultiAgentBase
from src.config.manager import MultiAgentConfig
from src.utils.input_processor import create_multimodal_message
from src.utils.fuzzy_match import find_best_match_from_dict
from src.utils import (
    log_router,
    log_agent,
    log_tool,
    log_flow,
    log_state,
    log_warning,
    log_error,
    init_default_logger,
)
from .schema import RouterReactState
from .prompt import ROUTER_SYSTEM_PROMPT, RESEARCH_AGENT_PROMPT, ANALYSIS_AGENT_PROMPT
from .config import ThreadConfiguration

logger = logging.getLogger(__name__)
# Initialize the colored logger for this module
init_default_logger(__name__)


class RouterReactAgent(MultiAgentBase):
    """
    Router React multi-agent system with handoff mechanism.

    This system consists of:
    - Router Agent: Coordinates tasks and manages handoffs using think_tool, plan_tool, handoff_to_other_agent, task_done
    - Research Agent: Specialized for information gathering and research
    - Analysis Agent: Specialized for data analysis and insights
    """

    name = "router_react"
    description = (
        "A multi-agent system with router coordinating specialized react agents."
    )

    def __init__(self, multi_agent_config: MultiAgentConfig):
        """Initialize the router react multi-agent system.

        Args:
            multi_agent_config: Configuration object containing multiple agent configurations.
        """
        super().__init__(multi_agent_config)
        self._load_system_prompts()
        log_router(f"{self.name} initialized successfully with handoff mechanism")

    def _load_system_prompts(self):
        """Load the system prompts for different agents."""
        self._system_prompts = {
            "router": ROUTER_SYSTEM_PROMPT,
            "research_agent": RESEARCH_AGENT_PROMPT,
            "analysis_agent": ANALYSIS_AGENT_PROMPT,
        }

    def build_graph(self) -> StateGraph:
        """Build the router react multi-agent graph with handoff mechanism.

        The graph flow:
        1. router_node: Router agent coordinates tasks and manages handoffs
        2. specialized_agent_node: Execute tasks from specialized agents
        3. human_node: Handle user interaction

        Returns:
            StateGraph: The constructed graph for the multi-agent system.
        """
        graph = StateGraph(state_schema=RouterReactState)

        # Add nodes
        graph.add_node("router_node", self.router_node)
        graph.add_node("specialized_agent_node", self.specialized_agent_node)
        graph.add_node("human_node", self.human_node)

        # Set up flow
        graph.add_edge(START, "router_node")

        return graph

    async def router_node(
        self, state: RouterReactState, config: Optional[RunnableConfig] = None
    ) -> Command:
        """
        Router agent node that coordinates tasks and manages handoffs.

        Handles multiple scenarios:
        1. think_tool call -> continue thinking, goto self
        2. plan_tool call -> store plan and start executing steps via handoffs
        3. handoff_to_other_agent -> delegate to specialized agent
        4. task_done -> complete task and go to human
        5. simple chat -> goto human_node
        """
        log_router("=== ROUTER NODE ACTIVATED ===")

        # Log current state
        current_plan = state.get("current_plan")
        current_step = state.get("current_step")
        if current_plan:
            log_state(
                f"Active plan: '{current_plan.get('thought', 'N/A')}' - Step {current_step + 1 if current_step is not None else '?'}/{len(current_plan.get('steps', []))}"
            )
        else:
            log_state("No active plan")

        # Get router agent configuration
        router_tools = self._agent_tools_registry.get("router", [])
        # Configure LLM to use only one tool at a time
        router_llm = self._agent_llm_registry.get("router").bind_tools(
            router_tools,
            tool_choice="auto",  # Let model choose but will only process first tool
        )

        # Prepare system message with context
        configurable = ThreadConfiguration.from_runnable_config(config)

        # Use the router system prompt
        router_prompt = self._system_prompts["router"].format(
            user_name=configurable.user_name or "用户"
        )

        system_message = SystemMessage(content=router_prompt)
        messages = [system_message] + state["messages"]

        log_router(f"Processing {len(state['messages'])} messages")

        # Get router agent response
        log_router("Invoking router LLM...")
        response = await router_llm.ainvoke(messages)

        # Log response details
        if response.content:
            content_preview = (
                response.content[:100] + "..."
                if len(response.content) > 100
                else response.content
            )
            log_router(f"Router response: {content_preview}")

        if response.tool_calls:
            tool_names = [tc["name"] for tc in response.tool_calls]
            log_router(f"Router wants to use tools: {tool_names}")

        # Handle tool calls - only process the first tool call to ensure single tool usage
        if response.tool_calls:
            # Only process the first tool call
            tool_call = response.tool_calls[0]
            tool_name = tool_call["name"]

            log_router(f"Processing single tool call: {tool_name}")

            if len(response.tool_calls) > 1:
                log_warning(
                    f"Multiple tool calls detected ({len(response.tool_calls)}), only processing the first: {tool_name}"
                )

            # Include the assistant message with tool calls, then the tool message
            tool_messages = [response]

            if tool_name == "think_tool":
                # Handle think_tool: add tool message and continue router self-loop
                reflection = tool_call["args"].get("reflection", "")
                tool_message = ToolMessage(
                    content=f"思考内容：{reflection}", tool_call_id=tool_call["id"]
                )
                tool_messages.append(tool_message)
                log_tool(
                    "think_tool",
                    f"Reflection: {reflection[:80]}..."
                    if len(reflection) > 80
                    else reflection,
                )

            elif tool_name == "plan_tool":
                # Handle plan_tool: store plan and initialize plan execution
                thought = tool_call["args"].get("thought", "")
                steps = tool_call["args"].get("steps", [])

                plan_data = {"thought": thought, "steps": steps}
                tool_message = ToolMessage(
                    content=f"已制定计划：\n思路：{thought}\n步骤：{steps}",
                    tool_call_id=tool_call["id"],
                )
                tool_messages.append(tool_message)

                log_tool("plan_tool", f"Created plan with {len(steps)} steps")
                log_state(
                    f"Plan thought: {thought[:60]}..." if len(thought) > 60 else thought
                )
                for i, step in enumerate(steps):
                    log_state(f"Step {i+1}: {step[:50]}..." if len(step) > 50 else step)

                # Add context for first step execution if plan has steps
                if steps:
                    first_step_context = HumanMessage(
                        content=f"现在开始执行计划的第 1 步（共 {len(steps)} 步）：\n"
                        f"步骤内容：{steps[0]}\n"
                        f"计划总体思路：{thought}\n"
                        f"请开始执行第一步。"
                    )
                    tool_messages.append(first_step_context)

                    log_flow(
                        "router_node",
                        "router_node",
                        "plan created, continuing with step 1",
                    )
                    log_state("Plan execution initialized - Step 1 context added")

                    # Initialize plan execution state
                    return Command(
                        goto="router_node",
                        update={
                            "messages": tool_messages,
                            "current_plan": plan_data,
                            "current_step": 0,
                            "step_results": [],
                            "agent_mode": "router",
                        },
                    )

            elif tool_name == "handoff_to_other_agent":
                # Handle handoff: delegate to specialized agent
                agent_name = tool_call["args"]["agent_name"]
                task_description = tool_call["args"]["task_description"]

                tool_message = ToolMessage(
                    content=f"正在将任务交给 {agent_name}：{task_description}",
                    tool_call_id=tool_call["id"],
                )
                tool_messages.append(tool_message)

                log_tool("handoff_to_other_agent", f"Delegating to {agent_name}")
                log_state(
                    f"Task: {task_description[:60]}..."
                    if len(task_description) > 60
                    else task_description
                )
                log_flow(
                    "router_node", "specialized_agent_node", f"handoff to {agent_name}"
                )

                return Command(
                    goto="specialized_agent_node",
                    update={
                        "task": task_description,
                        "messages": tool_messages,
                        "agent_mode": agent_name,
                        "sub_agent": agent_name,
                    },
                )

            elif tool_name == "step_done":
                # Handle step completion - advance to next step in plan
                step_description = tool_call["args"]["step_description"]
                step_result = tool_call["args"]["step_result"]

                tool_message = ToolMessage(
                    content=f"步骤已完成：{step_description}\n结果：{step_result}",
                    tool_call_id=tool_call["id"],
                )
                tool_messages.append(tool_message)

                log_tool(
                    "step_done",
                    f"Step completed: {step_description[:50]}..."
                    if len(step_description) > 50
                    else step_description,
                )
                log_state(
                    f"Step result: {step_result[:60]}..."
                    if len(step_result) > 60
                    else step_result
                )

                # Update step results and advance to next step
                current_plan = state.get("current_plan")
                current_step = state.get("current_step", 0)
                step_results = state.get("step_results", [])

                if current_plan:
                    step_results.append(f"步骤 {current_step + 1}: {step_result}")
                    next_step = current_step + 1
                    plan_steps = current_plan.get("steps", [])

                    # Add context for next step execution if more steps exist
                    if next_step < len(plan_steps):
                        next_step_description = plan_steps[next_step]
                        context_message = HumanMessage(
                            content=f"当前正在执行计划的第 {next_step + 1} 步（共 {len(plan_steps)} 步）：\n"
                            f"步骤内容：{next_step_description}\n"
                            f"计划总体思路：{current_plan.get('thought', '')}\n"
                            f"前面步骤的结果：{chr(10).join(step_results)}"
                        )
                        tool_messages.append(context_message)
                        log_state(
                            f"Advancing to step {next_step + 1}/{len(plan_steps)}: {next_step_description[:50]}..."
                        )
                    else:
                        # All steps completed, add completion context
                        completion_message = HumanMessage(
                            content=f"计划执行完成！所有 {len(plan_steps)} 个步骤已完成。\n"
                            f"计划思路：{current_plan.get('thought', '')}\n"
                            f"执行结果：{chr(10).join(step_results)}\n"
                            f"请总结计划执行结果并回复用户。"
                        )
                        tool_messages.append(completion_message)
                        log_state(
                            f"🎉 Plan execution complete! All {len(plan_steps)} steps finished"
                        )
                        log_flow(
                            "router_node",
                            "router_node",
                            "plan completed, preparing summary",
                        )

                    return Command(
                        goto="router_node",
                        update={
                            "messages": tool_messages,
                            "current_step": next_step
                            if next_step < len(plan_steps)
                            else None,
                            "step_results": step_results,
                            # Clear plan state if all steps completed
                            "current_plan": None
                            if next_step >= len(plan_steps)
                            else current_plan,
                            "agent_mode": "router",
                        },
                    )
                else:
                    # No active plan, just continue
                    log_warning("step_done called but no active plan found")
                    log_flow("router_node", "router_node", "no active plan")
                    return Command(
                        goto="router_node",
                        update={"messages": tool_messages, "agent_mode": "router"},
                    )

            else:
                log_error(f"Unknown tool: {tool_name}")
                raise ValueError(f"Unknown tool: {tool_name}")

            # Continue router loop with tool messages
            log_flow("router_node", "router_node", "processing tool results")
            return Command(
                goto="router_node",
                update={"messages": tool_messages, "agent_mode": "router"},
            )

        else:
            # Simple chat response - go to human
            log_router("No tool calls - providing direct response")
            log_flow("router_node", "human_node", "direct response")
            return Command(
                goto="human_node",
                update={"messages": [response], "agent_mode": "router"},
            )

    async def specialized_agent_node(
        self, state: RouterReactState, config: Optional[RunnableConfig] = None
    ) -> Command:
        """
        Specialized agent node that executes one task step and returns to router.

        Each specialized agent performs only one step of work before returning
        control to the router for coordination.
        """
        log_agent("SPECIALIZED", "=== SPECIALIZED AGENT NODE ACTIVATED ===")

        sub_agent = state.get("sub_agent")
        task = state.get("task")

        if sub_agent is None or task is None:
            log_error(
                f"Missing required parameters - sub_agent: {sub_agent}, task: {task}"
            )
            log_flow(
                "specialized_agent_node", "router_node", "error - missing parameters"
            )
            return Command(
                goto="router_node",
                update={
                    "messages": [
                        AIMessage(
                            content=f"错误：缺少子智能体或任务信息。sub_agent: {sub_agent}, task: {task}"
                        )
                    ],
                    "agent_mode": "router",
                },
            )

        log_agent(
            sub_agent, f"Received task: {task[:60]}..." if len(task) > 60 else task
        )

        # Check if the requested agent exists in the registry
        available_agents = list(self._agent_llm_registry.keys())
        matched_agent = find_best_match_from_dict(
            sub_agent, {agent: agent for agent in available_agents}
        )

        log_state(f"Available agents: {available_agents}")

        if matched_agent is None:
            log_warning(f"No matching agent found for '{sub_agent}'")
            error_message = (
                f"未找到匹配的专门智能体 '{sub_agent}'。可用智能体: {', '.join(available_agents)}"
            )
            log_flow("specialized_agent_node", "router_node", "no matching agent found")
            return Command(
                goto="router_node",
                update={
                    "messages": [AIMessage(content=error_message)],
                    "agent_mode": "router",
                },
            )

        # Log the match if it's different from the original request
        if matched_agent != sub_agent:
            log_agent(sub_agent, f"Matched to agent: {matched_agent}")
        else:
            log_agent(sub_agent, "Exact agent match found")

        # Get specialized agent configuration
        agent_llm = self._agent_llm_registry.get(matched_agent)
        agent_tools = self._agent_tools_registry.get(matched_agent)

        log_agent(matched_agent, f"Configuring with tools: {agent_tools}")

        # Create specialized react agent
        specialized_agent = create_react_agent(
            model=agent_llm,
            tools=agent_tools,
            prompt=self._system_prompts[matched_agent],
        )

        # Prepare task message for the specialized agent
        task_message = HumanMessage(content=f"请执行以下任务：{task}")

        log_agent(matched_agent, "Starting task execution...")

        # Get specialized agent response (single step execution)
        output = await specialized_agent.ainvoke({"messages": [task_message]})
        response = output.get("messages", [])[-1]

        # Log response details
        if hasattr(response, "content") and response.content:
            content_preview = (
                response.content[:100] + "..."
                if len(response.content) > 100
                else response.content
            )
            log_agent(matched_agent, f"Task completed: {content_preview}")
        else:
            log_agent(matched_agent, "Task completed with no text response")

        log_flow(
            "specialized_agent_node", "router_node", f"{matched_agent} task completed"
        )

        # Return to router with the result
        return Command(
            goto="router_node",
            update={
                "messages": [response],
                "agent_mode": "router",
                # Clear sub-agent state
                "sub_agent": None,
                "task": None,
            },
        )

    async def human_node(
        self, state: RouterReactState, config: Optional[RunnableConfig] = None
    ) -> Command:
        """Handle human interaction."""
        log_router("=== HUMAN NODE ACTIVATED ===")

        user_id = state.get("user_id")
        log_router(f"💬 Waiting for user input (user_id: {user_id})")

        # Interrupt for user input
        feedback = interrupt("你的回答：")

        # Create multimodal message from feedback
        human_message = create_multimodal_message(**feedback)

        # Log user input details
        if hasattr(human_message, "content") and human_message.content:
            content_preview = (
                human_message.content[:80] + "..."
                if len(human_message.content) > 80
                else human_message.content
            )
            log_router(f"📝 User input received: {content_preview}")

        log_flow("human_node", "router_node", "new user input")
        log_state("Resetting conversation state for new interaction")

        # Go back to router node with user input for new conversation
        return Command(
            goto="router_node",
            update={
                "messages": [human_message],
                "agent_mode": "router",
                # Reset state for new conversation
                "current_plan": None,
                "current_step": None,
                "step_results": None,
                "sub_agent": None,
                "task": None,
            },
        )
