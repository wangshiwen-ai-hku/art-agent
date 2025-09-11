import json
import logging
import os
from typing import Annotated, Literal
import aiohttp

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.types import Command, interrupt
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import create_react_agent

from .config import ThreadConfiguration
from .schema import Plan, StepType, State
from src.utils.json_utils import repair_json_output

from datetime import datetime


logger = logging.getLogger(__name__)


from ..base import MultiAgentBase
from src.config.manager import MultiAgentConfig
from .prompt import (
    RESEARCHER_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    BACKGROUND_INVESTIGATOR_SYSTEM_PROMPT,
    GENERATOR_SYSTEM_PROMPT,
    RESOURCE_QUERY_SYSTEM_PROMPT,
)


class DeepResearcher(MultiAgentBase):
    """Deep Researcher agent that can be used to do research."""

    name = "deep_researcher"
    description = "Deep Researcher agent that can be used to do research."

    def __init__(self, multi_agent_config: MultiAgentConfig):
        super().__init__(multi_agent_config)
        self._load_system_prompts()

    def _load_system_prompts(self):
        """Load the system prompt for the agent."""
        # Store all specialized prompts for different roles
        self._system_prompts = {
            "researcher": RESEARCHER_SYSTEM_PROMPT,
            "planner": PLANNER_SYSTEM_PROMPT,
            "background_investigator": BACKGROUND_INVESTIGATOR_SYSTEM_PROMPT,
            "generator": GENERATOR_SYSTEM_PROMPT,
            "resource_query": RESOURCE_QUERY_SYSTEM_PROMPT,
        }

    def build_graph(self):
        """Build the agent graph.

        Returns:
            StateGraph: The constructed graph for the agent.
        """
        builder = StateGraph(State)
        builder.add_edge(START, "background_investigator")
        builder.add_node("background_investigator", self.background_investigation_node)
        builder.add_node("planner", self.planner_node)
        builder.add_node("resource_query", self.resource_query_node)
        builder.add_node("research_team", self.research_team_node)
        builder.add_node("researcher", self.researcher_node)
        builder.add_node("generator", self.generator_node)
        builder.add_node("human_feedback", self.human_feedback_node)
        return builder

    def apply_prompt_template(
        self,
        system_prompt: str,
        state: State,
        system_prompt_only: bool = False,
        config: RunnableConfig = None,
    ) -> list:
        """
        Apply template variables to a prompt template and return formatted messages.

        Args:
            system_prompt: The system prompt template string
            state: Current agent state containing variables to substitute
            system_prompt_only: Whether to return only system prompt or include messages
            config: RunnableConfig containing configuration parameters

        Returns:
            List of messages with the system prompt as the first message
        """
        # Convert state to dict for template rendering
        if state.get("locale", "None") == "None":
            logger.warning("Locale is not set in state, using default locale: zh-CN")

        state_vars = {
            "CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S %z"),
            **state,
        }
        # Add configuration variables if provided
        if config:
            # Convert RunnableConfig to Configuration object to get proper typed access
            configuration = ThreadConfiguration.from_runnable_config(config)
            config_dict = configuration.to_dict()
            logger.info(f"Configuration dict: {config_dict}")
            state_vars.update(config_dict)

        # Format the system prompt with the state and config variables
        system_prompt = system_prompt.format(**state_vars)

        logger.info(f"{self.name} system prompt: {system_prompt}")
        if system_prompt_only:
            return [{"role": "system", "content": system_prompt}]
        else:
            return [{"role": "system", "content": system_prompt}] + state["messages"]

    async def planner_node(
        self, state: State, config: RunnableConfig
    ) -> Command[Literal["human_feedback"]]:
        """Planner node that generate the full plan."""
        logger.info("Planner generating full plan")
        llm = self._agent_llm_registry["planner"].with_structured_output(
            Plan,
            method="json_mode",
        )

        messages = self.apply_prompt_template(
            self._system_prompts["planner"],
            state,
            system_prompt_only=False,
            config=config,
        )

        full_response = ""
        response = llm.invoke(messages)
        full_response = response.model_dump_json(indent=4, exclude_none=True)
        logger.debug(f"Current state messages: {state['messages']}")
        logger.info(f"Planner response: {full_response}")

        try:
            curr_plan = json.loads(repair_json_output(full_response))
        except json.JSONDecodeError:
            logger.warning("Planner response is not a valid JSON")
            return Command(goto="__end__")

        return Command(
            update={
                "messages": [AIMessage(content=full_response, name="planner")],
                "current_plan": full_response,
            },
            goto="human_feedback",
        )

    async def background_investigation_node(
        self, state: State, config: RunnableConfig
    ) -> Command[Literal["planner"]]:
        logger.info("background investigation node is running.")
        return await self._setup_and_execute_agent_step(
            state, config, "background_investigator"
        )

    def human_feedback_node(
        self, state: State, config: RunnableConfig
    ) -> Command[Literal["planner", "resource_query"]]:
        configurable = ThreadConfiguration.from_runnable_config(config)
        current_plan = state.get("current_plan", "")
        # check if the plan is auto accepted
        auto_accepted_plan = configurable.auto_accepted_plan
        if not auto_accepted_plan:
            feedback = interrupt(
                {
                    "message": "请查看并修改教学内容规划：您可以在对话框中给出修改建议，也可以直接编辑教学内容规划。",
                    "plan": current_plan,
                }
            )
            if feedback and (
                (isinstance(feedback, dict) and feedback.get("type") == "direct_edit")
                or (
                    isinstance(feedback, str)
                    and feedback.upper().startswith("[DIRECT_EDIT]")
                )
            ):
                edited_plan = (
                    str(feedback["content"])
                    if isinstance(feedback, dict)
                    else feedback[len("[DIRECT_EDIT]") :].strip()
                )
                current_plan = edited_plan
            elif feedback and (
                (isinstance(feedback, dict) and feedback.get("type") == "accept")
                or (isinstance(feedback, str) and "accept" in feedback.lower())
            ):
                logger.info("Plan is accepted by user.")

            else:
                raise TypeError(f"Interrupt value of {feedback} is not supported.")

        # if the plan is accepted, run the following node
        try:
            current_plan = repair_json_output(current_plan)
            # parse the plan
            new_plan = json.loads(current_plan)

        except json.JSONDecodeError:
            logger.warning("Planner response is not a valid JSON")
            return Command(goto="__end__")

        return Command(
            update={
                "current_plan": Plan.model_validate(new_plan),
            },
            goto="resource_query",
        )

    async def resource_query_node(
        self, state: State, config: RunnableConfig
    ) -> Command[Literal["research_team"]]:
        logger.info("Resource query node is running.")
        configurable = ThreadConfiguration.from_runnable_config(config)
        task_description = state.get("task_description", "")

        # Initialize variables for API data
        api_data = None
        formatted_response = ""

        # Query for the knowledge maps and experiments
        try:
            async with aiohttp.ClientSession() as session:
                api_url = f"https://test2.actwise.xyz/query-api/match/interactive?query={task_description}"
                async with session.get(api_url) as response:
                    if response.status == 200:
                        api_data = await response.json()
                        logger.info(
                            f"API query successful for task: {task_description}"
                        )
                    else:
                        logger.warning(
                            f"API query failed with status {response.status}"
                        )
        except Exception as e:
            logger.error(f"Error querying API: {e}")

        # If API data is available, use LLM to analyze and summarize resources
        if api_data and (api_data.get("knowledge_maps") or api_data.get("experiments")):
            # Prepare the raw API data for LLM analysis
            raw_data_content = "## 检索到的教学资源\n\n"

            if api_data.get("knowledge_maps"):
                raw_data_content += "### 知识地图\n"
                for km in api_data["knowledge_maps"]:
                    name = km.get("name", "")
                    url = km.get("url", "")
                    raw_data_content += f"- **{name}**: {url}\n"
                raw_data_content += "\n"

            if api_data.get("experiments"):
                raw_data_content += "### 实验资源\n"
                for exp in api_data["experiments"]:
                    name = exp.get("name", "")
                    url = exp.get("url", "")
                    raw_data_content += f"- **{name}**: {url}\n"
                raw_data_content += "\n"

            # Use LLM to analyze and provide teaching guidance
            try:
                # Apply the resource filter prompt template
                messages = self.apply_prompt_template(
                    self._system_prompts["resource_query"],
                    state,
                    system_prompt_only=True,
                    config=config,
                )
                messages.append(
                    HumanMessage(
                        content=raw_data_content,
                        name="raw_resources",
                    )
                )

                # Get LLM response for analysis and guidance
                response = self._agent_llm_registry["resource_query"].invoke(messages)
                if "N/A" in response.content:
                    return Command(goto="research_team")

                formatted_response = (
                    response.content.strip("```markdown").strip("```").strip()
                )

                logger.info(f"LLM resource analysis response: {formatted_response}")

            except Exception as e:
                logger.error(f"Error in LLM resource analysis: {e}")
                # Fallback to simple listing
                formatted_response = raw_data_content

        else:
            logger.info("No API data available or API query failed")
            formatted_response = "## 教学资源分析与使用指导\n\n暂时没有检索到相关的实验或知识地图。\n\n"

            # In this case, directly goto the research_team node
            return Command(goto="research_team")

        # Update state with the analysis and guidance
        logger.info(f"Resource filter response: {formatted_response}")
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content=formatted_response,
                        name="resource_query",
                    )
                ],
                "filtered_resources": formatted_response,
            },
            goto="research_team",
        )

    async def research_team_node(
        self,
        state: State,
        config: RunnableConfig,
    ) -> Command[Literal["researcher", "generator"]]:
        """Research team node that collaborates on tasks."""
        logger.info("Research team is collaborating on tasks.")
        current_plan = state.get("current_plan")

        # check if all steps are executed
        if all(step.execution_res for step in current_plan.steps):
            return Command(goto="__end__")

        for step in current_plan.steps:
            if not step.execution_res:
                break
        if step.step_type and step.step_type == StepType.RESEARCH:
            return Command(goto="researcher")
        elif step.step_type and step.step_type == StepType.GENERATION:
            return Command(goto="generator")
        else:
            raise ValueError(f"Step type {step.step_type} is not supported.")

    async def researcher_node(
        self, state: State, config: RunnableConfig
    ) -> Command[Literal["research_team"]]:
        """Researcher node that do research"""
        logger.info("Researcher node is researching.")
        return await self._setup_and_execute_agent_step(
            state,
            config,
            "researcher",
        )

    async def generator_node(
        self, state: State, config: RunnableConfig
    ) -> Command[Literal["research_team"]]:
        """Generator node that generates educational content and materials."""
        logger.info("Generator node is generating educational content.")
        return await self._setup_and_execute_agent_step(
            state,
            config,
            "generator",
        )

    async def _setup_and_execute_agent_step(
        self,
        state: State,
        config: RunnableConfig,
        agent_name: str,
    ) -> Command[Literal["research_team"]]:
        """Setup and execute an agent step with the given LLM, state, and tools."""
        llm = self._agent_llm_registry[agent_name]
        tools = self._agent_tools_registry[agent_name]
        agent = create_react_agent(name=agent_name, model=llm, tools=tools)
        # configurable is already a Configuration object passed from the calling nodes

        try:
            if agent_name == "background_investigator":
                system_prompt = self._system_prompts[agent_name]
                # Convert Configuration object back to RunnableConfig format for consistency
                messages = self.apply_prompt_template(
                    system_prompt, state, system_prompt_only=True, config=config
                )
                messages = messages + [
                    {
                        "role": "user",
                        "content": "# 背景调研任务描述\n\n" + state["task_description"],
                    }
                ]
                logger.info(f"Background investigator is running.")
                result = await agent.ainvoke(
                    input={"messages": messages}, config=config
                )
                response_content = result["messages"][-1].content
                logger.info(f"背景调研结果: {response_content}")
                return Command(
                    update={
                        "messages": [
                            AIMessage(
                                content=response_content,
                                name=agent_name,
                            )
                        ],
                        "background_investigation_results": response_content,
                    },
                    goto="planner",
                )

            elif agent_name == "researcher" or agent_name == "generator":
                system_prompt = self._system_prompts[agent_name]
                messages = self.apply_prompt_template(
                    system_prompt, state, system_prompt_only=True, config=config
                )

                current_plan = state.get("current_plan")
                observations = state.get("observations", [])

                # Find the first unexecuted step
                current_step = None
                completed_steps = []
                for step in current_plan.steps:
                    if not step.execution_res:
                        current_step = step
                        break
                    else:
                        completed_steps.append(step)

                if not current_step:
                    logger.warning("No unexecuted step found")
                    return Command(goto="research_team")

                logger.info(f"Executing step: {current_step.title}")

                # Format completed steps information
                completed_steps_info = ""
                if completed_steps:
                    completed_steps_info = "# 已经执行的调研步骤\n\n"
                    for i, step in enumerate(completed_steps):
                        completed_steps_info += f"## 步骤 {i+1}: {step.title}\n\n"
                        # completed_steps_info += f"### 具体内容\n\n---\n{str(step.execution_res)}\n---\n"

                # Prepare the input for the agent with completed steps info
                agent_input = {
                    "messages": messages
                    + [
                        HumanMessage(
                            content=f"# 当前任务\n\n## 标题\n\n{current_step.title}\n\n## 描述\n\n{current_step.description}\n\n## 语言\n\n{state.get('locale', 'zh-CN')}"
                        )
                    ]
                }

                # Invoke the agent
                result = await agent.ainvoke(input=agent_input, config=config)

                # Process the result
                response_content = result["messages"][-1].content

                # Update the step with the execution result
                current_step.execution_res = response_content

                return Command(
                    update={
                        "messages": [
                            AIMessage(
                                content=response_content,
                                name=agent_name,
                            )
                        ],
                        "observations": observations + [response_content],
                        "retry_count": 0,
                    },
                    goto="research_team",
                )

            else:
                raise ValueError(f"Invalid agent name: {agent_name}")

        except Exception as e:
            error_msg = f"Agent {agent_name} execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)

            if agent_name == "background_investigator":
                return Command(
                    update={
                        "messages": [
                            AIMessage(
                                content=f"背景调研执行失败，错误信息: {str(e)}。",
                                name=agent_name,
                            )
                        ],
                        "background_investigation_results": "",
                    },
                    goto="planner",
                )
            elif agent_name == "researcher" or agent_name == "generator":
                observations = state.get("observations", [])

                current_step.execution_res = f"执行失败: {str(e)}"
                return Command(
                    update={
                        "messages": [
                            AIMessage(
                                content=f"步骤执行失败，错误信息: {str(e)}。",
                                name=agent_name,
                            )
                        ],
                        "observations": observations + [f"执行失败: {str(e)}"],
                    },
                    goto="research_team",
                )
            else:
                return Command(goto="research_team")
