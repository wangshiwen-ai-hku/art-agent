import dotenv
import os
import logging
from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


from ..config.configuration import Configuration

from .schema import State


dotenv.load_dotenv()

logger = logging.getLogger(__name__)


async def _execute_agent_step(
    state: State, agent, agent_name: str
) -> Command[Literal["research_team", "planner"]]:
    """Helper function to execute a step using the specified agent."""

    default_recursion_limit = 10

    try:
        if agent_name == "background_investigator":
            messages = apply_prompt_template(agent_name, state, system_prompt_only=True)
            messages = messages + [
                {
                    "role": "user",
                    "content": "# 背景调研任务描述\n\n" + state["task_description"],
                }
            ]
            logger.info(f"Background investigator is running.")
            result = await agent.ainvoke(
                input={"messages": messages},
                config={"recursion_limit": default_recursion_limit},
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
            messages = apply_prompt_template(agent_name, state, system_prompt_only=True)

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
            result = await agent.ainvoke(
                input=agent_input, config={"recursion_limit": default_recursion_limit}
            )

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
            # For background investigator, return with empty result so it can be retried
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
            # For researcher/generator, leave execution_res empty so step can be retried
            observations = state.get("observations", [])

            # If retry_count is greater than 3, fill in the execution_res with the error message
            # if state.get("retry_count", 0) > 3:
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
                    # "retry_count": 0,
                },
                goto="research_team",
            )
            # else:
            #     return Command(
            #     update={
            #         "retry_count": state.get("retry_count", 0) + 1,
            #     },
            #     goto="research_team",
            # )
        else:
            # For unknown agent types, still try to continue
            return Command(goto="research_team")


async def _setup_and_execute_agent_step(
    llm,
    state: State,
    configurable: Configuration,
    agent_type: str,
    tools: list,
) -> Command[Literal["research_team"]]:
    agent = create_react_agent(name=agent_type, model=llm, tools=tools)
    await _execute_agent_step(state, agent, agent_type)
