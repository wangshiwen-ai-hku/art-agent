"""Tools that trigger specific model abilities and actions without relying on external tools or resources."""

from langchain_core.tools import tool


@tool
def handoff_to_other_agent(agent_name: str, task_description: str):
    """
    Handoff the current task to another specialized agent.

    Args:
        agent_name: Name of the agent to hand off to (e.g., 'planning', 'research')
        task_description: Detailed description of the task to be handled

    Returns:
        Confirmation message with handoff details
    """
    return {
        "type": "handoff",
        "target_agent": agent_name,
        "task": task_description,
        "message": f"Handing off to {agent_name} agent: {task_description}",
    }


@tool(description="指示当前计划步骤已经完成，可以继续下一步")
def step_done(step_description: str, step_result: str):
    """
    指示当前计划步骤已经完成，可以继续执行下一步。

    Args:
        step_description: 已完成步骤的描述
        step_result: 步骤执行的结果摘要

    Returns:
        包含步骤完成详情的确认消息
    """
    return {
        "type": "step_done",
        "step": step_description,
        "result": step_result,
        "message": f"Step completed: {step_description}. Result: {step_result}",
    }


@tool(description="调研规划的策略性反思的工具")
def think_tool(reflection: str) -> str:
    """用于对调研进展和决策进行策略性反思的工具。

    在每次搜索后使用此工具来系统性地分析结果并规划下一步。
    这在研究工作流程中创造了一个深思熟虑的暂停，以便做出高质量的决策。

    何时使用：
    - 收到搜索结果后：我找到了哪些关键信息？
    - 决定下一步之前：我是否有足够的信息来全面回答？
    - 评估研究缺口时：我还缺少哪些具体信息？
    - 结束研究之前：我现在能够提供一个完整的答案吗？

    反思应该涉及：
    1. 当前发现的分析 - 我收集到了哪些具体信息？
    2. 缺口评估 - 还缺少哪些关键信息？
    3. 质量评估 - 我是否有足够的证据/例子来提供一个好的答案？
    4. 策略决策 - 我应该继续搜索还是提供我的答案？

    Args:
        reflection: 你对研究进展、发现、缺口和下一步的详细反思

    Returns:
        记录你的反思用于下一步决策制定
    """
    return f"你的反思：{reflection}"


@tool(description="任务分解和规划的工具")
def plan_tool(thought: str, steps: list[str]) -> dict:
    """用于将复杂任务分解为具体可执行步骤的规划工具。

    将大型或复杂的任务拆分成一系列有序的子任务，每个子任务都有明确的目标和可衡量的结果。
    这个工具有助于将抽象的目标转化为具体的行动计划。

    Args:
        thought: 你针对复杂任务以及如何分解任务的详细思考过程
        steps: 任务分解后的步骤列表，包含每一个步骤的详细描述

    Returns:
        记录你的计划，用于逐步执行
    """
    return {
        "thought": thought,
        "steps": steps,
    }
