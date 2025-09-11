"""
Test file using LLM to call tools from triggers.py
"""

import asyncio
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from src.infra.tools.triggers import think_tool, plan_tool, task_done


async def test_llm_tool_calls():
    print("=" * 60)
    print("TESTING LLM TOOL CALLS")
    print("=" * 60)

    # Initialize LLM with tools
    llm = init_chat_model(model="gpt-4.1-mini", temperature=0)

    # Bind tools to LLM
    tools = [think_tool, plan_tool, task_done]
    llm_with_tools = llm.bind_tools(tools)

    print(f"✅ LLM initialized: {llm.model_name}")
    print(f"✅ Tools bound: {[tool.name for tool in tools]}")

    # Test 1: LLM calls think_tool
    print("\n" + "=" * 40)
    print("TEST 1: LLM calls think_tool")
    print("=" * 40)

    messages1 = [
        SystemMessage(content="你是一个推理智能体。你有think_tool、plan_tool和task_done工具。"),
        HumanMessage(content="请思考一下如何制定一个学习Python的计划"),
    ]

    response1 = await llm_with_tools.ainvoke(messages1)
    print(f"LLM Response: {response1.content}")
    print(f"Tool calls: {response1.tool_calls}")

    if response1.tool_calls:
        for tool_call in response1.tool_calls:
            print(f"\n🔧 Tool Call Details:")
            print(f"   Name: {tool_call['name']}")
            print(f"   ID: {tool_call['id']}")
            print(f"   Args: {tool_call['args']}")
            print(f"   Args type: {type(tool_call['args'])}")

    # Test 2: LLM calls plan_tool
    print("\n" + "=" * 40)
    print("TEST 2: LLM calls plan_tool")
    print("=" * 40)

    messages2 = [
        SystemMessage(content="你是一个推理智能体。你有think_tool、plan_tool和task_done工具。"),
        HumanMessage(content="请制定一个具体的3个月Python学习计划，分解成具体步骤"),
    ]

    response2 = await llm_with_tools.ainvoke(messages2)
    print(f"LLM Response: {response2.content}")
    print(f"Tool calls: {response2.tool_calls}")

    if response2.tool_calls:
        for tool_call in response2.tool_calls:
            print(f"\n🔧 Tool Call Details:")
            print(f"   Name: {tool_call['name']}")
            print(f"   ID: {tool_call['id']}")
            print(f"   Args: {tool_call['args']}")
            print(f"   Args type: {type(tool_call['args'])}")

            # Test how we would extract plan data in the agent
            if tool_call["name"] == "plan_tool":
                thought = tool_call["args"].get("thought", "")
                steps = tool_call["args"].get("steps", [])
                plan_data = {"thought": thought, "steps": steps}

                print(f"\n📋 Plan Data Processing:")
                print(f"   Thought: '{thought}'")
                print(f"   Steps: {steps}")
                print(f"   Steps type: {type(steps)}")
                print(f"   Plan_data: {plan_data}")

                print(f"\n📝 Individual Steps:")
                for i, step in enumerate(steps):
                    print(f"   Step {i}: '{step}' ({type(step)})")

    # Test 3: LLM calls task_done
    print("\n" + "=" * 40)
    print("TEST 3: LLM calls task_done")
    print("=" * 40)

    messages3 = [
        SystemMessage(
            content="你是一个推理智能体。你有think_tool、plan_tool和task_done工具。当你完成一个任务步骤时，请使用task_done工具。"
        ),
        HumanMessage(content="我刚完成了Python基础语法的学习，请确认这个步骤完成"),
    ]

    response3 = await llm_with_tools.ainvoke(messages3)
    print(f"LLM Response: {response3.content}")
    print(f"Tool calls: {response3.tool_calls}")

    if response3.tool_calls:
        for tool_call in response3.tool_calls:
            print(f"\n🔧 Tool Call Details:")
            print(f"   Name: {tool_call['name']}")
            print(f"   ID: {tool_call['id']}")
            print(f"   Args: {tool_call['args']}")
            print(f"   Args type: {type(tool_call['args'])}")


if __name__ == "__main__":
    asyncio.run(test_llm_tool_calls())
