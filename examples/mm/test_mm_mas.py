"""
Lesson MAS Agent Example

This example demonstrates how to use the LessonMAS agent for comprehensive
lesson generation with planning and execution phases. The LessonMAS is a multi-agent system
that coordinates a supervisor agent for planning and section agents for content generation.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import logging
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import END, StateGraph, START
from src.agents.mm_mas.graph import LessonMAS
from src.config.manager import config
from src.infra.tools.manager import load_all_mcp_tools
from langgraph.graph import StateGraph

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def draw_graph(graph: StateGraph, filename: str):
    """Draws the agent graph and saves it to a file."""
    try:
        # Get the compiled graph representation
        compiled_graph = graph.compile()
        
        # Draw the graph and save as PNG
        with open(f"{filename}.png", "wb") as f:
            f.write(compiled_graph.get_graph().draw_mermaid_png())
        logger.info(f"✅ Graph visualization saved to {filename}.png")
        
    except Exception as e:
        logger.error(f"❌ Failed to draw graph: {e}")
        logger.error("Please ensure you have graphviz installed (`sudo apt-get install graphviz`) and pygraphviz (`pip install pygraphviz`).")


async def initialize_lesson_mas():
    """Initialize the Lesson MAS agent with proper MCP tools loading."""
    # Load MCP tools first
    logger.info("Ensuring MCP tools are loaded...")
    try:
        await load_all_mcp_tools()
        logger.info("MCP tools initialization completed")
    except Exception as e:
        logger.warning(f"MCP tools loading had issues: {e}")
        logger.info("Continuing with available tools...")

    # Get multi-agent configuration for mm_mas
    multi_agent_config = config.get_multi_agent_config("learner", "mm_mas")
    logger.info(
        f"Lesson MAS multi-agent configuration loaded with {len(multi_agent_config.agents)} sub-agents:"
    )
    for agent in multi_agent_config.agents:
        logger.info(f"  - {agent.agent_name}: {agent.tools}")

    # Create the Lesson MAS agent
    return LessonMAS(multi_agent_config=multi_agent_config)


async def run_lesson_mas_demo():
    """Run an automated demo for the Lesson MAS."""

    print("\n🤖 自动课程生成模式演示")
    print("=" * 50)

    # Initialize Lesson MAS agent
    lesson_mas_agent = await initialize_lesson_mas()

    # Draw the graphs before running
    print("\n🎨 正在绘制 Agent 工作流图...")
    # The main graph is built within the build_graph method
    main_graph_builder = lesson_mas_agent.build_graph()
    draw_graph(main_graph_builder, "graphs_viz/mm_mas/lesson_graph")
    
    # The section graph is built within a helper method
    section_graph_builder = lesson_mas_agent.build_section_graph()
    draw_graph(section_graph_builder, "graphs_viz/mm_mas/section_graph")
    print("=" * 50)


    # Runtime configuration
    thread_config = {
        "configurable": {
            "thread_id": "lesson_mas_demo_thread",
        },
        "recursion_limit": 50,
    }

    # Lesson generation topic
    topic = "二次函数的图像与性质教学设计"
    duration = "25min"
    depth = "初中二年级"
    
    print(f"🎯 教学主题: {topic}")
    print(f"🕒 教学时长: {duration}")
    print(f"🎓 学习深度: {depth}")

    _input = {
        "topic": topic,
        "duration": duration,
        "depth": depth,
        "messages": [], # Start with an empty list of messages
    }

    print("\n🚀 自动课程生成开始...")

    final_state = None
    # # Use the compiled graph from the drawing step
    compiled_graph = main_graph_builder.compile()
    async for state in compiled_graph.astream(
        input=_input,
        config=thread_config,
    ):
        final_state = state
        if "messages" in state and state["messages"]:
            # Get the last message to show progress
            last_message = state["messages"][-1]
            if isinstance(last_message, BaseMessage):
                agent_name = getattr(last_message, "name", "system")
                content = last_message.content
                
                # Show abbreviated content for readability
                abbreviated_content = (
                    content[:300] + "..." if len(content) > 300 else content
                )
                print(f"\n🗣️  {agent_name}: {abbreviated_content}")


    print("\n\n✅ 自动课程生成完成!")
    
    # if final_state and final_state.get("final_content"):
    #     print("\n" + "="*50)
    #     print("📋 最终生成的教学内容:")
    #     print("="*50)
    #     print(final_state["final_content"])
    # else:
    #     print("\n❌ 未能生成最终内容。请检查日志以获取更多信息。")
        # print(f"Final state: {final_state}")


if __name__ == "__main__":
    print("🚀 启动课程生成Agent (Lesson MAS)...")
    print("📋 注意: 确保MCP服务器正在运行 (如果配置了的话)")
    print("   例如: uv run scripts/run_mcp_servers.py")
    print("=" * 70)
    
    try:
        asyncio.run(run_lesson_mas_demo())

    except KeyboardInterrupt:
        print("\n👋 程序结束")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        logger.error(f"Startup error: {e}", exc_info=True)
