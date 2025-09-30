from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, ToolMessage
# from langchain_openai import ChatOpenAI  # 或你的LLM
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
import sys
sys.path.append('.')
from src.config.manager import AgentConfig, config
from dataclasses import asdict
from src.utils.print_utils import show_messages
from pydantic import BaseModel

import operator
import asyncio  # 用于async工具调用

# 从你的文档导入工具/代理（假设在同一项目中）
from src.infra.tools.draw_canvas_tools import draw_agent_with_tool
from src.infra.tools.edit_canvas_tools import edit_agent_with_tool

from src.service import agent_instance
import base64
from langchain_core.tools import tool
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DesignPlan(BaseModel):
    conceptual_analysis: str
    visual_elements: str
    layout_strategy: str
    complexity_roadmap: str
    execution_steps: str
    design_draft: str
# planner



# 补充的关键工具
@tool
def convert_svg_to_png_base64(svg_code: str) -> str:
    """svg_code: the svg code of the path you want to draw.
        example: <svg width="100" height="100">
                    <path d="M10,10 L50,50 Q70,30 90,90 Z" fill="red"/>
                </svg>
    return: the base64 str of the png image of the picked paths.
    """
    try:
        # 使用cairosvg或其他SVG渲染库
        import cairosvg
        png_data = cairosvg.svg2png(bytestring=svg_code.encode())
        base64_str = base64.b64encode(png_data).decode('utf-8')
        return f"data:image/png;base64,{base64_str}"
    except ImportError:
        # 备选方案：返回SVG的base64
        return f"data:image/svg+xml;base64,{base64.b64encode(svg_code.encode()).decode()}"


# 定义状态
class AgentState(TypedDict):
    topic: str  # 输入topic
    svg_history: List[str] = [] # 历史SVG代码
    current_svg: str = "" # 当前SVG代码（初始为空）
    critiques: Annotated[List[str], operator.add] = "" # 累积批评, Q: WHY the critique is accumulated?
    iteration: Annotated[int, operator.add] = 0  # 迭代计数
    final_approval: bool = False # 批准标志

class Critique(BaseModel):
    approval: bool  # 批准标志
    feedback: str  # 反馈

llm = None

def _init_agent():
    agent_config = config.get_agent_config("design_loop_agent", "core")
    model_config = agent_config.model
    global llm
    llm = init_chat_model(**asdict(model_config))


# 初始化LLM（替换为你的配置）
_init_agent()

DESIGN_SYSTEM_PROMPT = """
# Role
You are a expert logo designer. You will be given a `topic and requirement` and you need to design a logo for the topic. You have two agent as tool to help you draw or edit the svg code. You must give them detailed `task_description` as input. They are obliged to follow your instructions.
Besides, your agents has math agent to help it draw complex paths. You can hint it with its math tool when you want to create a complex path. You can even mention the task for their math agent. 

# Come up with Initial Design Draft
You must design the logo with creative mind and detailed instructions to your agents as tools. Your main task is design and brainstorm. The OBJECT, the LAYOUT, the COLOR and the MEANING.
USE just one agent at a time. You will have another agent to evaluate the svg code.
DONOT include any raw text in the svg code, all text should be PATH created by DRAW agent.

# Follow Feedback for your design draft
As you design, you also have a critic agent who will evaluate the svg code and give you `feedback`. You must follow the feedback to improve the `current svg code` by using your draw and edit agents.

# Output
The final output must be the pure svg code. DON'T include any other content.
"""
DESIGN_IN_LOOP_PROMPT = """

topic and requirement: {topic}
current svg code: {current_svg_code}
feedback: {feedback}

"""

# Designer节点：生成或编辑SVG
design_agent = create_react_agent(
    name="design_agent",
    model=llm,
    tools=[draw_agent_with_tool, edit_agent_with_tool],
    prompt=DESIGN_SYSTEM_PROMPT
)
async def designer_node(state: AgentState) -> AgentState:
    prompt = DESIGN_IN_LOOP_PROMPT.format(topic=state["topic"], current_svg_code=state["current_svg"], feedback=state.get("critiques", ""))
    msg = await design_agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
    # for chunk in design_agent.astream(msg):
    # show_messages(msg.get('messages', []))
    async for event in design_agent.astream(msg):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
    messages = msg.get('messages', []) # reverse order
    for message in reversed(messages):
        if (isinstance(message, ToolMessage)) and message.content.strip().startswith('<svg'):
            new_svg = message.content
            break
    open(f'{output_dir}/designer_node_{state["iteration"]}.svg', 'w', encoding='utf-8').write(new_svg)
    logger.info(f"designer_node: {new_svg}")
    return {
        "svg_history": state.get("svg_history", []) + [new_svg],
        "current_svg": new_svg,
        "iteration": 1,
        "critiques": []  # 清空当前轮批评，准备新轮
    }

async def design_planning_node(state: AgentState) -> AgentState:
    prompt = f"""
    You are a expert logo designer. You will be given a `topic and requirement` and you need to design a logo for the topic. You have two agent as tool to help you draw or edit the svg code. You must give them detailed `task_description` as input. They are obliged to follow your instructions.
    """
    return state

# make critic a tool of designer, and involve it into the design agent

# Critic节点：评估SVG
def critic_node(state: AgentState) -> AgentState:
    # USER Question: why not use schema output
    prompt = f"""
    You are a expert logo critic. Evaluate this SVG for topic '{state["topic"]}':
    
    {state["current_svg"]}
    
    Criteria:
    1. Relevance: Matches topic?
    2. Complexity: Intricate? (Multi-paths, gradients, math-based symmetries/curves?)
    3. Cleverness: Visual puns/metaphors?
    4. Quality: Aesthetically pleasing, professional, scalable?
    
    If excellent (scores 8+/10 on all), approve. Else, provide detailed feedback for improvement.
    I will give you the image rendered by the svg code.

    """
    current_image_base64 = convert_svg_to_png_base64(state["current_svg"])
    image_content = {"type": "image_url", "image_url": {"url": current_image_base64}}
    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        image_content
    ])
    response = llm.with_structured_output(Critique).invoke({"messages": [message]})

    feedback = response.feedback
    approval = response.approval

    return {
        "critiques": [feedback] if not approval else [],
        "final_approval": approval
    }


import uuid, os
output_dir = "output/design_loop/" + str(uuid.uuid4())
os.makedirs(output_dir, exist_ok=True)

# Router：决定下一步
def router(state: AgentState) -> str:
    if state["final_approval"] or state["iteration"] > 5:  # Max 5 iterations
        return "end"
    return "designer"

# 构建graph
workflow = StateGraph(state_schema=AgentState)

# 添加节点
workflow.add_node("designer", designer_node)
workflow.add_node("critic", critic_node)

# 入口
workflow.set_entry_point("designer")

# 边
workflow.add_edge("designer", "critic")
workflow.add_conditional_edges("critic", router, {"designer": "designer", "end": END})

# 编译（添加checkpoint可选）
graph = workflow.compile()

# 示例使用（async运行）
async def run_design_loop(topic: str):
    initial_state = {"topic": topic, "current_svg": "", "critiques": [], "iteration": 0, "final_approval": False}
    # for chunk in graph.astream(initial_state):
    #     show_messages(chunk.get("messages", []))
    final_state = await graph.ainvoke(initial_state)
    # show_messages(final_state.get("messages", []))

    return final_state["current_svg"]


logger.info("run_design_loop, output_dir: %s", output_dir)
# 测试：
asyncio.run(run_design_loop("设计一个logo，一个会议，首字母是 ICSML，由于是香港大学设立的，需要融入港大logo的元素"))