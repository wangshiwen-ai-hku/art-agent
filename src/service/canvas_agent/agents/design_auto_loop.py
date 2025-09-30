from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage, AIMessage
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
import sys
sys.path.append('.')
from src.config.manager import AgentConfig, config
from dataclasses import asdict
from src.utils.print_utils import show_messages
from pydantic import BaseModel

import operator
import asyncio
import uuid
import os
import base64
from langchain_core.tools import tool
import logging
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# 从工具文件中导入所有绘图和编辑工具
from src.infra.tools.draw_canvas_tools import DrawCanvasAgentTools
from src.infra.tools.edit_canvas_tools import EditCanvasAgentTools


logger = logging.getLogger(__name__)

# --- 1. Token Tracking and Logging ---

class TokenUsageCallback(BaseCallbackHandler):
    """Callback to track and log token usage for each LLM call."""
    def __init__(self):
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_tokens = 0

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        if response.llm_output and 'token_usage' in response.llm_output:
            token_usage = response.llm_output['token_usage']
            prompt_tokens = token_usage.get('prompt_tokens', 0)
            completion_tokens = token_usage.get('completion_tokens', 0)
            
            self._total_prompt_tokens += prompt_tokens
            self._total_completion_tokens += completion_tokens
            self._total_tokens += prompt_tokens + completion_tokens

            logger.info(f"LLM Call Token Usage: Prompt={prompt_tokens}, Completion={completion_tokens}")
    
    def get_total_usage(self):
        return {
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_tokens,
        }


# --- 2. 定义 Agent 的状态和 Pydantic 模型 ---

class DesignPlan(BaseModel):
    """一个结构化的设计计划，指导整个logo创作过程。"""
    conceptual_analysis: str
    visual_elements: str
    layout_strategy: str
    complexity_roadmap: str
    execution_steps: List[str]

class Critique(BaseModel):
    """对SVG设计的结构化评价。"""
    approval: bool
    feedback: str

class AgentState(TypedDict):
    """定义整个图（Graph）的状态。"""
    topic: str
    output_dir: str
    design_plan: DesignPlan | None
    svg_history: List[str]
    current_svg: str
    critiques: Annotated[List[str], operator.add]
    iteration: int
    max_iterations: int
    final_approval: bool

# --- 3. 初始化LLM和工具 ---

# 初始化共享的LLM实例
agent_config = config.get_agent_config("design_loop_agent", "core")
model_config = agent_config.model
llm = init_chat_model(**asdict(model_config))

# 合并所有可执行工具
ExecutionTools = DrawCanvasAgentTools + EditCanvasAgentTools
# 确保工具列表没有重复项 (基于函数名)
unique_tools = {func.name: func for func in ExecutionTools}
ExecutionTools = list(unique_tools.values())

@tool
def convert_svg_to_png_base64(svg_code: str) -> str:
    """将SVG代码转换为PNG的base64字符串，用于多模态评估。"""
    try:
        import cairosvg
        png_data = cairosvg.svg2png(bytestring=svg_code.encode())
        base64_str = base64.b64encode(png_data).decode('utf-8')
        return f"data:image/png;base64,{base64_str}"
    except ImportError:
        return f"data:image/svg+xml;base64,{base64.b64encode(svg_code.encode()).decode()}"

# --- 4. 定义图（Graph）的各个节点 ---

async def planner_node(state: AgentState) -> AgentState:
    """
    规划节点：接收用户主题，生成一个详细、结构化的设计计划。
    """
    logger.info("---进入规划节点(Planner Node)---")
    prompt = f"""
You are an expert logo design strategist. Based on the following topic and requirements, create a comprehensive and actionable design plan.
The plan should be structured to guide a separate execution agent to create a complex and creative SVG logo.

**Topic & Requirements:** {state['topic']}

**Your output MUST be a structured DesignPlan.**
Break down the creative process into the following sections:
1.  **conceptual_analysis:** What are the core ideas, metaphors, or themes to convey?
2.  **visual_elements:** What specific shapes, symbols, or objects will be used? Describe them in detail.
3.  **layout_strategy:** How will these elements be arranged? Describe the composition, balance, and flow.
4.  **complexity_roadmap:** How will complexity be achieved? Mention specific techniques like intricate Bezier curves, gradients, or symmetrical patterns that the math agent can help with.
5.  **execution_steps:** Provide a clear, step-by-step list of actions for the execution agent to follow. Each step should be a concrete instruction, like "Draw the main body of the dragon using a complex Bezier curve" or "Add the letters 'ICSML' as SVG paths using a specific font."
"""
    response = await llm.with_structured_output(DesignPlan).ainvoke([HumanMessage(content=prompt)])
    logger.info(f"生成的设计计划: {response}")
    return {"design_plan": response}


async def execution_node(state: AgentState) -> AgentState:
    """
    执行节点：一个ReAct Agent，根据计划和反馈使用工具集来创建或修改SVG。
    """
    logger.info("---进入执行节点(Execution Node)---")
    
    SYSTEM_PROMPT = """
# Role
You are an expert SVG artist. Your task is to execute a given design plan and create an SVG logo using your tools.
You must follow the `execution_steps` precisely. If you need to draw complex, mathematically precise curves, describe the task to the `calculate_path_data_with_math_agent` tool, which will return the necessary path data for you to draw.

- **Initial Drawing:** If the `current_svg_code` is empty, follow the plan to create the first version.
- **Modification:** If `current_svg_code` and `feedback` are provided, modify the existing SVG based on the feedback, while still adhering to the original design plan.
- **Tool Usage:** You have a rich set of tools for drawing and editing. Choose the right tool for each task. All text must be rendered as paths using `draw_text_paths`.
- **Output:** Your final message in the ReAct loop must be the complete, pure SVG code.
"""
    
    # 构建执行Agent
    executor_agent = create_react_agent(
        name="executor_agent",
        model=llm,
        tools=ExecutionTools,
        prompt=SYSTEM_PROMPT
    )

    # 准备输入
    prompt = f"""
**Design Plan:**
{state['design_plan']}

**Execution Steps:**
{state['design_plan'].execution_steps}

**Current SVG Code:**
{state['current_svg']}

**Feedback from Critic (if any):**
{state.get('critiques', [])}

Now, proceed with the execution.
"""
    
    msg = await executor_agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
    show_messages(msg.get('messages', []))
    
    # 从工具调用或最终回复中提取SVG
    new_svg = ""
    messages: List[BaseMessage] = msg.get('messages', [])
    for message in reversed(messages):
        if (isinstance(message, ToolMessage) or isinstance(message, HumanMessage)) and isinstance(message.content, str) and message.content.strip().startswith('<svg'):
            new_svg = message.content
            break
        elif isinstance(message, AIMessage) and isinstance(message.content, str) and message.content.strip().startswith('<svg'):
            new_svg = message.content
            break
            
    if not new_svg: # 如果没有找到SVG，可能出错了或者在思考
        logger.warning("在执行节点的输出中未找到SVG代码。")
        # 在这种情况下，我们可能不更新SVG，或者返回一个错误状态
        return {}

    output_dir = state["output_dir"]
    svg_path = os.path.join(output_dir, f'execution_node_{state["iteration"]}.svg')
    with open(svg_path, 'w', encoding='utf-8') as f:
        f.write(new_svg)
    logger.info(f"执行节点生成的SVG (第 {state['iteration']} 次迭代) 已保存至: {svg_path}")
    
    return {
        "current_svg": new_svg,
        "svg_history": state.get("svg_history", []) + [new_svg],
        "iteration": state.get("iteration", 0) + 1,
    }

def critic_node(state: AgentState) -> AgentState:
    """
    批判节点：一个多模态LLM，评估SVG并提供结构化的反馈。
    """
    logger.info("---进入批判节点(Critic Node)---")
    if not state["current_svg"]:
        logger.info("没有SVG可供批判，跳过此节点。")
        # 给予一个默认的“不批准”反馈，以启动第一个执行循环
        return {
            "critiques": ["No SVG was generated yet. Please start by executing the design plan."],
            "final_approval": False
        }

    prompt = f"""
You are a discerning logo design critic. Evaluate the provided SVG artwork based on the original design topic.

**Design Topic:** '{state["topic"]}'

**Evaluation Criteria:**
1.  **Relevance to Topic:** Does the logo effectively represent the core concepts of the topic?
2.  **Execution of Plan:** How well does the SVG adhere to the original design plan?
3.  **Complexity & Skill:** Does the artwork demonstrate technical skill (e.g., smooth Bezier curves, clever geometry, good composition)?
4.  **Aesthetics:** Is the logo visually appealing and well-balanced?

**Task:**
Provide your feedback in a structured format. If the logo is excellent and meets all criteria to a high standard (9/10 or above), set `approval` to `true`. Otherwise, set `approval` to `false` and provide concise, actionable `feedback` for the execution agent to improve upon.
"""
    
    current_image_base64 = convert_svg_to_png_base64(state["current_svg"])
    image_content = {"type": "image_url", "image_url": {"url": current_image_base64}}
    message = HumanMessage(content=[{"type": "text", "text": prompt}, image_content])
    
    # 使用同步调用，因为LangGraph节点可以是同步的
    response = llm.with_structured_output(Critique).invoke([message])
    logger.info(f"批判结果: Approval={response.approval}, Feedback='{response.feedback}'")
    
    return {
        "critiques": [response.feedback] if not response.approval else [],
        "final_approval": response.approval
    }

def router(state: AgentState) -> str:
    """
    路由节点：根据批判结果和迭代次数决定下一步。
    """
    logger.info("---进入路由节点(Router)---")
    if state["final_approval"]:
        logger.info("设计已批准。结束流程。")
        return "end"
    if state["iteration"] >= state["max_iterations"]:
        logger.info("达到最大迭代次数。结束流程。")
        return "end"
    
    logger.info("设计未批准，返回执行节点进行修改。")
    return "executor"

# --- 5. 构建并编译图 ---

workflow = StateGraph(state_schema=AgentState)

# 添加节点
workflow.add_node("planner", planner_node)
workflow.add_node("executor", execution_node)
workflow.add_node("critic", critic_node)

# 定义图的边
workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "critic")
workflow.add_conditional_edges("critic", router, {"executor": "executor", "end": END})

graph = workflow.compile()

# --- 6. 运行图 ---

async def run_design_loop(topic: str, max_iterations: int = 3):
    output_dir = "output/design_auto_loop/" + str(uuid.uuid4())
    os.makedirs(output_dir, exist_ok=True)

    # --- Setup Logging ---
    log_file_path = os.path.join(output_dir, 'design_loop.log')
    
    # Remove any existing handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler() # Also log to console
        ]
    )
    logger.info(f"Log file will be saved to: {log_file_path}")

    token_callback = TokenUsageCallback()

    initial_state = {
        "topic": topic,
        "output_dir": output_dir,
        "design_plan": None,
        "current_svg": "",
        "svg_history": [],
        "critiques": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "final_approval": False
    }
    final_state = await graph.ainvoke(initial_state, config={'callbacks': [token_callback]})
    
    logger.info("---设计流程结束---")

    total_usage = token_callback.get_total_usage()
    logger.info(f"---Total Token Usage---")
    logger.info(f"Prompt Tokens: {total_usage['total_prompt_tokens']}")
    logger.info(f"Completion Tokens: {total_usage['total_completion_tokens']}")
    logger.info(f"Total Tokens: {total_usage['total_tokens']}")
    
    # The iteration number is already incremented, so we subtract 1 for the last saved file.
    final_iteration_count = final_state.get('iteration', 1)
    final_svg_path = os.path.join(output_dir, f"execution_node_{final_iteration_count - 1}.svg")
    logger.info(f"最终SVG保存在: {final_svg_path}")

    return final_state["current_svg"]

if __name__ == "__main__":
    topic = "设计一个logo，一个会议，首字母是 ICSML，由于是香港大学设立的，需要融入港大logo的元素"
    asyncio.run(run_design_loop(topic))