import logging
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from .schema import LessonMASState, SectionGenerationState, LessonPlan
from langgraph.graph import StateGraph
import os
from datetime import datetime
import json
from langchain_core.callbacks import BaseCallbackHandler
from typing import Dict, Any, List
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)

# --- Custom Callback Handler for Sub-Graph Logging ---

class SubgraphDetailLogger(BaseCallbackHandler):
    """
    一个自定义的回调处理器，用于为 section-generation 子图中的事件
    提供详细的、实时的日志记录。
    """

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any,
    ) -> Any:
        """当聊天模型开始调用时记录其提示。"""
        logger.info("--- Sub-Graph: Chat Model Invoked ---")
        # `messages` 是一个消息列表的列表，对应批处理中的每个输入
        for i, msg_list in enumerate(messages):
            logger.info(f"  Input #{i+1}:")
            for msg in msg_list:
                # 安全地获取内容，处理不同的消息类型
                content = getattr(msg, 'content', str(msg))
                logger.info(f"    - {msg.type.upper()}: {content}")
        logger.info("--------------------------------------")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """
        将流式 token 打印到标准输出，以实现实时输出可见性。
        这用于观察“chunk的返回”的生成过程。
        """
        # 我们在这里使用 print() 是为了在控制台中获得即时、无缓冲的反馈。
        # 使用 logger.info() 可能会缓冲输出。
        print(token, end="", flush=True, encoding="utf-8")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """记录 LLM 调用的最终输出。"""
        # 在流式输出后，我们打印一个换行符以正确格式化后续的日志。
        print("\n") 
        logger.info("--- Sub-Graph: LLM Call Finished ---")
        # 简化最终结果的日志记录。完整内容已通过流式方式显示。
        logger.info("  (Full content streamed above)")
        logger.info("------------------------------------")

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """记录工具调用的开始。"""
        tool_name = serialized.get("name", "Unknown Tool")
        logger.info(f"--- Sub-Graph: Tool Call Started [{tool_name}] ---")
        logger.info(f"  Tool Input: {input_str}")
        logger.info("-------------------------------------------------")

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """记录工具调用的结束及其输出。"""
        logger.info(f"--- Sub-Graph: Tool Call Finished ---")
        logger.info(f"  Tool Output: {output}")
        logger.info("-----------------------------------")


# --- Detailed Logging Setup ---
LOG_DIR = "/graphs_viz/mm_mas/logs"
os.makedirs(LOG_DIR, exist_ok=True)

def create_log_entry(node_name: str, model_name: str, prompt: str, result: str):
    """Helper function to create detailed log entries."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        prompt_filename = f"{timestamp}_{node_name}_prompt.txt"
        result_filename = f"{timestamp}_{node_name}_result.json"
        prompt_filepath = os.path.join(LOG_DIR, prompt_filename)
        result_filepath = os.path.join(LOG_DIR, result_filename)
        
        prompt_len = len(prompt)
        result_len = len(result)
        
        with open(prompt_filepath, "w", encoding="utf-8") as f:
            f.write(prompt)
        with open(result_filepath, "w", encoding="utf-8") as f:
            f.write(result)
            
        logger.info(f"--- LLM Call Log: {node_name} ---")
        logger.info(f"Model Name: {model_name}")
        logger.info(f"Prompt Length: {prompt_len} characters")
        logger.info(f"Prompt Saved To: {prompt_filepath}")
        logger.info(f"Result Length: {result_len} characters")
        logger.info(f"Result Saved To: {result_filepath}")
        logger.info("------------------------------------")
    except Exception as e:
        logger.error(f"Failed to create log entry for {node_name}. Error: {e}")

# --- Graph Nodes ---

def lesson_supervisor_node(state: LessonMASState, config: RunnableConfig, structured_llm, model_name: str, system_prompt: str):
    """
    The top-level supervisor node. It plans the lesson on the first run
    and acts as a central checkpoint on subsequent runs.
    """
    logger.info("Lesson supervisor node is running.")
    
    if "section_plan" not in state or state.get("section_plan") is None:
        logger.info("No lesson plan found. Generating a new plan.")
        
        # Combine system and human prompts for a complete view
        human_prompt = f"""
请为以下学习请求创建一个详细的、分章节的学习计划。
您的回复必须是一个符合提供的 Pydantic Schema 的 JSON 对象。

# 学习主题
{state['topic']}

# 学习时长
{state['duration']}

# 学习深度
{state['depth']}
"""
        full_prompt = f"--- SYSTEM PROMPT ---\n{system_prompt}\n\n--- USER REQUEST ---\n{human_prompt}"
        
        # Invoke the LLM with structured output, passing the list of messages directly
        lesson_plan: LessonPlan = structured_llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)],
            # config={"configurable": {"system_prompt": system_prompt}} # Pass system prompt if model 
            # supports it
            config=config
        )
        
        # Log the interaction with the full combined prompt
        create_log_entry(
            node_name="lesson_supervisor",
            model_name=model_name,
            prompt=full_prompt,
            result=lesson_plan.model_dump_json(indent=2)
        )
        
        # Convert the Pydantic plan to the state's TypedDict format
        section_plan_for_state = [
            {"title": sec.title, "description": sec.description, "duration": sec.duration, "content": None}
            for sec in lesson_plan.sections
        ]

        return {
            "section_plan": section_plan_for_state,
            "current_section_index": 0,
            "messages": [AIMessage(content=f"课程规划完成，共计 {len(section_plan_for_state)} 个章节。", name="lesson_supervisor")]
        }
    else:
        logger.info("Lesson plan exists. Passing through for routing decision.")
        return {}


def section_supervisor_node(state: LessonMASState, config: RunnableConfig, section_supervisor_graph: StateGraph, system_prompt: str):
    """
    This node invokes the inner supervisor graph to generate content for a single section.
    """
    logger.info("Section supervisor node (invoker) is running.")
    logger.info("Invoking sub-graph with custom logger to trace tool calls and LLM streams.")

    current_index = state.get("current_section_index", 0)
    section_plan = state["section_plan"]
    
    if current_index >= len(section_plan):
        logger.warning("Section supervisor invoker called without a valid section.")
        return {}

    current_section = section_plan[current_index]
    logger.info(f"Invoking sub-graph for section {current_index + 1}: {current_section['title']}")

    # The user-facing task description that the supervisor will work on.
    task_description = f"""
    当下section的信息:
    
# 章节标题
{current_section['title']}

# 章节描述 (学习目标)
{current_section['description']}

# 预计时长
{current_section['duration']} 分钟

请根据以上信息，协调你的员工 Agents，生成该章节的详细教学内容。
"""
    
    logger.info(f"当下生成章节内容，请根据以下信息生成章节内容：\n{task_description}")
    # The input to the supervisor graph is a dict with a "messages" key.
    sub_graph_input = {"messages": [HumanMessage(content=task_description)]}

    # Run the sub-graph
    # 为了从子图中获取工具调用和 token 流（chunks）的详细日志，
    # 我们传递一个自定义的回调处理器。
    invocation_config = config.copy()
    invocation_config["callbacks"] = [SubgraphDetailLogger()]

    final_sub_graph_state = section_supervisor_graph.invoke(
        sub_graph_input, config=invocation_config
    )
    
    # Log the interaction
    full_prompt_for_log = f"--- SUPERVISOR SYSTEM PROMPT ---\n{system_prompt}\n\n--- INITIAL TASK ---\n{task_description}"
    
    # Convert state to a serializable format for logging
    final_state_json = json.dumps(final_sub_graph_state, indent=2, default=str, ensure_ascii=False)

    create_log_entry(
        node_name="section_supervisor",
        model_name='gemini-2.5-pro', # This is an approximation as the supervisor uses multiple models
        prompt=full_prompt_for_log,
        result=final_state_json
    )
    
    generated_content = final_sub_graph_state.get("compiled_content", "本章节内容生成失败。")
    
    section_plan[current_index]['content'] = generated_content
    
    logger.info(f"Sub-graph finished for section {current_index + 1}.")

    return {
        "section_plan": section_plan,
        "current_section_index": current_index + 1,
        "messages": [AIMessage(content=f"章节 '{current_section['title']}' 内容生成完毕。", name="section_supervisor")]
    }

def route_after_lesson_supervisor(state: LessonMASState) -> str:
    """
    Determines the next step after the lesson_supervisor has run.
    - If not all sections are done, route to the section supervisor.
    - If all sections are done, route to finish.
    """
    logger.info("Routing after lesson supervisor.")
    if not state.get("section_plan"):
         logger.error("Routing error: No plan found after supervisor node.")
         return "finish"
         
    if state.get("current_section_index", 0) < len(state.get("section_plan", [])):
        logger.info("More sections to process. Routing to section_supervisor_node.")
        return "section_supervisor_node"
    else:
        logger.info("All sections processed. Routing to finish.")
        return "finish" 