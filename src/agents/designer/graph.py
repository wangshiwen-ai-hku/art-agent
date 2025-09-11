"""
Base chat agent implementation that provides common chat functionality
for agents that need interactive conversation capabilities.
"""

import os
import logging
from abc import abstractmethod
from typing import Optional
from langchain_core.runnables import RunnableConfig

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.prebuilt import create_react_agent

from ..base import BaseAgent, BaseState
from .prompt import NOTEHELPER_SYSTEM_PROMPT, SUMMARY_SYSTEM_PROMPT
from .config import ThreadConfiguration
from .schema import NoteHelperState
from src.config.manager import AgentConfig, MultiAgentConfig
from src.utils.input_processor import create_multimodal_message
from .callback import SummaryNodeCallback

logger = logging.getLogger(__name__)


class BaseNoteHelperAgent(BaseAgent):
    """
    Base chat agent that provides common interactive conversation functionality.
    This class can be inherited by specific agents like tutor, solver, etc.
    """

    name = "base_notehelper_agent"
    description = "Base notehelper agent that provides interactive conversation capabilities."

    def __init__(self, agent_config: AgentConfig):
        """Initialize the notehelper agent.

        Args:
            agent_config: Configuration object containing model parameters and tools.
        """
        super().__init__(agent_config)
        logger.info(f"{self.name} chat agent initialized successfully")
        self._wrap_llm_with_callback()

    def _load_system_prompt(self):
        """Load the system prompt for the chat agent."""
        self._system_prompt = {
            "chat_node": NOTEHELPER_SYSTEM_PROMPT,
            "summary_node": SUMMARY_SYSTEM_PROMPT,
        }
        self._user_prompt = {
            "chat_node": open(os.path.join(os.path.dirname(__file__), "prompt/chat_prompt.txt"), "r").read(),
            "summary_node": open(os.path.join(os.path.dirname(__file__), "prompt/summary_prompt.txt"), "r").read(),
        }

    def _wrap_llm_with_callback(self):
        """给 self._llm 加上 callback，返回新的 llm 实例"""
        cb = SummaryNodeCallback(node_name="summary_node")
        # 关键：LangChain 的 Runnable 支持 callbacks 参数
        # 我们直接把 callback 绑定到 llm 上，后面所有 invoke/ainvoke 都会自动带
        self._llm = self._llm.with_config(callbacks=[cb])
  
    async def init_context_node(self, state: NoteHelperState, config=None):
        """把用户第一次传进来的要求固化到 state"""
        logger.info("[init_context] 进入")
        # 前端/测试脚本可以把要求写在 state 里，这里只做保底
        if not state.user_specified_format.requirements:
            state.user_specified_format.requirements = "默认：生成详细 markdown 课堂笔记"
        return {"user_specified_format": state.user_specified_format}

    async def summary_node(self, state: NoteHelperState, config=None):
        """真正的摘要节点，只接受 (state,config)"""
        logger.info("[summary_node] 进入")
        from langchain_core.messages import SystemMessage, HumanMessage

        sys_msg = SystemMessage(content=self._system_prompt["summary_node"])
        interaction_history_str = "\n".join(
            f"{i+1}. {s}" for i, s in enumerate(state.interaction_history)
        )
        user_prompt = (
            open(os.path.join(os.path.dirname(__file__), "prompt/summary_prompt.txt"))
            .read()
            .format(
                history_summary="\n".join(state.history_summaries),
                current_full_asr=state.currimulate_asr_result
                + state.current_asr_chunk,
                user_specific_format=state.user_specified_format.model_dump_json(),
                interaction_history=interaction_history_str,
                industry_domain=state.industry_domain or "",
            )
        )
        # logger.info(f"[summary_node] 📝 prompt: {user_prompt}")
        logger.info(f"[summary_node] 📝 current_full_asr: {state.currimulate_asr_result + state.current_asr_chunk}")
        full_prompt = self._system_prompt["summary_node"]+user_prompt
        logger.info(f"[summary_node] prompt: {full_prompt}")
        response = await self._llm.ainvoke([sys_msg, HumanMessage(content=user_prompt)])

        logger.info(f"[summary_node] 📝 current_final_notes: {state.final_notes + response.content}")
        # 累加 ASR 并保存新摘要
        return {
            "currimulate_asr_result": state.current_asr_chunk,
            "history_summaries": [response.content],
            "final_notes": state.final_notes + "\n\n" + response.content,
            "current_asr_chunk": "",   # 消费完清空
        }
        
    async def wait_node(self, state: NoteHelperState, config = None):
        """会阻塞，直到外部调用者通过 .ainvoke 传入新 ASR"""
        logger.info("[wait_node] 进入")
        # 如果当前已经有 current_asr_chunk（外部刚喂的），直接放行
        logger.info(f"[wait_node] state.current_asr_chunk: {state.current_asr_chunk}")
        logger.info(f"[wait_node] state.current_message: {state.current_message}")
        
        if state.current_asr_chunk.strip():
            return Command(goto="summary_node")
        if state.current_message.strip():
            return Command(goto="chat_node")
        if state.asr_end:
            return Command(goto=END)
        # 否则 interrupt 等待
        logger.info("[wait_node] 即将 interrupt …")
        # logger.info(f"[wait_node] payload: {payload}")
        payload = interrupt("等待输入")      # 这里会抛特殊异常，把控制权交回给用户
        # payload = payload.value
        
        if "current_asr_chunk" in payload or "industry_domain" in payload:
            updates = {}
            if "current_asr_chunk" in payload:
                updates["current_asr_chunk"] = payload.pop("current_asr_chunk")
            if "industry_domain" in payload:
                updates["industry_domain"] = payload.pop("industry_domain")
            return Command(goto="summary_node", update=updates)
        elif "current_message" in payload:
            return Command(goto="chat_node", update={"current_message": payload.pop("current_message")})
        elif "asr_end" in payload:
            return Command(goto=END, update={"asr_end": payload.pop("asr_end")})
        # else:
        #     return Command(goto="wait_node")
    
        # logger.info(f"ASR 输入: {msg}")
        # 外部调用者会把 ASR 写在 state.current_asr_chunk 里
        # return Command(goto="summary_node", update={"current_asr_chunk": asr})
        # return Command(goto="router_node")
    
    async def chat_node(
        self, state: NoteHelperState, config: Optional[RunnableConfig] = None
    ):
        """Handle chat conversation with tools support.

        Args:
            state: The current agent state.
            config: Optional configuration for the node.

        Returns:
            Command: Command to continue to human_chat_node.
        """
        logger.info("[chat_node] 进入")
        # Create react agent with tools for interactive conversation
        if self._tools:
            react_agent = create_react_agent(
                name=self.name,
                model=self._llm,
                tools=self._tools,
            )

            # Prepare messages with system context
            system_message = SystemMessage(
                content=self._system_prompt["chat_node"]
            )
            messages = [
                system_message
            ] + [
                HumanMessage(
                    content=self._user_prompt["chat_node"].format(
                        current_full_asr=state.currimulate_asr_result,
                        current_full_notes=state.final_notes,
                        user_question=state.current_message,
                    )
                )
            ]

            # Get response from react agent
            response = await react_agent.ainvoke({"messages": messages})
            last_message = response["messages"][-1]

            interaction = f"用户提问：{state.current_message}"
            return {
                "messages": [last_message],
                "interaction_history": [interaction],
                "current_message": "",
            }
        else:
            # Fallback to simple LLM without tools
            system_message = SystemMessage(content=self._system_prompt["chat_node"])

            messages = [
                system_message
            ] + [
                HumanMessage(
                    content=self._user_prompt["chat_node"].format(
                        current_full_asr=state.currimulate_asr_result,
                        current_full_notes=state.final_notes,
                        user_question=state.current_message,
                    )
                )
            ]

            response = await self._llm.ainvoke(messages)
            logger.info(f"chat agent response: {response.content}")

            interaction = f"用户提问：{state.current_message}"
            return {
                "messages": [response],
                "interaction_history": [interaction],
                "current_message": "",
            }

    # async def router_node(self, state: NoteHelperState, config=None) -> Command:
    #     """简单演示：如果外部不再喂 ASR 就结束"""
    #     logger.info("[router_node] 进入")
    #     logger.info(f"[router_node] state.current_message: {state.current_message}")
    #     # logger.info(f"[router_node] state.asr_end: {state.asr_end}")
        
    #     if state.current_asr_chunk:
    #         logger.info("[router_node] 进入 ASR")
    #         return Command(goto="summary_node")
    #     if state.current_message:
    #         logger.info("[router_node] 进入 chat")
    #         return Command(goto="chat_node")
    #     if state.asr_end:
    #         logger.info("[router_node] 进入 END")
    #         return Command(goto=END)
        
    #     logger.info("[router_node] 进入 wait")
    #     return Command(goto="wait_node")

    def build_graph(self):
        from langgraph.graph import StateGraph, START, END
        graph = StateGraph(NoteHelperState)
        graph.add_node("init_context", self.init_context_node)
        graph.add_node("wait_node", self.wait_node)
        graph.add_node("chat_node", self.chat_node)
        # graph.add_node("router_node", self.router_node)
        graph.add_node("summary_node", self.summary_node)
        # graph.add_node("should_continue", self.should_continue)

        graph.add_edge(START, "init_context")
        graph.add_edge("init_context", "wait_node")
        graph.add_edge("summary_node", "wait_node")
        graph.add_edge("chat_node", "wait_node")
        # graph.add_edge("wait_node", "router_node")
        return graph
    
    