"""
Base chat agent implementation that provides common chat functionality
for agents that need interactive conversation capabilities.
"""

from pathlib import Path
import os
import logging
from abc import abstractmethod
from re import A
from typing import Optional
from langchain_core.runnables import RunnableConfig
import base64

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.prebuilt import create_react_agent


## ===========gen ai ============#
from google import genai
from google.genai.types import GenerateContentConfig, Modality
from PIL import Image
from io import BytesIO

from ..base import BaseAgent, BaseState
from .prompt import FOV_SYSTEM_PROMPT, FOV_CHAT_SYSTEM_PROMPT
from .config import ThreadConfiguration
from .schema import ArtistState
from src.config.manager import AgentConfig, MultiAgentConfig
from src.utils.input_processor import create_vertex_multimodal_message as create_multimodal_message
from .callback import DesignerNodeCallback
from .models import client
# from src.utils import multi_modal_message_loader

logger = logging.getLogger(__name__)


WORK_DIR = Path(os.path.dirname(__file__))

class FoVAgent(BaseAgent):
    """
    Base chat agent that provides common interactive conversation functionality.
    This class can be inherited by specific agents like tutor, solver, etc.
    """

    name = "fov_agent"
    description = "fov agent that provides interactive conversation capabilities."

    def __init__(self, agent_config: AgentConfig):
        """Initialize the notehelper agent.

        Args:
            agent_config: Configuration object containing model parameters and tools.
        """
        super().__init__(agent_config)
        logger.info(f"{self.name} chat agent initialized successfully")
        # self._wrap_llm_with_callback()
        self.config = agent_config
        
    def _init_llm(self):
        """Load the model for the chat agent."""
        self._llm = client

    def _load_system_prompt(self):
        """Load the system prompt for the chat agent."""
        system_prompt_text = open(os.path.join(os.path.dirname(__file__), "prompt/system_prompt.txt"), "r").read()
        image_data_list = [WORK_DIR / 'examples/design_skill.jpg']
        # convert to base64 str
        image_data = [open(image_path, "rb").read() for image_path in image_data_list]
        self._system_prompt = create_multimodal_message(text=system_prompt_text, image_data=image_data)
        
        self._user_prompt = {
            "draw_node": open(os.path.join(os.path.dirname(__file__), "prompt/draw_prompt.txt"), "r").read(),
            "designer_node": open(os.path.join(os.path.dirname(__file__), "prompt/designer_prompt.txt"), "r").read(),
        }

    # def _wrap_llm_with_callback(self):
    #     """给 self._llm 加上 callback，返回新的 llm 实例"""
    #     cb = DesignerNodeCallback(node_name="designer_node")
    #     # 关键：LangChain 的 Runnable 支持 callbacks 参数
    #     # 我们直接把 callback 绑定到 llm 上，后面所有 invoke/ainvoke 都会自动带
    #     self._llm = self._llm.with_config(callbacks=[cb])
  
    async def init_context_node(self, state: ArtistState, config=None):
        """把用户第一次传进来的要求固化到 state"""
        logger.info("[init_context] 进入")
        if state.user_example_images: ## Path like
            images_data = [open(image_path, "rb").read() for image_path in state.user_example_images]
        else:
            images_data = []
        return {"topic": state.topic, "requirement": state.requirement, "user_example_images": images_data}

    async def desginer_node(self, state: ArtistState, config=None):
        """进入设计阶段。"""
        logger.info("[desinger_node] 进入")
        from langchain_core.messages import SystemMessage, HumanMessage
        user_prompt_text = (
            self._user_prompt["designer_node"]
            .format(
                topic=state.topic,
                requirement=state.requirement,
            )
        )
        user_prompt = create_multimodal_message(text=user_prompt_text, image_data=state.user_example_images)
        
        logger.info(f"[designer_node] 📝 user_prompt: {user_prompt_text}")
        full_prompt = [self._system_prompt] + [user_prompt]
        # logger.info(f"[designer_node] prompt: {full_prompt}")
        # response = await self._llm.ainvoke(full_prompt)
        response = await self._llm.models.generate_content(
            model='gemini-2.5-pro',
            contents=full_prompt,
        )

        logger.info(f"[designer_node] 📝 current_design_draft: {response.content}")
        
        return {
            "design_draft": response.content,
        }

    async def draw_node(self, state: ArtistState, config=None):
        """进入绘制阶段。"""
        logger.info("[draw_node] 进入")
        
        sys_msg = SystemMessage(content=self._system_prompt)
        
        user_prompt = (
            self._user_prompt["draw_node"]
            .format(
                topic=state.topic,
                requirement=state.requirement,
                design_draft=state.design_draft,
            )
        )
        
        ## load multi-modal messages 
        
        logger.info(f"[draw_node] 📝 user_prompt: {user_prompt}")
        full_prompt = self._system_prompt["draw_node"] + user_prompt
        logger.info(f"[draw_node] prompt: {full_prompt}")
        response = await self._llm.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            config=GenerateContentConfig(
            response_modalities=[Modality.TEXT, Modality.IMAGE],
            candidate_count=1,
            safety_settings=[
                {"method": "PROBABILITY"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT"},
                {"threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ],
            ),
        )
        
        draft = ""
        image_base64 = ""
        
        for part in response.candidates[0].content.parts:
            if part.text:
                draft = part.text
                
            elif part.inline_data:
                image_data = part.inline_data.data
                image = Image.open(BytesIO(image_data))
                image.save("output/draw_node.png")
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                image_base64_str = f"data:image/png;base64,{image_base64}"
        
        return {
            "history_drafts": draft,
            "history_images": image_base64,
        }

    # async def wait_node(self, state: A, config = None):
    #     """会阻塞，直到外部调用者通过 .ainvoke 传入新 ASR"""
    #     logger.info("[wait_node] 进入")
    #     # 如果当前已经有 current_asr_chunk（外部刚喂的），直接放行
    #     logger.info(f"[wait_node] state.current_asr_chunk: {state.current_asr_chunk}")
    #     logger.info(f"[wait_node] state.current_message: {state.current_message}")
        
    #     if state.current_asr_chunk.strip():
    #         return Command(goto="summary_node")
    #     if state.current_message.strip():
    #         return Command(goto="chat_node")
    #     if state.asr_end:
    #         return Command(goto=END)
    #     # 否则 interrupt 等待
    #     logger.info("[wait_node] 即将 interrupt …")
    #     # logger.info(f"[wait_node] payload: {payload}")
    #     payload = interrupt("等待输入")      # 这里会抛特殊异常，把控制权交回给用户
    #     # payload = payload.value
        
    #     if "current_asr_chunk" in payload or "industry_domain" in payload:
    #         updates = {}
    #         if "current_asr_chunk" in payload:
    #             updates["current_asr_chunk"] = payload.pop("current_asr_chunk")
    #         if "industry_domain" in payload:
    #             updates["industry_domain"] = payload.pop("industry_domain")
    #         return Command(goto="summary_node", update=updates)
    #     elif "current_message" in payload:
    #         return Command(goto="chat_node", update={"current_message": payload.pop("current_message")})
    #     elif "asr_end" in payload:
    #         return Command(goto=END, update={"asr_end": payload.pop("asr_end")})
    #     # else:
    #     #     return Command(goto="wait_node")
    
    #     # logger.info(f"ASR 输入: {msg}")
    #     # 外部调用者会把 ASR 写在 state.current_asr_chunk 里
    #     # return Command(goto="summary_node", update={"current_asr_chunk": asr})
    #     # return Command(goto="router_node")
    
    # async def chat_node(
    #     self, state: NoteHelperState, config: Optional[RunnableConfig] = None
    # ):
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
        graph = StateGraph(ArtistState)
        graph.add_node("init_context", self.init_context_node)
        graph.add_node("designer_node", self.desginer_node)
        graph.add_node("draw_node", self.draw_node)
        # graph.add_node("router_node", self.router_node)
        # graph.add_node("summary_node", self.summary_node)
        # graph.add_node("should_continue", self.should_continue)

        graph.add_edge(START, "init_context")
        graph.add_edge("init_context", "designer_node")
        graph.add_edge("designer_node", "draw_node")
        graph.add_edge("draw_node", END)
        # graph.add_edge("wait_node", "router_node")
        return graph
    
    