"""
Base chat agent implementation that provides common chat functionality
for agents that need interactive conversation capabilities.
"""

from pathlib import Path
import os
import logging
from abc import abstractmethod
from re import A
from typing import Optional, List, Union

from langchain_core.runnables import RunnableConfig
import base64
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.prebuilt import create_react_agent


## ===========gen ai ============#
from google import genai
from google.genai.types import GenerateContentConfig, Modality, GenerateImagesConfig, RawReferenceImage, MaskReferenceImage, MaskReferenceConfig, EditImageConfig
from PIL import Image
from io import BytesIO

from ..base import BaseAgent, BaseState
from .prompt import FOV_SYSTEM_PROMPT, FOV_CHAT_SYSTEM_PROMPT
from .config import ThreadConfiguration
from .schema import ArtistState
from src.config.manager import AgentConfig, MultiAgentConfig
from src.utils.input_processor import create_vertex_multimodal_message as create_multimodal_message
from src.utils.input_processor import bytes_to_raw_reference_image
from .callback import DesignerNodeCallback
from .models import client
# from src.utils import multi_modal_message_loader

logger = logging.getLogger(__name__)

generation_model = "imagen-3.0-generate-001"
# generation_model_fast = "imagen-4.0-fast-generate-001"
# generation_model_ultra = "imagen-4.0-ultra-generate-001"
edit_model = "imagen-3.0-capability-001"

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
        self._system_prompt = open(os.path.join(os.path.dirname(__file__), "prompt/system_prompt.txt"), "r").read()
        # self._system_prompt = self._system_prompt_text
        
        system_image_paths = ['design_skill.jpg', 'eg_8.jpg', 'eg_5.jpg', 'eg_3.png']
        self._system_image_data = [os.path.join(os.path.dirname(__file__), "examples", image_path) for image_path in system_image_paths]
        self._system_image_data = self._process_image_bytes(self._system_image_data)
        
        self._user_prompt = {
            "draw_node": open(os.path.join(os.path.dirname(__file__), "prompt/draw_prompt_imagegen.txt"), "r").read(),
            "designer_node": open(os.path.join(os.path.dirname(__file__), "prompt/designer_prompt_imagegen.txt"), "r").read(),
            "critic_node": open(os.path.join(os.path.dirname(__file__), "prompt/critic_prompt_imagegen.txt"), "r").read(),
            "edit_node": open(os.path.join(os.path.dirname(__file__), "prompt/edit_prompt_imagegen.txt"), "r").read(),
        }

    async def init_context_node(self, state: ArtistState, config=None):
        """把用户第一次传进来的要求固化到 state"""
        logger.info("[init_context] 进入")
        if state.user_example_images: ## Path like
            images_data = self._process_image_bytes(state.user_example_images)
        else:
            images_data = []
        self._system_prompt = self._system_prompt.format(topic=state.topic, requirement=state.requirement)
        self._system_prompt = create_multimodal_message(text=self._system_prompt, image_data=self._system_image_data)
        logger.info(f"[init_context] 📝 system_prompt: {self._system_prompt}")
        os.makedirs(state.project_dir, exist_ok=True)
        open(os.path.join(state.project_dir, "project_description.txt"), "w", encoding="utf-8").write(f"topic: {state.topic}\nrequirement: {state.requirement}")
        
        return {
            "project_dir": state.project_dir, 
            "topic": state.topic, "requirement": state.requirement, "user_example_images": images_data}
    
    def _process_image_bytes(self, image_path: Union[List[Union[str,Path]],str,Path]) -> List[bytes]:
        """
        read the path/str/ list of path of images to bytes
        """        
        if isinstance(image_path, str):
            image_path = [image_path]
        img_bytes_list = []
        for image in image_path:
            if isinstance(image, str):
                image = Path(image)
                img_bytes_list.append(image.read_bytes())
            elif isinstance(image, Path):
                img_bytes_list.append(image.read_bytes())
            else:
                raise ValueError(f"Unsupported image format: {type(image)}")
        return img_bytes_list

    async def critic_node(self, state: ArtistState, config=None):
        """进入阶段。"""
        logger.info("[critic_node] 进入")
        
        user_prompt_text = (
            self._user_prompt["critic_node"]
        )
        user_prompt_text = user_prompt_text.format(prompt=state.design_draft)
        
        image_data = [state.history_images[-1]]
        user_prompt = create_multimodal_message(text=user_prompt_text, image_data=image_data)
        
        logger.info(f"[critic_node] 📝 user_prompt: {user_prompt}")
        full_prompt = [self._system_prompt] + [user_prompt]
        logger.info(f"[designer_node] prompt: {full_prompt}")
        response = self._llm.models.generate_content(
            model='gemini-2.5-pro',
            contents=full_prompt,
        ).text
        logger.info(f"[critic_node] 📝 response: {response}")
        open(os.path.join(state.project_dir, "critic_node.txt"), "w", encoding="utf-8").write(response)
        
        # response = open("critic_node.txt", "r", encoding="utf-8").read()
        return {
            "history_drafts": [response],
        }
        
    async def desginer_node(self, state: ArtistState, config=None):
        """进入设计阶段。"""
        logger.info("[desinger_node] 进入")
        from langchain_core.messages import SystemMessage, HumanMessage
        user_prompt_text = (
            self._user_prompt["designer_node"]
        )
        user_prompt = create_multimodal_message(text=user_prompt_text, image_data=state.user_example_images)
        
        # logger.info(f"[designer_node] 📝 user_prompt: {user_prompt_text}")
        full_prompt = [self._system_prompt] + [user_prompt]
        logger.info(f"[designer_node] prompt: {full_prompt}")

        response = self._llm.models.generate_content(
            model='gemini-2.5-pro',
            contents=full_prompt,
        ).text

        logger.info(f"[designer_node] 📝 current_design_draft: {response}")
        open(os.path.join(state.project_dir, "designer_node.txt"), "w", encoding="utf-8").write(response)
        
        return {
            "design_draft": response,
            "history_drafts": [response],
        }

    async def draw_node(self, state: ArtistState, config=None):
        """进入绘制阶段。"""
        logger.info("[draw_node] 进入")
    
        
        design_prompt = (
            self._user_prompt["draw_node"]
            .format(
                design_draft=state.design_draft,
            )
        )
        ## load multi-modal messages 
        user_prompt = f"A minimalist black and white logo, vector art, high contrast, clean lines, isolated on a white background. {design_prompt}"
        logger.info(f"[draw_node] 📝 user_prompt: {user_prompt}")
        open(os.path.join(state.project_dir, "draw_node_prompt.txt"), "w", encoding="utf-8").write(user_prompt)

        image = client.models.generate_images(
            model=generation_model,
            prompt=user_prompt,
            config=GenerateImagesConfig(
                aspect_ratio="1:1",
                number_of_images=1,
                image_size="1K",
                safety_filter_level="BLOCK_MEDIUM_AND_ABOVE",
                person_generation="ALLOW_ADULT",
            ),
        )
        
        image.generated_images[0].image.save(os.path.join(state.project_dir, "draw_node.png"))
        
        return {
                "history_images": [open(os.path.join(state.project_dir, "draw_node.png"), "rb").read()],
        }

    async def edit_node(self, state: ArtistState, config=None):
        """进入重新绘制阶段，忽略之前的图像，使用critic生成的新prompt重新创作。"""
        logger.info("[redraw_node] 进入")
        
        # The prompt is the full, new design brief generated by the critic node.
        user_prompt = state.history_drafts[-1]
        
        # We prepend the style keywords to ensure consistency, same as in draw_node.
        full_user_prompt = f"A minimalist black and white logo, vector art, high contrast, clean lines, isolated on a white background. {user_prompt}"
        logger.info(f"[redraw_node] 📝 user_prompt: {full_user_prompt}")

        image = client.models.generate_images(
            model=generation_model,
            prompt=full_user_prompt,
            config=GenerateImagesConfig(
                aspect_ratio="1:1",
                number_of_images=1,
                image_size="1K",
                safety_filter_level="BLOCK_MEDIUM_AND_ABOVE",
                person_generation="ALLOW_ADULT",
            ),
        )
        
        image.generated_images[0].image.save(os.path.join(state.project_dir, "edit_node.png"))
        
        return {
                "history_images": state.history_images + [open(os.path.join(state.project_dir, "edit_node.png"), "rb").read()],
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
        # """Handle chat conversation with tools support.

        # Args:
        #     state: The current agent state.
        #     config: Optional configuration for the node.

        # Returns:
        #     Command: Command to continue to human_chat_node.
        # """
        # logger.info("[chat_node] 进入")
        # # Create react agent with tools for interactive conversation
        # if self._tools:
        #     react_agent = create_react_agent(
        #         name=self.name,
        #         model=self._llm,
        #         tools=self._tools,
        #     )

        #     # Prepare messages with system context
        #     system_message = SystemMessage(
        #         content=self._system_prompt["chat_node"]
        #     )
        #     messages = [
        #         system_message
        #     ] + [
        #         HumanMessage(
        #             content=self._user_prompt["chat_node"].format(
        #                 current_full_asr=state.currimulate_asr_result,
        #                 current_full_notes=state.final_notes,
        #                 user_question=state.current_message,
        #             )
        #         )
        #     ]

        #     # Get response from react agent
        #     response = await react_agent.ainvoke({"messages": messages})
        #     last_message = response["messages"][-1]

        #     interaction = f"用户提问：{state.current_message}"
        #     return {
        #         "messages": [last_message],
        #         "interaction_history": [interaction],
        #         "current_message": "",
        #     }
        # else:
        #     # Fallback to simple LLM without tools
        #     system_message = SystemMessage(content=self._system_prompt["chat_node"])

        #     messages = [
        #         system_message
        #     ] + [
        #         HumanMessage(
        #             content=self._user_prompt["chat_node"].format(
        #                 current_full_asr=state.currimulate_asr_result,
        #                 current_full_notes=state.final_notes,
        #                 user_question=state.current_message,
        #             )
        #         )
        #     ]

        #     response = await self._llm.ainvoke(messages)
        #     logger.info(f"chat agent response: {response.content}")

        #     interaction = f"用户提问：{state.current_message}"
        #     return {
        #         "messages": [response],
        #         "interaction_history": [interaction],
        #         "current_message": "",
        #     }

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
        graph.add_node("critic_node", self.critic_node)
        graph.add_node("edit_node", self.edit_node)
        # graph.add_node("router_node", self.router_node)
        # graph.add_node("summary_node", self.summary_node)
        # graph.add_node("should_continue", self.should_continue)

        graph.add_edge(START, "init_context")
        graph.add_edge("init_context", "designer_node")
        graph.add_edge("designer_node", "draw_node")
        graph.add_edge("draw_node", "critic_node")
        graph.add_edge("critic_node", "edit_node")
        graph.add_edge("edit_node", END)
        # graph.add_edge("wait_node", "router_node")
        return graph
    
    