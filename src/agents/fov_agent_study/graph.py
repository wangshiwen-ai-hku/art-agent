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
import json

from langchain_core.runnables import RunnableConfig
import base64
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool


## ===========gen ai ============#
from google import genai
from google.genai.types import GenerateImagesConfig, EditImageConfig, GenerateContentConfig
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
from .schema import DesignModel, EditModel
# from src.utils import multi_modal_message_loader

logger = logging.getLogger(__name__)

generation_model = "imagen-4.0-generate-001"
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
        
        system_image_paths = ['design_skill.jpg']
        self._system_image_data = [os.path.join(os.path.dirname(__file__), "examples", image_path) for image_path in system_image_paths]
        self._system_image_data = self._process_image_bytes(self._system_image_data)
        
        self._user_prompt = {
            "draw_node": open(os.path.join(os.path.dirname(__file__), "prompt/draw_prompt_imagegen.txt"), "r").read(),
            "designer_node": open(os.path.join(os.path.dirname(__file__), "prompt/designer_prompt_imagegen.txt"), "r").read(),
            "critic_node": open(os.path.join(os.path.dirname(__file__), "prompt/critic_prompt_imagegen.txt"), "r").read(),
            "edit_node": open(os.path.join(os.path.dirname(__file__), "prompt/edit_prompt_imagegen.txt"), "r").read(),
            "study_node": open(os.path.join(os.path.dirname(__file__), "prompt/study_prompt.txt"), "r").read(),
        }

    async def init_context_node(self, state: ArtistState, config=None):
        """把用户第一次传进来的要求固化到 state"""
        logger.info("[init_context] 进入")
        if state.user_example_image_paths: ## Path like
            user_example_images_data = self._process_image_bytes(state.user_example_image_paths)
            logger.info(f"[init_context] 📝 user_example_images_data: {len(user_example_images_data)}")
        else:
            user_example_images_data = []
        self._system_prompt = self._system_prompt.format(topic=state.topic, requirement=state.requirement)
        self._system_prompt = create_multimodal_message(text=self._system_prompt, image_data=self._system_image_data)
        logger.info(f"[init_context] 📝 system_prompt: {self._system_prompt}")
        os.makedirs(state.project_dir, exist_ok=True)
        open(os.path.join(state.project_dir, "project_description.txt"), "w", encoding="utf-8").write(f"topic: {state.topic}\nrequirement: {state.requirement}")
        
        return {
            "project_dir": state.project_dir, 
            "topic": state.topic, "requirement": state.requirement, 
            "user_example_images": user_example_images_data}
    
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

       
    async def desginer_node(self, state: ArtistState, config=None):
        """进入设计阶段。"""
        logger.info("[desinger_node] 进入")
        from langchain_core.messages import SystemMessage, HumanMessage
        user_prompt_text = (
            self._user_prompt["designer_node"]
        )
        
        user_prompt = create_multimodal_message(text=user_prompt_text, image_data=[state.history_images[-1]])
        
        # logger.info(f"[designer_node] 📝 user_prompt: {user_prompt_text}")
        full_prompt = [self._system_prompt] + [user_prompt]
        logger.info(f"[designer_node] prompt: {full_prompt}")
        if not os.path.exists(os.path.join(state.project_dir, "designer_node.txt")):
            response = self._llm.models.generate_content(
            model='gemini-2.5-pro',
            contents=full_prompt,
            config=GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=EditModel,
                ),
            # tools=[draw_image_tool, edit_image_tool],
        ).parsed
            edit_prompt = response.edit_prompt
            analysis_and_think = response.analysis_and_think
        else:
            edit_prompt = open(os.path.join(state.project_dir, "designer_node.txt"), "r", encoding="utf-8").read()
            analysis_and_think = open(os.path.join(state.project_dir, "analysis_and_think.txt"), "r", encoding="utf-8").read()
            
        open(os.path.join(state.project_dir, "designer_node.txt"), "w", encoding="utf-8").write(edit_prompt)
        open(os.path.join(state.project_dir, "analysis_and_think.txt"), "w", encoding="utf-8").write(analysis_and_think)
        
        logger.info(f"[designer_node] 📝 designer_node response: {edit_prompt}")
        logger.info(f"[designer_node] 📝 designer_node analysis_and_think: {analysis_and_think}")
        
        edit_image = edit_image_tool(edit_prompt, state.history_images[-1])
        edit_image.save(os.path.join(state.project_dir, "designer_node.jpg"))
        history_images = []
        if os.path.exists(os.path.join(state.project_dir, "designer_node.jpg")):
            history_images.append(open(os.path.join(state.project_dir, "designer_node.jpg"), "rb").read())
        return {
                "history_drafts": ["第一次作品分析：", edit_prompt, analysis_and_think],
                "history_images": history_images,
            }

    async def critic_node(self, state: ArtistState, config=None):
        """进入设计阶段。"""
        logger.info("[critic_node] 进入")
        from langchain_core.messages import SystemMessage, HumanMessage
        user_prompt_text = (
self._user_prompt["critic_node"].format(original_draft=open(os.path.join(state.project_dir, "original_draft.txt"), "r", encoding="utf-8").read())
        )
        
        user_prompt = create_multimodal_message(text=user_prompt_text, image_data=[state.history_images[-1]])
        
        # logger.info(f"[designer_node] 📝 user_prompt: {user_prompt_text}")
        full_prompt = [self._system_prompt] + [user_prompt]
        logger.info(f"[critic_node] prompt: {full_prompt}")
        if not os.path.exists(os.path.join(state.project_dir, "critic_node.txt")):
            response = self._llm.models.generate_content(
            model='gemini-2.5-pro',
            contents=full_prompt,
            # temperature=0.5,
            # config=GenerateContentConfig(
            #     # response_modalities=[Modality.TEXT],
            #     temperature=0.7,
            # ),
            # tools=[draw_image_tool, edit_image_tool],
        ).text
        else:
            response = open(os.path.join(state.project_dir, "critic_node.txt"), "r", encoding="utf-8").read()
            
        # open(os.path.join(state.project_dir, "critic_node.txt"), "w", encoding="utf-8").write(response)
        
        logger.info(f"[critic_node] 📝 critic_node response: {response}")
        
        return {
            "history_drafts": [response],
            "history_images": [state.history_images[-1]],
        }

    async def study_node(self, state: ArtistState, config=None):
        """进入设计阶段。"""
        logger.info("[study_node] 进入")
        from langchain_core.messages import SystemMessage, HumanMessage
        user_prompt_text = (
            self._user_prompt["study_node"]
        )
        user_prompt = create_multimodal_message(text=user_prompt_text, image_data=state.user_example_images)
        
        # logger.info(f"[designer_node] 📝 user_prompt: {user_prompt_text}")
        full_prompt = [self._system_prompt] + [user_prompt]
        logger.info(f"[study_node] prompt: {full_prompt}")
        # exit()
        if not os.path.exists(os.path.join(state.project_dir, "study_node.txt")):   
            response = self._llm.models.generate_content(
                model='gemini-2.5-pro',
                contents=full_prompt,
                # define the return schema
                config=GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=DesignModel,
                ),
            )
            try:
                response = response.parsed
                logger.info(f"[study_node] 📝 study_node response: {response}")
                
                positive_figure_gen_prompt = response.positive_figure_gen_prompt
                negative_figure_gen_prompt = response.negative_figure_gen_prompt
                mutual_border_gen_prompt = response.mutual_border_gen_prompt
                analysis_and_think = response.analysis_and_think
                aspect_ratio = response.aspect_ratio

            except:
                open(os.path.join(state.project_dir, "study_node.txt"), "w", encoding="utf-8").write(response)
        else:
            positive_figure_gen_prompt = open(os.path.join(state.project_dir, "positive_figure_gen_prompt.txt"), "r", encoding="utf-8").read()
            negative_figure_gen_prompt = open(os.path.join(state.project_dir, "negative_figure_gen_prompt.txt"), "r", encoding="utf-8").read()
            mutual_border_gen_prompt = open(os.path.join(state.project_dir, "mutual_border_gen_prompt.txt"), "r", encoding="utf-8").read()
            analysis_and_think = open(os.path.join(state.project_dir, "analysis_and_think.txt"), "r", encoding="utf-8").read()
            aspect_ratio = open(os.path.join(state.project_dir, "aspect_ratio.txt"), "r", encoding="utf-8").read()
            
        logger.info(f"[study_node] 📝 study_node positive_figure_gen_prompt: {positive_figure_gen_prompt}")
        logger.info(f"[study_node] 📝 study_node negative_figure_gen_prompt: {negative_figure_gen_prompt}")
        logger.info(f"[study_node] 📝 study_node mutual_border_gen_prompt: {mutual_border_gen_prompt}")
        logger.info(f"[study_node] 📝 study_node analysis_and_think: {analysis_and_think}")
        logger.info(f"[study_node] 📝 study_node aspect_ratio: {aspect_ratio}")

        
        open(os.path.join(state.project_dir, "positive_figure_gen_prompt.txt"), "w", encoding="utf-8").write(positive_figure_gen_prompt)
        open(os.path.join(state.project_dir, "negative_figure_gen_prompt.txt"), "w", encoding="utf-8").write(negative_figure_gen_prompt)
        open(os.path.join(state.project_dir, "mutual_border_gen_prompt.txt"), "w", encoding="utf-8").write(mutual_border_gen_prompt)
        open(os.path.join(state.project_dir, "analysis_and_think.txt"), "w", encoding="utf-8").write(analysis_and_think)
        open(os.path.join(state.project_dir, "aspect_ratio.txt"), "w", encoding="utf-8").write(aspect_ratio)
        # positive_image = draw_image_tool(positive_figure_gen_prompt)
        # negative_image = draw_image_tool(negative_figure_gen_prompt)
        # positive_image.save(os.path.join(state.project_dir, "positive_figure_gen_prompt.jpg"))
        # negative_image.save(os.path.join(state.project_dir, "negative_figure_gen_prompt.jpg"))
        mutual_border_image = draw_image_tool(mutual_border_gen_prompt if mutual_border_gen_prompt else analysis_and_think, aspect_ratio)
        mutual_border_image.save(os.path.join(state.project_dir, "mutual_border_gen_prompt.jpg"))
        
        # image = draw_image_tool(response.text)
        # image.save(os.path.join(state.project_dir, "study_node.jpg"))
        # exit()
        history_images = []
        if os.path.exists(os.path.join(state.project_dir, "positive_figure_gen_prompt.jpg")):
            history_images.append(open(os.path.join(state.project_dir, "positive_figure_gen_prompt.jpg"), "rb").read())
        if os.path.exists(os.path.join(state.project_dir, "negative_figure_gen_prompt.jpg")):
            history_images.append(open(os.path.join(state.project_dir, "negative_figure_gen_prompt.jpg"), "rb").read())
        if os.path.exists(os.path.join(state.project_dir, "mutual_border_gen_prompt.jpg")):
            history_images.append(open(os.path.join(state.project_dir, "mutual_border_gen_prompt.jpg"), "rb").read())
        return {
                "history_drafts": [positive_figure_gen_prompt, negative_figure_gen_prompt, mutual_border_gen_prompt, analysis_and_think],
                "history_images": history_images,
                "aspect_ratio": aspect_ratio,
            }
 
    def build_graph(self):
        from langgraph.graph import StateGraph, START, END
        graph = StateGraph(ArtistState)
        graph.add_node("init_context", self.init_context_node)
        graph.add_node("study_node", self.study_node)
        graph.add_node("designer_node", self.desginer_node)
        graph.add_node("critic_node", self.critic_node)

        # graph.add_node("router_node", self.router_node)
        # graph.add_node("summary_node", self.summary_node)
        # graph.add_node("should_continue", self.should_continue)

        graph.add_edge(START, "init_context")
        graph.add_edge("init_context", "study_node")
        # graph.add_edge("study_node", "designer_node")
        # graph.add_edge("designer_node", "critic_node")
        # graph.add_edge("critic_node", END)
        # graph.add_edge("study_node", END)
        # graph.add_edge("designer_node", END)
        # graph.add_edge("study_node", "draw_node")
        # graph.add_edge("draw_node", END)
        # graph.add_edge("designer_node", "router_node")
        # graph.add_edge("router_node", END)
        # graph.add_edge("wait_node", "router_node")
        return graph
    

def draw_image_tool(prompt: str, aspect_ratio: str):
    """
    draw image with image generation model
    """
    logger.info("[draw_image_tool] 进入")
    design_prompt = prompt
    ## load multi-modal messages 
    user_prompt = f"Highly artistic typography, logo, visual arts. no text. {design_prompt}"
    logger.info(f"[draw_tool] 📝 user_prompt: {user_prompt}")
    image = client.models.generate_images(
        model=generation_model,
        prompt=user_prompt,
        config=GenerateImagesConfig(
            aspect_ratio=aspect_ratio,
            number_of_images=1,
            image_size="1K",
            # safety_filter_level="BLOCK_MEDIUM_AND_ABOVE",
            # person_generation="ALLOW_ADULT",
        ),
    )
    
    return image.generated_images[0].image


def edit_image_tool(edit_prompt, reference_image):
    """
    params:
    edit_prompt: str
    reference_image: bytes
    """
    logger.info("[edit] 进入")
    user_prompt = edit_prompt

    ## load multi-modal messages 
    
    logger.info(f"[edit_node] 📝 user_prompt: {user_prompt}")
    ## from bytes to PIL Image
    raw_ref_image = bytes_to_raw_reference_image(reference_image)

    image = client.models.edit_image(
            model=edit_model,
            reference_images=[raw_ref_image],
            prompt=user_prompt,
            config=EditImageConfig(
                edit_mode="EDIT_MODE_DEFAULT",
                number_of_images=1,
                safety_filter_level="BLOCK_MEDIUM_AND_ABOVE",
                person_generation="ALLOW_ADULT",
            ),
        )
    
    return image.generated_images[0].image

if __name__ == "__main__":
  
    pass
 
 
    
    