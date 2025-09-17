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
from pydantic import BaseModel
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
# from .prompt import FOV_SYSTEM_PROMPT, FOV_CHAT_SYSTEM_PROMPT
from .config import ThreadConfiguration
from .schema import SketchState
from src.config.manager import AgentConfig, MultiAgentConfig
from src.utils.input_processor import create_vertex_multimodal_message as create_multimodal_message
from src.utils.input_processor import bytes_to_raw_reference_image
# from src.utils import multi_modal_message_loader
from src.utils.input_processor import process_image_bytes
from .schema import SketchDraft, SketchImaginary
from src.infra.tools.image_generate_edit import generate_image_tool, edit_image_tool

logger = logging.getLogger(__name__)

generation_model = "imagen-3.0-generate-001"
# generation_model_fast = "imagen-4.0-fast-generate-001"
# generation_model_ultra = "imagen-4.0-ultra-generate-001"
edit_model = "imagen-3.0-capability-001"

class SketchAgent(BaseAgent):
    """
    Base chat agent that provides common interactive conversation functionality.
    This class can be inherited by specific agents like tutor, solver, etc.
    """

    name = "sketch_agent"
    description = "sketch agent that provides interactive conversation capabilities."

    def __init__(self, agent_config: AgentConfig):
        """Initialize the  agent.

        Args:
            agent_config: Configuration object containing model parameters and tools.
        """
        super().__init__(agent_config)
        logger.info(f"{self.name} chat agent initialized successfully")
        # self._wrap_llm_with_callback()
        self.config = agent_config

    def _load_system_prompt(self):
        """Load the system prompt for the chat agent."""
        self._system_prompt = open(os.path.join(os.path.dirname(__file__), "prompt/system_prompt.txt"), "r").read()
        # self._system_prompt = self._system_prompt_text
        
        # self._system_image_data = [os.path.join(os.path.dirname(__file__), "examples", "sketch_example.jpg")]
        # self._system_image_data = process_image_bytes(self._system_image_data)
        
        self._user_prompt = {
            "sketch_node": open(os.path.join(os.path.dirname(__file__), "prompt/sketch_prompt.txt"), "r").read(),
        }


    async def imagine_node(self, state: SketchState, config=None):
        """进入想象阶段。"""
        logger.info("[imagine_node] 进入")

    def get_structure_llm(self, schema: BaseModel):
        return self._llm.with_structured_output(schema)
    
    async def sketch_node(self, state: SketchState, config=None):
        """进入草图绘制阶段。"""
        logger.info("[sketch_node] 进入")
        
        prompt = self._user_prompt["sketch_node"].format(topic=state['topic'], technique=state['technique'])
        
        prompt = [SystemMessage(content=self._system_prompt), HumanMessage(content=prompt)]
        # llm_output = self._llm.invoke(prompt)

        sketch_draft: SketchImaginary = await self.get_structure_llm(schema=SketchImaginary).ainvoke(prompt)
        
        sketch_draft_list: list = sketch_draft.sketch_imaginary
        design_analysis: str = sketch_draft.design_analysis
        logger.info(f"design_analysis: {design_analysis}")
        
        with open(os.path.join(state['project_dir'], "sketch_draft.json"), "w") as f:
            json.dump(sketch_draft.model_dump(), f, ensure_ascii=False, indent=4)
        
        open(os.path.join(state['project_dir'], "prompt.txt"), "w").write(self._system_prompt + "\n" + self._user_prompt["sketch_node"])

        for idx, d in enumerate(sketch_draft_list):
            logger.info(f"sketch_prompt: {d.sketch_prompt}")
            logger.info(f"sketch_description: {d.design_description}")
            logger.info(f"sketch_elements: {d.sketch_elements}")
            logger.info(f"sketch_style: {d.sketch_style}")
            
            sketch_image = generate_image_tool(d.sketch_prompt, '1:1')
            sketch_image.save(os.path.join(state['project_dir'], f"sketch_{idx}.png"))

        return state
     
        
        
 
    def build_graph(self):
        from langgraph.graph import StateGraph, START, END
        graph = StateGraph(SketchState)
        graph.add_node("imagine_node", self.imagine_node)
        graph.add_node("sketch_node", self.sketch_node)
        # graph.add_node("designer_node", self.desginer_node)
        # graph.add_node("critic_node", self.critic_node)

        graph.add_edge(START, "sketch_node")
        return graph
    



    
    