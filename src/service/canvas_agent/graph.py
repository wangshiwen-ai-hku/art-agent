"""
Base chat agent implementation that provides common chat functionality
for agents that need interactive conversation capabilities.
"""

from typing import List
import logging
from typing import List, Literal, Tuple
import json
import os
import asyncio
import base64
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langgraph.types import Command, interrupt
from dataclasses import asdict
from pydantic import BaseModel, Field
import re
from langchain_core.tools import tool
from src.agents.base import BaseAgent
from .schema import (
    CanvasState,
    SvgArtwork,
    create_initial_state,
)
from src.config.manager import AgentConfig
from src.infra.tools.image_tools import generate_image_tool
from .utils import svg_to_png, show_messages, convert_svg_to_png_base64
from src.infra.tools.svg_tools import PickPathTools

logger = logging.getLogger(__name__)


class CanvasAgent(BaseAgent):
    """
    A super tutor agent that guides users in designing SVG images.
    Workflow:
    - The user provides an instruction.
    - The tutor agent, using its tools (draw, edit, etc.), processes the request.
    - The agent returns the output (text and/or SVG) and waits for the next instruction.
    """

    name = "canvas_agent"
    description = "An agent that generates design sketches as SVG images."

    def __init__(self, agent_config: AgentConfig):
        super().__init__(agent_config)
        self._load_system_prompt()
        self.config = agent_config

        # The super tutor agent that can use various tools
        self.tutor_agent = create_react_agent(
            model=self.init_llm(),
            tools=self._tools,
            prompt=self._system_prompts["system_prompt"],
        )


    def init_llm(self, temperature: float = 0.7):
        config = self._model_config
        config.temperature = temperature
        return init_chat_model(**asdict(config))

    def _load_system_prompt(self):
        """Loads system prompts from text files."""
        self._system_prompts = {
            "system_prompt": open(
                os.path.join(os.path.dirname(__file__), "prompt/system_prompt.txt"),
                "r",
                encoding="utf-8",
            ).read(),
        }
    
    async def _init_context_node(self, state: CanvasState, config=None):
        logger.info("---CANVAS AGENT: INIT CONTEXT NODE---")
        # This node can be used to initialize the agent's state or context.
        return 
    
    async def wait_for_user_input(self, state: CanvasState, config=None):
        logger.info("---CANVAS AGENT: WAIT FOR USER INPUT---")
        # The graph interrupts here. The input from the next `ainvoke` call
        # is passed to this node as the return value of interrupt().
        payload = interrupt("Wait for user input")
        logger.info(f"-> Wait for user input: {payload['user_input']}")
        return payload

    async def tutor_node(self, state: CanvasState, config=None):
        logger.info("---CANVAS AGENT: TUTOR NODE---")
        try:
            user_input = state["user_input"]
            
            # Prepare agent input
            messages = state["conversation"]["messages"] + [HumanMessage(content=user_input)]
            agent_input = {"messages": messages}
            
            # Add image data if present
            image_contents = []
            if state["content"]["reference_images"]:
                for image_path in state["content"]["reference_images"]:
                    with open(image_path, "rb") as image_file:
                        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
                    image_contents.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                        }
                    )
                # Modify the last message to include images
                last_message = agent_input["messages"][-1]
                last_message.content = [{"type": "text", "text": last_message.content}] + image_contents


            final_state = await self.tutor_agent.ainvoke(agent_input)
            
            # Process the output from the tutor agent
            final_messages = final_state.get("messages", [])
            state["conversation"]["messages"] = final_messages
            show_messages(final_messages)

            # Extract the last valid SVG from tool messages
            new_svg_code = None
            for message in reversed(final_messages):
                if (isinstance(message, AIMessage) or isinstance(message, ToolMessage)) and message.content.strip().startswith('<svg'):
                    new_svg_code = message.content
                    break
            
            if new_svg_code:
                new_svg = SvgArtwork(svg_code=new_svg_code, elements=[]) # Elements can be parsed if needed
                state["content"]["svg_history"].append(new_svg)
                state["content"]["current_svg"] = new_svg

                # Save the SVG and a PNG preview
                if state["project"]["project_dir"]:
                    import time
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    svg_path = os.path.join(
                        state["project"]["project_dir"], f"artwork_{timestamp}.svg"
                    )
                    with open(svg_path, "w", encoding="utf-8") as f:
                        f.write(new_svg_code)
                    png_path = svg_to_png(svg_path)
                    logger.info(f"-> Saved artwork to {svg_path} and PNG to {png_path}")
                    state["project"]["saved_files"].append(png_path)
            
            return state

        except Exception as e:
            logger.error(f"Error in tutor_node: {e}")
            error_message = AIMessage(
                content=f"An error occurred in the tutor agent: {e}"
            )
            state["conversation"]["messages"].append(error_message)
            return state

    
    def build_graph(self):
        graph = StateGraph(CanvasState)
        
        graph.add_node("tutor_node", self.tutor_node)
        graph.add_node("wait_for_user_input", self.wait_for_user_input)
        graph.add_node("init_context_node", self._init_context_node)
        
        graph.set_entry_point("init_context_node")
        graph.add_edge("init_context_node", "wait_for_user_input")
        graph.add_edge("wait_for_user_input", "tutor_node")
        graph.add_edge("tutor_node", "wait_for_user_input")

        return graph
    
    


    
    