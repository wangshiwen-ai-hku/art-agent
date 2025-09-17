"""
Base chat agent implementation that provides common chat functionality
for agents that need interactive conversation capabilities.
"""

import logging
from typing import List
import json
import os
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

from src.agents.base import BaseAgent
from .schema import CanvasState, SketchDraft, SketchOutput
from src.config.manager import AgentConfig
from pydantic import BaseModel

from src.infra.tools import canvas_tools

try:
    import cairosvg
except ImportError:
    cairosvg = None

logger = logging.getLogger(__name__)


class SketchImaginary(BaseModel):
    """A model for parsing the output of the imaginer LLM call."""
    sketch_imaginary: List[SketchDraft]
    design_analysis: str


class CanvasAgent(BaseAgent):
    """
    An agent that generates design sketches as SVG images.
    Workflow:
    1.  Imaginer Node: Brainstorms textual descriptions and drawing steps for multiple sketch ideas.
    2.  Drawer Node: Converts each sketch idea into an SVG using drawing tools.
    """

    name = "canvas_agent"
    description = "An agent that generates design sketches as SVG images."

    def __init__(self, agent_config: AgentConfig):
        super().__init__(agent_config)
        self._load_system_prompt()
        self.config = agent_config
        
        # Create a fully-fledged agent that can manage the tool-calling loop internally.
        # This agent will use the "drawer" system prompt as its instructions.
        system_prompt = self._system_prompts["drawer"]
        self.drawer_agent = create_react_agent(model=self._llm, tools=self._tools, prompt=system_prompt)


    def _load_system_prompt(self):
        """Load the system prompts for the agent's nodes."""
        self._system_prompts = {
            "imaginer": open(os.path.join(os.path.dirname(__file__), "prompt/imaginer_prompt.txt"), "r", encoding="utf-8").read(),
            "drawer": open(os.path.join(os.path.dirname(__file__), "prompt/drawer_prompt.txt"), "r", encoding="utf-8").read(),
        }

    def get_structure_llm(self, schema: BaseModel):
        return self._llm.with_structured_output(schema)

    async def imagine_node(self, state: CanvasState, config=None):
        """Brainstorms sketch ideas and drawing instructions."""
        logger.info("---CANVAS AGENT: IMAGINER NODE---")
        
        prompt_template = self._system_prompts["imaginer"]
        prompt = prompt_template.format(topic=state['topic'], technique=state.get('technique', ''), requirement=state.get('requirement', ''))
        
        messages = [HumanMessage(content=prompt)]
        
        structured_llm = self.get_structure_llm(schema=SketchImaginary)
        llm_output = await structured_llm.ainvoke(messages)
        
        sketch_ideas = llm_output.sketch_imaginary

        # Save the raw output for debugging
        if state.get('project_dir'):
            # Save the full output from the imaginer LLM
            with open(os.path.join(state['project_dir'], "sketch_draft.json"), "w", encoding="utf-8") as f:
                json.dump(llm_output.model_dump(), f, ensure_ascii=False, indent=4)

            # Save just the sketch ideas list as requested
            ideas_as_dicts = [idea.model_dump() for idea in sketch_ideas]
            with open(os.path.join(state['project_dir'], "sketch_ideas.json"), "w", encoding="utf-8") as f:
                json.dump(ideas_as_dicts, f, ensure_ascii=False, indent=4)

        return {"sketch_ideas": sketch_ideas}

    async def draw_sketches_node(self, state: CanvasState, config=None):
        """Uses the drawer agent to execute the drawing prompts for each idea."""
        logger.info("---CANVAS AGENT: DRAWER NODE---")
        
        sketches: List[SketchOutput] = []

        for i, idea in enumerate(state['sketch_ideas']):
            logger.info(f"-> Drawing sketch {i+1}...")
            
            prompt = idea.drawing_prompt
            messages = [HumanMessage(content=prompt)]
            
            # The create_react_agent expects a dictionary with a "messages" key.
            agent_input = {"messages": messages}
            
            # Invoke the agent. It will now handle the entire tool-calling loop.
            final_state = await self.drawer_agent.ainvoke(agent_input)

            # Extract the tool outputs (SVG strings) from the final message list.
            svg_elements = [
                message.content 
                for message in final_state['messages'] 
                if isinstance(message, ToolMessage)
            ]

            # Assemble the final SVG
            final_svg = f'<svg width="1024" height="1024" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024">{"".join(svg_elements)}</svg>'
            
            # Save the SVG file and convert to PNG
            if state.get('project_dir'):
                svg_path = os.path.join(state['project_dir'], f"sketch_{i}.svg")
                jpg_path = os.path.join(state['project_dir'], f"sketch_{i}.png")
                with open(svg_path, "w", encoding="utf-8") as f:
                    f.write(final_svg)

                if cairosvg:
                    try:
                        cairosvg.svg2png(bytestring=final_svg.encode('utf-8'), write_to=jpg_path, background_color="white")
                        logger.info(f"-> Saved JPG sketch to {jpg_path}")
                    except Exception as e:
                        logger.error(f"-> Failed to convert SVG to JPG for sketch {i+1}: {e}")
                else:
                    logger.warning("-> `cairosvg` is not installed. Skipping JPG conversion. To install, run: pip install cairosvg")
            
            sketches.append({
                "idea": idea,
                "svg_elements": svg_elements,
                "final_svg": final_svg
            })

        return {"sketches": sketches}

    def build_graph(self):
        graph = StateGraph(CanvasState)
        graph.add_node("imagine_node", self.imagine_node)
        graph.add_node("draw_sketches_node", self.draw_sketches_node)

        graph.add_edge(START, "imagine_node")
        graph.add_edge("imagine_node", "draw_sketches_node")
        graph.add_edge("draw_sketches_node", END)
        
        return graph
    



    
    