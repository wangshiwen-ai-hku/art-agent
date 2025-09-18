"""
Base chat agent implementation that provides common chat functionality
for agents that need interactive conversation capabilities.
"""

import logging
from typing import List, Literal, Tuple
import json
import os
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

from src.agents.base import BaseAgent
from .schema import CanvasState, SketchDraft, SketchOutput
from src.config.manager import AgentConfig
from pydantic import BaseModel
import re

from src.infra.tools.math_tools import calculator

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
        self.load_math_llm()
        
        # The Drawer Agent: Executes drawing instructions
        drawer_system_prompt = self._system_prompts["drawer"]
        self.drawer_agent = create_react_agent(model=self._llm, tools=self._tools, prompt=drawer_system_prompt)

        # The Planner Agent: Designs the logo using mathematical tools
        planner_system_prompt = self._system_prompts["sketch_planner"]
        planner_tools = [calculator]
        self.planner_agent = create_react_agent(model=self._math_llm, tools=planner_tools + self._tools, prompt=planner_system_prompt)

    def load_math_llm(self):
        self._math_llm = self._init_llm()

    def _load_system_prompt(self):
        """Loads system prompts and prepares SVG examples."""
        
        def _extract_svg_path(file_name: str) -> str:
            """Reads an SVG file and extracts the path data from the 'd' attribute."""
            try:
                file_path = os.path.join(os.path.dirname(__file__), "examples", file_name)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                match = re.search(r'<path[^>]*d="([^"]+)"', content)
                if match:
                    return f'd="{match.group(1)}"'
                return "Path data not found."
            except FileNotFoundError:
                logger.error(f"Example SVG file not found: {file_name}")
                return f"[Could not load example: {file_name}]"
            except Exception as e:
                logger.error(f"Error processing SVG file {file_name}: {e}")
                return "[Error loading example]"

        # Load example paths into instance variables
        self.github_path_example = _extract_svg_path("github-svgrepo-com.svg")
        self.starbucks_path_example = _extract_svg_path("starbucks-svgrepo-com.svg")
        self.jtb_path_example = _extract_svg_path("jtbmd-svgrepo-com.svg")

        # Load prompts
        self._system_prompts = {
            "imaginer": open(os.path.join(os.path.dirname(__file__), "prompt/imaginer_prompt.txt"), "r", encoding="utf-8").read(),
            "drawer": open(os.path.join(os.path.dirname(__file__), "prompt/drawer_prompt.txt"), "r", encoding="utf-8").read(),
            "sketch_refiner_input": open(os.path.join(os.path.dirname(__file__), "prompt/sketch_refiner_prompt.txt"), "r", encoding="utf-8").read(),
            "sketch_planner": open(os.path.join(os.path.dirname(__file__), "prompt/sketch_planner_prompt.txt"), "r", encoding="utf-8").read(),
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

    async def refine_sketch_node(self, state: CanvasState, config=None):
        """Uses a planning agent with a calculator to generate a detailed drawing prompt."""
        logger.info("---CANVAS AGENT: PLANNER NODE---")
        
        refiner_input_template = self._system_prompts["sketch_refiner_input"]
        
        sketch_ideas = state['sketch_ideas']
        
        for idea in sketch_ideas:
            logger.info(f"-> Planning sketch with math: {idea.design_description}")
            
            # This is the input to the planner, containing examples and the specific request
            planner_input = refiner_input_template.format(
                sketch_description=idea.sketch_description,
                # github_path=self.github_path_example,
                # starbuck_path=self.starbucks_path_example,
                # jtb_path=self.jtb_path_example
            ) + self._system_prompts["sketch_planner"]
            
            messages = [HumanMessage(content=planner_input)]
            agent_input = {"messages": messages}
            
            # Invoke the planner agent. It will use the calculator to think.
            final_state = await self.planner_agent.ainvoke(agent_input)

            # The final answer from the planner is the drawing prompt for the next agent.
            drawing_prompt = final_state['messages'][-1].content
            
            idea.drawing_prompt = drawing_prompt
            logger.info(f"-> Generated Drawing Prompt: {idea.drawing_prompt}")

        return {"sketch_ideas": sketch_ideas}


    async def draw_sketches_node(self, state: CanvasState, config=None):
        """Uses the drawer agent to execute the drawing prompts for each idea."""
        logger.info("---CANVAS AGENT: DRAWER NODE---")
        
        sketches: List[SketchOutput] = []

        for i, idea in enumerate(state['sketch_ideas']):
            logger.info(f"-> Drawing sketch {i+1}...")
            
            if not idea.drawing_prompt:
                logger.warning(f"-> Skipping sketch {i+1} as it has no drawing prompt.")
                continue

            messages = [HumanMessage(content=idea.drawing_prompt)]
            
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
            final_svg = f'<svg width="1000" height="1000" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">{"".join(svg_elements)}</svg>'
            
            # Save the SVG file and convert to PNG
            if state.get('project_dir'):
                svg_path = os.path.join(state['project_dir'], f"sketch_{i}.svg")
                png_path = os.path.join(state['project_dir'], f"sketch_{i}.png")
                with open(svg_path, "w", encoding="utf-8") as f:
                    f.write(final_svg)

                if cairosvg:
                    try:
                        cairosvg.svg2png(bytestring=final_svg.encode('utf-8'), write_to=png_path, background_color="white")
                        logger.info(f"-> Saved PNG sketch to {png_path}")
                    except Exception as e:
                        logger.error(f"-> Failed to convert SVG to PNG for sketch {i+1}: {e}")
                else:
                    logger.warning("-> `cairosvg` is not installed. Skipping PNG conversion. To install, run: pip install cairosvg")
            
            sketches.append({
                "idea": idea,
                "svg_elements": svg_elements,
            })

        return {"sketches": sketches}

    def build_graph(self):
        graph = StateGraph(CanvasState)
        graph.add_node("imagine_node", self.imagine_node)
        graph.add_node("draw_sketches_node", self.draw_sketches_node)
        graph.add_node("refine_sketch_node", self.refine_sketch_node)

        graph.add_edge(START, "imagine_node")
        graph.add_edge("imagine_node", "refine_sketch_node")
        graph.add_edge("refine_sketch_node", "draw_sketches_node")
        graph.add_edge("draw_sketches_node", END)
        
        return graph
    
    


    
    