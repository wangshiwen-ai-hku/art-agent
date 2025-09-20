"""
Base chat agent implementation that provides common chat functionality
for agents that need interactive conversation capabilities.
"""

import logging
from typing import List, Literal, Tuple
import json
import os
import asyncio
import io
import base64
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

from PIL import Image

from src.agents.base import BaseAgent
from .schema import CanvasState, SketchDraft, SketchOutput
from src.config.manager import AgentConfig
from pydantic import BaseModel, Field
import re
from src.infra.tools.image_generate_edit import generate_image_tool
from src.infra.tools.math_tools import calculator, find_points_with_shape, calculate_points, arc_points_from_center, line_line_intersection, point_on_arc, line_circle_intersection
from src.agents.canvas_agent.utils import svg_to_png, show_messages

try:
    import cairosvg
except ImportError:
    cairosvg = None

logger = logging.getLogger(__name__)


class SketchImaginary(BaseModel):
    """A model for parsing the output of the imaginer LLM call."""
    sketch_imaginary: List[SketchDraft]
    design_analysis: str


class ImagePrompt(BaseModel):
    """A model for parsing the output of the imaginer LLM call."""
    prompt: str = Field(description="the prompt of image generation")


class Critique(BaseModel):
    """A model for parsing the output of the critique LLM call."""
    critique: str = Field(description="constructive critique of the design concept")

class DescribeOutput(BaseModel):
    """A model for parsing the output of the describe LLM call."""
    description: str = Field(description="a detailed description of the image to svg mode")

    
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
        planner_tools = [find_points_with_shape, calculate_points, line_circle_intersection, line_line_intersection, point_on_arc, arc_points_from_center]
        
        self.drawer_agent = create_react_agent(model=self._math_llm, tools= [find_points_with_shape, calculate_points, line_circle_intersection, line_line_intersection, point_on_arc, arc_points_from_center] + self._tools, prompt=drawer_system_prompt,)

        self.describe_agent = create_react_agent(model=self._math_llm, tools=[find_points_with_shape, calculate_points, line_circle_intersection, line_line_intersection, point_on_arc, arc_points_from_center], prompt=self._system_prompts["describe_prompt"])
        
        # The Planner Agent: Designs the logo using mathematical tools
        planner_system_prompt = self._system_prompts["sketch_planner"]
        
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
            "image_generator_prompt": open(os.path.join(os.path.dirname(__file__), "prompt/image_generator_prompt.txt"), "r", encoding="utf-8").read(),
            "critique_prompt": open(os.path.join(os.path.dirname(__file__), "prompt/critique_prompt.txt"), "r", encoding="utf-8").read(),
            "refine_prompt": open(os.path.join(os.path.dirname(__file__), "prompt/refine_prompt.txt"), "r", encoding="utf-8").read(),
            "describe_prompt": open(os.path.join(os.path.dirname(__file__), "prompt/describe_prompt.txt"), "r", encoding="utf-8").read(),
        }

    def get_structure_llm(self, schema: BaseModel, llm=None):
        if llm is None:
            llm = self._llm
        return llm.with_structured_output(schema)

    async def _generate_prompt(self, state: CanvasState):
        logger.info("---CANVAS AGENT: GENERATE PROMPT NODE---")
        prompt_template = self._system_prompts["image_generator_prompt"]
        prompt = prompt_template.format(topic=state['topic'])
        messages = [HumanMessage(content=prompt)]
        structured_llm = self.get_structure_llm(schema=ImagePrompt)
        
        llm_output = await structured_llm.ainvoke(messages)
        return llm_output.prompt

    async def generate_images_node(self, state: CanvasState, config=None):
        """Generates a single composite image with four unique variations based on the topic."""
        logger.info("---CANVAS AGENT: IMAGE GENERATOR NODE---")
        
        prompt = await self._generate_prompt(state)
        image_paths = []
        
        try:
            image_pil = await asyncio.to_thread(
                generate_image_tool,
                prompt=prompt,
                aspect_ratio="1:1"
            )
            save_path = os.path.join(state['project_dir'], "generated_image_composition.png")
            image_pil.save(save_path)
            logger.info(f"-> Saved generated image to {save_path}")
            image_paths.append(save_path)
           
        except Exception as e:
            logger.error(f"-> Failed to generate or save image: {e}")
        
        image_paths = [path for path in image_paths if path]
        return {"generated_image_paths": image_paths}
        
    async def imagine_node(self, state: CanvasState, config=None):
        """Brainstorms sketch ideas and drawing instructions."""
        logger.info("---CANVAS AGENT: IMAGINER NODE---")
        
        prompt_template = self._system_prompts["imaginer"]
        
        # Base64 encode the generated images
        image_contents = []
        if state.get("generated_image_paths"):
            for image_path in state["generated_image_paths"]:
                try:
                    with open(image_path, "rb") as image_file:
                        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
                        image_contents.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                        })
                except Exception as e:
                    logger.error(f"Error encoding image {image_path}: {e}")

        # Construct the multimodal message
        prompt_text = prompt_template.format(
            topic=state['topic'], 
            technique=state.get('technique', ''), 
            requirement=state.get('requirement', '')
        )
        
        messages = [
            HumanMessage(content=[
                {"type": "text", "text": prompt_text},
                *image_contents
            ])
        ]
        
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

    async def critique_and_refine_node(self, state: CanvasState, config=None):
        """Critiques and refines the initial sketch ideas."""
        logger.info("---CANVAS AGENT: CRITIQUE AND REFINE NODE---")
        
        initial_ideas = state['sketch_ideas']
        refined_ideas = []

        critique_llm = self.get_structure_llm(schema=Critique)
        refine_llm = self.get_structure_llm(schema=SketchDraft)

        critique_prompt_template = self._system_prompts["critique_prompt"]
        refine_prompt_template = self._system_prompts["refine_prompt"]

        for i, idea in enumerate(initial_ideas):
            logger.info(f"-> Critiquing idea #{i+1}...")
            
            # 1. Critique the idea
            critique_input = f"**Design Description:**\n{idea.design_description}\n\n**Sketch Description:**\n{idea.sketch_description}"
            critique_messages = [
                SystemMessage(content=critique_prompt_template),
                HumanMessage(content=critique_input)
            ]
            critique_output = await critique_llm.ainvoke(critique_messages)
            critique_text = critique_output.critique
            logger.info(f"-> Critique #{i+1}: {critique_text}")

            # 2. Refine the idea based on the critique
            logger.info(f"-> Refining idea #{i+1}...")
            refine_input = (
                f"**Original Idea:**\n"
                f"Design Description: {idea.design_description}\n"
                f"Sketch Description: {idea.sketch_description}\n\n"
                f"**Design Director's Critique:**\n{critique_text}"
            )
            refine_messages = [
                SystemMessage(content=refine_prompt_template),
                HumanMessage(content=refine_input)
            ]
            
            refined_idea = await refine_llm.ainvoke(refine_messages)
            refined_ideas.append(refined_idea)
            logger.info(f"-> Refined Idea #{i+1} Description: {refined_idea.design_description}")

        # Save the refined ideas for debugging
        if state.get('project_dir'):
            refined_ideas_as_dicts = [idea.model_dump() for idea in refined_ideas]
            with open(os.path.join(state['project_dir'], "refined_sketch_ideas.json"), "w", encoding="utf-8") as f:
                json.dump(refined_ideas_as_dicts, f, ensure_ascii=False, indent=4)

        return {"sketch_ideas": refined_ideas}

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
            show_messages(final_state.get("messages", []))
            
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
                with open(svg_path, "w", encoding="utf-8") as f:
                    f.write(final_svg)
                
                png_path = svg_to_png(svg_path)
                
            sketches.append({
                "idea": idea,
                "svg_elements": svg_elements,
            })

        return {"sketches": sketches}

    async def draw_sketches_only_node(self, state: CanvasState, config=None):
        """Uses the drawer agent to execute the drawing prompts for each idea."""
        logger.info("---CANVAS AGENT: DRAW ONLY NODE---")
        
        draw_prompt = state['draw_prompt']
        logger.info(f"-> Draw Prompt: {draw_prompt}")
        messages = [HumanMessage(content=draw_prompt)]
            
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
        output_dir = state.get('project_dir', 'output/test')
        os.makedirs(output_dir, exist_ok=True)
            # Save the SVG file and convert to PNG
        if output_dir:
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")  
            output_name = f"sketch_{timestamp}"
            svg_path = os.path.join(output_dir, f"{output_name}.svg")
            # png_path = os.path.join(output_dir, f"{output_name}.png")
            with open(svg_path, "w", encoding="utf-8") as f:
                f.write(final_svg)

            png_path = svg_to_png(svg_path)
            
        return {"generated_image_paths": [png_path]}

    async def describe_only_node(self, state: CanvasState, config=None):
        logger.info("---CANVAS AGENT: DESCRIBE ONLY NODE---")
        image_contents = []
        if state.get("generated_image_paths"):
            for image_path in state["generated_image_paths"]:
                try:
                    with open(image_path, "rb") as image_file:
                        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
                        image_contents.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                        })
                except Exception as e:
                    logger.error(f"Error encoding image {image_path}: {e}")
        
        messages = [
            HumanMessage(content=[
                {"type": "text", "text": "Please describe the image in svg desciption."},
                *image_contents
            ])
        ]
        
        # structured_llm = self.get_structure_llm(schema=DescribeOutput, llm=self._math_llm)
        llm_state = await self.describe_agent.ainvoke({"messages": messages})
        
        show_messages(llm_state["messages"])
        try:
            description = llm_state["messages"][-1].additional_kwargs["parsed"].description
            logger.info(f"-> Description: {description}")
        except Exception as e:
            logger.info(f"-> Error parsing description: {e}")
            description = llm_state["messages"][-1].content
            logger.info(f"-> Description: {description}")
        
        return {"draw_prompt": description} 
        
    def router_node(self, state: CanvasState, config=None):
        stage = state.get('stage', 'generate')
        logger.info(f"-> Router Node: {stage}")
        
        if stage == 'draw_only':
            return 'draw_only'
        elif stage == 'describe_only':
            return 'describe_only'
        else:
            return 'generate'
 

    def build_graph(self):
        graph = StateGraph(CanvasState)
        # graph.add_node("router_node", self.router_node)
        graph.add_node("imagine_node", self.imagine_node)
        graph.add_node("draw_sketches_node", self.draw_sketches_node)
        graph.add_node("draw_sketches_only_node", self.draw_sketches_only_node)
        graph.add_node("describe_only_node", self.describe_only_node)
        
        graph.add_node("refine_sketch_node", self.refine_sketch_node)
        graph.add_node("generate_images_node", self.generate_images_node)
        graph.add_node("critique_and_refine_node", self.critique_and_refine_node)

        # graph.add_edge(START, "router_node")
        graph.add_conditional_edges(START, self.router_node, {
            'draw_only': "draw_sketches_only_node",
            'describe_only': "describe_only_node",
            'generate': "generate_images_node",
        })
        # graph.add_edge("router_node", "generate_images_node")
        # graph.add_conditional_edges('router_node', 
        #                             self.router_node,
        #                             {
        #                                 'draw_only': 'draw_sketches_only_node',
        #                                 'describe_only': 'describe_only_node',
        #                                 'generate': 'generate_images_node',
        #                             })
        
        # graph.add_edge('generate_images_node', 'imagine_node')
        graph.add_edge("generate_images_node", "imagine_node")
        graph.add_edge("imagine_node", "critique_and_refine_node")
        graph.add_edge("critique_and_refine_node", "refine_sketch_node")
        graph.add_edge("refine_sketch_node", "draw_sketches_node")
        graph.add_edge("draw_sketches_node", END)
        
        
        graph.add_edge("draw_sketches_only_node", END)
        graph.add_edge("describe_only_node",      "draw_sketches_only_node")
        # graph.add_edge("draw_sketches_only_node", END)
        # graph.add_edge("describe_only_node", "draw_sketches_only_node")
        
        return graph
    
    


    
    