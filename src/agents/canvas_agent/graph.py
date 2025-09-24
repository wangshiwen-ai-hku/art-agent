"""
Base chat agent implementation that provides common chat functionality
for agents that need interactive conversation capabilities.
"""

from code import interact
import logging
from typing import List, Literal, Tuple
import json
import os
import asyncio
import io
import base64
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langgraph.types import Command

from dataclasses import asdict
from PIL import Image
from langgraph.types import interrupt

from src.agents.base import BaseAgent
from .schema import CanvasState, SketchDraft, SketchOutput, STATE_MAP
from src.config.manager import AgentConfig, config
from pydantic import BaseModel, Field
import re
from src.infra.tools.image_generate_edit import generate_image_tool, generate_image
from src.infra.tools.math_tools import calculator, find_points_with_shape, calculate_points, arc_points_from_center, line_line_intersection, point_on_arc, line_circle_intersection
from src.agents.canvas_agent.utils import svg_to_png, show_messages, convert_svg_to_png_base64
from src.infra.tools.svg_tools import PickPathTools
from .supervisor import create_supervisor, _prepare_tool_node

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
        
        # The Drawer Agent: Executes drawing instructions
        drawer_system_prompt = self._system_prompts["drawer"]
        planner_tools = [find_points_with_shape, calculate_points, line_circle_intersection, line_line_intersection, point_on_arc, arc_points_from_center]
        
        self.drawer_agent = create_react_agent(model=self.init_llm(0.2), name="drawer_agent", tools= [find_points_with_shape, calculate_points, line_circle_intersection, line_line_intersection, point_on_arc, arc_points_from_center] + self._tools, prompt=drawer_system_prompt,)
        
        # describe agent, with svg2png and png2base64 tools to help understand the svg code.
        self.describe_agent = create_react_agent(model=self.init_llm(), name="describe_agent", tools=planner_tools, prompt=self._system_prompts["describe_prompt"])
        
        self.pick_path_agent = create_react_agent(model=self.init_llm(), name='pick_path_agent',
                                                  tools=PickPathTools, prompt=self._system_prompts["pick_path_prompt"])
        # edit agent, possible with optimization tools
        self.edit_agent = create_react_agent(model=self.init_llm(), name="edit_agent",
                                             tools=planner_tools + self._tools, prompt=self._system_prompts["edit_prompt"])
        
        # brainstorm agent, image generation agent with tools
        # self.brainstorm_agent = create_react_agent(model=self.init_llm(1.0), tools=[generate_image], prompt=self._system_prompts["brainstorm_prompt"])
              
        # planner agent, math agent with tools
        self.planner_agent = create_react_agent(model=self.init_llm(), name="planner_agent", tools=planner_tools, prompt=self._system_prompts["sketch_planner"])
        
        self._tool_node = _prepare_tool_node(tools=self._tools, handoff_tool_prefix="", add_handoff_messages=True, agent_names=set('drawer_agent'))
        self._llm.bind_tools([self._tool_node])
        supervisor_graph = self.build_graph()
        

    def init_llm(self, temperature: float = 0.7):
        config = self._model_config
        config.temperature = temperature
        return init_chat_model(**asdict(config))

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
            "edit_prompt": open(os.path.join(os.path.dirname(__file__), "prompt/edit_prompt.txt"), "r", encoding="utf-8").read(),
            "chat_prompt": open(os.path.join(os.path.dirname(__file__), "prompt/chat_prompt.txt"), "r", encoding="utf-8").read(),
            "imaginer": open(os.path.join(os.path.dirname(__file__), "prompt/imaginer_prompt.txt"), "r", encoding="utf-8").read(),
            "drawer": open(os.path.join(os.path.dirname(__file__), "prompt/drawer_prompt.txt"), "r", encoding="utf-8").read(),
            "sketch_refiner_input": open(os.path.join(os.path.dirname(__file__), "prompt/sketch_refiner_prompt.txt"), "r", encoding="utf-8").read(),
            "sketch_planner": open(os.path.join(os.path.dirname(__file__), "prompt/sketch_planner_prompt.txt"), "r", encoding="utf-8").read(),
            "image_generator_prompt": open(os.path.join(os.path.dirname(__file__), "prompt/image_generator_prompt.txt"), "r", encoding="utf-8").read(),
            "critique_prompt": open(os.path.join(os.path.dirname(__file__), "prompt/critique_prompt.txt"), "r", encoding="utf-8").read(),
            "refine_prompt": open(os.path.join(os.path.dirname(__file__), "prompt/refine_prompt.txt"), "r", encoding="utf-8").read(),
            "describe_prompt": open(os.path.join(os.path.dirname(__file__), "prompt/describe_prompt.txt"), "r", encoding="utf-8").read(),
            "pick_path_prompt": open(os.path.join(os.path.dirname(__file__), "prompt/pick_path_prompt.txt"), "r", encoding="utf-8").read(),
        }

    def get_structure_llm(self, schema: BaseModel, llm=None):
        if llm is None:
            llm = self._llm
        return llm.with_structured_output(schema)
    
    async def _init_context_node(self, state: CanvasState, config=None):
        logger.info("---CANVAS AGENT: INIT CONTEXT NODE---")
        # state['messages'] = [SystemMessage(content=self._system_prompts["chat_prompt"])] + state.get("messages", [])
        return 
    
    async def pick_path_node(self, state: CanvasState, config=None):
        logger.info("---CANVAS AGENT: PICK PATH NODE---")
        try:
            user_message = state.get('user_message')
            input_svg_code = state.get('svg_history', [])[-1]
            
            input_image = convert_svg_to_png_base64(input_svg_code)
            prompt = self._system_prompts["pick_path_prompt"].format(
                user_message=user_message,
                svg_code=input_svg_code,
            )
            text_content = {
                "type": "text",
                "text": prompt
            }
            image_content = {
                "type": "image_url",
                "image_url": {"url": input_image}
            }
            agent_input = {"messages": [HumanMessage(content=[image_content,text_content])]}
            logger.info(f"-> Picking path with input: {prompt}")
            async for chunk in self.pick_path_agent.astream(agent_input):
                # print(chunk)
                if chunk.get('messages'):
                    show_messages(chunk.get('messages', []))

            # final_state = await self.pick_path_agent.ainvoke(agent_input)
            show_messages(final_state.get('messages', []))
            picked_path = final_state.get('messages', [])[-1].content
            return {"svg_history": [picked_path], "messages": [HumanMessage(content=prompt), AIMessage(content="Successfull picked path:" + picked_path)]}
        
        except Exception as e:
            logger.error(f"Error in pick_path_node: {e}")
            error_message = AIMessage(content=f"An error occurred during picking path: {e}")
            state["messages"].append(error_message)
            return Command(goto="wait_for_user_input", )
        
    async def wait_for_user_input(self, state: CanvasState, config=None):
        logger.info("---CANVAS AGENT: WAIT FOR USER INPUT---")
        # The graph interrupts here. The input from the next `ainvoke` call
        # is passed to this node as the return value of interrupt().
        payload = interrupt("Wait for user input")
        logger.info(f"-> Resuming with payload: {payload}")
        assert payload.get("user_message"), "user_message is required"
        stage = payload.get("stage", "chat")
        if stage == 'end':
            return Command(goto=END)
        
        if stage not in STATE_MAP:
            return Command(goto="wait_for_user_input", update={"messages": [AIMessage(content="Invalid stage-" + stage)]})
        
        logger.info(f"-> Routing to stage: '{stage}'")
        return Command(goto=STATE_MAP[stage], update=payload)
    
    async def chat_node(self, state: CanvasState, config=None):
        
        logger.info("---CANVAS AGENT: CHAT NODE---")

        messages = state['messages'] + [HumanMessage(content=state['user_message'])]
        print("====chat history====")
        show_messages(messages)
        
        try:
            response = await self._llm.ainvoke(messages)
        except Exception as e:
            logger.error(f"-> Error in chat node: {e}")
            err_msg = AIMessage(content=f"Error in chat node: {e}")
            return {"messages": [err_msg], "response": err_msg.content}
        
        # logger.info(f"-> Chat response: {response.content}")
        
        return {
            "messages": [response],
            "response": response.content
        }
    
    async def edit_node(self, state: CanvasState, config=None):
        logger.info("---CANVAS AGENT: EDIT NODE---")
        try:
            user_instruction = state.get('user_instruction')
            svg_history = state.get('svg_history', [])

            if not user_instruction:
                logger.warning("-> No user instruction found for editing. Skipping.")
                return {}

            if not svg_history:
                logger.warning("-> No SVG history found for editing. Skipping.")
                return {}
            
            input_svg_code = svg_history[-1]
            
            prompt = self._system_prompts["edit_prompt"].format(
                user_instruction=user_instruction,
                input_svg_code=input_svg_code
            )
            
            agent_input = {"messages": [HumanMessage(content=prompt)]}
            final_state = await self.edit_agent.ainvoke(agent_input)
            
            new_svg_elements = [
                message.content 
                for message in final_state.get('messages', []) 
                if isinstance(message, ToolMessage)
            ]
            
            if not new_svg_elements:
                logger.warning("-> Edit agent did not produce any SVG elements.")
                new_svg_elements = re.findall(r'(<path.*/>)', input_svg_code)

            new_svg = f'<svg width="1000" height="1000" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">{"".join(new_svg_elements)}</svg>'
            svg_history.append(new_svg)
            
            last_idea = state['sketches'][-1]['idea'] if state.get('sketches') else SketchDraft(
                design_description="Edited sketch", 
                sketch_description=user_instruction
            )
            
            new_sketch_output = SketchOutput(idea=last_idea, svg_elements=new_svg_elements)
            sketches = state.get('sketches', [])
            sketches.append(new_sketch_output)
            
            png_path = None
            if state.get('project_dir'):
                import time
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                svg_path = os.path.join(state['project_dir'], f"sketch_edited_{timestamp}.svg")
                with open(svg_path, "w", encoding="utf-8") as f:
                    f.write(new_svg)
                png_path = svg_to_png(svg_path)
                logger.info(f"-> Saved edited SVG to {svg_path} and PNG to {png_path}")
            
            return {
                "sketches": sketches,
                "svg_history": svg_history,
                "generated_image_paths": [png_path] if png_path else []
            }
        except Exception as e:
            logger.error(f"Error in edit_node: {e}")
            error_message = AIMessage(content=f"An error occurred during the edit process: {e}")
            state["messages"].append(error_message)
            return Command(goto="wait_for_user_input", )
        
    async def _generate_prompt(self, state: CanvasState):
        logger.info("---CANVAS AGENT: GENERATE PROMPT NODE---")
  
        try:
            prompt_template = self._system_prompts["image_generator_prompt"]
            prompt = prompt_template.format(topic=state['topic'])
            image_contents = []
            if state.get('input_image_paths'):
                for image_path in state['input_image_paths']:
                    with open(image_path, "rb") as image_file:
                        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
                        image_contents.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                        })
            
            structured_llm = self.get_structure_llm(schema=ImagePrompt)
            messages = [
                HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    *image_contents
                ])
            ]
            llm_output = await structured_llm.ainvoke(messages)
            return llm_output.prompt
        except Exception as e:
            logger.error(f"Error in _generate_prompt: {e}")
            # This is a helper, so we re-raise to be caught by the calling node
            raise
    
    async def generate_images_node(self, state: CanvasState, config=None):
        """Generates a single composite image with four unique variations based on the topic."""
        logger.info("---CANVAS AGENT: IMAGE GENERATOR NODE---")
        try:
            if not state.get("topic", ""):
                state['stage'] = ''
                return Command(goto="wait_for_user_input", update={"messages": [AIMessage(content="Please input a topic first.")]})
            
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
        except Exception as e:
            logger.error(f"Error in generate_images_node: {e}")
            error_message = AIMessage(content=f"An error occurred during image generation: {e}")
            return Command(goto="wait_for_user_input", update={"messages": [error_message]})
        
    async def imagine_node(self, state: CanvasState, config=None):
        """Brainstorms sketch ideas and drawing instructions."""
        logger.info("---CANVAS AGENT: IMAGINER NODE---")
        try:
            prompt_template = self._system_prompts["imaginer"]
            
            contents = [{"type": "text", "text": prompt_template.format(
                topic=state['topic'], 
                technique=state.get('technique', ''), 
                requirement=state.get('requirement', ''))}]
            
            if state.get("generated_image_paths"):
                for image_path in state["generated_image_paths"]:
                    try:
                        with open(image_path, "rb") as image_file:
                            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
                            contents.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                            })
                    except Exception as e:
                        logger.error(f"Error encoding image {image_path}: {e}")

            messages = [HumanMessage(content=contents)]
            
            structured_llm = self.get_structure_llm(schema=SketchImaginary)
            llm_output = await structured_llm.ainvoke(messages)
            
            sketch_ideas = llm_output.sketch_imaginary

            if state.get('project_dir'):
                with open(os.path.join(state['project_dir'], "sketch_draft.json"), "w", encoding="utf-8") as f:
                    json.dump(llm_output.model_dump(), f, ensure_ascii=False, indent=4)

                ideas_as_dicts = [idea.model_dump() for idea in sketch_ideas]
                with open(os.path.join(state['project_dir'], "sketch_ideas.json"), "w", encoding="utf-8") as f:
                    json.dump(ideas_as_dicts, f, ensure_ascii=False, indent=4)

            return {"sketch_ideas": sketch_ideas}
        except Exception as e:
            logger.error(f"Error in imagine_node: {e}")
            error_message = AIMessage(content=f"An error occurred during brainstorming: {e}")
            state["messages"].append(error_message)
            return Command(goto="wait_for_user_input")

    async def critique_and_refine_node(self, state: CanvasState, config=None):
        """Critiques and refines the initial sketch ideas."""
        logger.info("---CANVAS AGENT: CRITIQUE AND REFINE NODE---")
        try:
            initial_ideas = state['sketch_ideas']
            if not initial_ideas:
                raise ValueError("No sketch ideas to critique.")
            refined_ideas = []

            critique_llm = self.get_structure_llm(schema=Critique)
            refine_llm = self.get_structure_llm(schema=SketchDraft)

            critique_prompt_template = self._system_prompts["critique_prompt"]
            refine_prompt_template = self._system_prompts["refine_prompt"]

            for i, idea in enumerate(initial_ideas):
                logger.info(f"-> Critiquing idea #{i+1}...")
                
                critique_input = f"**Design Description:**\n{idea.design_description}\n\n**Sketch Description:**\n{idea.sketch_description}"
                critique_messages = [
                    SystemMessage(content=critique_prompt_template),
                    HumanMessage(content=critique_input)
                ]
                critique_output = await critique_llm.ainvoke(critique_messages)
                critique_text = critique_output.critique
                logger.info(f"-> Critique #{i+1}: {critique_text}")

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

            if state.get('project_dir'):
                refined_ideas_as_dicts = [idea.model_dump() for idea in refined_ideas]
                with open(os.path.join(state['project_dir'], "refined_sketch_ideas.json"), "w", encoding="utf-8") as f:
                    json.dump(refined_ideas_as_dicts, f, ensure_ascii=False, indent=4)

            return {"sketch_ideas": refined_ideas}
        except Exception as e:
            logger.error(f"Error in critique_and_refine_node: {e}")
            error_message = AIMessage(content=f"An error occurred during critique and refinement: {e}")
            state["messages"].append(error_message)
            return Command(goto="wait_for_user_input")

    async def refine_sketch_node(self, state: CanvasState, config=None):
        """Uses a planning agent with a calculator to generate a detailed drawing prompt."""
        logger.info("---CANVAS AGENT: PLANNER NODE---")
        try:
            refiner_input_template = self._system_prompts["sketch_refiner_input"]
            
            sketch_ideas = state['sketch_ideas']
            if not sketch_ideas:
                raise ValueError("No sketch ideas to refine.")
            
            for idea in sketch_ideas:
                logger.info(f"-> Planning sketch with math: {idea.design_description}")
                
                planner_input = refiner_input_template.format(
                    sketch_description=idea.sketch_description,
                ) + self._system_prompts["sketch_planner"]
                
                messages = [HumanMessage(content=planner_input)]
                agent_input = {"messages": messages}
                
                final_state = await self.planner_agent.ainvoke(agent_input)

                drawing_prompt = final_state['messages'][-1].content
                
                idea.drawing_prompt = drawing_prompt
                logger.info(f"-> Generated Drawing Prompt: {idea.drawing_prompt}")
                ideas_as_dicts = [idea.model_dump() for idea in sketch_ideas]
                with open(os.path.join(state['project_dir'], "sketch_idea_with_prompt.json"), "w", encoding="utf-8") as f:
                    json.dump(ideas_as_dicts, f, ensure_ascii=False, indent=4)
            return {"sketch_ideas": sketch_ideas}
        except Exception as e:
            logger.error(f"Error in refine_sketch_node: {e}")
            error_message = AIMessage(content=f"An error occurred during sketch planning: {e}")
            state["messages"].append(error_message)
            return Command(goto="wait_for_user_input", )

    async def draw_sketches_node(self, state: CanvasState, config=None):
        """Uses the drawer agent to execute the drawing prompts for each idea."""
        logger.info("---CANVAS AGENT: DRAWER NODE---")
        try:
            sketches: List[SketchOutput] = []

            for i, idea in enumerate(state['sketch_ideas']):
                logger.info(f"-> Drawing sketch {i+1}...")
                
                if not idea.drawing_prompt:
                    logger.warning(f"-> Skipping sketch {i+1} as it has no drawing prompt.")
                    continue

                messages = [HumanMessage(content=idea.drawing_prompt)]
                agent_input = {"messages": messages}
                
                final_state = await self.drawer_agent.ainvoke(agent_input)
                show_messages(final_state.get("messages", []))
                
                svg_elements = [
                    message.content 
                    for message in final_state['messages'] 
                    if isinstance(message, ToolMessage)
                ]

                final_svg = f'<svg width="1000" height="1000" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">{"".join(svg_elements)}</svg>'
                
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
        except Exception as e:
            logger.error(f"Error in draw_sketches_node: {e}")
            error_message = AIMessage(content=f"An error occurred during drawing: {e}")
            state["messages"].append(error_message)
            return Command(goto="wait_for_user_input", )

    async def draw_sketches_only_node(self, state: CanvasState, config=None):
        """Uses the drawer agent to execute the drawing prompts for each idea."""
        logger.info("---CANVAS AGENT: DRAW ONLY NODE---")
        try:
            draw_prompt = state['draw_prompt']
            logger.info(f"-> Draw Prompt: {draw_prompt}")
            messages = [HumanMessage(content=draw_prompt)]
            agent_input = {"messages": messages}
            
            final_state = await self.drawer_agent.ainvoke(agent_input)

            svg_elements = [
                    message.content 
                    for message in final_state['messages'] 
                    if isinstance(message, ToolMessage)
                ]
            
            final_svg = f'<svg width="1000" height="1000" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">{"".join(svg_elements)}</svg>'
            output_dir = state.get('project_dir', 'output/test')
            os.makedirs(output_dir, exist_ok=True)
            if output_dir:
                import time
                timestamp = time.strftime("%Y%m%d_%H%M%S")  
                output_name = f"sketch_{timestamp}"
                svg_path = os.path.join(output_dir, f"{output_name}.svg")
                with open(svg_path, "w", encoding="utf-8") as f:
                    f.write(final_svg)

                png_path = svg_to_png(svg_path)
                
            return {"generated_image_paths": [png_path]}
        except Exception as e:
            logger.error(f"Error in draw_sketches_only_node: {e}")
            error_message = AIMessage(content=f"An error occurred during the 'draw only' process: {e}")
            state["messages"].append(error_message)
            return Command(goto="wait_for_user_input", )

    async def describe_only_node(self, state: CanvasState, config=None):
        logger.info("---CANVAS AGENT: DESCRIBE ONLY NODE---")
        try:
            image_contents = []
            if state.get("generated_image_paths"):
                image_contents.append( {"type": "text", "text": "Please describe the image in svg desciption."})
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
                {"type": "text", "text": state['topic']},
                    *image_contents
                ])
            ]
            
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
        except Exception as e:
            logger.error(f"Error in describe_only_node: {e}")
            error_message = AIMessage(content=f"An error occurred during the 'describe only' process: {e}")
            state["messages"].append(error_message)
            return Command(goto="wait_for_user_input", )
        
    # def router_node(self, state: CanvasState, config=None):
    #     stage = state.get('stage', 'generate')
    #     logger.info(f"-> Router Node: {stage}")
        
    #     if stage == 'draw_only':
    #         return Command(goto="draw_sketches_only_node",)
    #     elif stage == 'describe_only':
    #         return Command(goto="describe_only_node")
    #     elif stage == 'chat':
    #         return Command(goto="chat_node")
    #     elif stage == 'edit':
    #         return Command(goto="edit_node")
    #     else: # generate
    #         return Command(goto="generate_images_node")
 
    
    def build_graph(self):
        graph = StateGraph(CanvasState)
        # graph.add_node("imagine_node", self.imagine_node)
        # graph.add_node("draw_sketches_node", self.draw_sketches_node)
        # graph.add_node("draw_sketches_only_node", self.draw_sketches_only_node)
        # graph.add_node("describe_only_node", self.describe_only_node)
        # graph.add_node("refine_sketch_node", self.refine_sketch_node)
        # graph.add_node("generate_images_node", self.generate_images_node)
        # graph.add_node("critique_and_refine_node", self.critique_and_refine_node)
        graph.add_node("wait_for_user_input", self.wait_for_user_input)
        # graph.add_node("pick_path_node", self.pick_path_node)
        graph.add_node("chat_node", self.chat_node)
        # graph.add_node("edit_node", self.edit_node)
        # graph.add_node("router_node", self.router_node)
        graph.add_node("init_context_node", self._init_context_node)
        
        graph.add_edge(START, "init_context_node")
        graph.add_edge("init_context_node", "wait_for_user_input")
        ## full generate loop
        # graph.add_edge("generate_images_node", "imagine_node")
        # graph.add_edge("imagine_node", "critique_and_refine_node")
        # graph.add_edge("critique_and_refine_node", "refine_sketch_node")
        # graph.add_edge("refine_sketch_node", "draw_sketches_node")
        # graph.add_edge("draw_sketches_node", "wait_for_user_input")
        # graph.add_edge("edit_node", "wait_for_user_input")
        # graph.add_edge("draw_sketches_only_node", "wait_for_user_input")
        # graph.add_edge("describe_only_node", "wait_for_user_input")
        graph.add_edge("chat_node", "wait_for_user_input")
        # graph.add_edge("pick_path_node", "wait_for_user_input")
        return graph
    
    


    
    