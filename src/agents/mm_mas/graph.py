import logging
import os
from functools import partial
from typing import Literal
import json

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

from src.agents.base.mas_graph import MultiAgentBase
from src.config.manager import MultiAgentConfig
from .schema import LessonMASState, SectionGenerationState, LessonPlan, StateSection
from .nodes import lesson_supervisor_node, section_supervisor_node, route_after_lesson_supervisor

logger = logging.getLogger(__name__)

# --- Graph Helper Functions ---

def compile_section_content(state: SectionGenerationState) -> dict:
    """A node that compiles the generated content from a section into a single markdown string."""
    logger.info("Compiling content for the section.")
    
    compiled_content = f"### {state.get('title', '无标题')}\n\n"
    
    if state.get('ppt_content'):
        compiled_content += "#### 演示文稿内容:\n"
        # Assuming ppt_content is a dict that can be nicely formatted
        compiled_content += f"```json\n{json.dumps(state['ppt_content'], indent=2, ensure_ascii=False)}\n```\n\n"

    if state.get('image_urls'):
        compiled_content += "#### 相关图片:\n"
        for url in state['image_urls']:
            compiled_content += f"- ![Image]({url})\n"
        compiled_content += "\n"

    if state.get('experiment_content'):
        exp = state['experiment_content']
        compiled_content += f"#### 互动实验:\n"
        compiled_content += f"- **标题:** {exp.get('title', 'N/A')}\n"
        compiled_content += f"- **链接:** [{exp.get('url', '#')}]({exp.get('url', '#')})\n\n"

    if state.get('flashcard_content'):
        compiled_content += "#### 学习闪卡:\n"
        for card in state['flashcard_content']:
            compiled_content += f"- **正面:** {card.get('front', '')}\n"
            compiled_content += f"- **背面:** {card.get('back', '')}\n\n"
            
    return {"compiled_content": compiled_content.strip()}


# --- Main MAS Class ---

class LessonMAS(MultiAgentBase):
    name = "lesson_mas"
    description = "A multi-agent system for generating 45-minute lessons."

    def __init__(self, multi_agent_config: MultiAgentConfig):
        super().__init__(multi_agent_config)
        # self.multi_agent_config = multi_agent_config
        self._load_system_prompts()

    def _load_system_prompts(self):
        prompts_dir = self._multi_agent_config.agent_specific_config.get("prompts_dir")
        if not prompts_dir:
            raise ValueError("prompts_dir is not configured for LessonMAS.")
        
        self.prompts = {}
        agent_names = [
            "lesson_supervisor", "section_supervisor", "ppt_agent", 
            "flashcard_agent", "experiment_agent", "image_agent"
        ]
        for name in agent_names:
            try:
                with open(os.path.join(prompts_dir, f"{name}_system_prompt.txt"), "r") as f:
                    self.prompts[name] = f.read()
            except FileNotFoundError:
                logger.warning(f"Prompt file for {name} not found. Skipping.")
                self.prompts[name] = ""
                

    def build_section_graph(self) -> StateGraph:
        """Builds the inner supervisor graph for generating content for a single section."""
        
        worker_agents = self._multi_agent_config.agent_specific_config.get("section_mas")
        members = ["section_supervisor"] + worker_agents
        prompts_dir = self._multi_agent_config.agent_specific_config.get("prompts_dir")
        
        # Create React agents or all workers
        agent_nodes = []
        for agent_name in worker_agents:
            agent = create_react_agent(
                name=agent_name,
                model=self._agent_llm_registry[agent_name],
                tools=self._agent_tools_registry.get(agent_name, []),
                prompt=open(os.path.join(prompts_dir, f"{agent_name}_system_prompt.txt"), "r").read()
            )
            agent_nodes.append(agent)

        # Create the supervisor
        supervisor_graph = create_supervisor(
            model=self._agent_llm_registry["section_supervisor"],
            agents=agent_nodes,
            supervisor_name="section_supervisor",
            state_schema=SectionGenerationState,
            # prompt=self.prompts["section_supervisor"],
            prompt=open(os.path.join(prompts_dir, "section_supervisor_system_prompt.txt"), "r").read(),
            task_prompt="""
# 章节标题
{title}

# 章节描述
{description}

# 预计时长
{duration} minutes
""",
            # Define entry and exit points for more complex flows
            entry_point="section_supervisor",
        )
        
        # Add a final node to compile the results into a single string
        supervisor_graph.add_node("compile_results", compile_section_content)
        supervisor_graph.add_edge("section_supervisor", "compile_results")
        supervisor_graph.add_edge("compile_results", END)

        return supervisor_graph.compile()


    def build_graph(self) -> StateGraph:
        """Builds the main lesson generation graph which orchestrates the section sub-graphs."""
        prompts_dir = self._multi_agent_config.agent_specific_config.get("prompts_dir")
        # 1. Build the inner graph for section content generation
        section_graph = self.build_section_graph()
        
        # 2. Create a structured LLM for the top-level lesson supervisor
        lesson_supervisor_llm = self._agent_llm_registry["lesson_supervisor"]
        structured_llm = lesson_supervisor_llm.with_structured_output(LessonPlan)
        
        # 3. Define nodes for the main graph
        lesson_supervisor_node_with_llm = partial(
            lesson_supervisor_node,
            structured_llm=structured_llm,
            model_name=getattr(lesson_supervisor_llm, "model", "unknown_model"),
            system_prompt=open(os.path.join(prompts_dir, "lesson_supervisor_system_prompt.txt"), "r").read(),
        )
        
        section_supervisor_node_with_graph = partial(
            section_supervisor_node,
            section_supervisor_graph=section_graph,
            system_prompt=open(os.path.join(prompts_dir, "section_supervisor_system_prompt.txt"), "r").read(),
        )

        # 4. Construct the main graph
        builder = StateGraph(LessonMASState)

        builder.add_node("lesson_supervisor", lesson_supervisor_node_with_llm)
        builder.add_node("section_supervisor_node", section_supervisor_node_with_graph)
        builder.add_node("finish", self.finish_node)

        builder.add_edge(START, "lesson_supervisor")
        
        # After a section is completed, control returns to the lesson supervisor for re-evaluation
        builder.add_edge("section_supervisor_node", "lesson_supervisor")
        
        # The lesson supervisor is now the central router for the main loop
        builder.add_conditional_edges(
            "lesson_supervisor",
            route_after_lesson_supervisor,
            {
                "section_supervisor_node": "section_supervisor_node",
                "finish": "finish",
            },
        )

        builder.add_edge("finish", END)

        return builder
    


    def finish_node(self, state: LessonMASState) -> dict:
        """Compiles the final lesson content from all sections."""
        logger.info("Finishing lesson generation.")
        
        section_plan = state.get("section_plan", [])
        final_content = f"# 课程：{state.get('topic', 'N/A')}\n\n"
        
        for i, section in enumerate(section_plan):
            final_content += f"## 章节 {i+1}: {section.get('title', '无标题')}\n\n"
            final_content += section.get('content', '本章节无内容。')
            final_content += "\n\n"
            
        return {"final_content": final_content.strip()}