"""
Wang Agent - Iterative SVG generation with weighted components and reflection.
"""

import logging
import os
import base64
import json
import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import asdict
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from src.agents.base import BaseAgent
from src.config.manager import AgentConfig
from src.infra.tools.image_tools import generate_image
from src.infra.tools.draw_canvas_tools import draw_agent_with_tool, DrawCanvasAgentTools
from src.infra.tools.edit_canvas_tools import edit_agent_with_tool
from src.infra.tools.math_tools import (
    MathAgentTools,
    auxiliary_line_circle_intersection,
    auxiliary_line_line_intersection,
)
from .enhanced_svg_tools import ENHANCED_SVG_TOOLS
from src.agents.canvas_agent.utils import svg_to_png, convert_svg_to_png_base64
from src.agents.wang_agent.svg_metrics_tools import (
    calculate_metrics_from_base64,
    analyze_svg_differences,
    format_metrics_for_llm,
    format_differences_for_llm
)
from src.agents.wang_agent.svg_parameter_tools import SVG_PARAMETER_TOOLS
from src.agents.wang_agent.png_structure_tools import PNG_STRUCTURE_TOOLS
from .schema import (
    WangState,
    DesignPrompt,
    Instruction,
    WeightedComponent,
    WeightedComponentsList,
    StageResult,
    ReflectionResult,
)

logger = logging.getLogger(__name__)


class WangAgent(BaseAgent):
    """
    Wang Agent: Iterative SVG generation agent with weighted step-by-step generation.
    
    Workflow:
    1. THINK: Convert user intention to design prompt
    2. Generate initial image and image prompt
    3. Analyze image into weighted components
    4. Generate SVG step by step (highest weight first)
    5. Reflect on each step
    6. If quality not met, rollback and retry
    7. Continue until all components meet criteria
    """

    name = "wang_agent"
    description = "Iterative SVG generation agent with reflection and weighted components"

    def __init__(self, agent_config: AgentConfig):
        super().__init__(agent_config)
        self._load_system_prompt()
        
        # Initialize drawer agent for SVG generation with enhanced Bezier curve tools
        # Use MathAgentTools which contains all available math tools
        planner_tools = MathAgentTools + [
            auxiliary_line_circle_intersection,
            auxiliary_line_line_intersection,
        ]
        
        # Combine all tools: base tools + math tools + enhanced Bezier tools + parameter tools + PNG structure tools
        all_drawer_tools = (
            DrawCanvasAgentTools +  # Base drawing tools
            planner_tools +  # Math tools for calculations
            ENHANCED_SVG_TOOLS +  # Enhanced Bezier curve tools
            SVG_PARAMETER_TOOLS +  # Parameter extraction and adjustment tools
            PNG_STRUCTURE_TOOLS +  # PNG structure extraction tools
            self._tools  # Agent-specific tools
        )
        
        # Load drawer prompt from canvas_agent if available, otherwise use default
        try:
            canvas_prompt_dir = Path(__file__).parent.parent / "canvas_agent" / "prompt"
            drawer_system_prompt = (canvas_prompt_dir / "drawer_prompt.txt").read_text(encoding="utf-8")
            # Enhance the prompt with Bezier curve guidance
            drawer_system_prompt += "\n\n## 重要：优先使用贝塞尔曲线\n"
            drawer_system_prompt += "优先使用贝塞尔曲线工具创建平滑、自然的形状：\n"
            drawer_system_prompt += "- draw_quadratic_bezier: 二次贝塞尔曲线（简单弧线）\n"
            drawer_system_prompt += "- draw_bezier_curve: 三次贝塞尔曲线（复杂曲线）\n"
            drawer_system_prompt += "- draw_smooth_cubic_bezier: 平滑连接的三次曲线\n"
            drawer_system_prompt += "- draw_bezier_path_from_points: 从点生成平滑路径\n"
            drawer_system_prompt += "- draw_organic_shape: 有机形状（自然形状）\n"
            drawer_system_prompt += "- draw_smooth_curve_through_points: 精确通过点的平滑曲线\n"
        except:
            drawer_system_prompt = """You are an expert SVG drawing agent specializing in Bezier curves.
            
Priority: Use Bezier curves for smooth, natural shapes:
- Use draw_quadratic_bezier for simple arcs
- Use draw_bezier_curve for complex curves  
- Use draw_smooth_cubic_bezier for smooth connections
- Use draw_bezier_path_from_points to create smooth paths from point lists
- Use draw_organic_shape for natural, organic forms
- Use draw_smooth_curve_through_points for curves that must pass through specific points

Always prefer Bezier curves over straight lines for curved shapes."""
        
        self.drawer_agent = create_react_agent(
            model=self.init_llm(0.2),
            tools=all_drawer_tools,
            prompt=drawer_system_prompt,
        )

    def init_llm(self, temperature: float = 0.7):
        """Initialize LLM with custom temperature."""
        config = self._model_config
        config.temperature = temperature
        return init_chat_model(**asdict(config))

    def _load_system_prompt(self):
        """Load system prompts."""
        prompt_dir = Path(__file__).parent / "prompt"
        self._system_prompts = {
            "system": (prompt_dir / "system_prompt.txt").read_text(encoding="utf-8"),
            "think": (prompt_dir / "think_prompt.txt").read_text(encoding="utf-8"),
            "weight_analysis": (prompt_dir / "weight_analysis_prompt.txt").read_text(encoding="utf-8"),
            "svg_generation": (prompt_dir / "svg_generation_prompt.txt").read_text(encoding="utf-8"),
            "reflect": (prompt_dir / "reflect_prompt.txt").read_text(encoding="utf-8"),
        }

    def get_structure_llm(self, schema):
        """Get LLM with structured output."""
        return self._llm.with_structured_output(schema)
    
    def _clean_and_validate_svg(self, svg_code: str) -> str:
        """Clean and validate SVG code to ensure it's well-formed."""
        import re
        from xml.etree import ElementTree as ET
        
        try:
            # Step 1: Fix escaped quotes (\" -> ")
            svg_code = svg_code.replace('\\"', '"')
            svg_code = svg_code.replace("\\'", "'")
            
            # Step 2: Remove namespace prefixes (ns0: -> empty)
            svg_code = re.sub(r'\s*xmlns:ns0="[^"]*"', '', svg_code)
            svg_code = re.sub(r'<ns0:(\w+)', r'<\1', svg_code)
            svg_code = re.sub(r'</ns0:(\w+)', r'</\1', svg_code)
            svg_code = svg_code.replace('ns0:', '')
            
            # Step 3: Extract only the first complete SVG block (remove anything after </svg>)
            svg_start = svg_code.find('<svg')
            if svg_start == -1:
                logger.warning("-> 未找到<svg>标签，返回空SVG")
                return '<svg width="1000" height="1000" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000"></svg>'
            
            # Find the first </svg> tag after <svg>
            svg_end = svg_code.find('</svg>', svg_start)
            if svg_end == -1:
                # No closing tag, try to add one
                svg_code = svg_code[svg_start:].rstrip() + '</svg>'
            else:
                # Extract only the first complete SVG block
                svg_code = svg_code[svg_start:svg_end + 6]  # +6 for '</svg>'
            
            # Step 4: Remove comments and extra whitespace between elements
            # Remove XML comments
            svg_code = re.sub(r'<!--.*?-->', '', svg_code, flags=re.DOTALL)
            
            # Step 4: Try to parse and re-serialize to ensure well-formedness
            try:
                root = ET.fromstring(svg_code)
                # If parsing succeeds, the SVG is well-formed
                # Reconstruct the SVG string
                svg_attrs = root.attrib
                width = svg_attrs.get('width', '1000')
                height = svg_attrs.get('height', '1000')
                viewBox = svg_attrs.get('viewBox', '0 0 1000 1000')
                xmlns = svg_attrs.get('xmlns', 'http://www.w3.org/2000/svg')
                
                # Extract all child elements as strings
                inner_content = ''.join([ET.tostring(elem, encoding='unicode') for elem in root])
                
                # Reconstruct clean SVG
                clean_svg = f'<svg width="{width}" height="{height}" xmlns="{xmlns}" viewBox="{viewBox}">{inner_content}</svg>'
                return clean_svg
            except ET.ParseError as parse_err:
                # If XML parsing fails, try to fix common issues
                logger.warning(f"-> SVG XML解析失败: {parse_err}，尝试修复...")
                
                # Try to extract just the first valid SVG block
                svg_match = re.search(r'(<svg[^>]*>.*?</svg>)', svg_code, re.DOTALL | re.IGNORECASE)
                if svg_match:
                    cleaned = svg_match.group(1)
                    # Try parsing again
                    try:
                        ET.fromstring(cleaned)
                        return cleaned
                    except:
                        pass
                
                # If still fails, try to build a minimal valid SVG from elements
                # Extract all SVG elements (path, circle, rect, etc.)
                elements = []
                for tag in ['path', 'circle', 'rect', 'ellipse', 'line', 'polygon', 'polyline', 'g', 'text']:
                    pattern = f'<{tag}[^>]*(?:/>|>.*?</{tag}>)'
                    matches = re.findall(pattern, svg_code, re.IGNORECASE | re.DOTALL)
                    elements.extend(matches)
                
                if elements:
                    # Build a clean SVG with extracted elements
                    clean_svg = f'<svg width="1000" height="1000" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">{"".join(elements)}</svg>'
                    try:
                        ET.fromstring(clean_svg)
                        return clean_svg
                    except:
                        pass
                
                # Last resort: return minimal valid SVG
                logger.warning("-> SVG验证失败，返回空SVG")
                return '<svg width="1000" height="1000" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000"></svg>'
        except Exception as e:
            logger.warning(f"-> SVG清理过程出错: {e}，尝试提取有效部分")
            # Fallback: extract first valid SVG block
            svg_match = re.search(r'(<svg[^>]*>.*?</svg>)', svg_code, re.DOTALL | re.IGNORECASE)
            if svg_match:
                return svg_match.group(1)
            # If no valid SVG found, return minimal
            return '<svg width="1000" height="1000" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000"></svg>'

    async def think_node(self, state: WangState, config=None):
        """
        THINK node: 使用LLM将用户意图转换为设计提示和图像生成提示
        
        流程：
        1. 接收用户的复杂意图（可能包含用户提供的图像）
        2. 使用LLM（GPT等模型）分析意图，生成：
           - design_prompt: 详细的设计描述
           - image_prompt: 用于图像生成的提示词
           - instruction: 包含intention和criteria的指令
        3. 这些输出将用于后续的图像生成和SVG生成
        """
        logger.info("---WANG AGENT: THINK NODE---")
        try:
            user_intention = state.get("user_intention", "")
            user_image_path = state.get("user_image_path")
            
            if not user_intention:
                error_msg = AIMessage(content="请提供用户意图")
                return {"messages": [error_msg]}

            logger.info(f"-> 用户意图: {user_intention}")
            logger.info("-> 使用LLM（GPT等模型）生成设计提示和图像提示...")

            # Prepare prompt for LLM
            think_prompt = self._system_prompts["think"].format(
                user_intention=user_intention
            )

            # Prepare messages with optional image
            contents = [{"type": "text", "text": think_prompt}]
            
            if user_image_path and os.path.exists(user_image_path):
                logger.info(f"-> 检测到用户提供的图像，将结合图像和文本生成提示")
                with open(user_image_path, "rb") as f:
                    encoded_image = base64.b64encode(f.read()).decode("utf-8")
                    contents.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                    })

            messages = [HumanMessage(content=contents)]
            
            # 调用LLM（使用配置的模型，可能是GPT、Gemini等）
            # 获取结构化输出
            structured_llm = self.get_structure_llm(DesignPrompt)
            result = await structured_llm.ainvoke(messages)
            
            logger.info(f"-> ✓ LLM已生成设计提示")
            logger.info(f"-> Design Prompt: {result.design_prompt[:200]}...")
            logger.info(f"-> Image Prompt (用于图像生成API): {result.image_prompt[:200]}...")
            logger.info(f"-> Intention: {result.instruction.intention}")
            logger.info(f"-> Criteria: {result.instruction.criteria}")

            return {
                "design_prompt": result.design_prompt,
                "image_prompt": result.image_prompt,  # 这个将用于调用图像生成API
                "instruction": result.instruction,
                "messages": [AIMessage(content=f"已使用LLM生成设计提示和图像提示。意图：{result.instruction.intention}")],
            }
        except Exception as e:
            logger.error(f"Error in think_node: {e}")
            error_msg = AIMessage(content=f"思考节点出错: {e}")
            return {"messages": [error_msg]}

    async def generate_image_node(self, state: WangState, config=None):
        """
        Generate initial image from image prompt.
        
        流程：
        1. 如果用户提供了图像，直接使用
        2. 否则，使用THINK节点生成的image_prompt调用图像生成API
        3. 保存生成的图像
        """
        logger.info("---WANG AGENT: GENERATE IMAGE NODE---")
        try:
            image_prompt = state.get("image_prompt")
            user_image_path = state.get("user_image_path")
            design_prompt = state.get("design_prompt", "")
            
            # If user provided image, use it
            if user_image_path and os.path.exists(user_image_path):
                logger.info(f"-> Using user-provided image: {user_image_path}")
                return {"initial_image_path": user_image_path}
            
            # Otherwise generate image using the prompt from THINK node
            if not image_prompt:
                error_msg = AIMessage(content="缺少图像生成提示，请先运行THINK节点")
                return {"messages": [error_msg]}

            project_dir = state.get("project_dir", "output/wang_agent")
            os.makedirs(project_dir, exist_ok=True)

            logger.info(f"-> 使用LLM生成的图像提示生成图像")
            logger.info(f"-> Image Prompt (from LLM): {image_prompt[:100]}...")
            logger.info(f"-> Design Prompt: {design_prompt[:100]}...")
            
            # 调用图像生成API（使用项目中已有的generate_image函数）
            # 这个函数内部调用Google GenAI的imagen模型
            logger.info("-> 调用图像生成API...")
            image_pil = await asyncio.to_thread(
                generate_image,
                prompt=image_prompt,
                aspect_ratio="1:1"
            )
            
            # Save image
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(project_dir, f"initial_image_{timestamp}.png")
            image_pil.save(image_path)
            logger.info(f"-> ✓ 图像已生成并保存: {image_path}")

            return {
                "initial_image_path": image_path,
                "messages": [AIMessage(content=f"已通过API生成初始图像: {image_path}")],
            }
        except Exception as e:
            logger.error(f"Error in generate_image_node: {e}")
            error_msg = AIMessage(content=f"图像生成出错: {e}")
            return {"messages": [error_msg]}

    async def analyze_weights_node(self, state: WangState, config=None):
        """Analyze image into weighted components."""
        logger.info("---WANG AGENT: ANALYZE WEIGHTS NODE---")
        try:
            initial_image_path = state.get("initial_image_path")
            design_prompt = state.get("design_prompt", "")
            
            if not initial_image_path or not os.path.exists(initial_image_path):
                error_msg = AIMessage(content="缺少初始图像")
                return {"messages": [error_msg]}

            # Read and encode image
            with open(initial_image_path, "rb") as f:
                encoded_image = base64.b64encode(f.read()).decode("utf-8")

            # Prepare prompt
            weight_prompt = self._system_prompts["weight_analysis"].format(
                image_description="参考图像",
                design_prompt=design_prompt
            )

            contents = [
                {"type": "text", "text": weight_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                }
            ]

            messages = [HumanMessage(content=contents)]
            
            # Get structured output - use wrapper model for List
            structured_llm = self.get_structure_llm(WeightedComponentsList)
            result = await structured_llm.ainvoke(messages)
            components = result.components
            
            # Sort by weight (descending)
            components.sort(key=lambda x: x.weight, reverse=True)
            
            # Extract structure information for each component
            logger.info(f"-> Analyzed {len(components)} components")
            design_prompt = state.get("design_prompt", "")
            
            for i, comp in enumerate(components):
                logger.info(f"  Component {i+1}: weight={comp.weight:.2f}, desc={comp.description[:50]}...")
                
                # Extract structure information from PNG for this component
                try:
                    from src.agents.wang_agent.png_structure_tools import analyze_component_structure
                    # Call the tool's underlying function
                    structure_info = analyze_component_structure.invoke({
                        "image_path": initial_image_path,
                        "component_description": comp.description,
                        "design_prompt": design_prompt,
                    })
                    
                    if "error" not in structure_info:
                        comp.structure_info = structure_info
                        logger.info(f"    -> 提取了结构信息: {structure_info.get('visual_features', {}).get('keypoint_count', 0)} 个关键点, {structure_info.get('visual_features', {}).get('contour_count', 0)} 个轮廓")
                    else:
                        logger.warning(f"    -> 结构信息提取失败: {structure_info.get('error')}")
                except Exception as e:
                    logger.warning(f"    -> 结构信息提取出错: {e}")

            return {
                "weighted_components": components,
                "messages": [AIMessage(content=f"已分析出{len(components)}个权重组件，并提取了结构信息")],
            }
        except Exception as e:
            logger.error(f"Error in analyze_weights_node: {e}", exc_info=True)
            # 如果权重分析失败，创建多个基础组件以确保分步生成
            # 这样可以避免单组件问题，即使分析失败也能分步生成
            logger.warning("-> 权重分析失败，创建基础组件以继续流程")
            
            # 根据设计提示和图像，创建多个基础组件
            # 对于树这样的对象，至少分解为：树干、树枝、树叶
            # 这是一个通用的fallback策略，适用于大多数图像
            fallback_components = [
                WeightedComponent(
                    weight=0.9,
                    description="主要结构部分（如树干、主体框架等）"
                ),
                WeightedComponent(
                    weight=0.7,
                    description="次要结构部分（如树枝、分支等）"
                ),
                WeightedComponent(
                    weight=0.5,
                    description="细节和装饰部分（如树叶、细节元素等）"
                ),
            ]
            
            # 尝试从design_prompt中提取关键词，优化组件描述
            design_prompt = state.get("design_prompt", "")
            if "树" in design_prompt or "tree" in design_prompt.lower():
                fallback_components = [
                    WeightedComponent(
                        weight=0.9,
                        description="树干部分：包括主树干和主要分支结构"
                    ),
                    WeightedComponent(
                        weight=0.7,
                        description="树冠部分：包括所有树叶和树冠的整体形状"
                    ),
                ]
            elif "人" in design_prompt or "person" in design_prompt.lower() or "人物" in design_prompt:
                fallback_components = [
                    WeightedComponent(
                        weight=0.9,
                        description="主体部分：包括头部、躯干和主要身体结构"
                    ),
                    WeightedComponent(
                        weight=0.7,
                        description="细节部分：包括四肢、面部特征和装饰元素"
                    ),
                ]
            # 可以继续添加其他常见对象的fallback策略
            
            logger.info(f"-> 创建了 {len(fallback_components)} 个fallback组件")
            return {
                "weighted_components": fallback_components,
                "messages": [AIMessage(content=f"权重分析出错，使用fallback组件继续: {e}")],
            }

    async def generate_svg_stage_node(self, state: WangState, config=None):
        """Generate SVG for current stage (highest weight component)."""
        logger.info("---WANG AGENT: GENERATE SVG STAGE NODE---")
        try:
            components = state.get("weighted_components", [])
            current_stage = state.get("current_stage", 0)
            stages = state.get("stages", [])
            last_reflection = state.get("last_reflection")
            
            # Handle rollback: restore SVG from previous successful stage
            update_fine_tune_attempts = False
            if last_reflection and not last_reflection.passed and current_stage > 0:
                # Rollback: use SVG from previous successful stage
                prev_stage_idx = current_stage - 1
                if prev_stage_idx < len(stages) and stages[prev_stage_idx].passed_reflection:
                    current_svg = stages[prev_stage_idx].svg_code
                    logger.info(f"-> Rollback: 回退到阶段 {prev_stage_idx + 1} 的成功SVG状态")
                    update_fine_tune_attempts = True
                else:
                    # If no previous successful stage, find the last successful one
                    for i in range(len(stages) - 1, -1, -1):
                        if stages[i].passed_reflection:
                            current_svg = stages[i].svg_code
                            logger.info(f"-> Rollback: 回退到阶段 {stages[i].stage_number} 的成功SVG状态")
                            update_fine_tune_attempts = True
                            break
                    else:
                        # No successful stage found, use initial empty SVG
                        current_svg = '<svg width="1000" height="1000" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000"></svg>'
                        logger.warning("-> Rollback: 没有找到成功的阶段，使用空SVG")
                        update_fine_tune_attempts = True
            else:
                # Normal flow: use current SVG
                current_svg = state.get("current_svg", '<svg width="1000" height="1000" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000"></svg>')
            
            design_prompt = state.get("design_prompt", "")
            initial_image_path = state.get("initial_image_path")
            
            # Check if we have components
            if not components:
                logger.warning("-> 没有权重组件，无法生成SVG")
                return {
                    "is_complete": True,
                    "messages": [AIMessage(content="没有权重组件，无法生成SVG。请先运行权重分析节点。")],
                }
            
            if current_stage >= len(components):
                logger.info("-> 所有组件已生成完成")
                return {
                    "is_complete": True,
                    "messages": [AIMessage(content="所有组件已生成完成")],
                }

            # Get current component
            component = components[current_stage]
            stage_number = current_stage + 1

            logger.info(f"-> Generating stage {stage_number}: {component.description[:50]}...")

            # Prepare image for context
            image_description = "参考图像"
            if initial_image_path and os.path.exists(initial_image_path):
                with open(initial_image_path, "rb") as f:
                    encoded_image = base64.b64encode(f.read()).decode("utf-8")
                    image_url = f"data:image/png;base64,{encoded_image}"
            else:
                image_url = None

            # Prepare prompt with reflection feedback if available
            reflection_context = ""
            if last_reflection and not last_reflection.passed:
                reflection_context = f"\n\n## 重要：上一轮反思反馈\n"
                reflection_context += f"上一轮生成未通过反思，反馈如下：\n"
                reflection_context += f"- 问题：{last_reflection.feedback}\n"
                if last_reflection.suggestions:
                    reflection_context += f"- 改进建议：\n"
                    for suggestion in last_reflection.suggestions[:3]:  # 最多3条建议
                        reflection_context += f"  * {suggestion}\n"
                reflection_context += f"\n请根据以上反馈，纠正错误，重新生成正确的SVG。\n"
            
            # Prepare structure information context
            structure_context = ""
            if component.structure_info:
                structure_info = component.structure_info
                visual_features = structure_info.get("visual_features", {})
                structure_info_data = structure_info.get("structure_info", {})
                
                structure_context = "\n\n## PNG结构信息（重要：用于指导SVG生成）\n"
                structure_context += f"**组件结构特征：**\n"
                
                if visual_features.get("keypoint_count", 0) > 0:
                    structure_context += f"- 关键点数量: {visual_features['keypoint_count']} 个\n"
                    keypoints = structure_info_data.get("keypoints", [])
                    if keypoints:
                        # Show first 5 keypoints as examples
                        sample_keypoints = keypoints[:5]
                        structure_context += f"- 关键点示例: {sample_keypoints}\n"
                
                if visual_features.get("contour_count", 0) > 0:
                    structure_context += f"- 轮廓数量: {visual_features['contour_count']} 个\n"
                    contours = structure_info_data.get("contours", [])
                    if contours:
                        # Show first contour as example
                        first_contour = contours[0] if contours else []
                        if first_contour:
                            structure_context += f"- 主要轮廓点数: {len(first_contour)} 个\n"
                            structure_context += f"- 主要轮廓点示例（前5个）: {first_contour[:5]}\n"
                
                if visual_features.get("dominant_colors"):
                    structure_context += f"- 主要颜色:\n"
                    for color_info in visual_features["dominant_colors"][:3]:
                        rgb = color_info["rgb"]
                        pct = color_info["percentage"]
                        structure_context += f"  * RGB({rgb[0]}, {rgb[1]}, {rgb[2]}) - {pct}%\n"
                
                shape_features = structure_info_data.get("shape_features")
                if shape_features:
                    bbox = shape_features.get("bbox", {})
                    center = shape_features.get("center", {})
                    structure_context += f"- 边界框: x={bbox.get('x', 0)}, y={bbox.get('y', 0)}, width={bbox.get('width', 0)}, height={bbox.get('height', 0)}\n"
                    structure_context += f"- 中心点: ({center.get('x', 0)}, {center.get('y', 0)})\n"
                
                structure_context += "\n**生成指导：**\n"
                structure_context += "- 使用关键点作为路径的起点、终点和控制点\n"
                structure_context += "- 使用轮廓点生成精确的SVG路径，优先使用贝塞尔曲线拟合轮廓\n"
                structure_context += "- 使用主要颜色作为填充和描边颜色\n"
                structure_context += "- 确保生成的SVG元素位置和大小与边界框匹配\n"
            
            # Prepare prompt
            svg_prompt = self._system_prompts["svg_generation"].format(
                stage_number=stage_number,
                component_description=component.description,
                weight=component.weight,
                design_prompt=design_prompt,
                current_svg=current_svg[:500] if current_svg else "",
                image_description=image_description
            )
            
            # Append structure context
            if structure_context:
                svg_prompt += structure_context
            
            # Append reflection context if available
            if reflection_context:
                svg_prompt += reflection_context

            # Use drawer agent to generate SVG
            messages = [HumanMessage(content=svg_prompt)]
            if image_url:
                messages = [HumanMessage(content=[
                    {"type": "text", "text": svg_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ])]

            agent_input = {"messages": messages}
            final_state = await self.drawer_agent.ainvoke(agent_input)

            # Extract SVG elements from all message types (ToolMessage and AIMessage)
            svg_elements = []
            import re
            
            def extract_svg_from_content(content: str) -> List[str]:
                """Extract SVG elements from content string."""
                extracted = []
                if not isinstance(content, str):
                    content = str(content)
                
                # Check if content contains SVG
                if "<svg" in content or "<path" in content or "<circle" in content or "<rect" in content or "<polygon" in content:
                    # Try to parse as JSON first
                    try:
                        import json
                        data = json.loads(content)
                        if isinstance(data, dict):
                            # Extract SVG from tool response
                            svg_value = data.get("value") or data.get("svg") or data.get("result") or content
                            if isinstance(svg_value, str):
                                content = svg_value
                    except:
                        pass  # Not JSON, use content as-is
                    
                    # Extract SVG elements
                    if "<svg" in content:
                        # Extract inner content from SVG wrapper
                        inner_match = re.search(r'<svg[^>]*>(.*?)</svg>', content, re.DOTALL)
                        if inner_match:
                            extracted.append(inner_match.group(1))
                        else:
                            # If no closing tag, try to extract what we can
                            match = re.search(r'<svg[^>]*>(.*)', content, re.DOTALL)
                            if match:
                                extracted.append(match.group(1))
                    else:
                        # Extract individual SVG elements
                        # Path elements
                        paths = re.findall(r'<path[^>]*(?:/>|>.*?</path>)', content, re.DOTALL)
                        extracted.extend(paths)
                        # Other elements
                        other_elements = re.findall(r'<(?:circle|rect|ellipse|line|polygon|polyline|g|text)[^>]*(?:/>|>.*?</\w+>)', content, re.DOTALL)
                        extracted.extend(other_elements)
                        # If no elements found but has SVG-like content, use as-is
                        if not paths and not other_elements and ("<path" in content or "<circle" in content):
                            extracted.append(content)
                
                return extracted
            
            # Check all messages for SVG content
            for msg in final_state.get("messages", []):
                if isinstance(msg, ToolMessage):
                    content = msg.content
                    if isinstance(content, str):
                        svg_elements.extend(extract_svg_from_content(content))
                    else:
                        svg_elements.extend(extract_svg_from_content(str(content)))
                elif isinstance(msg, AIMessage):
                    # Drawer agent may return SVG directly in AIMessage content
                    content = msg.content
                    if isinstance(content, str) and content.strip().startswith('<svg'):
                        # Full SVG document in AIMessage
                        extracted = extract_svg_from_content(content)
                        svg_elements.extend(extracted)
                    elif isinstance(content, str) and ("<svg" in content or "<path" in content or "<circle" in content):
                        # SVG fragments in AIMessage
                        extracted = extract_svg_from_content(content)
                        svg_elements.extend(extracted)

            # Combine with current SVG
            if svg_elements:
                import re
                
                # First, extract inner content from current_svg if it's a complete SVG document
                # This handles the case where current_svg is a full <svg>...</svg> tag
                inner_content = current_svg
                if current_svg.strip().startswith('<svg'):
                    # Extract content inside <svg> tags
                    inner_match = re.search(r'<svg[^>]*>(.*?)</svg>', current_svg, re.DOTALL | re.IGNORECASE)
                    if inner_match:
                        inner_content = inner_match.group(1)
                    else:
                        # If no closing tag, try to extract what we can
                        match = re.search(r'<svg[^>]*>(.*)', current_svg, re.DOTALL | re.IGNORECASE)
                        if match:
                            inner_content = match.group(1)
                
                # Extract all elements from inner content (both self-closing and closing tags)
                # Match self-closing elements: <path ... />
                self_closing = re.findall(r'<(?:path|circle|rect|ellipse|line|polygon|polyline|g|text)[^>]*/>', inner_content, re.IGNORECASE)
                # Match closing tag elements: <path ...>...</path>
                closing_tags = re.findall(r'<(?:path|circle|rect|ellipse|line|polygon|polyline|g|text)[^>]*>.*?</(?:path|circle|rect|ellipse|line|polygon|polyline|g|text)>', inner_content, re.IGNORECASE | re.DOTALL)
                # Combine existing elements (avoid duplicates by checking if element already exists)
                existing_elements = self_closing + closing_tags
                
                # Filter out duplicates from new elements
                new_elements = []
                existing_content = set()
                for elem in existing_elements:
                    # Extract a signature to detect duplicates (use more content for better detection)
                    sig = re.sub(r'\s+', ' ', elem.strip()[:200])
                    existing_content.add(sig)
                
                for elem in svg_elements:
                    sig = re.sub(r'\s+', ' ', elem.strip()[:200])
                    if sig not in existing_content:
                        new_elements.append(elem)
                        existing_content.add(sig)
                
                all_elements = existing_elements + new_elements
                # Create a clean, well-formed SVG document
                new_svg = f'<svg width="1000" height="1000" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">{"".join(all_elements)}</svg>'
                
                # Validate and clean SVG to ensure it's well-formed
                new_svg = self._clean_and_validate_svg(new_svg)
                
                logger.info(f"-> 合并了 {len(new_elements)} 个新SVG元素（共 {len(all_elements)} 个元素）")
            else:
                # Drawer agent没有生成SVG元素 - 提供详细的错误信息
                logger.warning("-> Drawer agent没有生成SVG元素")
                
                # 检查drawer agent的响应，提供诊断信息
                all_messages = final_state.get("messages", [])
                message_summaries = []
                for msg in all_messages[-5:]:  # 检查最后5条消息
                    if isinstance(msg, AIMessage):
                        content_preview = str(msg.content)[:100] if msg.content else "空内容"
                        message_summaries.append(f"AIMessage: {content_preview}...")
                    elif isinstance(msg, ToolMessage):
                        content_preview = str(msg.content)[:100] if msg.content else "空内容"
                        message_summaries.append(f"ToolMessage: {content_preview}...")
                
                logger.warning(f"-> Drawer agent最后的消息: {message_summaries}")
                
                # 如果这是第一次尝试且没有SVG，返回错误信息以便reflection可以处理
                # 但不要完全失败，保留当前SVG以便重试
                error_feedback = f"阶段{stage_number}生成失败：drawer agent未返回有效的SVG元素。组件描述：{component.description}"
                logger.error(error_feedback)
                
                new_svg = current_svg

            # Update component with generated SVG
            component.svg_elements = "".join(svg_elements) if svg_elements else None

            if svg_elements:
                logger.info(f"-> Generated SVG for stage {stage_number}")
            else:
                logger.warning(f"-> Stage {stage_number} failed: no SVG elements generated")

            # Update stage result
            stage_result = StageResult(
                stage_number=stage_number,
                component=component,
                svg_code=new_svg,
                passed_reflection=False,  # Will be updated after reflection
            )
            stages = state.get("stages", [])
            if current_stage < len(stages):
                stages[current_stage] = stage_result
            else:
                stages.append(stage_result)

            return {
                "current_svg": new_svg,
                "current_stage": current_stage,
                "weighted_components": components,
                "stages": stages,
            }
        except Exception as e:
            logger.error(f"Error in generate_svg_stage_node: {e}")
            error_msg = AIMessage(content=f"SVG生成出错: {e}")
            return {"messages": [error_msg]}

    async def reflect_node(self, state: WangState, config=None):
        """Reflect on current SVG quality."""
        logger.info("---WANG AGENT: REFLECT NODE---")
        try:
            current_svg = state.get("current_svg", "")
            instruction = state.get("instruction")
            design_prompt = state.get("design_prompt", "")
            initial_image_path = state.get("initial_image_path")
            current_stage = state.get("current_stage", 0)
            components = state.get("weighted_components", [])
            
            if not current_svg or not instruction:
                error_msg = AIMessage(content="缺少SVG代码或指令")
                return {"messages": [error_msg]}

            if current_stage >= len(components):
                # All stages complete, final reflection
                return {
                    "is_complete": True,
                    "last_reflection": ReflectionResult(
                        passed=True,
                        feedback="所有阶段已完成",
                        suggestions=[]
                    ),
                }

            # Prepare image
            image_description = "参考图像"
            if initial_image_path and os.path.exists(initial_image_path):
                with open(initial_image_path, "rb") as f:
                    encoded_image = base64.b64encode(f.read()).decode("utf-8")
                    image_url = f"data:image/png;base64,{encoded_image}"
            else:
                image_url = None

            # Convert SVG to image for visual comparison
            # Add error handling for invalid SVG
            metrics = None
            diff_analysis = None
            metrics_text = ""
            diff_text = ""
            
            try:
                svg_image_url = convert_svg_to_png_base64(current_svg)
                include_svg_image = True
                
                # Calculate quantitative metrics if we have target image
                if initial_image_path and os.path.exists(initial_image_path):
                    try:
                        # Calculate metrics from base64
                        metrics = calculate_metrics_from_base64(svg_image_url, initial_image_path)
                        metrics_text = format_metrics_for_llm(metrics)
                        logger.info(f"-> 定量指标: {metrics_text[:200]}...")
                        
                        # Analyze differences (save SVG to temp file first)
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                            # Decode base64 and save
                            if svg_image_url.startswith("data:image"):
                                base64_data = svg_image_url.split(",")[1]
                            else:
                                base64_data = svg_image_url
                            import base64 as b64
                            image_data = b64.b64decode(base64_data)
                            tmp_file.write(image_data)
                            tmp_svg_path = tmp_file.name
                        
                        diff_analysis = analyze_svg_differences(tmp_svg_path, initial_image_path)
                        diff_text = format_differences_for_llm(diff_analysis)
                        logger.info(f"-> 差异分析: 发现 {len(diff_analysis.get('top_differences', []))} 个高差异区域")
                        
                        # Clean up temp file
                        try:
                            os.unlink(tmp_svg_path)
                        except:
                            pass
                            
                    except Exception as e:
                        logger.warning(f"-> 指标计算失败: {e}")
                        metrics_text = "定量指标计算失败，将仅使用视觉评估"
                
            except Exception as e:
                logger.warning(f"-> SVG转换失败，跳过图像比较: {e}")
                svg_image_url = None
                include_svg_image = False

            # Prepare prompt with quantitative metrics
            criteria_text = "\n".join([f"- {c}" for c in instruction.criteria])
            
            # Add metrics and differences to prompt
            metrics_section = ""
            if metrics_text:
                metrics_section = f"\n\n## 定量指标评估\n{metrics_text}\n"
            
            differences_section = ""
            if diff_text:
                differences_section = f"\n\n## 差异区域分析\n{diff_text}\n"
            
            reflect_prompt = self._system_prompts["reflect"].format(
                criteria=criteria_text,
                svg_code=current_svg[:1000],  # Limit length
                image_description=image_description,
                design_prompt=design_prompt
            )
            
            # Append metrics and differences to prompt
            reflect_prompt += metrics_section + differences_section

            contents = [
                {"type": "text", "text": reflect_prompt},
            ]
            
            # Only add SVG image if conversion succeeded
            if include_svg_image and svg_image_url:
                contents.append({"type": "image_url", "image_url": {"url": svg_image_url}})
            
            if image_url:
                contents.append({"type": "image_url", "image_url": {"url": image_url}})

            messages = [HumanMessage(content=contents)]
            
            # Get structured output
            structured_llm = self.get_structure_llm(ReflectionResult)
            reflection = await structured_llm.ainvoke(messages)

            logger.info(f"-> Reflection: passed={reflection.passed}, feedback={reflection.feedback[:100]}...")

            # Add metrics and differences to reflection result
            if metrics:
                reflection.metrics = metrics
            if diff_analysis:
                reflection.difference_regions = diff_analysis.get("top_differences", [])
            
            # Update stage result with reflection
            stages = state.get("stages", [])
            current_stage = state.get("current_stage", 0)
            if current_stage < len(stages):
                stages[current_stage].passed_reflection = reflection.passed

            # Update iteration count and stage based on router decision
            # This will be handled by the conditional edge
            update = {
                "last_reflection": reflection,
                "stages": stages,
            }
            
            # If passed and not last stage, increment stage
            if reflection.passed:
                components = state.get("weighted_components", [])
                if current_stage < len(components) - 1:
                    update["current_stage"] = current_stage + 1
                    update["iteration_count"] = state.get("iteration_count", 0) + 1
                else:
                    # Last stage passed, mark as complete
                    update["is_complete"] = True
                    update["current_stage"] = current_stage + 1  # Mark as completed
            else:
                # Failed, increment iteration count for retry
                # Keep current_stage unchanged for rollback
                update["iteration_count"] = state.get("iteration_count", 0) + 1

            return update
        except Exception as e:
            logger.error(f"Error in reflect_node: {e}")
            # Default to passing if reflection fails
            return {
                "last_reflection": ReflectionResult(
                    passed=True,
                    feedback=f"反思过程出错，继续下一步: {e}",
                    suggestions=[]
                ),
            }

    async def fine_tune_node(self, state: WangState, config=None):
        """
        微调节点：基于差异分析，使用参数调整工具微调SVG，而不是完全重新生成。
        """
        logger.info("---WANG AGENT: FINE TUNE NODE---")
        try:
            current_svg = state.get("current_svg", "")
            last_reflection = state.get("last_reflection")
            fine_tune_attempts = state.get("fine_tune_attempts", 0)
            
            if not current_svg or not last_reflection:
                logger.warning("-> 缺少SVG或反思结果，跳过微调")
                return {
                    "fine_tune_attempts": fine_tune_attempts + 1,
                    "messages": [AIMessage(content="微调失败：缺少必要信息")]
                }
            
            # 提取差异区域信息
            difference_regions = last_reflection.difference_regions if hasattr(last_reflection, 'difference_regions') else []
            
            if not difference_regions:
                logger.warning("-> 无差异区域信息，跳过微调")
                return {
                    "fine_tune_attempts": fine_tune_attempts + 1,
                    "messages": [AIMessage(content="微调失败：无差异区域信息")]
                }
            
            # 提取SVG参数
            from src.agents.wang_agent.svg_parameter_tools import extract_svg_parameters
            svg_params = extract_svg_parameters(current_svg)
            
            if "error" in svg_params:
                logger.warning(f"-> 参数提取失败: {svg_params['error']}")
                return {
                    "fine_tune_attempts": fine_tune_attempts + 1,
                    "messages": [AIMessage(content=f"微调失败：参数提取错误 - {svg_params['error']}")]
                }
            
            # 准备微调提示
            # 告诉drawer agent需要微调哪些元素
            top_diff = difference_regions[0]  # 使用差异最大的区域
            diff_region = top_diff.get("region", (0, 0, 100, 100))
            diff_score = top_diff.get("difference", 0)
            
            # 找到该区域内的元素
            x, y, w, h = diff_region
            region_center_x = x + w // 2
            region_center_y = y + h // 2
            
            # 获取参考图像的结构信息，用于更精确的微调
            initial_image_path = state.get("initial_image_path")
            structure_guidance = ""
            
            if initial_image_path and os.path.exists(initial_image_path):
                try:
                    from src.agents.wang_agent.png_structure_tools import extract_image_structure
                    # 提取差异区域的结构信息
                    region_structure = extract_image_structure.invoke({
                        "image_path": initial_image_path,
                        "component_region": (x, y, w, h),
                        "extract_keypoints": True,
                        "extract_contours": True,
                        "extract_colors": True,
                    })
                    
                    if "error" not in region_structure:
                        structure_guidance = "\n\n## 参考图像差异区域的结构信息\n"
                        
                        if region_structure.get("keypoints"):
                            keypoints = region_structure["keypoints"][:10]  # 前10个关键点
                            structure_guidance += f"- 该区域的关键点: {keypoints}\n"
                            structure_guidance += f"  → 建议：将SVG元素的关键点（起点、终点、控制点）调整到这些位置附近\n"
                        
                        if region_structure.get("contours"):
                            contours = region_structure["contours"]
                            structure_guidance += f"- 该区域的轮廓数量: {len(contours)}\n"
                            if contours:
                                first_contour = contours[0][:10]  # 前10个点
                                structure_guidance += f"- 主要轮廓点: {first_contour}\n"
                                structure_guidance += f"  → 建议：使用这些点作为路径的参考，调整路径控制点使其更贴近轮廓\n"
                        
                        if region_structure.get("dominant_colors"):
                            colors = region_structure["dominant_colors"][:3]
                            structure_guidance += f"- 该区域的主要颜色:\n"
                            for color_info in colors:
                                rgb = color_info["rgb"]
                                pct = color_info["percentage"]
                                structure_guidance += f"  * RGB({rgb[0]}, {rgb[1]}, {rgb[2]}) - {pct}%\n"
                            structure_guidance += f"  → 建议：调整该区域元素的颜色，使其更接近这些颜色值\n"
                except Exception as e:
                    logger.warning(f"-> 提取差异区域结构信息失败: {e}")
            
            # 准备微调指令
            fine_tune_prompt = f"""
基于反思反馈，当前SVG需要微调。

差异分析：
- 差异最大的区域：({x},{y}) 到 ({x+w},{y+h})，中心点约({region_center_x},{region_center_y})
- 差异分数：{diff_score:.3f}
- 反思反馈：{last_reflection.feedback[:200]}

当前SVG参数：
- 总元素数：{len(svg_params.get('elements', []))}
- 前3个元素位置：{[elem.get('position') for elem in svg_params.get('elements', [])[:3]]}
{structure_guidance}
请使用以下工具进行微调：
1. extract_svg_parameters_tool - 如果需要，重新提取参数
2. adjust_element_position_tool - 调整元素位置
3. adjust_element_color_tool - 调整元素颜色
4. adjust_path_control_points_tool - 调整路径控制点
5. get_contour_points_for_svg - 获取参考图像的轮廓点（如果需要）

微调策略：
- 优先调整差异区域内的元素
- 根据反思反馈和参考图像结构信息，精确调整位置、颜色或控制点
- 如果提供了关键点或轮廓点，使用这些点作为调整目标
- 保持其他元素不变

请生成微调后的SVG代码。
"""
            
            # 使用drawer agent进行微调
            messages = [HumanMessage(content=fine_tune_prompt)]
            
            # 添加当前SVG作为上下文
            messages.append(HumanMessage(content=f"当前SVG代码：\n{current_svg[:2000]}"))
            
            # 调用drawer agent
            result = await self.drawer_agent.ainvoke({"messages": messages})
            
            # 提取微调后的SVG
            fine_tuned_svg = current_svg  # 默认保持原SVG
            for msg in result.get("messages", []):
                if isinstance(msg, ToolMessage):
                    # 检查是否是调整工具的结果
                    content = msg.content
                    if isinstance(content, str) and ("<svg" in content or "<path" in content):
                        # 尝试提取SVG
                        import re
                        svg_match = re.search(r'(<svg[^>]*>.*?</svg>)', content, re.DOTALL)
                        if svg_match:
                            fine_tuned_svg = svg_match.group(1)
                            logger.info("-> 从工具结果中提取到微调后的SVG")
                elif isinstance(msg, AIMessage):
                    # 检查AIMessage中是否有SVG
                    content = msg.content
                    if isinstance(content, str) and "<svg" in content:
                        import re
                        svg_match = re.search(r'(<svg[^>]*>.*?</svg>)', content, re.DOTALL)
                        if svg_match:
                            fine_tuned_svg = svg_match.group(1)
                            logger.info("-> 从AI消息中提取到微调后的SVG")
            
            # 如果SVG没有变化，说明微调可能失败
            if fine_tuned_svg == current_svg:
                logger.warning("-> 微调后SVG未变化，可能微调失败")
            
            # 清理和验证微调后的SVG
            fine_tuned_svg = self._clean_and_validate_svg(fine_tuned_svg)
            
            logger.info(f"-> 微调完成 (尝试 {fine_tune_attempts + 1})")
            
            return {
                "current_svg": fine_tuned_svg,
                "fine_tune_attempts": fine_tune_attempts + 1,
                "messages": [AIMessage(content=f"已完成微调 (尝试 {fine_tune_attempts + 1})")]
            }
            
        except Exception as e:
            logger.error(f"Error in fine_tune_node: {e}", exc_info=True)
            return {
                "fine_tune_attempts": state.get("fine_tune_attempts", 0) + 1,
                "messages": [AIMessage(content=f"微调过程出错: {e}")]
            }

    def should_continue(self, state: WangState) -> str:
        """Decide whether to continue, rollback, or finish.
        
        完成条件：只有当反思通过（reflection.passed = True）且所有阶段完成时，才标记为完成。
        不应该因为达到max_iterations就标记为完成。
        """
        # Check if already complete
        if state.get("is_complete", False):
            return "finish"
        
        reflection = state.get("last_reflection")
        current_stage = state.get("current_stage", 0)
        components = state.get("weighted_components", [])
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 7)
        
        # Check if all stages complete (only if reflection passed)
        if not components:
            return "finish"
        
        # Check if all stages are done AND last reflection passed
        if current_stage >= len(components):
            if reflection and reflection.passed:
                # All stages done and quality met - truly complete
                return "finish"
            else:
                # All stages done but quality not met - should not happen, but finish anyway
                logger.warning("-> 所有阶段已完成但质量未达标")
                return "finish"
        
        # Check iteration limit
        # 核心原则：只有当反思通过（SVG与图像接近）时才完成，而不是因为达到迭代次数
        # 但如果达到最大迭代次数，即使质量未达标，也要结束并输出当前SVG
        if iteration_count >= max_iterations:
            logger.warning(f"-> 达到最大迭代次数 ({max_iterations})")
            if reflection and reflection.passed and current_stage >= len(components) - 1:
                # Quality is good and last stage, finish with quality_passed
                logger.info("-> 虽然达到最大迭代次数，但质量已达标，标记为完成")
                # Note: is_complete will be set to True in reflect_node when reflection.passed
                return "finish"
            else:
                # 达到最大迭代次数但质量未达标，停止但标记为未完成
                logger.warning(f"-> 达到最大迭代次数 ({max_iterations}) 但质量未达标，停止并输出当前SVG")
                # Mark stop reason but keep is_complete=False
                return "finish"  # Will be handled to set stop_reason
        
        # Check reflection result
        if reflection:
            if reflection.passed:
                # Quality is good, reset fine-tune attempts
                # Check if this was the last stage
                if current_stage >= len(components) - 1:
                    # Last stage passed, finish
                    return "finish"
                # Move to next stage - update state will be handled by the node
                return "next_stage"
            else:
                # Quality not good - try fine-tuning first before rollback
                fine_tune_attempts = state.get("fine_tune_attempts", 0)
                max_fine_tune_attempts = state.get("max_fine_tune_attempts", 2)
                
                # Check if we have difference regions to guide fine-tuning
                has_difference_regions = (
                    hasattr(reflection, 'difference_regions') and
                    reflection.difference_regions and 
                    len(reflection.difference_regions) > 0
                )
                
                if fine_tune_attempts < max_fine_tune_attempts and has_difference_regions:
                    # Try fine-tuning instead of rollback
                    logger.info(f"-> 反思未通过，尝试微调 (尝试 {fine_tune_attempts + 1}/{max_fine_tune_attempts})")
                    return "fine_tune"  # New route for fine-tuning
                else:
                    # Fine-tuning failed or no difference regions, rollback
                    if fine_tune_attempts >= max_fine_tune_attempts:
                        logger.warning(f"-> 微调尝试已达上限 ({max_fine_tune_attempts})，回退到重新生成")
                        # Reset fine-tune attempts for next rollback cycle
                        state["fine_tune_attempts"] = 0
                    else:
                        logger.warning("-> 无差异区域信息，直接回退")
                    return "rollback"
        
        return "next_stage"

    def router_condition(self, state: WangState) -> str:
        """Route condition based on reflection result."""
        decision = self.should_continue(state)
        logger.info(f"-> Router decision: {decision}")
        
        # Set stop_reason when finishing
        if decision == "finish":
            reflection = state.get("last_reflection")
            current_stage = state.get("current_stage", 0)
            components = state.get("weighted_components", [])
            iteration_count = state.get("iteration_count", 0)
            max_iterations = state.get("max_iterations", 7)
            is_complete = state.get("is_complete", False)
            
            if is_complete:
                # Quality passed and all stages done
                state["stop_reason"] = "quality_passed"
            elif iteration_count >= max_iterations:
                # Reached max iterations
                state["stop_reason"] = "max_iterations"
            elif current_stage >= len(components) if components else False:
                # All stages done but quality not met
                state["stop_reason"] = "stages_complete_quality_failed"
            else:
                state["stop_reason"] = "unknown"
        
        return decision

    def build_graph(self):
        """Build the agent graph."""
        graph = StateGraph(WangState)
        
        # Add nodes
        graph.add_node("think_node", self.think_node)
        graph.add_node("generate_image_node", self.generate_image_node)
        graph.add_node("analyze_weights_node", self.analyze_weights_node)
        graph.add_node("generate_svg_stage_node", self.generate_svg_stage_node)
        graph.add_node("reflect_node", self.reflect_node)
        graph.add_node("fine_tune_node", self.fine_tune_node)
        
        # Add edges
        graph.add_edge(START, "think_node")
        graph.add_edge("think_node", "generate_image_node")
        graph.add_edge("generate_image_node", "analyze_weights_node")
        graph.add_edge("analyze_weights_node", "generate_svg_stage_node")
        graph.add_edge("generate_svg_stage_node", "reflect_node")
        
        # Conditional routing after reflection
        graph.add_conditional_edges(
            "reflect_node",
            self.router_condition,
            {
                "finish": END,
                "next_stage": "generate_svg_stage_node",
                "rollback": "generate_svg_stage_node",
                "fine_tune": "fine_tune_node",  # New route for fine-tuning
            }
        )
        
        # After fine-tuning, go back to reflection
        graph.add_edge("fine_tune_node", "reflect_node")
        
        return graph

