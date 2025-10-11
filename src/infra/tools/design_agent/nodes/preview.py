# src/infra/tools/design_agent/nodes/preview.py
import logging
import json
import os
import asyncio
import time

from langchain_core.messages import AIMessage, HumanMessage

from src.infra.tools.image_tools import generate_image
from src.infra.tools.draw_canvas_tools import draw_agent_with_tool
from src.infra.tools.design_agent.schemas import StageEnum
from src.infra.tools.design_agent.nodes.memory import note_helper

logger = logging.getLogger(__name__)

async def preview_image_node(state, config=None):
    logger.info(f"-> Preview image node entered")
    try:
        messages = state.get('messages', [])
        last_text = ""
        if state.get('compose_map_messages'):
            last_text = state['compose_map_messages'][-1].get('improved_design_prompt', '') or state['compose_map_messages'][-1].get('design_prompt', '')
            last_style_hint = state['compose_map_messages'][-1].get('imagen_style_prefix', '')
        if not last_text:
            last_text = "Please generate a preview image for the current design based on the history."

        logger.info(f"-> Preview image generation prompt: {last_text}")
        prompt = f"{last_style_hint}. {last_text}"

        async def generate_image_and_save(prompt, aspect_ratio):
            image_pil = generate_image(prompt=prompt, aspect_ratio=aspect_ratio)
            if state.get('project_dir'):
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                save_path = os.path.join(state['project_dir'], f'preview_{timestamp}.png')
                image_pil.save(save_path)
                png_path = save_path
            else:
                png_path = None
            return png_path

        png_path = await generate_image_and_save(prompt=prompt, aspect_ratio="1:1")
        image_paths = state.get('generated_image_paths', [])

        if png_path:
            image_paths.append(png_path)
        logger.info(f"-> Preview image node image paths: {image_paths}")

        ai_msg = AIMessage(content=json.dumps({"image_preview_url": png_path, "image_generation_prompt": prompt}, ensure_ascii=False))
        if state.get('compose_map_messages'):
            state['compose_map_messages'][-1].update({"image_preview_url": png_path})
        new_messages = [ai_msg]

        update = {
            "multi_modal_messages": state.get('multi_modal_messages', []) + [{"image_generation_prompt": prompt, "image_url": png_path, "next_step": StageEnum.REFLECT}],
            "generated_image_paths": image_paths,
            "stage": StageEnum.REFLECT,
            "messages": new_messages,
            "stage_iteration": state.get('stage_iteration', 0) + 1,
        }
        try:
            if note_helper is not None:
                note_text = f"Generated image: {png_path or 'in-memory'}; prompt: {prompt}"
                _ = await note_helper.do_notes([HumanMessage(content=note_text)])
                update['notes'] = note_helper.get_notes()
        except Exception:
            pass
        return update
    except Exception as e:
        logger.error(f"Error in preview_image_node: {e}")
        return {"messages": state.get('messages', [])}

async def preview_svg_node(state, config=None):
    if state.get('stage_iteration', 0) > 10:
        return {
            "messages": AIMessage(content="Task stop due to stage iteration limit. Summarize the current design."),
            "stage": StageEnum.SUMMARIZE,
        }

    logger.info("-> Preview SVG node entered (delegating to draw_canvas_tools.draw_agent_with_tool)")
    try:
        draw_desc = None
        if state.get('summarize_map'):
            draw_desc = state['summarize_map'].get('draw_description')
        if not draw_desc and state.get('compose_map_messages'):
            draw_desc = state['compose_map_messages'][-1].get('design_prompt')

        if not draw_desc:
            draw_desc = "Preview SVG for current design"

        width = 400
        height = 400

        plan_context = None
        critique_notes = None
        if state.get('plan_map'):
            try:
                plan_context = json.dumps(state['plan_map'], ensure_ascii=False)
            except Exception:
                plan_context = str(state['plan_map'])
        if state.get('reflect_map_messages'):
            try:
                critique_notes = json.dumps(state['reflect_map_messages'][-1], ensure_ascii=False)
            except Exception:
                critique_notes = str(state['reflect_map_messages'][-1])

        draw_result_json = await draw_agent_with_tool(task_description=draw_desc, width=width, height=height, plan_context=plan_context or "", critique_notes=critique_notes or "")
        try:
            draw_result = json.loads(draw_result_json)
        except Exception:
            draw_result = {"svg": draw_result_json}

        new_svg = draw_result.get('svg') or draw_result.get('value') or (draw_result_json if isinstance(draw_result_json, str) else '')

        svg_path = None
        if new_svg and state.get('project_dir'):
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f'preview_svg_{timestamp}.svg'
            svg_path = os.path.join(state['project_dir'], filename)
            open(svg_path, 'w', encoding='utf-8').write(new_svg)

        svg_paths = state.get('generated_svg_paths', [])
        if svg_path:
            svg_paths.append(svg_path)

        ai_msg = AIMessage(content=f"Draw agent produced svg: {svg_path or 'in-memory'}")

        update = {
            "multi_modal_messages": state.get('multi_modal_messages', []) + [{"image_generation_prompt": draw_desc, "image_url": svg_path or '', "next_step": StageEnum.REFLECT}],
            "generated_svg_paths": svg_paths,
            "messages": [ai_msg],
            "stage": StageEnum.REFLECT,
            "stage_iteration": state.get('stage_iteration', 0) + 1,
        }
        try:
            if note_helper is not None:
                note_text = f"Generated svg: {svg_path or 'in-memory'}; draw_desc: {draw_desc}"
                _ = await note_helper.do_notes([HumanMessage(content=note_text)])
                update['notes'] = note_helper.get_notes()
        except Exception:
            pass
        return update
    except Exception as e:
        logger.error(f"Error in preview_svg_node: {e}")
        return {"messages": state.get('messages', [])}
