#!/usr/bin/env python3
"""
Experiment Generation Tool
~~~~~~~~~~~~~~~~~~~~~~~~~~
A LangChain-compatible tool for calling the local experiment-generation API
(the one wrapped by your shell script).

Usage
-----
from exp_tool import submit_experiment_task

images = await submit_experiment_task.ainvoke({
    "course": "物理",
    "target": "高中",
    "experiment": "牛顿第二定律",
    "level": "中级",
    "interactive": "深度",
    "model": "gemini-2.5-flash",
    "description": "纸带法探究位移和速度/时间的关系"
})
"""

import logging
from typing import List, Dict
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

@tool
def generate_experiment(
    course: str,
    target: str,
    experiment: str,
    level: str = "中级",
    interactive: str = "深度",
    model: str = "gemini-2.5-flash",
    description: str = "",
) -> Dict:
    """
    【模拟】生成科学实验。
    本工具为模拟工具，仅记录调用参数并返回一个固定的、符合格式的示例结果。
    它不执行任何实际的实验生成操作。
    """
    # 1. 记录调用信息
    logger.info("--- MOCK TOOL CALL: generate_experiment ---")
    logger.info(f"  Course: {course}")
    logger.info(f"  Target: {target}")
    logger.info(f"  Experiment: {experiment}")
    logger.info(f"  Level: {level}")
    logger.info(f"  Interactive: {interactive}")
    logger.info(f"  Model: {model}")
    logger.info(f"  Description: {description}")
    
    # 2. 创建并返回一个符合格式的假数据
    mock_result = {
        "experiment_id": "mock_exp_12345",
        "title": f"模拟实验：{experiment}",
        "objective": f"通过互动实验理解 '{experiment}' 的原理。",
        "url": f"https://example.com/exp/{experiment.replace(' ', '_')}"
    }
    
    logger.info(f"  Returning mock result: {mock_result}")
    logger.info("-------------------------------------------")
    
    return mock_result

# ==============================================================================
#  以下的原始代码已被注释掉，以启用模拟模式
# ==============================================================================
#
# import os
# import time
# import asyncio
# ... (and so on for the rest of the original file)