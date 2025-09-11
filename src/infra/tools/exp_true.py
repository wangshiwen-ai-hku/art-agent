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

import os
import time
import asyncio
import uuid
from typing import List, Optional
from urllib.parse import urljoin

import requests
import dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import tool

dotenv.load_dotenv()

# -------------------- 配置 --------------------
EXP_API_BASE_URL: str = os.getenv("EXP_API_BASE_URL", "http://8.140.225.115:11234")
EXP_HEALTH_ENDPOINT: str = "/health"
EXP_SUBMIT_ENDPOINT: str = "/generate"
EXP_PROGRESS_ENDPOINT: str = "/progress"
EXP_POLL_INTERVAL: int = int(os.getenv("EXP_POLL_INTERVAL", "3"))
EXP_TIMEOUT: int = int(os.getenv("EXP_TIMEOUT", "300"))  # 秒
# --------------------------------------------


# ==================== 请求 / 响应模型 ====================
class ExperimentRequest(BaseModel):
    course: str = Field(description="学科名称，如 物理")
    target: str = Field(description="学段，如 高中")
    experiment: str = Field(description="实验名称，如 牛顿第二定律")
    level: str = Field(description="难度等级，如 中级")
    interactive: str = Field(description="交互深度，如 深度")
    model: str = Field(description="所用模型，如 gemini-2.5-flash")
    description: str = Field(description="实验描述/需求")


class ExperimentResult(BaseModel):
    task_id: str
    preview_url: str
    download_url: str
    local_html_path: Optional[str] = None


# ==================== 工具实现 ====================
def _health_check() -> bool:
    """检测服务是否存活"""
    try:
        r = requests.get(f"{EXP_API_BASE_URL}{EXP_HEALTH_ENDPOINT}", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _submit(req: ExperimentRequest) -> str:
    """提交任务，返回 task_id"""
    r = requests.post(
        f"{EXP_API_BASE_URL}{EXP_SUBMIT_ENDPOINT}",
        json=req.model_dump(),
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()["task_id"]


def _poll(task_id: str) -> ExperimentResult:
    """轮询直到终态，返回结果"""
    start = time.time()
    while True:
        if time.time() - start > EXP_TIMEOUT:
            raise TimeoutError("任务轮询超时")
        r = requests.get(
            f"{EXP_API_BASE_URL}{EXP_PROGRESS_ENDPOINT}/{task_id}", timeout=10
        )
        r.raise_for_status()
        data = r.json()
        state = data.get("state")
        if state == "SUCCESS":
            return ExperimentResult(
                task_id=task_id,
                preview_url=data["preview_url"],
                download_url=data["download_url"],
            )
        if state == "FAILURE":
            raise RuntimeError("服务端任务失败: " + str(data))
        time.sleep(EXP_POLL_INTERVAL)


def _download(result: ExperimentResult) -> ExperimentResult:
    """把 HTML 拉到本地"""
    local_filename = f"{result.task_id}.html"
    with requests.get(
        f"{EXP_API_BASE_URL}{result.download_url}", stream=True, timeout=30
    ) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    result.local_html_path = os.path.abspath(local_filename)
    return result


# -------------------- LangChain Tool --------------------
@tool
def generate_experiment(
    course: str,
    target: str,
    experiment: str,
    level: str = "中级",
    interactive: str = "深度",
    model: str = "gemini-2.5-flash",
    description: str = "",
) -> List[dict]:
    """
    调用本地实验生成 API，提交任务并轮询直到完成，最终返回本地 HTML 文件路径。

    参数
    ----
    course : str
        学科名称，如 "物理"
    target : str
        学段，如 "高中"
    experiment : str
        实验名称，如 "牛顿第二定律"
    level : str, optional
        难度等级，默认 "中级"
    interactive : str, optional
        交互深度，默认 "深度"
    model : str, optional
        所用模型，默认 "gemini-2.5-flash"
    description : str, optional
        实验描述/需求

    返回
    ----
    List[dict]
        仅含一个元素，字段包括：
        - task_id
        - preview_url
        - download_url
        - local_html_path  （本地可打开的 HTML 绝对路径）
    """
    return "call_exp_tool"
    if not _health_check():
        raise RuntimeError("实验生成服务未就绪")

    req = ExperimentRequest(
        course=course,
        target=target,
        experiment=experiment,
        level=level,
        interactive=interactive,
        model=model,
        description=description,
    )
    task_id = _submit(req)
    print(f"Task submitted: {task_id}")
    result = _poll(task_id)
    result = _download(result)
    return [result.model_dump()]


# ==================== 自测入口 ====================
async def _async_selftest():
    """async 自测"""
    print("=== Experiment Tool Self-test ===")
    try:
        res = await generate_experiment.ainvoke(
            {
                "course": "物理",
                "target": "高中",
                "experiment": "牛顿第二定律",
                "description": "纸带法探究位移和速度/时间的关系",
            }
        )
        print("Task finished:", res[0])
    except Exception as e:
        print("Error:", e)
    print("=== Done ===")


if __name__ == "__main__":
    asyncio.run(_async_selftest())