#!/usr/bin/env python3
"""
Flashcard Generation Tool
~~~~~~~~~~~~~~~~~~~~~~~~~
LangChain-compatible wrapper for the local flashcard-generation API
(the one wrapped by your shell script).

Usage
-----
from flashcard import generate_flashcards

cards = await generate_flashcards.ainvoke({
    "course": "语文",
    "target": "高中",
    "topic": "阿房宫赋的相关背景和文言知识",
    "level": "初级",
    "number": 10,
    "material": "侧重于一词多义的对比和分析"
})
"""

import os
import time
import asyncio
from typing import List
from urllib.parse import urljoin

import requests
import dotenv
from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.tools import tool

dotenv.load_dotenv()

# -------------------- 配置 --------------------
FLASHCARD_API_BASE_URL: str = os.getenv("FLASHCARD_API_BASE_URL", "http://8.140.225.115:13234")
FLASHCARD_HEALTH_ENDPOINT: str = "/health"
FLASHCARD_SUBMIT_ENDPOINT: str = "/generate"
FLASHCARD_PROGRESS_ENDPOINT: str = "/progress"
FLASHCARD_RESULT_ENDPOINT: str = "/result"
FLASHCARD_POLL_INTERVAL: int = int(os.getenv("FLASHCARD_POLL_INTERVAL", "3"))
FLASHCARD_TIMEOUT: int = int(os.getenv("FLASHCARD_TIMEOUT", "300"))  # 秒
# --------------------------------------------

from typing import Optional
from pydantic import BaseModel, Field

class FlashcardRequest(BaseModel):
    # ---------------- 必填 ----------------
    course:   str = Field(..., example="语文")
    target:   str = Field(..., example="高中")
    topic:    str = Field(..., example="阿房宫赋的相关背景和文言知识")
    level:    str = Field(..., example="初级")
    number:   int = Field(..., example=10)

    # ---------------- 选填 ----------------
    requirement: str = Field(
        "完整且准确的进行知识巩固和复习",
        example="侧重于一词多义的对比和分析",
        description="具体要求，比如侧重于一词多义的对比和分析"
    )
    material: Optional[str] = Field(
        None,      # 默认无参考材料
        description="参考材料，如果有的话，默认为模型内部生成"
    )
    model: str = Field("gemini-2.5-pro", example="gemini-2.5-pro")
   

class FlashcardResult(BaseModel):
    task_id: str
    # preview_folder: str
    download_url: str
    local_zip_path: str


# ==================== 内部辅助 ====================
def _health_check() -> bool:
    try:
        r = requests.get(f"{FLASHCARD_API_BASE_URL}{FLASHCARD_HEALTH_ENDPOINT}", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _submit(req: FlashcardRequest) -> str:
    r = requests.post(
        f"{FLASHCARD_API_BASE_URL}{FLASHCARD_SUBMIT_ENDPOINT}",
        json=req.model_dump(),
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()["task_id"]


def _poll(task_id: str) -> FlashcardResult:
    start = time.time()
    while True:
        if time.time() - start > FLASHCARD_TIMEOUT:
            raise TimeoutError("轮询超时")
        r = requests.get(
            f"{FLASHCARD_API_BASE_URL}{FLASHCARD_PROGRESS_ENDPOINT}/{task_id}", timeout=10
        )
        r.raise_for_status()
        data = r.json()
        state = data.get("state")
        if state == "SUCCESS":
            # 再拉一次 result 端点拿下载地址
            res = requests.get(
                f"{FLASHCARD_API_BASE_URL}{FLASHCARD_RESULT_ENDPOINT}/{task_id}", timeout=10
            )
            res.raise_for_status()
            res = res.json()
            return FlashcardResult(
                task_id=task_id,
                # preview_folder=res["preview_folder"],
                download_url=res["download_url"],
                local_zip_path="",  # 后面补充
            )
        if state == "FAILURE":
            raise RuntimeError("服务端任务失败: " + str(data))
        time.sleep(FLASHCARD_POLL_INTERVAL)


def _download(result: FlashcardResult) -> FlashcardResult:
    local_filename = f"{result.task_id}.zip"
    with requests.get(f"{FLASHCARD_API_BASE_URL}{result.download_url}", stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    result.local_zip_path = os.path.abspath(local_filename)
    return result


# -------------------- LangChain Tool --------------------
@tool
def generate_flashcard(
    course: str,
    target: str,
    topic: str,
    level: str = "初级",
    number: int = 10,
    material: str = "",
) -> List[dict]:
    """
    调用本地闪卡生成 API，提交任务并轮询直到完成，最终返回打包好的 zip 本地路径。

    参数
    ----
    course : str
        学科名称，如 "语文"
    target : str
        学段，如 "高中"
    topic : str
        知识点或课文，如 "阿房宫赋"
    level : str, optional
        难度等级，默认 "初级"
    number : int, optional
        需要生成的卡片数量，默认 10
    material : str, optional
        额外要求，如 "侧重一词多义"

    返回
    ----
    List[dict]
        仅含一个元素，字段包括：
        - task_id
        - download_url
    """
    return "call_flashcard_tool"
    if not _health_check():
        raise RuntimeError("闪卡生成服务未就绪")

    req = FlashcardRequest(
        course=course, target=target, topic=topic, level=level, number=number, material=material
    )
    task_id = _submit(req)
    print(f"Task submitted: {task_id}")
    result = _poll(task_id)
    result = _download(result)
    return [result.model_dump()]


# ==================== 自测入口 ====================
async def _async_selftest():
    print("=== Flashcard Tool Self-test ===")
    try:
        res = await generate_flashcard.ainvoke(
            {
                "course": "语文",
                "target": "高中",
                "topic": "阿房宫赋的相关背景和文言知识",
                "level": "初级",
                "number": 10,
                "requirement": "侧重于一词多义的对比和分析",
                "model": "gemini-2.5-flash",
                "material": "",
            }
        )
        print("Task finished:", res[0])
    except Exception as e:
        print("Error:", e)
    print("=== Done ===")


if __name__ == "__main__":
    # call for Health
    print("=== Flashcard Tool Health Check ===")
    print(_health_check())
    print("=== Done ===")
    asyncio.run(_async_selftest())