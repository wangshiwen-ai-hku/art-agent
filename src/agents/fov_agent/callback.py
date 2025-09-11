# src/utils/llm_callback.py
import time, json, logging
from typing import Any, Dict, Optional
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult

logger = logging.getLogger(__name__)

class DesignerNodeCallback(BaseCallbackHandler):
    """专门监控 designer_node 里 LLM 的耗时、输入、输出"""
    def __init__(self, node_name: str = "designer_node"):
        super().__init__()
        self.node_name = node_name

    # ---- 异步版本（LangGraph 默认走 async） ----
    async def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: list[str],
        *,
        run_id,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._start = time.time()
        # 把完整请求体落盘（OpenAI 接口能看到 messages 字段）
        logger.info("[designer] 📤 请求体: %s", json.dumps(kwargs.get("invocation_params", {}), ensure_ascii=False, indent=2))
        for idx, p in enumerate(prompts):
            logger.info("[designer] 📝 prompt-%d: %.800s", idx, p)
            logger.info(f"[{self.node_name}] ⏱  LLM 启动, run_id={run_id}")
            # 把 prompt 也落盘（太长可以只打印前 500 字符）
            for idx, p in enumerate(prompts):
                logger.info(f"[{self.node_name}] 📝 prompt-{idx}: {p[:500]}...")

    async def on_llm_end(
        self, response: LLMResult, *, run_id, **kwargs: Any
    ) -> None:
        cost = time.time() - self._start
        logger.info(f"[{self.node_name}] ✅ LLM 完成, run_id={run_id}, 耗时={cost:.2f}s")
        for g in response.generations:
            for chunk in g:
                logger.info(f"[{self.node_name}] 💬 输出: {chunk.text[:500]}...")

    # ---- 同步版本（防止有人手动 invoke） ----
    def on_llm_start(self, *args, **kwargs):
        pass   # 复用异步即可，这里留空
    def on_llm_end(self, *args, **kwargs):
        pass