from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
from mem0 import Memory

from src.config.mem import MEM_CONFIG

# TODO


class EduMemory:
    """
    Educational Memory wrapper for managing student learning experiences,
    progress tracking, and personalized educational content.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize EduMemory with mem0 backend."""
        config = MEM_CONFIG if config is None else config
        self.mem0 = Memory.from_config(config)
        self.subjects = [
            "math",
        ]  # TODO: add more subjects

    async def add_interaction(
        self,
        messages: List[Dict[str, str]],
        user_id: str,
        metadata: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Add a learning interaction to student's memory.
        """
        result = await asyncio.to_thread(
            self.mem0.add, messages, user_id=user_id, metadata=metadata, **kwargs
        )
        return result

    async def search(
        self,
        query: str,
        user_id: str,
        limit: Optional[int] = None,
        filters: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Search the memory for relevant information.
        """
        result = await asyncio.to_thread(
            self.mem0.search,
            query,
            user_id=user_id,
            limit=limit,
            filters=filters,
            **kwargs,
        )
        return result

    async def get_preference_profile(
        self, user_id: str, limit: Optional[int] = 20, **kwargs
    ) -> Dict[str, Any]:
        """
        Get preference profile of a user; then format the results into a structured format.
        """
        result = await self.search(
            query="learning preference", user_id=user_id, limit=limit, **kwargs
        )
        preference = "学生的学习偏好如下：\n"
        for item in result["results"]:
            preference += f"- {item['memory']}\n"
        if len(result["results"]) == 0:
            preference += "暂无学习偏好信息。"
        return preference

    async def get_all(self, user_id: str, **kwargs) -> Dict[str, Any]:
        """
        Get all interactions of a user.
        """
        result = await asyncio.to_thread(self.mem0.get_all, user_id=user_id, **kwargs)
        return result
