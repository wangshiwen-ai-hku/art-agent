import os
from dataclasses import dataclass, field, fields
from typing import Any, Optional
from abc import ABC

from langchain_core.runnables import RunnableConfig
from src.config.manager import config
from src.config.manager import BaseThreadConfiguration


@dataclass(kw_only=True)
class ThreadConfiguration(BaseThreadConfiguration):
    pass
