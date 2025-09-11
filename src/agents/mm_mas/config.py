from dataclasses import dataclass
from src.config.manager import BaseThreadConfiguration

@dataclass(kw_only=True)
class ThreadConfiguration(BaseThreadConfiguration):
    max_ppt_pages: int = 20
    max_cards: int = 10
    enable_digital_human: bool = True