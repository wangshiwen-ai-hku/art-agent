from dataclasses import dataclass

from src.config.manager import BaseThreadConfiguration


@dataclass(kw_only=True)
class ThreadConfiguration(BaseThreadConfiguration):
    """Deep Researcher agent configuration."""

    # Plan and execution configuration specific to deep researcher
    max_step_num: int = 8  # Maximum number of steps in a plan
    auto_accepted_plan: bool = False  # Whether to automatically accept plans
    max_search_results: int = 7  # Maximum number of search results
