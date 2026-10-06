"""Runtime settings.

Every value can be overridden from the environment (or a local .env file),
so nothing in the code needs editing to switch models or loop limits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Judges the article and decides whether to keep looping.
    judge_model: str = "openai/gpt-oss-120b"
    # Writes the first draft and every corrected version.
    writer_model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0
    # Hard cap on rectification rounds, enforced by the supervisor.
    max_iterations: int = 8
    # Web results fetched per search.
    search_results: int = 5
    # The evaluator searches on the first N characters of the article.
    search_query_chars: int = 300
    # Truncation applied before the article / feedback go back into a prompt.
    article_chars: int = 1000
    feedback_chars: int = 800

    @property
    def recursion_limit(self) -> int:
        """LangGraph step budget: 6 graph steps per round plus fixed overhead."""
        return 6 * self.max_iterations + 20


def load_settings() -> Settings:
    defaults = Settings()
    return Settings(
        judge_model=os.getenv("JUDGE_MODEL", defaults.judge_model),
        writer_model=os.getenv("WRITER_MODEL", defaults.writer_model),
        max_iterations=int(os.getenv("MAX_ITERATIONS", defaults.max_iterations)),
        search_results=int(os.getenv("SEARCH_RESULTS", defaults.search_results)),
    )
