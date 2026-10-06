"""Shared state passed between the nodes of the graph."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage


class NewsState(TypedDict, total=False):
    topic: str
    # Running log of what each node did (appended, never overwritten).
    messages: Annotated[list[BaseMessage], operator.add]
    next_agent: str

    generated_news: str
    rectified_news: str
    final_article: str

    evaluation: Literal["genuine", "needs_correction", "pending"]
    feedback: str
    # Factual-accuracy score for the latest version, 0.0 to 1.0.
    confidence: float
    # One score per evaluation round (appended).
    confidence_history: Annotated[list[float], operator.add]

    # Number of rectification rounds completed so far.
    iteration: int
    stopping_decision: Literal["stop", "continue", "pending"]
    stop_reason: str
    # True only if the evaluator's last verdict was "genuine".
    verified: bool


def initial_state(topic: str, article: str | None = None) -> NewsState:
    """Starting state. Pass `article` to verify your own text instead of a generated one."""
    return {
        "topic": topic,
        "messages": [],
        "generated_news": (article or "").strip(),
        "rectified_news": "",
        "evaluation": "pending",
        "feedback": "",
        "confidence": 0.0,
        "confidence_history": [],
        "iteration": 0,
        "stopping_decision": "pending",
        "stop_reason": "",
        "verified": False,
    }


def current_article(state: NewsState) -> str:
    """The latest version of the article: the last correction, else the first draft."""
    return state.get("rectified_news") or state.get("generated_news", "")
