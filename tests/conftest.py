"""Stand-ins for the models and the web search, so tests run offline and free."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from news_verifier import Settings, build_graph, initial_state  # noqa: E402


class ScriptedModel:
    """Replays a fixed list of replies and records every prompt it was sent."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def invoke(self, prompt):
        if not isinstance(prompt, str):
            prompt = "\n".join(message.content for message in prompt)
        self.prompts.append(prompt)
        if not self.replies:
            raise AssertionError("Model was called more times than the test scripted")
        return SimpleNamespace(content=self.replies.pop(0))


class RepeatingJudge:
    """A judge that never accepts the article and never agrees to stop."""

    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        if "DECISION:" in prompt:
            return SimpleNamespace(content="DECISION: continue\n\nSTOPPING REASON:\nStill wrong.")
        return SimpleNamespace(
            content="EVALUATION: needs_correction\n\nFEEDBACK:\nStill wrong.\n\nCONFIDENCE:\n0.3"
        )


class RepeatingWriter:
    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        marker = "RECTIFIED_ARTICLE:" if isinstance(prompt, str) else "GENERATED_ARTICLE:"
        return SimpleNamespace(content=f"{marker}\nDraft number {self.calls}.")


@pytest.fixture
def run_graph():
    def _run(writer, judge, *, article=None, max_iterations=8, search=None):
        settings = Settings(max_iterations=max_iterations)
        app = build_graph(
            settings=settings,
            writer_llm=writer,
            judge_llm=judge,
            search=search or (lambda query: "[1] Stub result (https://example.com)"),
        )
        return app.invoke(
            initial_state("FIFA", article),
            config={"recursion_limit": settings.recursion_limit},
        )

    return _run
