"""Assemble the LangGraph graph and provide a one-call `run` helper."""

from __future__ import annotations

from functools import partial
from typing import Callable

from langgraph.graph import END, START, StateGraph

from .config import Settings, load_settings
from .nodes import make_nodes, route_from_supervisor
from .search import web_search
from .state import NewsState, initial_state

WORKERS = ("generator", "evaluator", "stopping_agent", "rectifier")


def _groq_model(name: str, temperature: float):
    # Imported here so the graph can be built with stand-in models (as the
    # tests do) without langchain-groq or an API key being needed.
    from langchain_groq import ChatGroq

    return ChatGroq(model=name, temperature=temperature)


def build_graph(
    settings: Settings | None = None,
    writer_llm=None,
    judge_llm=None,
    search: Callable[[str], str] | None = None,
):
    """Compile the graph.

    Every worker reports back to the supervisor, which picks the next step:

        generator -> evaluator -> (stopping_agent -> rectifier -> evaluator)* -> finish
    """
    settings = settings or load_settings()
    writer_llm = writer_llm or _groq_model(settings.writer_model, settings.temperature)
    judge_llm = judge_llm or _groq_model(settings.judge_model, settings.temperature)
    search = search or partial(web_search, max_results=settings.search_results)

    nodes = make_nodes(writer_llm, judge_llm, search, settings)

    graph = StateGraph(NewsState)
    for name, node in nodes.items():
        graph.add_node(name, node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {**{worker: worker for worker in WORKERS}, "finish": "finish"},
    )
    for worker in WORKERS:
        graph.add_edge(worker, "supervisor")
    graph.add_edge("finish", END)

    return graph.compile()


def run(
    topic: str,
    article: str | None = None,
    settings: Settings | None = None,
    **graph_kwargs,
) -> NewsState:
    """Run the pipeline on a topic and return the final state.

    If `article` is given, that text is verified and corrected; otherwise the
    generator writes a draft first.
    """
    settings = settings or load_settings()
    app = build_graph(settings=settings, **graph_kwargs)
    return app.invoke(
        initial_state(topic, article),
        config={"recursion_limit": settings.recursion_limit},
    )
