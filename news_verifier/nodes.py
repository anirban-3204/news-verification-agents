"""The agents (graph nodes) and the supervisor that routes between them."""

from __future__ import annotations

import logging
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from . import prompts
from .config import Settings
from .parsing import extract_article, parse_evaluation, parse_stop_decision
from .search import trusted_sources_query
from .state import NewsState, current_article

log = logging.getLogger("news_verifier")

FINISH = "FINISH"


def route(state: NewsState, max_iterations: int) -> tuple[str, str]:
    """Decide which agent runs next. Returns (next_agent, reason_if_finishing).

    Routing is plain Python on the state rather than an LLM call: the order of
    the steps is fixed, so there is nothing for a model to decide here.
    """
    evaluation = state.get("evaluation", "pending")
    decision = state.get("stopping_decision", "pending")

    if not state.get("generated_news"):
        return "generator", ""
    if evaluation == "pending":
        return "evaluator", ""
    if evaluation == "genuine":
        return FINISH, "Evaluator judged the article genuine."
    if decision == "stop":
        return FINISH, state.get("stop_reason", "") or "Stopping agent ended the loop."
    if state.get("iteration", 0) >= max_iterations:
        return FINISH, f"Reached the limit of {max_iterations} rectification rounds."
    if decision == "pending":
        return "stopping_agent", ""
    return "rectifier", ""


def make_nodes(
    writer_llm,
    judge_llm,
    search: Callable[[str], str],
    settings: Settings,
) -> dict[str, Callable[[NewsState], dict]]:
    """Build the node functions around the given models and search function."""

    def supervisor_node(state: NewsState) -> dict:
        next_agent, reason = route(state, settings.max_iterations)
        log.info("\nSupervisor -> %s", next_agent)

        update: dict = {
            "next_agent": next_agent,
            "messages": [AIMessage(content=f"Supervisor routed to: {next_agent}")],
        }
        if next_agent == FINISH:
            update["stop_reason"] = reason
        return update

    def generator_node(state: NewsState) -> dict:
        log.info("\nGenerator running...")

        reply = writer_llm.invoke(
            [
                SystemMessage(content=prompts.GENERATOR_SYSTEM),
                HumanMessage(content=prompts.GENERATOR_USER.format(topic=state["topic"])),
            ]
        )
        article = extract_article(reply.content, prompts.GENERATOR_MARKER)
        log.info("\nGenerated article:\n\n%s", article)

        return {
            "generated_news": article,
            "evaluation": "pending",
            "messages": [AIMessage(content=f"Generated article:\n{article}")],
        }

    def evaluator_node(state: NewsState) -> dict:
        log.info("\nEvaluator running...")

        article = current_article(state)
        evidence = search(article[: settings.search_query_chars])

        reply = judge_llm.invoke(
            prompts.EVALUATOR.format(
                article=article,
                evidence=evidence or prompts.NO_EVIDENCE,
            )
        )
        verdict, feedback, score = parse_evaluation(reply.content)
        log.info("\nEvaluator output:\n\n%s", reply.content)
        log.info("\nVerdict: %s\nScore: %s", verdict, score)

        return {
            "evaluation": verdict,
            "feedback": feedback,
            "confidence": score,
            "confidence_history": [score],
            "messages": [AIMessage(content=f"Evaluation: {verdict} (score {score})")],
        }

    def stopping_node(state: NewsState) -> dict:
        log.info("\nStopping agent running...")

        reply = judge_llm.invoke(
            prompts.STOPPING.format(
                article=current_article(state)[: settings.article_chars],
                feedback=state.get("feedback", "")[: settings.feedback_chars],
                topic=state["topic"],
                confidence=state.get("confidence", 0.0),
            )
        )
        decision, reason = parse_stop_decision(reply.content)
        log.info("\nDecision: %s\nReason: %s", decision, reason)

        return {
            "stopping_decision": decision,
            "stop_reason": reason,
            "messages": [
                AIMessage(content=f"Stopping decision: {decision}. Reason: {reason}")
            ],
        }

    def rectifier_node(state: NewsState) -> dict:
        log.info("\nRectifier running...")

        sources = search(trusted_sources_query(state["topic"]))
        reply = writer_llm.invoke(
            prompts.RECTIFIER.format(
                topic=state["topic"],
                sources=sources or prompts.NO_EVIDENCE,
                article=current_article(state)[: settings.article_chars],
                feedback=state.get("feedback", "")[: settings.feedback_chars],
            )
        )
        rectified = extract_article(reply.content, prompts.RECTIFIER_MARKER)
        log.info("\nRectified article:\n\n%s", rectified)

        return {
            "rectified_news": rectified,
            # Send the new version back through evaluation.
            "evaluation": "pending",
            "stopping_decision": "pending",
            "iteration": state.get("iteration", 0) + 1,
            "messages": [AIMessage(content=f"Rectified article:\n{rectified}")],
        }

    def finisher_node(state: NewsState) -> dict:
        log.info("\nPipeline complete.")
        return {
            "final_article": current_article(state),
            "verified": state.get("evaluation") == "genuine",
        }

    return {
        "supervisor": supervisor_node,
        "generator": generator_node,
        "evaluator": evaluator_node,
        "stopping_agent": stopping_node,
        "rectifier": rectifier_node,
        "finish": finisher_node,
    }


def route_from_supervisor(state: NewsState) -> str:
    next_agent = state.get("next_agent", FINISH)
    return "finish" if next_agent == FINISH else next_agent
