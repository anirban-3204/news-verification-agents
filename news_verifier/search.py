"""Web search used as evidence by the evaluator and the rectifier."""

from __future__ import annotations

import logging

log = logging.getLogger("news_verifier")

TRUSTED_SITES = ("reuters.com", "bbc.com", "apnews.com")


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return numbered results as plain text.

    Returns an empty string if the search fails, so a network problem
    degrades the run instead of crashing it.
    """
    from ddgs import DDGS

    try:
        hits = DDGS().text(query, max_results=max_results)
    except Exception as exc:  # ddgs raises several unrelated error types
        log.warning("Search failed (%s): %s", type(exc).__name__, exc)
        return ""

    lines = []
    for number, hit in enumerate(hits, start=1):
        lines.append(f"[{number}] {hit.get('title', '')} ({hit.get('href', '')})")
        lines.append(f"    {hit.get('body', '')}")
    return "\n".join(lines)


def trusted_sources_query(topic: str) -> str:
    """Query that restricts results to a few established news outlets."""
    sites = " OR ".join(f"site:{site}" for site in TRUSTED_SITES)
    return f"latest news about {topic} {sites}"
