"""Turn the models' free-text replies into structured values.

The models are asked for a strict format but do not always follow it, so every
parser here has a safe default instead of raising.
"""

from __future__ import annotations

import re

_EVALUATION_RE = re.compile(r"EVALUATION:[\s*_]*(genuine|needs_correction)", re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r"CONFIDENCE:[\s*_]*(\d*\.?\d+)", re.IGNORECASE)
_DECISION_RE = re.compile(r"DECISION:[\s*_]*(stop|continue)", re.IGNORECASE)

# A reply after the marker that is this short and on one line is treated as a headline.
_HEADLINE_MAX_CHARS = 200


def extract_article(raw: str, marker: str) -> str:
    """Return the article from a reply that tags it with `marker`.

    Normally the article is everything after the marker. Sometimes a model
    writes the body first and then puts only a headline after the marker; in
    that case the headline and the body are joined, so the body is not lost.
    """
    if marker not in raw:
        return raw.strip()

    before, _, after = raw.rpartition(marker)
    before, after = before.strip(), after.strip()

    only_headline = "\n" not in after and len(after) <= _HEADLINE_MAX_CHARS
    if before and only_headline and len(before) > len(after):
        return f"{after}\n\n{before}" if after else before
    return after or before


def parse_evaluation(raw: str) -> tuple[str, str, float]:
    """Return (verdict, feedback, score) from the evaluator's reply.

    Defaults to ("needs_correction", <whole reply>, 0.0) so that an unreadable
    reply never lets an article through as genuine.
    """
    verdict = "needs_correction"
    match = _EVALUATION_RE.search(raw)
    if match:
        verdict = match.group(1).lower()

    feedback = raw.strip()
    if "FEEDBACK:" in raw:
        feedback = raw.split("FEEDBACK:", 1)[1]
        feedback = re.split(r"CONFIDENCE:", feedback, flags=re.IGNORECASE)[0].strip()

    score = 0.0
    scores = _CONFIDENCE_RE.findall(raw)
    if scores:
        score = float(scores[-1])
        if 1.0 < score <= 100.0:  # written as a percentage, e.g. "85"
            score /= 100.0
        score = min(max(score, 0.0), 1.0)

    return verdict, feedback, score


def parse_stop_decision(raw: str) -> tuple[str, str]:
    """Return (decision, reason) from the stopping agent's reply.

    Defaults to "continue"; the supervisor's iteration cap bounds the loop.
    """
    decision = "continue"
    match = _DECISION_RE.search(raw)
    if match:
        decision = match.group(1).lower()

    reason = "No reason provided"
    if "STOPPING REASON:" in raw:
        reason = raw.split("STOPPING REASON:", 1)[1].strip() or reason

    return decision, reason
