import pytest

from news_verifier.parsing import extract_article, parse_evaluation, parse_stop_decision
from news_verifier.prompts import GENERATOR_MARKER, RECTIFIER_MARKER


# --- extract_article -------------------------------------------------------

def test_article_after_marker_is_returned():
    raw = "RECTIFIED_ARTICLE:\nFirst paragraph.\n\nSecond paragraph."
    assert extract_article(raw, RECTIFIER_MARKER) == "First paragraph.\n\nSecond paragraph."


def test_reply_without_marker_is_used_whole():
    assert extract_article("  Just the article.  ", GENERATOR_MARKER) == "Just the article."


def test_body_is_kept_when_only_a_headline_follows_the_marker():
    # Seen in real runs: the body comes first and the marker carries just a title.
    body = "FIFA said on Tuesday that it had joined a UN climate programme. " * 3
    raw = f"{body}\n\nGENERATED_ARTICLE: FIFA Partners with UN"
    article = extract_article(raw, GENERATOR_MARKER)
    assert article.startswith("FIFA Partners with UN")
    assert body.strip() in article


def test_short_preamble_does_not_replace_the_article():
    raw = "Here is the corrected version.\n\nRECTIFIED_ARTICLE: FIFA and FIFPRO signed an agreement."
    assert extract_article(raw, RECTIFIER_MARKER) == "FIFA and FIFPRO signed an agreement."


# --- parse_evaluation ------------------------------------------------------

def test_well_formed_evaluation():
    raw = "EVALUATION: genuine\n\nFEEDBACK:\nAll claims check out.\n\nCONFIDENCE:\n0.85"
    assert parse_evaluation(raw) == ("genuine", "All claims check out.", 0.85)


@pytest.mark.parametrize(
    "tail, expected",
    [
        ("0.62 (factual accuracy score)", 0.62),
        ("0.95(factual accuracy score)", 0.95),
        ("0.38 (the article is largely inaccurate, score 0.5 would be too kind)", 0.38),
        ("**0.4**", 0.4),
        ("1", 1.0),
        (".7", 0.7),
    ],
)
def test_score_is_read_despite_trailing_text(tail, expected):
    raw = f"EVALUATION: needs_correction\n\nFEEDBACK:\nSome issues.\n\nCONFIDENCE:\n{tail}"
    verdict, feedback, score = parse_evaluation(raw)
    assert verdict == "needs_correction"
    assert feedback == "Some issues."
    assert score == expected


def test_percentage_score_is_converted():
    assert parse_evaluation("EVALUATION: genuine\nCONFIDENCE: 85")[2] == 0.85


def test_unreadable_evaluation_never_passes_as_genuine():
    verdict, feedback, score = parse_evaluation("I think this looks fine overall.")
    assert verdict == "needs_correction"
    assert score == 0.0
    assert feedback == "I think this looks fine overall."


def test_markdown_bold_verdict_is_read():
    assert parse_evaluation("**EVALUATION:** genuine\nCONFIDENCE: 0.9")[0] == "genuine"


# --- parse_stop_decision ---------------------------------------------------

def test_stop_decision_and_reason():
    raw = "DECISION: stop\n\nSTOPPING REASON:\nNo major issues remain."
    assert parse_stop_decision(raw) == ("stop", "No major issues remain.")


def test_stop_decision_defaults_to_continue():
    assert parse_stop_decision("Not sure.") == ("continue", "No reason provided")


def test_stop_decision_ignores_case_and_punctuation():
    assert parse_stop_decision("DECISION: **Continue.**")[0] == "continue"
