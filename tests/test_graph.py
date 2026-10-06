import pytest

from conftest import RepeatingJudge, RepeatingWriter, ScriptedModel
from news_verifier.nodes import FINISH, route

NEEDS_FIX = "EVALUATION: needs_correction\n\nFEEDBACK:\nThe match count is wrong.\n\nCONFIDENCE:\n0.4"
GENUINE = "EVALUATION: genuine\n\nFEEDBACK:\nAll claims are supported.\n\nCONFIDENCE:\n0.9"
CONTINUE = "DECISION: continue\n\nSTOPPING REASON:\nA factual error remains."
STOP = "DECISION: stop\n\nSTOPPING REASON:\nOnly minor wording issues remain."


# --- supervisor routing ----------------------------------------------------

@pytest.mark.parametrize(
    "state, expected",
    [
        ({}, "generator"),
        ({"generated_news": "x", "evaluation": "pending"}, "evaluator"),
        ({"generated_news": "x", "evaluation": "genuine"}, FINISH),
        (
            {"generated_news": "x", "evaluation": "needs_correction", "stopping_decision": "pending"},
            "stopping_agent",
        ),
        (
            {"generated_news": "x", "evaluation": "needs_correction", "stopping_decision": "continue"},
            "rectifier",
        ),
        (
            {"generated_news": "x", "evaluation": "needs_correction", "stopping_decision": "stop"},
            FINISH,
        ),
        (
            {
                "generated_news": "x",
                "evaluation": "needs_correction",
                "stopping_decision": "pending",
                "iteration": 3,
            },
            FINISH,
        ),
    ],
)
def test_route(state, expected):
    assert route(state, max_iterations=3)[0] == expected


# --- whole graph, offline --------------------------------------------------

def test_one_correction_then_accepted(run_graph):
    writer = ScriptedModel(
        ["GENERATED_ARTICLE:\nThe World Cup has 80 matches.", "RECTIFIED_ARTICLE:\nThe World Cup has 104 matches."]
    )
    judge = ScriptedModel([NEEDS_FIX, CONTINUE, GENUINE])

    result = run_graph(writer, judge)

    assert result["final_article"] == "The World Cup has 104 matches."
    assert result["verified"] is True
    assert result["iteration"] == 1
    assert result["confidence_history"] == [0.4, 0.9]
    # The rectifier was shown the evaluator's feedback and the previous draft.
    assert "The match count is wrong." in writer.prompts[1]
    assert "The World Cup has 80 matches." in writer.prompts[1]
    # The second evaluation judged the corrected draft, not the original.
    assert "104 matches" in judge.prompts[2]


def test_accepted_first_time_needs_no_correction(run_graph):
    writer = ScriptedModel(["GENERATED_ARTICLE:\nA correct article."])
    judge = ScriptedModel([GENUINE])

    result = run_graph(writer, judge)

    assert result["verified"] is True
    assert result["iteration"] == 0
    assert result["final_article"] == "A correct article."


def test_loop_stops_at_the_iteration_limit(run_graph):
    writer, judge = RepeatingWriter(), RepeatingJudge()

    result = run_graph(writer, judge, max_iterations=3)

    assert result["iteration"] == 3
    assert result["verified"] is False
    assert len(result["confidence_history"]) == 4  # first draft + 3 corrections
    assert "limit of 3" in result["stop_reason"]
    assert writer.calls == 4  # 1 draft + 3 corrections
    assert judge.calls == 7  # 4 evaluations + 3 stop checks


def test_stopping_agent_can_end_the_loop_unverified(run_graph):
    writer = ScriptedModel(["GENERATED_ARTICLE:\nMostly right."])
    judge = ScriptedModel([NEEDS_FIX, STOP])

    result = run_graph(writer, judge)

    assert result["verified"] is False
    assert result["iteration"] == 0
    assert result["stop_reason"] == "Only minor wording issues remain."


def test_supplied_article_skips_the_generator(run_graph):
    writer = ScriptedModel([])  # any writer call would fail the test
    judge = ScriptedModel([GENUINE])

    result = run_graph(writer, judge, article="My own article.")

    assert result["final_article"] == "My own article."
    assert "My own article." in judge.prompts[0]


def test_failed_search_does_not_crash_the_run(run_graph):
    writer = ScriptedModel(["GENERATED_ARTICLE:\nAn article."])
    judge = ScriptedModel([GENUINE])

    result = run_graph(writer, judge, search=lambda query: "")

    assert result["verified"] is True
    assert "No web evidence could be retrieved" in judge.prompts[0]
