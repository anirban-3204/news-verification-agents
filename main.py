"""Command-line entry point.

    python main.py "FIFA"
    python main.py "FIFA" --article-file my_draft.txt --plot outputs/scores.png
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from news_verifier import load_settings, run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate (or take) a news article, fact-check it against "
        "web search, and rewrite it until it passes."
    )
    parser.add_argument("topic", help='Topic of the article, e.g. "FIFA"')
    parser.add_argument(
        "--article-file",
        type=Path,
        help="Verify the article in this text file instead of generating one",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        help="Maximum rectification rounds (default: MAX_ITERATIONS or 8)",
    )
    parser.add_argument("--plot", type=Path, help="Save the score-per-round chart here (PNG)")
    parser.add_argument("--json", type=Path, help="Save the result as JSON here")
    parser.add_argument("--quiet", action="store_true", help="Print only the final result")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv()

    # Model output often contains characters the default Windows encoding
    # cannot write when output is redirected to a file.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
        return 1

    settings = load_settings()
    if args.max_iterations is not None:
        settings = dataclasses.replace(settings, max_iterations=args.max_iterations)

    article = None
    if args.article_file:
        article = args.article_file.read_text(encoding="utf-8")

    result = run(args.topic, article=article, settings=settings)
    history = result.get("confidence_history", [])

    print("\n" + "=" * 60)
    print("FINAL ARTICLE\n")
    print(result["final_article"])
    print("\n" + "-" * 60)
    print(f"Verified by evaluator : {'yes' if result['verified'] else 'NO'}")
    print(f"Final accuracy score  : {result.get('confidence', 0.0):.2f}")
    print(f"Rectification rounds  : {result.get('iteration', 0)}")
    print(f"Score per round       : {', '.join(f'{s:.2f}' for s in history)}")
    print(f"Stopped because       : {result.get('stop_reason', '')}")

    if args.plot:
        from news_verifier.plotting import plot_scores

        print(f"Chart saved to        : {plot_scores(history, args.plot)}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "topic": result["topic"],
            "final_article": result["final_article"],
            "verified": result["verified"],
            "final_score": result.get("confidence", 0.0),
            "score_history": history,
            "rectification_rounds": result.get("iteration", 0),
            "stop_reason": result.get("stop_reason", ""),
            "last_feedback": result.get("feedback", ""),
        }
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON saved to         : {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
