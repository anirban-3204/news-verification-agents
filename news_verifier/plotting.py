"""Plot how the accuracy score moved across evaluation rounds."""

from __future__ import annotations

from pathlib import Path


def plot_scores(history: list[float], path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")  # write a file; no display needed
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(history)), history, marker="o", linewidth=2, markersize=8)
    ax.axhline(y=1.0, color="g", linestyle="--", alpha=0.4, label="Perfect")
    ax.set_xlabel("Evaluation round")
    ax.set_ylabel("Factual accuracy score")
    ax.set_title("Article factual accuracy over iterations")
    ax.set_xticks(range(len(history)))
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
