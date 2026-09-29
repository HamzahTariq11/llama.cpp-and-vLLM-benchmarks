"""Render charts in docs/ from the benchmark JSON files in results/.

Usage:
    uv run python -m serve_llm.plot
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
DOCS_DIR = REPO_ROOT / "docs"

# Palette: single-hue bars on a near-white surface, recessive axes, text in ink tokens.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e6e5e1"
SERIES_1 = "#2a78d6"


def _style_axis(ax: plt.Axes, title: str, ylabel: str) -> None:
    """Apply the shared recessive-axis style to one panel."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)


def _bar_panel(ax: plt.Axes, labels: list[str], values: list[float], fmt: str,
               errors: list[float] | None = None) -> None:
    """Draw one single-series bar panel with small value labels above each bar."""
    bars = ax.bar(labels, values, color=SERIES_1, width=0.6, yerr=errors,
                  error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1, "capsize": 3})
    ax.bar_label(bars, labels=[fmt.format(v) for v in values], padding=3,
                 fontsize=8, color=INK_SECONDARY)
    ax.margins(y=0.15)


def plot_quant_sweep(data: dict, out: Path) -> Path:
    """Four small multiples over quantization level: size, prompt speed, generation speed, PPL."""
    rows = data["rows"]
    labels = [r["quant"] for r in rows]
    s = data["settings"]

    fig, grid = plt.subplots(2, 2, figsize=(12, 7.5), facecolor=SURFACE)
    size_ax, ppl_ax, pp_ax, tg_ax = grid.flat

    _bar_panel(size_ax, labels, [r["file_bytes"] / 1e9 for r in rows], "{:.2f}")
    _style_axis(size_ax, "Model file size", "GB")

    _bar_panel(ppl_ax, labels, [r["ppl"] for r in rows], "{:.2f}",
               errors=[r["ppl_err"] for r in rows])
    _style_axis(ppl_ax, "Perplexity (lower is better)", "PPL, wikitext-2")

    _bar_panel(pp_ax, labels, [r["pp_tps"] for r in rows], "{:.0f}",
               errors=[r["pp_tps_std"] for r in rows])
    _style_axis(pp_ax, f"Prompt processing ({s['pp_tokens']}-token prompt)", "tokens / sec")

    _bar_panel(tg_ax, labels, [r["tg_tps"] for r in rows], "{:.1f}",
               errors=[r["tg_tps_std"] for r in rows])
    _style_axis(tg_ax, f"Generation ({s['tg_tokens']} tokens)", "tokens / sec")

    fig.suptitle(
        f"{data['model']} quantization sweep on {data['hardware']['cpu']}",
        x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold",
    )
    fig.text(
        0.01, 0.005,
        f"llama.cpp {data['llama_cpp']} · {s['threads']} threads · speed: mean ± sd of "
        f"{s['reps']} reps · perplexity: {s['ppl_chunks']} chunks × {s['ppl_ctx']} ctx, "
        f"error bars are llama-perplexity's ± estimate",
        fontsize=8, color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    quant_file = RESULTS_DIR / "quant_sweep.json"
    if quant_file.exists():
        data = json.loads(quant_file.read_text(encoding="utf-8"))
        print(f"Wrote {plot_quant_sweep(data, DOCS_DIR / 'quant_sweep.png')}")


if __name__ == "__main__":
    main()
