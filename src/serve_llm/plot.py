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
# Categorical slots in fixed order; a series keeps its slot by position in the sorted run list.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


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


LOAD_PANELS = [
    ("throughput_tps", "Aggregate throughput", "output tokens / sec"),
    ("ttft_p95_s", "Time to first token (p95)", "seconds"),
    ("latency_p95_s", "Request latency (p95)", "seconds"),
]


def _runtime_footer(fig: plt.Figure, data: dict, extra: str = "") -> None:
    s = data["settings"]
    fig.text(0.01, 0.01,
             f"{data['model']} {data['quant']} · llama.cpp {data['llama_cpp']} · prompt: "
             f"{s['pp_tokens']} tokens · generation: {s['tg_tokens']} tokens · mean ± sd of "
             f"{s['reps']} reps{extra}",
             fontsize=8, color=MUTED)


def _save(fig: plt.Figure, out: Path, top: float = 0.9) -> Path:
    fig.tight_layout(rect=(0, 0.04, 1, top))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_threads(data: dict, out: Path) -> Path:
    """Two panels over thread count: prompt processing and generation speed."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE)
    for ax, kind, title in zip(axes, ("prompt", "generation"),
                               ("Prompt processing", "Generation"), strict=True):
        rows = sorted((r for r in data["rows"] if r["kind"] == kind), key=lambda r: r["threads"])
        _bar_panel(ax, [str(r["threads"]) for r in rows], [r["tps"] for r in rows], "{:.1f}",
                   errors=[r["tps_std"] for r in rows])
        _style_axis(ax, title, "tokens / sec")
        ax.set_xlabel("threads", color=INK_SECONDARY, fontsize=9)
    fig.suptitle(f"Thread count on {data['hardware']['cpu']}", x=0.01, ha="left",
                 fontsize=12, color=INK, fontweight="bold")
    _runtime_footer(fig, data)
    return _save(fig, out, top=0.92)


def plot_attention(data: dict, out: Path) -> Path:
    """Grouped bars: speed per attention config at empty vs filled context, plus KV memory."""
    rows = data["rows"]
    configs = list(dict.fromkeys(r["config"] for r in rows))  # keep run order
    depths = sorted({r["depth"] for r in rows})
    colors = dict(zip(configs, CATEGORICAL, strict=False))
    width = 0.8 / len(configs)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [2, 2, 1.3]})
    for ax, kind, title in zip(axes[:2], ("prompt", "generation"),
                               ("Prompt processing", "Generation"), strict=True):
        for i, cfg in enumerate(configs):
            by_depth = {r["depth"]: r for r in rows if r["kind"] == kind and r["config"] == cfg}
            xs = [d_i + (i - (len(configs) - 1) / 2) * width for d_i in range(len(depths))]
            vals = [by_depth[d]["tps"] for d in depths]
            bars = ax.bar(xs, vals, width=width * 0.92, color=colors[cfg], label=cfg,
                          yerr=[by_depth[d]["tps_std"] for d in depths],
                          error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1, "capsize": 2})
            ax.bar_label(bars, labels=[f"{v:.1f}" for v in vals], padding=3, fontsize=7.5,
                         color=INK_SECONDARY)
        ax.set_xticks(range(len(depths)),
                      ["empty context" if d == 0 else f"{d}-token context" for d in depths])
        _style_axis(ax, title, "tokens / sec")
        ax.margins(y=0.15)

    kv = {r["config"]: r["kv_cache_mib"] for r in rows}
    bars = axes[2].bar(configs, [kv[c] for c in configs], width=0.6,
                       color=[colors[c] for c in configs])
    axes[2].bar_label(bars, labels=[f"{kv[c]:.1f}" for c in configs], padding=3, fontsize=8,
                      color=INK_SECONDARY)
    axes[2].set_xticks(range(len(configs)), [c.replace(" · ", "\n") for c in configs])
    _style_axis(axes[2], f"KV cache at {data['settings']['kv_ctx']} ctx", "MiB")
    axes[2].margins(y=0.15)

    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=len(configs),
               frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    fig.suptitle("Flash attention and KV-cache quantization", x=0.01, ha="left",
                 fontsize=12, color=INK, fontweight="bold")
    _runtime_footer(fig, data, f" · {data['settings']['threads']} threads")
    return _save(fig, out)


LOAD_TITLES = {
    "load_baseline": "llama.cpp under concurrent load (laptop CPU)",
    "load_parallel_slots": "llama.cpp: 1 vs 4 parallel slots under concurrent load",
}


def plot_load(runs: list[dict], out: Path, title: str) -> Path:
    """Three small multiples over concurrency, one line per benchmark run."""
    if len(runs) > len(CATEGORICAL):
        raise ValueError(f"At most {len(CATEGORICAL)} runs per chart; split into several charts")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), facecolor=SURFACE)
    for ax, (key, panel_title, ylabel) in zip(axes, LOAD_PANELS, strict=True):
        for color, run in zip(CATEGORICAL, runs, strict=False):
            levels = [lv for lv in run["levels"] if lv.get(key) is not None]
            ax.plot([lv["concurrency"] for lv in levels], [lv[key] for lv in levels],
                    color=color, linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=run["label"])
        _style_axis(ax, panel_title, ylabel)
        ax.set_xscale("log", base=2)
        concurrencies = sorted({lv["concurrency"] for r in runs for lv in r["levels"]})
        ax.set_xticks(concurrencies, [str(c) for c in concurrencies])
        ax.minorticks_off()
        ax.set_xlabel("concurrent requests", color=INK_SECONDARY, fontsize=9)
        ax.set_ylim(bottom=0)
        ax.margins(y=0.15)

    if len(runs) > 1:
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=min(len(runs), 4),
                   frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    first = runs[0]
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    s = first["settings"]
    fig.text(0.01, 0.01,
             f"{s['num_prompts']} '{s['prompt_set']}' prompts per level · {s['max_tokens']} output "
             f"tokens each (ignore_eos={s['ignore_eos']}) · temperature 0 · streaming",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def load_runs(results_dir: Path) -> dict[str, list[dict]]:
    """Group load-test result files (experiment "load*") by experiment, sorted by label."""
    groups: dict[str, list[dict]] = {}
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if str(data.get("experiment", "")).startswith("load"):
            groups.setdefault(data["experiment"], []).append(data)
    for experiment, runs in groups.items():
        prompt_setups = {(r["settings"]["prompt_set"], r["settings"]["num_prompts"]) for r in runs}
        if len(prompt_setups) > 1:
            raise ValueError(f"{experiment}: runs use different prompt setups {prompt_setups}")
    return {k: sorted(v, key=lambda r: r["label"]) for k, v in groups.items()}


def main() -> None:
    quant_file = RESULTS_DIR / "quant_sweep.json"
    if quant_file.exists():
        data = json.loads(quant_file.read_text(encoding="utf-8"))
        print(f"Wrote {plot_quant_sweep(data, DOCS_DIR / 'quant_sweep.png')}")

    for name, plotter in (("runtime_threads", plot_threads), ("runtime_attention", plot_attention)):
        path = RESULTS_DIR / f"{name}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            print(f"Wrote {plotter(data, DOCS_DIR / f'{name}.png')}")

    for experiment, runs in load_runs(RESULTS_DIR).items():
        title = LOAD_TITLES.get(experiment, experiment)
        if len(runs) == 1:
            title += f" · {runs[0]['label']}"
        print(f"Wrote {plot_load(runs, DOCS_DIR / f'{experiment}.png', title)}")


if __name__ == "__main__":
    main()
