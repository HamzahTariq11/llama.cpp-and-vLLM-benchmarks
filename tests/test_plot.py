import json
from pathlib import Path

from serve_llm.plot import load_runs, plot_load, plot_quant_sweep


def _load_run(label: str, prompt_set: str = "varied") -> dict:
    levels = [{"concurrency": c, "throughput_tps": 10.0 * c, "ttft_p95_s": 0.2 * c,
               "latency_p95_s": 12.0 + c} for c in (1, 2, 4)]
    return {"experiment": "load", "label": label, "levels": levels,
            "settings": {"prompt_set": prompt_set, "num_prompts": 40, "max_tokens": 256,
                         "ignore_eos": True}}


def test_plot_load_writes_png_for_several_runs(tmp_path: Path) -> None:
    out = plot_load([_load_run("a"), _load_run("b")], tmp_path / "load.png", "Test")
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_load_runs_groups_by_prompt_set_and_ignores_other_experiments(tmp_path: Path) -> None:
    files = {"b.json": _load_run("b"), "a.json": _load_run("a"),
             "s.json": _load_run("s", "shared_prefix"), "q.json": {"experiment": "quant_sweep"}}
    for name, data in files.items():
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")

    groups = load_runs(tmp_path)

    assert set(groups) == {"varied", "shared_prefix"}
    assert [r["label"] for r in groups["varied"]] == ["a", "b"]


def test_plot_quant_sweep_writes_png(tmp_path: Path) -> None:
    row = {"file_bytes": 1e9, "pp_tps": 100.0, "pp_tps_std": 1.0,
           "tg_tps": 20.0, "tg_tps_std": 0.5, "ppl": 11.0, "ppl_err": 0.4}
    data = {
        "model": "test-model",
        "hardware": {"cpu": "Test CPU"},
        "llama_cpp": "b0 (abc)",
        "settings": {"threads": 4, "pp_tokens": 256, "tg_tokens": 64, "reps": 3,
                     "ppl_ctx": 512, "ppl_chunks": 20},
        "rows": [{"quant": "F16", **row}, {"quant": "Q4_K_M", **row}],
    }

    out = plot_quant_sweep(data, tmp_path / "chart.png")

    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
