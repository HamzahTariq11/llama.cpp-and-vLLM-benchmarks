import json
from pathlib import Path

import pytest

from serve_llm.plot import load_runs, plot_attention, plot_load, plot_quant_sweep, plot_threads


def _load_run(label: str, experiment: str = "load_x", num_prompts: int = 40) -> dict:
    levels = [{"concurrency": c, "throughput_tps": 10.0 * c, "ttft_p95_s": 0.2 * c,
               "latency_p95_s": 12.0 + c} for c in (1, 2, 4)]
    return {"experiment": experiment, "label": label, "levels": levels,
            "settings": {"prompt_set": "varied", "num_prompts": num_prompts, "max_tokens": 256,
                         "ignore_eos": True}}


def _write(tmp_path: Path, files: dict) -> None:
    for name, data in files.items():
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")


def test_plot_load_writes_png_for_several_runs(tmp_path: Path) -> None:
    out = plot_load([_load_run("a"), _load_run("b")], tmp_path / "load.png", "Test")
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_load_runs_groups_by_experiment_and_ignores_others(tmp_path: Path) -> None:
    _write(tmp_path, {"b.json": _load_run("b"), "a.json": _load_run("a"),
                      "s.json": _load_run("s", "load_y"), "q.json": {"experiment": "quant_sweep"}})

    groups = load_runs(tmp_path)

    assert set(groups) == {"load_x", "load_y"}
    assert [r["label"] for r in groups["load_x"]] == ["a", "b"]


def test_load_runs_rejects_mixed_prompt_setups_in_one_chart(tmp_path: Path) -> None:
    _write(tmp_path, {"a.json": _load_run("a"), "b.json": _load_run("b", num_prompts=16)})
    with pytest.raises(ValueError):
        load_runs(tmp_path)


def _runtime(rows: list[dict]) -> dict:
    return {"model": "m", "quant": "Q5_K_M", "hardware": {"cpu": "Test CPU"},
            "llama_cpp": "b0 (abc)",
            "settings": {"pp_tokens": 256, "tg_tokens": 64, "reps": 3, "threads": 4,
                         "depth": 2048, "kv_ctx": 4096},
            "rows": rows}


def test_plot_threads_writes_png(tmp_path: Path) -> None:
    rows = [{"config": f"{t} threads", "threads": t, "kind": kind, "depth": 0,
             "tps": 10.0 * t, "tps_std": 0.5}
            for t in (1, 2, 4) for kind in ("prompt", "generation")]
    out = plot_threads(_runtime(rows), tmp_path / "threads.png")
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_attention_writes_png(tmp_path: Path) -> None:
    rows = [{"config": cfg, "kind": kind, "depth": depth, "tps": 20.0, "tps_std": 0.5,
             "kv_cache_mib": 112.0 if "f16" in cfg else 59.5}
            for cfg in ("FA off · KV f16", "FA on · KV f16", "FA on · KV q8_0")
            for kind in ("prompt", "generation") for depth in (0, 2048)]
    out = plot_attention(_runtime(rows), tmp_path / "attention.png")
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


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
