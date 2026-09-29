from pathlib import Path

from serve_llm.plot import plot_quant_sweep


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
