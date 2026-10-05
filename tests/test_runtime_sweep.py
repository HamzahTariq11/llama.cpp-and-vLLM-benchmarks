from pathlib import Path

import pytest

from serve_llm.runtime_sweep import parse_bench_rows, parse_kv_size

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_kv_size_reads_real_server_log_line() -> None:
    line = ("0.00.906.478 I llama_kv_cache: size =   59.50 MiB (  4096 cells,  28 layers,  "
            "4/1 seqs), K (q8_0):   29.75 MiB, V (q8_0):   29.75 MiB")
    assert parse_kv_size(line) == pytest.approx(59.5)


def test_parse_kv_size_raises_without_size_line() -> None:
    with pytest.raises(ValueError):
        parse_kv_size("llama_server: model loaded")


def test_parse_bench_rows_labels_prompt_and_generation_tests() -> None:
    rows = parse_bench_rows((FIXTURES / "llama_bench.json").read_text(encoding="utf-8"))

    assert [r["test"] for r in rows] == ["pp64", "tg16"]
    assert [r["kind"] for r in rows] == ["prompt", "generation"]
    assert rows[1]["tps"] == pytest.approx(3.087313)
    assert rows[0]["type_k"] == "f16"
