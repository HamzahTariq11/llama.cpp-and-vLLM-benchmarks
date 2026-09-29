import json
from pathlib import Path

import pytest

from serve_llm.quant_sweep import parse_bench_json, parse_perplexity

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_bench_json_splits_prompt_and_generation_runs() -> None:
    result = parse_bench_json((FIXTURES / "llama_bench.json").read_text(encoding="utf-8"))

    assert result["pp_tokens"] == 64
    assert result["pp_tps"] == pytest.approx(74.996092)
    assert result["tg_tokens"] == 16
    assert result["tg_tps"] == pytest.approx(3.087313)
    assert result["tg_tps_std"] == 0.0
    assert result["cpu_info"].startswith("11th Gen Intel")
    assert result["build"] == "b11254 (8019dc563)"
    assert result["threads"] == 4


def test_parse_bench_json_is_order_independent() -> None:
    runs = json.loads((FIXTURES / "llama_bench.json").read_text(encoding="utf-8"))
    reversed_result = parse_bench_json(json.dumps(runs[::-1]))

    assert reversed_result["pp_tps"] == pytest.approx(74.996092)
    assert reversed_result["tg_tps"] == pytest.approx(3.087313)


def test_parse_perplexity_reads_final_estimate() -> None:
    ppl, err = parse_perplexity((FIXTURES / "llama_perplexity.txt").read_text(encoding="utf-8"))

    assert ppl > 1
    assert 0 < err < ppl


def test_parse_perplexity_raises_without_final_estimate() -> None:
    with pytest.raises(ValueError):
        parse_perplexity("[1]10.2,[2]11.3,")
