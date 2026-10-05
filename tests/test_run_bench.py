import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

BENCH = Path(__file__).resolve().parents[1] / "bench"

# run_bench.py is a standalone script (copied into the Kaggle notebook), not a package module.
_spec = importlib.util.spec_from_file_location("run_bench", BENCH / "run_bench.py")
rb = importlib.util.module_from_spec(_spec)
sys.modules["run_bench"] = rb  # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(rb)


def test_percentile_interpolates() -> None:
    values = [4.0, 1.0, 3.0, 2.0, 5.0]
    assert rb.percentile(values, 0) == 1.0
    assert rb.percentile(values, 50) == 3.0
    assert rb.percentile(values, 100) == 5.0
    assert rb.percentile(values, 95) == pytest.approx(4.8)
    assert rb.percentile([7.0], 95) == 7.0
    assert rb.percentile([], 50) is None


def test_decode_tps_excludes_first_token() -> None:
    # 101 tokens: first arrives at 1s, remaining 100 over the next 4s.
    assert rb.decode_tps(101, ttft_s=1.0, latency_s=5.0) == pytest.approx(25.0)
    assert rb.decode_tps(1, ttft_s=1.0, latency_s=1.0) is None


def test_summarize_level_on_fake_timings() -> None:
    results = [
        rb.RequestResult("a", ok=True, ttft_s=0.5, latency_s=10.0, output_tokens=100,
                         cached_tokens=0, decode_tps=10.4),
        rb.RequestResult("b", ok=True, ttft_s=1.5, latency_s=12.0, output_tokens=100,
                         cached_tokens=40, decode_tps=9.4),
        rb.RequestResult("c", ok=False, error="HTTP 500"),
    ]
    s = rb.summarize_level(2, results, wall_s=20.0)

    assert s["requests"] == 3
    assert s["errors"] == 1
    assert s["total_output_tokens"] == 200
    assert s["throughput_tps"] == pytest.approx(10.0)
    assert s["requests_per_s"] == pytest.approx(0.1)
    assert s["latency_p50_s"] == pytest.approx(11.0)
    assert s["ttft_p50_s"] == pytest.approx(1.0)
    assert s["ttft_p95_s"] == pytest.approx(1.45)
    assert s["cached_tokens_total"] == 40


def test_load_prompts_attaches_shared_system_prompt() -> None:
    varied = rb.load_prompts(BENCH / "prompts.jsonl", "varied", limit=None)
    shared = rb.load_prompts(BENCH / "prompts.jsonl", "shared_prefix", limit=3)

    assert len(varied) >= 40
    assert all(p["messages"][0]["role"] == "user" for p in varied)
    assert len(shared) == 3
    system_prompts = {p["messages"][0]["content"] for p in shared}
    assert len(system_prompts) == 1  # every request shares the exact same prefix
    assert shared[0]["messages"][0]["role"] == "system"


def test_load_prompts_repeat_makes_distinct_ids() -> None:
    items = rb.load_prompts(BENCH / "prompts.jsonl", "shared_prefix", limit=2, repeat=3)

    assert len(items) == 6
    assert len({it["id"] for it in items}) == 6
    assert items[0]["id"].endswith("#1") and items[-1]["id"].endswith("#3")


def test_bust_cache_tags_user_message_only() -> None:
    messages = [{"role": "system", "content": "shared"}, {"role": "user", "content": "q"}]
    first, second = rb.bust_cache(messages), rb.bust_cache(messages)

    assert first[0] == second[0] == {"role": "system", "content": "shared"}
    assert first[1]["content"].endswith("] q")
    assert first[1]["content"] != second[1]["content"]
    assert messages[1]["content"] == "q"  # input not mutated


def _sse(*chunks: dict) -> bytes:
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks] + ["data: [DONE]\n\n"]
    return "".join(lines).encode()


def test_stream_one_reads_ttft_and_usage() -> None:
    body = _sse(
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [], "usage": {"completion_tokens": 2, "prompt_tokens": 9,
                                  "prompt_tokens_details": {"cached_tokens": 4}}},
    )
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=body))

    async def go() -> rb.RequestResult:
        async with httpx.AsyncClient(transport=transport) as client:
            return await rb.stream_one(client, "http://test/v1/chat/completions", {}, "x")

    result = asyncio.run(go())
    assert result.ok
    assert result.output_tokens == 2
    assert result.prompt_tokens == 9
    assert result.cached_tokens == 4
    assert 0 <= result.ttft_s <= result.latency_s


def test_stream_one_reports_http_errors() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(503, text="overloaded"))

    async def go() -> rb.RequestResult:
        async with httpx.AsyncClient(transport=transport) as client:
            return await rb.stream_one(client, "http://test/v1/chat/completions", {}, "x")

    result = asyncio.run(go())
    assert not result.ok
    assert "503" in result.error
