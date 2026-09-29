"""Load-test any OpenAI-compatible chat endpoint (llama.cpp server, vLLM, ...).

Self-contained: depends only on httpx and the standard library, so it can be copied
into a notebook as-is.

For each concurrency level, every prompt in the chosen set is sent once as a streaming
request, with at most `concurrency` requests in flight. Streaming gives time to first
token (TTFT); `stream_options.include_usage` gives exact token counts.

Usage:
    python run_bench.py --base-url http://127.0.0.1:8080/v1 --model qwen2.5-1.5b-instruct \
        --label llamacpp-q5_k_m --concurrency 1,2,4 --out results/llamacpp-q5_k_m.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

DEFAULT_PROMPTS = Path(__file__).with_name("prompts.jsonl")


@dataclass
class RequestResult:
    """Timing and token counts for one streamed request."""

    id: str
    ok: bool
    ttft_s: float | None = None
    latency_s: float | None = None
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    output_tokens: int | None = None
    decode_tps: float | None = None
    error: str | None = None


# ---------------------------------------------------------------- statistics


def percentile(values: list[float], pct: float) -> float | None:
    """Linear-interpolated percentile (pct in 0..100); None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def decode_tps(output_tokens: int, ttft_s: float, latency_s: float) -> float | None:
    """Per-request generation speed after the first token: (tokens - 1) / decode time."""
    decode_time = latency_s - ttft_s
    if output_tokens < 2 or decode_time <= 0:
        return None
    return (output_tokens - 1) / decode_time


def summarize_level(concurrency: int, results: list[RequestResult], wall_s: float) -> dict:
    """Aggregate per-request results for one concurrency level."""
    ok = [r for r in results if r.ok]
    latencies = [r.latency_s for r in ok if r.latency_s is not None]
    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    speeds = [r.decode_tps for r in ok if r.decode_tps is not None]
    total_out = sum(r.output_tokens or 0 for r in ok)
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "errors": len(results) - len(ok),
        "wall_s": wall_s,
        "total_output_tokens": total_out,
        "throughput_tps": total_out / wall_s if wall_s > 0 else None,
        "requests_per_s": len(ok) / wall_s if wall_s > 0 else None,
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "ttft_p50_s": percentile(ttfts, 50),
        "ttft_p95_s": percentile(ttfts, 95),
        "decode_tps_p50": percentile(speeds, 50),
        "cached_tokens_total": sum(r.cached_tokens or 0 for r in ok),
    }


# ---------------------------------------------------------------- prompts


def load_prompts(path: Path, prompt_set: str, limit: int | None) -> list[dict]:
    """Return [{id, messages}] for the chosen set. The shared-prefix set gets its system prompt."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    system = next((r["prompt"] for r in records if r["set"] == f"{prompt_set}_system"), None)
    items = []
    for r in records:
        if r["set"] != prompt_set:
            continue
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": r["prompt"]})
        items.append({"id": r["id"], "messages": messages})
    if not items:
        raise SystemExit(f"No prompts in set '{prompt_set}' in {path}")
    return items[:limit] if limit else items


# ---------------------------------------------------------------- requests


async def stream_one(client: httpx.AsyncClient, url: str, payload: dict, rid: str) -> RequestResult:
    """Send one streaming chat request and time it."""
    start = time.perf_counter()
    ttft = None
    usage: dict = {}
    try:
        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")[:300]
                return RequestResult(rid, ok=False, error=f"HTTP {resp.status_code}: {body}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    if ttft is None and choice.get("delta", {}).get("content"):
                        ttft = time.perf_counter() - start
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        return RequestResult(rid, ok=False, error=f"{type(exc).__name__}: {exc}"[:300])

    latency = time.perf_counter() - start
    out_tokens = usage.get("completion_tokens")
    if ttft is None or out_tokens is None:
        return RequestResult(rid, ok=False, latency_s=latency, error="no content or no usage")
    return RequestResult(
        rid, ok=True, ttft_s=ttft, latency_s=latency,
        prompt_tokens=usage.get("prompt_tokens"),
        cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        output_tokens=out_tokens,
        decode_tps=decode_tps(out_tokens, ttft, latency),
    )


def bust_cache(messages: list[dict]) -> list[dict]:
    """Prefix the user message with a unique tag so no request reuses an earlier full prompt.

    Every concurrency level re-sends the same prompts, and both llama.cpp and vLLM cache
    prompt prefixes; without this, later levels would get artificially fast TTFT. The tag
    goes on the user message, so a shared system prompt stays cacheable (that is the
    behavior the shared-prefix experiment measures).
    """
    tagged = [dict(m) for m in messages]
    user = next(m for m in reversed(tagged) if m["role"] == "user")
    user["content"] = f"[request {uuid.uuid4().hex[:8]}] {user['content']}"
    return tagged


async def run_level(args: argparse.Namespace, prompts: list[dict], concurrency: int) -> dict:
    """Send every prompt once with at most `concurrency` in flight; return the level summary."""
    url = f"{args.base_url.rstrip('/')}/chat/completions"
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency + 1)

    async with httpx.AsyncClient(timeout=args.timeout, limits=limits, headers=_auth()) as client:

        async def worker(item: dict) -> RequestResult:
            messages = bust_cache(item["messages"]) if args.cache_bust else item["messages"]
            async with sem:
                return await stream_one(client, url, _payload(args, messages), item["id"])

        start = time.perf_counter()
        results = await asyncio.gather(*(worker(p) for p in prompts))
        wall = time.perf_counter() - start

    summary = summarize_level(concurrency, list(results), wall)
    summary["per_request"] = [asdict(r) for r in results]
    return summary


def _payload(args: argparse.Namespace, messages: list[dict]) -> dict:
    payload = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.ignore_eos:
        # Supported by both llama.cpp server and vLLM: every request generates max_tokens.
        payload["ignore_eos"] = True
    return payload


def _auth() -> dict:
    key = os.environ.get("OPENAI_API_KEY")
    return {"Authorization": f"Bearer {key}"} if key else {}


# ---------------------------------------------------------------- environment


def hardware_info() -> dict:
    """Best-effort CPU, RAM and GPU description using only the standard library."""
    info = {"os": platform.platform(), "python": platform.python_version(),
            "cpu": platform.processor() or platform.machine(), "cpu_threads": os.cpu_count()}
    if sys.platform == "win32":
        try:
            import ctypes
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            info["cpu"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong)] + [
                            (f"_{i}", ctypes.c_ulonglong) for i in range(6)]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            info["ram_gb"] = round(status.ullTotalPhys / 1024**3, 1)
        except OSError:
            pass
    elif Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                info["ram_gb"] = round(int(line.split()[1]) / 1024**2, 1)
                break
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            info["gpus"] = [line.strip() for line in out.splitlines() if line.strip()]
        except (OSError, subprocess.SubprocessError):
            pass
    return info


def server_models(base_url: str) -> object:
    """Whatever the server reports at /models (records server-side model metadata)."""
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/models", headers=_auth(), timeout=10)
        return resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------- main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8080/v1")
    p.add_argument("--model", required=True, help="model name the server accepts")
    p.add_argument("--label", required=True, help="name for this run, e.g. llamacpp-q5_k_m")
    p.add_argument("--concurrency", default="1,2,4", help="comma-separated levels")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    p.add_argument("--prompt-set", default="varied", help="varied | shared_prefix")
    p.add_argument("--limit", type=int, help="use only the first N prompts of the set")
    p.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false",
                   help="let the model stop early (output lengths then vary per backend)")
    p.add_argument("--no-cache-bust", dest="cache_bust", action="store_false",
                   help="send prompts verbatim (repeat runs may then hit the prompt cache)")
    p.add_argument("--no-warmup", dest="warmup", action="store_false")
    p.add_argument("--server-config", default="{}",
                   help='JSON describing server settings, e.g. \'{"quant": "Q5_K_M"}\'')
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


async def main_async(args: argparse.Namespace) -> dict:
    prompts = load_prompts(args.prompts, args.prompt_set, args.limit)
    levels = [int(c) for c in args.concurrency.split(",")]

    if args.warmup:
        # One short request so model load / first-call overheads don't land in level 1.
        async with httpx.AsyncClient(timeout=args.timeout, headers=_auth()) as client:
            url = f"{args.base_url.rstrip('/')}/chat/completions"
            warm = dict(_payload(args, prompts[0]["messages"]), max_tokens=8)
            await stream_one(client, url, warm, "warmup")

    runs = []
    for c in levels:
        print(f"[{args.label}] concurrency {c}: {len(prompts)} requests ...", flush=True)
        level = await run_level(args, prompts, c)
        print(
            f"[{args.label}] c={c}  throughput {level['throughput_tps'] or 0:.1f} tok/s  "
            f"p50 TTFT {level['ttft_p50_s'] or 0:.2f}s  "
            f"p95 latency {level['latency_p95_s'] or 0:.1f}s  errors {level['errors']}",
            flush=True,
        )
        runs.append(level)

    return {
        "experiment": "load",
        "label": args.label,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_url": args.base_url,
        "model": args.model,
        "server_models": server_models(args.base_url),
        "server_config": json.loads(args.server_config),
        "hardware": hardware_info(),
        "settings": {
            "prompt_set": args.prompt_set, "num_prompts": len(prompts),
            "max_tokens": args.max_tokens, "ignore_eos": args.ignore_eos, "temperature": 0,
            "cache_bust": args.cache_bust,
        },
        "levels": runs,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = asyncio.run(main_async(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
