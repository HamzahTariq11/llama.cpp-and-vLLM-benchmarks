"""llama.cpp runtime tuning on one quantized model, measured with llama-bench.

Experiments:
    threads    generation / prompt speed for several thread counts
    attention  flash attention on/off and KV-cache type (f16 vs q8_0), at empty and
               filled context, plus the KV-cache memory each setting needs

Usage:
    uv run python -m serve_llm.runtime_sweep threads
    uv run python -m serve_llm.runtime_sweep attention
"""

import argparse
import json
import platform
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from serve_llm.quant_sweep import BIN_DIR, GGUF_DIR, MODEL_SLUG, REPO_ROOT, run

KV_SIZE_RE = re.compile(r"llama_kv_cache: size =\s*([\d.]+) MiB")

# (label, llama-bench flags) for the attention experiment.
# A quantized V cache requires flash attention, so there is no "FA off · KV q8_0" config.
ATTENTION_CONFIGS = [
    ("FA off · KV f16", {"fa": "off", "ctk": "f16", "ctv": "f16"}),
    ("FA on · KV f16", {"fa": "on", "ctk": "f16", "ctv": "f16"}),
    ("FA on · KV q8_0", {"fa": "on", "ctk": "q8_0", "ctv": "q8_0"}),
]


def parse_bench_rows(text: str) -> list[dict]:
    """Flatten `llama-bench -o json` output into one row per test."""
    rows = []
    for r in json.loads(text):
        is_pp = r["n_prompt"] > 0
        rows.append({
            "test": f"pp{r['n_prompt']}" if is_pp else f"tg{r['n_gen']}",
            "kind": "prompt" if is_pp else "generation",
            "threads": r["n_threads"],
            "flash_attn": r["flash_attn"],
            "type_k": r["type_k"],
            "type_v": r["type_v"],
            "depth": r["n_depth"],
            "tps": r["avg_ts"],
            "tps_std": r["stddev_ts"],
            "cpu_info": r["cpu_info"],
            "build": f"b{r['build_number']} ({r['build_commit']})",
        })
    return rows


def parse_kv_size(text: str) -> float:
    """Return the KV-cache size in MiB from llama-server / llama-cli startup logs."""
    match = KV_SIZE_RE.search(text)
    if match is None:
        raise ValueError("No 'llama_kv_cache: size' line found")
    return float(match.group(1))


def kv_cache_mib(gguf: Path, ctx: int, flags: dict) -> float:
    """Start llama-server just long enough to read the KV-cache size it allocates."""
    cmd = [str(BIN_DIR / "llama-server.exe"), "-m", str(gguf), "-c", str(ctx), "--port", "8099",
           "-fa", flags["fa"], "-ctk", flags["ctk"], "-ctv", flags["ctv"]]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    try:
        for line in proc.stdout:
            if KV_SIZE_RE.search(line):
                return parse_kv_size(line)
        raise RuntimeError("llama-server exited before reporting its KV-cache size")
    finally:
        proc.kill()
        proc.wait()


def bench(gguf: Path, args: argparse.Namespace, extra: list[str]) -> list[dict]:
    """Run llama-bench with the shared pp/tg settings plus `extra` flags."""
    out = run([
        str(BIN_DIR / "llama-bench.exe"), "-m", str(gguf), "-p", str(args.pp), "-n", str(args.tg),
        "-r", str(args.reps), "-o", "json", *extra,
    ])
    return parse_bench_rows(out[: out.rindex("]") + 1])


def threads_experiment(gguf: Path, args: argparse.Namespace) -> list[dict]:
    """One llama-bench call sweeping thread counts (flash attention left at its default)."""
    rows = bench(gguf, args, ["-t", args.threads_list])
    for r in rows:
        r["config"] = f"{r['threads']} threads"
    return rows


def attention_experiment(gguf: Path, args: argparse.Namespace) -> list[dict]:
    """Each attention config at depth 0 and `--depth`, plus its KV-cache size."""
    rows = []
    for label, flags in ATTENTION_CONFIGS:
        print(f"  {label} ...", flush=True)
        kv_mib = kv_cache_mib(gguf, args.kv_ctx, flags)
        extra = ["-t", str(args.threads), "-d", f"0,{args.depth}",
                 "-fa", flags["fa"], "-ctk", flags["ctk"], "-ctv", flags["ctv"]]
        for r in bench(gguf, args, extra):
            rows.append({**r, "config": label, "kv_cache_mib": kv_mib, "kv_ctx": args.kv_ctx})
    return rows


EXPERIMENTS = {"threads": threads_experiment, "attention": attention_experiment}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS))
    parser.add_argument("--quant", default="q5_k_m")
    parser.add_argument("--threads", type=int, default=4, help="threads for non-thread sweeps")
    parser.add_argument("--threads-list", default="1,2,4,6,8")
    parser.add_argument("--pp", type=int, default=256)
    parser.add_argument("--tg", type=int, default=64)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--depth", type=int, default=2048, help="filled-context depth to test")
    parser.add_argument("--kv-ctx", type=int, default=4096, help="context size for KV memory")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    gguf = GGUF_DIR / f"{MODEL_SLUG}-{args.quant}.gguf"
    print(f"[{args.experiment}] {gguf.name}", flush=True)
    rows = EXPERIMENTS[args.experiment](gguf, args)
    for r in rows:
        print(f"  {r['config']:18} {r['test']:6} depth {r['depth']:5}  "
              f"{r['tps']:7.1f} ± {r['tps_std']:.1f} t/s", flush=True)

    result = {
        "experiment": f"runtime_{args.experiment}",
        "model": MODEL_SLUG,
        "quant": args.quant.upper(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "hardware": {"cpu": rows[0]["cpu_info"], "os": platform.platform()},
        "llama_cpp": rows[0]["build"],
        "settings": {"pp_tokens": args.pp, "tg_tokens": args.tg, "reps": args.reps,
                     "threads": args.threads, "depth": args.depth, "kv_ctx": args.kv_ctx},
        "rows": [{k: v for k, v in r.items() if k not in ("cpu_info", "build")} for r in rows],
    }
    out = args.out or REPO_ROOT / "results" / f"runtime_{args.experiment}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
