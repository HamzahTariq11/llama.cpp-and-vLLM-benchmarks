"""Measure size, speed and quality for every quantization level of the model.

For each GGUF: file size, prompt-processing and generation speed (llama-bench),
and perplexity on a fixed slice of wikitext-2 (llama-perplexity). Writes one JSON file.

Usage:
    uv run python -m serve_llm.quant_sweep [--quants f16,q4_k_m] [--ppl-chunks 20]
"""

import argparse
import json
import platform
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

# Defaults mirror llamacpp/config.ps1.
REPO_ROOT = Path(__file__).resolve().parents[2]
LLAMA_TAG = "b11254"
MODEL_SLUG = "qwen2.5-1.5b-instruct"
BIN_DIR = REPO_ROOT / "tools" / f"llama.cpp-{LLAMA_TAG}-bin"
GGUF_DIR = REPO_ROOT / "models" / "gguf"
WIKITEXT = REPO_ROOT / "tools" / "wikitext" / "wikitext-2-raw" / "wiki.test.raw"
DEFAULT_QUANTS = ["f16", "q8_0", "q5_k_m", "q4_k_m", "q3_k_m", "q2_k"]

PPL_RE = re.compile(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)")


def parse_bench_json(text: str) -> dict:
    """Extract prompt-processing and generation tokens/sec from `llama-bench -o json` output."""
    result: dict = {}
    for run in json.loads(text):
        kind = "pp" if run["n_prompt"] > 0 else "tg"
        result[f"{kind}_tokens"] = run["n_prompt"] or run["n_gen"]
        result[f"{kind}_tps"] = run["avg_ts"]
        result[f"{kind}_tps_std"] = run["stddev_ts"]
        result.setdefault("cpu_info", run["cpu_info"])
        result.setdefault("build", f"b{run['build_number']} ({run['build_commit']})")
        result.setdefault("n_params", run["model_n_params"])
        result.setdefault("threads", run["n_threads"])
    return result


def parse_perplexity(text: str) -> tuple[float, float]:
    """Return (ppl, ppl_error) from llama-perplexity output."""
    match = PPL_RE.search(text)
    if match is None:
        raise ValueError("No 'Final estimate' line found in llama-perplexity output")
    return float(match.group(1)), float(match.group(2))


def run(cmd: list[str]) -> str:
    """Run a command and return stdout+stderr, raising on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({proc.returncode}):\n{proc.stderr[-2000:]}")
    return proc.stdout + proc.stderr


def measure(gguf: Path, args: argparse.Namespace) -> dict:
    """Benchmark and evaluate one GGUF file."""
    bench_out = run([
        str(BIN_DIR / "llama-bench.exe"), "-m", str(gguf), "-t", str(args.threads),
        "-p", str(args.pp), "-n", str(args.tg), "-r", str(args.reps), "-o", "json",
    ])
    # llama-bench prints the JSON on stdout; logs go to stderr after it.
    bench = parse_bench_json(bench_out[: bench_out.rindex("]") + 1])

    ppl_out = run([
        str(BIN_DIR / "llama-perplexity.exe"), "-m", str(gguf), "-f", str(args.wikitext),
        "-t", str(args.threads), "-c", str(args.ctx), "--chunks", str(args.ppl_chunks),
    ])
    ppl, ppl_err = parse_perplexity(ppl_out)
    return {**bench, "file_bytes": gguf.stat().st_size, "ppl": ppl, "ppl_err": ppl_err}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quants", default=",".join(DEFAULT_QUANTS))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--pp", type=int, default=256, help="prompt tokens for llama-bench")
    parser.add_argument("--tg", type=int, default=64, help="generated tokens for llama-bench")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--ctx", type=int, default=512, help="perplexity context size")
    parser.add_argument("--ppl-chunks", type=int, default=20)
    parser.add_argument("--wikitext", type=Path, default=WIKITEXT)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "quant_sweep.json")
    args = parser.parse_args()

    rows = []
    for quant in args.quants.split(","):
        gguf = GGUF_DIR / f"{MODEL_SLUG}-{quant}.gguf"
        print(f"[{quant}] measuring {gguf.name} ...", flush=True)
        row = {"quant": quant.upper(), **measure(gguf, args)}
        print(
            f"[{quant}] {row['file_bytes'] / 1e9:.2f} GB  pp {row['pp_tps']:.1f} t/s  "
            f"tg {row['tg_tps']:.1f} t/s  PPL {row['ppl']:.3f} ± {row['ppl_err']:.3f}",
            flush=True,
        )
        rows.append(row)

    result = {
        "experiment": "quant_sweep",
        "model": MODEL_SLUG,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "hardware": {"cpu": rows[0]["cpu_info"], "os": platform.platform()},
        "llama_cpp": rows[0]["build"],
        "settings": {
            "threads": args.threads, "pp_tokens": args.pp, "tg_tokens": args.tg,
            "reps": args.reps, "ppl_ctx": args.ctx, "ppl_chunks": args.ppl_chunks,
            "ppl_dataset": "wikitext-2-raw test",
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
