"""Run every laptop (llama.cpp) experiment in one session, with drift checks in between.

Before the first step and after every step, a short reference measurement (llama-bench,
Q5_K_M, 4 threads) is appended to results/reference_checks.json. If the machine's state
changes mid-session (background load, thermals, power mode), it shows up there.

Steps: quant, threads, baseline, parallel1, parallel4, attention

Usage:
    uv run python -m serve_llm.laptop_suite                  # everything (~1.5 h)
    uv run python -m serve_llm.laptop_suite --steps parallel1,parallel4
"""

import argparse
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime

import httpx

from serve_llm.quant_sweep import BIN_DIR, GGUF_DIR, MODEL_SLUG, REPO_ROOT, parse_bench_json, run

RESULTS = REPO_ROOT / "results"
LOG_DIR = REPO_ROOT / "tools" / "logs"
REFERENCE_FILE = RESULTS / "reference_checks.json"
QUANT = "q5_k_m"
GGUF = GGUF_DIR / f"{MODEL_SLUG}-{QUANT}.gguf"
PORT = 8080
BASE_URL = f"http://127.0.0.1:{PORT}/v1"
CTX = 4096


def reference_check(after: str) -> dict:
    """Standard llama-bench measurement; appended to the session's drift log."""
    out = run([str(BIN_DIR / "llama-bench.exe"), "-m", str(GGUF), "-t", "4",
               "-p", "256", "-n", "64", "-r", "3", "-o", "json"])
    bench = parse_bench_json(out[: out.rindex("]") + 1])
    entry = {
        "after": after,
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "pp_tps": bench["pp_tps"], "pp_tps_std": bench["pp_tps_std"],
        "tg_tps": bench["tg_tps"], "tg_tps_std": bench["tg_tps_std"],
    }
    log = json.loads(REFERENCE_FILE.read_text(encoding="utf-8")) if REFERENCE_FILE.exists() else {
        "description": "Q5_K_M llama-bench, 4 threads, pp256/tg64, 3 reps; run between steps",
        "checks": [],
    }
    log["checks"].append(entry)
    REFERENCE_FILE.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"  reference after {after}: pp {entry['pp_tps']:.1f} ± {entry['pp_tps_std']:.1f}  "
          f"tg {entry['tg_tps']:.1f} ± {entry['tg_tps_std']:.1f} t/s", flush=True)
    return entry


@contextmanager
def llama_server(name: str, extra: list[str]):
    """Start llama-server on the Q5_K_M model, wait for /health, stop it on exit."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = (LOG_DIR / f"server-{name}.log").open("w", encoding="utf-8")
    cmd = [str(BIN_DIR / "llama-server.exe"), "-m", str(GGUF), "--alias", MODEL_SLUG,
           "--port", str(PORT), "-c", str(CTX), *extra]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 120
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early; see {log.name}")
            try:
                if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("llama-server did not become healthy within 120 s")
            time.sleep(1)
        yield
    finally:
        proc.terminate()
        proc.wait(timeout=30)
        log.close()


def load_test(label: str, experiment: str, server_config: dict, extra_args: list[str]) -> None:
    """Run bench/run_bench.py against the running server."""
    cmd = [sys.executable, str(REPO_ROOT / "bench" / "run_bench.py"),
           "--base-url", BASE_URL, "--model", MODEL_SLUG, "--label", label,
           "--experiment", experiment, "--concurrency", "1,2,4", "--max-tokens", "256",
           "--server-config", json.dumps(server_config),
           "--out", str(RESULTS / f"{label}.json"), *extra_args]
    subprocess.run(cmd, check=True)


def _server_config(**kw: object) -> dict:
    return {"engine": "llama.cpp", "build": BIN_DIR.name.split("-")[1], "quant": QUANT.upper(),
            "threads": 4, "ctx_size": CTX, **kw}


def step_quant() -> None:
    subprocess.run([sys.executable, "-m", "serve_llm.quant_sweep"], check=True)


def step_threads() -> None:
    subprocess.run([sys.executable, "-m", "serve_llm.runtime_sweep", "threads"], check=True)


def step_attention() -> None:
    subprocess.run([sys.executable, "-m", "serve_llm.runtime_sweep", "attention"], check=True)


def step_baseline() -> None:
    with llama_server("baseline", []):
        load_test("llamacpp-q5_k_m-default", "load_baseline",
                  _server_config(parallel_slots=4, flags="defaults"), [])


def _parallel(slots: int) -> None:
    with llama_server(f"parallel{slots}", ["--parallel", str(slots)]):
        load_test(f"llamacpp-q5_k_m-parallel{slots}", "load_parallel_slots",
                  _server_config(parallel_slots=slots), ["--limit", "16"])


def step_parallel1() -> None:
    _parallel(1)


def step_parallel4() -> None:
    _parallel(4)


# Order matters: attention is measured after the load tests so every attention config
# runs on an already-warm (thermally steady) machine.
STEPS = {"quant": step_quant, "threads": step_threads, "baseline": step_baseline,
         "parallel1": step_parallel1, "parallel4": step_parallel4, "attention": step_attention}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", default=",".join(STEPS), help=f"subset of {','.join(STEPS)}")
    args = parser.parse_args()
    steps = args.steps.split(",")
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise SystemExit(f"Unknown steps: {sorted(unknown)}")

    reference_check("start")
    for name in steps:
        print(f"== step: {name}", flush=True)
        started = time.monotonic()
        STEPS[name]()
        print(f"== {name} done in {(time.monotonic() - started) / 60:.1f} min", flush=True)
        reference_check(name)


if __name__ == "__main__":
    main()
