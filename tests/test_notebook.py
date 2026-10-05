import json
import re
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "vllm" / "vllm_benchmark.ipynb"


def _load() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(nb: dict) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_is_valid_nbformat_4() -> None:
    nb = _load()
    assert nb["nbformat"] == 4
    assert nb["cells"][0]["cell_type"] == "markdown"
    assert all("id" in c for c in nb["cells"])


def test_code_cells_compile_and_have_no_outputs() -> None:
    nb = _load()
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell["outputs"] == [] and cell["execution_count"] is None
        # Shell (!) and magic (%) lines are IPython syntax, not Python.
        lines = [line for line in cell["source"] if not line.lstrip().startswith(("!", "%"))]
        source = "".join(lines)
        compile(source, f"<cell {cell['id']}>", "exec")


def test_cells_run_in_the_documented_order() -> None:
    code = "\n".join(_code_cells(_load()))
    steps = ["nvidia-smi", "pip install", "MODEL =", "RUN_BENCH =", "def start_server",
             '"load_vllm"', '"load_prefix_caching"', "AWQ_MODEL, log_name", "make_archive"]
    positions = [code.index(step) for step in steps]
    assert positions == sorted(positions)


def test_no_secrets() -> None:
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", text)          # Hugging Face tokens
    assert not re.search(r"sk-[A-Za-z0-9]{20,}", text)          # OpenAI-style keys
    assert not re.search(r"(?i)(api[_-]?key|token)\s*=\s*['\"][^'\"<]", text)
