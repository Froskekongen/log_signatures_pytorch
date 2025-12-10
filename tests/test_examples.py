import doctest
import importlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


DOC_MODULES = [
    "log_signatures_pytorch.__init__",
    "log_signatures_pytorch.bch_coefficients",
    "log_signatures_pytorch.hall_bch",
    "log_signatures_pytorch.hall_projection",
    "log_signatures_pytorch.log_signature",
    "log_signatures_pytorch.signature",
    "log_signatures_pytorch.tensor_ops",
]


@pytest.mark.parametrize("module_name", DOC_MODULES)
def test_docstrings_run(module_name: str) -> None:
    """Ensure all doctest examples execute without failure."""
    module = importlib.import_module(module_name)
    failures, _ = doctest.testmod(
        module, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    )
    assert failures == 0, f"Doctests failed for {module_name}"


def _run_python_fences(doc_path: Path) -> None:
    """Execute Python code fences in order, sharing a namespace within the file."""
    ns: dict[str, object] = {"__name__": "__main__"}
    lines = doc_path.read_text().splitlines()
    in_block = False
    block: list[str] = []
    start_line = 0
    for idx, line in enumerate(lines, 1):
        if not in_block:
            if line.strip().startswith("```python"):
                in_block = True
                block = []
                start_line = idx + 1
            continue
        if line.strip().startswith("```"):
            code = "\n".join(block)
            compiled = compile(code, f"{doc_path}:{start_line}", "exec")
            exec(compiled, ns)
            in_block = False
            block = []
            continue
        block.append(line)
    if in_block:
        raise AssertionError(f"Unclosed code fence in {doc_path}")


def _docs_markdown_files() -> list[Path]:
    docs_root = ROOT / "docs"
    return sorted(p for p in docs_root.rglob("*.md") if p.is_file())


@pytest.mark.parametrize("doc_path", [ROOT / "README.md", *_docs_markdown_files()])
def test_markdown_examples(doc_path: Path) -> None:
    """Run all Python code fences in markdown docs to keep examples fresh."""
    _run_python_fences(doc_path)
