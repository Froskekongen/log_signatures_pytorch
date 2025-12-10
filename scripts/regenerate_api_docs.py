"""Regenerate mkdocstrings API stub pages from the current codebase.

This script writes one Markdown file per module under ``docs/api`` with
``:::`` directives for each public function/class defined in that module.
Run it whenever the API surface changes to keep the docs in sync.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Iterable, List

MODULES = [
    "hall_projection",
    "lyndon_words",
    "signature",
    "log_signature",
    "hall_bch",
    "tensor_ops",
    "bch_coefficients",
]

TITLE_OVERRIDES = {
    "hall_bch": "Hall BCH",
    "bch_coefficients": "BCH Coefficients",
}


def public_callables(module) -> List[str]:
    """Return sorted public callables defined in the given module.

    Includes lru_cache-wrapped functions, classes, and regular functions whose
    ``__module__`` matches the target module.
    """
    names: List[str] = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        if inspect.isclass(obj) or inspect.isfunction(obj) or inspect.isbuiltin(obj):
            names.append(name)
            continue
        if callable(obj):
            names.append(name)
    return sorted(dict.fromkeys(names))


def write_module_page(module_name: str, targets: Iterable[str], docs_dir: Path) -> None:
    """Write a single Markdown page with mkdocstrings directives."""
    title = TITLE_OVERRIDES.get(module_name, module_name.replace("_", " ").title())
    header = f"# {title}\n\n"
    body_lines = [
        f"::: log_signatures_pytorch.{module_name}.{target}\n" for target in targets
    ]
    (docs_dir / f"{module_name}.md").write_text(
        header + "".join(body_lines), encoding="utf-8"
    )


def main() -> None:
    import importlib

    docs_api_dir = Path(__file__).resolve().parent.parent / "docs" / "api"
    docs_api_dir.mkdir(parents=True, exist_ok=True)

    for mod_name in MODULES:
        module = importlib.import_module(f"log_signatures_pytorch.{mod_name}")
        targets = public_callables(module)
        write_module_page(mod_name, targets, docs_api_dir)
        print(f"Wrote docs/api/{mod_name}.md with {len(targets)} entries")


if __name__ == "__main__":
    main()
