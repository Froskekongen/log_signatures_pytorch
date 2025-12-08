# Repository Guidelines

## Project Structure & Module Organization
Core kernels live in `src/log_signatures_pytorch/`, where `signature.py` and `log_signature.py` expose the public API, `basis.py` and `hall_projection.py` cover Hall-basis math, and `tensor_ops.py` hosts shared Torch tensor utilities. Tests sit in `tests/`, grouped by property-based specs (`test_log_signature_properties.py`) and integration checks (`test_log_signature.py`, `test_mathematical_verification.py`). `main.py` is a thin smoke-test entry point, while `pyproject.toml` and `uv.lock` define dependencies (PyTorch, NumPy, SciPy plus dev extras like pytest/esig).

## Build, Test, and Development Commands
- `uv venv && source .venv/bin/activate` — create/activate the local virtualenv (Python 3.13+).
- `uv sync --group dev` — install runtime + dev deps (torch, pytest, esig, mkdocs, etc.). Use `uv sync` without `--group dev` for runtime-only.
- `uv run pytest tests -q` — run the full test matrix; add `-k pattern` or `-m "not external"` to skip esig parity tests when esig isn't installed.
- `uv run python -m pytest tests/test_log_signature.py -vv` — focus on the log-signature suite during kernel work.
- `uv run python main.py` — quick smoke test that the CLI entry runs.

## Coding Style & Naming Conventions
Write Python 3.13-compatible code with 4-space indentation, type hints, and docstrings describing tensor shapes. Favor pure functions that operate on `torch.Tensor` batches, and keep public names snake_case (`log_signature`, `hall_basis`). Internal helper tensors should read `*_tensor` or `*_mat` to mirror existing patterns. Use Torch ops over NumPy to stay on GPU, and prefer vectorized expressions inside `tensor_ops` rather than Python loops. Add numpy docstrings to new function and classes.

## Testing Guidelines
All new functionality needs at least one deterministic unit test in `tests/`, mirroring existing `test_*` files. Tests should name the mathematical property they enforce (`test_signature_invariance_under_translation`). Use pytest parametrization for width/depth sweeps instead of manual loops. When adding low-level kernels, compare against the reference `mathematical_verification` helpers and ensure `pytest --maxfail=1 --disable-warnings` passes locally before pushing.

## Commit & Pull Request Guidelines
Follow the current history: concise, imperative commit subjects (`Add signature computations`, `Start with cursor agent implementation`). Each PR should describe the tensor-level change, note affected modules, link any tracking issue, and attach benchmarking numbers or screenshots for perf work. Include reproduction commands for reviewers (e.g., `pytest tests/test_log_signature.py`). Keep PRs focused on one feature/bugfix and call out any new dependencies in the description.

## Security & Configuration Tips
Never commit large datasets or proprietary paths; only Torch tensors derived from synthetic data belong in tests. Respect the restricted network policy by keeping downloads out of runtime code. When enabling GPU-specific behavior, gate it behind `gpu_optimized` flags and fall back to CPU paths so CI can run without CUDA.
