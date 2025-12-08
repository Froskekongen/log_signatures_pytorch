# log_signatures_pytorch

Differentiable log-signature and signature kernels implemented in PyTorch with both CPU-friendly and GPU-parallel execution paths.

## What you'll find

- Batched signature and log-signature computation for tensors shaped ``(batch, length, dim)`` with optional streaming outputs at every step. For a single path, add a leading dimension via ``unsqueeze(0)``.
- Hall-basis utilities (`hall_basis`, `logsigdim`, `logsigkeys`) for inspecting dimensions and basis labels.
- Two log-signature backends: the default signature→log path, and an incremental sparse BCH implementation for depths up to 4 (falls back otherwise).
- The implementation of signatures is structured after keras_sig, but only focuses on pytorch.
- Dependencies are kept minimal.


## Installation

Requires Python 3.13+ and PyTorch ≥ 2.9 (CPU or CUDA builds work). From the repository root:

```bash
uv venv
source .venv/bin/activate
uv sync                    # installs runtime deps + project in editable mode
# uv sync --group dev      # adds pytest/esig for running the full test suite
```

Verify PyTorch is available:

```bash
python - <<'PY'
import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
PY
```

## Quick start

### Signature and log-signature of a single path

```python
import torch
from log_signatures_pytorch import signature, log_signature, logsigdim

path = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]).unsqueeze(0)

sig = signature(path, depth=2)
print(sig.shape)           # torch.Size([1, 6]) = sum(width**k for k in 1..depth)

log_sig = log_signature(path, depth=2)
print(log_sig.shape)       # torch.Size([1, 3]) = logsigdim(2, 2)
print("logsigdim:", logsigdim(2, 2))  # 3
```

### Batched computation and streaming outputs

```python
batch_paths = torch.tensor([
    [[0.0, 0.0], [1.0, 1.0]],
    [[0.0, 0.0], [2.0, 2.0]],
])

sig = signature(batch_paths, depth=2)
print(sig.shape)                 # torch.Size([2, 6])

log_sig_stream = log_signature(batch_paths, depth=2, stream=True)
print(log_sig_stream.shape)      # torch.Size([2, 1, 3]) -> (batch, steps, logsigdim)

# Streaming for a single path returns one row per increment (batch=1)
sig_stream = signature(path, depth=2, stream=True)
print(sig_stream.shape)          # torch.Size([1, 2, 6])
```

### Hall basis helpers

```python
from log_signatures_pytorch import hall_basis, logsigkeys

basis = hall_basis(width=2, depth=2)
print(basis)          # [1, 2, (1, 2)]

keys = logsigkeys(width=2, depth=2)
print(keys)           # ['1', '2', '[1,2]'] (matches esig format)
```

### Choosing computation mode

- `gpu_optimized`: defaults to True when the input tensor is on CUDA. Set False to force the CPU scan path.
- `chunk_size`: optional on CPU to trade a small amount of extra compute for lower peak memory when sequences are long.
- `method`: `log_signature(..., method="bch_sparse")` uses the incremental BCH routine for depths supported by `HallBCH` (depth ≤ 4); otherwise it falls back to the default path.

Signature outputs exclude the empty word (dimension is `sum(width**k for k=1..depth)`); use `logsigdim(width, depth)` to size log-signature outputs.

### GPU compile recommendations

- For fixed shapes with many repeated calls, `torch.compile` with mode `reduce-overhead` gives the fastest runtime for `_batch_signature_gpu` (~0.08–0.12 ms in our sweeps) and for `log_signature` (similarly sized speedups). First-call compile time can be large—especially for log-signatures—so cache per shape if you need to reuse compiled artifacts.
- For workloads with many varying shapes or when compile latency matters more than per-call speed, prefer `none` (no compile) or the lighter `default` mode. `default` is slower than `reduce-overhead` at runtime but compiles much faster; this tradeoff is more pronounced for log-signatures.
- The benchmark helper `benchmarks/benchmark_batch_signature_gpu.py` supports `--target signature|log_signature`, `--compile-modes none reduce-overhead default`, `--measure-compile-time`, and per-shape compile caching. CSVs land under `benchmarks/results/` by default.

## Testing and verification

- Run the suite: `pytest tests -q`
- Quick smoke check: `python main.py`
- Mathematical property checks are documented in `tests/mathematical_verification_guide.md`.

## Documentation

Documentation is built using MkDocs with mkdocstrings and Material theme. To build and serve the documentation:

```bash
# Build static site
uv run mkdocs build

# Serve locally (with auto-reload on changes)
uv run mkdocs serve
```

The documentation will be available at `http://127.0.0.1:8000` when serving locally.

## References

- esig: https://github.com/datasig-ac-uk/esig
- signatory: https://github.com/patrick-kidger/signatory
- keras-sig: https://github.com/remigenet/keras_sig
- Hall basis: "On the bases of free Lie algebras". M. Hall (1950)

## License

MIT


## Citation

If you use this software in your research or in your project, please cite it as follows:

### BibTeX

```bibtex
@software{log_signatures_pytorch,
  author = {Aune, Erlend},
  title = {log-signatures-pytorch: Differentiable log-signature and signature kernels in PyTorch},
  version = {0.1.x},
  url = {https://github.com/froskekongen/log_signatures_pytorch},
  year = {2025},
  license = {MIT}
}
```

### Plain text

Aune, Erlend. (2025). log-signatures-pytorch: Differentiable log-signature and signature kernels in PyTorch (Version 0.1.x). [Computer software]: https://github.com/froskekongen/log_signatures_pytorch
