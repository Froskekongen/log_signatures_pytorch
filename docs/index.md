# log-signatures-pytorch

Differentiable log-signature and signature kernels implemented in PyTorch with both CPU-friendly and GPU-parallel execution paths.

## What you'll find

- Batched signature and log-signature computation for tensors shaped `(batch, length, dim)` with optional streaming outputs at every step. For a single path, add a leading dimension via `unsqueeze(0)`.
- Hall-basis utilities (`hall_basis`, `logsigdim`, `logsigkeys`) for inspecting dimensions and basis labels.
- Two log-signature coordinate systems:
  - `mode="hall"` (default): classic Hall basis with dense projection.
  - `mode="words"`: Signatory-style Lyndon words basis using a gather-only projection for faster CPU/GPU throughput.
- Two computation backends: the default signature→log path, and an incremental sparse BCH implementation for depths up to 4 (falls back otherwise).
- The implementation of signatures is structured after keras_sig, but only focuses on pytorch.
- Dependencies are kept minimal.

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

# Lyndon (words) coordinates for comparison
log_sig_words = log_signature(path, depth=2, mode="words")
print(log_sig_words.shape)  # torch.Size([1, 3])
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

# Lyndon words (Signatory ordering)
from log_signatures_pytorch import lyndon_words, logsigkeys_words
words = lyndon_words(width=2, depth=3)
print(words)          # [(1,), (2,), (1, 2), (1, 1, 2), (1, 2, 2)]
print(logsigkeys_words(width=2, depth=3))
```

## Installation

Requires Python 3.13+ and PyTorch ≥ 2.9 (CPU or CUDA builds work). 

To install from pypi using pip, run:
```bash
pip install log-signatures-pytorch
```

## Testing Hall vs Words modes

- Deterministic unit tests (both bases):  
  `uv run pytest tests/test_log_signature.py -k "words or hall"`
- Mathematical verification suite for both bases (free Lie identities, log-exp consistency):  
  `uv run pytest tests/test_mathematical_verification.py`
- Microbenchmark comparison:  
  `PYTHONPATH=src uv run python benchmarks/words_vs_hall.py --width 3 --depth 4 --batch 64 --length 100`

Notes:
- `mode="words"` is available on the default signature→log path; BCH streaming currently supports `mode="hall"`.
- The verification tests reuse the same path examples and assert both modes satisfy the algebraic identities where applicable.

## References

- esig: https://github.com/datasig-ac-uk/esig
- signatory: https://github.com/patrick-kidger/signatory
- keras-sig: https://github.com/remigenet/keras_sig
- Hall basis: "On the bases of free Lie algebras" — M. Hall (1950)
- Lyndon words mode: Signatory documentation “mode='words'” and original Lyndon/Chen–Fox–Lyndon constructions

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
