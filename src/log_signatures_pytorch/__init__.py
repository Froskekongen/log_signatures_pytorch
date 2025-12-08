"""Log- and signature computation for PyTorch (Hall and Lyndon bases).

This package provides efficient, differentiable computation of signatures and
log-signatures for paths/streams using PyTorch. Two coordinate systems for the
log-signature are available:

- Hall basis (default)
- Lyndon \"words\" basis (Signatory-compatible gather projection)

Main entry points:

- :func:`signature`: Compute the signature of batched paths
- :func:`log_signature`: Compute the log-signature of batched paths (Hall or words)
- :func:`hall_basis`: Generate Hall basis elements
- :func:`lyndon_words`: Generate Lyndon words up to a depth
- :func:`logsigdim` / :func:`logsigdim_words`: Dimension helpers
- :func:`logsigkeys` / :func:`logsigkeys_words`: Human-readable labels

Examples
--------
>>> import torch
>>> from log_signatures_pytorch import signature, log_signature, logsigdim
>>>
>>> # Single path (add batch dimension)
>>> path = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]).unsqueeze(0)
>>> sig = signature(path, depth=2)
>>> sig.shape
torch.Size([1, 6])
>>>
>>> # Log-signature
>>> log_sig = log_signature(path, depth=2)
>>> log_sig.shape
torch.Size([1, 3])
>>> logsigdim(2, 2)
3
"""

from .basis import (
    hall_basis,
    logsigdim,
    logsigkeys,
    logsigdim_words,
    logsigkeys_words,
    lyndon_words,
)
from .log_signature import log_signature
from .signature import signature

__all__ = [
    "signature",
    "log_signature",
    "hall_basis",
    "logsigdim",
    "logsigkeys",
    "logsigdim_words",
    "logsigkeys_words",
    "lyndon_words",
]
