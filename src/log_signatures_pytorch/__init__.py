"""Log-signature computation for PyTorch.

This package provides efficient, differentiable computation of signatures and
log-signatures for paths/streams using PyTorch.

The main entry points are:

- :func:`signature`: Compute the signature of batched paths
- :func:`log_signature`: Compute the log-signature of batched paths
- :func:`hall_basis`: Generate Hall basis elements
- :func:`logsigdim`: Get the dimension of the log-signature
- :func:`logsigkeys`: Get human-readable labels for Hall basis elements

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

from .basis import hall_basis, logsigdim, logsigkeys
from .log_signature import log_signature
from .signature import signature

__all__ = [
    "signature",
    "log_signature",
    "hall_basis",
    "logsigdim",
    "logsigkeys",
]
