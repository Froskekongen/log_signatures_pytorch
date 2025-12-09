"""Lyndon words (\"words\"/efficient basis) utilities.

This module generates Lyndon words and provides dimension/key helpers for the
\"words\" basis used by Signatory. Letters are 1-based to match Hall basis
conventions throughout the library.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import torch

# WordsBasisElement represents a Lyndon word as a tuple of ints (1-based letters)
WordsBasisElement = Tuple[int, ...]


def _lyndon_fixed_length(width: int, length: int) -> List[WordsBasisElement]:
    """Generate Lyndon words of exact ``length`` over an alphabet ``1..width``."""
    if length < 1:
        return []
    # Working array (1-based indexing for simplicity)
    a: List[int] = [0] * (length + 1)
    words: List[WordsBasisElement] = []

    def generate(t: int, p: int):
        if t > length:
            if p == length:
                # Convert to 1-based letters for external consistency
                words.append(tuple(x + 1 for x in a[1 : length + 1]))
            return
        a[t] = a[t - p]
        generate(t + 1, p)
        for j in range(a[t - p] + 1, width):
            a[t] = j
            generate(t + 1, t)

    generate(1, 1)
    return words


@lru_cache(maxsize=None)
def lyndon_words(width: int, depth: int) -> Tuple[WordsBasisElement, ...]:
    """Lyndon words up to ``depth`` in Signatory-compatible ordering."""
    if width < 1:
        raise ValueError("width must be >= 1")
    if depth < 1:
        raise ValueError("depth must be >= 1")

    words: List[WordsBasisElement] = []
    for length in range(1, depth + 1):
        words.extend(_lyndon_fixed_length(width, length))
    return tuple(words)


def logsigdim_words(width: int, depth: int) -> int:
    """Dimension of the truncated log-signature in the Lyndon \"words\" basis."""
    return len(lyndon_words(width, depth))


def logsigkeys_words(width: int, depth: int) -> List[str]:
    """Human-readable labels for the Lyndon \"words\" basis (Signatory style)."""

    def _to_str(word: WordsBasisElement) -> str:
        return ",".join(str(x) for x in word)

    return [_to_str(word) for word in lyndon_words(width, depth)]


@lru_cache(maxsize=None)
def _words_indices(width: int, depth: int) -> Tuple[Tuple[int, ...], ...]:
    """Cached tensor indices for Lyndon words grouped by length."""
    grouped = [[] for _ in range(depth)]
    for word in lyndon_words(width, depth):
        idx = 0
        for letter in word:
            idx = idx * width + (letter - 1)
        grouped[len(word) - 1].append(idx)
    return tuple(tuple(group) for group in grouped)


def _project_to_words_basis(
    log_sig_tensors: list[torch.Tensor], width: int, depth: int
) -> torch.Tensor:
    """Project log-signature tensors onto the Lyndon \"words\" basis."""
    if not log_sig_tensors:
        return torch.zeros(
            0,
            device=torch.device("cpu"),
            dtype=torch.float32,  # pragma: no cover
        )

    indices_by_depth = _words_indices(width, depth)
    slices = []
    for k, indices in enumerate(indices_by_depth, start=1):
        if not indices:
            continue
        tensor = log_sig_tensors[k - 1].reshape(log_sig_tensors[k - 1].shape[0], -1)
        gather_idx = torch.tensor(indices, device=tensor.device, dtype=torch.long)
        slices.append(torch.index_select(tensor, dim=1, index=gather_idx))
    if not slices:
        return torch.zeros(
            log_sig_tensors[0].shape[0],
            0,
            device=log_sig_tensors[0].device,
            dtype=log_sig_tensors[0].dtype,
        )
    return torch.cat(slices, dim=1)

