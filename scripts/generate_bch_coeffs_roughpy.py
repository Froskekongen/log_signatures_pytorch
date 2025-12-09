#!/usr/bin/env python3
"""Generate BCH coefficients via roughpy and compare to the SymPy/Dynkin helper.

This is intended as an external oracle for validating BCH implementations.
It prints word coefficients (in the tensor basis) for a width-2 algebra and
verifies they match the internal SymPy-based generator.
"""

from __future__ import annotations

from typing import Dict, Tuple

import roughpy as rp

from log_signatures_pytorch.bch_coefficients import bch_coeffs as sympy_bch_coeffs


Word = Tuple[int, ...]


def bch_words(depth: int) -> Dict[Word, float]:
    """Return word -> coefficient for log(exp(X) exp(Y)) truncated to depth."""
    ctx = rp.get_context(width=2, depth=depth, coeffs=rp.Rational)
    X = rp.FreeTensor({rp.TensorKey((1,), ctx=ctx): rp.Rational(1)}, ctx=ctx)
    Y = rp.FreeTensor({rp.TensorKey((2,), ctx=ctx): rp.Rational(1)}, ctx=ctx)
    log_elem = (X.exp() * Y.exp()).log()
    coeffs: Dict[Word, float] = {}
    for item in log_elem:
        word = tuple(item.key().to_letters())
        if len(word) > depth:
            continue
        coeffs[word] = item.value().to_float()
    return coeffs


def bch_words_sympy(depth: int) -> Dict[Word, float]:
    """Return word -> coefficient computed via the SymPy-based helper."""
    coeffs = sympy_bch_coeffs(width=2, depth=depth)
    return {word: float(coeff) for word, coeff in coeffs.items()}


def assert_matching_coeffs(
    depth: int, rough: Dict[Word, float], sympy: Dict[Word, float], tol: float = 1e-9
) -> None:
    """Validate that roughpy and sympy agree up to tolerance."""
    mismatches = []
    for word in set(rough) | set(sympy):
        r_val = float(rough.get(word, 0.0))
        s_val = float(sympy.get(word, 0.0))
        if abs(r_val - s_val) > tol:
            mismatches.append((word, r_val, s_val))
    if mismatches:
        details = "\n".join(
            f"{w}: roughpy={r} sympy={s}" for w, r, s in sorted(mismatches)
        )
        raise SystemExit(f"BCH coefficient mismatch at depth {depth}:\n{details}")


def main() -> None:
    for depth in range(1, 6):
        rough_coeffs = bch_words(depth)
        sympy_coeffs = bch_words_sympy(depth)
        assert_matching_coeffs(depth, rough_coeffs, sympy_coeffs)
        print(f"Depth {depth}: {len(sympy_coeffs)} terms (SymPy matches roughpy)")
        for word, coeff in sorted(
            sympy_coeffs.items(), key=lambda kv: (len(kv[0]), kv[0])
        ):
            print(f"{word}: {coeff}")
        print()


if __name__ == "__main__":
    main()
