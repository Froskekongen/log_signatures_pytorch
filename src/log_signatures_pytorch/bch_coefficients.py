"""Simple BCH coefficient generator (tensor-algebra series with exact rationals).

This module provides functions to generate Baker-Campbell-Hausdorff coefficients
using exact rational arithmetic. It computes word -> rational coefficient mappings
for log(prod_i exp(E_i)), where E_i are the one-letter generators (1..width),
truncated to a given depth.

This is intended for verification/offline coefficient generation (tests, debug),
not runtime. Arithmetic uses Python's Fractions and concatenation; no external
CAS is required.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Dict, Tuple

Word = Tuple[int, ...]
CoeffMap = Dict[Word, Fraction]


def _mul_series(a: CoeffMap, b: CoeffMap, depth: int) -> CoeffMap:
    """Cauchy product of two series, truncating words beyond `depth`."""
    out: CoeffMap = {}
    for w1, c1 in a.items():
        for w2, c2 in b.items():
            w = w1 + w2
            if len(w) > depth:
                continue
            out[w] = out.get(w, Fraction(0)) + c1 * c2
    return out


def _pow_series(a: CoeffMap, n: int, depth: int) -> CoeffMap:
    """Binary exponentiation of a series (non-negative integer power)."""
    if n < 0:
        raise ValueError("Series exponent must be non-negative.")
    if n == 0:
        return {(): Fraction(1)}
    result: CoeffMap = {(): Fraction(1)}
    base = a
    exp = n
    while exp:
        if exp & 1:
            result = _mul_series(result, base, depth)
        exp //= 2
        if exp:
            base = _mul_series(base, base, depth)
    return result


def _exp_single(letter: int, depth: int) -> CoeffMap:
    """Truncated exp(e_letter) = sum_{k>=0} e_letter^k / k! up to `depth`."""
    out: CoeffMap = {(): Fraction(1)}
    word = (letter,)
    for k in range(1, depth + 1):
        out[word * k] = Fraction(1, math.factorial(k))
    return out


def _log_series(series: CoeffMap, depth: int) -> CoeffMap:
    """Formal log of a series with constant term 1, truncated to `depth`."""
    if series.get((), None) != 1:
        raise ValueError("log_series expects identity coefficient 1.")
    U = dict(series)
    U.pop((), None)
    result: CoeffMap = {}
    for n in range(1, depth + 1):
        term = _pow_series(U, n, depth)
        sign = -1 if (n % 2 == 0) else 1
        coeff = Fraction(sign, n)
        for w, c in term.items():
            if len(w) == 0:
                continue
            result[w] = result.get(w, Fraction(0)) + coeff * c
    return result


def bch_coeffs(width: int, depth: int) -> CoeffMap:
    """Coefficients for log(prod_{i=1}^width exp(e_i)) truncated to `depth`.

    Computes the exact rational coefficients for the Baker-Campbell-Hausdorff
    expansion of the product of exponentials of generators, truncated to the
    specified depth.

    Parameters
    ----------
    width : int
        Number of generators (path dimension). Must be >= 1.
    depth : int
        Truncation depth. Must be >= 1.

    Returns
    -------
    CoeffMap
        Dictionary mapping tensor-algebra words (tuples of generator indices)
        to rational coefficients (Fraction objects) in the series expansion.
        Words are tuples of integers from 1 to width, representing the
        generators in order.

    Examples
    --------
    >>> from log_signatures_pytorch.bch_coefficients import bch_coeffs
    >>> from fractions import Fraction
    >>>
    >>> # BCH coefficients for width=2, depth=2
    >>> coeffs = bch_coeffs(2, 2)
    >>> # Check some expected coefficients
    >>> coeffs[(1,)] == Fraction(1, 1)
    True
    >>> coeffs[(2,)] == Fraction(1, 1)
    True
    >>> # The bracket [1, 2] has coefficient 1/2, split across (1, 2) and (2, 1)
    >>> coeffs.get((1, 2), 0) == Fraction(1, 2)
    True
    >>> coeffs.get((2, 1), 0) == Fraction(-1, 2)
    True
    >>> coeffs.get((1, 2), 0) - coeffs.get((2, 1), 0) == Fraction(1, 1)
    True
    >>>
    >>> # BCH coefficients for width=3, depth=2
    >>> coeffs = bch_coeffs(3, 2)
    >>> # Three degree-1 generators
    >>> len([w for w in coeffs.keys() if len(w) == 1])
    3
    """
    if depth < 1 or width < 1:
        return {}
    # Multiply exp of each generator in order
    prod: CoeffMap = {(): Fraction(1)}
    for letter in range(1, width + 1):
        prod = _mul_series(prod, _exp_single(letter, depth), depth)
    log_series = _log_series(prod, depth)
    return {w: c for w, c in log_series.items() if len(w) <= depth and c != 0}
