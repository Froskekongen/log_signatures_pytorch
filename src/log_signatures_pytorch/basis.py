"""Hall basis generation following the standard Hall set construction.

This module provides functions to generate and work with Hall basis elements,
which form a basis for the free Lie algebra. The Hall basis is used to
represent log-signatures in a compact form.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple, Union

# HallBasisElement represents Hall set elements as ints or nested brackets.
HallBasisElement = Union[int, Tuple["HallBasisElement", "HallBasisElement"]]

# WordsBasisElement represents a Lyndon word as a tuple of ints (1-based letters)
WordsBasisElement = Tuple[int, ...]


def _lyndon_fixed_length(width: int, length: int) -> List[WordsBasisElement]:
    """Generate Lyndon words of exact ``length`` over an alphabet ``1..width``.

    Parameters
    ----------
    width : int
        Alphabet size (path dimension), must be >= 1.
    length : int
        Target word length, must be >= 1.

    Returns
    -------
    list[tuple[int, ...]]
        Lyndon words of the requested length, emitted in lexicographic order
        using 1-based letters.

    Notes
    -----
    Implements the classic Duval/FKM recursive construction in
    O(number_of_words * length) time.
    """
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
    """Lyndon words up to ``depth`` in Signatory-compatible ordering.

    Parameters
    ----------
    width : int
        Alphabet size (path dimension), must be >= 1.
    depth : int
        Maximum word length to include, must be >= 1.

    Returns
    -------
    tuple[tuple[int, ...], ...]
        All Lyndon words of lengths 1..depth, ordered first by length then
        lexicographically within each length. Letters are 1-based to align
        with Hall basis conventions.

    Raises
    ------
    ValueError
        If ``width < 1`` or ``depth < 1``.
    """
    if width < 1:
        raise ValueError("width must be >= 1")
    if depth < 1:
        raise ValueError("depth must be >= 1")

    words: List[WordsBasisElement] = []
    for length in range(1, depth + 1):
        words.extend(_lyndon_fixed_length(width, length))
    return tuple(words)


def _hall_basis_key(elem: HallBasisElement):
    if isinstance(elem, int):
        return (0, elem)
    left, right = elem
    return (1, _hall_basis_key(left), _hall_basis_key(right))


def _hall_is_valid_pair(left: HallBasisElement, right: HallBasisElement) -> bool:
    """Check Hall ordering constraints for a candidate bracket (left, right)."""
    if _hall_basis_key(left) >= _hall_basis_key(right):
        return False
    if isinstance(right, tuple):
        right_left, _ = right
        if _hall_basis_key(right_left) > _hall_basis_key(left):
            return False
    return True


def hall_basis(width: int, depth: int) -> List[HallBasisElement]:
    """Return Hall basis elements up to ``depth`` over an alphabet of size ``width``.

    The Hall basis is a particular basis for the free Lie algebra. Elements are
    ordered first by depth, then lexicographically by the recursive Hall ordering.
    Degree-1 elements are labeled 1..width and higher degrees are nested tuples
    representing Lie brackets.

    Parameters
    ----------
    width : int
        Size of the alphabet (path dimension). Must be >= 1.
    depth : int
        Maximum depth to generate basis elements. Must be >= 1.

    Returns
    -------
    List[HallBasisElement]
        Hall basis elements, where each element is either an integer (degree 1)
        or a nested tuple representing a Lie bracket (higher degrees).

    Raises
    ------
    ValueError
        If ``width < 1`` or ``depth < 1``.

    Examples
    --------
    >>> from log_signatures_pytorch import hall_basis
    >>>
    >>> # Hall basis for width=2, depth=2
    >>> basis = hall_basis(2, 2)
    >>> basis
    [1, 2, (1, 2)]
    >>>
    >>> # Hall basis for width=2, depth=3
    >>> basis = hall_basis(2, 3)
    >>> len(basis)
    5
    >>> basis[:3]
    [1, 2, (1, 2)]
    >>>
    >>> # Hall basis for width=3, depth=2
    >>> basis = hall_basis(3, 2)
    >>> len(basis)
    6  # 3 degree-1 + 3 degree-2 elements
    """
    if width < 1:
        raise ValueError("width must be >= 1")
    if depth < 1:
        raise ValueError("depth must be >= 1")

    depth_groups: Dict[int, List[HallBasisElement]] = {}
    letters = list(range(1, width + 1))
    depth_groups[1] = letters
    basis: List[HallBasisElement] = list(letters)

    for current_depth in range(2, depth + 1):
        candidates: List[HallBasisElement] = []
        for left_depth in range(1, current_depth):
            right_depth = current_depth - left_depth
            for left in depth_groups[left_depth]:
                for right in depth_groups[right_depth]:
                    if _hall_is_valid_pair(left, right):
                        candidates.append((left, right))
        candidates.sort(key=_hall_basis_key)
        depth_groups[current_depth] = candidates
        basis.extend(candidates)

    return basis


def logsigdim(width: int, depth: int) -> int:
    """Dimension of the truncated log-signature in the Hall basis.

    This is the number of Hall basis elements up to the given depth, which
    determines the output dimension of log-signature computations.

    Parameters
    ----------
    width : int
        Path dimension (size of the alphabet). Must be >= 1.
    depth : int
        Truncation depth. Must be >= 1.

    Returns
    -------
    int
        The dimension of the log-signature, equal to the number of Hall basis
        elements up to the specified depth.

    Raises
    ------
    ValueError
        If ``width < 1`` or ``depth < 1``.

    Examples
    --------
    >>> from log_signatures_pytorch import logsigdim
    >>>
    >>> # Dimension for width=2, depth=2
    >>> logsigdim(2, 2)
    3
    >>>
    >>> # Dimension for width=2, depth=3
    >>> logsigdim(2, 3)
    5
    >>>
    >>> # Dimension for width=3, depth=2
    >>> logsigdim(3, 2)
    6
    >>>
    >>> # Compare with signature dimension
    >>> # Signature: 2 + 4 = 6 for width=2, depth=2
    >>> # Log-signature: 3 for width=2, depth=2
    >>> logsigdim(2, 2) < 2 + 2**2
    True
    """
    return len(hall_basis(width, depth))


def logsigdim_words(width: int, depth: int) -> int:
    """Dimension of the truncated log-signature in the Lyndon \"words\" basis.

    Parameters
    ----------
    width : int
        Alphabet size (path dimension), must be >= 1.
    depth : int
        Truncation depth, must be >= 1.

    Returns
    -------
    int
        Number of Lyndon words of lengths 1..depth (Witt formula).
    """
    return len(lyndon_words(width, depth))


def logsigkeys(width: int, depth: int) -> List[str]:
    """Human-readable labels for Hall basis elements (esig-compatible).

    Converts Hall basis elements to string representations that are compatible
    with the esig library format. Degree-1 elements are represented as integers,
    and higher degree elements are represented as nested bracket notation.

    Parameters
    ----------
    width : int
        Path dimension (size of the alphabet). Must be >= 1.
    depth : int
        Truncation depth. Must be >= 1.

    Returns
    -------
    List[str]
        List of string labels for each Hall basis element, in the same order
        as returned by :func:`hall_basis`.

    Raises
    ------
    ValueError
        If ``width < 1`` or ``depth < 1``.

    Examples
    --------
    >>> from log_signatures_pytorch import logsigkeys
    >>>
    >>> # Keys for width=2, depth=2
    >>> keys = logsigkeys(2, 2)
    >>> keys
    ['1', '2', '[1,2]']
    >>>
    >>> # Keys for width=2, depth=3
    >>> keys = logsigkeys(2, 3)
    >>> keys
    ['1', '2', '[1,2]', '[1,[1,2]]', '[[1,2],2]']
    >>>
    >>> # Keys for width=3, depth=2
    >>> keys = logsigkeys(3, 2)
    >>> keys
    ['1', '2', '3', '[1,2]', '[1,3]', '[2,3]']
    """

    def _to_str(elem: HallBasisElement) -> str:
        if isinstance(elem, int):
            return str(elem)
        left, right = elem
        return f"[{_to_str(left)},{_to_str(right)}]"

    return [_to_str(elem) for elem in hall_basis(width, depth)]


def logsigkeys_words(width: int, depth: int) -> List[str]:
    """Human-readable labels for the Lyndon \"words\" basis (Signatory style).

    Parameters
    ----------
    width : int
        Alphabet size (path dimension), must be >= 1.
    depth : int
        Maximum word length to include, must be >= 1.

    Returns
    -------
    list[str]
        Labels of the same length/order as :func:`lyndon_words`, where a word
        like (1, 2, 1) is rendered as ``\"1,2,1\"``.
    """

    def _to_str(word: WordsBasisElement) -> str:
        return ",".join(str(x) for x in word)

    return [_to_str(word) for word in lyndon_words(width, depth)]
