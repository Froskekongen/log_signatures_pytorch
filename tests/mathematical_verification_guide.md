# Mathematical Verification Guide for Log-Signature Implementation

This document outlines the mathematical properties that should hold for each function in the log-signature implementation. Treat it as a checklist for analytical reasoning and targeted tests (e.g., pytest/property checks), not as an LLM prompt.

## 1. Hall Basis Generation (`src/log_signatures_pytorch/basis.py`)

### Functions to Verify:
- `hall_basis(width, depth)`: Generates Hall basis elements
- `logsigdim(width, depth)`: Computes dimension of log-signature
- `logsigkeys(width, depth)`: Generates string representations

### Mathematical Properties to Verify:

1. **Hall Basis Definition**:
   - Verify that generated elements satisfy Hall ordering conditions:
     - A bracket [a, b] is in Hall basis if a < b (lexicographically)
     - If a = [c, d] is a bracket, then d <= c
     - If a = [c, d] and b = [e, f] are brackets, then if c < e, then [a, b] is in basis
   
2. **Dimension Formula**:
   - Verify that `logsigdim(width, depth)` matches the Witt formula for free Lie algebra dimension
   - For width=2: depth 1→2, depth 2→3, depth 3→5, depth 4→8, etc.
   - Formula: dim = (1/d) * Σ(μ(d/k) * width^k) where μ is Möbius function
   
3. **Basis Completeness**:
   - Verify that Hall basis elements span the free Lie algebra
   - Check that all elements are linearly independent
   - Verify uniqueness of each basis element

4. **Ordering Properties**:
   - Elements should be ordered by depth (lower depth first)
   - Within same depth, should be lexicographically ordered
   - Verify that ordering matches esig library output

## 2. Lie Bracket Operations (`src/log_signatures_pytorch/tensor_ops.py`)

### Functions to Verify:
- `lie_brackets(x, y)`: Computes [x, y] = x ⊗ y - y ⊗ x
- `batch_lie_brackets(x, y)`: Batched version

### Mathematical Properties to Verify:

1. **Anti-commutativity**:
   - Verify: [a, b] = -[b, a] for all tensors a, b
   - This is a fundamental property of Lie brackets

2. **Jacobi Identity**:
   - Verify: [a, [b, c]] + [b, [c, a]] + [c, [a, b]] = 0
   - This is the defining property of Lie algebras

3. **Bilinearity**:
   - Verify: [αa + βb, c] = α[a, c] + β[b, c] for scalars α, β
   - Verify: [a, αb + βc] = α[a, b] + β[a, c]

4. **Distributivity**:
   - Verify: [a, b + c] = [a, b] + [a, c]
   - Verify: [a + b, c] = [a, c] + [b, c]

5. **Zero Properties**:
   - Verify: [a, 0] = 0 and [0, a] = 0

## 3. Baker-Campbell-Hausdorff Formula (`src/log_signatures_pytorch/tensor_ops.py` and `src/log_signatures_pytorch/hall_bch.py`)

### Functions to Verify:
- `batch_bch_formula(a, b, depth)`: Batched low-order merge in tensor algebra
- `HallBCH.bch(x, y)`: Hall-basis closed form implemented up to depth 4

### Mathematical Properties to Verify:

1. **BCH Series Expansion**:
   - For `batch_bch_formula`, verify the included terms (a + b and +[a, b]/2 when depth ≥ 2) match the truncated series it implements.
   - For `HallBCH.bch`, verify coefficients against the closed-form truncation up to depth 4:
     - z = a + b + 1/2 [a, b]
     - + 1/12 [a, [a, b]] + 1/12 [b, [b, a]]
     - - 1/24 [b, [a, [a, b]]] (depth ≥ 4)

2. **Associativity Property**:
   - BCH(BCH(a, b), c) should relate to BCH(a, BCH(b, c)) via higher-order terms
   - This is complex but should be verified for small depth

3. **Special Cases**:
   - BCH(a, 0) = a
   - BCH(0, b) = b
   - BCH(a, -a) ≈ 0 (for small a)

4. **Truncation Error**:
   - Verify that truncation to depth d gives correct result up to that depth
   - Check error bounds for truncated BCH formula

## 4. Log-Signature Computation (`src/log_signatures_pytorch/log_signature.py`)

### Functions to Verify:
- `log_signature(path, depth)`: Main API function
- `_batch_log_signature(path, depth)`: CPU implementation
- `_batch_log_signature_gpu(path, depth)`: GPU implementation
- `_signature_to_logsignature_tensor(sig_tensors, width, depth)`: Conversion function
- `_project_to_hall_basis(log_sig_tensors, width, depth)`: Projection function

### Mathematical Properties to Verify:

1. **Exponential Relationship**:
   - Verify: signature(path) = exp(log_signature(path)) (up to truncation)
   - This is the fundamental relationship between signature and log-signature
   - Check: exp(log_sig) should match signature for simple paths

2. **Chen's Identity**:
   - Verify that log-signature satisfies Chen's identity for path concatenation
   - For paths X and Y: log_sig(X * Y) = BCH(log_sig(X), log_sig(Y))

3. **Path Increment Property**:
   - For a single increment dx, verify: log_sig ≈ dx (at depth 1)
   - For small increments, higher-order terms should be small

4. **Projection Correctness**:
   - Verify that projection onto Hall basis extracts correct coefficients
   - Check that tensor components map correctly to Hall basis elements
   - Verify that projection preserves the free Lie algebra structure

5. **Dimension Consistency**:
   - Verify that output dimension matches logsigdim(width, depth)
   - Check that log-signature dimension < signature dimension

6. **Special Cases**:
   - Zero path: log_sig(0) = 0
   - Straight line: log_sig should match analytical result
   - Path reversal: relationship between log_sig(X) and log_sig(X_rev)

7. **Differentiability**:
   - Verify that log-signature is differentiable with respect to path
   - Check that gradients are well-behaved (finite, no NaN)
   - Verify chain rule applies correctly

## 5. Numerical Properties

### Properties to Verify:

1. **Stability**:
   - Verify numerical stability for small/large increments
   - Check behavior with repeated points
   - Verify no NaN or Inf in outputs

2. **Precision**:
   - Verify that exp(log_sig) matches sig within numerical precision
   - Check error bounds for truncation
   - Verify consistency between CPU and GPU implementations

3. **Edge Cases**:
   - Zero increments
   - Very small increments
   - Very large increments
   - Degenerate paths (collinear points)

## Verification Methodology

For each function, the verification should:

1. **Check Mathematical Definitions**: Verify that implementation matches mathematical definition
2. **Verify Properties**: Check that all known mathematical properties hold
3. **Compare with References**: Compare with esig/signatory libraries where possible
4. **Test Special Cases**: Verify behavior on known analytical cases
5. **Check Relationships**: Verify relationships with other functions (e.g., exp(log_sig) = sig)

## Known Limitations / Checks

1. **BCH Depth Limits**: `HallBCH.bch` is implemented up to depth 4; callers should fall back to the default signature→log path beyond that.
2. **Truncation Awareness**: `batch_bch_formula` includes only the first commutator term; expectations/tests should align with that truncation.

## References

- Hall basis: "On the bases of free Lie algebras" by M. Hall (1950)
- BCH formula: Baker-Campbell-Hausdorff formula in Lie theory
- Log-signatures: "A Primer on the Signature Method in Machine Learning"
- esig library: https://github.com/datasig-ac-uk/esig
- signatory library: https://github.com/patrick-kidger/signatory
