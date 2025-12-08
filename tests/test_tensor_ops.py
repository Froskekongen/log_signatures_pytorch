import pytest
import torch

from log_signatures_pytorch.tensor_ops import (
    add_tensor_product,
    batch_add_tensor_product,
    batch_mult_fused_restricted_exp,
    batch_restricted_exp,
    batch_tensor_product,
    lie_brackets,
    tensor_product,
)


def test_tensor_product_matches_outer_product() -> None:
    x = torch.tensor([1.0, 2.0], dtype=torch.float64)
    y = torch.tensor([3.0, 4.0], dtype=torch.float64)
    expected = torch.outer(x, y)
    torch.testing.assert_close(tensor_product(x, y), expected)


def test_add_tensor_product_matches_manual_definition() -> None:
    x = torch.zeros(2, 2, dtype=torch.float64)
    y = torch.tensor([1.0, 0.5], dtype=torch.float64)
    z = torch.tensor([2.0, -1.0], dtype=torch.float64)
    manual = x + torch.outer(y, z)
    torch.testing.assert_close(add_tensor_product(x, y, z), manual)


def test_restricted_exp_depth_three() -> None:
    vec = torch.tensor([[0.5, -1.0]], dtype=torch.float64)
    tensors = batch_restricted_exp(vec, depth=3)
    assert len(tensors) == 3
    torch.testing.assert_close(tensors[0], vec)
    torch.testing.assert_close(tensors[1], batch_tensor_product(vec, vec / 2.0))
    torch.testing.assert_close(
        tensors[2],
        batch_tensor_product(tensors[1], vec / 3.0),
    )


def test_restricted_exp_grad_propagates() -> None:
    vec = torch.tensor([[0.3, -0.1]], dtype=torch.float64, requires_grad=True)
    tensors = batch_restricted_exp(vec, depth=3)
    loss = sum(t.sum() for t in tensors)
    loss.backward()
    assert vec.grad is not None
    assert torch.all(torch.isfinite(vec.grad))


def test_batch_restricted_exp_matches_singleton_path() -> None:
    single = torch.tensor([[0.2, -0.4]], dtype=torch.float64)
    batch = batch_restricted_exp(single, depth=2)
    assert len(batch) == 2
    torch.testing.assert_close(batch[0], single)
    torch.testing.assert_close(batch[1], batch_tensor_product(single, single / 2.0))


@pytest.mark.parametrize("depth", [2, 3, 4])
def test_mult_fused_restricted_exp_commutative_case(depth: int) -> None:
    # In one dimension, successive increments commute and the signature
    # equals the restricted exponential of their sum.
    increment1 = torch.tensor([[0.4]], dtype=torch.float64)
    increment2 = torch.tensor([[-0.1]], dtype=torch.float64)
    carry = batch_restricted_exp(increment1, depth=depth)
    updated = batch_mult_fused_restricted_exp(increment2, carry)
    expected = batch_restricted_exp(increment1 + increment2, depth=depth)
    for got, want in zip(updated, expected):
        torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("depth", [2, 3, 4])
def test_batch_mult_fused_restricted_exp_commutative_case(depth: int) -> None:
    increment1 = torch.tensor([[0.4], [-0.6]], dtype=torch.float64)
    increment2 = torch.tensor([[0.1], [0.5]], dtype=torch.float64)
    carry = batch_restricted_exp(increment1, depth=depth)
    updated = batch_mult_fused_restricted_exp(increment2, carry)
    expected = batch_restricted_exp(increment1 + increment2, depth=depth)
    for got, want in zip(updated, expected):
        torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)


def test_batch_add_tensor_product_agrees_with_manual_tensor_product() -> None:
    x = torch.ones(2, 2, dtype=torch.float64)
    y = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    z = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float64)
    manual = x + batch_tensor_product(y, z)
    torch.testing.assert_close(batch_add_tensor_product(x, y, z), manual)


def test_lie_bracket_zero_for_parallel_vectors() -> None:
    vec = torch.randn(3, dtype=torch.float64)
    result = lie_brackets(vec, vec)
    torch.testing.assert_close(result, torch.zeros_like(result))
