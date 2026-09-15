from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tokenspeed.runtime.distributed.comm_backend import hccl as hccl_module
from tokenspeed.runtime.distributed.comm_backend import registry
from tokenspeed.runtime.distributed.comm_backend.hccl import HcclBackend


def test_registry_selects_hccl_on_npu(monkeypatch):
    monkeypatch.setattr(registry, "_global_backend", None)
    monkeypatch.setattr(
        registry,
        "current_platform",
        lambda: SimpleNamespace(is_npu=True),
    )

    backend = registry.initialize_comm_backend()

    assert isinstance(backend, HcclBackend)


def test_hccl_all_reduce_uses_hccl_process_group(monkeypatch):
    process_group = Mock()
    get_process_group = Mock(return_value=process_group)
    all_reduce = Mock()
    monkeypatch.setattr(hccl_module.pg_manager, "get_process_group", get_process_group)
    monkeypatch.setattr(hccl_module.dist, "all_reduce", all_reduce)
    tensor = torch.ones(4)

    output = HcclBackend().all_reduce(tensor, (0, 1))

    assert output is tensor
    get_process_group.assert_called_once_with("hccl", (0, 1))
    all_reduce.assert_called_once_with(
        tensor,
        op=torch.distributed.ReduceOp.SUM,
        group=process_group,
    )


@pytest.mark.parametrize("world,tokens", [(2, 0), (2, 1), (2, 16384), (8, 3)])
@pytest.mark.parametrize("contiguous", [True, False])
def test_token_reduce_scatter_equal_splits_skip_padding(
    monkeypatch, world, tokens, contiguous
):
    process_group = object()
    backend = HcclBackend()
    monkeypatch.setattr(backend, "_process_group", lambda group: process_group)
    value = torch.arange(world * tokens * 16, dtype=torch.float32).reshape(
        world * tokens, 16
    )
    if not contiguous:
        value = value[:, ::2]
    calls = []

    def forbidden_zeros(*args, **kwargs):
        raise AssertionError("equal splits must not allocate a zero padding buffer")

    def reduce_scatter(output, source, group):
        assert group is process_group and source.is_contiguous()
        if value.is_contiguous():
            assert source.data_ptr() == value.data_ptr()
        torch.testing.assert_close(source, value)
        # Simulate rank 0 with identical inputs on every rank.
        output.copy_(source[:tokens] * world)
        calls.append(source)

    monkeypatch.setattr(torch.Tensor, "new_zeros", forbidden_zeros)
    monkeypatch.setattr(hccl_module.dist, "reduce_scatter_tensor", reduce_scatter)
    result = backend.token_reduce_scatter(value, tuple(range(world)), [tokens] * world)
    torch.testing.assert_close(result, value[:tokens] * world)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "splits,rank", [([1, 3], 0), ([1, 3], 1), ([0, 3], 0), ([0, 3], 1)]
)
def test_token_reduce_scatter_unequal_splits_keep_zero_padding(
    monkeypatch, splits, rank
):
    backend = HcclBackend()
    process_group = object()
    monkeypatch.setattr(backend, "_process_group", lambda group: process_group)
    monkeypatch.setattr(hccl_module.dist, "get_rank", lambda: rank)
    value = torch.arange(sum(splits) * 4, dtype=torch.float32).reshape(-1, 4)
    expected_padded = torch.zeros(2 * max(splits), 4)
    expected_padded[: splits[0]].copy_(value[: splits[0]])
    expected_padded[max(splits) : max(splits) + splits[1]].copy_(value[splits[0] :])

    def reduce_scatter(output, source, group):
        assert group is process_group
        torch.testing.assert_close(source, expected_padded)
        output.copy_(source[rank * max(splits) : (rank + 1) * max(splits)] * 2)

    monkeypatch.setattr(hccl_module.dist, "reduce_scatter_tensor", reduce_scatter)
    result = backend.token_reduce_scatter(value, (0, 1), splits)
    start = sum(splits[:rank])
    torch.testing.assert_close(result, value[start : start + splits[rank]] * 2)


def test_token_reduce_scatter_single_rank_keeps_copy_semantics(monkeypatch):
    monkeypatch.setattr(hccl_module.dist, "get_rank", lambda: 0)
    value = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    result = HcclBackend().token_reduce_scatter(value, (0,), [3])
    torch.testing.assert_close(result, value)
    assert result.data_ptr() != value.data_ptr()
