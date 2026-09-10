# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Contracts for optional fused exchange, resource isolation and shared overlap."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.ops.moe.gmoe import (
    GMoEContext,
    select_gmoe_stages,
)
from tokenspeed_kernel_npu.ops import gmoe_comm as backend

from tokenspeed.runtime.distributed.process_group_manager import ProcessGroupManager


def test_dedicated_group_isolation_and_creation_order(monkeypatch):
    import tokenspeed.runtime.distributed.process_group_manager as manager_module

    created = []
    monkeypatch.setattr(manager_module.dist, "get_world_size", lambda: 32)

    def new_group(ranks, **kwargs):
        pg = object()
        created.append((ranks, kwargs, pg))
        return pg

    monkeypatch.setattr(manager_module.dist, "new_group", new_group)
    manager = ProcessGroupManager()
    manager._device_backend = "hccl"
    ranks = tuple(range(8, 16))
    first = manager.get_dedicated_device_group(ranks, "exchange")
    assert first is created[1][2]
    assert [x[0] for x in created] == [
        tuple(range(start, start + 8)) for start in range(0, 32, 8)
    ]
    assert manager.get_dedicated_device_group(ranks, "exchange") is first
    assert len(created) == 4
    assert not manager.has_process_group("hccl", ranks)


@pytest.fixture
def preparation(monkeypatch, tmp_path):
    events = []
    library = tmp_path / "runtime.so"
    library.touch()
    monkeypatch.setattr(backend, "_active", None)
    npu = SimpleNamespace(
        get_device_name=lambda device: "Ascend910B2C",
        get_device_limit=lambda device: {"cube_core_num": 24, "vector_core_num": 48},
        Stream=lambda **kwargs: object(),
        set_stream_limit=lambda *args, **kwargs: events.append(("limit", kwargs)),
        synchronize=lambda: events.append(("sync",)),
    )
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    for name in (
        "gmoe_dispatch",
        "gmoe_combine",
        "initialize_group_first_rdma",
        "finalize_group_first_rdma",
    ):

        def call(*args, _name=name, **kwargs):
            events.append((_name, args, kwargs))

        monkeypatch.setattr(torch.ops.custom, name, call, raising=False)
    monkeypatch.setattr(
        torch.ops.custom.gmoe_dispatch,
        "router",
        lambda *a, **kw: events.append(("router", a, kw)),
        raising=False,
    )
    monkeypatch.setattr(backend.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(backend.dist, "get_world_size", lambda group: 16)
    pg = SimpleNamespace(
        _get_backend=lambda device: SimpleNamespace(
            get_hccl_comm_name=lambda rank: "dedicated"
        )
    )

    def create(ranks, namespace):
        events.append(("group", ranks, namespace))
        return pg

    args = dict(
        ep_group=tuple(range(16)),
        num_groups=8,
        num_experts=384,
        top_k=16,
        device="npu:0",
        create_group=create,
        options=dict(rdma_library=str(library), rendezvous="tcp://localhost:43211"),
    )
    return args, events


def test_prepare_once_and_collective_close(preparation):
    args, events = preparation
    first = backend.prepare_gmoe_exchange(**args)
    assert backend.prepare_gmoe_exchange(**args) is first
    assert len([e for e in events if e[0] == "group"]) == 1
    assert len([e for e in events if e[0] == "initialize_group_first_rdma"]) == 1
    assert first.shared_stream is None
    assert not any(event[0] == "limit" for event in events)
    assert first.zero_cores == 8
    with pytest.raises(RuntimeError, match="one compatible"):
        backend.prepare_gmoe_exchange(
            **{
                **args,
                "options": {**args["options"], "zero_cores": 4},
            }
        )
    first.close()
    first.close()
    assert [e[0] for e in events[-2:]] == ["sync", "finalize_group_first_rdma"]
    with pytest.raises(RuntimeError, match="closed"):
        first.check_shape(32, 512)


def test_explicit_shared_overlap_retains_limited_stream(preparation):
    args, events = preparation
    args["options"]["shared_overlap"] = True
    exchange = backend.prepare_gmoe_exchange(**args)
    assert exchange.shared_stream is not None
    assert any(event[0] == "limit" for event in events)


def test_default_shared_runs_inline_without_wait(preparation):
    args, _ = preparation
    exchange = backend.prepare_gmoe_exchange(**args)
    hidden = torch.ones(2, 4)
    output, wait = exchange.run_shared(lambda x: x + 1, hidden)
    assert torch.equal(output, hidden + 1)
    assert wait is None


def test_main_stream_must_not_alias_limited_shared(preparation, monkeypatch):
    args, _ = preparation
    resource = backend.prepare_gmoe_exchange(**args)
    shared = SimpleNamespace(device="npu:0")
    resource.shared_stream = shared
    monkeypatch.setattr(torch.npu, "current_stream", lambda **kw: shared, raising=False)
    with pytest.raises(RuntimeError, match="aliases the limited shared stream"):
        resource.check_execution_stream()
    monkeypatch.setattr(torch.npu, "current_stream", lambda **kw: object())
    resource.check_execution_stream()


@pytest.mark.parametrize(
    "change,match",
    [
        ({"ep_group": tuple(range(4))}, "contiguous"),
        ({"ep_group": tuple(range(1, 17))}, "contiguous"),
        ({"num_groups": 1}, "contiguous"),
        ({"num_groups": 3}, "contiguous"),
        ({"top_k": 65}, "top_k"),
    ],
)
def test_invalid_topology_fails_before_collectives(preparation, change, match):
    args, events = preparation
    with pytest.raises(ValueError, match=match):
        backend.prepare_gmoe_exchange(**{**args, **change})
    assert not events


@pytest.mark.parametrize(
    "options,match",
    [
        ({"rendezvous": None}, "independent"),
        ({"rendezvous": "localhost:42"}, "tcp://"),
        ({"window_bytes": 1024}, "window_bytes"),
        ({"zero_cores": 9}, "zero_cores"),
        ({"shared_vector_cores": 48}, "dispatch"),
        ({"shared_cube_cores": 24}, "cube"),
    ],
)
def test_invalid_options_fail_before_collectives(preparation, options, match):
    args, events = preparation
    args["options"]["shared_overlap"] = True
    args["options"].update(options)
    with pytest.raises(ValueError, match=match):
        backend.prepare_gmoe_exchange(**args)
    assert not events


def test_capacity_checked_without_launch(preparation):
    args, events = preparation
    resource = backend.prepare_gmoe_exchange(**args)
    resource.check_shape(32, 512)
    count = len(events)
    for tokens, hidden in ((0, 512), (32, 513), (65536, 512), (65535, 32768)):
        with pytest.raises(ValueError):
            resource.check_shape(tokens, hidden)
    assert len(events) == count


def test_chunked_binding_accepts_large_logical_tokens(preparation, monkeypatch):
    args, events = preparation
    monkeypatch.setattr(
        torch.ops.custom, "gmoe_comm_capabilities", lambda: 1, raising=False
    )
    resource = backend.prepare_gmoe_exchange(**args)
    assert resource.token_chunking
    for tokens in (1024, 65535, 65536, 1_000_000):
        resource.check_shape(tokens, 512)
        resource.check_router_shape(tokens, 512, 16)
    resource.window_bytes = 32768
    with pytest.raises(ValueError, match="window"):
        resource.check_shape(1, 512)


@pytest.mark.parametrize("world,groups", [(8, 8), (16, 8), (16, 4)])
def test_zero_gets_half_of_launch_by_default(preparation, monkeypatch, world, groups):
    args, _ = preparation
    args.update(ep_group=tuple(range(world)), num_groups=groups)
    monkeypatch.setattr(backend.dist, "get_world_size", lambda group: world)
    resource = backend.prepare_gmoe_exchange(**args)
    assert resource.zero_cores == 8


def test_fused_selection_is_opt_in():
    default = select_gmoe_stages(input_dtype=torch.bfloat16, traits={})
    assert default.pre.name == "composed_gmoe_pre"
    assert default.post.name == "composed_gmoe_post"


def test_router_core_budget_and_post_reuse(preparation):
    args, events = preparation
    args["options"]["shared_overlap"] = True
    args["options"]["router_cores"] = 32
    resource = backend.prepare_gmoe_exchange(**args)
    assert resource.router_cores == 32
    assert ("limit", {"cube_num": 4, "vector_num": 8}) in events
    from tokenspeed_kernel.ops.moe import gmoe_ascend as stages

    context = SimpleNamespace(exchange=resource)
    assert stages.prepare(context, device="npu:0", options=args["options"]) is context


@pytest.mark.parametrize(
    "options",
    [
        {"router_cores": 32, "shared_cube_cores": 8},
        {"router_cores": 32, "shared_vector_cores": 10},
        {"router_cores": 33},
        {"router_cores": 3},
        {"router_cores": -2},
    ],
)
def test_router_rejects_oversubscribed_cores(preparation, options):
    args, events = preparation
    args["options"]["shared_overlap"] = True
    args["options"].update(options)
    with pytest.raises(ValueError, match="cores"):
        backend.prepare_gmoe_exchange(**args)
    assert not events


def test_router_metadata_window_capacity(preparation):
    args, events = preparation
    args["options"].update(router_cores=32, window_bytes=180000)
    resource = backend.prepare_gmoe_exchange(**args)
    resource.check_shape(32, 64)  # Pure exchange fits; TopK64 metadata does not.
    with pytest.raises(ValueError, match="metadata"):
        resource.check_router_shape(32, 64, 64)
    with pytest.raises(ValueError, match="aligned"):
        resource.check_router_shape(1, 80, 16)


def test_router_wide_hidden_contract(preparation):
    args, _ = preparation
    args["options"]["window_bytes"] = 32 * 1024 * 1024
    resource = backend.prepare_gmoe_exchange(**args)
    for hidden in (2048, 4096, 8192):
        resource.check_router_shape(1, hidden, 16)
    for hidden in (8256, 16384):
        with pytest.raises(ValueError, match="8192"):
            resource.check_router_shape(1, hidden, 16)


def test_missing_router_overload_fails_before_collectives(preparation, monkeypatch):
    args, events = preparation
    args["options"]["router_cores"] = 32
    monkeypatch.delattr(torch.ops.custom.gmoe_dispatch, "router")
    with pytest.raises(RuntimeError, match="gmoe_dispatch.router"):
        backend.prepare_gmoe_exchange(**args)
    assert not events


def test_router_passes_existing_weights_and_policy(preparation):
    args, events = preparation
    args["options"]["router_cores"] = 32
    resource = backend.prepare_gmoe_exchange(**args)
    router = SimpleNamespace(
        classifier=SimpleNamespace(weight=torch.empty(416, 512, dtype=torch.float32)),
        e_score_correction_bias=torch.empty(416, dtype=torch.float32),
    )
    x = torch.empty(32, 8, 512, dtype=torch.bfloat16)
    resource.exchange_router(x, router, 16, 2.5, True)
    name, positional, keywords = events[-1]
    assert name == "router"
    assert positional[0] is x
    assert positional[1] is router.classifier.weight
    assert positional[2] is router.e_score_correction_bias
    assert positional[3:] == ("dedicated", 16, 2, 16)
    assert keywords == dict(
        routed_scaling_factor=2.5, renormalize=True, router_cores=32, overlap=True
    )


@pytest.mark.parametrize(
    "change", ["experts", "hidden", "dtype", "bias", "topk", "scale", "cores"]
)
def test_router_invalid_weights_fail_before_resource_setup(monkeypatch, change):
    from tokenspeed_kernel.ops.moe import gmoe_ascend as stages

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid router must fail before resource setup")

    monkeypatch.setattr(stages, "prepare", forbidden)
    weight = torch.empty(
        1025 if change == "experts" else 416,
        80 if change == "hidden" else 512,
        dtype=torch.bfloat16 if change == "dtype" else torch.float32,
    )
    context = SimpleNamespace(
        router=SimpleNamespace(
            classifier=SimpleNamespace(weight=weight),
            e_score_correction_bias=torch.empty(
                415 if change == "bias" else weight.shape[0], dtype=torch.float32
            ),
        ),
        top_k=65 if change == "topk" else 16,
        routed_scaling_factor=float("nan") if change == "scale" else 6.0,
    )
    with pytest.raises(ValueError, match="Fused router"):
        stages.prepare_router(
            context,
            device="cpu",
            options={"router_cores": 0} if change == "cores" else {},
        )


@pytest.mark.parametrize("fused_router", [False, True])
def test_router_selection_remains_explicit(fused_router):
    stages = select_gmoe_stages(
        input_dtype=torch.bfloat16,
        traits={"gmoe_exchange_enabled": True},
        pre_solution="flash_npu_router" if fused_router else None,
        post_solution="flash_npu",
    )
    assert stages.pre.name == (
        "flash_npu_router_gmoe_pre" if fused_router else "flash_npu_gmoe_pre"
    )


@pytest.mark.parametrize(
    "fused_router,fused_shared", [(False, False), (True, False), (True, True)]
)
def test_pre_joins_shared_after_dispatch_before_experts(fused_router, fused_shared):
    from tokenspeed_kernel.ops.moe import gmoe_ascend as stages

    if not hasattr(stages, "gmoe_pre"):
        pytest.skip("Ascend registration unavailable on this host")
    events = []
    groups, egp, tokens, hidden = 8, 2, 3, 512
    x = torch.randn(tokens, groups * hidden, dtype=torch.bfloat16)

    def shared(module, value):
        assert value is x
        events.append("shared_fork")
        return module(value), lambda: events.append("shared_join")

    def projection(value):
        events.append("projection")
        return value

    def norm(value):
        events.append("norm")
        return value

    def exchange(value):
        events.append("dispatch")
        return value.permute(1, 0, 2).flatten(0, 1).repeat(egp, 1)

    def route(value):
        events.append("router")
        return torch.ones(value.shape[0], 2), torch.zeros(
            value.shape[0], 2, dtype=torch.int32
        )

    def forbidden(*args):
        raise AssertionError("Fused stages must not call old framework collectives")

    def exchange_router(value, *args):
        received = exchange(value)
        weights, ids = route(received)
        return received, weights, ids

    resource = SimpleNamespace(
        check_execution_stream=lambda: None,
        check_shape=lambda *a: None,
        check_router_shape=lambda *a: None,
        run_shared=shared,
        exchange=exchange,
        exchange_router=exchange_router,
    )
    identity = torch.nn.Identity()
    context = GMoEContext(
        num_groups=groups,
        egp_group=(0, 1),
        egp_rank=1,
        exchange_group=tuple(range(groups)),
        num_experts=384,
        norm_scale=1,
        proj_input=projection,
        norm=norm,
        shared_experts=identity,
        proj_output=identity,
        route=forbidden if fused_router else route,
        router=identity,
        top_k=2,
        routed_scaling_factor=1,
        renormalize_topk=False,
        all_to_all=forbidden,
        all_gather=forbidden,
        reduce_scatter=forbidden,
        exchange=resource,
        prepared_shared=lambda value: value * 2,
    )
    pre = (
        stages.gmoe_router_ffn_pre
        if fused_shared
        else stages.gmoe_router_pre if fused_router else stages.gmoe_pre
    )
    result = pre(hidden_states=x, context=context)
    assert events == [
        "projection",
        "shared_fork",
        "norm",
        "dispatch",
        "router",
        "shared_join",
    ]
    assert result.local_received.storage_offset() == groups * tokens * hidden
    assert (
        result.local_received.untyped_storage().data_ptr()
        == result.received.untyped_storage().data_ptr()
    )
    assert torch.equal(result.shared_output, x * 2 if fused_shared else x)


def test_shared_ffn_selection_is_explicit():
    stages = select_gmoe_stages(
        input_dtype=torch.bfloat16,
        traits={"gmoe_exchange_enabled": True},
        pre_solution="flash_npu_router_ffn",
        post_solution="flash_npu",
    )
    assert stages.pre.name == "flash_npu_router_ffn_gmoe_pre"
    assert stages.post.name == "flash_npu_gmoe_post"
