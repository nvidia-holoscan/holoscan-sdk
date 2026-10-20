# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Partition, region, lifecycle, and deployment plan-parity tests."""

import gc
import os
import subprocess
from datetime import timedelta

import holoscan_test_support  # noqa: F401
import pytest

import holoscan as hs

COUNTER = "holoscan.test.CounterSource"
DOUBLER = "holoscan.test.DoubleIt"
SINK = "holoscan.test.Sink"


def pipeline_graph(name):
    """Compose the pipeline used by test_thread_placement_plan_matches_independent_cpp_authoring."""
    graph = hs.Graph(name)
    source = graph.op(COUNTER, "source", limit=4)
    doubler = graph.op(DOUBLER, "doubler")
    sink = graph.op(SINK, "sink")
    graph.add_flow(source.out, doubler["in"], queue_depth=4)
    graph.add_flow(doubler.out, sink["in"], queue_depth=4)
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    return graph, source, doubler, sink


def cpp_plan(mode, *arguments):
    """Load independent plan bytes for representative thread-placement parity."""
    helper = os.environ["HOLOSCAN_POLICY_PLAN_HELPER"]
    return subprocess.run(
        [helper, mode, *(str(argument) for argument in arguments)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


# REQ: PY-API-006
def test_thread_placement_plan_matches_independent_cpp_authoring():
    # Key setup: Place the complete pipeline in one sleeping-waitset partition.
    #
    #   [worker: source --uint32--> doubler --uint32--> sink]
    graph, source, doubler, sink = pipeline_graph("thread-placement")
    graph.partition("worker").add(source, doubler, sink)
    options = hs.CompileOptions()
    options.deployment.bind_thread_placement(
        "worker",
        hs.execution.ThreadPlacement(wake_mode=hs.execution.WakeMode.SLEEPING_WAITSET),
    )

    plan = hs.compile(graph, options)

    # Key assertion: Python placement authoring produces the independent C++ plan bytes.
    assert plan.ok
    assert plan.partitions.mode is hs.execution.PartitioningMode.MANUAL
    assert plan.json() == cpp_plan("thread")


# REQ: COMPILE-096
# REQ: COMPILE-099
# REQ: COMPILE-100
# REQ: PY-API-006
def test_support_thread_affinity_plan_matches_independent_cpp_authoring():
    # Key setup: Bind both Runtime support roles to CPUs available to this process.
    if not hasattr(os, "sched_getaffinity"):
        pytest.skip("support-thread affinity probing is supported only on Linux")
    cores = sorted(os.sched_getaffinity(0))
    if not cores:
        pytest.skip("the compiling thread has no effective CPUs")
    controller_cpu = cores[0]
    pump_cpu = cores[-1]
    graph, _, _, _ = pipeline_graph("support-thread-affinity")
    options = hs.CompileOptions()
    options.deployment.bind_support_thread_affinity(
        hs.execution.SupportThreadRole.CONTROLLER,
        hs.execution.CPUAffinity([controller_cpu]),
    ).bind_support_thread_affinity(
        hs.execution.SupportThreadRole.CHANNEL_PUMP,
        hs.execution.CPUAffinity([pump_cpu]),
    )

    plan = hs.compile(graph, options)

    # Key assertion: Python and independent C++ authoring emit byte-identical plan evidence.
    assert plan.ok
    assert plan.json() == cpp_plan("support-thread", controller_cpu, pump_cpu)


# REQ: PY-API-006
def test_green_context_plan_matches_independent_cpp_authoring():
    # Key setup: Split the pipeline into critical and background CUDA partitions.
    #
    #   [critical: source] --uint32--> [background: doubler --uint32--> sink]
    try:
        capability = hs.execution.query_cuda_green_context_capability(hs.execution.DeviceId(0))
    except hs.HoloscanError as error:
        pytest.skip(f"CUDA Green Context capability unavailable: {error}")
    if capability.structured_sm_quantum == 0 or capability.max_structured_context_count < 2:
        pytest.skip("target cannot provide a dedicated Green Context plus residual")

    graph, source, doubler, sink = pipeline_graph("green-context")
    graph.partition("critical").add(source)
    graph.partition("background").add(doubler, sink)
    options = hs.CompileOptions()
    options.deployment.bind_cuda_green_context(
        "critical",
        hs.execution.CudaGreenContextPlacement(
            device=capability.device,
            sm_count=capability.structured_sm_quantum,
        ),
    )

    plan = hs.compile(graph, options)

    # Key assertion: Supported Green Context placement matches independent C++ authoring.
    assert plan.ok
    assert plan.json() == cpp_plan("green")


# REQ: PY-API-007
@pytest.mark.parametrize("mode", ["automatic", "include-all"])
def test_automatic_partitioning_plans_match_independent_cpp_authoring(mode):
    # Key setup: Partition the pipeline through constrained-automatic or include-all authoring.
    #
    #   source --uint32--> doubler --uint32--> sink
    if mode == "automatic":
        graph, source, _, _ = pipeline_graph("automatic-partitioning")
        graph.partition("anchor").add(source)
        graph.auto_partition_remaining()
        expected_mode = hs.execution.PartitioningMode.CONSTRAINED_AUTOMATIC
    else:
        graph, _, _, _ = pipeline_graph("include-all")
        graph.partition("whole-application").include_all()
        expected_mode = hs.execution.PartitioningMode.MANUAL

    plan = hs.compile(graph)

    # Key assertion: Both partition modes preserve their expected mode and C++ plan bytes.
    assert plan.ok
    assert plan.partitions.mode is expected_mode
    assert plan.json() == cpp_plan(mode)


# REQ: PY-API-007
def test_variadic_partition_add_is_atomic_on_rejection():
    # Key setup: Reject a duplicate variadic member before adding the complete pipeline.
    #
    #   [worker: source --uint32--> doubler --uint32--> sink]
    graph, source, doubler, sink = pipeline_graph("partition-batch")
    partition = graph.partition("worker")

    with pytest.raises(hs.errors.InvalidArgumentError, match="GRAPH_PARTITION_MEMBER_DUPLICATE"):
        partition.add(source, source)

    partition.add(source, doubler, sink)

    # Key assertion: Rejection leaves no partial membership and the later complete add compiles.
    assert hs.compile(graph).ok


# REQ: PY-API-007
def test_variadic_region_add_is_atomic_on_rejection():
    # Key setup: Reject a duplicate region batch before adding the corrected membership.
    #
    #   source --uint32--> doubler --uint32--> sink
    #     |------ critical region ------|
    graph, source, doubler, _ = pipeline_graph("region-batch")
    region = graph.region("critical")

    with pytest.raises(hs.errors.DuplicateNameError, match="GRAPH_REGION_MEMBER_DUPLICATE"):
        region.add(source, doubler, source)

    region.add(source, doubler)

    # Key assertion: Rejection leaves no partial membership and the corrected add compiles.
    assert hs.compile(graph).ok


# REQ: PY-API-007
@pytest.mark.parametrize("policy", ["partition", "region"])
def test_variadic_policy_add_rejects_wrong_type_before_mutation(policy):
    # Key setup: Put a non-Operator in the middle of a policy batch, then retry with valid handles.
    #
    #   source --uint32--> doubler --uint32--> sink
    graph, source, doubler, sink = pipeline_graph(f"{policy}-type-error")
    handle = graph.partition("worker") if policy == "partition" else graph.region("critical")

    with pytest.raises(TypeError, match=r"argument 2 must be holoscan.OperatorHandle"):
        handle.add(source, "doubler", sink)

    handle.add(source, doubler, sink)

    # Key assertion: Type validation runs before Core mutation and the corrected batch succeeds.
    assert hs.compile(graph).ok


# REQ: PY-API-007
def test_policy_handles_do_not_extend_graph_lifetime():
    # Key setup: Retain all three policy handles and one Operator after releasing the owning Graph.
    #
    #   source
    graph = hs.Graph("retired-policy-handles")
    source = graph.op(COUNTER, "source", limit=1)
    region = graph.region("critical")
    partition = graph.partition("worker")
    lifecycle = graph.lifecycle()
    del graph
    gc.collect()

    # Key assertion: Each weak policy handle reports its Core-owned stale-handle token.
    with pytest.raises(hs.errors.InvalidArgumentError) as region_error:
        region.add(source)
    assert region_error.value.token == "GRAPH_REGION_HANDLE_INVALID"
    with pytest.raises(hs.errors.InvalidArgumentError) as partition_error:
        partition.add(source)
    assert partition_error.value.token == "GRAPH_PARTITION_HANDLE_INVALID"
    with pytest.raises(hs.errors.InvalidArgumentError) as lifecycle_error:
        lifecycle.allow_self_restart(source)
    assert lifecycle_error.value.token == "GRAPH_LIFECYCLE_HANDLE_INVALID"


# REQ: PY-API-007
def test_region_and_lifecycle_plan_matches_independent_cpp_authoring():
    # Key setup: Add region constraints and lifecycle edges to the canonical pipeline.
    #
    #   source --uint32--> doubler --uint32--> sink
    #     |------ critical region ------|
    graph, source, doubler, _ = pipeline_graph("graph-policies")
    (
        graph.region("critical")
        .add(source)
        .add(doubler)
        .constrain(hs.execution.RegionConstraint.static_topology())
        .constrain(hs.execution.RegionConstraint.bounded_memory())
        .constrain(hs.execution.RegionConstraint.device_compatible_ops())
        .constrain(hs.execution.RegionConstraint.latency_budget(hs.time.us(250)))
        .constrain(hs.execution.RegionConstraint.cuda_graph())
        .profile(hs.execution.Profile.ROBOTICS)
    )
    (
        graph.lifecycle()
        .before(source, doubler, hs.lifecycle.LifecycleStage.CONFIGURE)
        .allow_self_restart(source)
        .allow_external_restart(doubler)
        .allow_restart_request(source, doubler)
    )

    plan = hs.compile(graph)

    # Key assertion: The hard CUDA Graph constraint fails closed with exact C++ plan parity.
    assert not plan.ok
    assert plan.errors[0].token == "CUDA_GRAPH_REGION_INVALID"
    assert plan.json() == cpp_plan("policy")
    assert (
        hs.execution.RegionConstraint.latency_budget(timedelta(microseconds=250)).latency_ns
        == 250_000
    )
    with pytest.raises(hs.errors.InvalidArgumentError) as latency_error:
        hs.execution.RegionConstraint.latency_budget(250)
    assert latency_error.value.token == "GRAPH_REGION_LATENCY_INVALID"


# REQ: PY-API-007
@pytest.mark.parametrize(
    ("mode", "expected_token"),
    [
        ("partition-diagnostic", "PARTITION_COVER_INVALID"),
        ("region-diagnostic", "PROFILE_REGION_LOOSENS_GRAPH"),
    ],
)
def test_partition_and_region_diagnostics_match_cpp(mode, expected_token):
    # Key setup: Author an incomplete partition or graph-loosening region over the pipeline.
    #
    #   source --uint32--> doubler --uint32--> sink
    if mode == "partition-diagnostic":
        graph, source, _, _ = pipeline_graph("partition-diagnostic")
        graph.partition("incomplete").add(source)
    else:
        graph, source, _, _ = pipeline_graph("region-diagnostic")
        graph.region("looser").add(source).profile(hs.execution.Profile.RESEARCH)

    plan = hs.compile(graph)

    # Key assertion: Rejected Python and C++ plans retain the same stable diagnostic.
    assert not plan.ok
    assert any(d.token == expected_token for d in plan.errors)
    assert plan.json() == cpp_plan(mode)


# REQ: PY-API-006
def test_tensor_output_device_diagnostic_matches_cpp():
    # Key setup: Bind a Tensor device to the scalar output of the canonical pipeline source.
    #
    #   source --uint32--> doubler --uint32--> sink
    graph, _, _, _ = pipeline_graph("tensor-device-diagnostic")
    options = hs.CompileOptions()
    options.deployment.bind_tensor_output_device(
        operator_path="source", output_port="out", device=hs.execution.DeviceId(0)
    )

    plan = hs.compile(graph, options)

    # Key assertion: Invalid Tensor placement matches the independent C++ rejected plan.
    assert not plan.ok
    assert any(d.token == "PHYSICAL_TENSOR_DEVICE_BINDING_INVALID" for d in plan.errors)
    assert plan.json() == cpp_plan("tensor-diagnostic")
