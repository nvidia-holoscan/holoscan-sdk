# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployment value and CompileOptions binding tests."""

import gc
import json
import os

import holoscan_test_support  # noqa: F401
import pytest

import holoscan as hs

COUNTER = "holoscan.tests.CounterSource"


def deployment_graph():
    """Compose the source graph used by test_deployment_binders_reach_core_compile_diagnostics."""
    graph = hs.Graph("deployment-values")
    source = graph.op(COUNTER, "source", limit=2)
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    return graph, source


def effective_cpu_ids():
    """Return the compiling thread's effective Linux CPU set or skip the placement test."""
    if not hasattr(os, "sched_getaffinity"):
        pytest.skip("support-thread affinity probing is supported only on Linux")
    cores = sorted(os.sched_getaffinity(0))
    if not cores:
        pytest.skip("the compiling thread has no effective CPUs")
    return cores


# REQ: PY-API-006
def test_cpu_affinity_accepts_python_core_ids_ranges_and_tuples():
    # Key setup: Construct affinities from IDs, ranges, tuples, mixed forms, and invalid inputs.

    # Key assertion: Accepted forms normalize to sorted unique cores and invalid forms fail closed.
    assert hs.execution.CPUAffinity([5, 3, 5, 1]).cores == (1, 3, 5)
    assert hs.execution.CPUAffinity(range(2, 5)).cores == (2, 3, 4)
    assert hs.execution.CPUAffinity([(2, 4), (7, 8)]).cores == (2, 3, 4, 7, 8)
    assert hs.execution.CPUAffinity([1, (3, 4), 3]).cores == (1, 3, 4)
    assert hs.execution.CPUAffinity.from_ranges([(9, 10)]).cores == (9, 10)
    assert hs.execution.CPUAffinity().empty
    assert "[1, 3, 5]" in repr(hs.execution.CPUAffinity([5, 1, 3]))

    with pytest.raises(hs.errors.InvalidArgumentError) as range_error:
        hs.execution.CPUAffinity([(4, 2)])
    assert range_error.value.token == "CPU_AFFINITY_RANGE_INVALID"
    with pytest.raises(hs.errors.InvalidArgumentError):
        hs.execution.CPUAffinity([True])
    with pytest.raises(hs.errors.InvalidArgumentError) as iterable_error:
        hs.execution.CPUAffinity("not an affinity")
    assert iterable_error.value.token == "CPU_AFFINITY_ITERABLE_REQUIRED"
    with pytest.raises(hs.errors.InvalidArgumentError, match="CPU_AFFINITY_RANGE_INVALID"):
        hs.execution.CPUAffinity.from_ranges([1, 2])

    affinity = hs.execution.CPUAffinity([1, 2])
    with pytest.raises(AttributeError):
        affinity.cores.append(3)


# REQ: PY-API-006
def test_thread_and_cuda_deployment_values_preserve_core_fields():
    # Key setup: Construct fully populated thread and CUDA Green Context placement values.
    deadline = hs.execution.DeadlineScheduling(runtime_ns=10, deadline_ns=20, period_ns=30)
    placement = hs.execution.ThreadPlacement(
        affinity=hs.execution.CPUAffinity([2, 4]),
        policy=hs.execution.SchedulingPolicy.DEADLINE,
        deadline=deadline,
        wake_mode=hs.execution.WakeMode.BUSY_POLL,
        irq_isolation=hs.execution.IRQIsolation.PREFER_ISOLATED,
        probe_failure_policy=hs.topics.ProbeFailurePolicy.FAIL_ADMISSION,
    )

    # Key assertion: Python value objects preserve every Core field and validation rule.
    assert placement.affinity.cores == (2, 4)
    assert placement.policy is hs.execution.SchedulingPolicy.DEADLINE
    assert placement.scheduling.policy is hs.execution.SchedulingPolicy.DEADLINE
    assert placement.deadline.period_ns == 30
    assert placement.wake_mode is hs.execution.WakeMode.BUSY_POLL
    assert placement.irq_isolation is hs.execution.IRQIsolation.PREFER_ISOLATED
    assert placement.probe_failure_policy is hs.topics.ProbeFailurePolicy.FAIL_ADMISSION

    placement.deadline.period_ns = 40
    assert placement.deadline.period_ns == 40
    assert placement.scheduling.deadline.period_ns == 40

    device = hs.execution.DeviceId(0)
    green = hs.execution.CudaGreenContextPlacement(device=device, sm_count=16)
    assert device.valid
    assert green.device.value == 0
    assert green.sm_count == 16
    assert not hs.execution.DeviceId(-1).valid

    with pytest.raises(hs.errors.InvalidArgumentError) as capability_error:
        hs.execution.query_cuda_green_context_capability(hs.execution.DeviceId(-1))
    assert capability_error.value.token == "CUDA_GREEN_CONTEXT_INVALID_PLACEMENT"


# REQ: PY-API-006
def test_deployment_values_and_capability_query_have_public_docstrings():
    # Key setup: Enumerate the deployment classes and function added by this slice.
    values = (
        hs.execution.DeviceId,
        hs.execution.CPUAffinity,
        hs.execution.SupportThreadRole,
        hs.execution.DeadlineScheduling,
        hs.execution.ThreadScheduling,
        hs.execution.ThreadPlacement,
        hs.execution.CudaGreenContextPlacement,
        hs.execution.CudaGreenContextCapability,
        hs.execution.DeploymentBinding,
        hs.execution.RateRatio,
        hs.execution.AutomaticQueueSizingOptions,
        hs.lifecycle.LifecycleTransactionOptions,
        hs.execution.query_cuda_green_context_capability,
    )

    # Key assertion: Every public deployment entry point provides concise API help.
    assert all(value.__doc__ for value in values)
    assert hs.execution.DeploymentBinding.bind_support_thread_affinity.__doc__


# REQ: COMPILE-096
# REQ: PY-API-006
def test_support_thread_affinity_is_fluent_owned_and_accepts_optional_pump():
    # Key setup: Bind both closed roles with keyword arguments, then release the authored values.
    # This graph needs no process-local Channel pump, so the pump binding is accepted but dormant.
    cores = effective_cpu_ids()
    authored_cores = [cores[0]]
    affinity = hs.execution.CPUAffinity(authored_cores)
    options = hs.CompileOptions()
    deployment = options.deployment

    returned = deployment.bind_support_thread_affinity(
        role=hs.execution.SupportThreadRole.CONTROLLER,
        affinity=affinity,
    )
    assert returned is deployment
    returned = returned.bind_support_thread_affinity(
        role=hs.execution.SupportThreadRole.CHANNEL_PUMP,
        affinity=hs.execution.CPUAffinity([cores[-1]]),
    )
    assert returned is deployment
    authored_cores.clear()
    del affinity
    gc.collect()

    graph, _ = deployment_graph()
    plan = hs.compile(graph, options)
    records = json.loads(plan.json())["support_thread_affinities"]

    # Key assertion: The enum is stable and Core retains canonical, deep-owned evidence by role.
    assert hs.execution.SupportThreadRole.CONTROLLER.value == 0
    assert hs.execution.SupportThreadRole.CHANNEL_PUMP.value == 1
    assert plan.ok
    assert [record["role"] for record in records] == [0, 1]
    assert records[0]["requested"]["affinity"] == [cores[0]]
    assert records[0]["accepted"]["affinity"] == [cores[0]]
    assert records[1]["requested"]["affinity"] == [cores[-1]]
    assert records[1]["accepted"]["affinity"] == [cores[-1]]


# REQ: COMPILE-096
# REQ: PY-API-006
def test_support_thread_affinity_rejects_duplicate_empty_and_unavailable_requests():
    # Key setup: Compile one duplicate role, one empty affinity, and one unavailable CPU request.
    cores = effective_cpu_ids()

    graph, _ = deployment_graph()
    duplicate = hs.CompileOptions()
    duplicate.deployment.bind_support_thread_affinity(
        hs.execution.SupportThreadRole.CONTROLLER,
        hs.execution.CPUAffinity([cores[0]]),
    ).bind_support_thread_affinity(
        hs.execution.SupportThreadRole.CONTROLLER,
        hs.execution.CPUAffinity([cores[-1]]),
    )
    duplicate_plan = hs.compile(graph, duplicate)
    duplicate_error = next(
        error
        for error in duplicate_plan.errors
        if error.token == "SUPPORT_THREAD_PLACEMENT_UNKNOWN_OR_DUPLICATE"
    )
    duplicate_mismatch = duplicate_error.detail.mismatches[0]

    graph, _ = deployment_graph()
    empty = hs.CompileOptions()
    empty.deployment.bind_support_thread_affinity(
        hs.execution.SupportThreadRole.CHANNEL_PUMP,
        hs.execution.CPUAffinity(),
    )
    empty_plan = hs.compile(graph, empty)
    empty_error = next(
        error for error in empty_plan.errors if error.token == "THREAD_PLACEMENT_REJECTED"
    )
    empty_mismatch = empty_error.detail.mismatches[0]

    available = set(cores)
    unavailable_cpu = next(cpu for cpu in range(max(available) + 2) if cpu not in available)
    graph, _ = deployment_graph()
    unavailable = hs.CompileOptions()
    unavailable.deployment.bind_support_thread_affinity(
        hs.execution.SupportThreadRole.CHANNEL_PUMP,
        hs.execution.CPUAffinity([unavailable_cpu]),
    )
    unavailable_plan = hs.compile(graph, unavailable)
    unavailable_error = next(
        error for error in unavailable_plan.errors if error.token == "THREAD_PLACEMENT_REJECTED"
    )
    unavailable_mismatch = unavailable_error.detail.mismatches[0]

    # Key assertion: Python reaches the same stable tokens and structured C++ mismatch evidence.
    assert not duplicate_plan
    assert duplicate_error.detail.operator_path == "support-thread/controller"
    assert duplicate_mismatch.field == "support_thread_role"
    assert duplicate_mismatch.provided == "role=controller,bindings=2"
    assert duplicate_mismatch.required == "at most one affinity binding per support-thread role"
    assert not empty_plan
    assert empty_error.detail.operator_path == "support-thread/channel_pump"
    assert empty_mismatch.field == "support_thread_affinity"
    assert empty_mismatch.provided == "role=channel_pump,requested=[]"
    assert empty_mismatch.required == "a nonempty affinity-only placement"
    assert not unavailable_plan
    assert unavailable_error.detail.operator_path == "support-thread/channel_pump"
    assert unavailable_mismatch.field == "support_thread_affinity"
    assert f"role=channel_pump,requested=[{unavailable_cpu}]" in unavailable_mismatch.provided
    assert unavailable_mismatch.required == (
        "requested CPUs must be in the compiling thread's effective affinity"
    )


# REQ: PY-API-006
def test_deployment_binders_reach_core_compile_diagnostics():
    # Key setup: Bind invalid thread, Green Context, and Tensor-device targets in separate graphs.
    #
    #   source
    graph, _ = deployment_graph()
    options = hs.CompileOptions()
    returned = options.deployment.bind_thread_placement("missing", hs.execution.ThreadPlacement())
    assert isinstance(returned, hs.execution.DeploymentBinding)
    plan = hs.compile(graph, options)
    assert not plan
    assert any(d.token == "PARTITION_PLACEMENT_UNKNOWN_OR_DUPLICATE" for d in plan.errors)

    graph, _ = deployment_graph()
    options = hs.CompileOptions()
    options.deployment.bind_cuda_green_context(
        "missing",
        hs.execution.CudaGreenContextPlacement(device=hs.execution.DeviceId(0), sm_count=1),
    )
    plan = hs.compile(graph, options)
    assert not plan
    assert any(d.token == "CUDA_GREEN_CONTEXT_UNKNOWN_OR_DUPLICATE" for d in plan.errors)

    graph, _ = deployment_graph()
    deployment = hs.execution.DeploymentBinding()
    deployment.bind_tensor_output_device(
        operator_path="source", output_port="out", device=hs.execution.DeviceId(0)
    )
    plan = hs.compile(graph, hs.CompileOptions(deployment=deployment))

    # Key assertion: Each binding reaches Core and retains its stable compile diagnostic token.
    assert not plan
    assert any(d.token == "PHYSICAL_TENSOR_DEVICE_BINDING_INVALID" for d in plan.errors)
    assert all(d.severity is hs.diagnostics.DiagnosticSeverity.ERROR for d in plan.errors)
