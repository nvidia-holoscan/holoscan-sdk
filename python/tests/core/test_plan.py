# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compilation and immutable Plan binding tests."""

import gc
import json
import os
import subprocess

import holoscan_test_support  # noqa: F401
import pytest

import holoscan as hs

COUNTER = "holoscan.test.CounterSource"
DOUBLER = "holoscan.test.DoubleIt"
SINK = "holoscan.test.Sink"


def first_pipeline_graph():
    """Compose the graph used by the plan parity tests.

    Representative: test_plan_matches_independently_authored_cpp_and_outlives_graph.
    """
    graph = hs.Graph("first-pipeline")
    source = graph.op(COUNTER, "source", limit=10)
    doubler = graph.op(DOUBLER, "doubler")
    sink = graph.op(SINK, "sink")
    graph.add_flow(source.out, doubler["in"], queue_depth=4)
    graph.add_flow(doubler.out, sink["in"], queue_depth=4)
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    return graph, source


# REQ: PY-API-004
def test_plan_matches_independently_authored_cpp_and_outlives_graph():
    # Key setup: Compile the canonical pipeline and retain its plan beyond graph lifetime.
    #
    #   source --uint32--> doubler --uint32--> sink
    graph, source = first_pipeline_graph()
    plan = hs.compile(graph)

    assert plan.ok
    assert bool(plan)
    assert plan.errors == []
    assert plan.json()
    assert '"endpoints":[]' in plan.endpoint_catalog.json()
    assert len(plan.partitions) > 0
    assert plan.partitions.of(source).operator_paths

    helper = os.environ["HOLOSCAN_FIRST_PIPELINE_PLAN_HELPER"]
    cpp_json = subprocess.run(
        [helper],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    # Key assertion: Python authoring matches C++ bytes and the immutable plan owns its state.
    assert plan.json() == cpp_json

    canonical_json = plan.json()
    del source
    del graph
    gc.collect()
    assert plan.json() == canonical_json


# REQ: PY-API-004
def test_partition_lookup_uses_canonical_operator_paths():
    # Key setup: Compile an operator whose local name requires URI percent-encoding.
    #
    #   camera 1
    graph = hs.Graph("encoded-operator-path")
    source = graph.op(COUNTER, "camera 1", limit=1)
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    plan = hs.compile(graph)

    # Key assertion: Handle lookup uses the same canonical path stored in the plan.
    assert plan.ok
    assert plan.partitions.of(source).operator_paths == ["camera%201"]


# REQ: PY-API-004
def test_partition_lookup_preserves_operator_graph_identity_and_generation():
    # Key setup: Compile one graph while retaining another graph with the same local Operator name.
    graph, source = first_pipeline_graph()
    foreign_graph, foreign_source = first_pipeline_graph()
    plan = hs.compile(graph)
    assert plan.partitions.of(source).operator_paths

    # Key assertion: A plausible path from another Graph cannot select this plan's partition.
    with pytest.raises(hs.HoloscanError) as foreign_error:
        plan.partitions.of(foreign_source)
    assert foreign_error.value.code is hs.errors.ErrorCode.INCOMPATIBLE

    # Mutating the source Graph advances its generation, so its formerly matching handle is stale
    # relative to the immutable plan rather than being re-resolved by name.
    graph.op(SINK, "later")
    with pytest.raises(hs.HoloscanError) as stale_error:
        plan.partitions.of(source)
    assert stale_error.value.code is hs.errors.ErrorCode.STALE_OPERATOR_REF

    # A handle whose Graph has retired keeps the established weak-handle failure behavior.
    del foreign_graph
    gc.collect()
    with pytest.raises(hs.HoloscanError) as invalid_error:
        plan.partitions.of(foreign_source)
    assert invalid_error.value.code is hs.errors.ErrorCode.INVALID_ARGUMENT


# REQ: PY-API-004
def test_rejected_plan_exposes_complete_diagnostics_and_compile_error():
    # Key setup: Compile a clock-triggered source without a default clock.
    #
    #   source
    graph = hs.Graph("missing-clock")
    graph.op(COUNTER, "source")

    plan = hs.compile(graph)

    # Key assertion: Rejection remains inspectable data until require_no_errors raises it.
    assert not plan.ok
    assert not plan
    assert plan.errors
    assert all(d.severity is hs.diagnostics.DiagnosticSeverity.ERROR for d in plan.errors)
    assert all(d.severity is hs.diagnostics.DiagnosticSeverity.WARNING for d in plan.warnings)
    assert len(plan.diagnostics) == len(plan.errors) + len(plan.warnings)
    assert any(d.token == "TEMPORAL_CLOCK_UNDECLARED" for d in plan.errors)
    diagnostic = plan.errors[0]
    assert isinstance(diagnostic.id, int)
    assert diagnostic.severity is hs.diagnostics.DiagnosticSeverity.ERROR
    assert isinstance(diagnostic.token, str)
    assert isinstance(diagnostic.source.graph_generation, int)
    assert isinstance(diagnostic.source.graph, int)
    assert isinstance(diagnostic.source.vertex, int)
    assert isinstance(diagnostic.source.port, int)
    assert isinstance(diagnostic.service, str)
    assert isinstance(diagnostic.endpoint, str)
    assert isinstance(diagnostic.json_pointer, str)
    assert isinstance(diagnostic.required, str)
    assert isinstance(diagnostic.provided, str)
    assert diagnostic.cause_id is None or isinstance(diagnostic.cause_id, int)
    assert "error TEMPORAL_CLOCK_UNDECLARED" in plan.summary()
    assert "REJECTED" in repr(plan)

    with pytest.raises(hs.errors.CompileError) as caught:
        plan.require_no_errors()
    assert isinstance(caught.value, hs.HoloscanError)
    assert isinstance(caught.value, hs.errors.FailureError)
    assert caught.value.token == "EXECUTION_PLAN_REJECTED"


# REQ: PY-API-004
def test_diagnostic_detail_is_typed_read_only_and_complete():
    # Key setup: Obtain a Core-owned diagnostic fixture with every supplemental field populated.
    diagnostic = holoscan_test_support._structured_diagnostic_fixture()
    detail = diagnostic.detail

    # Key assertion: Python preserves the complete typed evidence without parsing rendered text.
    assert isinstance(detail, hs.diagnostics.DiagnosticDetail)
    assert detail.operation is hs.diagnostics.DiagnosticOperation.SETUP
    assert detail.error.code is hs.errors.ErrorCode.INVALID_ARGUMENT
    assert detail.error.message == "primary fixture cause"
    assert detail.stage is hs.lifecycle.LifecycleStage.WARM_UP
    assert detail.phase is hs.lifecycle.LifecycleTransitionPhase.RESTART_FORWARD
    assert detail.transaction_id == 41
    assert detail.role is hs.diagnostics.DiagnosticFailureRole.ROLLBACK
    assert (detail.operator_path, detail.port_name) == ("source", "out")
    assert (detail.related_source.vertex, detail.related_source.port) == (6, 8)
    assert (detail.related_operator_path, detail.related_port_name) == ("sink", "in")
    assert len(detail.mismatches) == 1
    assert (
        detail.mismatches[0].field,
        detail.mismatches[0].provided,
        detail.mismatches[0].required,
    ) == ("dtype", "float32", "uint8")
    assert len(detail.causes) == 1
    assert detail.causes[0].code is hs.errors.ErrorCode.NOT_FOUND
    assert detail.causes[0].message == "nested fixture cause"
    assert detail.primary_diagnostic_index == 2
    assert detail.truncated is True
    assert all(
        value.__doc__
        for value in (
            hs.diagnostics.DiagnosticDetail,
            hs.diagnostics.DiagnosticError,
            hs.diagnostics.DiagnosticFailureRole,
            hs.diagnostics.DiagnosticMismatch,
            hs.diagnostics.DiagnosticOperation,
            hs.lifecycle.LifecycleTransitionPhase,
        )
    )
    with pytest.raises(AttributeError):
        detail.operator_path = "replacement"


# REQ: PY-API-004
# REQ: PY-API-023
def test_clean_plan_filters_diagnostics_and_options_are_value_objects():
    # Key setup: Compile the canonical pipeline with explicit lifecycle and queue options.
    #
    #   source --uint32--> doubler --uint32--> sink
    lifecycle = hs.lifecycle.LifecycleTransactionOptions(
        max_pending_requests=1,
        max_pending_transactions=2,
        max_receipts=3,
        max_fenced_operators=4,
        max_candidate_generations=5,
        max_events=6,
        max_audit_records=7,
        max_quiesce_records=8,
        max_rollback_records=9,
    )
    parameters = hs.params.ParameterCompileOptions()
    parameters.runtime_updates = hs.params.ParamRuntimeUpdatePolicy.ENABLED_IF_ADMITTED
    parameters.max_event_cursors = 2
    deployment = hs.execution.DeploymentBinding()
    options = hs.CompileOptions(
        deployment=deployment,
        profile=hs.execution.Profile.STANDARD,
        parameters=parameters,
        opaque_object_payloads=hs.OpaqueObjectPayloadPolicy.WARN,
        lifecycle_transactions=lifecycle,
        automatic_queue_sizing=hs.execution.AutomaticQueueSizingOptions(
            headroom_ratio=hs.execution.RateRatio(2, 1)
        ),
        cuda_execution=hs.execution.CudaExecutionCompileOptions(
            compute_stream_policy=hs.execution.CudaComputeStreamPolicy.SHARED_PROCESS_DEVICE
        ),
    )
    graph, _ = first_pipeline_graph()
    plan = hs.compile(graph, options)

    # Key assertion: A clean plan filters diagnostics and preserves every options value.
    assert plan.ok
    assert plan.diagnostics == []
    assert plan.errors == []
    assert plan.warnings == []
    assert bool(plan)
    assert options.profile is hs.execution.Profile.STANDARD
    assert (
        options.cuda_execution.compute_stream_policy
        is hs.execution.CudaComputeStreamPolicy.SHARED_PROCESS_DEVICE
    )
    assert (
        plan.cuda_compute_stream_policy
        is hs.execution.CudaComputeStreamPolicy.SHARED_PROCESS_DEVICE
    )
    assert json.loads(plan.json())["cuda_compute_stream_policy"] == 1
    assert options.opaque_object_payloads is hs.OpaqueObjectPayloadPolicy.WARN
    assert isinstance(options.deployment, hs.execution.DeploymentBinding)
    assert (
        options.parameters.runtime_updates is hs.params.ParamRuntimeUpdatePolicy.ENABLED_IF_ADMITTED
    )
    assert options.parameters.max_event_cursors == 2
    options.opaque_object_payloads = hs.OpaqueObjectPayloadPolicy.REJECT
    assert options.opaque_object_payloads is hs.OpaqueObjectPayloadPolicy.REJECT
    assert hs.CompileOptions().opaque_object_payloads is hs.OpaqueObjectPayloadPolicy.ALLOW
    assert options.automatic_queue_sizing.headroom_ratio.numerator == 2
    assert options.automatic_queue_sizing.headroom_ratio.denominator == 1
    assert options.lifecycle_transactions.max_pending_requests == 1
    assert options.lifecycle_transactions.max_rollback_records == 9
    options.lifecycle_transactions.max_events = 11
    assert options.lifecycle_transactions.max_events == 11
    options.lifecycle_transactions = None
    assert options.lifecycle_transactions is None
    options.lifecycle_transactions = lifecycle
    assert options.lifecycle_transactions.max_events == 6
    assert hs.CompileOptions().lifecycle_transactions is None
    assert (
        hs.CompileOptions().cuda_execution.compute_stream_policy
        is hs.execution.CudaComputeStreamPolicy.PER_EXECUTION_PARTITION
    )

    with pytest.raises(TypeError):
        hs.lifecycle.LifecycleTransactionOptions()
