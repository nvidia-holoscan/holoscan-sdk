# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hosted Python Operator lifecycle callback coverage."""

import threading

import holoscan_test_support  # noqa: F401  # Register the C++ source fixture.
import pytest

import holoscan as hs

COUNTER = "holoscan.tests.CounterSource"


def diagnostic_error_messages(diagnostics):
    """Return owned error messages from structured diagnostics that carry details."""
    return [item.detail.error.message for item in diagnostics if item.detail is not None]


# REQ: PY-OP-009
def test_named_lifecycle_callbacks_receive_phase_parameters_and_expire():
    """Exercise cold name resolution and callback-lifetime lifecycle views."""

    # Key setup: Register required and optional stage methods by name on a Python sink.
    #
    #   source[C++] --uint32--> probe[Python]
    class LifecycleProbe(hs.Operator):
        inp = hs.Input("in", hs.payload.uint32)
        amount = hs.Param(3)
        shape = hs.Param([2, 3], max_elements=2, recompile_required=True)
        observations = []
        retained_context = None
        computed = threading.Event()

        def setup(self, spec):
            spec.input(self.inp, queue_depth=2)
            spec.lifecycle().stage(hs.lifecycle.LifecycleStage.CONFIGURE, "configure")
            spec.lifecycle().optional(hs.lifecycle.LifecycleStage.ALLOCATE, "allocate")
            spec.lifecycle().stage(hs.lifecycle.LifecycleStage.RELEASE, "release")

        def configure(self, context):
            restart_target = context.restart_target("probe")
            type(self).observations.append(
                (
                    "configure",
                    context.params,
                    context.params_generation,
                    context.transition_phase,
                    context.stop_requested,
                    restart_target.id,
                    restart_target.run_generation,
                )
            )
            type(self).retained_context = context
            return hs.lifecycle.LifecycleStatus.OK

        def allocate(self, context):
            type(self).observations.append(("allocate", context.params))
            return hs.lifecycle.LifecycleStatus.SKIPPED

        def release(self, context):
            type(self).observations.append(("release", context.transition_phase))

        def compute(self, context):
            self.inp.receive()
            type(self).computed.set()

    LifecycleProbe.observations = []
    LifecycleProbe.retained_context = None
    LifecycleProbe.computed.clear()
    graph = hs.Graph("native-lifecycle-hooks")
    source = graph.op(COUNTER, "source", limit=1)
    probe = graph.op(LifecycleProbe, "probe", amount=5, shape=[4, 5])
    graph.add_flow(source.out, probe["in"], queue_depth=2)
    graph.lifecycle().allow_self_restart(probe)
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    lifecycle_capacity = hs.lifecycle.LifecycleTransactionOptions(
        max_pending_requests=64,
        max_pending_transactions=64,
        max_receipts=128,
        max_fenced_operators=64,
        max_candidate_generations=64,
        max_events=1024,
        max_audit_records=128,
        max_quiesce_records=256,
        max_rollback_records=256,
    )
    with hs.run_async(
        hs.compile(graph, hs.CompileOptions(lifecycle_transactions=lifecycle_capacity))
    ) as session:
        assert LifecycleProbe.computed.wait(timeout=5.0)
        session.request_stop()
        assert session.wait(timeout=5.0)

    # Key assertion: Stages see the selected generation and a retained context fails closed.
    assert not session.diagnostics
    configure_observation = LifecycleProbe.observations[0]
    assert configure_observation[:5] == (
        "configure",
        {"amount": 5, "shape": [4, 5]},
        0,
        hs.lifecycle.LifecycleTransitionPhase.STARTUP_FORWARD,
        False,
    )
    assert configure_observation[5] > 0
    assert configure_observation[6] > 0
    assert ("allocate", {"amount": 5, "shape": [4, 5]}) in LifecycleProbe.observations
    assert LifecycleProbe.observations[-1] == (
        "release",
        hs.lifecycle.LifecycleTransitionPhase.SHUTDOWN_TEARDOWN,
    )
    with pytest.raises(hs.errors.NotReadyError, match="PYTHON_LIFECYCLE_CONTEXT_EXPIRED"):
        _ = LifecycleProbe.retained_context.params
    with pytest.raises(hs.errors.NotReadyError, match="PYTHON_LIFECYCLE_CONTEXT_EXPIRED"):
        LifecycleProbe.retained_context.report_error(
            hs.errors.ErrorCode.INVALID_ARGUMENT, "expired lifecycle detail"
        )


# REQ: PY-OP-009
def test_lifecycle_context_resolves_live_clocks_in_the_owning_graph_scope():
    """Resolve default and imported clocks without leaking across embedded Graph scopes."""

    observations = []
    retained_contexts = []
    resolutions_complete = threading.Event()
    missing_scope = hs.Graph("missing-clock-scope")
    missing_clock = missing_scope.add_clock(hs.RealtimeClock, "missing")

    class ClockResolutionGraph(hs.Graph):
        def compose(self):
            timeline = self.import_clock("timeline")
            reference = self.import_clock("reference")
            self.set_default_clock(timeline)
            self.timeline = timeline
            self.reference = reference

            class ClockProbe(hs.Operator):
                def setup(self, spec):
                    spec.lifecycle().stage(hs.lifecycle.LifecycleStage.WARM_UP, "warm_up")

                def contract(self):
                    return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1)))

                def warm_up(self, context):
                    default = context.resolve_clock()
                    named = context.resolve_clock(timeline)
                    scoped = context.resolve_clock(reference)
                    current = default.now()
                    try:
                        context.resolve_clock(missing_clock)
                    except hs.errors.NotFoundError as error:
                        missing_token = error.token
                    else:
                        raise AssertionError("undeclared clock unexpectedly resolved")
                    observations.append(
                        {
                            "default": default,
                            "named_id": named.id,
                            "scoped_id": scoped.id,
                            "current": current,
                        }
                    )
                    assert missing_token == "NOT_FOUND"
                    retained_contexts.append(context)
                    if len(observations) == 2:
                        resolutions_complete.set()
                    return hs.lifecycle.LifecycleStatus.OK

                def compute(self, context):
                    pass

            self.op(ClockProbe, "probe")

    # Key setup: Both children share their scheduling clock but bind the same local reference name
    # to different parent clocks. The missing key is declared in an unrelated Graph scope only.
    graph = hs.Graph("native-lifecycle-clock-resolution")
    first = graph.embed(ClockResolutionGraph, "first")
    second = graph.embed(ClockResolutionGraph, "second")
    timeline = graph.add_clock(hs.RealtimeClock, "timeline")
    first_reference = graph.add_clock(hs.RealtimeClock, "first-reference")
    second_reference = graph.add_clock(hs.RealtimeClock, "second-reference")
    graph.set_default_clock(timeline)
    graph.bind_clock(first, first.timeline, timeline)
    graph.bind_clock(second, second.timeline, timeline)
    graph.bind_clock(first, first.reference, first_reference)
    graph.bind_clock(second, second.reference, second_reference)

    with hs.run_async(hs.compile(graph)) as session:
        assert resolutions_complete.wait(timeout=5.0)
        session.request_stop()
        assert session.wait(timeout=5.0)

    # Key assertion: Default resolution is live, local import names remain instance-scoped, and
    # callback authority expires without erasing the diagnostic parts of a returned ClockRef.
    assert session.diagnostics == []
    assert len(observations) == 2
    assert len({row["scoped_id"] for row in observations}) == 2
    for row in observations:
        reference = row["default"]
        assert reference.id == row["named_id"]
        assert isinstance(reference, hs.clocks.ClockRef)
        assert isinstance(reference.capabilities, hs.clocks.ClockCapabilities)
        assert reference.capabilities.can_read_now
        assert all(
            isinstance(value, bool)
            for value in (
                reference.capabilities.can_wait_until,
                reference.capabilities.steady,
                reference.capabilities.has_outgoing_conversions,
                reference.capabilities.has_incoming_conversions,
            )
        )
        assert row["current"].valid
        assert row["current"].clock_id == reference.id
        assert reference.at_ns(37) == hs.time.TimePoint(37, reference.id)
        with pytest.raises(hs.errors.StaleEndpointError):
            reference.now()
    for context in retained_contexts:
        with pytest.raises(hs.errors.NotReadyError, match="PYTHON_LIFECYCLE_CONTEXT_EXPIRED"):
            context.resolve_clock()


# REQ: PY-OP-007, PY-OP-009
@pytest.mark.parametrize(
    "outcome",
    [
        "fatal",
        "reported_fatal",
        "reported_success",
        "exception",
        "typed_exception",
        "invalid",
        "none",
    ],
)
def test_named_lifecycle_callback_status_and_failures_are_preserved(outcome):
    # Key setup: Return each success/failure shape accepted or rejected by the hosted callback.
    #
    #   source[C++] --uint32--> probe[Python]
    class LifecycleStatusProbe(hs.Operator):
        inp = hs.Input("in", hs.payload.uint32)
        computed = threading.Event()

        def setup(self, spec):
            spec.input(self.inp, queue_depth=1)
            spec.lifecycle().stage(hs.lifecycle.LifecycleStage.CONFIGURE, "configure")

        def configure(self, context):
            if outcome == "fatal":
                return hs.lifecycle.LifecycleStatus.FATAL_FAILURE
            if outcome == "reported_fatal":
                context.report_error(
                    hs.errors.ErrorCode.INVALID_ARGUMENT, "reported lifecycle sentinel"
                )
                context.report_error(hs.errors.ErrorCode.FAILURE, "ignored second report")
                return hs.lifecycle.LifecycleStatus.FATAL_FAILURE
            if outcome == "reported_success":
                context.report_error(
                    hs.errors.ErrorCode.INVALID_ARGUMENT, "discarded success detail"
                )
                return hs.lifecycle.LifecycleStatus.OK
            if outcome == "exception":
                raise RuntimeError("advanced lifecycle sentinel")
            if outcome == "typed_exception":
                error = hs.errors.InvalidArgumentError("advanced lifecycle sentinel")
                error.code = hs.errors.ErrorCode.INVALID_ARGUMENT
                raise error
            if outcome == "invalid":
                return object()
            return None

        def compute(self, context):
            self.inp.receive()
            type(self).computed.set()

    LifecycleStatusProbe.computed.clear()
    graph = hs.Graph(f"native-lifecycle-status-{outcome}")
    source = graph.op(COUNTER, "source", limit=1)
    probe = graph.op(LifecycleStatusProbe, "probe")
    graph.add_flow(source.out, probe["in"], queue_depth=1)
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    plan = hs.compile(graph)
    if outcome in {"none", "reported_success"}:
        with hs.run_async(plan) as session:
            assert LifecycleStatusProbe.computed.wait(timeout=5.0)
            session.request_stop()
            assert session.wait(timeout=5.0)
    else:
        session = hs.run(plan)

    # Key assertion: None means OK, while explicit or translated failures terminate the run and
    # preserve bounded Python detail when one exists.
    if outcome in {"none", "reported_success"}:
        assert session.termination is hs.RunTermination.STOPPED
        assert session.diagnostics == []
    else:
        assert session.termination is hs.RunTermination.FAILED
        assert session.diagnostics
        if outcome in {"exception", "typed_exception"}:
            expected_code = (
                hs.errors.ErrorCode.INVALID_ARGUMENT
                if outcome == "typed_exception"
                else hs.errors.ErrorCode.FAILURE
            )
            errors = [
                item.detail.error
                for item in session.diagnostics
                if item.detail is not None
                and "advanced lifecycle sentinel" in item.detail.error.message
            ]
            assert errors
            for error in errors:
                assert error.code is expected_code
                assert "test_native_lifecycle.py" in error.message
        if outcome == "reported_fatal":
            detail = session.diagnostics[0].detail.error
            assert detail.code is hs.errors.ErrorCode.INVALID_ARGUMENT
            assert detail.message == "reported lifecycle sentinel"
        if outcome == "invalid":
            assert any(
                "PYTHON_LIFECYCLE_STATUS_REQUIRED" in message
                for message in diagnostic_error_messages(session.diagnostics)
            )


def test_named_lifecycle_callbacks_reject_legacy_stage_conflicts():
    """Keep context-free start/stop distinct from advanced stage callbacks."""

    # Key setup: Attempt to register an advanced callback for the legacy start slot.
    class ConflictingStart(hs.Operator):
        def setup(self, spec):
            spec.lifecycle().stage(hs.lifecycle.LifecycleStage.START, "on_start")

        def on_start(self, context):
            pass

        def compute(self, context):
            pass

    graph = hs.Graph("native-lifecycle-conflict")

    # Key assertion: Authoring rejects the duplicate semantic slot before graph publication.
    with pytest.raises(
        hs.errors.InvalidArgumentError, match="PYTHON_LIFECYCLE_STAGE_NOT_HOSTABLE"
    ) as raised:
        graph.op(ConflictingStart, "conflict")
    assert raised.value.token == "GRAPH_OPERATOR_SETUP_RUNTIME_ERROR"


# REQ: PY-OP-009
@pytest.mark.parametrize("arity", [0, 2])
def test_named_lifecycle_callbacks_reject_wrong_arity_during_authoring(arity):
    # Key setup: Bind a callable that cannot accept exactly one LifecycleContext argument.
    class WrongArity(hs.Operator):
        def setup(self, spec):
            spec.lifecycle().stage(hs.lifecycle.LifecycleStage.CONFIGURE, "configure")

        if arity == 0:

            def configure(self):
                return None

        else:

            def configure(self, first, second):
                del first, second

        def compute(self, context):
            pass

    graph = hs.Graph(f"native-lifecycle-arity-{arity}")

    # Key assertion: The invalid method is rejected at graph authoring, before a run starts.
    with pytest.raises(
        hs.errors.InvalidArgumentError, match="PYTHON_LIFECYCLE_CALLBACK_SIGNATURE_INVALID"
    ):
        graph.op(WrongArity, "wrong-arity")


# REQ: PY-OP-009
def test_named_lifecycle_callback_property_errors_are_not_reported_as_missing():
    # Key setup: Resolve a declared callback through a descriptor that raises its own error.
    class RaisingDescriptor:
        def __get__(self, instance, owner):
            del instance, owner
            raise RuntimeError("lifecycle descriptor sentinel")

    class PropertyFailure(hs.Operator):
        configure = RaisingDescriptor()

        def setup(self, spec):
            spec.lifecycle().stage(hs.lifecycle.LifecycleStage.CONFIGURE, "configure")

        def compute(self, context):
            pass

    graph = hs.Graph("native-lifecycle-descriptor-failure")

    # Key assertion: Attribute resolution preserves the originating descriptor exception.
    with pytest.raises(hs.errors.FailureError, match="lifecycle descriptor sentinel"):
        graph.op(PropertyFailure, "property-failure")
