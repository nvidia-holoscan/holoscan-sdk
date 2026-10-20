# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporal contracts authored by native Python operators."""

import json
import os
import subprocess
import time
from datetime import timedelta

import holoscan_test_support  # noqa: F401
import pytest

import holoscan as hs

COUNTER = "holoscan.test.CounterSource"


# REQ: PY-OP-011
def test_max_start_delay_removes_legacy_names_and_preserves_reason_value():
    contract = hs.Contract()
    assert contract.max_start_delay(timedelta(milliseconds=2)) is contract
    assert not hasattr(hs.Contract, "deadline")
    assert not hasattr(hs.ActivationReason, "DEADLINE_EXCEEDED")
    assert hs.ActivationReason.MAX_START_DELAY_EXCEEDED.value == 1


class SensorFusion(hs.Operator):
    """Reproduces the first_operator teaching contract for plan-parity coverage."""

    camera = hs.Input("camera", hs.payload.uint32)
    imu = hs.Input("imu", hs.payload.uint32)
    max_imu_age = hs.Param(timedelta(milliseconds=5))

    def setup(self, spec):
        spec.input(self.camera, queue_depth=8)
        spec.input(self.imu, queue_depth=8)

    def contract(self):
        return (
            hs.Contract()
            .trigger(hs.OnEach(self.camera))
            .input(self.imu, hs.LatestWithin(max_age=self.max_imu_age))
            .on_missing(self.imu, hs.ReuseLast(max_age=timedelta(milliseconds=20)))
        )

    def compute(self, context):
        pass


def temporal_graph(operator_type=SensorFusion, name="native-temporal-contract", queue_depth=8):
    """Compose the two-source graph used by the temporal contract tests."""
    graph = hs.Graph(name)
    camera = graph.op(COUNTER, "camera")
    imu = graph.op(COUNTER, "imu")
    fusion = graph.op(operator_type, "fusion")
    graph.add_flow(camera.out, fusion.camera, queue_depth=queue_depth)
    graph.add_flow(imu.out, fusion.imu, queue_depth=queue_depth)
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    return graph


def stable_temporal_records(plan_json):
    """Remove process-local generation values before comparing temporal records."""
    document = json.loads(plan_json)

    def canonicalize(value):
        if isinstance(value, dict):
            return {
                key: "<plan-generation>" if key == "plan_generation" else canonicalize(child)
                for key, child in value.items()
            }
        if isinstance(value, list):
            return [canonicalize(child) for child in value]
        return value

    return {
        key: canonicalize(document[key])
        for key in (
            "clock_triggers",
            "input_selections",
            "missing_aggregates",
            "temporal_triggers",
        )
    }


# Implements the unwired Exact/SafeDefault contract used by
# test_optional_unwired_input_runs_with_a_missing_sample.
class OptionalClockProbe(hs.Operator):
    side = hs.Input("side", hs.payload.uint32)
    samples = []

    def setup(self, spec):
        spec.input(self.side, optional=True)

    def contract(self):
        return (
            hs.Contract()
            .trigger(hs.OnClock(period=hs.time.ms(1)))
            .input(self.side, hs.Exact())
            .on_missing(self.side, hs.SafeDefault())
        )

    def compute(self, context):
        sample = self.side.receive()
        type(self).samples.append((sample.data, sample.is_missing))


# Omits optional permission while retaining the same contract. Used by
# test_unwired_input_remains_required_when_optional_is_omitted.
class RequiredClockProbe(OptionalClockProbe):
    def setup(self, spec):
        spec.input(self.side)


# REQ: PY-OP-005
def test_optional_unwired_input_runs_with_a_missing_sample():
    # Graph:
    #
    #   [unwired] --> optional.side
    #
    # Key setup: Compile and run one clock-driven Exact/SafeDefault input with explicit permission.
    OptionalClockProbe.samples.clear()
    graph = hs.Graph("native-optional-unwired-input")
    graph.op(OptionalClockProbe, "optional")
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    plan = hs.compile(graph)
    with hs.run_async(plan) as session:
        deadline = time.monotonic() + 5.0
        while not OptionalClockProbe.samples and time.monotonic() < deadline:
            time.sleep(0.01)
        session.request_stop()
        stopped = session.wait(timeout=5.0)

    # Key assertion: Core admits absence with its stable warning, and receive reports it as missing.
    assert plan.ok, plan.json()
    assert any(item.token == "TEMPORAL_INPUT_UNCONNECTED_SAFE_DEFAULT" for item in plan.diagnostics)
    assert stopped
    assert session.termination == hs.RunTermination.STOPPED
    assert OptionalClockProbe.samples
    assert all(sample == (None, True) for sample in OptionalClockProbe.samples)


# REQ: PY-OP-005
def test_unwired_input_remains_required_when_optional_is_omitted():
    # Graph:
    #
    #   [unwired] --> required.side
    #
    # Key setup: Compile the same eligible contract without explicit optional permission.
    graph = hs.Graph("native-required-unwired-input")
    graph.op(RequiredClockProbe, "required")
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    plan = hs.compile(graph)

    # Key assertion: omission preserves required-input behavior and its stable diagnostic token.
    assert not plan.ok
    assert any(item.token == "TEMPORAL_INPUT_UNCONNECTED" for item in plan.diagnostics)


# REQ: PY-OP-005
@pytest.mark.parametrize(
    ("invalid_contract", "expected_token"),
    [
        ("selector", "TEMPORAL_INPUT_OPTIONAL_INVALID"),
        ("trigger", "TEMPORAL_CONTRACT_INVALID"),
        ("missing", "TEMPORAL_INPUT_OPTIONAL_INVALID"),
    ],
)
def test_optional_permission_does_not_legalize_unsupported_contracts(
    invalid_contract, expected_token
):
    # Graph:
    #
    #   [unwired] --> invalid.side
    #
    # Key setup: Keep optional permission while violating one eligibility rule at a time.
    class InvalidOptionalContract(hs.Operator):
        side = hs.Input("side", hs.payload.uint32)

        def setup(self, spec):
            spec.input(self.side, optional=True)

        def contract(self):
            trigger = (
                hs.OnEach(self.side)
                if invalid_contract == "trigger"
                else hs.OnClock(period=hs.time.ms(1))
            )
            selector = (
                hs.LatestWithin(max_age=hs.time.ms(1))
                if invalid_contract == "selector"
                else hs.Exact()
            )
            missing = hs.Skip() if invalid_contract == "missing" else hs.SafeDefault()
            return (
                hs.Contract()
                .trigger(trigger)
                .input(self.side, selector)
                .on_missing(self.side, missing)
            )

        def compute(self, context):
            pass

    graph = hs.Graph(f"native-invalid-optional-{invalid_contract}")
    graph.op(InvalidOptionalContract, "invalid")
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    plan = hs.compile(graph)

    # Key assertion: Core distinguishes an invalid trigger from ineligible optional-input policy.
    assert not plan.ok
    assert any(item.token == expected_token for item in plan.diagnostics)


# REQ: PY-OP-005
def test_teaching_contract_matches_independent_cpp_temporal_records():
    #   camera[C++] --uint32--\
    #                              fusion[Python]
    #   imu[C++] --------uint32--/
    # Key setup: Compile the first_operator teaching contract through the Python host.
    plan = hs.compile(temporal_graph())
    cpp_plan = subprocess.run(
        [os.environ["HOLOSCAN_TEMPORAL_PLAN_HELPER"]],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout

    # Key assertion: The hosted and typed C++ forms lower to identical temporal plan records.
    assert plan.ok
    assert stable_temporal_records(plan.json()) == stable_temporal_records(cpp_plan)


# REQ: PY-OP-005
def test_contract_replay_mismatch_fails_before_activation():
    #   camera[C++] --uint32--\
    #                              fusion[Python]
    #   imu[C++] --------uint32--/
    # Key setup: Change the OnEach driver between compile and session-start replay.
    class ConditionalContract(SensorFusion):
        contract_calls = 0

        def contract(self):
            type(self).contract_calls += 1
            driver = self.camera if type(self).contract_calls == 1 else self.imu
            return hs.Contract().trigger(hs.OnEach(driver))

    plan = hs.compile(temporal_graph(ConditionalContract, "conditional-temporal-contract"))
    assert plan.ok
    session = hs.run_async(plan)
    assert session.wait(timeout=5.0)

    # Key assertion: Core's ordinary replay validator rejects the changed contract.
    assert session.diagnostics
    assert any(
        item.token == "RUNTIME_OPERATOR_REPLAY_FINGERPRINT_MISMATCH" for item in session.diagnostics
    )


# REQ: PY-OP-005
def test_invalid_contract_normalization_during_replay_fails_before_compute():
    # Key setup: Compile a valid contract, then add max_start_delay without a violation policy
    # only during session-start replay.
    class InvalidNormalizedReplay(SensorFusion):
        contract_calls = 0
        compute_calls = 0

        def contract(self):
            type(self).contract_calls += 1
            contract = hs.Contract().trigger(hs.OnEach(self.camera))
            if type(self).contract_calls > 1:
                contract.max_start_delay(hs.time.ms(1))
            return contract

        def compute(self, context):
            type(self).compute_calls += 1

    plan = hs.compile(temporal_graph(InvalidNormalizedReplay, "invalid-normalized-replay"))
    assert plan.ok, plan.json()
    session = hs.run_async(plan)
    assert session.wait(timeout=5.0)

    # Key assertion: Replay reports the bounded normalization failure rather than dereferencing a
    # failed result, and the invalid operator never reaches compute.
    diagnostic = next(
        item
        for item in session.diagnostics
        if item.token == "RUNTIME_OPERATOR_REPLAY_FINGERPRINT_MISMATCH"
    )
    assert diagnostic.detail is not None
    assert (
        "max_start_delay and on_violation must be declared together"
        in diagnostic.detail.error.message
    )
    assert InvalidNormalizedReplay.compute_calls == 0


# REQ: PY-OP-005
def test_contract_callback_requires_a_contract_value():
    #   camera[C++] --uint32--\
    #                              fusion[Python]
    #   imu[C++] --------uint32--/
    # Key setup: Return a non-Contract value from the cold Python contract callback.
    class InvalidContract(SensorFusion):
        def contract(self):
            return None

    # Key assertion: Compilation retains a stable Core token and the callback's precise detail.
    plan = hs.compile(temporal_graph(InvalidContract, "invalid-temporal-contract"))
    assert not plan.ok
    diagnostic = next(
        item for item in plan.diagnostics if item.token == "TEMPORAL_CONTRACT_REPLAY_FAILED"
    )
    assert diagnostic.detail is not None
    assert "PYTHON_OPERATOR_CONTRACT_REQUIRED" in diagnostic.detail.error.message


# REQ: PY-OP-005
def test_contract_callback_exception_reaches_runtime_diagnostic():
    #   camera[C++] --uint32--\
    #                              fusion[Python]
    #   imu[C++] --------uint32--/
    # Key setup: Let compilation succeed, then raise during session-start contract replay.
    class FailingReplay(SensorFusion):
        contract_calls = 0

        def contract(self):
            type(self).contract_calls += 1
            if type(self).contract_calls == 2:
                raise ValueError("contract replay exploded")
            return super().contract()

    plan = hs.compile(temporal_graph(FailingReplay, "failing-temporal-replay"))
    assert plan.ok, plan.json()
    session = hs.run_async(plan)
    assert session.wait(timeout=5.0)

    # Key assertion: The callback failure is distinct from a semantic replay mismatch.
    diagnostic = next(
        item for item in session.diagnostics if item.token == "RUNTIME_OPERATOR_CALLBACK_FAILED"
    )
    assert diagnostic.detail is not None
    assert "contract replay exploded" in diagnostic.detail.error.message


# REQ: PY-OP-005
# REQ: PY-OP-010
# REQ: PY-OP-011
def test_temporal_vocabulary_uses_explicit_duration_values():
    # Key setup: Construct every fieldless value plus representative bounded temporal values.
    assert isinstance(hs.OnAnyInput(), hs.OnAnyInput)
    assert isinstance(hs.Exact(), hs.Exact)
    assert isinstance(hs.DrainAll(), hs.DrainAll)
    assert isinstance(hs.Skip(), hs.Skip)
    assert isinstance(hs.SafeDefault(), hs.SafeDefault)
    latest = hs.LatestWithin(max_age=timedelta(microseconds=5))
    reuse = hs.ReuseLast(max_age=hs.time.us(20))
    bracketed = hs.BracketedWindow(lookbehind=2, lookahead=3, max_samples=8)
    start_at = hs.time.TimePoint(1_000_000, 1)
    clock = hs.OnClock(period=timedelta(milliseconds=1), start_at=start_at)

    # Key assertion: Durations retain nanosecond precision and bare numbers fail at construction.
    assert latest.max_age.nanoseconds == 5_000
    assert reuse.max_age.nanoseconds == 20_000
    assert bracketed.lookbehind == 2
    assert bracketed.lookahead == 3
    assert bracketed.max_samples == 8
    assert clock.period.nanoseconds == 1_000_000
    assert clock.clock.is_default
    assert clock.start_at == start_at
    with pytest.raises(hs.errors.InvalidArgumentError, match="AUTHORING_PARAMETER_TYPE_MISMATCH"):
        hs.OnClock(period=1)
    with pytest.raises(hs.errors.InvalidArgumentError, match="field_name=lookbehind"):
        hs.BracketedWindow(lookbehind=True)
    with pytest.raises(hs.errors.InvalidArgumentError, match="field_name=limit"):
        hs.Contract().complete_after(True)
    with pytest.raises(hs.errors.InvalidArgumentError, match="field_name=clock"):
        hs.OnClock(period=hs.time.ms(1), clock="wall")


# REQ: PY-OP-005
@pytest.mark.parametrize(
    "contract_builder",
    [
        lambda _op: hs.Contract().trigger(hs.OnClock(period=1)),
        lambda op: hs.Contract().input(op.imu, hs.LatestWithin(max_age=1)),
        lambda op: hs.Contract().on_missing(op.imu, hs.ReuseLast(max_age=1)),
        lambda op: hs.Contract().trigger(hs.OnAligned(ports=[op.camera, op.imu], window=1)),
        lambda op: hs.Contract().trigger(hs.OnWindow(anchor=op.camera, max_wait=1)),
        lambda _op: hs.Contract().max_start_delay(1),
    ],
)
def test_every_temporal_duration_rejects_bare_numbers(contract_builder):
    # Key setup: Route a bare integer through each duration-bearing vocabulary path.
    class InvalidDuration(SensorFusion):
        def contract(self):
            return contract_builder(self)

    plan = hs.compile(temporal_graph(InvalidDuration, "invalid-temporal-duration"))

    # Key assertion: Every path fails closed through the same explicit-unit diagnostic.
    assert not plan.ok
    diagnostic = next(
        item for item in plan.diagnostics if item.token == "TEMPORAL_CONTRACT_REPLAY_FAILED"
    )
    assert diagnostic.detail is not None
    assert "AUTHORING_PARAMETER_TYPE_MISMATCH" in diagnostic.detail.error.message


# REQ: PY-OP-005
def test_retained_port_trigger_owns_its_declaration():
    #   camera[C++] --uint32--\
    #                              fusion[Python]
    #   imu[C++] --------uint32--/
    # Key setup: Cache a trigger beyond the hosted instance that supplied its OperatorPort.
    class RetainedTrigger(SensorFusion):
        trigger = None

        def contract(self):
            if type(self).trigger is None:
                type(self).trigger = hs.OnEach(self.camera)
            return hs.Contract().trigger(type(self).trigger)

    graph = temporal_graph(RetainedTrigger, "retained-temporal-trigger")
    first = hs.compile(graph)
    second = hs.compile(graph)

    # Key assertion: Reusing the trigger on a fresh replay remains deterministic and safe.
    assert first.ok, first.json()
    assert second.ok, second.json()
    assert RetainedTrigger.trigger.port.name == "camera"


# REQ: PY-OP-005
def test_aligned_window_max_start_delay_and_completion_vocabulary_compiles():
    #   left[C++] --uint32--\
    #                            aligned[Python]
    #   right[C++] ------uint32--/
    # Key setup: Exercise trigger, selector, missing, max_start_delay, and completion forms.
    class AlignedProbe(hs.Operator):
        camera = hs.Input("camera", hs.payload.uint32)
        imu = hs.Input("imu", hs.payload.uint32)
        observed_ports = ()
        observed_window_ns = 0

        def contract(self):
            class PortSequence:
                def __init__(self, *values):
                    self._values = values

                def __len__(self):
                    return len(self._values)

                def __getitem__(self, index):
                    return self._values[index]

            trigger = hs.OnAligned(
                ports=PortSequence(self.camera, self.imu),
                window=timedelta(microseconds=250),
            )
            type(self).observed_ports = tuple(port.name for port in trigger.ports)
            type(self).observed_window_ns = trigger.window.nanoseconds
            return (
                hs.Contract()
                .trigger(trigger)
                .input(self.camera, hs.Exact())
                .input(self.imu, hs.Exact())
                .max_start_delay(timedelta(milliseconds=5))
                .on_violation(hs.OnViolation.FAIL_RUN)
            )

        def compute(self, context):
            pass

    class FiniteSource(hs.Operator):
        out = hs.Output("out", hs.payload.uint32)

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(4)

        def compute(self, context):
            pass

    plan = hs.compile(temporal_graph(AlignedProbe, "aligned-temporal-contract"))
    # Graph:
    #
    # finite-source[Python]
    source_graph = hs.Graph("finite-temporal-source")
    source_graph.op(FiniteSource, "source")
    source_graph.set_default_clock(source_graph.add_clock(hs.RealtimeClock, "clock"))
    source_plan = hs.compile(source_graph)

    # Key assertion: The public vocabulary reaches the ordinary admitted temporal plan.
    assert plan.ok, plan.json()
    assert source_plan.ok, source_plan.json()
    assert AlignedProbe.observed_ports == ("camera", "imu")
    assert AlignedProbe.observed_window_ns == 250_000
    document = json.loads(plan.json())
    source_document = json.loads(source_plan.json())
    assert any(trigger["kind"] == 7 for trigger in document["temporal_triggers"])
    assert document["max_start_delays"]
    assert source_document["operator_completion_rules"][0]["limit"] == 4


# REQ: PY-OP-005
def test_window_trigger_and_bracketed_selection_compile():
    #   camera[C++] --uint32--\
    #                              window[Python]
    #   imu[C++] --------uint32--/
    # Key setup: Build one bounded anchor window with a safe-default bracketed side input.
    class WindowProbe(SensorFusion):
        observed_window = None

        def contract(self):
            trigger = hs.OnWindow(
                anchor=self.camera,
                max_wait=timedelta(milliseconds=2),
                max_pending_windows=2,
            )
            type(self).observed_window = (
                trigger.anchor.name,
                trigger.max_wait.nanoseconds,
                trigger.wait_clock.is_default,
                trigger.max_pending_windows,
            )
            return (
                hs.Contract()
                .trigger(trigger)
                .input(self.camera, hs.Exact())
                .input(self.imu, hs.BracketedWindow(max_samples=8))
                .on_missing(self.imu, hs.SafeDefault())
            )

    plan = hs.compile(temporal_graph(WindowProbe, "window-temporal-contract", queue_depth=16))

    # Key assertion: Window and bracketed records are emitted by the C++ compiler, not Python.
    assert plan.ok, plan.json()
    assert WindowProbe.observed_window == ("camera", 2_000_000, True, 2)
    document = json.loads(plan.json())
    assert document["window_triggers"]
    assert any(selection["batch_selection_mode"] != 0 for selection in document["input_selections"])


# REQ: PY-OP-005
def test_any_input_exact_drain_all_and_skip_vocabulary_compiles():
    #   camera[C++] --uint32--\
    #                              fusion[Python]
    #   imu[C++] --------uint32--/
    # Key setup: Exercise the remaining trigger, selectors, and missing policy in one contract.
    class RemainingVocabulary(SensorFusion):
        def contract(self):
            return (
                hs.Contract()
                .trigger(hs.OnAnyInput())
                .input(self.camera, hs.Exact())
                .on_missing(self.camera, hs.Skip())
                .input(self.imu, hs.DrainAll())
            )

    plan = hs.compile(temporal_graph(RemainingVocabulary, "remaining-temporal-vocabulary"))

    # Key assertion: Registered payload thunks preserve both exact and batch selector identities.
    assert plan.ok, plan.json()
    document = json.loads(plan.json())
    assert len(document["input_selections"]) == 2


# REQ: PY-OP-005
def test_depth_one_reuse_last_preserves_the_initial_native_python_sample():
    #   sparse-source[C++] --uint32--> probe[Python]
    class ReuseProbe(hs.Operator):
        value = hs.Input("value", hs.payload.uint32)
        observations = []

        def setup(self, spec):
            spec.input(self.value, queue_depth=1)

        def contract(self):
            return (
                hs.Contract()
                .trigger(hs.OnClock(period=hs.time.ms(10)))
                .input(self.value, hs.LatestWithin(max_age=timedelta(seconds=5)))
                .on_missing(self.value, hs.ReuseLast(max_age=timedelta(seconds=5)))
            )

        def compute(self, context):
            sample = self.value.receive()
            type(self).observations.append((sample.data, sample.is_stale, sample.is_missing))

    # Key setup: Publish one timestamped value while an independent clock drives a depth-one
    # native-Python consumer.
    ReuseProbe.observations.clear()
    graph = hs.Graph("depth-one-native-reuse-last", default_clock=hs.RealtimeClock)
    source = graph.op("holoscan.test.SparseTimestampedSource", "a-source")
    probe = graph.op(ReuseProbe, "z-probe")
    graph.add_flow(source.out, probe.value, queue_depth=1)
    graph.partition("source").add(source)
    graph.partition("probe").add(probe)
    plan = hs.compile(graph)
    assert plan.ok, plan.json()
    with hs.run_async(plan) as session:
        deadline = time.monotonic() + 5.0
        while len(ReuseProbe.observations) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        session.request_stop()
        stopped = session.wait(timeout=5.0)

    # Key assertion: The first value is selected once, then remains a valid reusable sample.
    assert stopped
    assert not session.diagnostics
    assert len(ReuseProbe.observations) >= 3
    assert ReuseProbe.observations[0] == (17, False, False)
    assert all(observation == (17, True, False) for observation in ReuseProbe.observations[1:])


# REQ: PY-OP-005
def test_window_clock_type_mismatch_is_tokened():
    #   camera[C++] --uint32--\
    #                              fusion[Python]
    #   imu[C++] --------uint32--/
    # Key setup: Supply a string where OnWindow requires a ClockKey.
    class InvalidWindowClock(SensorFusion):
        def contract(self):
            return hs.Contract().trigger(
                hs.OnWindow(anchor=self.camera, max_wait=hs.time.ms(1), wait_clock="wall")
            )

    plan = hs.compile(temporal_graph(InvalidWindowClock, "invalid-window-clock"))

    # Key assertion: The mismatch retains the argument name instead of surfacing std::bad_cast.
    assert not plan.ok
    diagnostic = next(
        item for item in plan.diagnostics if item.token == "TEMPORAL_CONTRACT_REPLAY_FAILED"
    )
    assert diagnostic.detail is not None
    assert "AUTHORING_TEMPORAL_CLOCK_TYPE_MISMATCH" in diagnostic.detail.error.message
    assert "field_name=wait_clock" in diagnostic.detail.error.message
