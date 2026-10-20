# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Managed testing preserves native restart receipts, recovery, and capture ownership."""

import ctypes
import threading

import holoscan_test_support
import pytest
from observation import CALLBACK_RELEASE_SECONDS, OBSERVATION_MS, OBSERVATION_SECONDS

import holoscan as hs
from holoscan.lifecycle import (
    LifecycleAccessPolicy,
    LifecycleRestartOptions,
    LifecycleStage,
    LifecycleStatus,
    LifecycleTransactionOptions,
    LifecycleTransactionState,
    LifecycleTransitionPhase,
)
from holoscan.testing import FrameRelease, OperatorTestHarness, Retain, TestLifecycleOptions
from holoscan.testing import TestRunOptions as RunOptions


class ProbeState:
    def __init__(self, failure=None):
        self.blocked = threading.Event()
        self.release = threading.Event()
        self.failure = failure
        self.stages = []
        self.starts = 0

    def stage(self, stage, context):
        phase = context.transition_phase
        self.stages.append((stage, phase))
        fail = (
            (
                self.failure in ("forward", "rollback", "restore")
                and phase == LifecycleTransitionPhase.RESTART_FORWARD
                and stage == LifecycleStage.WARM_UP
            )
            or (
                self.failure == "rollback"
                and phase == LifecycleTransitionPhase.FORWARD_ROLLBACK
                and stage == LifecycleStage.RELEASE
            )
            or (
                self.failure == "restore"
                and phase == LifecycleTransitionPhase.RESTORE_OLD
                and stage == LifecycleStage.CONFIGURE
            )
        )
        if fail:
            context.report_error(hs.errors.ErrorCode.INVALID_ARGUMENT, f"{self.failure} failure")
            return LifecycleStatus.FATAL_FAILURE
        return LifecycleStatus.OK

    def hold_compute(self):
        self.blocked.set()
        assert self.release.wait(timeout=CALLBACK_RELEASE_SECONDS), "compute was not released"


def scalar_probe(state):
    class ScalarProbe(hs.Operator):
        inp = hs.Input("in", hs.payload.uint32)
        out = hs.Output("out", hs.payload.uint32)

        def setup(self, spec):
            spec.input(self.inp)
            spec.output(self.out)
            lifecycle = spec.lifecycle()
            lifecycle.stage(LifecycleStage.CONFIGURE, "configure")
            lifecycle.stage(LifecycleStage.ALLOCATE, "allocate")
            lifecycle.stage(LifecycleStage.WARM_UP, "warm_up")
            lifecycle.stage(LifecycleStage.ARM, "arm")
            lifecycle.stage(LifecycleStage.RELEASE, "release")

        def configure(self, context):
            return state.stage(LifecycleStage.CONFIGURE, context)

        def allocate(self, context):
            return state.stage(LifecycleStage.ALLOCATE, context)

        def warm_up(self, context):
            return state.stage(LifecycleStage.WARM_UP, context)

        def arm(self, context):
            return state.stage(LifecycleStage.ARM, context)

        def release(self, context):
            return state.stage(LifecycleStage.RELEASE, context)

        def start(self):
            state.starts += 1
            return LifecycleStatus.OK

        def compute(self, context):
            value = self.inp.receive_data()
            if value == 1:
                state.hold_compute()
                if state.failure == "compute":
                    self.out.emit(99)
                    raise RuntimeError("compute failed after accepted restart")
                return
            self.out.emit(value * 10)

    return ScalarProbe


def configure_restart(harness, *, pending=8, receipts=16):
    harness.set_compile_options(
        hs.CompileOptions(
            lifecycle_transactions=LifecycleTransactionOptions(
                max_pending_requests=pending,
                max_pending_transactions=8,
                max_receipts=receipts,
                max_fenced_operators=16,
                max_candidate_generations=16,
                max_events=128,
                max_audit_records=32,
                max_quiesce_records=64,
                max_rollback_records=64,
            )
        )
    )
    harness.set_lifecycle_options(TestLifecycleOptions(allow_external_restart=True))


def stage_frames(harness, *values):
    for value in values:
        harness.push_frame({"in": value})


def control_for(running):
    target = running.operator_ref()
    control = running.lifecycle_control(
        LifecycleAccessPolicy(
            targets=[target],
            from_stages=[
                LifecycleStage.CONFIGURE,
                LifecycleStage.ALLOCATE,
                LifecycleStage.WARM_UP,
                LifecycleStage.ARM,
                LifecycleStage.START,
            ],
        )
    )
    return target, control


def receipt_state(view):
    return (
        view.state,
        view.transaction_id,
        view.coalesced_into_transaction,
        view.related_transaction_id,
        view.fence_id,
        view.effective_stage,
        view.candidate_generation,
    )


@pytest.mark.parametrize("second_stage", [LifecycleStage.CONFIGURE, LifecycleStage.ALLOCATE])
# REQ: PY-API-031
def test_pending_requests_preserve_coalescing_and_capacity(second_stage):
    state = ProbeState()
    with OperatorTestHarness(scalar_probe(state)) as harness:
        configure_restart(harness, pending=2)
        stage_frames(harness, 1, 2)
        with harness.run_async(RunOptions(frames=FrameRelease.MANUAL)) as running:
            try:
                running.release_frame(timeout_ms=OBSERVATION_MS)
                assert state.blocked.wait(timeout=OBSERVATION_SECONDS)
                target, control = control_for(running)
                first = control.request_restart(target)
                second = control.request_restart(
                    target, LifecycleRestartOptions(from_stage=second_stage)
                )
                before = control.inspect(first)
                if second_stage == LifecycleStage.CONFIGURE:
                    coalesced = control.inspect(second)
                    assert coalesced.coalesced_into_transaction == before.transaction_id
                    assert coalesced.fence_id == before.fence_id
                with pytest.raises(hs.errors.LifecycleRestartQueueFullError):
                    control.request_restart(
                        target, LifecycleRestartOptions(from_stage=LifecycleStage.WARM_UP)
                    )
                assert receipt_state(control.inspect(first)) == receipt_state(before)
                with pytest.raises(hs.errors.HoloscanTimeoutError):
                    control.await_terminal(first, timeout=0.0)
                assert not any(
                    phase == LifecycleTransitionPhase.RESTART_FORWARD for _, phase in state.stages
                )
            finally:
                # Release before owner cleanup even if a receipt assertion fails.
                state.release.set()
            for receipt in (first, second):
                assert control.await_terminal(receipt, timeout=OBSERVATION_SECONDS).state == (
                    LifecycleTransactionState.PUBLISHED
                )
            running.release_frame(timeout_ms=OBSERVATION_MS)
            assert running.finish().values("out") == [20]
    assert control.inspect(second).state == LifecycleTransactionState.PUBLISHED
    with pytest.raises(hs.errors.ClosedError):
        control.request_restart(target)


# REQ: PY-API-031
def test_terminal_receipts_keep_consuming_the_finite_receipt_budget():
    state = ProbeState()
    with OperatorTestHarness(scalar_probe(state)) as harness:
        configure_restart(harness, pending=2, receipts=4)
        stage_frames(harness, 0, 2)
        with harness.run_async(RunOptions(frames=FrameRelease.MANUAL)) as running:
            running.release_frame(timeout_ms=OBSERVATION_MS)
            assert running.wait_for_output("out", 1, timeout_ms=OBSERVATION_MS)
            target, control = control_for(running)
            receipts = []
            for _ in range(4):
                receipt = control.request_restart(
                    target, LifecycleRestartOptions(from_stage=LifecycleStage.START)
                )
                assert control.await_terminal(receipt, timeout=OBSERVATION_SECONDS).state == (
                    LifecycleTransactionState.PUBLISHED
                )
                receipts.append(receipt)
            with pytest.raises(hs.errors.LifecycleRestartQueueFullError):
                control.request_restart(
                    target, LifecycleRestartOptions(from_stage=LifecycleStage.START)
                )
            running.release_frame(timeout_ms=OBSERVATION_MS)
            assert running.finish().values("out") == [0, 20]
    assert all(
        control.inspect(receipt).state == LifecycleTransactionState.PUBLISHED
        for receipt in receipts
    )


@pytest.mark.parametrize("failure", [None, "compute"])
# REQ: PY-API-031
def test_pending_restart_cancellation_preserves_the_captured_prefix(failure):
    state = ProbeState(failure)
    with OperatorTestHarness(scalar_probe(state)) as harness:
        configure_restart(harness)
        stage_frames(harness, 0, 1, 2)
        with harness.run_async(RunOptions(frames=FrameRelease.MANUAL)) as running:
            running.release_frame(timeout_ms=OBSERVATION_MS)
            assert running.wait_for_output("out", 1, timeout_ms=OBSERVATION_MS)
            try:
                running.release_frame(timeout_ms=OBSERVATION_MS)
                assert state.blocked.wait(timeout=OBSERVATION_SECONDS)
                target, control = control_for(running)
                receipt = control.request_restart(target)
                if failure is None:
                    running.request_stop()
            finally:
                state.release.set()
            result = running.finish(check=False)
            assert result.values("out") == [0]
            assert result.termination == (
                hs.RunTermination.FAILED if failure else hs.RunTermination.STOPPED
            )
            assert control.inspect(receipt).state == LifecycleTransactionState.CANCELED
            assert not any(
                phase == LifecycleTransitionPhase.RESTART_FORWARD for _, phase in state.stages
            )
            if failure:
                with pytest.raises(hs.HoloscanError) as caught:
                    running.finish()
                assert caught.value.result is result
                assert "compute failed after accepted restart" in result.describe_failure()


@pytest.mark.parametrize("failure", ["forward", "rollback", "restore"])
# REQ: PY-API-031
def test_restart_recovery_keeps_receipt_and_diagnostic_failure_channels(failure):
    state = ProbeState(failure)
    with OperatorTestHarness(scalar_probe(state)) as harness:
        configure_restart(harness)
        stage_frames(harness, 0, 2)
        with harness.run_async(RunOptions(frames=FrameRelease.MANUAL)) as running:
            running.release_frame(timeout_ms=OBSERVATION_MS)
            assert running.wait_for_output("out", 1, timeout_ms=OBSERVATION_MS)
            target, control = control_for(running)
            receipt = control.request_restart(target)
            terminal = control.await_terminal(receipt, timeout=OBSERVATION_SECONDS)
            recovered = failure == "forward"
            assert terminal.state == (
                LifecycleTransactionState.ROLLED_BACK
                if recovered
                else LifecycleTransactionState.FAILED_STOPPED
            )
            if recovered:
                running.release_frame(timeout_ms=OBSERVATION_MS)
            result = running.finish(check=False)
            assert result.ok == recovered
            assert result.values("out") == ([0, 20] if recovered else [0])
            details = [item.detail for item in result.diagnostics if item.detail is not None]
            assert any(
                detail.phase == LifecycleTransitionPhase.RESTART_FORWARD for detail in details
            )
            if not recovered:
                assert result.termination == hs.RunTermination.FAILED
                secondary = (
                    LifecycleTransitionPhase.FORWARD_ROLLBACK
                    if failure == "rollback"
                    else LifecycleTransitionPhase.RESTORE_OLD
                )
                assert any(detail.phase == secondary for detail in details)
                with pytest.raises(hs.HoloscanError) as caught:
                    running.finish()
                assert caught.value.result is result


@pytest.mark.parametrize("stage", [LifecycleStage.START, LifecycleStage.ALLOCATE])
# REQ: PY-API-031
def test_hosted_restart_executes_the_selected_suffix_and_fresh_run_starts_over(stage):
    state = ProbeState()
    with OperatorTestHarness(scalar_probe(state)) as harness:
        configure_restart(harness)
        stage_frames(harness, 0, 2)
        with harness.run_async(RunOptions(frames=FrameRelease.MANUAL)) as running:
            running.release_frame(timeout_ms=OBSERVATION_MS)
            assert running.wait_for_output("out", 1, timeout_ms=OBSERVATION_MS)
            target, control = control_for(running)
            receipt = control.request_restart(target, LifecycleRestartOptions(from_stage=stage))
            terminal = control.await_terminal(receipt, timeout=OBSERVATION_SECONDS)
            assert terminal.state == LifecycleTransactionState.PUBLISHED
            assert terminal.effective_stage == stage
            current = running.operator_ref()
            assert (current.id, current.run_generation) == (target.id, target.run_generation)
            restarted = [
                observed
                for observed, phase in state.stages
                if phase == LifecycleTransitionPhase.RESTART_FORWARD
            ]
            assert restarted == (
                [LifecycleStage.ALLOCATE, LifecycleStage.WARM_UP, LifecycleStage.ARM]
                if stage == LifecycleStage.ALLOCATE
                else []
            )
            assert state.starts == 2
            running.release_frame(timeout_ms=OBSERVATION_MS)
            result = running.finish()
        assert result.values("out") == [0, 20]
        stage_frames(harness, 3)
        with harness.run_async() as fresh:
            assert fresh.operator_ref().run_generation != target.run_generation
            assert fresh.finish().values("out") == [30]
        assert state.starts == 3
        assert result.values("out") == [0, 20]


@pytest.mark.parametrize("native_capture", [False, True])
@pytest.mark.parametrize("stage", [LifecycleStage.ARM, LifecycleStage.ALLOCATE])
# REQ: PY-API-031
def test_tensor_captures_survive_restart_with_a_real_unpublished_loan(stage, native_capture):
    state = ProbeState()
    calls = []

    class TensorProbe(hs.Operator):
        out = hs.Output("out", hs.payload.Tensor)

        def setup(self, spec):
            spec.output(self.out).produces_tensor(
                hs.tensor.OutputSpec(
                    storage="runtime_pool",
                    memory_kind=hs.tensor.MemoryKind.HOST,
                    dtype=hs.tensor.DLDataType(1, 8, 1),
                    rank=1,
                ),
                hs.tensor.byte_span(4),
            )
            spec.lifecycle().stage(LifecycleStage.ALLOCATE, "allocate")
            spec.lifecycle().stage(LifecycleStage.ARM, "arm")

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(3)

        def allocate(self, context):
            return state.stage(LifecycleStage.ALLOCATE, context)

        def arm(self, context):
            return state.stage(LifecycleStage.ARM, context)

        def compute(self, context):
            calls.append(len(calls) + 1)
            with self.out.allocate_tensor(shape=(4,), dtype=hs.tensor.DLDataType(1, 8, 1)) as loan:
                (ctypes.c_uint8 * 4).from_address(loan.tensor.data)[:] = [11 * calls[-1]] * 4
                if len(calls) == 2:
                    # The restart fence discards this callback's subsequently queued output.
                    state.hold_compute()

    with OperatorTestHarness(TensorProbe) as harness:
        configure_restart(harness)
        if native_capture:
            harness.capture_native("out", Retain(max_records=3))
        with harness.run_async() as running:
            try:
                assert state.blocked.wait(timeout=OBSERVATION_SECONDS)
                target, control = control_for(running)
                receipt = control.request_restart(target, LifecycleRestartOptions(from_stage=stage))
                with pytest.raises(hs.errors.HoloscanTimeoutError):
                    control.await_terminal(receipt, timeout=0.0)
                assert not any(
                    phase == LifecycleTransitionPhase.RESTART_FORWARD for _, phase in state.stages
                )
            finally:
                state.release.set()
            terminal = control.await_terminal(receipt, timeout=OBSERVATION_SECONDS)
            assert terminal.state == LifecycleTransactionState.PUBLISHED
            assert terminal.effective_stage == stage
            assert running.wait_for_output("out", 2, timeout_ms=OBSERVATION_MS)
            if native_capture:
                # Dropping a queued generation cannot satisfy the native Retain proof. Stop
                # explicitly so failure evidence does not depend on a watchdog deadline.
                running.request_stop()
                result = running.finish(check=False)
                assert not result.ok
                assert result.termination == hs.RunTermination.STOPPED
                assert "Retain completion boundary was not reached" in result.describe_failure()
            else:
                result = running.finish()
    assert calls == [1, 2, 3]
    if native_capture:
        tensors = result.values("out")
        assert len(tensors) == 2
        contents = [bytes((ctypes.c_uint8 * 4).from_address(value.data)) for value in tensors]
        assert tensors[0].data != tensors[1].data
    else:
        snapshots = result.snapshots("out")
        contents = [value.bytes for value in snapshots]
        assert all(value.shape == [4] for value in snapshots)
    assert contents == [bytes([value] * 4) for value in (11, 33)]


@pytest.mark.parametrize("stage", [LifecycleStage.ARM, LifecycleStage.ALLOCATE])
# REQ: PY-API-031
def test_native_tensor_retention_completes_across_same_run_restart(stage):
    holoscan_test_support._register_tensor_restart_source_for_test()
    with OperatorTestHarness("holoscan.test.TensorRestartSource") as harness:
        configure_restart(harness)
        harness.capture_native("out", Retain(max_records=2))
        with harness.run_async() as running:
            assert holoscan_test_support._post_tensor_restart_source(
                generation=1, timeout=OBSERVATION_SECONDS
            )
            assert running.wait_for_output("out", 1, timeout_ms=OBSERVATION_MS)
            target, control = control_for(running)
            receipt = control.request_restart(target, LifecycleRestartOptions(from_stage=stage))
            terminal = control.await_terminal(receipt, timeout=OBSERVATION_SECONDS)
            assert terminal.state == LifecycleTransactionState.PUBLISHED
            assert terminal.effective_stage == stage
            assert running.operator_ref().run_generation == target.run_generation
            assert running.output_count("out") == 1
            assert holoscan_test_support._post_tensor_restart_source(
                generation=2, timeout=OBSERVATION_SECONDS
            )
            result = running.finish()
    assert result.ok
    tensors = result.values("out")
    assert len(tensors) == 2
    assert tensors[0].data != tensors[1].data
    assert [bytes((ctypes.c_uint8 * 4).from_address(value.data)) for value in tensors] == [
        bytes([11] * 4),
        bytes([22] * 4),
    ]
