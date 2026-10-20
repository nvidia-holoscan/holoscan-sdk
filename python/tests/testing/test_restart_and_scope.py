# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical lifecycle controls and owning lifecycle configuration."""

from dataclasses import FrozenInstanceError

import holoscan_test_support
import pytest
from observation import OBSERVATION_MS, OBSERVATION_SECONDS

import holoscan as hs
from holoscan.lifecycle import (
    LifecycleAccessPolicy,
    LifecycleRestartOptions,
    LifecycleStage,
    LifecycleStatus,
    LifecycleTransactionOptions,
    LifecycleTransactionState,
)
from holoscan.testing import FrameRelease, OperatorTestHarness, TestLifecycleOptions
from holoscan.testing import TestRunOptions as RunOptions


class Restartable(hs.Operator):
    inp = hs.Input("in", hs.payload.uint32)
    out = hs.Output("out", hs.payload.uint32)

    def setup(self, spec):
        spec.input(self.inp)
        spec.output(self.out)
        spec.lifecycle().stage(LifecycleStage.ARM, "arm")

    def __init__(self):
        self.factor = 1

    def arm(self, context):
        return LifecycleStatus.OK

    def start(self):
        self.factor += 1
        return LifecycleStatus.OK

    def compute(self, context):
        self.out.emit(self.inp.receive_data() * self.factor)


def options():
    return hs.CompileOptions(
        lifecycle_transactions=LifecycleTransactionOptions(
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
    )


# REQ: PY-API-031
def test_same_session_external_restart_with_canonical_control():
    with OperatorTestHarness(Restartable) as harness:
        harness.set_compile_options(options())
        harness.set_lifecycle_options(TestLifecycleOptions(allow_external_restart=True))
        harness.push_frame({"in": 21})
        harness.push_frame({"in": 5})
        with harness.run_async(RunOptions(frames=FrameRelease.MANUAL)) as running:
            target = running.operator_ref()
            control = running.lifecycle_control(
                LifecycleAccessPolicy(targets=[target], from_stages=[LifecycleStage.ARM])
            )
            running.release_frame(timeout_ms=OBSERVATION_MS)
            assert running.wait_for_output("out", 1, timeout_ms=OBSERVATION_MS)
            receipt = control.request_restart(
                target, LifecycleRestartOptions(from_stage=LifecycleStage.ARM)
            )
            terminal = control.await_terminal(receipt, timeout=OBSERVATION_SECONDS)
            assert terminal.state == LifecycleTransactionState.PUBLISHED
            assert running.operator_ref().run_generation == target.run_generation
            running.release_frame(timeout_ms=OBSERVATION_MS)
            result = running.finish()
        assert result.values("out") == [42, 15]


# REQ: PY-API-031
def test_restart_rejects_foreign_targets_invalid_stages_and_missing_permission():
    with OperatorTestHarness(Restartable) as first, OperatorTestHarness(Restartable) as second:
        for harness in (first, second):
            harness.set_compile_options(options())
            harness.set_lifecycle_options(TestLifecycleOptions(allow_external_restart=True))
            harness.push_frame({"in": 3})
            harness.push_frame({"in": 4})
        manual = RunOptions(frames=FrameRelease.MANUAL, expect_stop=True)
        with first.run_async(manual) as running, second.run_async(manual) as other:
            running.release_frame(timeout_ms=OBSERVATION_MS)
            assert running.wait_for_output("out", 1, timeout_ms=OBSERVATION_MS)
            target = running.operator_ref()
            foreign = other.operator_ref()
            control = running.lifecycle_control(
                LifecycleAccessPolicy(targets=[target], from_stages=[LifecycleStage.ARM])
            )
            with pytest.raises(hs.errors.StaleOperatorRefError):
                running.operator_ref(foreign)
            with pytest.raises(hs.errors.LifecycleRestartUnauthorizedError):
                control.request_restart(foreign)
            with pytest.raises(hs.errors.LifecycleInvalidRestartStageError):
                control.request_restart(
                    target, LifecycleRestartOptions(from_stage=LifecycleStage.STOP)
                )
            with pytest.raises(hs.errors.LifecycleRestartUnauthorizedError):
                control.request_restart(
                    target, LifecycleRestartOptions(from_stage=LifecycleStage.ALLOCATE)
                )
        first.push_frame({"in": 4})
        with first.run_async(manual) as newer:
            current = newer.operator_ref()
            assert current.run_generation != target.run_generation
            control = newer.lifecycle_control(
                LifecycleAccessPolicy(targets=[current], from_stages=[LifecycleStage.ARM])
            )
            with pytest.raises(hs.errors.StaleOperatorRefError):
                newer.operator_ref(target)
            with pytest.raises(hs.errors.LifecycleRestartUnauthorizedError):
                control.request_restart(target)

    with OperatorTestHarness(Restartable) as denied:
        denied.set_compile_options(options())
        denied.push_frame({"in": 1})
        denied.push_frame({"in": 2})
        with denied.run_async(manual) as running:
            running.release_frame(timeout_ms=OBSERVATION_MS)
            assert running.wait_for_output("out", 1, timeout_ms=OBSERVATION_MS)
            target = running.operator_ref()
            with pytest.raises(hs.errors.LifecycleRestartUnauthorizedError):
                running.lifecycle_control(
                    LifecycleAccessPolicy(targets=[target], from_stages=[LifecycleStage.ARM])
                )


# REQ: PY-API-031
def test_registered_operator_self_restart_preserves_the_same_running_attempt():
    holoscan_test_support._register_self_restart_probe_for_test()
    with OperatorTestHarness("holoscan.test.SelfRestartProbe") as harness:
        harness.set_compile_options(options())
        harness.set_lifecycle_options(
            TestLifecycleOptions(allow_self_restart=True, allow_external_restart=True)
        )
        harness.push_frame({"in": 0})
        harness.push_frame({"in": 1})
        harness.push_frame({"in": 2})
        with harness.run_async(RunOptions(frames=FrameRelease.MANUAL)) as running:
            target = running.operator_ref()
            control = running.lifecycle_control(
                LifecycleAccessPolicy(targets=[target], from_stages=[LifecycleStage.START])
            )
            running.release_frame(timeout_ms=OBSERVATION_MS)
            assert running.wait_for_output("out", 1, timeout_ms=OBSERVATION_MS)
            running.release_frame(timeout_ms=OBSERVATION_MS)
            # Restart drops the submitting callback's queued output. Observe its receipt
            # directly instead of treating that discarded output as a synchronization point.
            receipt = holoscan_test_support._self_restart_receipt(timeout=OBSERVATION_SECONDS)
            assert control.await_terminal(receipt, timeout=OBSERVATION_SECONDS).state == (
                LifecycleTransactionState.PUBLISHED
            )
            assert running.operator_ref().run_generation == target.run_generation
            running.release_frame(timeout_ms=OBSERVATION_MS)
            result = running.finish()
        assert result.values("out") == [0, 4]
        assert control.inspect(receipt).state == LifecycleTransactionState.PUBLISHED


# REQ: PY-API-031
def test_lifecycle_options_are_immutable_strict_and_keyword_only():
    value = TestLifecycleOptions(allow_external_restart=True)
    assert value.allow_external_restart
    assert not value.allow_self_restart
    with pytest.raises(FrozenInstanceError):
        value.allow_external_restart = False
    for bad in (None, 0, 1, "true"):
        with pytest.raises(TypeError, match="must be bool"):
            TestLifecycleOptions(allow_self_restart=bad)
        with pytest.raises(TypeError, match="must be bool"):
            TestLifecycleOptions(allow_external_restart=bad)
    with pytest.raises(TypeError):
        TestLifecycleOptions(True)


# REQ: PY-API-031
def test_lifecycle_options_replace_clear_and_reject_mutation_during_run():
    with OperatorTestHarness(Restartable) as harness:
        harness.set_compile_options(options())
        selected = TestLifecycleOptions(allow_external_restart=True)
        for mode, allowed in [
            ("default", False),
            ("set", True),
            ("reuse", True),
            ("replace", False),
            ("set", True),
            ("none", False),
            ("set", True),
            ("clear", False),
        ]:
            # Replacing or clearing permissions must preserve inputs already staged.
            harness.push_frame({"in": 17})
            if mode == "set":
                harness.set_lifecycle_options(selected)
                with pytest.raises(TypeError):
                    harness.set_lifecycle_options(object())
            elif mode == "replace":
                harness.set_lifecycle_options(TestLifecycleOptions(allow_self_restart=True))
            elif mode == "none":
                harness.set_lifecycle_options(None)
            elif mode == "clear":
                harness.clear_lifecycle_options()
            with harness.run_async(RunOptions(frames=FrameRelease.MANUAL)) as running:
                with pytest.raises(RuntimeError, match="active attempt"):
                    harness.clear_lifecycle_options()
                with pytest.raises(RuntimeError, match="active attempt"):
                    harness.set_lifecycle_options(selected)
                target = running.operator_ref()
                policy = LifecycleAccessPolicy(targets=[target], from_stages=[LifecycleStage.ARM])
                if allowed:
                    running.lifecycle_control(policy)
                else:
                    with pytest.raises(hs.errors.LifecycleRestartUnauthorizedError):
                        running.lifecycle_control(policy)
                running.release_frame(timeout_ms=OBSERVATION_MS)
                assert running.finish().value("out") == 34
    with pytest.raises(RuntimeError, match="harness is closed"):
        harness.set_lifecycle_options(selected)
    with pytest.raises(RuntimeError, match="harness is closed"):
        harness.clear_lifecycle_options()


# REQ: PY-API-030
def test_compile_options_are_copied_before_attempts():
    class Echo(hs.Operator):
        inp = hs.Input("in", hs.pyobject)
        out = hs.Output("out", hs.pyobject)

        def compute(self, context):
            self.out.emit(self.inp.receive_data())

    config = hs.CompileOptions(opaque_object_payloads=hs.OpaqueObjectPayloadPolicy.ALLOW)
    with OperatorTestHarness(Echo) as harness:
        harness.set_compile_options(config)
        config.opaque_object_payloads = hs.OpaqueObjectPayloadPolicy.REJECT
        harness.push("in", {"owned": True})
        assert harness.run().value("out") == {"owned": True}
