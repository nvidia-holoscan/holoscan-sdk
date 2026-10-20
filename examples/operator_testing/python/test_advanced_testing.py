# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit failure inspection and same-attempt restart with finite manual input."""

import pytest
from test_operator_testing import ScaleOp

import holoscan as hs
from holoscan.testing import (
    FrameRelease,
    OperatorTestHarness,
    TestLifecycleOptions,
    TestRunOptions,
    result_of,
)


def test_expected_capture_failure():
    with OperatorTestHarness(ScaleOp) as harness:
        harness.capture("out", capacity=1)
        harness.set_output_flow("out", queue_depth=2)
        harness.push_many("in", [1, 2])
        with pytest.raises(hs.errors.ResourceExhaustedError) as caught:
            harness.run()
    result = result_of(caught.value)
    assert result is not None
    assert not result.ok
    assert result.values("out") == [2]


class Restartable(ScaleOp):
    def setup(self, spec):
        spec.input(self.inp)
        spec.output(self.out)
        spec.lifecycle().stage(hs.lifecycle.LifecycleStage.ARM, "arm")

    def arm(self, context):
        return hs.lifecycle.LifecycleStatus.OK


def test_managed_restart():
    with OperatorTestHarness(Restartable) as harness:
        harness.set_compile_options(
            hs.CompileOptions(
                lifecycle_transactions=hs.lifecycle.LifecycleTransactionOptions(
                    max_pending_requests=8,
                    max_pending_transactions=8,
                    max_receipts=16,
                    max_fenced_operators=8,
                    max_candidate_generations=8,
                    max_events=128,
                    max_audit_records=32,
                    max_quiesce_records=32,
                    max_rollback_records=64,
                )
            )
        )

        harness.set_lifecycle_options(TestLifecycleOptions(allow_external_restart=True))
        harness.push_frame({"in": 21})
        harness.push_frame({"in": 5})
        with harness.run_async(TestRunOptions(frames=FrameRelease.MANUAL)) as running:
            target = running.operator_ref()
            control = running.lifecycle_control(
                hs.lifecycle.LifecycleAccessPolicy(
                    targets=[target], from_stages=[hs.lifecycle.LifecycleStage.ARM]
                )
            )
            running.release_frame(timeout_ms=2000)
            assert running.wait_for_output("out", 1, timeout_ms=2000)
            receipt = control.request_restart(
                target,
                hs.lifecycle.LifecycleRestartOptions(from_stage=hs.lifecycle.LifecycleStage.ARM),
            )
            terminal = control.await_terminal(receipt, timeout=2.0)
            assert terminal.state == hs.lifecycle.LifecycleTransactionState.PUBLISHED
            running.release_frame(timeout_ms=2000)
            result = running.finish()
    assert result.values("out") == [42, 10]
