# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static contract checked against the installed optional package."""

from typing import assert_type

import holoscan as hs
from holoscan.lifecycle import LifecycleAccessPolicy, LifecycleOperatorRef, LifecycleStage
from holoscan.tensor import MemoryConversionPolicy
from holoscan.testing import (
    OperatorTestHarness,
    RunningFixture,
    TestLifecycleOptions,
    TestResult,
    TestRunOptions,
    result_of,
    sample_at,
)


class Echo(hs.Operator):
    inp = hs.Input("in", hs.payload.uint32)
    out = hs.Output("out", hs.payload.uint32)

    def compute(self, context: hs.ExecutionContext) -> None:
        self.out.emit(self.inp.receive_data())


def contract() -> None:
    with OperatorTestHarness(Echo, params={}, outputs=["out"]) as harness:
        harness.push_samples("in", [sample_at(1, capture_timestamp_ns=0)])
        harness.set_input_flow(
            "in", queue_depth=2, memory_conversion=MemoryConversionPolicy.DEFAULT
        )
        harness.set_lifecycle_options(TestLifecycleOptions(allow_external_restart=True))
        assert_type(harness.run(check=False), TestResult)
        assert_type(harness.run().values("out"), list[object])
        assert_type(harness.run_async(TestRunOptions(timeout_ms=1000)), RunningFixture)
        with harness.run_async() as running:
            target = running.operator_ref()
            assert_type(target, LifecycleOperatorRef)
            running.lifecycle_control(LifecycleAccessPolicy([target], [LifecycleStage.ARM]))
            assert_type(running.finish(check=False), TestResult)


def failure_contract(error: BaseException) -> None:
    result = result_of(error)
    assert_type(result, TestResult | None)
    if result is not None:
        assert_type(result, TestResult)
        assert_type(result.values("out"), list[object])
