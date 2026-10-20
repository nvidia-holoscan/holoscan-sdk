# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run with python -m pytest test_operator_testing.py using installed SDK packages."""

from typing import cast

import holoscan as hs
from holoscan.testing import OperatorTestHarness, sample_at


class ScaleOp(hs.Operator):
    inp = hs.Input("in", hs.payload.uint32)
    out = hs.Output("out", hs.payload.uint32)
    factor = hs.Param(2, range=(1, 8))

    def compute(self, context):
        sample = self.inp.receive()
        self.out.forward(sample, cast(int, sample.data) * self.factor)


def test_scale():
    with OperatorTestHarness(ScaleOp, params={"factor": 2}) as harness:
        harness.push_many("in", [1, 2, 3])
        result = harness.run()
    assert result.values("out") == [2, 4, 6]


def test_timestamped_input():
    with OperatorTestHarness(ScaleOp) as harness:
        harness.push_samples(
            "in",
            [
                sample_at(10, capture_timestamp_ns=0),
                sample_at(20, capture_timestamp_ns=100_000_000),
            ],
        )
        result = harness.run()
        assert result.values("out") == [20, 40]
        assert [sample.metadata.capture_timestamp_ns for sample in result.samples("out")] == [
            0,
            100_000_000,
        ]
