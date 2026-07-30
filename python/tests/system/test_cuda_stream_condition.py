"""
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""  # noqa: E501

import pytest

from holoscan.conditions import CountCondition, CudaStreamCondition
from holoscan.core import Application, ConditionType, IOSpec, Operator, OperatorSpec
from holoscan.operators import PingTensorTxOp

"""
Tests for CudaStreamCondition and CudaStreamCondition.

These are Python equivalents of a subset of the C++ tests in
public/tests/system/cuda_stream_condition.cpp.
"""


class DualPortsRxOp(Operator):
    """Receiver operator with two regular input ports.

    This tests CudaStreamCondition(s) monitoring multiple regular input ports.
    """

    def __init__(self, fragment, *args, **kwargs):
        self.count = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        # Two input ports - conditions will be added externally
        spec.input("in1").condition(ConditionType.NONE)
        spec.input("in2").condition(ConditionType.NONE)

    def compute(self, op_input, op_output, context):
        msg1 = op_input.receive("in1")
        msg2 = op_input.receive("in2")

        has_msg1 = msg1 is not None
        has_msg2 = msg2 is not None

        self.count += 1
        print(
            f"{self.name}: compute() called, count = {self.count}, "
            f"in1 = {'received' if has_msg1 else 'empty'}, "
            f"in2 = {'received' if has_msg2 else 'empty'}"
        )


class CudaStreamConditionDualApp(Application):
    """Test application for CudaStreamCondition(s) with two input ports.

    When use_legacy=False (default): Uses a single CudaStreamCondition to monitor both ports.
    When use_legacy=True: Uses TWO separate CudaStreamConditions (one per port).

    This demonstrates the advantage of CudaStreamCondition over the legacy approach.
    """

    def __init__(self, *args, use_legacy=False, **kwargs):
        self.use_legacy = use_legacy
        super().__init__(*args, **kwargs)

    def compose(self):
        # Common transmitter arguments
        tx_kwargs = dict(
            rows=64,
            columns=64,
            channels=4,
            storage_type="device",
            async_device_allocation=True,
        )

        tx1 = PingTensorTxOp(
            self,
            CountCondition(self, 5),
            name="tx1",
            **tx_kwargs,
        )
        tx2 = PingTensorTxOp(
            self,
            CountCondition(self, 5),
            name="tx2",
            **tx_kwargs,
        )

        if self.use_legacy:
            # Legacy API approach: need TWO separate CudaStreamConditions, one for each port
            stream_cond1 = CudaStreamCondition(self, receiver="in1", name="stream_cond1")
            stream_cond2 = CudaStreamCondition(self, receiver="in2", name="stream_cond2")
            rx = DualPortsRxOp(self, stream_cond1, stream_cond2, name="rx")
        else:
            # New API approach: single condition monitors both ports
            stream_cond = CudaStreamCondition(
                self,
                receivers=["in1", "in2"],
                name="stream_cond",
            )
            rx = DualPortsRxOp(self, stream_cond, name="rx")

        self.add_flow(tx1, rx, {("out", "in1")})
        self.add_flow(tx2, rx, {("out", "in2")})


class MixedPortsRxOp(Operator):
    """Receiver operator with BOTH a regular port AND a kAnySize multi-receiver port.

    This tests CudaStreamCondition monitoring both types of ports simultaneously.
    """

    def __init__(self, fragment, *args, **kwargs):
        self.count = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        # Regular input port (disable default MessageAvailable condition)
        spec.input("regular_in").condition(ConditionType.NONE)
        # Multi-receiver port (disable default MessageAvailable condition)
        spec.input("multi_in", size=IOSpec.ANY_SIZE).condition(ConditionType.NONE)

    def compute(self, op_input, op_output, context):
        # Receive from regular port
        regular_msg = op_input.receive("regular_in")
        has_regular = regular_msg is not None

        # Receive from multi-receiver port
        multi_msgs = op_input.receive("multi_in")
        multi_count = len(multi_msgs) if multi_msgs is not None else 0

        self.count += 1
        print(
            f"{self.name}: compute() called, count = {self.count}, "
            f"regular_in = {'received' if has_regular else 'empty'}, "
            f"multi_in count = {multi_count}"
        )


class CudaStreamConditionMixedApp(Application):
    """Test application for CudaStreamCondition with BOTH kAnySize and regular ports.

    Tests that CudaStreamCondition correctly handles a mix of regular ports and
    kAnySize multi-receiver ports.
    """

    def compose(self):
        # Common transmitter arguments
        # PingTensorTxOp will use its default internal CudaStreamPool
        tx_kwargs = dict(
            rows=64,
            columns=64,
            channels=4,
            storage_type="device",
            async_device_allocation=True,
        )

        # One source for regular port
        tx_regular = PingTensorTxOp(
            self,
            CountCondition(self, 5),
            name="tx_regular",
            **tx_kwargs,
        )

        # Two sources for multi-receiver port
        tx_multi1 = PingTensorTxOp(
            self,
            CountCondition(self, 5),
            name="tx_multi1",
            **tx_kwargs,
        )
        tx_multi2 = PingTensorTxOp(
            self,
            CountCondition(self, 5),
            name="tx_multi2",
            **tx_kwargs,
        )

        # Create CudaStreamCondition monitoring both "regular_in" and "multi_in" (base name)
        stream_cond = CudaStreamCondition(
            self,
            receivers=["regular_in", "multi_in"],
            name="stream_cond",
        )

        rx = MixedPortsRxOp(self, stream_cond, name="rx")

        # Connect regular port
        self.add_flow(tx_regular, rx, {("out", "regular_in")})
        # Connect multi-receiver port (creates multi_in:0, multi_in:1)
        self.add_flow(tx_multi1, rx, {("out", "multi_in")})
        self.add_flow(tx_multi2, rx, {("out", "multi_in")})


def test_cuda_stream_condition_mixed_ports(capfd):
    """Test CudaStreamCondition with both regular and multi-receiver ports."""
    app = CudaStreamConditionMixedApp()

    # Run the application
    app.run()

    # Check the output
    captured = capfd.readouterr()

    # Verify that compute was called with both regular and multi inputs
    # regular_in should be received, multi_in should have 2 messages (from 2 sources)
    expected_msg = "rx: compute() called, count = 5, regular_in = received, multi_in count = 2"
    assert expected_msg in captured.out, (
        f"Expected 5 compute() calls with regular_in received and 2 multi_in messages:\n"
        f"=== OUTPUT ===\n{captured.out}\n==============\n"
        f"=== STDERR ===\n{captured.err}\n==============\n"
    )

    # Verify no errors in stderr
    assert "error" not in captured.err.lower(), (
        f"Unexpected error in output:\n=== STDERR ===\n{captured.err}\n==============\n"
    )


@pytest.mark.parametrize("use_legacy", [False, True], ids=["NewAPI", "LegacyAPI"])
def test_cuda_stream_condition_dual_ports(use_legacy, capfd):
    """Test CudaStreamCondition(s) with two regular input ports.

    Parameterized test that runs with both:
    - use_legacy=False: Single CudaStreamCondition for both ports ("receivers" parameter).
    - use_legacy=True (Legacy): Two separate CudaStreamConditions, one per port
      (using the legacy "receiver" parameter)
    """
    app = CudaStreamConditionDualApp(use_legacy=use_legacy)

    # Run the application
    app.run()

    # Check the output
    captured = capfd.readouterr()

    # Verify that compute was called and both inputs were received
    expected_msg = "rx: compute() called, count = 5, in1 = received, in2 = received"
    assert expected_msg in captured.out, (
        f"Expected 5 compute() calls with both inputs received:\n"
        f"=== OUTPUT ===\n{captured.out}\n==============\n"
        f"=== STDERR ===\n{captured.err}\n==============\n"
    )

    # Verify no errors in stderr
    assert "error" not in captured.err.lower(), (
        f"Unexpected error in output:\n=== STDERR ===\n{captured.err}\n==============\n"
    )
