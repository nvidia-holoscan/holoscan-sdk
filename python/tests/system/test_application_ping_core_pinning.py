"""
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""  # noqa: E501

from holoscan.conditions import CountCondition
from holoscan.core import Application, IOSpec, Operator, OperatorSpec
from holoscan.operators import PingTxOp
from holoscan.schedulers import EventBasedScheduler


class PingRxOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        self.count = 1
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("receivers", size=IOSpec.ANY_SIZE)

    def compute(self, op_input, op_output, context):
        values = op_input.receive("receivers")
        assert values is not None
        print(f"received message {self.count}", flush=True)
        self.count += 1


class MyPingApp(Application):
    def __init__(
        self,
        *args,
        count=10,
        **kwargs,
    ):
        self.count = count
        super().__init__(*args, **kwargs)

    def compose(self):
        tx = PingTxOp(self, CountCondition(self, self.count), name="tx")
        rx = PingRxOp(self, name="rx")
        self.add_flow(tx, rx)


def test_event_based_scheduler_with_pin_cores(ping_config_file, capfd):
    """Test EventBasedScheduler with pin_cores parameter."""
    count = 5
    app = MyPingApp(count=count)
    app.config(ping_config_file)

    # Test with pin_cores parameter
    scheduler = EventBasedScheduler(
        app, worker_thread_number=2, name="ebs_pin_cores", max_duration_ms=5000, pin_cores=[0, 1]
    )
    app.scheduler(scheduler)
    app.run()

    # assert that the expected number of messages were received
    captured = capfd.readouterr()

    assert f"received message {count}" in captured.out
    assert f"received message {count + 1}" not in captured.out
    assert "error" not in captured.out
    assert "pinned to CPU cores: 0,1" in captured.err


def test_event_based_scheduler_without_pin_cores(ping_config_file, capfd):
    """Test EventBasedScheduler without pin_cores parameter."""
    count = 5
    app = MyPingApp(count=count)
    app.config(ping_config_file)

    # Test without pin_cores parameter
    scheduler = EventBasedScheduler(
        app, worker_thread_number=2, name="ebs_no_pin_cores", max_duration_ms=5000
    )
    app.scheduler(scheduler)
    app.run()

    # assert that the expected number of messages were received
    captured = capfd.readouterr()

    assert f"received message {count}" in captured.out
    assert f"received message {count + 1}" not in captured.out
    assert "pinned to CPU cores" not in captured.err
    assert "error" not in captured.out
