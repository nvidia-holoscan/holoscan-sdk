"""
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""  # noqa: E501

import pytest

from holoscan.conditions import CountCondition
from holoscan.core import Application, IOSpec, Operator, OperatorSpec
from holoscan.operators import PingTxOp
from holoscan.resources import SchedulingPolicy
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
        print(f"received message {self.count}")
        self.count += 1


class MyPingApp(Application):
    def __init__(
        self,
        *args,
        sched_policy,
        count=10,
        **kwargs,
    ):
        self.count = count
        self.sched_policy = sched_policy
        super().__init__(*args, **kwargs)

    def compose(self):
        tx = PingTxOp(self, CountCondition(self, self.count), name="tx")
        rx = PingRxOp(self, name="rx")

        pool1 = self.make_thread_pool("pool1", 0)
        if self.sched_policy == SchedulingPolicy.SCHED_DEADLINE:
            pool1.add_realtime(
                tx,
                self.sched_policy,
                pin_operator=True,
                pin_cores=[0],
                sched_runtime=1000000,
                sched_deadline=10000000,
                sched_period=10000000,
            )
        elif (
            self.sched_policy == SchedulingPolicy.SCHED_FIFO
            or self.sched_policy == SchedulingPolicy.SCHED_RR
        ):
            pool1.add_realtime(
                tx, self.sched_policy, pin_operator=True, pin_cores=[0], sched_priority=1
            )
        else:
            raise ValueError(f"Invalid scheduling policy: {self.sched_policy}")

        pool1.add(rx, True)
        self.add_flow(tx, rx)


def file_contains_string(filename, string):
    try:
        with open(filename) as f:
            return string in f.read()

    except FileNotFoundError:
        return False


@pytest.mark.realtime
@pytest.mark.parametrize(
    "sched_policy",
    [SchedulingPolicy.SCHED_DEADLINE, SchedulingPolicy.SCHED_FIFO, SchedulingPolicy.SCHED_RR],
)
def test_my_realtime_ping_app(ping_config_file, sched_policy, capfd):
    count = 10
    app = MyPingApp(count=count, sched_policy=sched_policy)
    app.config(ping_config_file)
    worker_thread_number = 3
    scheduler = EventBasedScheduler(
        app, worker_thread_number=worker_thread_number, name="ebs", max_duration_ms=10000
    )
    app.scheduler(scheduler)
    app.run()

    # assert that the expected number of messages were received
    captured = capfd.readouterr()

    if "Failed to set SCHED_DEADLINE policy" in captured.err:
        pytest.skip("Environment has insufficient permissions to set the SCHED_DEADLINE policy")
    assert f"received message {count}" in captured.out
    assert f"received message {count + 1}" not in captured.out
    assert "error" not in captured.out

    # there is a single real-time worker thread in pool1 (corresponding to operator "tx")
    assert captured.err.count("started real-time worker thread [pool name: pool1") == 1
    # there is a single non-real-time worker thread in pool1 (corresponding to operator "rx")
    assert captured.err.count("started worker thread [pool name: pool1") == 1
    # there are 3 threads in the default pool (corresponding to worker_thread_number)
    default_thread_pattern = "Event Based scheduler started worker thread [pool name: default_pool"
    assert captured.err.count(default_thread_pattern) == worker_thread_number
