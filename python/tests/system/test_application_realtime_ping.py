"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
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
    scheduler = EventBasedScheduler(app, worker_thread_number=3, name="ebs", max_duration_ms=10000)
    app.scheduler(scheduler)
    app.run()

    # assert that the expected number of messages were received
    captured = capfd.readouterr()

    assert f"received message {count}" in captured.out
    assert f"received message {count + 1}" not in captured.out
    assert "error" not in captured.out
