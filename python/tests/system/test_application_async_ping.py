"""
SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import datetime
import time
from concurrent.futures import Future, ThreadPoolExecutor

import pytest

from holoscan.conditions import (
    AsynchronousCondition,
    AsynchronousEventState,
    CountCondition,
    PeriodicCondition,
)
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import PingRxOp
from holoscan.schedulers import EventBasedScheduler, GreedyScheduler, MultiThreadScheduler


class AsyncPingTxOp(Operator):
    """Asynchronous transmit operator.

    This operator sends a message asynchronously, where the delay between when the async event will
    be set to EVENT_DONE is specified by `delay`. After a total count of 10, the statue will be set
    to EVENT_NEVER, and the operator will stop sending messages.
    """

    def __init__(self, fragment, *args, delay=0.2, count=10, **kwargs):
        self.index = 0

        # counter to keep track of number of times compute was called
        self.iter = 0
        self.count = count
        self.delay = delay

        # add an asynchronous condition
        self.async_cond_ = AsynchronousCondition(fragment, name="async_cond")

        # thread pool with 1 worker to run async_send
        self.executor_ = ThreadPoolExecutor(max_workers=1)
        self.future_ = None  # will be set during start()

        # Need to call the base class constructor last
        # Note: It is essential that we pass self.async_cond_ to the parent
        # class constructor here.
        super().__init__(fragment, self.async_cond_, *args, **kwargs)

    def aysnc_send(self):
        """Function to be submitted to self.executor by start()

        When the condition's even_state is EVENT_WAITING, set to EVENT_DONE. This function will
        only exit once the condition is set to EVENT_NEVER.
        """
        while True:
            time.sleep(self.delay)
            print("in async_send")
            if self.async_cond_.event_state == AsynchronousEventState.EVENT_WAITING:
                self.async_cond_.event_state = AsynchronousEventState.EVENT_DONE
            elif self.async_cond_.event_state == AsynchronousEventState.EVENT_NEVER:
                break
        return

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def start(self):
        self.future_ = self.executor_.submit(self.aysnc_send)
        assert isinstance(self.future_, Future)

    def compute(self, op_input, op_output, context):
        print("in compute")
        self.iter += 1
        if self.iter < self.count:
            self.async_cond_.event_state = AsynchronousEventState.EVENT_WAITING
        else:
            self.async_cond_.event_state = AsynchronousEventState.EVENT_NEVER

        op_output.emit(self.iter, "out")

    def stop(self):
        self.async_cond_.event_state = AsynchronousEventState.EVENT_NEVER
        self.future_.result()


class MyAsyncPingApp(Application):
    def __init__(self, *args, count=10, delay=0.1, ext_count=None, ext_delay=None, **kwargs):
        self.count = count
        self.delay = delay
        self.ext_count = ext_count
        self.ext_delay = ext_delay
        super().__init__(*args, **kwargs)

    def compose(self):
        tx_args = []
        if self.ext_count:
            # add a count condition external to the count on the operator itself
            tx_args.append(CountCondition(self, count=self.ext_count))
        if self.ext_delay:
            # add a count condition external to the count on the operator itself
            tx_args.append(PeriodicCondition(self, recess_period=self.ext_delay))

        tx = AsyncPingTxOp(
            self,
            *tx_args,
            count=self.count,
            delay=self.delay,
            name="tx",
        )
        rx = PingRxOp(self, name="rx")
        self.add_flow(tx, rx)


@pytest.mark.parametrize(
    "scheduler_class", [GreedyScheduler, MultiThreadScheduler, EventBasedScheduler]
)
@pytest.mark.parametrize("extra_count_condition", [False, True])
@pytest.mark.parametrize("extra_periodic_condition", [False, True])
def test_my_ping_async_multicondition_event_wait_app(
    scheduler_class, extra_count_condition, extra_periodic_condition, capfd
):
    count = 8
    delay = 0.025

    if extra_count_condition:
        # add another CountCondition that would terminate the app earlier
        ext_count = max(count // 2, 1)
        expected_count = ext_count
    else:
        ext_count = None
        expected_count = count

    if extra_periodic_condition:
        # add another PeriodicCondition that would cause a longer period than delay
        expected_delay = 2 * delay
        ext_delay = datetime.timedelta(seconds=expected_delay)
    else:
        expected_delay = delay
        ext_delay = None

    app = MyAsyncPingApp(
        count=count,
        delay=delay,
        ext_count=ext_count,
        ext_delay=ext_delay,
    )

    # set scheduler (using its default options)
    app.scheduler(scheduler_class(app))

    t_start = time.time()
    app.run()
    duration = time.time() - t_start

    # overall duration must be longer than delays caused by async_send
    assert duration > expected_delay * (expected_count - 1)

    # assert that the expected number of messages were received
    captured = capfd.readouterr()
    assert f"Rx message value: {expected_count}" in captured.out
    assert f"Rx message value: {expected_count + 1}" not in captured.out
