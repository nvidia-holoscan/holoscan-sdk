# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from argparse import ArgumentParser
from concurrent.futures import Future, ThreadPoolExecutor

from holoscan.conditions import AsynchronousCondition, AsynchronousEventState, CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import PingRxOp, PingTxOp
from holoscan.schedulers import EventBasedScheduler, GreedyScheduler, MultiThreadScheduler


class AsyncPingTxOp(Operator):
    """Asynchronous transmit operator.

    This operator sends a message asynchronously, where the delay between when the async event will
    be set to EVENT_DONE is specified by `delay`. After a specified count, the status will be set
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

        When the condition's event_state is EVENT_WAITING, set to EVENT_DONE. This function will
        only exit once the condition is set to EVENT_NEVER.
        """

        while True:
            try:
                print(f"waiting for {self.delay} s in AsyncPingTxOp.async_send")
                time.sleep(self.delay)
                if self.async_cond_.event_state == AsynchronousEventState.EVENT_WAITING:
                    self.async_cond_.event_state = AsynchronousEventState.EVENT_DONE
                elif self.async_cond_.event_state == AsynchronousEventState.EVENT_NEVER:
                    break
            except Exception as e:
                self.async_cond_.event_state = AsynchronousEventState.EVENT_NEVER
                raise (e)
        return

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def start(self):
        self.future_ = self.executor_.submit(self.aysnc_send)
        assert isinstance(self.future_, Future)

    def compute(self, op_input, op_output, context):
        self.iter += 1
        if self.iter < self.count:
            self.async_cond_.event_state = AsynchronousEventState.EVENT_WAITING
        else:
            self.async_cond_.event_state = AsynchronousEventState.EVENT_NEVER

        op_output.emit(self.iter, "out")

    def stop(self):
        self.async_cond_.event_state = AsynchronousEventState.EVENT_NEVER
        self.future_.result()


class AsyncPingRxOp(Operator):
    """Asynchronous transmit operator.

    This operator receives a message asynchronously, where the delay between when the async event
    will be set to EVENT_DONE is specified by `delay`.
    """

    def __init__(self, fragment, *args, delay=0.2, count=10, **kwargs):
        self.index = 0

        # delay used by async_receive
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

    def async_receive(self):
        """Function to be submitted to self.executor by start()

        When the condition's event_state is EVENT_WAITING, set to EVENT_DONE. This function will
        only exit once the condition is set to EVENT_NEVER.
        """

        while True:
            try:
                print(f"waiting for {self.delay} s in AsyncPingRxOp.async_receive")
                time.sleep(self.delay)
                if self.async_cond_.event_state == AsynchronousEventState.EVENT_WAITING:
                    self.async_cond_.event_state = AsynchronousEventState.EVENT_DONE
                elif self.async_cond_.event_state == AsynchronousEventState.EVENT_NEVER:
                    break
            except Exception as e:
                self.async_cond_.event_state = AsynchronousEventState.EVENT_NEVER
                raise (e)
        return

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def start(self):
        self.future_ = self.executor_.submit(self.async_receive)
        assert isinstance(self.future_, Future)

    def compute(self, op_input, op_output, context):
        value = op_input.receive("in")
        print(f"Rx message value: {value}")
        self.async_cond_.event_state = AsynchronousEventState.EVENT_WAITING

    def stop(self):
        self.async_cond_.event_state = AsynchronousEventState.EVENT_NEVER
        self.future_.result()


class MyPingApp(Application):
    def __init__(self, *args, delay=10, count=0, async_rx=True, async_tx=True, **kwargs):
        self.delay = delay
        self.count = count
        self.async_rx = async_rx
        self.async_tx = async_tx
        super().__init__(*args, **kwargs)

    def compose(self):
        # Define the tx and rx operators, allowing tx to execute 10 times
        if self.async_tx:
            tx = AsyncPingTxOp(self, count=self.count, delay=self.delay, name="tx")
        else:
            tx = PingTxOp(self, CountCondition(self, self.count), name="tx")
        if self.async_rx:
            rx = AsyncPingRxOp(self, delay=self.delay, name="rx")
        else:
            rx = PingRxOp(self, name="rx")

        # Define the workflow:  tx -> rx
        self.add_flow(tx, rx)


def main(delay_ms=100, count=10, async_rx=False, async_tx=False, scheduler="event_based"):
    app = MyPingApp(delay=delay_ms / 1000.0, count=count, async_rx=async_rx, async_tx=async_tx)
    if scheduler == "greedy":
        app.scheduler(GreedyScheduler(app))
    elif scheduler == "multi_thread":
        app.scheduler(MultiThreadScheduler(app, worker_thread_number=2))
    elif scheduler == "event_based":
        app.scheduler(EventBasedScheduler(app, worker_thread_number=2))
    else:
        raise ValueError(
            f"unrecognized scheduler '{scheduler}', should be one of ('greedy', 'multi_thread', "
            "'event_based')"
        )
    app.run()


if __name__ == "__main__":
    parser = ArgumentParser(
        description=(
            "Asynchronous operator example. By default, both message transmit and receive use an "
            "AsynchronousCondition."
        )
    )
    parser.add_argument(
        "-t",
        "--delay",
        type=int,
        default=100,
        help=(
            "The delay in ms that the async function will wait before updating from "
            "EVENT_WAITING to EVENT_DONE."
        ),
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=10,
        help=("The number of messages to transmit."),
    )
    parser.add_argument(
        "--sync_tx",
        action="store_true",
        help=(
            "Sets the application to use the synchronous PingTx transmitter instead of AsyncPingTx."
        ),
    )
    parser.add_argument(
        "--sync_rx",
        action="store_true",
        help=(
            "Sets the application to use the synchronous PingRx receiver instead of AsyncPingRx."
        ),
    )
    parser.add_argument(
        "-s",
        "--scheduler",
        type=str,
        default="event_based",
        choices=["event_based", "greedy", "multi_thread"],
        help="The scheduler to use for the application.",
    )
    args = parser.parse_args()
    if args.delay < 0:
        raise ValueError(f"delay must be non-negative, got {args.delay}")
    if args.count < 0:
        raise ValueError(f"count must be positive, got {args.count}")

    main(
        delay_ms=args.delay,
        count=args.count,
        async_rx=not args.sync_rx,
        async_tx=not args.sync_tx,
        scheduler=args.scheduler,
    )
