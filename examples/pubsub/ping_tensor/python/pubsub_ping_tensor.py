# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Pub/sub ping tensor example using the Fast-DDS backend.

Run the subscriber in one terminal:
    python pubsub_ping_tensor.py --role subscriber

Run the publisher in a second terminal:
    python pubsub_ping_tensor.py --role publisher --gpu --count 10
"""

import argparse

import cupy as cp
import numpy as np

from holoscan.conditions import (
    CountCondition,
    PeriodicCondition,
    PublisherAvailableCondition,
    SubscriberAvailableCondition,
)
from holoscan.core import Application, Operator, OperatorSpec, Tracker
from holoscan.network_contexts import FastDdsPubSubNetworkContext
from holoscan.schedulers import EventBasedScheduler

try:
    from holoscan.conditions import PendingExportCondition
except ImportError:
    PendingExportCondition = None

TOPIC_NAME = "ping_tensor_topic"
MAX_PENDING_NATIVE_EXPORTS = 4


class TensorTxOp(Operator):
    """Generates and publishes tensors on the pub/sub topic."""

    def __init__(
        self,
        fragment,
        *args,
        gpu=False,
        batch_size=0,
        rows=32,
        columns=64,
        channels=0,
        data_type="uint8",
        **kwargs,
    ):
        self.gpu = gpu
        self.batch_size = batch_size
        self.rows = rows
        self.columns = columns
        self.channels = channels
        self.data_type = data_type
        self._count = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out").topic(TOPIC_NAME)

    def compute(self, op_input, op_output, context):
        shape = []
        if self.batch_size > 0:
            shape.append(self.batch_size)
        shape.extend([self.rows, self.columns])
        if self.channels > 0:
            shape.append(self.channels)

        xp = cp if self.gpu else np
        tensor = xp.zeros(shape, dtype=self.data_type)

        self._count += 1
        op_output.emit({"tensor": tensor}, "out")
        print(
            f"tx sent message {self._count}: shape={tuple(shape)}, dtype={self.data_type}, "
            f"storage={'GPU' if self.gpu else 'host'}"
        )


class TensorRxOp(Operator):
    """Receives tensors from the pub/sub topic."""

    def __init__(self, fragment, *args, **kwargs):
        self._count = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in").topic(TOPIC_NAME)

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("in")
        if msg is None:
            return

        if isinstance(msg, dict):
            for key, tensor in msg.items():
                self._count += 1
                shape = tensor.shape if hasattr(tensor, "shape") else "unknown"
                print(f"rx received message {self._count}: key='{key}', shape={shape}")
        else:
            self._count += 1
            shape = msg.shape if hasattr(msg, "shape") else "unknown"
            print(f"rx received message {self._count}: shape={shape}")


class PublisherApp(Application):
    def __init__(self, args):
        super().__init__()
        self._args = args

    def compose(self):
        ctx = FastDdsPubSubNetworkContext(
            self,
            native_buffer_policy=self._args.native_buffer_policy,
            name="pubsub_context",
        )
        self.network_context(ctx)

        self.scheduler(
            EventBasedScheduler(
                self,
                worker_thread_number=2,
                stop_on_deadlock_timeout=300000,
                name="scheduler",
            )
        )

        period_ns = self._args.tx_period_ms * 1_000_000
        conditions = [
            CountCondition(self, self._args.count, name="tx_count"),
            PeriodicCondition(self, recess_period=period_ns, name="tx_period"),
            SubscriberAvailableCondition(
                self,
                transmitter="out",
                min_subscriber_count=1,
                poll_period_ms=100,
                stabilization_ms=500,
                latch_ready=True,
                name="subscriber_available",
            ),
        ]
        if PendingExportCondition is not None and self._args.native_buffer_policy != "disabled":
            conditions.append(
                PendingExportCondition(
                    self,
                    max_pending=MAX_PENDING_NATIVE_EXPORTS,
                    network_context=ctx,
                    name="pending_export_cond",
                )
            )

        tx = TensorTxOp(
            self,
            *conditions,
            gpu=self._args.gpu,
            batch_size=self._args.batch_size,
            rows=self._args.rows,
            columns=self._args.columns,
            channels=self._args.channels,
            data_type=self._args.data_type,
            name="tx",
        )
        self.add_operator(tx)


class SubscriberApp(Application):
    def __init__(self, args):
        super().__init__()
        self._args = args

    def compose(self):
        ctx = FastDdsPubSubNetworkContext(
            self,
            native_buffer_policy=self._args.native_buffer_policy,
            native_buffer_use_eager_acquire=self._args.eager,
            name="pubsub_context",
        )
        self.network_context(ctx)

        self.scheduler(
            EventBasedScheduler(
                self,
                worker_thread_number=2,
                stop_on_deadlock_timeout=300000,
                name="scheduler",
            )
        )

        conditions = [
            PublisherAvailableCondition(
                self,
                receiver="in",
                poll_period_ms=100,
                latch_ready=True,
                name="publisher_available",
            ),
        ]
        if self._args.count >= 0:  # negative = run indefinitely
            conditions.append(CountCondition(self, self._args.count, name="rx_count"))

        rx = TensorRxOp(self, *conditions, name="rx")
        self.add_operator(rx)


def main():
    parser = argparse.ArgumentParser(description="Pub/sub ping tensor example (Fast-DDS)")
    parser.add_argument(
        "--role", required=True, choices=["publisher", "subscriber"], help="Process role"
    )
    parser.add_argument("--gpu", action="store_true", help="Place tensors in GPU memory")
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Message count: publisher send count or subscriber receive target "
        "(default: 10). Negative = run indefinitely.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=0, help="Batch size (0 to omit dimension)"
    )
    parser.add_argument(
        "--rows", type=int, default=32, help="Number of rows in the tensor (default: 32)"
    )
    parser.add_argument(
        "--columns", type=int, default=64, help="Number of columns in the tensor (default: 64)"
    )
    parser.add_argument(
        "--channels", type=int, default=0, help="Number of channels (0 to omit dimension)"
    )
    parser.add_argument("--data_type", default="uint8", help="Tensor element type (default: uint8)")
    parser.add_argument(
        "--tx_period_ms",
        type=int,
        default=100,
        help="Publisher delay between messages in ms (default: 100)",
    )
    parser.add_argument(
        "--native_buffer_policy",
        default="preferred",
        choices=["disabled", "preferred", "required"],
        help="Native buffer policy (default: preferred)",
    )
    parser.add_argument(
        "--eager", action="store_true", help="Subscriber-side: enable eager CUDA IPC acquire"
    )
    parser.add_argument("--track", action="store_true", help="Enable Data Flow Tracking output")
    args = parser.parse_args()

    app = PublisherApp(args) if args.role == "publisher" else SubscriberApp(args)
    if args.track:
        with Tracker(app, num_start_messages_to_skip=0, num_last_messages_to_discard=0) as tracker:
            app.run()
            tracker.print()
    else:
        app.run()


if __name__ == "__main__":
    main()
