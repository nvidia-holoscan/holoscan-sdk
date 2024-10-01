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

import pytest

from holoscan.conditions import CountCondition
from holoscan.core import Application, MetadataPolicy, Operator, OperatorSpec
from holoscan.schedulers import EventBasedScheduler, GreedyScheduler, MultiThreadScheduler


class PingMetadataTxOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        self.index = 0
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out1")
        spec.output("out2")

    def compute(self, op_input, op_output, context):
        self.index += 1
        meta = self.metadata
        if self.is_metadata_enabled:
            meta["channel_1_id"] = "odds"
        op_output.emit(self.index, "out1")

        self.index += 1
        if self.is_metadata_enabled:
            del meta["channel_1_id"]
            meta["channel_2_id"] = "evens"
        op_output.emit(self.index, "out2")


class PingMetadataMiddleOp(Operator):
    def __init__(self, fragment, multiplier=2, *args, **kwargs):
        self.count = 1
        self.multiplier = multiplier

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in1")
        spec.input("in2")
        spec.output("out1")
        spec.output("out2")

    def compute(self, op_input, op_output, context):
        value1 = op_input.receive("in1")
        value2 = op_input.receive("in2")
        self.count += 1

        # add the multiplier parameter used in the metadata
        if self.is_metadata_enabled:
            self.metadata["multiplier"] = self.multiplier

        op_output.emit(value1 * self.multiplier, "out1")
        op_output.emit(value2 * self.multiplier, "out2")


class PingMetadataRxOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        self.count = 1
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.param("receivers", kind="receivers")

    def compute(self, op_input, op_output, context):
        values = op_input.receive("receivers")
        print(f"rx: {len(values)=}")
        assert values is not None
        print(f"received message {self.count}")
        self.count += 1
        if self.is_metadata_enabled:
            print("metadata is enabled")
            assert "multiplier" in self.metadata
            assert "channel_1_id" in self.metadata
            assert "channel_2_id" in self.metadata
        else:
            print("metadata is disabled")
            assert "multiplier" not in self.metadata
            assert "channel_1_id" not in self.metadata
            assert "channel_2_id" not in self.metadata


class MyPingApp(Application):
    def __init__(self, *args, count=10, **kwargs):
        self.count = count
        super().__init__(*args, **kwargs)

    def compose(self):
        tx = PingMetadataTxOp(
            self,
            CountCondition(self, self.count),
            name="tx",
        )
        mx = PingMetadataMiddleOp(self, name="mx")
        rx = PingMetadataRxOp(self, name="rx")
        rx.metadata_policy = MetadataPolicy.UPDATE
        self.add_flow(tx, mx, {("out1", "in1"), ("out2", "in2")})
        self.add_flow(mx, rx, {("out1", "receivers"), ("out2", "receivers")})


class MyPingParallelApp(Application):
    """
    App with one transmitter, tx, being broadcast to 4 parallel branches that have two multiplier
    operators each. The outputs of all parallel branches connect to a common receiver, rx.

          | -> mx11 -> mx21 -> |
          | -> mx12 -> mx22 -> |
    tx -> | -> mx13 -> mx23 -> | -> rx
          | -> mx14 -> mx24 -> |

    """

    def __init__(self, *args, count=10, rx_policy=MetadataPolicy.RAISE, **kwargs):
        self.count = count
        self.rx_policy = rx_policy
        super().__init__(*args, **kwargs)

    def compose(self):
        tx = PingMetadataTxOp(
            self,
            CountCondition(self, self.count),
            name="tx",
        )
        mx11 = PingMetadataMiddleOp(self, name="mx11")
        mx11.metadata_policy = MetadataPolicy.UPDATE
        mx12 = PingMetadataMiddleOp(self, name="mx12")
        mx12.metadata_policy = MetadataPolicy.UPDATE
        mx13 = PingMetadataMiddleOp(self, name="mx13")
        mx13.metadata_policy = MetadataPolicy.UPDATE
        mx14 = PingMetadataMiddleOp(self, name="mx14")
        mx14.metadata_policy = MetadataPolicy.UPDATE
        mx21 = PingMetadataMiddleOp(self, name="mx21")
        mx21.metadata_policy = MetadataPolicy.UPDATE
        mx22 = PingMetadataMiddleOp(self, name="mx22")
        mx22.metadata_policy = MetadataPolicy.UPDATE
        mx23 = PingMetadataMiddleOp(self, name="mx23")
        mx23.metadata_policy = MetadataPolicy.UPDATE
        mx24 = PingMetadataMiddleOp(self, name="mx24")
        mx24.metadata_policy = MetadataPolicy.UPDATE
        rx = PingMetadataRxOp(self, name="rx")
        # leave at default policy for rx_policy=None
        if self.rx_policy is None:
            assert rx.metadata_policy == MetadataPolicy.RAISE
        else:
            rx.metadata_policy = self.rx_policy

        self.add_flow(tx, mx11, {("out1", "in1"), ("out2", "in2")})
        self.add_flow(tx, mx12, {("out1", "in1"), ("out2", "in2")})
        self.add_flow(tx, mx13, {("out1", "in1"), ("out2", "in2")})
        self.add_flow(tx, mx14, {("out1", "in1"), ("out2", "in2")})
        self.add_flow(mx11, mx21, {("out1", "in1"), ("out2", "in2")})
        self.add_flow(mx12, mx22, {("out1", "in1"), ("out2", "in2")})
        self.add_flow(mx13, mx23, {("out1", "in1"), ("out2", "in2")})
        self.add_flow(mx14, mx24, {("out1", "in1"), ("out2", "in2")})
        self.add_flow(mx21, rx, {("out1", "receivers"), ("out2", "receivers")})
        self.add_flow(mx22, rx, {("out1", "receivers"), ("out2", "receivers")})
        self.add_flow(mx23, rx, {("out1", "receivers"), ("out2", "receivers")})
        self.add_flow(mx24, rx, {("out1", "receivers"), ("out2", "receivers")})


@pytest.mark.parametrize("is_metadata_enabled", [False, True])
def test_my_ping_app(capfd, is_metadata_enabled):
    count = 100
    app = MyPingApp(count=count)
    app.is_metadata_enabled = is_metadata_enabled
    app.run()

    # assert that the expected number of messages were received
    captured = capfd.readouterr()

    assert "rx: len(values)=2" in captured.out
    assert f"received message {count}" in captured.out
    assert f"received message {count + 1}" not in captured.out
    assert f"metadata is {'enabled' if is_metadata_enabled else 'disabled'}" in captured.out


@pytest.mark.parametrize(
    "scheduler_class", [GreedyScheduler, MultiThreadScheduler, EventBasedScheduler]
)
@pytest.mark.parametrize("is_metadata_enabled", [False, True])
@pytest.mark.parametrize(
    "update_policy", [MetadataPolicy.UPDATE, MetadataPolicy.RAISE, MetadataPolicy.REJECT, None]
)
def test_my_ping_parallel_app(capfd, scheduler_class, is_metadata_enabled, update_policy):
    count = 3
    app = MyPingParallelApp(count=count, rx_policy=update_policy)
    app.is_metadata_enabled = is_metadata_enabled

    app.scheduler(scheduler_class(app))
    if is_metadata_enabled and (update_policy is None or update_policy == MetadataPolicy.RAISE):
        with pytest.raises(RuntimeError):
            app.run()

        # assert that the expected error message was logged
        captured = capfd.readouterr()
        assert "duplicate metadata keys" in captured.err

        return
    else:
        app.run()

        # assert that the expected number of messages were received
        captured = capfd.readouterr()

        assert "rx: len(values)=8" in captured.out
        assert f"received message {count}" in captured.out
        assert f"received message {count + 1}" not in captured.out
        assert f"metadata is {'enabled' if is_metadata_enabled else 'disabled'}" in captured.out
