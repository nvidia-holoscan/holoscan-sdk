"""
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from holoscan.data_loggers import BasicConsoleLogger, SimpleTextSerializer
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
        rx.metadata_policy = MetadataPolicy.INPLACE_UPDATE
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
            assert rx.metadata_policy == MetadataPolicy.DEFAULT
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
    "update_policy",
    [
        MetadataPolicy.UPDATE,
        MetadataPolicy.INPLACE_UPDATE,
        MetadataPolicy.RAISE,
        MetadataPolicy.REJECT,
        None,
    ],
)
def test_metadata_duplicate_keys_app(capfd, scheduler_class, is_metadata_enabled, update_policy):
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


@pytest.mark.parametrize(
    "scheduler_class", [GreedyScheduler, MultiThreadScheduler, EventBasedScheduler]
)
@pytest.mark.parametrize("log_inputs,log_outputs", [(False, True), (True, False), (True, True)])
@pytest.mark.parametrize("log_metadata", [False, True])
@pytest.mark.parametrize("log_python_object_contents", [False, True])
def test_metadata_logging(
    scheduler_class,
    log_inputs,
    log_outputs,
    log_metadata,
    log_python_object_contents,
    capfd,
):
    count = 3
    app = MyPingParallelApp(count=count, rx_policy=MetadataPolicy.UPDATE)
    app.enable_metadata(True)

    app.scheduler(scheduler_class(app))

    data_logging_enabled = log_inputs or log_outputs
    if data_logging_enabled:
        basic_logger = BasicConsoleLogger(
            fragment=app,
            name="console-logger",
            log_inputs=log_inputs,
            log_outputs=log_outputs,
            log_metadata=log_metadata,
            log_tensor_data_content=False,
            serializer=SimpleTextSerializer(
                app,
                name="text-serializer",
                log_python_object_contents=log_python_object_contents,
            ),
        )
        app.add_data_logger(basic_logger)

    app.run()

    # assert that the expected logging data was recorded
    captured = capfd.readouterr()

    assert "rx: len(values)=8" in captured.out
    assert f"received message {count}" in captured.out
    assert f"received message {count + 1}" not in captured.out
    assert "metadata is enabled" in captured.out

    # The app sends a holoscan::Message containing an int in the std::any
    # Category info present whenever data logging is enabled
    category_str = "Category:Message (std::any)"
    expect_found = data_logging_enabled
    assert (category_str in captured.err) == expect_found

    # Python object contents only present when log_python_object_contents=True
    object_contents_str = "Python(int): 20"
    expect_found = data_logging_enabled and log_python_object_contents
    assert (object_contents_str in captured.err) == expect_found

    # metadata only present when log_metadata is enabled
    if log_python_object_contents:
        metadata_str = (
            "MetadataDictionary(size=3) {"
            "'multiplier': Python(int): 2, "
            "'channel_2_id': Python(str): 'evens', "
            "'channel_1_id': Python(str): 'odds'}"
        )
    else:
        metadata_str = (
            "MetadataDictionary(size=3) {"
            "'multiplier': Python Object, "
            "'channel_2_id': Python Object, "
            "'channel_1_id': Python Object}"
        )
    expect_found = data_logging_enabled and log_metadata
    assert (metadata_str in captured.err) == expect_found

    # output port names only present when log_inputs is enabled
    for rx_log_sr in ["ID:mx11.in1", "ID:rx.receivers"]:
        expect_found = data_logging_enabled and log_inputs
        assert (rx_log_sr in captured.err) == expect_found

    # output port names only present when log_outputs is enabled
    for tx_log_sr in ["ID:mx11.out1", "ID:tx.out1"]:
        expect_found = data_logging_enabled and log_outputs
        assert (tx_log_sr in captured.err) == expect_found


@pytest.mark.parametrize(
    "allowlist_patterns,denylist_patterns,expect_mx11,expect_mx12,expect_mx13",
    [
        ([], [], True, True, True),  # allow all
        ([".*mx11.*"], [], True, False, False),  # allow only mx11
        ([], [".*mx11.*", ".*mx13.*"], False, True, False),  # deny mx11 and mx13
        ([".*mx12.*"], [".*mx11.*"], False, True, False),  # warns, only allowlist is used
    ],
)
def test_data_logging_allowlist_denylist(
    allowlist_patterns,
    denylist_patterns,
    expect_mx11,
    expect_mx12,
    expect_mx13,
    capfd,
):
    count = 3
    app = MyPingParallelApp(count=count, rx_policy=MetadataPolicy.UPDATE)
    app.enable_metadata(False)

    app.scheduler(GreedyScheduler(app))

    expect_warn = (len(allowlist_patterns) > 0) and (len(denylist_patterns) > 0)
    warn_msg_cpp = "Both allowlist_patterns and denylist_patterns are specified"
    warn_msg_python = warn_msg_cpp

    logger_kwargs = dict(
        fragment=app,
        name="console-logger",
        log_inputs=True,
        log_outputs=True,
        log_metadata=False,
        log_tensor_data_content=False,
        allowlist_patterns=allowlist_patterns,
        denylist_patterns=denylist_patterns,
    )
    if expect_warn:
        with pytest.warns(UserWarning, match=warn_msg_python):
            basic_logger = BasicConsoleLogger(**logger_kwargs)
    else:
        basic_logger = BasicConsoleLogger(**logger_kwargs)

    app.add_data_logger(basic_logger)

    app.run()

    # assert that the expected logging data was recorded
    captured = capfd.readouterr()

    assert "rx: len(values)=8" in captured.out
    assert f"received message {count}" in captured.out
    assert f"received message {count + 1}" not in captured.out

    warn_str = "Both allowlist_patterns and denylist_patterns are specified"
    assert (warn_str in captured.err) == expect_warn

    mx11_str = "BasicConsoleLogger[ID:mx11"
    assert (mx11_str in captured.err) == expect_mx11

    mx12_str = "BasicConsoleLogger[ID:mx12"
    assert (mx12_str in captured.err) == expect_mx12

    mx13_str = "BasicConsoleLogger[ID:mx13"
    assert (mx13_str in captured.err) == expect_mx13
