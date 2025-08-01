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
from holoscan.core import (
    Application,
    AsyncQueuePolicy,
    DataLoggerResource,
    MetadataPolicy,
    Operator,
    OperatorSpec,
)
from holoscan.data_loggers import (
    AsyncConsoleLogger,
    BasicConsoleLogger,
    GXFConsoleLogger,
    SimpleTextSerializer,
)
from holoscan.operators import PingRxOp, PingTensorRxOp, PingTensorTxOp
from holoscan.resources import UnboundedAllocator
from holoscan.schedulers import EventBasedScheduler, GreedyScheduler, MultiThreadScheduler

try:
    import cupy as cp
except ImportError:
    cp = None

logging_timeout = 100  # 0.1 seconds to complete data logging


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
        # add a second metadata key
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

    scheduler_kwargs = {"worker_thread_number": 8} if scheduler_class != GreedyScheduler else {}
    app.scheduler(scheduler_class(app, **scheduler_kwargs))

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
@pytest.mark.parametrize("logger_class", [AsyncConsoleLogger, BasicConsoleLogger, GXFConsoleLogger])
def test_metadata_logging(
    scheduler_class,
    log_inputs,
    log_outputs,
    log_metadata,
    log_python_object_contents,
    logger_class,
    capfd,
):
    count = 3
    app = MyPingParallelApp(count=count, rx_policy=MetadataPolicy.UPDATE)
    app.enable_metadata(True)

    scheduler_kwargs = {"worker_thread_number": 8} if scheduler_class != GreedyScheduler else {}
    scheduler_kwargs["stop_on_deadlock_timeout"] = logging_timeout

    app.scheduler(scheduler_class(app, **scheduler_kwargs))

    data_logging_enabled = log_inputs or log_outputs
    if data_logging_enabled:
        basic_logger = logger_class(
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

    if log_metadata and log_outputs:
        # Verify that both metadata entries are present for the first operator's output ports
        # (If AsyncDataLogger did not make a copy of the metadata dictionary this might not be true
        #  due to in-place updates made in PingMetadataTxOp between emit on "out1" and "out2")
        if log_python_object_contents:
            assert (
                "MetadataDictionary(size=1) {'channel_2_id': Python(str): 'evens'}" in captured.err
            )
            assert (
                "MetadataDictionary(size=1) {'channel_1_id': Python(str): 'odds'}" in captured.err
            )
        else:
            assert "MetadataDictionary(size=1) {'channel_2_id': Python Object}" in captured.err
            assert "MetadataDictionary(size=1) {'channel_1_id': Python Object}" in captured.err


@pytest.mark.parametrize(
    "allowlist_patterns,denylist_patterns,expect_mx11,expect_mx12,expect_mx13",
    [
        ([], [], True, True, True),  # allow all
        ([".*mx11.*"], [], True, False, False),  # allow only mx11
        ([], [".*mx11.*", ".*mx13.*"], False, True, False),  # deny mx11 and mx13
        ([".*mx12.*"], [".*mx11.*"], False, True, False),
    ],
)
@pytest.mark.parametrize("logger_class", [AsyncConsoleLogger, BasicConsoleLogger, GXFConsoleLogger])
def test_data_logging_allowlist_denylist(
    allowlist_patterns,
    denylist_patterns,
    expect_mx11,
    expect_mx12,
    expect_mx13,
    logger_class,
    capfd,
):
    count = 3
    app = MyPingParallelApp(count=count, rx_policy=MetadataPolicy.UPDATE)
    app.enable_metadata(False)

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
    console_logger = logger_class(**logger_kwargs)
    app.add_data_logger(console_logger)

    scheduler_kwargs = {"stop_on_deadlock_timeout": logging_timeout}
    app.scheduler(GreedyScheduler(fragment=app, **scheduler_kwargs))
    app.run()

    # assert that the expected logging data was recorded
    captured = capfd.readouterr()

    assert "rx: len(values)=8" in captured.out
    assert f"received message {count}" in captured.out
    assert f"received message {count + 1}" not in captured.out

    mx11_str = "ConsoleLogger[ID:mx11"
    assert (mx11_str in captured.err) == expect_mx11

    mx12_str = "ConsoleLogger[ID:mx12"
    assert (mx12_str in captured.err) == expect_mx12

    mx13_str = "ConsoleLogger[ID:mx13"
    assert (mx13_str in captured.err) == expect_mx13


class TensorMetadataMiddleOp(Operator):
    def __init__(self, fragment, value=5, emit_mode="tensormap", *args, **kwargs):
        self.count = 1
        self.value = value
        self.emit_mode = emit_mode

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in_tensor")
        spec.output("out_tensor")

    def compute(self, op_input, op_output, context):
        tensormap = op_input.receive("in_tensor")
        if not tensormap:
            raise RuntimeError("no tensor received")
        tensor = cp.asarray(tensormap["image"])
        tensor[:] = self.value

        meta = self.metadata
        if self.is_metadata_enabled:
            meta["value"] = self.value

        if self.emit_mode == "tensormap":
            op_output.emit(dict(constant_tensor=tensor), "out_tensor")
        elif self.emit_mode == "tensor":
            op_output.emit(tensor, "out_tensor", emitter_name="holoscan::Tensor")
        elif self.emit_mode == "ndarray":
            # emit the Python object directly
            op_output.emit(tensor, "out_tensor")
        else:
            raise ValueError("emit_mode must be one of {'tensormap', 'tensor', 'ndarray'}")


class TensorConsoleLoggingApp(Application):
    def __init__(
        self,
        *args,
        count=10,
        value=5,
        max_elements=10,
        logger_class=BasicConsoleLogger,
        extra_logger_kwargs=None,
        mx_emit_mode="tensormap",
        **kwargs,
    ):
        self.count = count
        self.value = value
        self.max_elements = max_elements
        if logger_class != "multiple" and not issubclass(logger_class, DataLoggerResource):
            raise ValueError(
                "logger_class must be the string 'multiple' or an instance of DataLoggerResource"
            )
        self.logger_class = logger_class
        self.mx_emit_mode = mx_emit_mode
        self.extra_logger_kwargs = extra_logger_kwargs or {}
        super().__init__(*args, **kwargs)

    def compose(self):
        tx = PingTensorTxOp(
            self,
            CountCondition(self, self.count),
            allocator=UnboundedAllocator(self, name="alloc"),
            storage_type="system",
            rows=512,
            columns=256,
            channels=4,
            dtype="uint16_t",
            tensor_name="image",
            name="tx",
        )
        mx = TensorMetadataMiddleOp(
            self, value=self.value, emit_mode=self.mx_emit_mode, name="value-setter"
        )
        if self.mx_emit_mode == "ndarray":
            # PingRxOp can receive arbitrary Python objects
            rx = PingRxOp(self, name="rx")
        else:
            # PingTensorRxOp wraps a C++ operator that only receives holoscan::Tensor, not Python
            # objects
            rx = PingTensorRxOp(self, name="rx")
        self.add_flow(tx, mx)
        self.add_flow(mx, rx)

        logger_kwargs = dict(
            fragment=self,
            log_inputs=True,
            log_outputs=True,
            log_metadata=True,
            log_tensor_data_content=True,
            serializer=SimpleTextSerializer(
                self,
                log_python_object_contents=True,
                max_elements=self.max_elements,
            ),
        )
        if self.logger_class == AsyncConsoleLogger:
            logger_kwargs.update(self.extra_logger_kwargs)

        if self.logger_class == "multiple":
            for logger_cls in [AsyncConsoleLogger, GXFConsoleLogger, BasicConsoleLogger]:
                self.add_data_logger(logger_cls(**logger_kwargs))
        else:
            # enable data logging from compose
            self.add_data_logger(self.logger_class(**logger_kwargs))


@pytest.mark.parametrize(
    "logger_class, extra_logger_kwargs",
    [
        (AsyncConsoleLogger, {"enable_large_data_queue": False}),
        (AsyncConsoleLogger, {"enable_large_data_queue": True}),
        (BasicConsoleLogger, {}),
        (GXFConsoleLogger, {}),
    ],
)
@pytest.mark.parametrize("mx_emit_mode", ["tensormap", "tensor", "ndarray"])
@pytest.mark.skipif(cp is None, reason="cupy not available")
def test_tensor_content_logging(logger_class, extra_logger_kwargs, mx_emit_mode, capfd):
    count = 2
    value = 5
    max_elements = 25
    app = TensorConsoleLoggingApp(
        count=count,
        value=value,
        max_elements=max_elements,
        logger_class=logger_class,
        mx_emit_mode=mx_emit_mode,
        extra_logger_kwargs=extra_logger_kwargs,
    )
    scheduler_kwargs = {"stop_on_deadlock_timeout": logging_timeout}
    app.scheduler(GreedyScheduler(fragment=app, **scheduler_kwargs))
    app.run()

    # assert that the expected logging data was recorded
    captured = capfd.readouterr()

    # each port was logged exactly once
    assert captured.err.count("[ID:rx.in]") == count
    assert captured.err.count("[ID:tx.out]") == count
    assert captured.err.count("[ID:value-setter.in_tensor]") == count
    assert captured.err.count("[ID:value-setter.out_tensor]") == count

    if mx_emit_mode == "tensormap":
        assert "constant_tensor" in captured.err
        assert "#cupy: tensor" not in captured.err
    elif mx_emit_mode == "tensor":
        # unnamed GPU tensor receives name "#cupy: tensor"
        assert "constant_tensor" not in captured.err
        assert "#cupy: tensor" in captured.err

    if mx_emit_mode != "ndarray":
        # tensor shape printed unless raw Python object was emitted
        assert "Tensor(shape=[512, 256, 4], dtype=kDLUInt:16:1" in captured.err

    data_values_logged = not (
        logger_class == AsyncConsoleLogger and not extra_logger_kwargs["enable_large_data_queue"]
    )
    if data_values_logged:
        if mx_emit_mode == "ndarray":
            # Python object logging via repr()
            # "[[[5, 5, 5, 5],"" because we have a 4-channel RGBA-style image
            expected_data = "Python(ndarray): array([[[5, 5, 5, 5],"
            assert expected_data in captured.err
        else:
            # exactly max_elements values were logged
            expected_data = "data=[" + ", ".join([f"{value}"] * max_elements) + ", ..."
            assert expected_data in captured.err

    # tx operator will have logged empty metadata
    assert "MetadataDictionary(size=0)" in captured.err

    # metadata value added by 'mx' operator was logged
    assert f"MetadataDictionary(size=1) {{'value': Python(int): {value}}}" in captured.err


@pytest.mark.parametrize("large_data_max_queue_size", [1, 1000])
@pytest.mark.parametrize(
    "large_data_queue_policy", [AsyncQueuePolicy.REJECT, AsyncQueuePolicy.RAISE]
)
@pytest.mark.parametrize(
    "scheduler_class", [GreedyScheduler, MultiThreadScheduler, EventBasedScheduler]
)
@pytest.mark.skipif(cp is None, reason="cupy not available")
def test_async_logger_small_entry_fallback(
    large_data_max_queue_size, large_data_queue_policy, scheduler_class, capfd
):
    """Limit large data queue size to 1 to test fallback to small data entry
    code path."""
    count = 25
    value = 3
    max_elements = 5000  # log many elements to increase likelihood of queue overflow
    app = TensorConsoleLoggingApp(
        count=count,
        value=value,
        max_elements=max_elements,
        logger_class=AsyncConsoleLogger,
        mx_emit_mode="tensormap",
        extra_logger_kwargs=dict(
            enable_large_data_queue=True,
            large_data_max_queue_size=large_data_max_queue_size,
            large_data_queue_policy=large_data_queue_policy,
        ),
    )
    scheduler_kwargs = {"worker_thread_number": 3} if scheduler_class != GreedyScheduler else {}
    scheduler_kwargs["stop_on_deadlock_timeout"] = logging_timeout

    app.scheduler(scheduler_class(fragment=app, **scheduler_kwargs))

    app.run()

    # assert that the expected logging data was recorded
    captured = capfd.readouterr()

    # BasicConsoleLogger doesn't log GXF entities so only message and metadata on rx.in will be
    # logged in that case.
    assert captured.err.count("[ID:rx.in]") == count
    assert captured.err.count("[ID:tx.out]") == count
    assert captured.err.count("[ID:value-setter.in_tensor]") == count
    assert captured.err.count("[ID:value-setter.out_tensor]") == count

    assert "constant_tensor" in captured.err
    assert "#cupy: tensor" not in captured.err

    # tensor shape printed unless raw Python object was emitted
    assert "Tensor(shape=[512, 256, 4], dtype=kDLUInt:16:1" in captured.err

    # exactly max_elements values were logged
    expected_data = "data=[" + ", ".join([f"{value}"] * max_elements) + ", ..."
    assert expected_data in captured.err

    # tx operator will have logged empty metadata
    assert "MetadataDictionary(size=0)" in captured.err

    # metadata value added by 'mx' operator was logged
    assert f"MetadataDictionary(size=1) {{'value': Python(int): {value}}}" in captured.err

    # all ports should be logging tensor data
    num_ports = 4  # tx.out, value-setter.in_tensor, value-setter.out_tensor, rx.in
    num_large_entries = count * num_ports
    if large_data_max_queue_size == 1:
        # Likely that at least some large data entries to the log_entry code path instead due to
        # the large data queue being full.
        no_fallbacks = "entries: 0" in captured.err

        # check that exceptions were raised as expected, but app will not have terminated
        if large_data_queue_policy == AsyncQueuePolicy.RAISE:
            if not no_fallbacks:
                assert "Exception during large data enqueueing" in captured.err
        else:
            assert "Exception during large data enqueueing" not in captured.err

        # Note: in this case there may be some residual errors logged at the GXF level:
        # [error] [program.cpp:614] Event notification 2 for entity [tx] with id [7] received in an unexpected state [Activated]  # noqa: E501
        # ("Event notification 2" corresponds to GXF_EVENT_MESSAGE_SYNC)
    elif large_data_max_queue_size >= num_large_entries:
        assert f"Large entries: {num_large_entries}" in captured.err
        assert "Exception during large data enqueueing" not in captured.err


@pytest.mark.parametrize(
    "scheduler_class", [GreedyScheduler, MultiThreadScheduler, EventBasedScheduler]
)
@pytest.mark.skipif(cp is None, reason="cupy not available")
def test_simultaneous_console_loggers(scheduler_class, capfd):
    """Test simultaneous console loggers."""
    count = 5
    value = 7
    max_elements = 10
    app = TensorConsoleLoggingApp(
        count=count,
        value=value,
        max_elements=max_elements,
        logger_class="multiple",
        mx_emit_mode="tensormap",
    )
    scheduler_kwargs = {"worker_thread_number": 3} if scheduler_class != GreedyScheduler else {}
    scheduler_kwargs["stop_on_deadlock_timeout"] = logging_timeout

    app.scheduler(scheduler_class(fragment=app, **scheduler_kwargs))
    app.run()

    # assert that the expected logging data was recorded
    captured = capfd.readouterr()

    num_loggers = 3  # logger_class='multiple' configures three loggers
    assert captured.err.count("[ID:rx.in]") == count * num_loggers
    assert captured.err.count("[ID:tx.out]") == count * num_loggers
    assert captured.err.count("[ID:value-setter.in_tensor]") == count * num_loggers
    assert captured.err.count("[ID:value-setter.out_tensor]") == count * num_loggers
