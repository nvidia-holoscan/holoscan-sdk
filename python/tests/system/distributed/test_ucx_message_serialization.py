"""
SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""  # noqa: E501

import contextlib
import sys

import pytest

from holoscan.conditions import CountCondition
from holoscan.core import Application, Fragment, IOSpec, Operator, OperatorSpec
from holoscan.operators.holoviz import HolovizOp

from ...utils import is_torch_cuda_compatible
from .utils import remove_ignored_errors

with contextlib.suppress(ImportError):
    import cupy as cp
    import numpy as np
    import torch

# test HolovizOp InputSpec
test_input_specs = []
test_text_spec = HolovizOp.InputSpec("dynamic_text", "text")
test_text_spec.text = ["Text1"]
test_text_spec.color = [1.0, 0.0, 0.0, 1.0]
test_input_specs.append(test_text_spec)
test_input_specs.append(HolovizOp.InputSpec("triangles", HolovizOp.InputType.TRIANGLES))
# create a full spec with all fields different from default
test_full_spec = HolovizOp.InputSpec("full", HolovizOp.InputType.COLOR)
test_full_spec.color = [0.5, 0.1, 0.2, 0.8]
test_full_spec.opacity = 0.124
test_full_spec.priority = 12
test_full_spec.image_format = HolovizOp.ImageFormat.R32G32B32A32_SFLOAT
test_full_spec.line_width = 12.0
test_full_spec.point_size = 24.0
test_full_spec.text = ["abc"]
test_full_spec.yuv_model_conversion = HolovizOp.YuvModelConversion.YUV_2020
test_full_spec.yuv_range = HolovizOp.YuvRange.ITU_NARROW
test_full_spec.x_chroma_location = HolovizOp.ChromaLocation.MIDPOINT
test_full_spec.y_chroma_location = HolovizOp.ChromaLocation.MIDPOINT
test_full_spec.depth_map_render_mode = HolovizOp.DepthMapRenderMode.LINES
test_view = HolovizOp.InputSpec.View()
test_view.offset_x = 0.2
test_view.offset_y = 1.3
test_view.width = 4.0
test_view.height = 2.8
test_view.matrix = np.arange(16.0, dtype=float)
test_full_spec.views = [test_view]
test_input_specs.append(test_full_spec)


class PingMessageTxOp(Operator):
    """Simple transmitter operator.

    This operator has a single output port:
        output: "out"

    On each tick, it transmits the object passed via `value`.
    If `value` is a list, it cycles through the values on each compute call.
    """

    def __init__(self, fragment, *args, value=1, **kwargs):
        self.value = value
        self.values = value if isinstance(value, list) else [value]
        self.index = 0
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        if not self.values:
            raise ValueError("values list cannot be empty")
        current_value = self.values[self.index]
        self.index = (self.index + 1) % len(self.values)

        # Get a string representation of the current test case
        if isinstance(current_value, str):
            test_case_name = current_value
        elif isinstance(current_value, dict):
            test_case_name = "dict"
        elif isinstance(current_value, list):
            test_case_name = "input_specs"
        else:
            test_case_name = str(type(current_value).__name__)

        print(f"Transmitting test case: {test_case_name}", file=sys.stderr)

        if current_value == "numpy-tensormap":
            tensormap = dict(
                r=np.arange(10000, dtype=np.int16),
                z=np.zeros((10, 5), dtype=float),
            )
            op_output.emit(tensormap, "out")
        elif current_value == "cupy-tensormap":
            tensormap = dict(
                r=cp.arange(10000, dtype=cp.int16),
                z=cp.zeros((10, 5), dtype=float),
            )
            op_output.emit(tensormap, "out")
        elif current_value == "numpy":
            z = np.zeros((16, 8, 4), dtype=np.float32)
            op_output.emit(z, "out")
        elif current_value == "cupy":
            z = cp.zeros((16, 8, 4), dtype=cp.float32)
            op_output.emit(z, "out")
        elif current_value == "cupy-as-holoscan-tensor":
            z = cp.zeros((16, 8, 4), dtype=cp.float32)
            op_output.emit(z, "out", emitter_name="holoscan::Tensor")
        elif current_value == "torch":
            z = torch.zeros((16, 8, 4), dtype=torch.float32, device="cuda")
            op_output.emit(z, "out")
        elif current_value == "torch-as-holoscan-tensor":
            z = torch.zeros((16, 8, 4), dtype=torch.float32, device="cuda")
            op_output.emit(z, "out", emitter_name="holoscan::Tensor")
        elif current_value == "cupy-complex":
            tensormap = dict(z=cp.ones((16, 8, 4), dtype=cp.complex64))
            op_output.emit(tensormap, "out")
        else:
            op_output.emit(current_value, "out")


def _check_value(value, expected_value):
    # checks the specific expected values as set in PingMessageTxOp.compute
    if expected_value == "numpy-tensormap":
        assert isinstance(value, dict)
        r = np.asarray(value["r"])
        z = np.asarray(value["z"])
        assert r.shape == (10000,)
        assert r.max() == 9999
        assert z.shape == (10, 5)
        assert z.max() == 0.0
    elif expected_value == "cupy-tensormap":
        assert isinstance(value, dict)
        r = cp.asarray(value["r"])
        z = cp.asarray(value["z"])
        assert r.shape == (10000,)
        assert int(r.max()) == 9999
        assert z.shape == (10, 5)
        assert float(z.max()) == 0.0
    elif expected_value == "numpy":
        # object with __array_attribute__ is deserialized as a NumPy array
        assert isinstance(value, np.ndarray)
        assert value.shape == (16, 8, 4)
    elif expected_value == "cupy" or expected_value == "cupy-as-holoscan-tensor":
        # object with __cuda_array_attribute__ is deserialized as a CuPy array
        assert isinstance(value, cp.ndarray)
        assert value.shape == (16, 8, 4)
    elif expected_value == "torch" or expected_value == "torch-as-holoscan-tensor":
        # PyTorch tensors are preserved as torch.Tensor on receive
        assert isinstance(value, torch.Tensor)
        assert value.shape == (16, 8, 4)
    elif expected_value == "cupy-complex":
        # object with __cuda_array_attribute__ is deserialized as a CuPy array
        assert isinstance(value, dict)
        z = cp.asarray(value["z"])
        assert isinstance(z, cp.ndarray)
        assert z.dtype == cp.complex64
        assert z.shape == (16, 8, 4)
    elif isinstance(expected_value, list) and isinstance(expected_value[0], HolovizOp.InputSpec):
        assert isinstance(value, list)
        assert len(value) == 3
        assert all(isinstance(v, HolovizOp.InputSpec) for v in value)
        assert value[0].type == test_text_spec.type
        assert value[0].color == test_text_spec.color
        assert value[1].type == HolovizOp.InputType.TRIANGLES
        assert value[2].type == test_full_spec.type
        assert value[2].color == test_full_spec.color
        assert value[2].opacity == test_full_spec.opacity
        assert value[2].priority == test_full_spec.priority
        assert value[2].image_format == test_full_spec.image_format
        assert value[2].line_width == test_full_spec.line_width
        assert value[2].point_size == test_full_spec.point_size
        assert value[2].text == test_full_spec.text
        assert value[2].yuv_model_conversion == test_full_spec.yuv_model_conversion
        assert value[2].yuv_range == test_full_spec.yuv_range
        assert value[2].x_chroma_location == test_full_spec.x_chroma_location
        assert value[2].y_chroma_location == test_full_spec.y_chroma_location
        assert value[2].depth_map_render_mode == test_full_spec.depth_map_render_mode
        assert all(isinstance(v, HolovizOp.InputSpec.View) for v in value[2].views)
        assert value[2].views[0].offset_x == test_view.offset_x
        assert value[2].views[0].offset_y == test_view.offset_y
        assert value[2].views[0].width == test_view.width
        assert value[2].views[0].height == test_view.height
        assert value[2].views[0].matrix == test_view.matrix


class PingMessageRxOp(Operator):
    """Simple receiver operator.

    This operator has a single input port:
        input: "in"

    This is an example of a native operator with one input port.
    On each tick, it receives an integer from the "in" port.
    If `expected_value` is a list, it cycles through the values on each compute call.
    """

    def __init__(self, fragment, *args, expected_value=1, **kwargs):
        # Need to call the base class constructor last
        self.expected_value = expected_value
        self.expected_values = (
            expected_value if isinstance(expected_value, list) else [expected_value]
        )
        self.index = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        value = op_input.receive("in")
        if not self.expected_values:
            raise ValueError("expected_values list cannot be empty")
        current_expected = self.expected_values[self.index]
        self.index = (self.index + 1) % len(self.expected_values)

        # Get a string representation of the current test case
        if isinstance(current_expected, str):
            test_case_name = current_expected
        elif isinstance(current_expected, dict):
            test_case_name = "dict"
        elif isinstance(current_expected, list):
            test_case_name = "input_specs"
        else:
            test_case_name = str(type(current_expected).__name__)

        try:
            _check_value(value, expected_value=current_expected)
            print(f"received expected value for test case: {test_case_name}", file=sys.stderr)
        except AssertionError as e:
            print(f"FAILED test case: {test_case_name}", file=sys.stderr)
            raise AssertionError(f"Test case '{test_case_name}' failed: {e}") from e


class TxFragment(Fragment):
    def __init__(self, *args, value=1, **kwargs):
        self.value = value
        super().__init__(*args, **kwargs)

    def compose(self):
        count = len(self.value) if isinstance(self.value, list) else 1
        tx = PingMessageTxOp(self, CountCondition(self, count), name="tx", value=self.value)
        self.add_operator(tx)


class RxFragment(Fragment):
    def __init__(self, *args, expected_value=1, **kwargs):
        self.expected_value = expected_value
        super().__init__(*args, **kwargs)

    def compose(self):
        count = len(self.expected_value) if isinstance(self.expected_value, list) else 1
        rx = PingMessageRxOp(
            self, CountCondition(self, count), name="rx", expected_value=self.expected_value
        )
        self.add_operator(rx)


class MultiFragmentPyObjectPingApp(Application):
    def __init__(self, *args, value=1, **kwargs):
        self.value = value
        super().__init__(*args, **kwargs)

    def compose(self):
        tx_fragment = TxFragment(self, name="tx_fragment", value=self.value)
        rx_fragment = RxFragment(self, name="rx_fragment", expected_value=self.value)

        # Connect the two fragments (tx.out -> rx.in)
        # We can skip the "out" and "in" suffixes, as they are the default
        self.add_flow(tx_fragment, rx_fragment, {("tx", "rx")})


class SingleFragmentDataPingApp(Application):
    def __init__(self, *args, value=1, expected_value=1, **kwargs):
        self.expected_value = expected_value
        self.value = value
        super().__init__(*args, **kwargs)

    def compose(self):
        tx = PingMessageTxOp(self, CountCondition(self, 1), name="tx", value=self.value)
        rx = PingMessageRxOp(
            self, CountCondition(self, 1), name="rx", expected_value=self.expected_value
        )
        self.add_flow(tx, rx)


@pytest.mark.parametrize(
    "value",
    [
        3.5,
        dict(a=5, b=7, c=[1, 2, 3], d="abc"),
        "numpy-tensormap",  # dict of numpy arrays
        "cupy-tensormap",  # dict of cupy arrays
        "numpy",  # single numpy array
        "cupy",  # single cupy array
        "cupy-as-holoscan-tensor",  # single cupy array as holoscan::Tensor
        "cupy-complex",  # single complex-valued cupy array
        "torch",  # single PyTorch tensor
        "torch-as-holoscan-tensor",  # single PyTorch tensor as holoscan::Tensor
        "input_specs",  # list of HolovizOp.InputSpec
    ],
)
def test_single_fragment_data_ping_app(value, capfd):
    """Testing UCX-based serialization of PyObject, tensors, etc."""
    if value in ["numpy", "numpy-tensormap"]:
        pytest.importorskip("numpy")
    elif value in ["cupy", "cupy-as-holoscan-tensor", "cupy-complex", "cupy-tensormap"]:
        pytest.importorskip("cupy")
    elif value in ["torch", "torch-as-holoscan-tensor"]:
        if not is_torch_cuda_compatible():
            pytest.skip("Torch CUDA unavailable or SM incompatible.")
    elif value == "input_specs":
        value = test_input_specs

    app = SingleFragmentDataPingApp(value=value)
    app.run()

    # assert that no errors were logged
    captured = capfd.readouterr()
    assert "received expected value" in captured.err
    assert "error" not in captured.err
    assert "Exception occurred" not in captured.err


def test_ucx_object_serialization_app(capfd):
    """Testing UCX-based serialization of PyObject, tensors, etc. with all types in one run.

    This test cycles through all data types in a single application run to reduce overhead
    from network connection setup and teardown.
    """
    pytest.importorskip("numpy")
    pytest.importorskip("cupy")

    # List of all test values to cycle through
    test_values = [
        3.5,
        dict(a=5, b=7, c=[1, 2, 3], d="abc"),
        "numpy-tensormap",  # dict of numpy arrays
        "cupy-tensormap",  # dict of cupy arrays
        "numpy",  # single numpy array
        "cupy",  # single cupy array
        "cupy-as-holoscan-tensor",  # single cupy array as holoscan::Tensor
        "cupy-complex",  # single complex-valued cupy array
        test_input_specs,  # list of HolovizOp.InputSpec
    ]

    # Optionally add PyTorch tensor tests if torch is available with compatible CUDA
    if is_torch_cuda_compatible():
        test_values.extend(["torch", "torch-as-holoscan-tensor"])

    app = MultiFragmentPyObjectPingApp(value=test_values)
    app.run()

    # assert that no errors were logged
    captured = capfd.readouterr()
    # avoid catching the expected error message
    # : "error handling callback was invoked with status -25 (Connection reset by remote peer)"
    captured_error = captured.err.replace("error handling callback", "ucx handling callback")

    # Check that we received the expected value for each test case
    expected_count = len(test_values)
    actual_count = captured_error.count("received expected value for test case:")

    # If the count doesn't match, print detailed information about which cases passed
    if actual_count != expected_count:
        print("\n=== Test case results ===", file=sys.stderr)
        for line in captured_error.split("\n"):
            if "test case:" in line:
                print(line, file=sys.stderr)
        print(f"Expected {expected_count} test cases, got {actual_count}", file=sys.stderr)

    assert actual_count == expected_count, (
        f"Expected {expected_count} successful receives, got {actual_count}. "
        f"Check stderr for details on which test cases passed/failed."
    )

    assert "error" not in remove_ignored_errors(captured_error)
    assert "Exception occurred" not in captured_error


class PingMessageReceiversRxOp(Operator):
    """Simple receiver operator.

    This operator has a single input port:
        input: "in"

    This is an example of a native operator with one input port.
    On each tick, it receives an integer from the "in" port.
    If `expected_value` is a list, it cycles through the values on each compute call.
    """

    def __init__(self, fragment, *args, expected_value=1, **kwargs):
        # Need to call the base class constructor last
        self.expected_value = expected_value
        self.expected_values = (
            expected_value if isinstance(expected_value, list) else [expected_value]
        )
        self.index = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("receivers", size=IOSpec.ANY_SIZE)

    def compute(self, op_input, op_output, context):
        values = op_input.receive("receivers")
        assert values is not None
        assert len(values) == 1
        current_expected = self.expected_values[self.index]
        self.index = (self.index + 1) % len(self.expected_values)

        # Get a string representation of the current test case
        if isinstance(current_expected, str):
            test_case_name = current_expected
        elif isinstance(current_expected, dict):
            test_case_name = "dict"
        elif isinstance(current_expected, list):
            test_case_name = "input_specs"
        else:
            test_case_name = str(type(current_expected).__name__)

        try:
            _check_value(values[0], expected_value=current_expected)
            print(f"received expected value for test case: {test_case_name}", file=sys.stderr)
        except AssertionError as e:
            print(f"FAILED test case: {test_case_name}", file=sys.stderr)
            raise AssertionError(f"Test case '{test_case_name}' failed: {e}") from e


class RxReceiversFragment(Fragment):
    def __init__(self, *args, expected_value=1, **kwargs):
        self.expected_value = expected_value
        super().__init__(*args, **kwargs)

    def compose(self):
        count = len(self.expected_value) if isinstance(self.expected_value, list) else 1
        rx = PingMessageReceiversRxOp(
            self, CountCondition(self, count), name="rx", expected_value=self.expected_value
        )
        self.add_operator(rx)


class MultiFragmentPyObjectReceiversPingApp(Application):
    def __init__(self, *args, value=1, **kwargs):
        self.value = value
        super().__init__(*args, **kwargs)

    def compose(self):
        tx_fragment = TxFragment(self, name="tx_fragment", value=self.value)
        rx_fragment = RxReceiversFragment(self, name="rx_fragment", expected_value=self.value)

        # Connect the two fragments (tx.out -> rx.in)
        self.add_flow(tx_fragment, rx_fragment, {("tx.out", "rx.receivers")})


def test_ucx_object_receivers_serialization_app(capfd):
    """Testing UCX-based serialization of PyObject, tensors, etc. with all types in one run.

    This test cycles through all data types in a single application run to reduce overhead
    from network connection setup and teardown.
    """
    pytest.importorskip("numpy")
    pytest.importorskip("cupy")

    # List of all test values to cycle through
    test_values = [
        3.5,
        dict(a=5, b=7, c=[1, 2, 3], d="abc"),
        "numpy-tensormap",  # dict of numpy arrays
        "cupy-tensormap",  # dict of cupy arrays
        "numpy",  # single numpy array
        "cupy",  # single cupy array
        "cupy-as-holoscan-tensor",  # single cupy array as holoscan::Tensor
        test_input_specs,  # list of HolovizOp.InputSpec
    ]

    # Optionally add PyTorch tensor tests if torch is available with compatible CUDA
    if is_torch_cuda_compatible():
        test_values.extend(["torch", "torch-as-holoscan-tensor"])

    app = MultiFragmentPyObjectReceiversPingApp(value=test_values)
    app.run()

    # assert that no errors were logged
    captured = capfd.readouterr()
    # avoid catching the expected error message
    # : "error handling callback was invoked with status -25 (Connection reset by remote peer)"
    captured_error = captured.err.replace("error handling callback", "ucx handling callback")

    # Check that we received the expected value for each test case
    expected_count = len(test_values)
    actual_count = captured_error.count("received expected value for test case:")

    # If the count doesn't match, print detailed information about which cases passed
    if actual_count != expected_count:
        print("\n=== Test case results ===", file=sys.stderr)
        for line in captured_error.split("\n"):
            if "test case:" in line:
                print(line, file=sys.stderr)
        print(f"Expected {expected_count} test cases, got {actual_count}", file=sys.stderr)

    assert actual_count == expected_count, (
        f"Expected {expected_count} successful receives, got {actual_count}. "
        f"Check stderr for details on which test cases passed/failed."
    )

    assert "error" not in remove_ignored_errors(captured_error)
    assert "Exception occurred" not in captured_error
