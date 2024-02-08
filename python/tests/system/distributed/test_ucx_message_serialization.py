"""
 SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import contextlib
import sys

import pytest

from holoscan.conditions import CountCondition
from holoscan.core import Application, Fragment, Operator, OperatorSpec
from holoscan.operators.holoviz import HolovizOp

with contextlib.suppress(ImportError):
    import cupy as cp
    import numpy as np


class PingMessageTxOp(Operator):
    """Simple transmitter operator.

    This operator has a single output port:
        output: "out"

    On each tick, it transmits the object passed via `value`.
    """

    def __init__(self, fragment, *args, value=1, **kwargs):
        self.value = value
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        if self.value == "numpy-tensormap":
            tensormap = dict(
                r=np.arange(10000, dtype=np.int16),
                z=np.zeros((10, 5), dtype=float),
            )
            op_output.emit(tensormap, "out")
        elif self.value == "cupy-tensormap":
            tensormap = dict(
                r=cp.arange(10000, dtype=cp.int16),
                z=cp.zeros((10, 5), dtype=float),
            )
            op_output.emit(tensormap, "out")
        elif self.value == "numpy":
            z = np.zeros((16, 8, 4), dtype=np.float32)
            op_output.emit(z, "out")
        elif self.value == "cupy":
            z = cp.zeros((16, 8, 4), dtype=cp.float32)
            op_output.emit(z, "out")
        elif self.value == "cupy-complex":
            tensormap = dict(z=cp.ones((16, 8, 4), dtype=cp.complex64))
            op_output.emit(tensormap, "out")
        else:
            op_output.emit(self.value, "out")


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
    elif expected_value == "cupy":
        # object with __cuda_array_attribute__ is deserialized as a CuPy array
        assert isinstance(value, cp.ndarray)
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
        assert len(value) == 2
        assert all(isinstance(v, HolovizOp.InputSpec) for v in value)
        assert value[0].type == HolovizOp.InputType.TEXT
        assert value[1].type == HolovizOp.InputType.TRIANGLES


class PingMessageRxOp(Operator):
    """Simple receiver operator.

    This operator has a single input port:
        input: "in"

    This is an example of a native operator with one input port.
    On each tick, it receives an integer from the "in" port.
    """

    def __init__(self, fragment, *args, expected_value=1, **kwargs):
        # Need to call the base class constructor last
        self.expected_value = expected_value
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        value = op_input.receive("in")
        _check_value(value, expected_value=self.expected_value)
        print("received expected value", file=sys.stderr)


class TxFragment(Fragment):
    def __init__(self, *args, value=1, period=None, **kwargs):
        self.value = value
        super().__init__(*args, **kwargs)

    def compose(self):
        tx = PingMessageTxOp(self, CountCondition(self, 1), name="tx", value=self.value)
        self.add_operator(tx)


class RxFragment(Fragment):
    def __init__(self, *args, expected_value=1, period=None, **kwargs):
        self.expected_value = expected_value
        super().__init__(*args, **kwargs)

    def compose(self):
        rx = PingMessageRxOp(
            self, CountCondition(self, 1), name="rx", expected_value=self.expected_value
        )
        self.add_operator(rx)


class MultiFragmentPyObjectPingApp(Application):
    def __init__(self, *args, value=1, period=None, **kwargs):
        self.value = value
        super().__init__(*args, **kwargs)

    def compose(self):
        tx_fragment = TxFragment(self, name="tx_fragment", value=self.value)
        rx_fragment = RxFragment(self, name="rx_fragment", expected_value=self.value)

        # Connect the two fragments (tx.out -> rx.in)
        # We can skip the "out" and "in" suffixes, as they are the default
        self.add_flow(tx_fragment, rx_fragment, {("tx", "rx")})


@pytest.mark.parametrize(
    "value",
    [
        3.5,
        dict(a=5, b=7, c=[1, 2, 3], d="abc"),
        "numpy-tensormap",  # dict of numpy arrays
        "cupy-tensormap",  # dict of cupy arrays
        "numpy",  # single numpy array
        "cupy",  # single cupy array
        "cupy-complex",  # single complex-valued cupy array
        "input_specs",  # list of HolovizOp.InputSpec
    ],
)
def test_ucx_object_serialization_app(ping_config_file, value, capfd):
    """Testing UCX-based serialization of PyObject, tensors, etc."""
    if value in ["numpy", "numpy-tensormap"]:
        pytest.importorskip("numpy")
    elif value in ["cupy", "cupy-complex", "cupy-tensormap"]:
        pytest.importorskip("cupy")
    elif value == "input_specs":
        specs = []
        text_spec = HolovizOp.InputSpec("dynamic_text", "text")
        text_spec.text = ["Text1"]
        text_spec.color = [1.0, 0.0, 0.0, 1.0]
        specs.append(text_spec)
        specs.append(HolovizOp.InputSpec("triangles", HolovizOp.InputType.TRIANGLES))
        value = specs

    app = MultiFragmentPyObjectPingApp(value=value)
    app.run()

    # assert that no errors were logged
    captured = capfd.readouterr()
    # avoid catching the expected error message
    # : "error handling callback was invoked with status -25 (Connection reset by remote peer)"
    captured_error = captured.err.replace("error handling callback", "ucx handling callback")
    assert "received expected value" in captured_error
    assert "error" not in captured_error
    assert "Exception occurred" not in captured_error


class PingMessageReceiversRxOp(Operator):
    """Simple receiver operator.

    This operator has a single input port:
        input: "in"

    This is an example of a native operator with one input port.
    On each tick, it receives an integer from the "in" port.
    """

    def __init__(self, fragment, *args, expected_value=1, **kwargs):
        # Need to call the base class constructor last
        self.expected_value = expected_value
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.param("receivers", kind="receivers")

    def compute(self, op_input, op_output, context):
        values = op_input.receive("receivers")
        assert values is not None
        assert len(values) == 1
        _check_value(values[0], expected_value=self.expected_value)
        print("received expected value", file=sys.stderr)


class RxReceiversFragment(Fragment):
    def __init__(self, *args, expected_value=1, period=None, **kwargs):
        self.expected_value = expected_value
        super().__init__(*args, **kwargs)

    def compose(self):
        rx = PingMessageReceiversRxOp(
            self, CountCondition(self, 1), name="rx", expected_value=self.expected_value
        )
        self.add_operator(rx)


class MultiFragmentPyObjectReceiversPingApp(Application):
    def __init__(self, *args, value=1, period=None, **kwargs):
        self.value = value
        super().__init__(*args, **kwargs)

    def compose(self):
        tx_fragment = TxFragment(self, name="tx_fragment", value=self.value)
        rx_fragment = RxReceiversFragment(self, name="rx_fragment", expected_value=self.value)

        # Connect the two fragments (tx.out -> rx.in)
        self.add_flow(tx_fragment, rx_fragment, {("tx.out", "rx.receivers")})


@pytest.mark.parametrize(
    "value",
    [
        3.5,
        dict(a=5, b=7, c=[1, 2, 3], d="abc"),
        "numpy-tensormap",  # dict of numpy arrays
        "cupy-tensormap",  # dict of cupy arrays
        "numpy",  # single numpy array
        "cupy",  # single cupy array
        "input_specs",  # list of HolovizOp.InputSpec
    ],
)
def test_ucx_object_receivers_serialization_app(ping_config_file, value, capfd):
    """Testing UCX-based serialization of PyObject, tensors, etc."""
    if value == "numpy":
        pytest.importorskip("numpy")
    elif value == "cupy":
        pytest.importorskip("cupy")
    elif value == "input_specs":
        specs = []
        text_spec = HolovizOp.InputSpec("dynamic_text", "text")
        text_spec.text = ["Text1"]
        text_spec.color = [1.0, 0.0, 0.0, 1.0]
        specs.append(text_spec)
        specs.append(HolovizOp.InputSpec("triangles", HolovizOp.InputType.TRIANGLES))
        value = specs

    app = MultiFragmentPyObjectReceiversPingApp(value=value)
    app.run()

    # assert that no errors were logged
    captured = capfd.readouterr()
    # avoid catching the expected error message
    # : "error handling callback was invoked with status -25 (Connection reset by remote peer)"
    captured_error = captured.err.replace("error handling callback", "ucx handling callback")
    assert "received expected value" in captured_error
    assert "error" not in captured_error
    assert "Exception occurred" not in captured_error
