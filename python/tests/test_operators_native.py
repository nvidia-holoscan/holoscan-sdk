# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from holoscan.conditions import CountCondition
from holoscan.core import Arg, Operator, OperatorSpec, Tensor
from holoscan.gxf import GXFExtensionRegistrar
from holoscan.resources import UnboundedAllocator

try:
    import numpy as np

    unsigned_dtypes = [np.uint8, np.uint16, np.uint32, np.uint64]
    signed_dtypes = [np.int8, np.int16, np.int32, np.int64]
    float_dtypes = [np.float16, np.float32, np.float64]
    complex_dtypes = [np.complex64, np.complex128]

except ImportError:
    unsigned_dtypes = signed_dtypes = float_dtypes = []


class TestOperator:
    def test_default_init(self, fragment, capfd):
        op = Operator()
        assert op.name == ""
        assert op.fragment is None
        assert op.id == -1
        assert op.operator_type == Operator.OperatorType.NATIVE
        capfd.readouterr()

    def test_operator_type(self):
        assert hasattr(Operator, "OperatorType")
        assert hasattr(Operator.OperatorType, "NATIVE")
        assert hasattr(Operator.OperatorType, "GXF")

    @pytest.mark.parametrize("with_fragment", [False, True, "as_kwarg"])
    @pytest.mark.parametrize("with_name", [False, True])
    @pytest.mark.parametrize("with_condition", [False, True, "as_kwarg"])
    @pytest.mark.parametrize("with_resource", [False, True, "as_kwarg"])
    def test_init(self, app, with_fragment, with_name, with_condition, with_resource, capfd):
        kwargs = dict(a=5, b=(13.7, 15.2), c="abcd")
        args = ()
        if with_name:
            kwargs["name"] = "my op"
        if with_fragment == "as_kwarg":
            kwargs["fragment"] = app
        elif with_fragment:
            args = (app,)
        else:
            args = ()
        if with_condition:
            if with_condition == "as_kwarg":
                kwargs["count"] = CountCondition(app, count=15)
            else:
                args += (CountCondition(app, count=15, name="count"),)
        if with_resource:
            if with_condition == "as_kwarg":
                kwargs["pool"] = UnboundedAllocator(app)
            else:
                args += (UnboundedAllocator(app, name="pool"),)
        op = Operator(*args, **kwargs)

        # check operator name
        expected_name = "my op" if with_name else ""
        assert op.name == expected_name

        # check operator fragment
        if with_fragment:
            assert op.fragment is app
        else:
            assert op.fragment is None

        # check all args that were not of Condition or Resource type
        assert len(op.args) == 3
        assert [arg.name for arg in op.args] == ["a", "b", "c"]

        # check conditions
        if with_condition:
            assert len(op.conditions) == 1
            assert "count" in op.conditions
            assert isinstance(op.conditions["count"], CountCondition)
        else:
            assert len(op.conditions) == 0

        # check resources
        if with_resource:
            assert len(op.resources) == 1
            assert "pool" in op.resources
            assert isinstance(op.resources["pool"], UnboundedAllocator)
        else:
            assert len(op.resources) == 0
        capfd.readouterr()

    def test_error_on_multiple_fragments(self, app, capfd):
        with pytest.raises(RuntimeError):
            Operator(app, app)
        with pytest.raises(RuntimeError):
            Operator(app, fragment=app)
        capfd.readouterr()

    def test_name(self, capfd):
        op = Operator()
        op.name = "op1"
        assert op.name == "op1"

        op = Operator(name="op3")
        assert op.name == "op3"
        capfd.readouterr()

    def test_fragment(self, fragment, capfd):
        op = Operator()
        assert op.fragment is None
        op.fragment = fragment
        assert op.fragment is fragment

    def test_add_arg(self, capfd):
        op = Operator()
        op.add_arg(Arg("a1"))
        capfd.readouterr()

    def test_initialize(self, app, config_file, capfd):
        spec = OperatorSpec(app)

        op = Operator()
        # Operator.__init__ will have added op.spec for us
        assert isinstance(op.spec, OperatorSpec)

        app.config(config_file)

        # initialize context
        context = app.executor.context
        assert context is not None

        # follow order of operations in Fragment::make_operator
        op.name = "my operator"

        #  initialize() will segfault op.fragment is not assigned?
        op.fragment = app

        op.setup(spec)
        op.spec = spec
        assert isinstance(op.spec, OperatorSpec)

        # Have to first register the GXFWrapper type before we can call
        # op.initialize() on an operator of type kNative.
        registrar = GXFExtensionRegistrar(context, "my_registrar")
        # add_component is currently templated only for
        # <holoscan::gxf::GXFWrapper, nvidia::gxf::Codelet>
        registrar.add_component("GXF wrapper to support Holoscan SDK native operators")
        # add_type is currently templated only for <holoscan::Message>
        registrar.add_type("Holoscan message type")
        # will be True if extension was registered successfully
        assert registrar.register_extension()

        op.initialize()
        op.id != -1
        op.operator_type == Operator.OperatorType.NATIVE
        capfd.readouterr()

    def test_operator_setup_and_assignment(self, fragment, capfd):
        spec = OperatorSpec(fragment)
        op = Operator(fragment)
        op.setup(spec)
        op.spec = spec
        capfd.readouterr()

    def test_dynamic_attribute_allowed(self, capfd):
        obj = Operator()
        obj.custom_attribute = 5
        capfd.readouterr()


class TestTensor:
    def _check_dlpack_attributes(self, t):
        assert hasattr(t, "__dlpack__")
        type(t.__dlpack__()).__name__ == "PyCapsule"

        assert hasattr(t, "__dlpack_device__")
        dev = t.__dlpack_device__()
        assert isinstance(dev, tuple) and len(dev) == 2

    def _check_array_interface_attribute(self, t, arr, cuda=False):
        if cuda:
            assert hasattr(t, "__cuda_array_interface__")
            interface = t.__cuda_array_interface__
            reference_interface = arr.__cuda_array_interface__
        else:
            assert hasattr(t, "__array_interface__")
            interface = t.__array_interface__
            reference_interface = arr.__array_interface__

        assert interface["version"] == 3

        assert interface["typestr"] == arr.dtype.str
        assert interface["shape"] == arr.shape
        assert len(interface["data"]) == 2
        if cuda:
            assert interface["data"][0] == arr.data.mem.ptr
            # no writeable flag present on CuPy arrays
        else:
            assert interface["data"][0] == arr.ctypes.data
            assert interface["data"][1] == (not arr.flags.writeable)
        if interface["strides"] is None:
            assert arr.flags.c_contiguous
        else:
            assert interface["strides"] == arr.strides
        assert interface["descr"] == [("", arr.dtype.str)]

        if reference_interface["version"] == interface["version"]:
            interface["shape"] == reference_interface["shape"]
            interface["typestr"] == reference_interface["typestr"]
            interface["descr"] == reference_interface["descr"]
            interface["data"] == reference_interface["data"]
            if reference_interface["strides"] is not None:
                interface["strides"] == reference_interface["strides"]

    def _check_tensor_property_values(self, t, arr, cuda=False):
        assert t.size == arr.size
        assert t.nbytes == arr.nbytes
        assert t.ndim == arr.ndim
        assert t.itemsize == arr.dtype.itemsize

        assert t.shape == arr.shape
        assert t.strides == arr.strides

        type(t.data).__name__ == "PyCapsule"

    @pytest.mark.parametrize(
        "dtype", unsigned_dtypes + signed_dtypes + float_dtypes + complex_dtypes
    )
    @pytest.mark.parametrize("order", ["F", "C"])
    def test_numpy_as_tensor(self, dtype, order):
        np = pytest.importorskip("numpy")
        a = np.zeros((4, 8, 12), dtype=dtype, order=order)
        t = Tensor.as_tensor(a)
        assert isinstance(t, Tensor)

        self._check_dlpack_attributes(t)
        self._check_array_interface_attribute(t, a, cuda=False)
        self._check_tensor_property_values(t, a)

    @pytest.mark.parametrize(
        "dtype", unsigned_dtypes + signed_dtypes + float_dtypes + complex_dtypes
    )
    @pytest.mark.parametrize("order", ["F", "C"])
    def test_cupy_as_tensor(self, dtype, order):
        cp = pytest.importorskip("cupy")
        a = cp.zeros((4, 8, 12), dtype=dtype, order=order)
        t = Tensor.as_tensor(a)
        assert isinstance(t, Tensor)

        self._check_dlpack_attributes(t)
        self._check_array_interface_attribute(t, a, cuda=True)
        self._check_tensor_property_values(t, a)

    def test_tensor_properties_are_readonly(self):
        np = pytest.importorskip("numpy")
        a = np.zeros((4, 8, 12), dtype=np.uint8)
        t = Tensor.as_tensor(a)
        with pytest.raises(AttributeError):
            t.size = 8
        with pytest.raises(AttributeError):
            t.nbytes = 8
        with pytest.raises(AttributeError):
            t.ndim = 2
        with pytest.raises(AttributeError):
            t.itemsize = 3
        with pytest.raises(AttributeError):
            t.shape = (t.size,)
        with pytest.raises(AttributeError):
            t.strides = (8,)
        with pytest.raises(AttributeError):
            t.data = 0

    @pytest.mark.parametrize(
        "dtype", unsigned_dtypes + signed_dtypes + float_dtypes + complex_dtypes
    )
    @pytest.mark.parametrize("order", ["F", "C"])
    @pytest.mark.parametrize("module", ["cupy", "numpy"])
    def test_tensor_round_trip(self, dtype, order, module):
        xp = pytest.importorskip(module)
        a = xp.zeros((4, 8, 12), dtype=dtype, order=order)
        t = Tensor.as_tensor(a)
        b = xp.asarray(t)
        xp.testing.assert_array_equal(a, b)


class MinimalOp(Operator):
    def __init__(self, *args, **kwargs):
        self.count = 1
        self.param_value = None
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.param("param_value", 500)

    def compute(self, op_input, op_output, context):
        self.count += 1


class ValueData:
    def __init__(self, value):
        self.data = value

    def __repr__(self):
        return f"ValueData({self.data})"

    def __eq__(self, other):
        return self.data == other.data

    def __hash__(self):
        return hash(self.data)


class PingTxOp(Operator):
    def __init__(self, *args, **kwargs):
        self.index = 0
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out1")
        spec.output("out2")

    def compute(self, op_input, op_output, context):
        value1 = ValueData(self.index)
        self.index += 1
        op_output.emit(value1, "out1")

        value2 = ValueData(self.index)
        self.index += 1
        op_output.emit(value2, "out2")


class PingMiddleOp(Operator):
    def __init__(self, *args, **kwargs):
        # If `self.multiplier` is set here (e.g., `self.multiplier = 4`), then
        # the default value by `param()` in `setup()` will be ignored.
        # (you can just call `spec.param("multiplier")` in `setup()` to use the
        # default value)
        #
        # self.multiplier = 4
        self.count = 1

        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in1")
        spec.input("in2")
        spec.output("out1")
        spec.output("out2")
        spec.param("multiplier", 2)

    def compute(self, op_input, op_output, context):
        value1 = op_input.receive("in1")
        value2 = op_input.receive("in2")
        self.count += 1

        # Multiply the values by the multiplier parameter
        value1.data *= self.multiplier
        value2.data *= self.multiplier

        op_output.emit(value1, "out1")
        op_output.emit(value2, "out2")


class PingRxOp(Operator):
    def __init__(self, *args, **kwargs):
        self.count = 1
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.param("receivers", kind="receivers")

    def compute(self, op_input, op_output, context):
        values = op_input.receive("receivers")
        assert values is not None
        self.count += 1
