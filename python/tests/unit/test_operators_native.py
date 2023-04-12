# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os

import pytest

from holoscan.conditions import CountCondition
from holoscan.core import Application, Arg, Operator, OperatorSpec, Tensor, _Operator
from holoscan.gxf import Entity
from holoscan.operators import (
    AJASourceOp,
    FormatConverterOp,
    HolovizOp,
    MultiAIInferenceOp,
    MultiAIPostprocessorOp,
    NTV2Channel,
    SegmentationPostprocessorOp,
    VideoStreamRecorderOp,
    VideoStreamReplayerOp,
    _holoviz_str_to_depth_map_render_mode,
    _holoviz_str_to_input_type,
)
from holoscan.resources import BlockMemoryPool, MemoryStorageType, UnboundedAllocator

try:
    import numpy as np

    unsigned_dtypes = [np.uint8, np.uint16, np.uint32, np.uint64]
    signed_dtypes = [np.int8, np.int16, np.int32, np.int64]
    float_dtypes = [np.float16, np.float32, np.float64]
    complex_dtypes = [np.complex64, np.complex128]

except ImportError:
    unsigned_dtypes = signed_dtypes = float_dtypes = []

sample_data_path = os.environ.get("HOLOSCAN_SAMPLE_DATA_PATH", "../data")


class TestOperator:
    def test_default_init(self):
        with pytest.raises(TypeError):
            Operator()

    def test_invalid_init(self):
        with pytest.raises(ValueError):
            Operator(5)

    def test_invalid_init2(self, fragment):
        # C++ PyOperator class will throw std::runtime_error if >1 fragment provided
        with pytest.raises(RuntimeError):
            Operator(fragment, fragment)

    def test_basic_init(self, fragment, capfd):
        op = Operator(fragment)
        assert op.name == ""
        assert op.fragment is fragment
        assert op.id != -1
        assert op.operator_type == Operator.OperatorType.NATIVE
        capfd.readouterr()

    def test_basic_kwarg_init(self, fragment, capfd):
        op = Operator(fragment=fragment)
        assert op.name == ""
        assert op.fragment is fragment
        assert op.id != -1
        assert op.operator_type == Operator.OperatorType.NATIVE
        capfd.readouterr()

    def test_operator_type(self):
        assert hasattr(Operator, "OperatorType")
        assert hasattr(Operator.OperatorType, "NATIVE")
        assert hasattr(Operator.OperatorType, "GXF")

    @pytest.mark.parametrize("with_name", [False, True])
    @pytest.mark.parametrize("with_condition", [False, True, "as_kwarg"])
    @pytest.mark.parametrize("with_resource", [False, True, "as_kwarg"])
    def test_init(self, app, with_name, with_condition, with_resource, capfd):
        kwargs = dict(a=5, b=(13.7, 15.2), c="abcd")
        args = ()
        if with_name:
            kwargs["name"] = "my op"
        args = (app,)
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
        assert op.fragment is app

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
        with pytest.raises(TypeError):
            Operator(app, fragment=app)
        capfd.readouterr()

    def test_name(self, fragment, capfd):
        op = Operator(fragment)
        op.name = "op1"
        assert op.name == "op1"

        op = Operator(fragment, name="op3")
        assert op.name == "op3"
        capfd.readouterr()

    def test_add_arg(self, fragment, capfd):
        op = Operator(fragment)
        op.add_arg(Arg("a1"))
        capfd.readouterr()

    def test_initialize(self, app, config_file, capfd):
        spec = OperatorSpec(app)

        op = Operator(app)
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

    def test_dynamic_attribute_allowed(self, fragment, capfd):
        obj = Operator(fragment)
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


# Test cases for specific bundled native operators


class TestAJASourceOp:
    def test_kwarg_based_initialization(self, app, config_file, capfd):
        app.config(config_file)
        op = AJASourceOp(
            fragment=app,
            name="source",
            channel=NTV2Channel.NTV2_CHANNEL1,
            **app.kwargs("aja"),
        )
        assert isinstance(op, _Operator)
        assert op.operator_type == Operator.OperatorType.NATIVE

        # assert no warnings or errors logged
        captured = capfd.readouterr()

        # Initializing outside the context of app.run() will result in the
        # following error being logged because the GXFWrapper will not have
        # been created for the operator:
        #     [error] [gxf_executor.cpp:452] Unable to get GXFWrapper for Operator 'segmentation_postprocessor  # noqa: E501
        assert captured.err.count("[error]") <= 1
        assert "warning" not in captured.err


class TestFormatConverterOp:
    def test_kwarg_based_initialization(self, app, config_file, capfd):
        app.config(config_file)
        op = FormatConverterOp(
            fragment=app,
            name="recorder_format_converter",
            pool=BlockMemoryPool(
                name="pool",
                fragment=app,
                storage_type=MemoryStorageType.DEVICE,
                block_size=16 * 1024**2,
                num_blocks=4,
            ),
            **app.kwargs("recorder_format_converter"),
        )
        assert isinstance(op, _Operator)
        len(op.args) == 12
        assert op.operator_type == Operator.OperatorType.NATIVE

        # assert no warnings or errors logged
        captured = capfd.readouterr()

        # Initializing outside the context of app.run() will result in the
        # following error being logged because the GXFWrapper will not have
        # been created for the operator:
        #     [error] [gxf_executor.cpp:452] Unable to get GXFWrapper for Operator 'recorder_format_converter'  # noqa: E501
        assert captured.err.count("[error]") <= 1
        assert "warning" not in captured.err


class TestMultiAIInferenceOp:
    def test_kwarg_based_initialization(self, app, config_file, capfd):
        app.config(config_file)
        model_path = os.path.join(sample_data_path, "multiai_ultrasound", "models")

        model_path_map = {
            "plax_chamber": os.path.join(model_path, "plax_chamber.onnx"),
            "aortic_stenosis": os.path.join(model_path, "aortic_stenosis.onnx"),
            "bmode_perspective": os.path.join(model_path, "bmode_perspective.onnx"),
        }

        op = MultiAIInferenceOp(
            app,
            name="multiai_inference",
            allocator=UnboundedAllocator(app, name="pool"),
            model_path_map=model_path_map,
            **app.kwargs("multiai_inference"),
        )
        assert isinstance(op, _Operator)
        assert op.id != -1
        assert op.operator_type == Operator.OperatorType.NATIVE

        # assert no warnings or errors logged
        captured = capfd.readouterr()

        # Initializing outside the context of app.run() will result in the
        # following error being logged because the GXFWrapper will not have
        # been created for the operator:
        #     [error] [gxf_executor.cpp:452] Unable to get GXFWrapper for Operator 'multi_ai_inference'  # noqa: E501
        assert captured.err.count("[error]") <= 1
        assert "warning" not in captured.err


class TestMultiAIPostprocessorOp:
    def test_kwarg_based_initialization(self, app, config_file, capfd):
        app.config(config_file)
        op = MultiAIPostprocessorOp(
            app,
            name="multiai_postprocessor",
            allocator=UnboundedAllocator(app, name="pool"),
            **app.kwargs("multiai_postprocessor"),
        )
        assert isinstance(op, _Operator)
        assert op.id != -1
        assert op.operator_type == Operator.OperatorType.NATIVE

        # assert no warnings or errors logged
        captured = capfd.readouterr()

        # Initializing outside the context of app.run() will result in the
        # following error being logged because the GXFWrapper will not have
        # been created for the operator:
        #     [error] [gxf_executor.cpp:452] Unable to get GXFWrapper for Operator 'multi_ai_inference'  # noqa: E501
        assert captured.err.count("[error]") <= 1
        assert "warning" not in captured.err


class TestSegmentationPostprocessorOp:
    def test_kwarg_based_initialization(self, app, capfd):
        op = SegmentationPostprocessorOp(
            fragment=app,
            allocator=UnboundedAllocator(fragment=app, name="allocator"),
            name="segmentation_postprocessor",
        )
        assert isinstance(op, _Operator)
        assert op.id != -1
        assert op.operator_type == Operator.OperatorType.NATIVE

        # assert no warnings or errors logged
        captured = capfd.readouterr()

        # Initializing outside the context of app.run() will result in the
        # following error being logged because the GXFWrapper will not have
        # been created for the operator:
        #     [error] [gxf_executor.cpp:452] Unable to get GXFWrapper for Operator 'segmentation_postprocessor'  # noqa: E501
        assert captured.err.count("[error]") <= 1
        assert "warning" not in captured.err


class TestVideoStreamRecorderOp:
    def test_kwarg_based_initialization(self, app, config_file, capfd):
        app.config(config_file)
        op = VideoStreamRecorderOp(name="recorder", fragment=app, **app.kwargs("recorder"))
        assert isinstance(op, _Operator)
        assert op.id != -1
        assert op.operator_type == Operator.OperatorType.NATIVE

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        # Initializing outside the context of app.run() will result in the
        # following error being logged because the GXFWrapper will not have
        # been created for the operator:
        #     [error] [gxf_executor.cpp:452] Unable to get GXFWrapper for Operator 'recorder'  # noqa: E501
        assert captured.err.count("[error]") <= 1
        assert "warning" not in captured.err


class TestVideoStreamReplayerOp:
    def test_kwarg_based_initialization(self, app, config_file, capfd):
        app.config(config_file)
        data_path = os.environ.get("HOLOSCAN_SAMPLE_DATA_PATH", "../data")
        op = VideoStreamReplayerOp(
            name="replayer",
            fragment=app,
            directory=os.path.join(data_path, "endoscopy", "video"),
            **app.kwargs("replayer"),
        )
        assert isinstance(op, _Operator)
        assert op.id != -1
        assert op.operator_type == Operator.OperatorType.NATIVE

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        # Initializing outside the context of app.run() will result in the
        # following error being logged because the GXFWrapper will not have
        # been created for the operator:
        #     [error] [gxf_executor.cpp:452] Unable to get GXFWrapper for Operator 'replayer'  # noqa: E501
        assert captured.err.count("[error]") <= 1
        assert "warning" not in captured.err


@pytest.mark.parametrize(
    "type_str",
    [
        "unknown",
        "color",
        "color_lut",
        "points",
        "lines",
        "line_strip",
        "triangles",
        "crosses",
        "rectangles",
        "ovals",
        "text",
        "depth_map",
        "depth_map_color",
    ],
)
def test_holoviz_input_types(type_str):
    assert isinstance(_holoviz_str_to_input_type[type_str], HolovizOp.InputType)


@pytest.mark.parametrize(
    "depth_type_str",
    [
        "points",
        "lines",
        "triangles",
    ],
)
def test_holoviz_depth_types(depth_type_str):
    assert isinstance(
        _holoviz_str_to_depth_map_render_mode[depth_type_str], HolovizOp.DepthMapRenderMode
    )


class TestHolovizOp:
    def test_kwarg_based_initialization(self, app, config_file, capfd):
        app.config(config_file)
        op = HolovizOp(app, name="visualizer", **app.kwargs("holoviz"))
        assert isinstance(op, _Operator)
        assert op.id != -1
        assert op.operator_type == Operator.OperatorType.NATIVE

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        # Initializing outside the context of app.run() will result in the
        # following error being logged because the GXFWrapper will not have
        # been created for the operator:
        #     [error] [gxf_executor.cpp:452] Unable to get GXFWrapper for Operator 'visualizer  # noqa: E501
        assert captured.err.count("[error]") <= 1
        assert "warning" not in captured.err

    @pytest.mark.parametrize(
        "tensor",
        [
            # color not length 4
            dict(
                name="scaled_coords",
                type="crosses",
                line_width=4,
                color=[1.0, 0.0],
            ),
            # color cannot be a str
            dict(
                name="scaled_coords",
                type="crosses",
                line_width=4,
                color="red",
            ),
            # color values out of range
            dict(
                name="scaled_coords",
                type="crosses",
                line_width=4,
                color=[255, 255, 255, 255],
            ),
            # unrecognized type
            dict(
                name="scaled_coords",
                type="invalid",
                line_width=4,
            ),
            # type not specified
            dict(
                name="scaled_coords",
                line_width=4,
            ),
            # name not specified
            dict(
                type="crosses",
                line_width=4,
            ),
            # unrecognized key specified
            dict(
                name="scaled_coords",
                type="crosses",
                line_width=4,
                color=[1.0, 1.0, 1.0, 0.0],
                invalid_key=None,
            ),
        ],
    )
    def test_invalid_tensors(self, tensor, app):
        with pytest.raises(ValueError):
            HolovizOp(
                name="visualizer",
                fragment=app,
                tensors=[tensor],
            )


class HolovizDepthMapSourceOp(Operator):
    def __init__(self, fragment, width, height, *args, **kwargs):
        self.width = width
        self.height = height
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        import cupy as cp

        depth_map = np.empty((self.height, self.width, 1), dtype=np.uint8)
        index = 0
        for y in range(self.height):
            for x in range(self.width):
                depth_map[y][x][0] = index
                index *= 4
        out_message = Entity(context)
        out_message.add(Tensor.as_tensor(cp.asarray(depth_map)), "depth_map")
        op_output.emit(out_message, "out")


class HolovizDepthMapSinkOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        op_input.receive("in")
        # Holoviz outputs a video buffer, but there is no support for video buffers in Python yet
        # message = op_input.receive("in")
        # image = message.get("render_buffer_output")


class MyHolovizDepthMapApp(Application):
    def compose(self):
        depth_map_width = 8
        depth_map_height = 4

        render_width = 32
        render_height = 32

        source = HolovizDepthMapSourceOp(
            self, depth_map_width, depth_map_height, CountCondition(self, 1), name="source"
        )

        alloc = UnboundedAllocator(
            fragment=self,
            name="allocator",
        )
        holoviz = HolovizOp(
            self,
            name="holoviz",
            width=render_width,
            height=render_height,
            headless=True,
            enable_render_buffer_output=True,
            allocator=alloc,
            tensors=[
                dict(name="depth_map", type="depth_map"),
            ],
        )

        sink = HolovizDepthMapSinkOp(self, name="sink")

        self.add_flow(source, holoviz, {("", "receivers")})
        self.add_flow(holoviz, sink, {("render_buffer_output", "")})


def test_holoviz_depth_map_app(capfd):
    pytest.importorskip("cupy")
    app = MyHolovizDepthMapApp()
    app.run()

    # assert no errors logged
    captured = capfd.readouterr()
    assert captured.err.count("[error]") == 0
