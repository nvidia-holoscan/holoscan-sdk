"""
SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from contextlib import suppress

import pytest

import holoscan.operators
from holoscan.conditions import CountCondition
from holoscan.core import (
    Application,
    Arg,
    MetadataDictionary,
    MetadataPolicy,
    Operator,
    OperatorSpec,
    Tensor,
    _Operator,
)
from holoscan.core._core import OperatorSpec as OperatorSpecBase
from holoscan.gxf import Entity

# Import AJA if possible
with suppress(ImportError):
    from holoscan.operators.aja_source import AJASourceOp, NTV2Channel

from holoscan.operators import (
    BayerDemosaicOp,
    FormatConverterOp,
    HolovizOp,
    InferenceOp,
    InferenceProcessorOp,
    PingTensorRxOp,
    PingTensorTxOp,
    SegmentationPostprocessorOp,
    V4L2VideoCaptureOp,
    VideoStreamRecorderOp,
    VideoStreamReplayerOp,
)
from holoscan.operators.holoviz import (
    _holoviz_str_to_chroma_location,
    _holoviz_str_to_color_space,
    _holoviz_str_to_depth_map_render_mode,
    _holoviz_str_to_image_format,
    _holoviz_str_to_input_type,
    _holoviz_str_to_yuv_model_conversion,
    _holoviz_str_to_yuv_range,
)
from holoscan.resources import (
    BlockMemoryPool,
    CudaStreamPool,
    MemoryStorageType,
    UnboundedAllocator,
)

try:
    import numpy as np

    unsigned_dtypes = [np.uint8, np.uint16, np.uint32, np.uint64]
    signed_dtypes = [np.int8, np.int16, np.int32, np.int64]
    float_dtypes = [np.float16, np.float32, np.float64]
    complex_dtypes = [np.complex64, np.complex128]

except ImportError:
    unsigned_dtypes = signed_dtypes = float_dtypes = []

sample_data_path = os.environ.get("HOLOSCAN_INPUT_PATH", "../data")


class TestOperator:
    def test_default_init(self):
        with pytest.raises(TypeError):
            Operator()

    def test_invalid_init(self):
        with pytest.raises(ValueError):  # noqa: PT011
            Operator(5)

    def test_invalid_init2(self, fragment):
        # C++ PyOperator class will throw std::runtime_error if >1 fragment provided
        with pytest.raises(RuntimeError):
            Operator(fragment, fragment)

    def test_basic_init(self, fragment):
        op = Operator(fragment)
        assert op.name.startswith("unnamed_operator")
        assert op.fragment is fragment
        assert op.operator_type == Operator.OperatorType.NATIVE

    def test_metadata(self, fragment):
        op = Operator(fragment)
        assert op.metadata_policy == MetadataPolicy.RAISE
        assert isinstance(op.metadata, MetadataDictionary)
        meta = op.metadata
        assert len(meta) == 0
        meta["name"] = "abcd"
        op.metadata["age"] = 50
        assert len(op.metadata) == 2
        assert "name" in op.metadata
        assert "age" in op.metadata

        # metadata transmission is disabled by default
        assert not fragment.is_metadata_enabled
        assert not op.is_metadata_enabled

    def test_basic_kwarg_init(self, fragment, capfd):
        op = Operator(fragment=fragment)
        assert op.name.startswith("unnamed_operator")
        assert op.fragment is fragment
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
        if with_name:
            assert op.name == "my op"
        else:
            assert op.name.startswith("unnamed_operator")

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
        spec = OperatorSpecBase(app)

        op = Operator(app)
        # Operator.__init__ will have added op.spec for us
        assert isinstance(op.spec, OperatorSpecBase)

        app.config(config_file)

        # initialize context
        context = app.executor.context
        assert context is not None

        # follow order of operations in Fragment::make_operator
        op.name = "my operator"

        op.setup(spec)
        op.spec = spec
        assert isinstance(op.spec, OperatorSpecBase)

        op.initialize()
        assert op.id == -1  # native operator
        assert op.operator_type == Operator.OperatorType.NATIVE
        capfd.readouterr()

    def test_operator_setup_and_assignment(self, fragment, capfd):
        spec = OperatorSpecBase(fragment)
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
        assert type(t.__dlpack__()).__name__ == "PyCapsule"

        assert hasattr(t, "__dlpack_device__")
        dev = t.__dlpack_device__()
        assert isinstance(dev, tuple)
        assert len(dev) == 2

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
            assert interface["shape"] == reference_interface["shape"]
            assert interface["typestr"] == reference_interface["typestr"]
            assert interface["descr"] == reference_interface["descr"]
            assert interface["data"] == reference_interface["data"]
            if reference_interface["strides"] is not None:
                assert interface["strides"] == reference_interface["strides"]

    def _check_tensor_property_values(self, t, arr, cuda=False):
        assert t.size == arr.size
        assert t.nbytes == arr.nbytes
        assert t.ndim == arr.ndim
        assert t.itemsize == arr.dtype.itemsize

        assert t.shape == arr.shape
        assert t.strides == arr.strides

        assert isinstance(t.data, int)

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

    @pytest.mark.parametrize("module", ["cupy", "numpy"])
    @pytest.mark.parametrize("convert_method", ["as_tensor", "from_dlpack"])
    def test_from_dlpack(self, module, convert_method):
        # Check if module is numpy and numpy version is less than 1.23 then skip the test
        # because numpy.from_dlpack is not available in numpy versions less than 1.23
        if module == "numpy" and tuple(map(int, np.__version__.split("."))) < (1, 23):
            pytest.skip("requires numpy version >= 1.23")

        xp = pytest.importorskip(module)
        arr_in = xp.random.randn(1, 2, 3, 4).astype(xp.float32)
        converter = getattr(Tensor, convert_method)
        tensor = converter(arr_in)
        arr_out1 = xp.asarray(tensor)
        arr_out2 = xp.from_dlpack(tensor)
        xp.testing.assert_array_equal(arr_in, arr_out1)
        xp.testing.assert_array_equal(arr_in, arr_out2)


# Test cases for specific bundled native operators
@pytest.mark.skipif(not hasattr(holoscan.operators, "AJASourceOp"), reason="AJA not built")
class TestAJASourceOp:
    def test_kwarg_based_initialization(self, app, config_file, capfd):
        app.config(config_file)
        name = "source"
        op = AJASourceOp(
            fragment=app,
            name=name,
            channel=NTV2Channel.NTV2_CHANNEL1,
            width=1920,
            height=1080,
            rdma=True,
            enable_overlay=False,
            overlay_rdma=True,
        )
        assert isinstance(op, _Operator)
        assert op.operator_type == Operator.OperatorType.NATIVE
        assert f"name: {name}" in repr(op)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    def test_channel_kwarg_string_variant(self, app, config_file, capfd):
        app.config(config_file)
        name = "source"
        op = AJASourceOp(
            fragment=app,
            name=name,
            channel="NTV2_CHANNEL1",
            width=1920,
            height=1080,
            rdma=True,
            enable_overlay=False,
            overlay_channel="NTV2_CHANNEL2",
            overlay_rdma=True,
        )
        assert isinstance(op, _Operator)
        assert op.operator_type == Operator.OperatorType.NATIVE
        assert f"name: {name}" in repr(op)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    def test_initialization_from_yaml(self, app, config_file, capfd):
        app.config(config_file)
        name = "source"
        op = AJASourceOp(
            fragment=app,
            name=name,
            **app.kwargs("aja"),
        )
        assert isinstance(op, _Operator)
        assert op.operator_type == Operator.OperatorType.NATIVE
        assert f"name: {name}" in repr(op)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err


class TestFormatConverterOp:
    def test_kwarg_based_initialization(self, app, config_file, capfd):
        app.config(config_file)
        name = "recorder_format_converter"
        print(f"kwargs={app.kwargs('recorder_format_converter')}")
        op = FormatConverterOp(
            fragment=app,
            name=name,
            pool=BlockMemoryPool(
                app,
                name="pool",
                storage_type=MemoryStorageType.DEVICE,
                block_size=16 * 1024**2,
                num_blocks=4,
            ),
            **app.kwargs("recorder_format_converter"),
        )
        assert isinstance(op, _Operator)
        assert len(op.args) == 12
        assert op.operator_type == Operator.OperatorType.NATIVE
        assert f"name: {name}" in repr(op)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err


class TestInferenceOp:
    def test_kwarg_based_initialization(self, app, config_file, capfd):
        app.config(config_file)
        model_path = os.path.join(sample_data_path, "multiai_ultrasound", "models")

        model_path_map = {
            "plax_chamber": os.path.join(model_path, "plax_chamber.onnx"),
            "aortic_stenosis": os.path.join(model_path, "aortic_stenosis.onnx"),
            "bmode_perspective": os.path.join(model_path, "bmode_perspective.onnx"),
        }
        name = "inference"
        op = InferenceOp(
            app,
            name=name,
            allocator=UnboundedAllocator(app, name="pool"),
            model_path_map=model_path_map,
            **app.kwargs("inference"),
        )
        assert isinstance(op, _Operator)
        assert op.operator_type == Operator.OperatorType.NATIVE

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err


class TestInferenceProcessorOp:
    def test_kwarg_based_initialization(self, app, config_file, capfd):
        app.config(config_file)
        name = "inference_processor"
        op = InferenceProcessorOp(
            app,
            name=name,
            allocator=UnboundedAllocator(app, name="pool"),
            **app.kwargs("inference_processor"),
        )
        assert isinstance(op, _Operator)
        assert op.operator_type == Operator.OperatorType.NATIVE
        assert f"name: {name}" in repr(op)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err


class TestSegmentationPostprocessorOp:
    def test_kwarg_based_initialization(self, app, capfd):
        name = "segmentation_postprocessor"
        op = SegmentationPostprocessorOp(
            fragment=app,
            allocator=UnboundedAllocator(fragment=app, name="allocator"),
            name=name,
        )
        assert isinstance(op, _Operator)
        assert op.operator_type == Operator.OperatorType.NATIVE
        assert f"name: {name}" in repr(op)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err


class TestVideoStreamRecorderOp:
    def test_kwarg_based_initialization(self, app, config_file, capfd):
        app.config(config_file)
        name = "recorder"
        op = VideoStreamRecorderOp(name=name, fragment=app, **app.kwargs("recorder"))
        assert isinstance(op, _Operator)
        assert op.operator_type == Operator.OperatorType.NATIVE
        assert f"name: {name}" in repr(op)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err


class TestVideoStreamReplayerOp:
    def test_kwarg_based_initialization(self, app, config_file, capfd):
        app.config(config_file)
        name = "replayer"
        data_path = os.environ.get("HOLOSCAN_INPUT_PATH", "../data")
        op = VideoStreamReplayerOp(
            name=name,
            fragment=app,
            directory=os.path.join(data_path, "racerx"),
            **app.kwargs("replayer"),
        )
        assert isinstance(op, _Operator)
        assert op.operator_type == Operator.OperatorType.NATIVE
        assert f"name: {name}" in repr(op)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
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
        "points_3d",
        "lines_3d",
        "line_strip_3d",
        "triangles_3d",
    ],
)
def test_holoviz_input_types(type_str):
    assert isinstance(_holoviz_str_to_input_type[type_str], HolovizOp.InputType)


@pytest.mark.parametrize(
    "image_format_str",
    [
        "auto_detect",
        "r8_uint",
        "r8_sint",
        "r8_unorm",
        "r8_snorm",
        "r8_srgb",
        "r16_uint",
        "r16_sint",
        "r16_unorm",
        "r16_snorm",
        "r16_sfloat",
        "r32_uint",
        "r32_sint",
        "r32_sfloat",
        "r8g8b8_unorm",
        "r8g8b8_snorm",
        "r8g8b8_srgb",
        "r8g8b8a8_unorm",
        "r8g8b8a8_snorm",
        "r8g8b8a8_srgb",
        "r16g16b16a16_unorm",
        "r16g16b16a16_snorm",
        "r16g16b16a16_sfloat",
        "r32g32b32a32_sfloat",
        "d16_unorm",
        "x8_d24_unorm",
        "d32_sfloat",
        "a2b10g10r10_unorm_pack32",
        "a2r10g10b10_unorm_pack32",
        "b8g8r8a8_unorm",
        "b8g8r8a8_srgb",
        "a8b8g8r8_unorm_pack32",
        "a8b8g8r8_srgb_pack32",
        "y8u8y8v8_422_unorm",
        "u8y8v8y8_422_unorm",
        "y8_u8v8_2plane_420_unorm",
        "y8_u8v8_2plane_422_unorm",
        "y8_u8_v8_3plane_420_unorm",
        "y8_u8_v8_3plane_422_unorm",
        "y16_u16v16_2plane_420_unorm",
        "y16_u16v16_2plane_422_unorm",
        "y16_u16_v16_3plane_420_unorm",
        "y16_u16_v16_3plane_422_unorm",
    ],
)
def test_holoviz_image_formats(image_format_str):
    assert isinstance(_holoviz_str_to_image_format[image_format_str], HolovizOp.ImageFormat)


@pytest.mark.parametrize(
    "yuv_model_conversion",
    [
        "yuv_601",
        "yuv_709",
        "yuv_2020",
    ],
)
def test_holoviz_yuv_model_conversions(yuv_model_conversion):
    assert isinstance(
        _holoviz_str_to_yuv_model_conversion[yuv_model_conversion], HolovizOp.YuvModelConversion
    )


@pytest.mark.parametrize(
    "yuv_range",
    [
        "itu_full",
        "itu_narrow",
    ],
)
def test_holoviz_yuv_ranges(yuv_range):
    assert isinstance(_holoviz_str_to_yuv_range[yuv_range], HolovizOp.YuvRange)


@pytest.mark.parametrize(
    "chroma_location",
    [
        "cosited_even",
        "midpoint",
    ],
)
def test_holoviz_chroma_locations(chroma_location):
    assert isinstance(_holoviz_str_to_chroma_location[chroma_location], HolovizOp.ChromaLocation)


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


@pytest.mark.parametrize(
    "color_space",
    [
        "srgb_nonlinear",
        "extended_srgb_linear",
        "bt2020_linear",
        "hdr10_st2084",
        "pass_through",
        "bt709_linear",
        "auto",
    ],
)
def test_holoviz_color_spaces(color_space):
    assert isinstance(_holoviz_str_to_color_space[color_space], HolovizOp.ColorSpace)


class TestHolovizOpInputSpec:
    def test_input_type_based_initialization(self):
        HolovizOp.InputSpec("tensor1", HolovizOp.InputType.TRIANGLES)

    def test_string_based_initialization(self):
        HolovizOp.InputSpec("tensor1", "triangles")

    def test_opacity(self):
        spec = HolovizOp.InputSpec("tensor1", HolovizOp.InputType.COLOR)

        opacity = spec.opacity
        assert 0.0 <= opacity <= 1.0

        spec.opacity = 0.5

    def test_priority(self):
        spec = HolovizOp.InputSpec("tensor1", HolovizOp.InputType.LINES)

        priority = spec.priority
        assert priority >= 0

        spec.priority = 5

    def test_image_format(self):
        spec = HolovizOp.InputSpec("tensor1", HolovizOp.InputType.COLOR)

        image_format = spec.image_format
        assert image_format == HolovizOp.ImageFormat.AUTO_DETECT

        spec.image_format = HolovizOp.ImageFormat.R8G8B8A8_SRGB

    def test_color(self):
        spec = HolovizOp.InputSpec("tensor1", HolovizOp.InputType.TRIANGLES_3D)

        color = spec.color
        assert len(color) == 4

        spec.color = [1.0, 0.0, 0.0, 1.0]

    def test_line_width(self):
        spec = HolovizOp.InputSpec("tensor1", HolovizOp.InputType.LINES)

        line_width = spec.line_width
        assert line_width > 0

        spec.line_width = 5

    def test_point_size(self):
        spec = HolovizOp.InputSpec("tensor1", HolovizOp.InputType.POINTS_3D)

        point_size = spec.point_size
        assert point_size > 0

        spec.point_size = 2

    def test_text(self):
        spec = HolovizOp.InputSpec("tensor1", HolovizOp.InputType.TEXT)

        text = spec.text
        assert text == []

        spec.text = ["abc", "de", "fghij"]

    def test_yuv_model_conversion(self):
        spec = HolovizOp.InputSpec("tensor1", HolovizOp.InputType.COLOR)

        yuv_model_conversion = spec.yuv_model_conversion
        assert yuv_model_conversion == HolovizOp.YuvModelConversion.YUV_601

        spec.yuv_model_conversion = HolovizOp.YuvModelConversion.YUV_709

        with pytest.raises(TypeError):
            spec.yuv_model_conversion = 0

    def test_yuv_range(self):
        spec = HolovizOp.InputSpec("tensor1", HolovizOp.InputType.COLOR)

        yuv_range = spec.yuv_range
        assert yuv_range == HolovizOp.YuvRange.ITU_FULL

        spec.yuv_range = HolovizOp.YuvRange.ITU_NARROW

        with pytest.raises(TypeError):
            spec.yuv_range = 0

    def test_x_chroma_location(self):
        spec = HolovizOp.InputSpec("tensor1", HolovizOp.InputType.COLOR)

        x_chroma_location = spec.x_chroma_location
        assert x_chroma_location == HolovizOp.ChromaLocation.COSITED_EVEN

        spec.x_chroma_location = HolovizOp.ChromaLocation.MIDPOINT

        with pytest.raises(TypeError):
            spec.x_chroma_location = 0

    def test_y_chroma_location(self):
        spec = HolovizOp.InputSpec("tensor1", HolovizOp.InputType.COLOR)

        y_chroma_location = spec.y_chroma_location
        assert y_chroma_location == HolovizOp.ChromaLocation.COSITED_EVEN

        spec.y_chroma_location = HolovizOp.ChromaLocation.MIDPOINT

        with pytest.raises(TypeError):
            spec.y_chroma_location = 0

    def test_depth_map_render_mode(self):
        spec = HolovizOp.InputSpec("tensor1", HolovizOp.InputType.DEPTH_MAP)

        depth_map_render_mode = spec.depth_map_render_mode
        assert depth_map_render_mode == HolovizOp.DepthMapRenderMode.POINTS

        spec.depth_map_render_mode = HolovizOp.DepthMapRenderMode.TRIANGLES

        with pytest.raises(TypeError):
            spec.depth_map_render_mode = 0

    def test_views(self):
        spec = HolovizOp.InputSpec("tensor1", HolovizOp.InputType.COLOR)

        assert spec.views == []

        # add views to the spec
        v = HolovizOp.InputSpec.View()
        spec.views = [v, v, v]
        assert len(spec.views) == 3
        assert all(isinstance(v, HolovizOp.InputSpec.View) for v in spec.views)

    def test_description(self):
        spec = HolovizOp.InputSpec("tensor1", HolovizOp.InputType.COLOR)

        # add views to the spec
        v = HolovizOp.InputSpec.View()
        spec.views = [v, v, v]

        description = spec.description()

        assert isinstance(description, str)
        assert "name: tensor1" in description
        assert "type: color" in description

        # check that parameters were printed for all three views
        assert description.count("offset_x") == 3
        assert description.count("offset_y") == 3
        assert description.count("width") == 3
        assert description.count("height") == 3


class TestHolovizOpInputSpecView:
    def test_default_initialization(self):
        HolovizOp.InputSpec.View()

    def test_offset_x(self):
        v = HolovizOp.InputSpec.View()
        v.offset_x = 0.3
        assert abs(v.offset_x - 0.3) < 1e-5

    def test_offset_y(self):
        v = HolovizOp.InputSpec.View()
        v.offset_y = 0.3
        assert abs(v.offset_y - 0.3) < 1e-5

    def test_width(self):
        v = HolovizOp.InputSpec.View()
        v.width = 0.8
        assert abs(v.width - 0.8) < 1e-5

    def test_height(self):
        v = HolovizOp.InputSpec.View()
        v.height = 0.8
        assert abs(v.height - 0.8) < 1e-5

    def test_matrix(self):
        np = pytest.importorskip("numpy")

        view = HolovizOp.InputSpec.View()
        assert view.matrix is None

        with pytest.raises(TypeError):
            # can only set matrix with 16 element sequence
            view.matrix = (1.0, 0.0, 0.0)

        mat = np.eye(4)
        view.matrix = mat.ravel()
        assert view.matrix == mat.ravel().tolist()


class TestHolovizOp:
    def test_kwarg_based_initialization(self, app, config_file, capfd):
        app.config(config_file)
        op = HolovizOp(app, name="visualizer", **app.kwargs("holoviz"))
        assert isinstance(op, _Operator)
        assert op.operator_type == Operator.OperatorType.NATIVE

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
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
            # unrecognized image format
            dict(
                name="scaled_coords",
                type="color",
                image_format="invalid",
            ),
            # unrecognized display color space
            dict(
                name="scaled_coords",
                type="color",
                display_color_space="invalid",
            ),
        ],
    )
    def test_invalid_tensors(self, tensor, app):
        if "type" in tensor and tensor.get("type") == "invalid":
            # unrecognized type name will raise RuntimeError, not ValueError
            exception_type = RuntimeError
        else:
            exception_type = ValueError
        with pytest.raises(exception_type):
            HolovizOp(
                name="visualizer",
                fragment=app,
                tensors=[tensor],
            )


class HolovizDepthMapSourceOp(Operator):
    def __init__(self, fragment, width, height, depth_data_type, *args, **kwargs):
        self.width = width
        self.height = height
        self.depth_data_type = depth_data_type
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        import cupy as cp

        depth_map = np.empty((self.height, self.width, 1), dtype=self.depth_data_type)
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
        # TODO(unknown): Holoviz outputs a video buffer, but there is no support for video buffers in Python  # noqa: FIX002, E501
        # yet
        pass
        # message = op_input.receive("in")
        # image = message.get("render_buffer_output")


class MyHolovizDepthMapApp(Application):
    def __init__(self, depth_data_type, *args, **kwargs):
        self.depth_data_type = depth_data_type
        super().__init__(*args, **kwargs)

    def compose(self):
        depth_map_width = 8
        depth_map_height = 4

        render_width = 32
        render_height = 32

        source = HolovizDepthMapSourceOp(
            self,
            depth_map_width,
            depth_map_height,
            self.depth_data_type,
            CountCondition(self, 1),
            name="source",
        )

        alloc = UnboundedAllocator(
            fragment=self,
            name="allocator",
        )
        name = "holoviz"
        holoviz = HolovizOp(
            self,
            name=name,
            width=render_width,
            height=render_height,
            headless=True,
            enable_render_buffer_output=True,
            allocator=alloc,
            tensors=[
                dict(name="depth_map", type="depth_map"),
            ],
        )
        assert f"name: {name}" in repr(holoviz)

        # Since HolovizOp's render_buffer_output has ConditionType::kNone, we cannot depends on
        # deadlocks to be terminated. Instead, we use a CountCondition to terminate the operator.
        # Otherwise, the operator will run forever.
        sink = HolovizDepthMapSinkOp(self, CountCondition(self, 1), name="sink")

        self.add_flow(source, holoviz, {("", "receivers")})
        self.add_flow(holoviz, sink, {("render_buffer_output", "")})


@pytest.mark.parametrize("depth_data_type", [np.uint8, np.float32])
def test_holoviz_depth_map_app(capfd, depth_data_type):
    pytest.importorskip("cupy")
    app = MyHolovizDepthMapApp(depth_data_type)
    app.run()

    # assert no errors logged
    captured = capfd.readouterr()
    assert captured.err.count("[error]") == 0


class HolovizCallbackSourceOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        op_output.emit({"data": np.ones((1, 1, 4), dtype=np.uint8)}, "out")


class MyHolovizCallbackApp(Application):
    callback_invocations = []
    width = 640
    height = 512

    def callback(self, *args):
        self.callback_invocations.append(list(args))

    def compose(self):
        source = HolovizCallbackSourceOp(
            self,
            CountCondition(self, 1),
            name="source",
        )

        holoviz = HolovizOp(
            self,
            name="viz",
            width=self.width,
            height=self.height,
            key_callback=self.callback,
            unicode_char_callback=self.callback,
            mouse_button_callback=self.callback,
            scroll_callback=self.callback,
            cursor_pos_callback=self.callback,
            framebuffer_size_callback=self.callback,
            window_size_callback=self.callback,
        )

        self.add_flow(source, holoviz, {("", "receivers")})


def test_holoviz_callbacks(capfd):
    app = MyHolovizCallbackApp()
    app.run()

    # when setting the framebuffer and window callbacks the callback is called once with the
    # initial size
    assert len(app.callback_invocations) == 2
    assert app.callback_invocations[0] == [app.width, app.height]
    assert app.callback_invocations[1] == [app.width, app.height]

    # assert no errors logged
    captured = capfd.readouterr()
    assert captured.err.count("[error]") == 0


class TestBayerDemosaicOp:
    def test_kwarg_based_initialization(self, app, config_file, capfd):
        app.config(config_file)
        demosaic_stream_pool = CudaStreamPool(
            app,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )
        name = "demosaic"
        op = BayerDemosaicOp(
            app,
            name=name,
            pool=BlockMemoryPool(
                fragment=app,
                name="device_alloc",
                storage_type=MemoryStorageType.DEVICE,
                block_size=16 * 1024**2,
                num_blocks=4,
            ),
            cuda_stream_pool=demosaic_stream_pool,
            **app.kwargs("demosaic"),
        )
        assert isinstance(op, _Operator)
        assert op.operator_type == Operator.OperatorType.NATIVE
        assert f"name: {name}" in repr(op)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err


class TestV4L2VideoCaptureOp:
    def test_kwarg_based_initialization(self, app, capfd):
        name = "video_capture"
        op = V4L2VideoCaptureOp(
            app,
            name=name,
            width=320,
            height=240,
            pixel_format="auto",
            pass_through=False,
            device="/dev/video0",
            allocator=UnboundedAllocator(app, name="pool"),
            exposure_time=500,
            gain=100,
        )
        assert isinstance(op, _Operator)
        assert len(op.args) == 9
        assert op.operator_type == Operator.OperatorType.NATIVE
        assert f"name: {name}" in repr(op)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    def test_default_initialization(self, app, capfd):
        name = "video_capture"
        op = V4L2VideoCaptureOp(
            app,
            name=name,
            allocator=UnboundedAllocator(app, name="pool"),
        )
        assert isinstance(op, _Operator)
        assert len(op.args) == 7  # No hardcoded defaults for exposure and gain
        assert op.operator_type == Operator.OperatorType.NATIVE
        assert f"name: {name}" in repr(op)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    def test_initialization_from_yaml(self, app, config_file, capfd):
        app.config(config_file)
        name = "video_capture"
        print(f"{app.kwargs('v4l2_video_capture')}")
        op = V4L2VideoCaptureOp(
            app,
            name=name,
            allocator=UnboundedAllocator(app, name="pool"),
            **app.kwargs("v4l2_video_capture"),
        )
        assert isinstance(op, _Operator)
        assert len(op.args) == 9
        assert op.operator_type == Operator.OperatorType.NATIVE
        assert f"name: {name}" in repr(op)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err


class TestPingTensorRxOp:
    def test_default_initialization(self, app, capfd):
        name = "tensor_rx"
        op = PingTensorRxOp(
            fragment=app,
            name=name,
        )
        assert isinstance(op, _Operator)
        assert op.operator_type == Operator.OperatorType.NATIVE
        assert f"name: {name}" in repr(op)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err

        assert "warning" not in captured.err


class TestPingTensorTxOp:
    def test_default_initialization(self, app, capfd):
        name = "tensor_tx"
        op = PingTensorTxOp(
            fragment=app,
            name=name,
        )
        assert isinstance(op, _Operator)
        assert op.operator_type == Operator.OperatorType.NATIVE
        assert f"name: {name}" in repr(op)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    @pytest.mark.parametrize(
        "dtype",
        [
            np.dtype("float32"),
            np.dtype("float64"),
            np.dtype("complex64"),
            np.dtype("complex128"),
            np.dtype("int8"),
            np.dtype("int16"),
            np.dtype("int32"),
            np.dtype("int64"),
            np.dtype("uint8"),
            np.dtype("uint16"),
            np.dtype("uint32"),
            np.dtype("uint64"),
            np.dtype("bool"),  # will use uint8_t
            "float",
            "double",
            "complex<float>",
            "complex<double>",
            "int8_t",
            "int16_t",
            "int32_t",
            "int64_t",
            "uint8_t",
            "uint16_t",
            "uint32_t",
            "uint64_t",
        ],
    )
    def test_kwarg_based_initialization(self, app, capfd, dtype):
        name = "tensor_tx"
        op = PingTensorTxOp(
            fragment=app,
            allocator=UnboundedAllocator(app, name="alloc"),
            storage_type="system",
            batch_size=10,
            rows=16,
            columns=24,
            channels=3,
            dtype=dtype,
            tensor_name="image",
            name=name,
        )
        assert isinstance(op, _Operator)
        assert op.operator_type == Operator.OperatorType.NATIVE
        assert f"name: {name}" in repr(op)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err
