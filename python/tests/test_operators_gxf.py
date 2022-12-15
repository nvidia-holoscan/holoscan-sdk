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

import os

import pytest

from holoscan.core import Operator, _Operator
from holoscan.gxf import GXFOperator
from holoscan.operators import (
    AJASourceOp,
    BayerDemosaicOp,
    FormatConverterOp,
    HolovizOp,
    LSTMTensorRTInferenceOp,
    MultiAIInferenceOp,
    MultiAIPostprocessorOp,
    NTV2Channel,
    SegmentationPostprocessorOp,
    TensorRTInferenceOp,
    ToolTrackingPostprocessorOp,
    VideoStreamRecorderOp,
    VideoStreamReplayerOp,
    VisualizerICardioOp,
    _holoviz_str_to_input_type,
)
from holoscan.resources import (
    BlockMemoryPool,
    CudaStreamPool,
    MemoryStorageType,
    UnboundedAllocator,
)

try:
    from holoscan.operators import EmergentSourceOp

    have_emergent_op = True
except ImportError:
    have_emergent_op = False

sample_data_path = os.environ.get("HOLOSCAN_SAMPLE_DATA_PATH", "../data")


class TestAJASourceOp:
    def test_kwarg_based_initialization(self, app, config_file, capfd):
        app.config(config_file)
        op = AJASourceOp(
            fragment=app,
            name="source",
            channel=NTV2Channel.NTV2_CHANNEL1,
            **app.kwargs("aja"),
        )
        assert isinstance(op, GXFOperator)
        assert isinstance(op, _Operator)
        assert op.id != -1
        assert op.operator_type == Operator.OperatorType.GXF
        assert op.gxf_typename == "nvidia::holoscan::AJASource"

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        # assert "error" not in captured.err
        # TODO: resolve error logged by GXF:  "Unable to handle parameter 'channel'"
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
        assert isinstance(op, GXFOperator)
        assert isinstance(op, _Operator)
        len(op.args) == 12
        assert op.id != -1
        assert op.operator_type == Operator.OperatorType.GXF
        assert op.gxf_typename == "nvidia::holoscan::formatconverter::FormatConverter"

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err


class TestLSTMTensorRTInferenceOp:
    def test_kwarg_based_initialization(self, app, config_file, capfd):
        app.config(config_file)
        lstm_cuda_stream_pool = CudaStreamPool(
            fragment=app,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )
        op = LSTMTensorRTInferenceOp(
            fragment=app,
            name="lstm_inferer",
            pool=UnboundedAllocator(fragment=app, name="pool"),
            cuda_stream_pool=lstm_cuda_stream_pool,
            model_file_path=os.path.join(
                sample_data_path, "ultrasound", "model", "us_unet_256x256_nhwc.onnx"
            ),
            engine_cache_dir=os.path.join(
                sample_data_path, "ultrasound", "model", "us_unet_256x256_nhwc_engines"
            ),
            **app.kwargs("lstm_inference"),
        )
        assert isinstance(op, GXFOperator)
        assert isinstance(op, _Operator)
        assert len(op.args) == 17
        assert op.id != -1
        assert op.operator_type == Operator.OperatorType.GXF
        assert op.gxf_typename == "nvidia::holoscan::custom_lstm_inference::TensorRtInference"

        # assert no warnings or errors logged
        # captured = capfd.readouterr()
        # assert "error" not in captured.err
        # assert "warning" not in captured.err
        # Note: currently warning and error are present due to non-specified
        #       optional parameters
        #       error GXF_FAILURE setting GXF parameter 'clock'
        #       error GXF_FAILURE setting GXF parameter 'dla_core'


class TestTensorRTInferenceOp:
    def test_kwarg_based_initialization(self, app, config_file, capfd):
        app.config(config_file)
        lstm_cuda_stream_pool = CudaStreamPool(
            fragment=app,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )
        op = TensorRTInferenceOp(
            fragment=app,
            name="inferer",
            pool=UnboundedAllocator(fragment=app, name="pool"),
            cuda_stream_pool=lstm_cuda_stream_pool,
            model_file_path=os.path.join(
                sample_data_path, "ultrasound", "model", "us_unet_256x256_nhwc.onnx"
            ),
            engine_cache_dir=os.path.join(
                sample_data_path, "ultrasound", "model", "us_unet_256x256_nhwc_engines"
            ),
            **app.kwargs("segmentation_inference"),
        )
        assert isinstance(op, GXFOperator)
        assert isinstance(op, _Operator)
        assert op.id != -1
        assert op.operator_type == Operator.OperatorType.GXF
        assert op.gxf_typename == "nvidia::gxf::TensorRtInference"

        # assert no warnings or errors logged
        # captured = capfd.readouterr()
        # assert "error" not in captured.err
        # assert "warning" not in captured.err
        # Note: currently warning and error are present due to non-specified
        #       optional parameters
        #       error GXF_FAILURE setting GXF parameter 'clock'
        #       error GXF_FAILURE setting GXF parameter 'dla_core'


class TestVideoStreamRecorderOp:
    def test_kwarg_based_initialization(self, app, config_file, capfd):
        app.config(config_file)
        op = VideoStreamRecorderOp(name="recorder", fragment=app, **app.kwargs("recorder"))
        assert isinstance(op, GXFOperator)
        assert isinstance(op, _Operator)
        assert op.id != -1
        assert op.operator_type == Operator.OperatorType.GXF
        assert op.gxf_typename == "nvidia::gxf::EntityRecorder"

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err


class TestVideoStreamReplayerOp:
    def test_kwarg_based_initialization(self, app, config_file, capfd):
        app.config(config_file)
        op = VideoStreamReplayerOp(name="replayer", fragment=app, **app.kwargs("replayer"))
        assert isinstance(op, GXFOperator)
        assert isinstance(op, _Operator)
        assert op.id != -1
        assert op.operator_type == Operator.OperatorType.GXF
        assert op.gxf_typename == "nvidia::holoscan::stream_playback::VideoStreamReplayer"

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err


class TestSegmentationPostprocessorOp:
    def test_kwarg_based_initialization(self, app, capfd):
        op = SegmentationPostprocessorOp(
            fragment=app,
            allocator=UnboundedAllocator(fragment=app, name="allocator"),
            name="segmentation_postprocessor",
        )
        assert isinstance(op, GXFOperator)
        assert isinstance(op, _Operator)
        assert op.id != -1
        assert op.operator_type == Operator.OperatorType.GXF
        assert op.gxf_typename == "nvidia::holoscan::segmentation_postprocessor::Postprocessor"

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err


class TestToolTrackingPostprocessorOp:
    def test_kwarg_based_initialization(self, app, capfd):
        op = ToolTrackingPostprocessorOp(
            fragment=app,
            name="tool_tracking_postprocessor",
            device_allocator=BlockMemoryPool(
                fragment=app,
                name="device_alloc",
                storage_type=MemoryStorageType.DEVICE,
                block_size=16 * 1024**2,
                num_blocks=4,
            ),
            host_allocator=UnboundedAllocator(fragment=app, name="host_alloc"),
        )
        assert isinstance(op, GXFOperator)
        assert isinstance(op, _Operator)
        assert op.id != -1
        assert op.operator_type == Operator.OperatorType.GXF
        assert op.gxf_typename == "nvidia::holoscan::tool_tracking_postprocessor::Postprocessor"

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err


# TODO:
#    For holoviz, need to implement std::vector<InputSpec> "tensors" argument


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
    ],
)
def test_holoviz_input_types(type_str):
    assert isinstance(_holoviz_str_to_input_type[type_str], HolovizOp.InputType)


class TestHolovizOp:
    def test_kwarg_based_initialization(self, app, config_file, capfd):
        app.config(config_file)
        op = HolovizOp(app, name="visualizer", **app.kwargs("holoviz"))
        assert isinstance(op, GXFOperator)
        assert isinstance(op, _Operator)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err
        assert op.id != -1
        assert op.operator_type == Operator.OperatorType.GXF
        assert op.gxf_typename == "nvidia::holoscan::Holoviz"

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


class TestMultiAIInferenceOp:
    def test_kwarg_based_initialization(self, app, config_file, capfd):
        app.config(config_file)
        model_path = os.path.join(sample_data_path, "multiai_ultrasound", "models")

        model_path_map = {
            "icardio_plax_chamber": os.path.join(model_path, "plax_chamber.onnx"),
            "icardio_aortic_stenosis": os.path.join(model_path, "aortic_stenosis.onnx"),
            "icardio_bmode_perspective": os.path.join(model_path, "bmode_perspective.onnx"),
        }

        op = MultiAIInferenceOp(
            app,
            name="multiai_inference",
            allocator=UnboundedAllocator(app, name="pool"),
            model_path_map=model_path_map,
            **app.kwargs("multiai_inference"),
        )
        assert isinstance(op, GXFOperator)
        assert isinstance(op, _Operator)
        assert op.id != -1
        assert op.gxf_typename == "nvidia::holoscan::multiai::MultiAIInference"

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
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
        assert isinstance(op, GXFOperator)
        assert isinstance(op, _Operator)
        assert op.gxf_typename == "nvidia::holoscan::multiai::MultiAIPostprocessor"


class TestVisualizerICardioOp:
    def test_kwarg_based_initialization(self, app, config_file, capfd):
        app.config(config_file)
        op = VisualizerICardioOp(
            app,
            name="visualizer_icardio",
            allocator=UnboundedAllocator(app, name="pool"),
            **app.kwargs("visualizer_icardio"),
        )
        assert isinstance(op, GXFOperator)
        assert isinstance(op, _Operator)
        assert op.gxf_typename == "nvidia::holoscan::multiai::VisualizerICardio"


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
        op = BayerDemosaicOp(
            app,
            name="demosaic",
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
        assert isinstance(op, GXFOperator)
        assert isinstance(op, _Operator)
        assert op.id != -1
        assert op.gxf_typename == "nvidia::holoscan::BayerDemosaic"

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err


@pytest.mark.skipif(not have_emergent_op, reason="requires EmergentSourceOp")
class TestEmergentSourceOp:
    def test_kwarg_based_initialization(self, app, config_file, capfd):
        app.config(config_file)
        op = EmergentSourceOp(
            app,
            name="emergent",
            **app.kwargs("emergent"),
        )
        assert isinstance(op, GXFOperator)
        assert isinstance(op, _Operator)
        assert op.id != -1
        assert op.gxf_typename == "nvidia::holoscan::EmergentSource"

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err
