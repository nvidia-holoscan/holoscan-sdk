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
"""This module provides a Python API to underlying C++ API Operators.

.. autosummary::

    holoscan.operators.AJASourceOp
    holoscan.operators.BayerDemosaicOp
    holoscan.operators.FormatConverterOp
    holoscan.operators.HolovizOp
    holoscan.operators.LSTMTensorRTInferenceOp
    holoscan.operators.MultiAIInferenceOp
    holoscan.operators.MultiAIPostprocessorOp
    holoscan.operators.NTV2Channel
    holoscan.operators.SegmentationPostprocessorOp
    holoscan.operators.TensorRTInferenceOp
    holoscan.operators.ToolTrackingPostprocessorOp
    holoscan.operators.VideoStreamRecorderOp
    holoscan.operators.VideoStreamReplayerOp
    holoscan.operators.VisualizerICardioOp
"""

import os
from collections.abc import MutableMapping, Sequence
from os.path import join

from ..config import holoscan_gxf_extensions_path
from ..core import IOSpec
from ..resources import UnboundedAllocator
from ._operators import AJASourceOp, BayerDemosaicOp, FormatConverterOp
from ._operators import HolovizOp as _HolovizOp
from ._operators import LSTMTensorRTInferenceOp as _LSTMTensorRTInferenceOp
from ._operators import (
    MultiAIInferenceOp,
    MultiAIPostprocessorOp,
    NTV2Channel,
    SegmentationPostprocessorOp,
    TensorRTInferenceOp,
    ToolTrackingPostprocessorOp,
    VideoStreamRecorderOp,
    VideoStreamReplayerOp,
    VisualizerICardioOp,
)

# emergent source operator is not always available, depending on build options
try:
    from ._operators import EmergentSourceOp

    have_emergent_op = True
except ImportError:
    have_emergent_op = False


__all__ = [
    "AJASourceOp",
    "BayerDemosaicOp",
    "FormatConverterOp",
    "HolovizOp",
    "LSTMTensorRTInferenceOp",
    "MultiAIInferenceOp",
    "MultiAIPostprocessorOp",
    "NTV2Channel",
    "SegmentationPostprocessorOp",
    "TensorRTInferenceOp",
    "ToolTrackingPostprocessorOp",
    "VideoStreamRecorderOp",
    "VideoStreamReplayerOp",
    "VisualizerICardioOp",
]
if have_emergent_op:
    __all__ += ["EmergentSourceOp"]


shader_path = join(holoscan_gxf_extensions_path, "visualizer_tool_tracking", "glsl")
font_path = join(holoscan_gxf_extensions_path, "visualizer_tool_tracking", "fonts")
del holoscan_gxf_extensions_path

VIZ_TOOL_DEFAULT_COLORS = [
    [0.12, 0.47, 0.71],
    [0.20, 0.63, 0.17],
    [0.89, 0.10, 0.11],
    [1.00, 0.50, 0.00],
    [0.42, 0.24, 0.60],
    [0.69, 0.35, 0.16],
    [0.65, 0.81, 0.89],
    [0.70, 0.87, 0.54],
    [0.98, 0.60, 0.60],
    [0.99, 0.75, 0.44],
    [0.79, 0.70, 0.84],
    [1.00, 1.00, 0.60],
]


class LSTMTensorRTInferenceOp(_LSTMTensorRTInferenceOp):
    def __init__(
        self,
        fragment,
        pool,
        cuda_stream_pool,
        model_file_path,
        engine_cache_dir,
        input_tensor_names,
        input_state_tensor_names,
        input_binding_names,
        output_tensor_names,
        output_state_tensor_names,
        output_binding_names,
        max_workspace_size,
        force_engine_update=False,
        verbose=True,
        enable_fp16_=True,
        name="lstm_inferer",
    ):
        if not os.path.exists(model_file_path):
            raise ValueError(f"Could not locate model file: ({model_file_path=})")
        if not os.path.exists(engine_cache_dir):
            raise ValueError(f"Could not engine cache directory: ({engine_cache_dir=})")

        if len(input_binding_names) != len(input_tensor_names):
            raise ValueError("lengths of `input_binding_names` and `input_tensor_names` must match")
        if len(output_binding_names) != len(output_tensor_names):
            raise ValueError(
                "lengths of `output_binding_names` and `output_tensor_names` must match"
            )

        if pool is None:
            cuda_stream_pool = UnboundedAllocator(fragment=self, name="pool")

        super().__init__(
            fragment=fragment,
            pool=pool,
            cuda_stream_pool=cuda_stream_pool,
            model_file_path=model_file_path,
            engine_cache_dir=engine_cache_dir,
            input_tensor_names=input_tensor_names,
            input_state_tensor_names=input_state_tensor_names,
            input_binding_names=input_binding_names,
            output_tensor_names=output_tensor_names,
            output_state_tensor_names=output_state_tensor_names,
            output_binding_names=output_binding_names,
            max_workspace_size=max_workspace_size,
            force_engine_update=force_engine_update,
            verbose=verbose,
            enable_fp16_=enable_fp16_,
            name=name,
        )


# copy docstrings defined in operators_pydoc.hpp
LSTMTensorRTInferenceOp.__doc__ = _LSTMTensorRTInferenceOp.__doc__
LSTMTensorRTInferenceOp.__init__.__doc__ = _LSTMTensorRTInferenceOp.__init__.__doc__  # noqa
# TODO: remove from operators_pydoc.hpp and just define docstrings in this file?


_holoviz_str_to_input_type = {
    "unknown": _HolovizOp.InputType.UNKNOWN,
    "color": _HolovizOp.InputType.COLOR,
    "color_lut": _HolovizOp.InputType.COLOR_LUT,
    "points": _HolovizOp.InputType.POINTS,
    "lines": _HolovizOp.InputType.LINES,
    "line_strip": _HolovizOp.InputType.LINE_STRIP,
    "triangles": _HolovizOp.InputType.TRIANGLES,
    "crosses": _HolovizOp.InputType.CROSSES,
    "rectangles": _HolovizOp.InputType.RECTANGLES,
    "ovals": _HolovizOp.InputType.OVALS,
    "text": _HolovizOp.InputType.TEXT,
}


class HolovizOp(_HolovizOp):
    def __init__(
        self,
        fragment,
        allocator=None,
        receivers=[],
        tensors=[],
        color_lut=[],
        window_title="Holoviz",
        display_name="DP-0",
        width=1920,
        height=1080,
        framerate=60,
        use_exclusive_display=False,
        fullscreen=False,
        headless=False,
        enable_render_buffer_input=False,
        enable_render_buffer_output=False,
        name="holoviz_op",
    ):

        if allocator is None:
            allocator = UnboundedAllocator(fragment)

        receiver_iospecs = []
        for receiver in receivers:
            if isinstance(receiver, str):
                continue  # skip
                # raise NotImpelementedError(
                #     "TODO: need to enable access to self.spec for the OperatorSpec"
                # )
                # receiver = IOSpec(
                #     op_spec=self.spec,
                #     name=receiver,
                #     io_type=IOSpec.IOType.kInput
                # )
            elif not isinstance(receiver, IOSpec):
                raise ValueError(
                    "receivers must be a string containing the receiver name or an "
                    "IOSpec object."
                )
            if not receiver.io_type == IOSpec.IOType.kInput:
                raise ValueError("IOType of receiver IOSpec objects must be 'kInput'")
            receiver_iospecs.append(receiver)

        tensor_input_specs = []
        for tensor in tensors:
            if not isinstance(tensor, MutableMapping):
                raise ValueError(
                    "Tensors must be a sequence of MutableMappings " "(e.g. list of dict)."
                )
            if "name" not in tensor or not isinstance(tensor["name"], str):
                raise ValueError(
                    "Tensor dictionaries must contain key 'name' with a value " "that is a str."
                )
            if "type" not in tensor:
                raise ValueError("tensor dictionaries must contain key 'type'")
            valid_keys = {
                "name",
                "type",
                "opacity",
                "priority",
                "color",
                "line_width",
                "point_size",
                "text",
            }
            unrecognized_keys = set(tensor.keys()) - valid_keys
            if unrecognized_keys:
                raise ValueError(f"Unrecognized keys found in tensor: {unrecognized_keys}")
            input_type = tensor["type"]
            if isinstance(input_type, str):
                input_type.lower()
                if input_type not in _holoviz_str_to_input_type:
                    raise ValueError(f"unrecognized input_type name: {input_type}")
                input_type = _holoviz_str_to_input_type[input_type]
            elif not isinstance(input_type, _HolovizOp.InputType):
                raise ValueError(
                    "value corresponding to key 'type' must be either a "
                    "HolovizOp.InputType object or one of the following "
                    f"strings: {tuple(_holoviz_str_to_input_type.keys())}"
                )
            ispec = _HolovizOp.InputSpec(tensor["name"], input_type)
            ispec._opacity = tensor.get("opacity", 1.0)
            ispec._priority = tensor.get("priority", 0)
            if "color" in tensor:
                color = tensor["color"]
                if not isinstance(color, Sequence) or len(color) != 4:
                    raise ValueError(
                        "Colors must be specified as a sequence of 4 values: " "(R, G, B, A)."
                    )
                color = list(map(float, color))
                for val in color:
                    if val < 0.0 or val > 1.0:
                        raise ValueError("color values must be in range [0, 1]")
            else:
                color = (1.0,) * 4
            ispec._color = color
            ispec._line_width = tensor.get("line_width", 1.0)
            ispec._point_size = tensor.get("point_size", 1.0)
            if "text" in tensor:
                text = tensor["text"]
                is_seq = isinstance(text, Sequence) and not isinstance(text, str)
                if not (is_seq and all(isinstance(v, str) for v in text)):
                    raise ValueError("The text value of a tensor must be a sequence of strings")
            else:
                text = []
            ispec._text = tensor.get("text", text)
            tensor_input_specs.append(ispec)

        super().__init__(
            fragment=fragment,
            allocator=allocator,
            receivers=receiver_iospecs,
            tensors=tensor_input_specs,
            color_lut=color_lut,
            window_title=window_title,
            display_name=display_name,
            width=width,
            height=height,
            framerate=framerate,
            use_exclusive_display=use_exclusive_display,
            fullscreen=fullscreen,
            headless=headless,
            enable_render_buffer_input=enable_render_buffer_input,
            enable_render_buffer_output=enable_render_buffer_output,
            name=name,
        )


# copy docstrings defined in operators_pydoc.hpp
HolovizOp.__doc__ = _HolovizOp.__doc__
HolovizOp.__init__.__doc__ = _HolovizOp.__init__.__doc__
# TODO: remove from operators_pydoc.hpp and just define docstrings in this file?
