/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef PYHOLOSCAN_OPERATORS_PYDOC_HPP
#define PYHOLOSCAN_OPERATORS_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace FormatConverterOp {

PYDOC(FormatConverterOp, R"doc(
Format conversion operator.
)doc")

// PyFormatConverterOp Constructor
PYDOC(FormatConverterOp_python, R"doc(
Format conversion operator.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment that the operator belongs to.
pool : holoscan.resources.Allocator
    Memory pool allocator used by the operator.
out_dtype : str
    Destination data type (e.g. "RGB888" or "RGBA8888").
in_dtype : str, optional
    Source data type (e.g. "RGB888" or "RGBA8888").
in_tensor_name : str, optional
    The name of the input tensor.
out_tensor_name : str, optional
    The name of the output tensor.
scale_min : float, optional
    Output will be clipped to this minimum value.
scale_max : float, optional
    Output will be clipped to this maximum value.
alpha_value : int, optional
    Unsigned integer in range [0, 255], indicating the alpha channel value to use
    when converting from RGB to RGBA.
resize_height : int, optional
    Desired height for the (resized) output. Height will be unchanged if `resize_height` is 0.
resize_width : int, optional
    Desired width for the (resized) output. Width will be unchanged if `resize_width` is 0.
resize_mode : int, optional
    Resize mode enum value corresponding to NPP's nppiInterpolationMode (default=NPPI_INTER_CUBIC).
channel_order : sequence of int
    Sequence of integers describing how channel values are permuted.
cuda_stream_pool : holoscan.resources.CudaStreamPool, optional
    CudaStreamPool instance to allocate CUDA streams.
name : str, optional
    The name of the operator.
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Parameters
----------
spec : holoscan.core.OperatorSpec
    The operator specification.
)doc")

}  // namespace FormatConverterOp

namespace AJASourceOp {

PYDOC(AJASourceOp, R"doc(
Operator to get a video stream from an AJA capture card.
)doc")

// PyAJASourceOp Constructor
PYDOC(AJASourceOp_python, R"doc(
Operator to get a video stream from an AJA capture card.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment that the operator belongs to.
device : str, optional
    The device to target (e.g. "0" for device 0)
channel : holoscan.operators.NTV2Channel or int, optional
    The camera NTV2Channel to use for output.
width : int, optional
    Width of the video stream.
height : int, optional
    Height of the video stream.
framerate : int, optional
    Frame rate of the video stream.
rdma : bool, optional
    Boolean indicating whether RDMA is enabled.
enable_overlay : bool, optional
    Boolean indicating whether a separate overlay channel is enabled.
overlay_channel : holoscan.operators.NTV2Channel or int, optional
    The camera NTV2Channel to use for overlay output.
overlay_rdma : bool, optional
    Boolean indicating whether RDMA is enabled for the overlay.
name : str, optional
    The name of the operator.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Parameters
----------
spec : holoscan.core.OperatorSpec
    The operator specification.
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization.
)doc")

}  // namespace AJASourceOp

namespace HolovizOp {

PYDOC(HolovizOp, R"doc(
Holoviz visualization operator using Holoviz module.

This is a Vulkan-based visualizer.
)doc")

// PyHolovizOp Constructor
PYDOC(HolovizOp_python, R"doc(
Holoviz visualization operator using Holoviz module.

This is a Vulkan-based visualizer.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment that the operator belongs to.
allocator : holoscan.core.Allocator, optional
    Allocator used to allocate render buffer output. If None, will default to
    `holoscan.core.UnboundedAllocator`.
receivers : sequence of holoscan.core.IOSpec, optional
    List of input receivers.
tensors : sequence of dict, optional
    List of input tensors. Each tensor is defined by a dictionary where the 'name' key must
    correspond to a tensor sent to the operator's input. See the notes section below for further
    details on how the tensor dictionary is defined.
color_lut : list of list of float, optional
    Color lookup table for tensors of type 'color_lut'. Should be shape `(n_colors, 4)`.
window_title : str, optional
    Title on window canvas.
display_name : str, optional
    In exclusive mode, name of display to use as shown with xrandr.
width : int, optional
    Window width or display resolution width if in exclusive or fullscreen mode.
height : int, optional
    Window height or display resolution width if in exclusive or fullscreen mode.
framerate : float, optional
    Display framerate in Hz if in exclusive mode.
use_exclusive_display : bool, optional
    Enable exclusive display.
fullscreen : bool, optional
    Enable fullscreen window.
headless : bool, optional
    Enable headless mode. No window is opened, the render buffer is output to
    port `render_buffer_output`.
enable_render_buffer_input : bool, optional
    If ``True``, an additional input port, named 'render_buffer_input' is added to the
    operator.
enable_render_buffer_output : bool, optional
    If ``True``, an additional output port, named 'render_buffer_output' is added to the
    operator.
font_path : str, optional
    File path for the font used for rendering text.
cuda_stream_pool : holoscan.resources.CudaStreamPool, optional
    CudaStreamPool instance to allocate CUDA streams.
name : str, optional
    The name of the operator.

Notes
-----
The `tensors` argument is used to specify the tensors to display. Each tensor is defined using a
dictionary, that must, at minimum include a 'name' key that corresponds to a tensor found on the
operator's input. A 'type' key should also be provided to indicate the type of entry to display.
The 'type' key will be one of {"color", "color_lut", "crosses", "lines", "line_strip", "ovals",
"points", "rectangles", "text", "triangles", "depth_map", "depth_map_color", "unknown"}. The
default type is "unknown" which will attempt to guess the corresponding type based on the tensor
dimensions. Concrete examples are given below.

To show a single 2D RGB or RGBA image, use a list containing a single tensor of type 'color'.

.. code-block:: python

    tensors = [dict(name="video", type="color", opacity=1.0, priority=0)]

Here, the optional key `opacity` is used to scale the opacity of the tensor. The `priority` key
is used to specify the render priority for layers. Layers with a higher priority will be rendered
on top of those with a lower priority.

If we also had a "boxes" tensor representing rectangular bounding boxes, we could display them
on top of the image like this.

.. code-block:: python

    tensors = [
        dict(name="video", type="color", priority=0),
        dict(name="boxes", type="rectangles", color=[1.0, 0.0, 0.0], line_width=2, priority=1),
    ]

where the `color` and `line_width` keys specify the color and line width of the bounding box.
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Parameters
----------
spec : holoscan.core.OperatorSpec
    The operator specification.
)doc")

}  // namespace HolovizOp

namespace SegmentationPostprocessorOp {

PYDOC(SegmentationPostprocessorOp, R"doc(
Operator carrying out post-processing operations used in the ultrasound demo app.
)doc")

// PySegmentationPostprocessorOp Constructor
PYDOC(SegmentationPostprocessorOp_python, R"doc(
Operator carrying out post-processing operations used in the ultrasound demo app.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment that the operator belongs to.
allocator : holoscan.resources.Allocator
    Memory allocator to use for the output.
in_tensor_name : str, optional
    Name of the input tensor.
network_output_type : str, optional
    Network output type (e.g. 'softmax').
data_format : str, optional
    Data format of network output.
cuda_stream_pool : holoscan.resources.CudaStreamPool, optional
    CudaStreamPool instance to allocate CUDA streams.

name : str, optional
    The name of the operator.
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Parameters
----------
spec : holoscan.core.OperatorSpec
    The operator specification.
)doc")

}  // namespace SegmentationPostprocessorOp

namespace VideoStreamRecorderOp {

PYDOC(VideoStreamRecorderOp, R"doc(
Operator class to record the video stream to a file.
)doc")

// PyVideoStreamRecorderOp Constructor
PYDOC(VideoStreamRecorderOp_python, R"doc(
Operator class to record the video stream to a file.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment that the operator belongs to.
directory : str
    Directory path for storing files.
basename : str
    User specified file name without extension.
flush_on_tick : bool, optional
    Flushes output buffer on every tick when ``True``.
name : str, optional
    The name of the operator.
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Parameters
----------
spec : holoscan.core.OperatorSpec
    The operator specification.
)doc")

}  // namespace VideoStreamRecorderOp

namespace VideoStreamReplayerOp {

PYDOC(VideoStreamReplayerOp, R"doc(
Operator class to replay a video stream from a file.
)doc")

// PyVideoStreamReplayerOp Constructor
PYDOC(VideoStreamReplayerOp_python, R"doc(
Operator class to replay a video stream from a file.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment that the operator belongs to.
directory : str
    Directory path for reading files from.
basename : str
    User specified file name without extension.
batch_size : int, optional
    Number of entities to read and publish for one tick.
ignore_corrupted_entities : bool, optional
    If an entity could not be deserialized, it is ignored by default;
    otherwise a failure is generated.
frame_rate : float, optional
    Frame rate to replay. If zero value is specified, it follows timings in
    timestamps.
realtime : bool, optional
    Playback video in realtime, based on frame_rate or timestamps.
repeat : bool, optional
    Repeat video stream in a loop.
count : int, optional
    Number of frame counts to playback. If zero value is specified, it is
    ignored. If the count is less than the number of frames in the video, it
    would finish early.
name : str, optional
    The name of the operator.
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Parameters
----------
spec : holoscan.core.OperatorSpec
    The operator specification.
)doc")

}  // namespace VideoStreamReplayerOp

namespace TensorRTInferenceOp {

PYDOC(TensorRTInferenceOp, R"doc(
Operator class to perform inference using TensorRT.
)doc")

// PyTensorRTInferenceOp Constructor
PYDOC(TensorRTInferenceOp_python, R"doc(
Operator class to perform inference using TensorRT.

Parameters
----------
fragment : Fragment
    The fragment that the operator belongs to.
model_file_path : str
    Path to the ONNX model to be loaded.
engine_cache_dir : str
    Path to a folder containing cached engine files to be serialized and loaded from.
input_tensor_names : sequence of str
    Names of input tensors in the order to be fed into the model.
input_binding_names : sequence of str
    Names of input bindings as in the model in the same order of
    what is provided in `input_tensor_names`.
output_tensor_names : sequence of str
    Names of output tensors in the order to be retrieved from the model.
output_binding_names : sequence of str
    Names of output bindings as in the model in the same order of
    what is provided in `output_tensor_names`.
pool : ``holoscan.resources.Allocator``
    Allocator instance for output tensors.
cuda_stream_pool : ``holoscan.resources.CudaStreamPool``
    CudaStreamPool instance to allocate CUDA streams.
plugins_lib_namespace : str
    Namespace used to register all the plugins in this library.
force_engine_update : bool, optional
    Always update engine regardless of whether there is an existing engine file.
    Warning: this may take minutes to complete, so is False by default.
max_workspace_size : int, optional
    Size of working space in bytes.
max_batch_size : int, optional
    Maximum possible batch size in case the first dimension is dynamic and used
    as batch size.
enable_fp16 : bool, optional
    Enable inference with FP16 and FP32 fallback.
relaxed_dimension_check : bool, optional
    Ignore dimensions of 1 for input tensor dimension check.
verbose : bool, optional
    Enable verbose logging to the console.
name : str, optional
    The name of the operator.
)doc")

PYDOC(gxf_typename, R"doc(
The GXF type name of the resource.

Returns
-------
str
    The GXF type name of the resource
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Parameters
----------
spec : ``holoscan.core.OperatorSpec``
    The operator specification.
)doc")

}  // namespace TensorRTInferenceOp

namespace MultiAIInferenceOp {

PYDOC(MultiAIInferenceOp, R"doc(
Multi-AI inference operator.
)doc")

// PyMultiAIInferenceOp_python Constructor
PYDOC(MultiAIInferenceOp_python, R"doc(
Multi-AI inference operator.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment that the operator belongs to.
backend : {"trt", "onnxrt"}
    Backend to use for inference. Set "trt" for TensorRT and "onnxrt" for the
    ONNX runtime.
allocator : holoscan.resources.Allocator
    Memory allocator to use for the output.
inference_map : holoscan.operators.MultiAIInferenceOp.DataVecMap
    Tensor to model map.
model_path_map : holoscan.operators.MultiAIInferenceOp.DataMap
    Path to the ONNX model to be loaded.
pre_processor_map : holoscan.operators.MultiAIInferenceOp::DataVecMap
    Pre processed data to model map.
in_tensor_names : sequence of str, optional
    Input tensors.
out_tensor_names : sequence of str, optional
    Output tensors.
infer_on_cpu : bool, optional
    Whether to run the computation on the CPU instead of GPU.
parallel_inference : bool, optional
    Whether to enable parallel execution.
input_on_cuda : bool, optional
    Whether the input buffer is on the GPU.
output_on_cuda : bool, optional
    Whether the output buffer is on the GPU.
transmit_on_cuda : bool, optional
    Whether to transmit the message on the GPU.
enable_fp16 : bool, optional
    Use 16-bit floating point computations.
is_engine_path : bool, optional
    Whether the input model path mapping is for trt engine files
name : str, optional
    The name of the operator.
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Parameters
----------
spec : holoscan.core.OperatorSpec
    The operator specification.
)doc")

}  // namespace MultiAIInferenceOp

namespace MultiAIPostprocessorOp {

PYDOC(MultiAIPostprocessorOp, R"doc(
Multi-AI post-processing operator.
)doc")

// PyMultiAIPostprocessorOp Constructor
PYDOC(MultiAIPostprocessorOp_python, R"doc(
Multi-AI post-processing operator.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment that the operator belongs to.
allocator : holoscan.resources.Allocator
    Memory allocator to use for the output.
post_processor_map : holoscan.operators.MultiAIPostprocessorOp.DataVecMap
    All post processor settings for each model.
process_operations : holoscan.operators.MultiAIPostprocessorOp.DataVecMap
    Operations in sequence on tensors.
processed_map : holoscan.operators.MultiAIPostprocessorOp::DataVecMap
    Input-output tensor mapping.
in_tensor_names : sequence of str, optional
    Names of input tensors in the order to be fed into the operator.
out_tensor_names : sequence of str, optional
    Names of output tensors in the order to be fed into the operator.
input_on_cuda : bool, optional
    Whether the input buffer is on the GPU.
output_on_cuda : bool, optional
    Whether the output buffer is on the GPU.
transmit_on_cuda : bool, optional
    Whether to transmit the message on the GPU.
disable_transmitter : bool, optional
    If ``True``, disable the transmitter output port of the operator.
name : str, optional
    The name of the operator.
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Parameters
----------
spec : holoscan.core.OperatorSpec
    The operator specification.
)doc")

}  // namespace MultiAIPostprocessorOp

namespace BayerDemosaicOp {

// Constructor
PYDOC(BayerDemosaicOp, R"doc(
Format conversion operator.
)doc")

// PyBayerDemosaicOp Constructor
PYDOC(BayerDemosaicOp_python, R"doc(
Format conversion operator.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment that the operator belongs to.
pool : holoscan.resources.Allocator
    Memory pool allocator used by the operator.
cuda_stream_pool : holoscan.resources.CudaStreamPool
    CUDA Stream pool to create CUDA streams
in_tensor_name : str, optional
    The name of the input tensor.
out_tensor_name : str, optional
    The name of the output tensor.
interpolation_mode : int, optional
    The interpolation model to be used for demosaicing. Values available at:
    https://docs.nvidia.com/cuda/npp/group__typedefs__npp.html#ga2b58ebd329141d560aa4367f1708f191
bayer_grid_pos : int, optional
    The Bayer grid position (default of 2 = GBRG). Values available at:
    https://docs.nvidia.com/cuda/npp/group__typedefs__npp.html#ga5597309d6766fb2dffe155990d915ecb
generate_alpha : bool, optional
    Generate alpha channel.
alpha_value : int, optional
    Alpha value to be generated if `generate_alpha` is set to ``True``.
name : str, optional
    The name of the operator.
)doc")

PYDOC(gxf_typename, R"doc(
The GXF type name of the resource.

Returns
-------
str
    The GXF type name of the resource
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Parameters
----------
spec : holoscan.core.OperatorSpec
    The operator specification.
)doc")

}  // namespace BayerDemosaicOp

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_OPERATORS_PYDOC_HPP
