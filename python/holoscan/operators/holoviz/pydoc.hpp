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

#ifndef HOLOSCAN_OPERATORS_HOLOVIZ_PYDOC_HPP
#define HOLOSCAN_OPERATORS_HOLOVIZ_PYDOC_HPP

#include <string>

#include "../../macros.hpp"

namespace holoscan::doc::HolovizOp {

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
The 'type' key will be one of {"color", "color_lut", "crosses", "lines", "lines_3d", "line_strip",
"line_strip_3d", "ovals", "points", "points_3d", "rectangles", "text", "triangles", "triangles_3d",
"depth_map", "depth_map_color", "unknown"}. The default type is "unknown" which will attempt to
guess the corresponding type based on the tensor dimensions. Concrete examples are given below.

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

}  // namespace holoscan::doc::HolovizOp

namespace holoscan::doc::HolovizOp::InputSpec {

// HolovizOp.InputSpec Constructor
PYDOC(InputSpec, R"doc(
InputSpec for the HolovizOp operator.

Parameters
----------
tensor_name : str
    The tensor name for this input.
type : holoscan.operators.HolovizOp.InputType or str
    The type of data that this tensor represents.

Attributes
----------
type : holoscan.operators.HolovizOp.InputType
    The type of data that this tensor represents.
opacity : float
    The opacity of the object. Must be in range [0.0, 1.0] where 1.0 is fully opaque.
priority : int
    Layer priority, determines the render order. Layers with higher priority values are rendered
    on top of layers with lower priority.
color : 4-tuple of float
    RGBA values in range [0.0, 1.0] for rendered geometry.
line_width : float
    Line width for geometry made of lines.
point_size : float
    Point size for geometry made of points.
text : sequence of str
    Sequence of strings used when type is `HolovizOp.InputType.TEXT`.
depth_map_render_mode : holoscan.operators.HolovizOp.DepthMapRenderMode
    The depth map render mode. Used only if `type` is `HolovizOp.InputType.DEPTH_MAP` or
    `HolovizOp.InputType.DEPTH_MAP_COLOR`.
views : list of HolovizOp.InputSpec.View
    Sequence of layer views. By default a layer will fill the whole window. When using a view, the
    layer can be placed freely within the window. When multiple views are specified, the layer is
    drawn multiple times using the specified layer views.
)doc")

PYDOC(InputSpec_description, R"doc(
Returns
-------
description : str
    YAML string representation of the InputSpec class.
)doc")

// HolovizOp.InputSpec.View Constructor
PYDOC(View, R"doc(
View for the InputSpec of a HolovizOp operator.

Attributes
----------
offset_x, offset_y : float
    Offset of top-left corner of the view. (0, 0) is the upper left and (1, 1) is the lower
    right.
width : float
    Normalized width (range [0.0, 1.0]).
height : float
    Normalized height (range [0.0, 1.0]).
matrix : sequence of float
    16-elements representing a 4x4 transformation matrix.

Notes
-----
Layers can also be placed in 3D space by specifying a 3D transformation `matrix`. Note that for
geometry layers there is a default matrix which allows coordinates in the range of [0 ... 1]
instead of the Vulkan [-1 ... 1] range. When specifying a matrix for a geometry layer, this
default matrix is overwritten.

When multiple views are specified, the layer is drawn multiple times using the specified layer
views.

It's possible to specify a negative term for height, which flips the image. When using a
negative height, one should also adjust the y value to point to the lower left corner of the
viewport instead of the upper left corner.
)doc")

}  // namespace holoscan::doc::HolovizOp::InputSpec

#endif /* HOLOSCAN_OPERATORS_HOLOVIZ_PYDOC_HPP */
