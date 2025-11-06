(visualization)=

```{eval-rst}
.. cpp:namespace:: holoscan
```

# Visualization

## Overview

Holoviz provides the functionality to composite real-time streams of frames with multiple different other layers like segmentation mask layers, geometry layers, and GUI layers.

For maximum performance, Holoviz makes use of [Vulkan](https://www.vulkan.org/), which is already installed as part of the NVIDIA GPU driver.

Holoscan provides the [Holoviz operator](#holoviz-operator) which is sufficient for many, even complex visualization tasks. The [Holoviz operator](#holoviz-operator) is used by multiple Holoscan [example applications](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/examples#holoscan-sdk-examples).

Additionally, for more advanced use cases, the [Holoviz module](#holoviz-module) can be used to create application-specific visualization operators. The [Holoviz module](#holoviz-module) provides a C++ API and is also used by the [Holoviz operator](#holoviz-operator).

The term Holoviz is used for both the [Holoviz operator](#holoviz-operator) and the [Holoviz module](#holoviz-module) below. Both the operator and the module roughly support the same feature set. Where applicable, information on how to use a feature with the operator and the module is provided. It's explicitly mentioned below when features are not supported by the operator.

## Layers

The core entity of Holoviz are layers. A layer is a two-dimensional image object. Multiple layers are composited to create the final output.

These layer types are supported by Holoviz:
- Image layer
- Geometry layer
- GUI layer

All layers have common attributes which define the look and also the way layers are finally composited.

The priority determines the rendering order of the layers. Before rendering, the layers are sorted by priority. The layers with the lowest priority are rendered first, so that the layer with the highest priority is rendered on top of all other layers. If layers have the same priority, then the render order of these layers is undefined.

The example below draws a transparent geometry layer on top of an image layer (geometry data and image data creation is omitted in the code). Although the geometry layer is specified first, it is drawn last because it has a higher priority (`1`) than the image layer (`0`).

`````{tab-set}
````{tab-item} Operator
The operator has a `receivers` port which accepts tensors and video buffers produced by other operators. Each tensor or video buffer will result in a layer.

The operator autodetects the layer type for certain input types (e.g., a video buffer will result in an image layer).

For other input types or more complex use cases, input specifications can be provided either at initialization time as a parameter or dynamically at runtime.

```cpp
std::vector<ops::HolovizOp::InputSpec> input_specs;

auto& geometry_spec =
    input_specs.emplace_back(ops::HolovizOp::InputSpec("point_tensor", ops::HolovizOp::InputType::POINTS));
geometry_spec.priority_ = 1;
geometry_spec.opacity_ = 0.5;

auto& image_spec =
    input_specs.emplace_back(ops::HolovizOp::InputSpec("image_tensor", ops::HolovizOp::InputType::IMAGE));
image_spec.priority_ = 0;

auto visualizer = make_operator<ops::HolovizOp>("holoviz", Arg("tensors", input_specs));

// the source provides two tensors named "point_tensor" and "image_tensor" at the "outputs" port.
add_flow(source, visualizer, {{"outputs", "receivers"}});
````
````{tab-item} Module
The definition of a layer is started by calling one of the layer begin functions {func}`viz::BeginImageLayer`, {func}`viz::BeginGeometryLayer` or  {func}`viz::BeginImGuiLayer`. The layer definition ends with {func}`viz::EndLayer`.

The start of a layer definition is resetting the layer attributes like priority and opacity to their defaults. So for the image layer, there is no need to set the opacity to `1.0` since the default is already `1.0`.

```cpp
namespace viz = holoscan::viz;

viz::Begin();

viz::BeginGeometryLayer();
viz::LayerPriority(1);
viz::LayerOpacity(0.5);
/// details omitted
viz::EndLayer();

viz::BeginImageLayer();
viz::LayerPriority(0);
/// details omitted
viz::EndLayer();

viz::End();
```
````
`````

### Image Layers

`````{tab-set}
````{tab-item} Operator
Image data can either be on host or device (GPU); both tensors and video buffers are accepted.

```cpp
std::vector<ops::HolovizOp::InputSpec> input_specs;

auto& image_spec =
    input_specs.emplace_back(ops::HolovizOp::InputSpec("image", ops::HolovizOp::InputType::IMAGE));

auto visualizer = make_operator<ops::HolovizOp>("holoviz", Arg("tensors", input_specs));

// the source provides an image named "image" at the "outputs" port.
add_flow(source, visualizer, {{"output", "receivers"}});
```

````
````{tab-item} Module
The function {func}`viz::BeginImageLayer` starts an image layer. An image layer displays a rectangular 2D image.

The image data is defined by calling {func}`viz::ImageCudaDevice`, {func}`viz::ImageCudaArray` or {func}`viz::ImageHost`. Various input formats are supported, see {enum}`viz::ImageFormat`.

For single channel image formats image colors can be looked up by defining a lookup table with {func}`viz::LUT`.

```cpp
viz::BeginImageLayer();
viz::ImageHost(width, height, format, data);
viz::EndLayer();
```

````
`````

#### Supported Image Formats

`````{tab-set}
````{tab-item} Operator
Supported formats for `nvidia::gxf::VideoBuffer`.

| nvidia::gxf::VideoFormat | Supported | Description |
|-|-|-|
| GXF_VIDEO_FORMAT_CUSTOM | - | |
| GXF_VIDEO_FORMAT_YUV420 | &check; | BT.601 multi planar 4:2:0 YUV |
| GXF_VIDEO_FORMAT_YUV420_ER | &check; | BT.601 multi planar 4:2:0 YUV ER |
| GXF_VIDEO_FORMAT_YUV420_709 | &check; | BT.709 multi planar 4:2:0 YUV |
| GXF_VIDEO_FORMAT_YUV420_709_ER | &check; | BT.709 multi planar 4:2:0 YUV ER |
| GXF_VIDEO_FORMAT_NV12 | &check; | BT.601 multi planar 4:2:0 YUV with interleaved UV |
| GXF_VIDEO_FORMAT_NV12_ER | &check; | BT.601 multi planar 4:2:0 YUV ER with interleaved UV |
| GXF_VIDEO_FORMAT_NV12_709 | &check; | BT.709 multi planar 4:2:0 YUV with interleaved UV |
| GXF_VIDEO_FORMAT_NV12_709_ER | &check; | BT.709 multi planar 4:2:0 YUV ER with interleaved UV |
| GXF_VIDEO_FORMAT_RGBA | &check; | RGBA-8-8-8-8 single plane |
| GXF_VIDEO_FORMAT_BGRA | &check; | BGRA-8-8-8-8 single plane |
| GXF_VIDEO_FORMAT_ARGB | &check; | ARGB-8-8-8-8 single plane |
| GXF_VIDEO_FORMAT_ABGR | &check; | ABGR-8-8-8-8 single plane |
| GXF_VIDEO_FORMAT_RGBX | &check; | RGBX-8-8-8-8 single plane |
| GXF_VIDEO_FORMAT_BGRX | &check; | BGRX-8-8-8-8 single plane |
| GXF_VIDEO_FORMAT_XRGB | &check; | XRGB-8-8-8-8 single plane |
| GXF_VIDEO_FORMAT_XBGR | &check; | XBGR-8-8-8-8 single plane |
| GXF_VIDEO_FORMAT_RGB | &check; |  RGB-8-8-8 single plane |
| GXF_VIDEO_FORMAT_BGR | &check; |  BGR-8-8-8 single plane |
| GXF_VIDEO_FORMAT_R8_G8_B8 | - | RGB - unsigned 8 bit multiplanar |
| GXF_VIDEO_FORMAT_B8_G8_R8 | - | BGR - unsigned 8 bit multiplanar |
| GXF_VIDEO_FORMAT_GRAY | &check; | 8 bit GRAY scale single plane |
| GXF_VIDEO_FORMAT_GRAY16 | &check; | 16 bit GRAY scale single plane |
| GXF_VIDEO_FORMAT_GRAY32 | - | 32 bit GRAY scale single plane |
| GXF_VIDEO_FORMAT_GRAY32F | &check; | float 32 bit GRAY scale single plane |
| GXF_VIDEO_FORMAT_RGB16 | - | RGB-16-16-16 single plane |
| GXF_VIDEO_FORMAT_BGR16 | - | BGR-16-16-16 single plane |
| GXF_VIDEO_FORMAT_RGB32 | - | RGB-32-32-32 single plane |
| GXF_VIDEO_FORMAT_BGR32 | - | BGR-32-32-32 single plane |
| GXF_VIDEO_FORMAT_R16_G16_B16 | - | RGB - signed 16 bit multiplanar |
| GXF_VIDEO_FORMAT_B16_G16_R16 | - | BGR - signed 16 bit multiplanar |
| GXF_VIDEO_FORMAT_R32_G32_B32 | - | RGB - signed 32 bit multiplanar |
| GXF_VIDEO_FORMAT_B32_G32_R32 | - | BGR - signed 32 bit multiplanar |
| GXF_VIDEO_FORMAT_NV24 | - | multi planar 4:4:4 YUV with interleaved UV |
| GXF_VIDEO_FORMAT_NV24_ER | - |  multi planar 4:4:4 YUV ER with interleaved UV |
| GXF_VIDEO_FORMAT_R8_G8_B8_D8 | - | RGBD unsigned 8 bit multiplanar |
| GXF_VIDEO_FORMAT_R16_G16_B16_D16 | - | RGBD unsigned 16 bit multiplanar |
| GXF_VIDEO_FORMAT_R32_G32_B32_D32 | - | RGBD unsigned 32 bit multiplanar |
| GXF_VIDEO_FORMAT_RGBD8 | - | RGBD 8 bit unsigned single plane |
| GXF_VIDEO_FORMAT_RGBD16 | - | RGBD 16 bit unsigned single plane |
| GXF_VIDEO_FORMAT_RGBD32 | - | RGBD 32 bit unsigned single plane |
| GXF_VIDEO_FORMAT_D32F | &check; | Depth 32 bit float single plane |
| GXF_VIDEO_FORMAT_D64F | - | Depth 64 bit float single plane |
| GXF_VIDEO_FORMAT_RAW16_RGGB | - | RGGB-16-16-16-16 single plane |
| GXF_VIDEO_FORMAT_RAW16_BGGR | - | BGGR-16-16-16-16 single plane |
| GXF_VIDEO_FORMAT_RAW16_GRBG | - | GRBG-16-16-16-16 single plane |
| GXF_VIDEO_FORMAT_RAW16_GBRG | - | GBRG-16-16-16-16 single plane |

Image format detection for `nvidia::gxf::Tensor`. Tensors don't have image format information attached. The Holoviz operator detects the image format from the tensor configuration.

| nvidia::gxf::PrimitiveType | Channels | Color format | Index for color lookup |
|-|-|-|-|
| kUnsigned8 | 1 | 8 bit GRAY scale single plane | &check; |
| kInt8 | 1 | signed 8 bit GRAY scale single plane | &check; |
| kUnsigned16 | 1 | 16 bit GRAY scale single plane | &check; |
| kInt16 | 1 | signed 16 bit GRAY scale single plane | &check; |
| kUnsigned32 | 1 | - | &check; |
| kInt32 | 1 | - | &check; |
| kFloat32 | 1 | float 32 bit GRAY scale single plane | &check; |
| kUnsigned8 | 3 | RGB-8-8-8 single plane | - |
| kInt8 | 3 | signed RGB-8-8-8 single plane | - |
| kUnsigned8 | 4 | RGBA-8-8-8-8 single plane | - |
| kInt8 | 4 | signed RGBA-8-8-8-8 single plane | - |
| kUnsigned16 | 4 | RGBA-16-16-16-16 single plane | - |
| kInt16 | 4 | signed RGBA-16-16-16-16 single plane | - |
| kFloat32 | 4 | RGBA float 32 single plane | - |

````
````{tab-item} Module
See {enum}`viz::ImageFormat` for supported image formats. Additionally {func}`viz::ImageComponentMapping` can be used to map the color components of an image to the color components of the output.
````
`````

### Geometry Layers

A geometry layer is used to draw 2d or 3d geometric primitives. 2d primitives are points, lines, line strips, rectangles, ovals or text and are defined with 2d coordinates (x, y). 3d primitives are points, lines, line strips or triangles and are defined with 3d coordinates (x, y, z).

Coordinates start with (0, 0) in the top left and end with (1, 1) in the bottom right for 2d primitives.

`````{tab-set}
````{tab-item} Operator

See [holoviz_geometry.cpp](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/examples/holoviz/cpp/holoviz_geometry.cpp) and [holoviz_geometry.py](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/examples/holoviz/python/holoviz_geometry.py) for 2d geometric primitives and and [holoviz_geometry.py](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/examples/holoviz/python/holoviz_geometry_3d.py) for 3d geometric primitives.

````
````{tab-item} Module
The function {func}`viz::BeginGeometryLayer` starts a geometry layer.

See {enum}`viz::PrimitiveTopology` for supported geometry primitive topologies.

There are functions to set attributes for geometric primitives like color ({func}`viz::Color`), line width ({func}`viz::LineWidth`), and point size ({func}`viz::PointSize`).

The code below draws a red rectangle and a green text.

```cpp
namespace viz = holoscan::viz;

viz::BeginGeometryLayer();

// draw a red rectangle
viz::Color(1.f, 0.f, 0.f, 0.f);
const float data[]{0.1f, 0.1f, 0.9f, 0.9f};
viz::Primitive(viz::PrimitiveTopology::RECTANGLE_LIST, 1, sizeof(data) / sizeof(data[0]), data);

// draw green text
viz::Color(0.f, 1.f, 0.f, 0.f);
viz::Text(0.5f, 0.5f, 0.2f, "Text");

viz::EndLayer();
```
````
`````

### ImGui Layers

:::{note}
ImGui layers are not supported when using the Holoviz operator.
:::

The Holoviz module supports user interface layers created with [Dear ImGui](https://github.com/ocornut/imgui).

Calls to the Dear ImGui API are allowed between {func}`viz::BeginImGuiLayer` and {func}`viz::EndImGuiLayer`, and are used to draw to the ImGui layer. The ImGui layer behaves like other layers and is rendered with the layer opacity and priority.

The code below creates a Dear ImGui window with a checkbox used to conditionally show a image layer.

```cpp
namespace viz = holoscan::viz;

bool show_image_layer = false;
while (!viz::WindowShouldClose()) {
    viz::Begin();

    viz::BeginImGuiLayer();

    ImGui::Begin("Options");
    ImGui::Checkbox("Image layer", &show_image_layer);
    ImGui::End();

    viz::EndLayer();

    if (show_image_layer) {
        viz::BeginImageLayer();
        viz::ImageHost(...);
        viz::EndLayer();
    }

    viz::End();
}
```

ImGUI is a static library and has no stable API. Therefore, the application and Holoviz have to use the same ImGUI version. When the link target `holoscan::viz::imgui` is exported, make sure to link your application against that target.

### Depth Map Layers

A depth map is a single channel 2d array where each element represents a depth value. The data is rendered as a 3D object using points, lines, or triangles. The color for the elements can also be specified.

Supported formats for the depth map:
- 8-bit unsigned normalized format that has a single 8-bit depth component
- 32-bit signed float format that has a single 32-bit depth component

Supported format for the depth color map:
- 32-bit unsigned normalized format that has an 8-bit R component in byte 0, an 8-bit G component in byte 1, an 8-bit B component in byte 2, and an 8-bit A component in byte 3

Depth maps are rendered in 3D and support camera movement.

`````{tab-set}
````{tab-item} Operator
```cpp
std::vector<ops::HolovizOp::InputSpec> input_specs;

auto& depth_map_spec =
    input_specs.emplace_back(ops::HolovizOp::InputSpec("depth_map", ops::HolovizOp::InputType::DEPTH_MAP));
depth_map_spec.depth_map_render_mode_ = ops::HolovizOp::DepthMapRenderMode::TRIANGLES;

auto visualizer = make_operator<ops::HolovizOp>("holoviz",
    Arg("tensors", input_specs));

// the source provides an depth map named "depth_map" at the "output" port.
add_flow(source, visualizer, {{"output", "receivers"}});
```
````
````{tab-item} Module
See [holoviz depth map demo](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/modules/holoviz/examples/depth_map).
````
`````

## Views

By default, a layer will fill the whole window. When using a view, the layer can be placed freely within the window.

Layers can also be placed in 3D space by specifying a 3D transformation matrix.
:::{note}
For geometry layers, there is a default matrix which allows coordinates in the range of [0 ... 1] instead of the Vulkan [-1 ... 1] range. When specifying a matrix for a geometry layer, this default matrix is overwritten.
:::

When multiple views are specified, the layer is drawn multiple times using the specified layer view.

It's possible to specify a negative term for height, which flips the image. When using a negative height, you should also adjust the y value to point to the lower left corner of the viewport instead of the upper left corner.

`````{tab-set}
````{tab-item} Operator
See [holoviz_views.py](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/examples/holoviz/python/holoviz_views.py).
````
````{tab-item} Module
Use {func}`viz::LayerAddView()` to add a view to a layer.
````
`````

## Camera

When rendering 3d geometry using a geometry layer with 3d primitives or using a depth map layer the camera properties can either be set by the application or interactively changed by the user.

To interactively change the camera, use the mouse:

- Orbit        (LMB)
- Pan          (LMB + CTRL  | MMB)
- Dolly        (LMB + SHIFT | RMB | Mouse wheel)
- Look Around  (LMB + ALT   | LMB + CTRL + SHIFT)
- Zoom         (Mouse wheel + SHIFT)

`````{tab-set}
````{tab-item} Operator
See [holoviz_camera.cpp](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/examples/holoviz/cpp/holoviz_camera.cpp).
````
````{tab-item} Module
Use {func}`viz::SetCamera()` to change the camera.
````
`````

(holoviz-display-mode)=

## Using a Display in Exclusive Mode

Typically, Holoviz opens a normal window on the Linux desktop. In that case, the desktop compositor is combining the Holoviz image with all other elements on the desktop. To avoid this extra compositing step, Holoviz can render to a display directly.

### Configure a Display for Exclusive Use

`````{tab-set}
````{tab-item} Single display
SSH into the machine and stop the X server:
```
sudo systemctl stop display-manager
```
To resume the `display manager`, run:
```bash
sudo systemctl start display-manager
```
````

````{tab-item} Multiple displays
The display to be used in exclusive mode needs to be disabled in the NVIDIA Settings application (`nvidia-settings`): open the `X Server Display Configuration` tab, select the display and under `Configuration` select `Disabled`. Press `Apply`.

````
`````

### Enable Exclusive Display in Holoviz

`````{tab-set}
````{tab-item} Operator
Arguments to pass to the Holoviz operator:

```cpp
auto visualizer = make_operator<ops::HolovizOp>("holoviz",
    Arg("use_exclusive_display", true), // required
    Arg("display_name", "DP-2"), // optional
    Arg("width", 2560), // optional
    Arg("height", 1440), // optional
    Arg("framerate", 240) // optional
    );

```
````
````{tab-item} Module
Provide the name of the display and desired display mode properties to {func}`viz::Init()`.

If the name is `nullptr`, then the first display is selected.
````
`````

The name of the display can either be the EDID name as displayed in the NVIDIA Settings, or the output name provided by `xrandr` or
`hwinfo --monitor`.

:::{tip}
`````{tab-set}
````{tab-item} X11
In this example output of `xrandr`, `DP-2` would be an adequate display name to use:
```bash
Screen 0: minimum 8 x 8, current 4480 x 1440, maximum 32767 x 32767
DP-0 disconnected (normal left inverted right x axis y axis)
DP-1 disconnected (normal left inverted right x axis y axis)
DP-2 connected primary 2560x1440+1920+0 (normal left inverted right x axis y axis) 600mm x 340mm
   2560x1440     59.98 + 239.97*  199.99   144.00   120.00    99.95
   1024x768      60.00
   800x600       60.32
   640x480       59.94
USB-C-0 disconnected (normal left inverted right x axis y axis)
```
````
````{tab-item} Wayland and X11
In this example output of `hwinfo`, `MSI MPG343CQR would be an adequate display name to use:
```bash
$ hwinfo --monitor | grep Model
  Model: "MSI MPG343CQR"
```
````
`````
:::

## CUDA Streams

By default, Holoviz is using CUDA stream `0` for all CUDA operations. Using the default stream can affect concurrency of CUDA operations, see [stream synchronization behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html#stream-sync-behavior__default-stream) for more information.

`````{tab-set}
````{tab-item} Operator
The operator is using a {class}`holoscan::CudaStreamPool` instance if provided by the `cuda_stream_pool` argument.
The stream pool is used to create a CUDA stream used by all Holoviz operations.

```cpp
const std::shared_ptr<holoscan::CudaStreamPool> cuda_stream_pool =
    make_resource<holoscan::CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);
auto visualizer =
    make_operator<holoscan::ops::HolovizOp>("visualizer",
        Arg("cuda_stream_pool") = cuda_stream_pool);
```

````
````{tab-item} Module
When providing CUDA resources to Holoviz through (e.g., {func}`viz::ImageCudaDevice`), Holoviz is using CUDA operations to use that memory. The CUDA stream used by these operations can be set by calling {func}`viz::SetCudaStream`. The stream can be changed at any time.
````
`````

## Reading the Frame Buffer

The rendered frame buffer can be read back. This is useful when doing offscreen rendering or running Holoviz in a headless environment.

`````{tab-set}
````{tab-item} Operator

To read back the color or depth frame buffer, set the `enable_render_buffer_output` or `enable_depth_buffer_output_` parameter to `true` and provide an allocator to the operator.

The color frame buffer is emitted on the `render_buffer_output` port, the depth frame buffer is emitted on the `depth_buffer_output` port.

```cpp
std::shared_ptr<holoscan::ops::HolovizOp> visualizer =
    make_operator<ops::HolovizOp>("visualizer",
        Arg("enable_render_buffer_output", true),
        Arg("enable_depth_buffer_output", true),
        Arg("allocator") = make_resource<holoscan::UnboundedAllocator>("allocator"),
        Arg("cuda_stream_pool") = cuda_stream_pool);

add_flow(visualizer, destination, {{"render_buffer_output", "color_input"}});
add_flow(visualizer, destination, {{"depth_buffer_output", "depth_input"}});
```

````
````{tab-item} Module
The rendered color or depth buffer can be read back using {func}`viz::ReadFramebuffer`.
````
`````

## sRGB

The sRGB color space is supported for both images and the framebuffer. By default Holoviz is using a linear encoded framebuffer.

`````{tab-set}
````{tab-item} Operator
To switch the framebuffer color format set the `framebuffer_srgb` parameter to `true`.

To use sRGB encoded images set the `image_format` field of  the `InputSpec` structure to a sRGB image format.
````
````{tab-item} Module
Use the {func}`viz::SetSurfaceFormat()` to set the framebuffer surface format to a sRGB color format.

To use sRGB encoded images set the `fmt` parameter of {func}`viz::ImageCudaDevice()`, {func}`viz::ImageCudaArray()` or {func}`viz::ImageHost()` to a sRGB image format.
````
`````

## HDR

Holoviz supports selecting the framebuffer surface image format and color space. To enable HDR select a HDR color space.

`````{tab-set}
````{tab-item} Operator
Set the `framebuffer_color_space` parameter to a supported HDR color space.
````
````{tab-item} Module
Use the {func}`viz::GetSurfaceFormats()` to query the available surface formats.

Use the {func}`viz::SetSurfaceFormat()` to set the framebuffer surface format to a surface format with a HDR color space.
````
`````

### Distributions supporting HDR

At the time of writing (08/2024) there is currently no official support for HDR on Linux. However there is experimental HDR support for Gnome version 44 and KDE Plasma 6.

#### KDE Plasma 6

Experimental HDR support is described in this [blog post](https://zamundaaa.github.io/wayland/2023/12/18/update-on-hdr-and-colormanagement-in-plasma.html). Three steps are required to use HDR with Holoviz:

1. Enable HDR in the display configuration
1. Install the [Vulkan HDR layer](https://github.com/Zamundaaa/VK_hdr_layer)
2. Set the `ENABLE_HDR_WSI` environment variable to `1`.

Run `vulkaninfo` to verify that HDR color spaces are reported

```shell
> vulkaninfo
...
GPU id : 0 (NVIDIA RTX 6000 Ada Generation):
        Surface type = VK_KHR_wayland_surface
        Formats: count = 11
                SurfaceFormat[0]:
                        format = FORMAT_A2B10G10R10_UNORM_PACK32
                        colorSpace = COLOR_SPACE_SRGB_NONLINEAR_KHR
                SurfaceFormat[1]:
                        format = FORMAT_A2R10G10B10_UNORM_PACK32
                        colorSpace = COLOR_SPACE_SRGB_NONLINEAR_KHR
                SurfaceFormat[2]:
                        format = FORMAT_R8G8B8A8_SRGB
                        colorSpace = COLOR_SPACE_SRGB_NONLINEAR_KHR
                SurfaceFormat[3]:
                        format = FORMAT_R8G8B8A8_UNORM
                        colorSpace = COLOR_SPACE_SRGB_NONLINEAR_KHR
                SurfaceFormat[4]:
                        format = FORMAT_B8G8R8A8_SRGB
                        colorSpace = COLOR_SPACE_SRGB_NONLINEAR_KHR
                SurfaceFormat[5]:
                        format = FORMAT_B8G8R8A8_UNORM
                        colorSpace = COLOR_SPACE_SRGB_NONLINEAR_KHR
                SurfaceFormat[6]:
                        format = FORMAT_R16G16B16A16_SFLOAT
                        colorSpace = COLOR_SPACE_SRGB_NONLINEAR_KHR
                SurfaceFormat[7]:
                        format = FORMAT_A2B10G10R10_UNORM_PACK32
                        colorSpace = COLOR_SPACE_HDR10_ST2084_EXT
                SurfaceFormat[8]:
                        format = FORMAT_A2R10G10B10_UNORM_PACK32
                        colorSpace = COLOR_SPACE_HDR10_ST2084_EXT
                SurfaceFormat[9]:
                        format = FORMAT_R16G16B16A16_SFLOAT
                        colorSpace = COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT
                SurfaceFormat[10]:
                        format = FORMAT_R16G16B16A16_SFLOAT
                        colorSpace = COLOR_SPACE_BT709_LINEAR_EXT
...
```

#### Gnome version 44

Gnome version 44 is part of Ubuntu 24.04.

Experimental HDR support had been added with this [MR](https://gitlab.gnome.org/GNOME/mutter/-/merge_requests/2879). To enable HDR, make sure Wayland is used (no HDR with X11), press `Alt+F2`, then type `lg` in the prompt to start the `looking glass` console. Enter `global.compositor.backend.get_monitor_manager().experimental_hdr = 'on'` to enable HDR, `global.compositor.backend.get_monitor_manager().experimental_hdr = 'off'` to disable HDR.

Gnome is not yet passing the HDR color space to Vulkan. The color space and other information can be queried using `drm_info`, check modes where `HDR_OUTPUT_METADATA` is not `0`.

For the case below (`Microstep MSI MPG343CQR` display) the HDR color space is `BT2020_RGB` with `SMPTE ST 2084 (PQ)` EOTF. Max luminance is 446 nits.

```shell
> drm_info | grep -B 1 -A 11 HDR
...
│   │       ├───"Colorspace": enum {Default, BT2020_RGB, BT2020_YCC} = BT2020_RGB
│   │       ├───"HDR_OUTPUT_METADATA": blob = 114
│   │       │   ├───Type: Static Metadata Type 1
│   │       │   ├───EOTF: SMPTE ST 2084 (PQ)
│   │       │   ├───Display primaries:
│   │       │   │   ├───Red: (0.6768, 0.3096)
│   │       │   │   ├───Green: (0.2764, 0.6445)
│   │       │   │   └───Blue: (0.1514, 0.0693)
│   │       │   ├───White point: (0.3135, 0.3291)
│   │       │   ├───Max display mastering luminance: 446 cd/m²
│   │       │   ├───Min display mastering luminance: 0.0001 cd/m²
│   │       │   ├───Max content light level: 446 cd/m²
│   │       │   └───Max frame average light level: 446 cd/m²
-
...
```

## Callbacks

Callbacks can be used to receive updates on key presses, mouse position and buttons, and window size.

`````{tab-set}
````{tab-item} Operator
C++
```cpp
visualizer = make_operator<ops::HolovizOp>(
            "holoviz",
            Arg("key_callback",
                ops::HolovizOp::KeyCallbackFunction([](ops::HolovizOp::Key key,
                                                       ops::HolovizOp::KeyAndButtonAction action,
                                                       ops::HolovizOp::KeyModifiers modifiers) -> void {
                HOLOSCAN_LOG_INFO(
                  "key {} action {} modifiers {}", int(key), int(action), *(uint32_t*)(&modifiers));
            }))
        )
```
Python
```python
def callback(
    key: HolovizOp.Key, action: HolovizOp.KeyAndButtonAction, modifiers: HolovizOp.KeyModifiers
):
    print(key, action, modifiers)

visualizer = HolovizOp(
            self,
            name="holoviz",
            key_callback=callback)
```
````
````{tab-item} Module
```cpp
void key_callback(void *user_pointer, Key key, KeyAndButtonAction action, KeyModifiers modifiers) {
    ...
}
...
viz::SetKeyCallback(user_pointer, &key_callback);
...
```
````
`````

## Scheduling Conditions

Holoviz provides specialized scheduling conditions that enable operators to synchronize their execution with display events. These conditions are particularly useful for rate-limiting and ensuring that frame generation is synchronized with the display's refresh cycle, reducing latency and improving visual quality.

### FirstPixelOut Condition

The {cpp:class}`C++ <holoscan::FirstPixelOutCondition>` allows an operator to synchronize its execution with the time when the first pixel of the next display refresh cycle leaves the display engine for the display (FirstPixelOut). By synchronizing with FirstPixelOut, applications can ensure minimal tearing and optimal frame pacing.

The condition internally waits for the next FirstPixelOut signal from the display hardware. When FirstPixelOut occurs, the condition transitions to a ready state, allowing the operator to execute. After execution, the condition transitions back to a waiting state until the next FirstPixelOut.

**Use cases:**
- Synchronizing frame generation with display refresh rate
- Minimizing display tearing in real-time visualization
- Rate-limiting operators to match the display's capabilities

### Present Done Condition

The {cpp:class}`C++ <holoscan::PresentDoneCondition>` allows an operator to synchronize its execution with the completion of frame presentation. This condition waits until a specific frame has been fully presented to the display before allowing the next operator execution.

The condition tracks presentation IDs, which increment with each frame presented. It blocks until a specific presentation ID is reached or a timeout occurs, ensuring that the application doesn't generate frames faster than the display can present them.

**Use cases:**
- Preventing frame queue buildup by matching generation to presentation rate
- Reducing end-to-end latency in interactive applications
- Ensuring smooth frame pacing without dropped frames

### Example Usage

The [holoviz_conditions](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/examples/holoviz/cpp/holoviz_conditions.cpp) example demonstrates how to use these conditions to rate-limit a source operator:

```cpp
#include <holoscan/operators/holoviz/conditions/present_done.hpp>

void compose() override {
  using namespace holoscan;

  auto visualizer = make_operator<ops::HolovizOp>("holoviz");

  auto source = make_operator<ops::SourceOp>(
      "source",
      // Run on every present completion
      make_condition<PresentDoneCondition>("present_limiter", visualizer));

  add_flow(source, visualizer, {{"output", "receivers"}});
}
```

## Multiprocess Synchronization

The HoloViz operator supports multiprocess synchronization using a file-based
FIFO mutex in the `/tmp/` directory. This enables better predictability with the
maximum end-to-end latency of the Holoscan SDK applications.

The support for this mutex is enabled by setting the environment variable
`HOLOSCAN_HOLOVIZ_MUTEX` to `1`. 

```bash
export HOLOSCAN_HOLOVIZ_MUTEX=1
```

When the environmentvariable is not set, the HoloViz operator will not use the
mutex, and there will be no attempt at explicit multiprocess synchronization.

When using the mutex, developers can optionally enable dropping frames for
an upper bound on the latency of the HoloViz operator. Setting HoloViz
operator's `multiprocess_framedrop_waittime_ms` parameter to a non-zero value
will drop frames when the mutex is not available for locking, even after waiting
for the specified amount of time in ms. In this case, the number of dropped
frames will be logged at the end of an application. Setting the parameter to 0,
which is the default value, will not drop any frames.

:::{note}
Although we do support automatic cleanup of the Holoviz mutex lock and mutex
queue files, an unexpected application crash or undefined application behavior
may not properly clear those files before shutting down. In that case, manually remove
those files (e.g., `rm -f /tmp/holoscan_holoviz_mutex*`).
:::


## Holoviz operator

### Class Documentation

{cpp:class}`C++ <holoscan::ops::HolovizOp>`

{py:class}`Python <holoscan.operators.HolovizOp>`.

### Holoviz Operator Examples

There are multiple [examples](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/holoviz), both in Python and C++, showing how to use various features of the Holoviz operator.

## Holoviz module

### Concepts

The Holoviz module uses the concept of the immediate mode design pattern for its API, inspired by the [Dear ImGui](https://github.com/ocornut/imgui) library. The difference to the retained mode, for which most APIs are designed, is that there are no objects created and stored by the application. This makes it fast and easy to make visualization changes in a Holoscan application.

### Instances

The Holoviz module uses a thread-local instance object to store its internal state. The instance object is created when calling the Holoviz module is first called from a thread. All Holoviz module functions called from that thread use this instance.

When calling into the Holoviz module from other threads other than the thread from which the Holoviz module functions were first called, make sure to call {func}`viz::GetCurrent()` and {func}`viz::SetCurrent()` in the respective threads.

There are usage cases where multiple instances are needed; for example, to open multiple windows. Instances can be created by calling {func}`viz::Create()`. Call {func}`viz::SetCurrent()` to make the instance current before calling the Holoviz module function to be executed for the window the instance belongs to.

### Getting Started

The code below creates a window and displays an image.

First, the Holoviz module needs to be initialized. This is done by calling {func}`viz::Init()`.

The elements to display are defined in the render loop; termination of the loop is checked with {func}`viz::WindowShouldClose`.

The definition of the displayed content starts with {func}`viz::Begin`, and ends with {func}`viz::End`. {func}`viz::End` starts the rendering and displays the rendered result.

Finally, the Holoviz module is shutdown with {func}`viz::Shutdown`.

```cpp
#include "holoviz/holoviz.hpp"

namespace viz = holoscan::viz;

viz::Init("Holoviz Example");

while (!viz::WindowShouldClose()) {
    viz::Begin();
    viz::BeginImageLayer();
    viz::ImageHost(width, height, viz::ImageFormat::R8G8B8A8_UNORM, image_data);
    viz::EndLayer();
    viz::End();
}

viz::Shutdown();
```

Result:

:::{figure-md} holoviz-example
:align: center

![](images/holoviz_example.png)

Holoviz example app

:::

### API

{ref}`namespace_holoscan__viz`

### Holoviz Module Examples

There are multiple [examples](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/modules/holoviz/examples) showing how to use various features of the Holoviz module.

### Fullscreen vs Exclusive Display Mode

Holoviz offers two different modes for rendering to a display taking up the
entire screen: **fullscreen mode** and **exclusive display mode**. While both modes
render content to fill the entire display, they differ significantly in how they
interact with the windowing system.

#### Fullscreen Mode

Fullscreen mode creates a borderless window that covers the entire display using GLFW (Graphics Library Framework). In this mode:

- **Window Manager Integration**: The window is still managed by the Linux window manager. Keyboard and mouse interactions are supported.
- **Desktop Compositor**: When the window covers the entire screen, the desktop
  compositor switches to flip mode, bypassing the compositing step. Therefore,
  the performance in fullscreen mode is better than in the windowed mode.
  However, the desktop compositor is still involved in the rendering path.
- **Simpler Setup**: No special display configuration is required. The
  application can switch to full-screen mode at runtime.
- **G-SYNC Support**: [G-SYNC](https://developer.nvidia.com/g-sync) is supported in fullscreen mode once it is enabled
  from NVIDIA Control Panel.

To enable fullscreen mode, set the `fullscreen` parameter to `true`:

`````{tab-set}
````{tab-item} Operator (C++)
```cpp
auto visualizer = make_operator<ops::HolovizOp>("holoviz",
    Arg("fullscreen", true)
    );
```
````
````{tab-item} Operator (Python)
```python
visualizer = HolovizOp(
    self,
    name="holoviz",
    fullscreen=True
)
```
````
`````

#### Exclusive Display Mode

Exclusive display mode renders directly to a display using Vulkan's
`VK_KHR_display` extension, completely bypassing the window manager and desktop
compositor. This mode takes up the entire screen, just like the fullscreen mode.
However, there are a few differences with the fullscreen mode. In this mode:

- **Direct Rendering**: The GPU writes pixels to a memory area, and the display controller reads from this memory and sends the pixels to the display. Vulkan's present call switches which memory region the display controller reads from, bypassing the window manager and compositor.
- **Performance**: Eliminating the involvement of the window manager anddesktop
  compositor altogether, this mode provides maximum rendering performance and minimal
  latency.
- **Display Control**: Takes exclusive control of the specified display,
  preventing other applications from using it. Keyboard and mouse interactions
  are not supported.
- **Setup Required**: Requires the display to be disabled in the X server configuration or the display manager to be stopped (see [Configure a Display for Exclusive Use](#configure-a-display-for-exclusive-use)).
- **Vulkan Extensions**: Uses Vulkan's direct display extensions.
- **G-SYNC Support**: G-SYNC is currently not supported in exclusive display mode.

To enable exclusive display mode, set the `use_exclusive_display` parameter to
`true` (as shown in [Enable Exclusive Display in
Holoviz](#enable-exclusive-display-in-holoviz)).
