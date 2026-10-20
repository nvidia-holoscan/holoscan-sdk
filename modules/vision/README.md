# Holoscan Vision

Holoscan Vision is a module in Holoscan SDK 5. It provides pipeline-native
wrappers for eight CV-CUDA operations and one NPP-backed Bayer demosaic operation:

The backend-derived names `CustomCropOp`, `CopyMakeBorderOp`, `ConvertToOp`,
and `ReformatOp` are deliberate: they keep the public operators discoverable
against the corresponding CV-CUDA documentation.

| Holoscan operator | Registry name | Ports | Configuration |
| --- | --- | --- | --- |
| `BayerDemosaicOp` | `holoscan.vision.BayerDemosaicOp` | `input` → `output` | Reconstruct RGB or RGBA from a Bayer mosaic using NPP |
| `ResizeOp` | `holoscan.vision.ResizeOp` | `input` → `output` | Width, height, interpolation |
| `CustomCropOp` | `holoscan.vision.CustomCropOp` | `input` → `output` | Crop rectangle |
| `CopyMakeBorderOp` | `holoscan.vision.CopyMakeBorderOp` | `input` → `output` | Border widths, mode, value |
| `RemapOp` | `holoscan.vision.RemapOp` | `input` + `map` → `output` | Size, interpolation, map mode, map synchronization/age, border |
| `NormalizeOp` | `holoscan.vision.NormalizeOp` | `input` → `output` | Base, scale, global scale, shift, epsilon |
| `ConvertToOp` | `holoscan.vision.ConvertToOp` | `input` → `output` | Output dtype, alpha, beta, rounding |
| `ReformatOp` | `holoscan.vision.ReformatOp` | `input` → `output` | Interleaved/planar direction |
| `NonMaximumSuppressionOp` | `holoscan.vision.NonMaximumSuppressionOp` | `boxes` + `scores` → `mask` | Score/IoU thresholds |

The eight image operators expose `Input<schema::ImageT>` and
`Output<schema::ImageT>` directly. Their `Image.data` field carries the CUDA
tensor; `RemapOp::map` and all NMS ports carry `Tensor` payloads. Image producers
and consumers must use the typed image ports; there are no Tensor compatibility
ports or payload template variants.

The CV-CUDA operators use `nvcv::TensorWrapData` around the existing Holoscan input
tensor and the plan-owned output tensor loan. They do not create
`cvcuda::Image`, stage tensor contents, copy between tensor APIs, or synchronize
the host. Bayer passes input/output pointers directly to NPP. Both backends
submit work on the operator's HSDK runtime CUDA stream.

## Supported tensors

- Image operators accept rank 3 `HWC` and rank 4 `NHWC`, except
  `BayerDemosaicOp`, which accepts only rank 3 `HWC`. `ReformatOp` also
  accepts and produces `CHW` and `NCHW`.
- After `ReformatOp` produces `CHW` or `NCHW`, the next Holoscan Vision image
  operator must be `ReformatOp` configured to convert back to `HWC` or `NHWC`;
  otherwise, the tensor must leave the Vision module.
- All dtypes in the table are scalar, byte-addressable DLPack dtypes.
- `Image.width` and `Image.height` must match the tensor's spatial dimensions.
  Named encodings must agree with its dtype, channel count, and layout; supplied
  plane descriptors must agree with its strides. `CUSTOM` is accepted when its
  scalar storage satisfies the same operator requirements.
- Packed and chroma-subsampled encodings are unsupported. They are rejected,
  never reinterpreted as scalar HWC/NHWC data.

| Operator input | Channels | Accepted dtypes |
| --- | --- | --- |
| `BayerDemosaicOp` | 1 | U8 or U16 |
| `ResizeOp` | 1, 3, or 4 | U8, U16, S16, or F32 |
| `CustomCropOp` | 1–4 | U8, S8, U16, S16, S32, F32, or F64 |
| `CopyMakeBorderOp` | 1, 3, or 4; 2 only for U8 | U8, U16, S16, or F32 |
| `RemapOp` image | 1, 3, or 4 for U8; 1 for F32 | U8 or F32 |
| `RemapOp` map | 2 | F32 |
| `NormalizeOp` | 1, 3, or 4 | U8, S8, U16, S16, S32, or F32 |
| `ConvertToOp` | 1–4 | U8, S8, U16, S16, S32, F32, or F64 |
| `ReformatOp` | 1–4 | U8, S8, U16, S16, S32, F32, or F64 |

- Interleaved inputs require tightly packed channels and pixels. Row and batch
  pitch may include padding. Planar `ReformatOp` inputs require tightly packed
  pixels, while row, channel, and batch pitch may include padding. Vision
  outputs use contiguous storage.
- The underlying CV-CUDA operators use 32-bit addressing. Every byte stride and
  each tensor's total strided byte span must not exceed `INT32_MAX`.
- `NonMaximumSuppressionOp`: S16 `[N, W, 4]` boxes and F32 `[N, W]` scores;
  output is a contiguous U8 `[N, W]` mask. Boxes and scores may have row pitch,
  but coordinates and score values must be tightly packed within each row.
  Boxes require an 8-byte-aligned base address and batch pitch; scores require
  4-byte alignment for both.
- All tensor inputs and outputs participating in one operator invocation must
  use the same CUDA device.

`ResizeOp` supports nearest-neighbor, linear, cubic, and area interpolation.
`RemapOp` supports absolute, absolute-normalized, and relative-normalized maps.
Its F32 two-channel map may have a batch size of one or the image batch size.
The map's spatial dimensions may differ from the input and output because
CV-CUDA samples it over the destination grid.
`CustomCropOp` requires the complete crop rectangle to lie within the input.

`VisionElementType` maps to scalar DLPack types as follows:

| Enum | DLPack code | Bits | Lanes |
| --- | --- | ---: | ---: |
| `kU8` | `kDLUInt` | 8 | 1 |
| `kS8` | `kDLInt` | 8 | 1 |
| `kU16` | `kDLUInt` | 16 | 1 |
| `kS16` | `kDLInt` | 16 | 1 |
| `kS32` | `kDLInt` | 32 | 1 |
| `kF32` | `kDLFloat` | 32 | 1 |
| `kF64` | `kDLFloat` | 64 | 1 |

`NormalizeOp` preserves dtype. Use `ConvertToOp` with output type `kF32`
before normalization when a U8 input must produce an F32 output.

## Bayer demosaic

`BayerDemosaicOp` uses NPP's CFA-to-RGB(A) kernels because CV-CUDA has no
corresponding Bayer demosaic operation. Its `ImageT` input carries a CUDA-device
U8 or U16 tensor shaped `[H, W, 1]`, with even width and height and tightly
packed samples. Row pitch may include padding. The image dimensions, encoding,
and supplied plane records must agree with the tensor.

`bayer_grid_pos` declares the CFA phase at the input ROI origin and defaults to
`kGBRG`. Each named `BAYER_RGGB`, `BAYER_BGGR`, `BAYER_GRBG`, or `BAYER_GBRG`
encoding must match the corresponding parameter value; a contradiction returns
`kInvalidArgument` before allocation. For `CUSTOM`, `MONO8`, or `MONO16`, the
parameter supplies the otherwise unnamed Bayer phase. A producer using these
encodings must supply mosaic samples rather than an already reconstructed image.

Output has the same dimensions and element type, with contiguous RGB storage
or RGBA when `generate_alpha` is true. `alpha_value` is unscaled: its default
255 remains 255 in U16 output; use 65535 for full-scale U16 alpha. U8 RGBA
rejects values above 255 before allocation. When alpha generation is disabled,
`alpha_value` is unused.

Output encoding is `RGB8`/`RGBA8` or `RGB16`/`RGBA16`. ROI offsets, image header,
exposure, gain, sample metadata, and provenance are preserved. `color_space` is
cleared and `significant_bits` becomes unknown because interpolation changes
sample values. Plane layout is rebuilt for the output tensor, and publication
attaches its committed runtime-pool loan without copying pixels.

## Image metadata

Outputs retain sample capture time, frame identity, trace identity, and flags,
as well as the image header, exposure, and gain. Width, height, encoding, and
plane layouts describe the output tensor. Plane offsets and row strides are
rebuilt from actual output strides; for a batch they describe the first image,
and the tensor batch stride locates subsequent images.

| Operation | Metadata transformation |
| --- | --- |
| Bayer demosaic | Preserve geometry and ROI; set RGB8/RGBA8 or RGB16/RGBA16 encoding, mark significant bits unknown, and clear color space. |
| Crop | Add the crop origin to ROI offsets with overflow checks; shift Bayer phase by crop parity. Preserve color space and significant bits. |
| Border | Subtract left/top padding from ROI offsets if both remain nonnegative; otherwise reset both to zero. Any padding changes Bayer encoding to `CUSTOM`. Zero or sample-copy borders preserve significant bits; nonzero constant borders mark precision unknown. No padding preserves metadata. |
| Resize | Changed dimensions reset both ROI offsets and change Bayer encoding to `CUSTOM`. Nearest-neighbor preserves significant bits; other interpolation marks precision unknown. Unchanged dimensions preserve these claims. |
| Remap | Always reset ROI offsets and change Bayer encoding to `CUSTOM`. Preserve significant bits only with nearest-neighbor image sampling and zero or sample-copy borders. |
| Normalize | Preserve geometry and dtype. Value-changing normalization marks precision unknown and clears `color_space`; identity normalization preserves both fields. |
| Convert | Preserve geometry. Unscaled lossless casts retain source precision; lossy casts or nonidentity alpha/beta mark it unknown. Dtype changes or nonidentity alpha/beta clear `color_space`. Select a corresponding named encoding for the new dtype when available, otherwise `CUSTOM`. |
| Reformat | Preserve geometry, color space, and significant bits. Map `RGB8`/`BGR8` to `R8_G8_B8`/`B8_G8_R8` for planar storage and back for interleaved storage. Use `CUSTOM` when no named encoding describes the target layout. |

Geometric operations preserve color space. Unknown precision is represented by
`schema::kUnknownImageSignificantBits` (255); zero still means full container
width. An unscaled lossless cast retains source precision, expanding zero to
the input container width when the dtype changes. For example, a 10-bit
`MONO16` image cast to F32 still reports 10 significant bits. This field does
not specify a code range or valid-bit alignment: casting does not rescale
white, shift bits, or normalize integer values to [0, 1]. Consumers must handle
unknown precision explicitly instead of inferring it from the output dtype.
The image descriptor is emitted with the committed runtime-pool tensor loan through `output.emit_tensor`, without copying pixel storage.

Unary operators consume each input independently. By default, `RemapOp` consumes
each image with the newest causal map and can reuse that map for later images up
to `map_max_age`; its default maximum age permits run-lifetime reuse. Set
`map_synchronization` to `kExact` to require identical image and map capture
times. `NonMaximumSuppressionOp` always requires identical capture times for its
boxes and scores. NMS unions both inputs' flags, retains frame identity only
on agreement, and retires conflicting trace roots. It preserves their common
capture time and rejects mixed clock domains. Remap treats the map as
configuration: capture time, frame identity, and trace identity come from the
image, while flags are unioned with the selected map's flags, even when that
map is retained. `RemapOp` applies the image metadata rules above.

## Authoring a graph

Output pools are finite plan resources. Vision operators are default-constructible, and their typed
Parameters contain the structural fields used to derive pool capacity. Fixed-output operators use
their output dimensions directly. `NormalizeOp` and `ReformatOp` inherit the connected input byte
bound, and `ConvertToOp` scales that bound for its selected output dtype. `CopyMakeBorderOp` uses
explicit maximum output dimensions, and NMS uses an explicit maximum box count. Complete
configuration loading requires the retained capacity fields marked `required_on_load`; direct C++
aggregate setup should likewise set them to the application's real bound instead of relying on
single-sample defaults. Batch capacities cannot exceed `holoscan::vision::kMaximumBatchSize`
(65,535).

The module catalog registers every default-constructible operator with
`HOLOSCAN_REGISTER_OPERATOR`. Registry metadata, keyword validation, and generated Python typing
therefore come directly from each operator's canonical `Params` schema. The rows intentionally do
not advertise graph-owned configuration binding; use the typed C++ handle and its actual `Params`
type when attaching configuration to an existing operator.

```cpp
#include <holoscan/vision/custom_crop_op.hpp>
#include <holoscan/vision/copy_make_border_op.hpp>
#include <holoscan/vision/resize_op.hpp>

holoscan::Graph graph{"vision"};

auto resize = graph.op<holoscan::vision::ResizeOp>("resize");
resize.set_params(holoscan::vision::ResizeOpParams{
    .width = 1280,
    .height = 720,
    .max_batch_size = 4U,
    .interpolation = holoscan::vision::InterpolationMode::kLinear,
});

auto crop = graph.op<holoscan::vision::CustomCropOp>("crop");
crop.set_params(holoscan::vision::CustomCropOpParams{
    .x = 64,
    .y = 32,
    .width = 640,
    .height = 480,
    .max_batch_size = 4U,
});

auto border = graph.op<holoscan::vision::CopyMakeBorderOp>("border");
border.set_params(holoscan::vision::CopyMakeBorderOpParams{
    .max_width = 672,
    .max_height = 512,
    .max_batch_size = 4U,
    .top = 16,
    .bottom = 16,
    .left = 16,
    .right = 16,
});

graph.add_flow(resize->output, crop->input);
graph.add_flow(crop->output, border->input);
```

Python uses the module-owned `holoscan_vision` package. Its `operators`
submodule exposes callable, schema-derived proxies with typed handles and
keyword arguments:

```python
import holoscan as hs
from holoscan_vision import operators as vision_ops

graph = hs.Graph("vision")
resize = vision_ops.ResizeOp(
    graph,
    "resize",
    max_batch_size=4,
    width=1280,
    height=720,
    interpolation="linear",
)

bayer = vision_ops.BayerDemosaicOp(
    graph,
    "bayer",
    bayer_grid_pos="rggb",
    generate_alpha=True,
    alpha_value=255,
)
```

`bayer_grid_pos` accepts the stable names `"bggr"`, `"rggb"`, `"gbrg"`, and
`"grbg"`, or their numeric values 0 through 3. The operator reconstructs the
CUDA-resident `holoscan.schema.Image` input through NPP and publishes an RGB or
RGBA Image without a host staging copy.

The Python operator metadata and generated documentation expose `input` and
`output` as `holoscan.schema.Image`. `RemapOp.map` and all
`NonMaximumSuppressionOp` ports remain `holoscan.Tensor`. The Image identity is
owned and registered by Sensor I/O; Vision consumes that row and does not define
a competing Python value or codec.

The Python package is built and installed when the SDK is configured with both
`HOLOSCAN_BUILD_PYTHON=ON` and `HOLOSCAN_BUILD_VISION=ON`.

Every runtime-pool CUDA tensor producer needs an explicit deployment binding,
including all three Vision image outputs below. This is a general Holoscan 5
rule: omitting a binding fails compilation because the compiler does not infer
device zero.

```cpp
holoscan::CompileOptions options;
options.deployment
    .bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "resize",
        .output_port = "output",
        .device = holoscan::DeviceId{0},
    })
    .bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "crop",
        .output_port = "output",
        .device = holoscan::DeviceId{0},
    })
    .bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "border",
        .output_port = "output",
        .device = holoscan::DeviceId{0},
    });
const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
```

The complete two-branch image-processing and detection graph is in
[`examples/vision_pipeline/cpp/vision_pipeline.cpp`](examples/vision_pipeline/cpp/vision_pipeline.cpp).

## Build and test

The module uses SDK-level dependencies, linting, tests, containers, and
packaging; it has no module-local runner or container image.

```bash
./run build --config \
  "-DHOLOSCAN_BUILD_VISION=ON -DHOLOSCAN_BUILD_EXAMPLES=ON -DHOLOSCAN_BUILD_TESTS=ON"
./run test --options "-L vision"
./run lint
```

The SDK development image installs CV-CUDA and NPP development and runtime
files for shared builds. GPU tests return a CTest skip
only when no usable CUDA device is present.

## Installed CMake package

```cmake
find_package(holoscan COMPONENTS vision REQUIRED)
target_link_libraries(my_pipeline PRIVATE holoscan::vision)
```

Standalone consumers can instead use:

```cmake
find_package(holoscan-vision REQUIRED)
target_link_libraries(my_pipeline PRIVATE holoscan::vision)
```

Configure with `-Dholoscan-vision_DIR=<prefix>/lib/cmake/holoscan` when the package is outside
CMake's default search locations.

`HOLOSCAN_BUILD_VISION` defaults to `ON`. The granular CPack Debian path packages
Vision as the `holoscan-vision` payload, which depends on the matching
`holoscan-core` version. Set `HOLOSCAN_BUILD_VISION=OFF` for a source build without
the module.

Holoscan Vision uses CV-CUDA internally without exposing it in public headers.
Published Vision Debian packages bundle the Vision shared library and the
required CV-CUDA runtime libraries. To develop with CV-CUDA directly, install
the matching Debian packages for your architecture and CUDA version from the
[CV-CUDA release assets](https://github.com/CVCUDA/CV-CUDA/releases).

Bayer demosaic also requires NPP, declared as a dependency of the Vision Debian
package and Python wheel.

See the [developer packaging notes](DEVELOPER.md#packaging) for source-build options.

## License

Apache-2.0; see the repository [LICENSE.txt](../../LICENSE.txt).
