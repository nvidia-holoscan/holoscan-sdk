# Holoscan Vision

Holoscan Vision is a module in Holoscan SDK 5. It provides pipeline-native
wrappers for CV-CUDA operations:

The backend-derived names `CustomCropOp`, `CopyMakeBorderOp`, `ConvertToOp`,
and `ReformatOp` are deliberate: they keep the public operators discoverable
against the corresponding CV-CUDA documentation.

| Holoscan operator | Ports | Configuration |
| --- | --- | --- |
| `ResizeOp` | `input` → `output` | Width, height, interpolation |
| `CustomCropOp` | `input` → `output` | Crop rectangle |
| `CopyMakeBorderOp` | `input` → `output` | Border widths, mode, value |
| `RemapOp` | `input` + `map` → `output` | Size, interpolation, map mode, map synchronization/age, border |
| `NormalizeOp` | `input` → `output` | Base, scale, global scale, shift, epsilon |
| `ConvertToOp` | `input` → `output` | Output dtype, alpha, beta, rounding |
| `ReformatOp` | `input` → `output` | Interleaved/planar direction |
| `NonMaximumSuppressionOp` | `boxes` + `scores` → `mask` | Score/IoU thresholds |

The operators use `nvcv::TensorWrapData` around the existing Holoscan input
tensor and the plan-owned output tensor loan. They do not create
`cvcuda::Image`, stage tensor contents, copy between tensor APIs, or synchronize
the host. CV-CUDA work is submitted on the operator's HSDK runtime CUDA stream.

## Supported tensors

- Image operators accept rank 3 `HWC` and rank 4 `NHWC`; `ReformatOp` also
  accepts and produces `CHW` and `NCHW`.
- After `ReformatOp` produces `CHW` or `NCHW`, the next Holoscan Vision image
  operator must be `ReformatOp` configured to convert back to `HWC` or `NHWC`;
  otherwise, the tensor must leave the Vision module.
- All dtypes in the table are scalar, byte-addressable DLPack dtypes.

| Operator input | Channels | Accepted dtypes |
| --- | --- | --- |
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

Outputs retain capture time, frame identity, trace identity, and sample flags.
Each operator documents its shape, dtype, and layout transformation in its
public header.

Unary operators consume each input independently. By default, `RemapOp` consumes
each image with the newest causal map and can reuse that map for later images up
to `map_max_age`; its default maximum age permits run-lifetime reuse. Set
`map_synchronization` to `kExact` to require identical image and map capture
times. `NonMaximumSuppressionOp` always requires identical capture times for its
boxes and scores. Both operators forward metadata from their primary input.

## Authoring a graph

Output pools are finite plan resources. Vision operators are default-constructible, and their typed
Parameters contain the structural fields used to derive pool capacity. Fixed-output operators use
their output dimensions directly. `NormalizeOp` and `ReformatOp` inherit the connected input byte
bound, and `ConvertToOp` scales that bound for its selected output dtype. `CopyMakeBorderOp` uses
explicit maximum output dimensions, and NMS uses an explicit maximum box count. Complete
configuration loading requires these capacity-only maxima and every `max_batch_size`; direct C++
aggregate setup should likewise set them to the application's real bound instead of relying on the
single-sample defaults. Batch capacities cannot exceed `holoscan::vision::kMaximumBatchSize`
(65,535).

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

Every runtime-pool CUDA Tensor producer needs an explicit deployment binding,
including both Vision outputs. This is a general Holoscan 5 rule: omitting a
binding fails compilation because the compiler does not infer device zero.

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
./public/run build --config \
  "-DHOLOSCAN_BUILD_VISION=ON -DHOLOSCAN_BUILD_EXAMPLES=ON -DHOLOSCAN_BUILD_TESTS=ON"
./public/run test --options "-L vision"
./public/run lint
```

The SDK development image installs the CV-CUDA development and runtime
packages selected by the build configuration. GPU tests return a CTest skip
only when no usable CUDA device is present.

## Installed CMake package

```cmake
find_package(holoscan CONFIG REQUIRED COMPONENTS vision)
target_link_libraries(my_pipeline PRIVATE holoscan::vision)
```

Standalone consumers can instead use:

```cmake
find_package(holoscan-vision CONFIG REQUIRED)
target_link_libraries(my_pipeline PRIVATE holoscan::vision)
```

Configure with `-Dholoscan-vision_DIR=<prefix>/lib/cmake/holoscan` when the package is outside
CMake's default search locations.

`HOLOSCAN_BUILD_VISION` defaults to `ON`. The granular CPack Debian path packages
Vision as the `holoscan-vision` payload, which depends on the matching
`holoscan-core` version. Set `HOLOSCAN_BUILD_VISION=OFF` for a source build without
the module.

Shared Debian packages bundle the matching CV-CUDA 0.18.x runtime, so they do
not require a separately configured CV-CUDA APT repository. Static packages
declare the matching development window. Public headers intentionally contain
no CV-CUDA types.

## License

Apache-2.0; see the repository [LICENSE.txt](../../LICENSE.txt).
