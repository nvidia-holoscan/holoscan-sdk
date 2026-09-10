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

Output pools are finite plan resources. Image operators take a named
`VisionImageCapacity`; NMS takes `NMSCapacity`. Typed parameters select active
geometry within that structural capacity. HSDK rejects a parameter override
that exceeds the authored capacity before compiling the graph. Batch capacities are limited to
`holoscan::vision::kMaximumBatchSize` (65,535).

```cpp
#include <holoscan/vision/custom_crop_op.hpp>
#include <holoscan/vision/resize_op.hpp>

holoscan::Graph graph{"vision"};

auto resize = graph.op<holoscan::vision::ResizeOp>(
    "resize", holoscan::vision::VisionImageCapacity{
                  .max_width = 1920,
                  .max_height = 1080,
                  .max_batch_size = 4U,
              });
resize.set_params(holoscan::vision::ResizeOpParams{
    .width = 1280,
    .height = 720,
    .interpolation = holoscan::vision::InterpolationMode::kLinear,
});

auto crop = graph.op<holoscan::vision::CustomCropOp>(
    "crop", holoscan::vision::VisionImageCapacity{
                .max_width = 640,
                .max_height = 480,
                .max_batch_size = 4U,
            });
crop.set_params(holoscan::vision::CustomCropOpParams{
    .x = 64,
    .y = 32,
    .width = 640,
    .height = 480,
});

graph.add_flow(resize->output, crop->input);
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
find_package(holoscan-vision CONFIG REQUIRED)
target_link_libraries(my_pipeline PRIVATE holoscan::vision)
```

`HOLOSCAN_BUILD_VISION` defaults to `ON`. The current distribution deliberately
includes the `holoscan-vision` CPack component in the single `holoscan` Debian
package and installs CV-CUDA in the common SDK development image. Therefore
default SDK packages and images carry the Vision dependency. Set
`HOLOSCAN_BUILD_VISION=OFF` for a source build without the module; no separate
Vision package or image layer is produced today.

Shared packages declare the supported CV-CUDA 0.17.x runtime window; static
packages declare the matching development window. Public headers intentionally
contain no CV-CUDA types.

## License

Apache-2.0; see the repository [LICENSE.txt](../../LICENSE.txt).
