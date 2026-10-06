# Holoscan Vision developer guide

Holoscan Vision keeps the recognizable Holoscan module layout while using the
Holoscan SDK's top-level development workflow.

```text
vision/
├── examples/vision_pipeline/cpp/      # HSDK 5 example graph
├── cmake/                              # installed package configuration template
├── include/holoscan/vision/            # installed public API
├── src/
│   └── detail/                         # private tensor adapter and image metadata rules
└── tests/
    ├── cpp/                            # unit and CUDA pipeline tests
    └── installed_consumer/             # installed CMake package smoke test
```

There is deliberately no module-local Dockerfile, CLI wrapper, CI setup, or
lint configuration. Those responsibilities belong to HSDK.

## Build workflow

From the repository root:

```bash
./public/run build --config \
  "-DHOLOSCAN_BUILD_VISION=ON -DHOLOSCAN_BUILD_EXAMPLES=ON -DHOLOSCAN_BUILD_TESTS=ON"
./public/run test --options "-L vision" --verbose
./public/run lint
```

For a focused iteration in an existing SDK build tree:

```bash
./public/run launch . --run-cmd \
  "cmake --build build-arm64 \
   --target holoscan_vision_tests holoscan_vision_nvcv_warmup_tests vision_pipeline -j 5 && \
   ctest --test-dir build-arm64 --output-on-failure -L vision"
```

## Operator invariants

- Public headers expose only HSDK types; CV-CUDA types stay in `.cpp` files.
- Port names are stable public API and describe their payload (`input`,
  `output`, `map`, `boxes`, `scores`, or `mask`).
- Structural fields in each operator's typed Parameters determine its immutable
  plan capacity. Fixed-output operators use their output dimensions directly;
  shape-preserving operators inherit the connected input byte bound, with
  `ConvertToOp` additionally scaling for its output dtype. Only genuinely
  capacity-only dimensions remain explicit. Complete parameter loading requires
  those maxima and each `max_batch_size`.
- The seven image operators expose `Input<schema::ImageT>` and
  `Output<schema::ImageT>` directly, without payload templates or Tensor
  compatibility variants. `Image.data` contains a CUDA HWC/NHWC tensor with
  scalar, pitch-linear interleaved storage. `ReformatOp` additionally supports
  CHW/NCHW. `RemapOp::map` and NMS ports remain `Tensor` ports.
- Validate image width/height, encoding, and any supplied plane descriptors
  against tensor dtype, channels, layout, and strides. Compatible scalar
  `CUSTOM` images are supported. Reject packed and chroma-subsampled encodings
  explicitly; do not reinterpret their storage.
- Every tensor byte stride and total strided byte span must fit CV-CUDA's
  32-bit kernel addressing.
- Read Image inputs with `selected_sample()`, never an owning generated descriptor
  copy in the compute hot path. Validate and describe input pixels from the borrow.
  Keep the view, its references, and all nested pointees immutable and callback-local.
- Each image transform cold-prepares `sensor_io::ImageDescriptorScratch` in its
  private `Impl`. Initialize missing scratch before the operation-ready early return
  in `start()`, so partial initialization can be retried. Reuse prepared state across
  idempotent starts and restarts. Copy metadata field-wise into scratch, preserving
  its owners and capacities; never assign an Image into it or populate its `data`.
  Source Tensor ownership must not remain in an operator member after compute.
- With the supported CV-CUDA 0.18.x line, the adapter passes the original base
  pointer, shape, dtype, and byte strides to `nvcv::TensorWrapData` without a
  cleanup callback. The operators in this module copy wrapper metadata during
  submission and retain no per-submit device workspace. These are
  version-specific implementation properties, not CV-CUDA API guarantees.
- Retain every CV-CUDA operation handle through `Operator::stop()`. Runtime
  performs its terminal CUDA-stream quiescence after `stop()`, so the handle is
  destroyed later with the operator. `start()` must reuse a retained handle on
  a lifecycle restart instead of replacing it.
- Before creating that operation handle, cold-prepare two simultaneous NVCV Tensor handles
  (three for Remap and NMS). The private helper constructs and destroys metadata-only wrappers
  over a temporary host placeholder; NVCV 0.18 copies/validates these descriptors without
  dereferencing the address or querying CUDA. Never submit that placeholder to a GPU operation.
  This version-specific warmup performs no device allocation, retains no handle/backing, and
  changes no process-global pool policy. Preparation failure is retryable; an application-imposed
  insufficient handle limit fails startup with `kResourceExhausted`. Successful idempotent starts
  skip this preparation along with operation creation. Revalidate the metadata-only assumption
  on an NVCV upgrade.
- Operators submit on `ExecutionContext::cuda_stream()` and commit the output
  loan without a host synchronization. Image operators publish the updated
  descriptor with `output.emit_tensor`, attaching the runtime-pool loan without
  copying pixels.
  Runtime's Tensor read leases protect asynchronous pixel reads after compute returns;
  the host descriptor borrow does not. Unregistered side streams are not supported.
- Before adding an operator or changing the supported CV-CUDA line, recheck
  wrapper cleanup behavior, when metadata is copied, and whether submission
  retains asynchronous metadata or workspace. Extend the ownership strategy if
  any of those assumptions changes.
- New operator behavior needs an adapter/unit test and, when it touches device
  execution, a CUDA-gated graph test.

## Shared image metadata rules

`src/detail/image_metadata.hpp` defines the backend-independent
`ImageTensorView` and metadata transformation contract. It imports no CV-CUDA
or other backend headers, so future NPP adapters can use the same validation and
metadata rules. `ImageGeometryChange` describes preserve, crop, pad, resize,
and warp effects; `values_changed` and `preserves_precision` compose numeric
and sampling effects independently. Backend adapters supply actual tensor
shape, dtype, and byte strides instead of duplicating descriptor policy.

Keep the image header, exposure, and gain, along with sample capture time,
frame identity, trace identity, and flags. Rebuild width, height, encoding,
and planes from the output view. For batches, planes describe the first image;
the tensor batch stride locates each subsequent image.

Include `<holoscan/sensor_io/provenance.hpp>` and use
`sensor_io::derive_from(sample->metadata())` for a checked scalar borrow. The updated
descriptor carries image metadata; the emit options carry sample provenance:

```cpp
auto sample = input.selected_sample();
if (!sample) return make_unexpected(sample.error());
// Describe input pixels from sample->data(), not from metadata-only scratch.
auto copied = impl_->image_metadata->copy_metadata_from(sample->data());
if (!copied) return make_unexpected(copied.error());
auto& image = impl_->image_metadata->image();
// Update image metadata, fill the output loan on the admitted stream, and commit.
return output.emit_tensor(std::move(*loan), image, sensor_io::derive_from(sample->metadata()));
```

Scratch reserves four planes and a 512-byte frame ID; it preserves optional Header/Time
presence and rejects oversize copies before mutation. Complete wire-size validation
still happens at publication. Unchanged descriptors can bypass scratch and publish
`sample->data()` directly. See the SDK's
[compiled borrowed Image transform](../../examples/borrowed_image_transform/README.md)
for a self-contained authoring example. The module's Image sink also uses a read-only
borrow; Remap maps and NMS Tensor receives retain their existing ownership path.
This removes descriptor-copy allocations, not necessarily CV-CUDA/NVCV or transport
allocations; whole-operator performance requires separate measurement.

### Descriptor allocation qualification

The maintained [Core/Sensor I/O benchmarks](../../benchmark/runtime/README.md) retain owning,
borrowed, and prepared metadata-edit cases, with collection commands and counter boundaries.
The receive benchmark isolates descriptor access; the production graph benchmark separates
receive, metadata-copy, and publication allocations. These measure host descriptor work,
**not Vision callback or GPU timings**. Save generated JSON outside the SDK source tree, together
with the build/host provenance and interpretation.

`VisionImageMetadata.PreparedScratchEditsDoNotAllocateOnFirstOrRepeatedUse` adds a small
Vision-specific regression in the existing unit target: it counts C++ allocations around
scratch copying and the production Resize/Reformat metadata helpers on the first and 63 repeated
edits, including 512-byte frame IDs and three/four-plane output. It requires zero allocations,
but does not instrument receive, publication, native NVCV/CV-CUDA allocation, or a full callback.
The separate `VisionBorrowedImage` tests cover all seven operators' preparation/retry/restart
and a repeated CUDA Resize graph with fanout, provenance, and source immutability checks.

Run these checks with the SDK container's existing `holoscan_vision_tests` target and
`ctest -R 'VisionBorrowedImage|VisionImageMetadata.PreparedScratch' --output-on-failure`.
The existing `BENCHMARK_RUNTIME_DESCRIPTOR_RECEIVE` and `BENCHMARK_RUNTIME_DESCRIPTOR_GRAPH`
CTest smokes guard their harnesses; they are not controlled timing collection. No Vision-specific
benchmark target or production allocation instrumentation is needed for this qualification.

This establishes removal of the reviewed owning descriptor-copy cost. It does not establish a
whole-callback zero-allocation property or an end-to-end speedup: CUDA execution can dominate
these sub-microsecond host savings. Profile native wrappers or whole operators separately if a
workload needs that evidence; retain new measurements separately from the historical results.

The dedicated `holoscan_vision_nvcv_warmup_tests` executable keeps native `malloc` interposition
out of `holoscan_vision_tests`. Its hook is live throughout the dedicated process even when the
thread-local counting/failure flags are off. The `VisionNvcvWarmupDeathTest` cases re-execute a
fresh process for each operator,
then require zero first/repeated `malloc` calls when wrapping its two or three Tensors after
`start()`. A glibc-only, thread-local test probe is needed because the distributed NVCV library
statically links its C++ allocator and bypasses the existing `operator new` counter. A cold control
proves that the native probe observes the installed library's allocations; non-glibc skips do not
qualify this behavior. This probe does not intercept other native allocation entry points.
Other cases check cold-allocation failure, partial-handle cleanup, caller-controlled fixed capacity,
and startup retry. Build `holoscan_vision_nvcv_warmup_tests`, then run
`ctest -R VisionNvcvWarmup --output-on-failure`. Build both Vision test targets before running
the complete `ctest -L vision` suite.
This avoids normal cold handle-pool initialization during first compute, but is not an exclusive
reservation: other NVCV users, higher concurrent demand, or later application pool reconfiguration
can still cause growth or failure. CUDA kernel/module and NVTX initialization are not warmed here.

NMS observes both boxes and scores with `sensor_io::ProvenanceAccumulator` and
emits its Tensor with the merged options. Remap treats its selected map as
configuration: preserve image capture time, frame identity, and trace identity,
and union the image and map flags, including when the map is retained.
After updating the descriptor, the metadata helper validates its agreement with
the output view. Backend adapters must still validate tensor geometry, strides,
and schema field representability before calling the helper; schema encoding
also validates the descriptor against the committed output tensor.

- Crop adds the origin to ROI offsets with checked arithmetic and shifts Bayer
  phase according to crop parity.
- Padding subtracts left/top from ROI offsets only if both remain nonnegative;
  otherwise reset both offsets. Any padding invalidates a Bayer encoding,
  yielding `CUSTOM`.
- Resize invalidates ROI and Bayer claims only when spatial dimensions change.
  Warp always resets ROI offsets and converts Bayer encoding to `CUSTOM`.
- Nearest resize preserves significant bits. Nearest image sampling during
  remap also preserves them when borders are zero or copy existing samples.
  Zero or sample-copy padding preserves them as well. Other interpolation and
  nonzero synthetic constant borders mark significant bits unknown. Resize and border
  operations that leave geometry unchanged preserve these claims.
- Unscaled lossless casts retain source `significant_bits`, expanding zero to
  the input container width when the dtype changes. Lossy casts and value-changing
  transforms set `schema::kUnknownImageSignificantBits` (255); zero retains its
  meaning of full container width. Dtype changes or value-changing transforms
  clear `color_space`; identity normalization and identity conversion preserve
  both fields. Dtype conversion chooses a corresponding named encoding where one
  exists and `CUSTOM` otherwise. Geometric operations preserve color space.
- Layout conversion preserves color space and significant bits. RGB8/BGR8 map
  to planar `R8_G8_B8`/`B8_G8_R8`, and those map back to interleaved RGB8/BGR8.
  Unsupported encoding/layout combinations become `CUSTOM`.

Test this helper independently of GPU execution, then cover operator wiring and
pixel behavior in CUDA-gated graph tests. See the public headers and
[image metadata contract](README.md#image-metadata) for operator-specific rules.

## Packaging

`HOLOSCAN_BUILD_VISION` controls the module and defaults to `ON`. Installation exports
`holoscan::vision` through the `vision` component of `holoscan-config.cmake` and
assigns files to the `holoscan-vision` CPack component. The granular CPack Debian
path packages the module separately from `holoscan-core`; set `HOLOSCAN_BUILD_VISION=OFF`
for a source build that omits it.

Published SDK Debian packages use shared SDK libraries and bundle the CV-CUDA
runtime. Developers can instead build the SDK from source with
`-DBUILD_SHARED_LIBS=OFF -DHOLOSCAN_BUILD_PYTHON=OFF` and generate a local static
Debian package.

The static Vision package declares a `cvcuda0-dev` dependency in the supported
version range for downstream development. Manually install the matching
CV-CUDA development and runtime Debian packages for your architecture and CUDA
version from the [CV-CUDA release assets](https://github.com/CVCUDA/CV-CUDA/releases).

Static builds also require NPP development files, including the static archives
that the standard SDK container removes.
