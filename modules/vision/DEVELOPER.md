# Holoscan Vision developer guide

Holoscan Vision keeps the recognizable Holoscan module layout while using the
Holoscan SDK's top-level development workflow.

```text
vision/
├── examples/vision_pipeline/cpp/      # HSDK 5 example graph
├── cmake/                              # installed package configuration template
├── include/holoscan/vision/            # installed public API
├── src/
│   └── detail/                         # private zero-copy tensor adapter
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
  "cmake --build build-arm64 --target holoscan_vision_tests vision_pipeline -j && \
   ctest --test-dir build-arm64 --output-on-failure -L vision"
```

## Operator invariants

- Public headers expose only HSDK types; CV-CUDA types stay in `.cpp` files.
- Port names are stable public API and describe their payload (`input`,
  `output`, `map`, `boxes`, `scores`, or `mask`).
- `VisionImageCapacity` and `NMSCapacity` are immutable plan structure. Image
  shape parameters may be smaller but cannot exceed the corresponding
  capacity; batch admission always uses the structural capacity.
- Image inputs are CUDA HWC/NHWC tensors with pitch-linear interleaved storage;
  `ReformatOp` additionally supports CHW/NCHW and NMS uses its documented
  detection tensor layouts.
- Every tensor byte stride and total strided byte span must fit CV-CUDA's
  32-bit kernel addressing.
- With the supported CV-CUDA 0.17.x line, the adapter passes the original base
  pointer, shape, dtype, and byte strides to `nvcv::TensorWrapData` without a
  cleanup callback. The operators in this module copy wrapper metadata during
  submission and retain no per-submit device workspace. These are
  version-specific implementation properties, not CV-CUDA API guarantees.
- Retain every CV-CUDA operation handle through `Operator::stop()`. Runtime
  performs its terminal CUDA-stream quiescence after `stop()`, so the handle is
  destroyed later with the operator. `start()` must reuse a retained handle on
  a lifecycle restart instead of replacing it.
- Operators submit on `ExecutionContext::cuda_stream()`, commit the output loan,
  and emit without a host synchronization.
- Before adding an operator or changing the supported CV-CUDA line, recheck
  wrapper cleanup behavior, when metadata is copied, and whether submission
  retains asynchronous metadata or workspace. Extend the ownership strategy if
  any of those assumptions changes.
- New operator behavior needs an adapter/unit test and, when it touches device
  execution, a CUDA-gated graph test.

## Packaging

`HOLOSCAN_BUILD_VISION` controls the module and defaults to `ON`. Installation exports
`holoscan::vision` through `holoscan-vision-config.cmake` and assigns files to
the `holoscan-vision` CPack component. The current distribution intentionally
combines all enabled components in the `holoscan` Debian package and installs
CV-CUDA in the common SDK development image. Consequently, default packages
and images include the Vision dependency even when an application does not use
Vision. Set `HOLOSCAN_BUILD_VISION=OFF` for a source build that omits the module;
separate Vision packages or image layers are not currently produced.

Shared packages declare `libcvcuda0` in the supported CV-CUDA version window.
Static packages declare the matching `cvcuda0-dev` window and discover the
corresponding link-only `cvcuda` and `nvcv_types` targets in the installed
config.
