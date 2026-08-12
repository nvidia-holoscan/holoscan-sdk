# HoloMQ

HoloMQ is a C++ library for zero-copy GPU IPC over publish–subscribe messaging.
It can be built as part of the Holoscan SDK or as a standalone CMake package
(`holomq::holomq`). Implementation sources live under `src/`; application API
details are in [`docs/PUBLIC_API.md`](docs/PUBLIC_API.md).

In SDK builds, third-party dependencies are wired through `cmake/deps/` and the shared
dependency helper in `cmake/setup_dependencies.cmake`. HoloMQ uses the **iceoryx2** messaging backend,
resolved at configure time: CMake searches for a prebuilt prefix first, then falls back to
RAPIDS CPM clone/patch/build (requires `cargo`).

## Build With Holoscan SDK

HoloMQ is built with Holoscan Core by default. It keeps its own package target,
`holomq::holomq`, even when built from the SDK tree.

By default, the HoloMQ package artifacts install to the same prefix as the SDK.
Use `HOLOMQ_INSTALL_PREFIX` or `--holomq-installprefix` when you want the
standalone HoloMQ package installed to a separate prefix:

```bash
./run build \
  --holomq-installprefix install-holomq
```

SDK builds use iceoryx2 (`HOLOSCAN_MQ_BACKEND=iceoryx2` / `HOLOMQ_DEFAULT_BACKEND=iceoryx2`,
mapped to `HOLOMQ_TRANSPORT=iox2` at configure time).

HoloMQ first searches for a prebuilt `iceoryx2-cxx` install prefix (`HOLOMQ_ICEORYX2_PREFIX`, `_deps/iceoryx2-install`,
or `/opt/holomq/iceoryx2-install`). If none is found, CMake clones the pinned
upstream commit, applies the ordered patch series under
`cmake/deps/patches/iceoryx2/`, and builds from source (requires a Rust
`cargo` toolchain).

From the SDK root:

```bash
./run build --config "-DHOLOSCAN_MQ_BACKEND=iceoryx2"
```

The upstream pin and ordered patch series live under `cmake/deps/`
(`iceoryx2.lock` and `patches/iceoryx2/`).

Holoscan Core links HoloMQ privately, but HoloMQ remains a standalone package.
SDK users who need HoloMQ APIs should still consume `holomq::holomq` and the
public headers described in [`docs/PUBLIC_API.md`](docs/PUBLIC_API.md).

## Build Standalone

The module does not carry its own `run` wrapper or Dockerfile. Use raw CMake for
standalone host builds, and use the top-level SDK `public/run` workflow for
containerized or SDK-integrated builds.

Raw CMake workflow:

```bash
cmake -S public/modules/holomq -B /tmp/holomq-build -G Ninja \
  -DCMAKE_INSTALL_PREFIX=/tmp/holomq-install \
  -DHOLOMQ_DEFAULT_BACKEND=iceoryx2
cmake --build /tmp/holomq-build --target holomq
cmake --build /tmp/holomq-build \
  --target holomq-check-export-macro-semantics holomq-check-symbols
ctest --test-dir /tmp/holomq-build --output-on-failure
cmake --install /tmp/holomq-build
```

Standalone HoloMQ builds shared libraries by default and installs
`libholomq.so`. Use `-DBUILD_SHARED_LIBS=OFF` only when a static library is
explicitly needed. HoloMQ requires the CUDA Toolkit (`find_package(CUDAToolkit)`)
and links `CUDA::cuda_driver`. The `holomq-check-symbols` target is a
Linux/GNU shared-build regression check and is intentionally absent from
static and unsupported toolchain configurations.

The `iceoryx2-cxx` dependency first searches for a prebuilt install prefix. Set `HOLOMQ_ICEORYX2_PREFIX` when using a
custom install location; otherwise HoloMQ also checks
`public/modules/holomq/_deps/iceoryx2-install` and
`/opt/holomq/iceoryx2-install`. If no prebuilt package is found, CMake clones
the pinned upstream iceoryx2 commit, applies
the ordered patch series under `cmake/deps/patches/iceoryx2/`, and builds it from source,
which requires a Rust `cargo` toolchain:

```bash
cmake -S public/modules/holomq -B /tmp/holomq-build -G Ninja \
  -DHOLOMQ_DEFAULT_BACKEND=iceoryx2 \
  -DHOLOMQ_ICEORYX2_PREFIX=/opt/holomq/iceoryx2-install
```

Library diagnostics use `<holomq/logger.hpp>` (`HOLOMQ_LOG_*`, optional
`set_log_handler`, and IOX2 transport logging via `enable_transport_logging()` or
`ContextBuilder::with_transport_logging(true)`). See [`docs/PUBLIC_API.md`](docs/PUBLIC_API.md#logging).

## SDK Container Workflows

Run SDK container workflows from `public/` with the top-level `run` script:

```bash
cd public
./run build --config "-DHOLOMQ_BUILD_EXAMPLES=ON"
./run test --name fd_pass_test
./run test --name cuda_pool_test
./run launch --run-cmd "cmake --build . --target holomq_main"
```

`public/run launch` starts the SDK build container with the same Dockerfile and
runtime settings used by the broader project, including NVIDIA runtime and
shared-memory settings needed by HoloMQ IPC tests.

### Examples and tests

Upstream examples include `holomq_main` (pub/sub demo). Unit tests cover
FD-pass, slot regions, WaitSet, CUDA pools, host-pinned pools, and VMM import.
Enable examples with `-DHOLOMQ_BUILD_EXAMPLES=ON`.

```bash
cd public
./run test --options "-N"
./run test --name fd_pass_test
./run test --name slot_region_test
./run test --name wait_set_test
./run test --name cuda_host_pool_test
./run test --name cuda_pool_test
./run test --name cuda_vmm_import_test
```

For standalone host builds, use `ctest --test-dir <build-dir> --output-on-failure`
or build and run the `holomq_main` executable from the chosen CMake build tree.

## Consume The Package

Standalone consumers can use the installed package config:

```cmake
find_package(holomq CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE holomq::holomq)
```

Application headers are exposed through CMake as part of the `holomq` target
(see [`docs/PUBLIC_API.md`](docs/PUBLIC_API.md) for the supported entry points
such as `<holomq/context.hpp>`). The standalone package intentionally does not provide a
`holoscan::ipc` compatibility target. Holoscan SDK code should consume HoloMQ
through `holomq::holomq` when SDK targets need IPC.

## Contributor Policy

HoloMQ public headers use `#pragma once` and natural include paths for the
supported API surface. Public DSO ABI declarations use the generated header
`<holomq/holomq_export.h>` and `HOLOMQ_EXPORT`; do not recreate or include
`holomq/export.hpp`.

Any `detail/` path segment is internal and not part of the HoloMQ ABI. See
[`docs/README.md`](docs/README.md), [`docs/PUBLIC_API.md`](docs/PUBLIC_API.md),
[`docs/abi_and_exports.md`](docs/abi_and_exports.md), and
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the standalone contributor workflow and
checks.

## Runtime Requirements

iceoryx2 requires shared-memory resources such as `/dev/shm`. SDK container
workflows use `--ipc=host` by default. Clean up iceoryx2 runtime artifacts after
crashes when tests fail to tear down cleanly.
