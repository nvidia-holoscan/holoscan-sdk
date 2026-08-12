# HoloMQ ABI, Visibility, and Export Policy

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Contributor policy for HoloMQ headers, visibility, and shared-library exports.
It is complete on its own and does not require reading Holoscan Core
documentation first.

Application-facing API names, types, and call patterns are in
[`PUBLIC_API.md`](PUBLIC_API.md). For codebase navigation, see
[`CONTRIBUTOR_READING_ORDER.md`](CONTRIBUTOR_READING_ORDER.md).

## Header Guards

Every hand-written HoloMQ C++ header uses `#pragma once`. Do not add macro
include guards. The generated `holomq_export.h` header is the only exception
because CMake owns that file.

## Public Includes

Application headers live under `include/holomq/` (installed as `<holomq/...>`;
see [`PUBLIC_API.md`](PUBLIC_API.md)):

```cpp
#include <holomq/context.hpp>
#include <holomq/buffer_pool.hpp>
```

Use angle-bracket includes with the `holomq/` prefix. Do not require
application code to include internal paths such as `include/holomq/detail/`.
The build adds the generated `holomq_export.h` from the build directory to the
include path.

Any path segment named `detail` is internal, such as
`include/holomq/detail/...`. `detail/` headers are not ABI, must not use
`HOLOMQ_EXPORT`, and must not be included by applications. `detail/` files may
change without warning between releases.

## Generated export header (`holomq_export.h`)

This file is **not checked into the repository**. CMake creates it at configure
time via `generate_export_header()` and writes it to the build tree at
`include/holomq/holomq_export.h` (under the module build directory). Installed
packages ship the same path as `<holomq/holomq_export.h>`.

```233:241:public/modules/holomq/CMakeLists.txt
generate_export_header(holomq
    BASE_NAME HOLOMQ
    EXPORT_MACRO_NAME HOLOMQ_EXPORT
    NO_EXPORT_MACRO_NAME HOLOMQ_NO_EXPORT
    DEPRECATED_MACRO_NAME HOLOMQ_DEPRECATED
    STATIC_DEFINE HOLOMQ_STATIC_DEFINE
    INCLUDE_GUARD_NAME HOLOMQ_EXPORT_H_GENERATED
    EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/include/holomq/holomq_export.h"
)
```

On a typical Linux shared-library build, the generated header defines:

| Macro | Role |
|-------|------|
| `HOLOMQ_EXPORT` | Marks symbols exported from (or imported from) the `holomq` shared library |
| `HOLOMQ_NO_EXPORT` | Explicit opt-out of export |
| `HOLOMQ_DEPRECATED` | Deprecation attribute wrapper |
| `HOLOMQ_DEPRECATED_EXPORT` | `HOLOMQ_EXPORT` + `HOLOMQ_DEPRECATED` |
| `HOLOMQ_DEPRECATED_NO_EXPORT` | `HOLOMQ_NO_EXPORT` + `HOLOMQ_DEPRECATED` |
| `HOLOMQ_NO_DEPRECATED` | Suppresses deprecation warnings when defined before include |

When `HOLOMQ_STATIC_DEFINE` is set, `HOLOMQ_EXPORT` and `HOLOMQ_NO_EXPORT`
expand to nothing (see **Export macro usage** below).

The generated file uses a macro include guard (`HOLOMQ_EXPORT_H_GENERATED`), not
`#pragma once` — that is the only header guard exception in the tree.

## Export macro usage

HoloMQ uses a generated export header for out-of-line symbols that belong in
the shared-library ABI:

```cpp
#include <holomq/holomq_export.h>

HOLOMQ_EXPORT void example_public_function();
```

Do not create or include `include/holomq/export.hpp`. Do not hand-write
`HOLOMQ_EXPORT`, `HOLOMQ_NO_EXPORT`, or `HOLOMQ_DEPRECATED`; CMake generates
them with `GenerateExportHeader`.

Static builds (`BUILD_SHARED_LIBS=OFF`) define `HOLOMQ_STATIC_DEFINE` on the
`holomq` target so `HOLOMQ_EXPORT` / `HOLOMQ_NO_EXPORT` expand to nothing and
symbols do not leak from a consumer shared object.

Public headers live under `include/holomq/`; `.cpp` implementations live under
`src/` with hidden visibility by default. Most API types are declared in headers
and implemented out-of-line in `src/` (often via pimpl); only symbols explicitly
marked `HOLOMQ_EXPORT` are part of the shared-library ABI. Today that is the
logging API in `include/holomq/logger.hpp` (implemented in `src/logger.cpp`).
Add `HOLOMQ_EXPORT` when promoting new out-of-line entry points to the stable
DSO surface.

## Visibility and ABI

The `holomq` target builds with hidden default visibility and explicitly exports
only the curated public DSO ABI. Use `HOLOMQ_EXPORT` for out-of-line public
functions, public classes whose type identity crosses the shared-library
boundary, and exception types intended to cross the boundary.

Do not blanket-export inline helpers, implementation types, primary templates, or
headers under `detail/`. If a public polymorphic class is added, give it an
out-of-line exported key function so RTTI and vtables have one owner. Template
exports require an explicit instantiation policy.

During EA the exported DSO surface is expected to change frequently. The
module-owned `abi/holomq.exports.baseline.txt` check is a **regression guard** against
accidental export leaks, not a promise of binary compatibility across Holoscan
or HoloMQ major releases. Downstream applications should link `libholomq.so`,
include only public `<holomq/...>` headers, and rebuild when upgrading SDK
packages. Semver and ABI stability expectations will be documented once the DSO
surface stabilizes post-EA.

Public option aggregates and enums in `<holomq/...>` headers are part of the source-facing EA API
even when they add no out-of-line DSO symbol. Adding an endpoint-local field such as
`PublisherPubSubOptions::pressure_policy` therefore requires a matching public API update and
installed-consumer rebuild evidence. It does not require a symbol-baseline update unless the
curated `nm -D` inventory changes. Pre-1.0 consumers must rebuild against the matching headers and
library rather than relying on aggregate layout compatibility between SDK packages.

## Shipped library type

When `BUILD_SHARED_LIBS=ON` (the default for SDK and standalone HoloMQ builds),
application authors link the shared `libholomq.so` target (`holomq::holomq`).
Holoscan 5.x binary artifacts ship `libholomq.so`. The static archive
(`holomq_static` / `holomq::holomq_internals`) exists only for in-tree **unit
tests** that white-box `src/` implementation headers; it is not the distributed
consumer library. In-tree examples link `holomq::holomq` and use the public API
only. Set `BUILD_SHARED_LIBS=OFF` for all-static embedding scenarios.

HoloMQ owns its package version through `VERSION`. Embedded SDK builds must not
replace that version with the Holoscan SDK version. `SOVERSION` tracks the major
component of `VERSION` (currently `0` for `0.1.0` while pre-1.0).

Holoscan Core links HoloMQ privately for implementation use in SDK builds. That
does not move HoloMQ into the Core public ABI: HoloMQ public headers are under
`include/holomq/`, and direct HoloMQ users should continue linking
`holomq::holomq`.

In-tree **unit tests** that white-box `src/` implementation headers link
`holomq::holomq_internals` (a static archive built from the same object files)
when `BUILD_SHARED_LIBS=ON`. That is not a shared-library ABI defect for
application authors: tests need symbols and types hidden from the public DSO
surface. In-tree examples link `holomq::holomq` and include only public
`<holomq/...>` headers. External applications should use the installed
`holomq::holomq` shared target and the documented public API only. The CMake
package name remains `holomq::` for EA1 standalone `find_package(holomq)`
compatibility; a `holoscan::holomq` alias may be added when HoloMQ is fully
integrated into the SDK export set.

## Local Checks

From `public`:

```sh
./run lint
./run build --config "-DHOLOMQ_BUILD_TESTS=ON"
```

Or run pre-commit directly from the repository root:

```sh
pre-commit run --config .pre-commit-config.yaml --all-files
```

From a CMake build tree (requires `compile_commands.json`), run the heavier
export checks. The symbol-baseline target is provided for the reference
Linux/GNU shared-library configuration in both embedded SDK and standalone
HoloMQ builds:

```sh
cmake --build <build-dir> --target holomq-check-export-macro-semantics
cmake --build <build-dir> --target holomq-check-symbols
```

The hooks reject macro include guards, missing `#pragma once`, hand-written
export macros, old `holomq/export.hpp` includes, and export usage in `detail/`.
The symbol check compares demangled `nm -D` output against the module-owned
`abi/holomq.exports.baseline.txt`. Keeping the checker and baseline inside the
HoloMQ source tree makes the same target available in embedded SDK builds,
isolated installed-source builds, and standalone HoloMQ builds. During EA the
public DSO surface is expected to change; the baseline is a regression guard
against accidental export drift, not a stability contract. Regenerate the
baseline from a shared library build when the exported DSO surface changes
intentionally:

```sh
holomq_root=<path-to-holomq-source>
library=<build-dir>/libholomq.so
export LC_ALL=C
nm -D --defined-only --extern-only "${library}" \
  | awk '{print $3}' | c++filt | sort -u > "${holomq_root}/abi/holomq.exports.baseline.txt"
```
