# HoloMQ ABI, Visibility, and Export Policy

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Contributor policy for HoloMQ headers, visibility, and shared-library exports.
HoloMQ is an SDK-owned backend library defined by `cmake/HoloMQ.cmake` and built through
the top-level SDK project. Paths and commands below are relative to the SDK
source root, which contains `run`, `include/`, `src/`, and `ci/`.

Application-facing API names, types, and call patterns are in
[`PUBLIC_API.md`](PUBLIC_API.md). For codebase navigation, see
[`CONTRIBUTOR_READING_ORDER.md`](CONTRIBUTOR_READING_ORDER.md).

**API stability: Evolving.** HoloMQ's direct public API and shared-library ABI
may change between SDK releases without a guaranteed deprecation cycle.
Upgrading may require source changes and rebuilding applications or plugins
that use HoloMQ directly. This classification describes compatibility across
upgrades; testing and release qualification are separate considerations.

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

Headers under `include/holomq/detail/` are implementation details. Public
template headers may include them transitively; applications should include
the supported public headers and avoid depending on `detail` types directly.
Keep `HOLOMQ_EXPORT` out of `detail/` headers. Installed implementation headers
do not acquire a public compatibility guarantee merely by being installed.

## Generated export header (`holomq_export.h`)

This file is **not checked into the repository**. CMake creates it at configure
time via `generate_export_header()` and writes it to the SDK build tree at
`include/holomq/holomq_export.h`. Installed packages ship the same path as `<holomq/holomq_export.h>`.

```cmake
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
`holomq` target and propagate it to consumers through the CMake target.
`HOLOMQ_EXPORT` and `HOLOMQ_NO_EXPORT` then expand to nothing. This disables
the shared-library import/export annotations; the final consumer's visibility
and linker settings determine which static-library symbols it exposes.

Public headers live under `include/holomq/`; `.cpp` implementations live under
`src/holomq/` with hidden visibility by default. Exported out-of-line entry
points cover contexts and builders, buffers and pools, readiness and wait
handles, error helpers, capabilities, and logging. Use `HOLOMQ_EXPORT` on
declarations that must be reachable across the shared-library boundary, and
include their declaring header in the implementation source.

The exported symbol inventory also contains SDK-private implementation bridges
and toolchain-generated symbols. An exported symbol's presence in
`ci/abi/holomq.exports.baseline.txt` does not by itself make it a supported
application API or classify it as Stable.

## Visibility and ABI

The `holomq` target builds with hidden default visibility. Use `HOLOMQ_EXPORT`
to expose out-of-line entry points across the shared-library boundary: public
functions, public classes whose type identity crosses the shared-library
boundary, and exception types intended to cross the boundary.

Do not blanket-export inline helpers, implementation types, primary templates, or
headers under `detail/`. If a public polymorphic class is added, give it an
out-of-line exported key function so RTTI and vtables have one owner. Template
exports require an explicit instantiation policy.

Distinguish source compatibility, binary compatibility, and cross-process
protocol compatibility when reviewing a change:

| Boundary | What to review |
| --- | --- |
| Source API | Public declarations, template behavior, option aggregates, and enum values used when an application is rebuilt. |
| Shared-library ABI | Exported symbols, signatures, layouts, vtables, calling conventions, and the matching compiler and dependency ABI. |
| Cross-process protocol | Payload layout, endpoint contracts, and transport configuration used by independently running peers. |

The `ci/abi/holomq.exports.baseline.txt` check compares exported symbol names
for the reference Linux/GNU shared build. It catches additions and removals
from that inventory, including accidental export leaks. It does not compare
class layouts, enum values, inline or template behavior, or wire protocols;
an unchanged symbol list alone does not establish binary compatibility.

Downstream applications should use public `<holomq/...>` headers and link the
installed CMake target so they inherit its dependency and compile definitions.
Build applications and plugins against matching SDK headers and libraries.
Consult the release notes for version-specific compatibility requirements and
rebuild requirements when upgrading. The symbol inventory and SONAME alone
do not establish a cross-release compatibility guarantee.

Public option aggregates and enums in `<holomq/...>` headers are part of the source-facing API
even when they add no out-of-line DSO symbol. Adding an endpoint-local field such as
`PublisherPubSubOptions::pressure_policy` therefore requires a matching public API update and
installed-consumer rebuild evidence. It does not require a symbol-baseline update unless the
curated `nm -D` inventory changes. Consumers must rebuild against the matching headers and
library rather than relying on aggregate layout compatibility between SDK packages.

## Shipped library type

When `BUILD_SHARED_LIBS=ON`, application authors link the shared `libholomq.so`
target. Holoscan binary artifacts ship `libholomq.so`; static SDK builds provide
the matching static archive. HoloMQ uses the SDK version and SOVERSION.

Holoscan Core links HoloMQ privately for implementation use in SDK builds. That
does not move HoloMQ into the Core public ABI: HoloMQ public headers are under
`include/holomq/`, and direct HoloMQ users should continue linking
`holoscan::holomq` or its compatibility alias `holomq::holomq`.

In-tree tests use SDK-private object compilation where implementation coverage is
needed; no second HoloMQ archive is produced. External applications use
`find_package(holoscan REQUIRED)` and link `holoscan::holomq` or the
compatibility alias `holomq::holomq`. HoloMQ is not installed as a separate
CMake package.

## Local Checks

From the SDK source root:

```bash
./run lint
./run build --config "-DHOLOSCAN_BUILD_TESTS=ON"
```

Or run pre-commit directly from the repository root:

```bash
pre-commit run --config .pre-commit-config.yaml --all-files
```

From an SDK development environment, run the configured build's export checks.
The semantic target checks public declarations and accepts the build's
`compile_commands.json`; the symbol-baseline target builds and inspects the
shared library. The latter is available only for a Linux shared build with
the GNU C++ compiler:

```bash
cmake --build <build-dir> --target holomq-check-export-macro-semantics
cmake --build <build-dir> --target holomq-check-symbols
```

The hooks reject macro include guards, missing `#pragma once`, hand-written
export macros, old `holomq/export.hpp` includes, and export usage in `detail/`.
The symbol check compares demangled `nm -D` output against
`ci/abi/holomq.exports.baseline.txt`, using
`scripts/check_holomq_symbol_baseline.sh`. Inspect the diff for every symbol
addition or removal and assess compatibility before accepting a baseline
update. Regenerate the baseline only for an intentional, reviewed inventory
change using the same reference Linux/GNU configuration. From the SDK source
root:

```bash
build_dir=/path/to/sdk-build
library="${build_dir}/lib/libholomq.so"
export LC_ALL=C
nm -D --defined-only --extern-only "${library}" \
  | awk '{print $3}' | c++filt | sort -u > ci/abi/holomq.exports.baseline.txt
bash scripts/check_holomq_symbol_baseline.sh \
  "${library}" ci/abi/holomq.exports.baseline.txt
```
