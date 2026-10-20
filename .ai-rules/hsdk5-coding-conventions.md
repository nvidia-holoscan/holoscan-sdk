---
globs: "**/*.cpp,**/*.cu,**/*.cuh,**/*.hpp,**/*.fbs,**/*.py"
alwaysApply: false
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# HSDK 5 Coding Conventions

## Repository Scope

- Paths and commands are relative to the SDK source root containing `run` and
  `CMakeLists.txt` (the `public/` directory in the internal checkout).
- Preserve natural public include paths: `<holoscan/...>` for Core and `<holomq/...>` for HoloMQ.
  This applies to headers under the public include roots. Source-tree-only code (`src/`,
  `tests/`, and benchmarks gated on `holoscan_SOURCE_DIR`) may include private
  implementation headers under `src/` with quoted paths, for example
  `#include "channel/channel_payload_limits.hpp"`; do not rewrite those to `<holoscan/...>` or
  move the headers into the public include roots to satisfy the convention.

## Applicability to source-preserving operations

- Apply these conventions when authoring or substantively editing source. An
  exact release projection, re-rooting, or reviewed cherry-pick is a
  source-preserving operation: its approved source commit or range defines the
  payload.
- Do not modify that payload merely to normalize it to current conventions. If
  the projected source exposes a pre-existing convention or lint issue, report
  it separately and obtain review before adding a corrective commit.
- Follow the applicable release procedure for provenance, path selection,
  validation, and exact-tree checks. A source-preserving operation is not an
  exemption for unrelated newly authored changes in the same branch.

## File Organization

- Use `.cpp` for C++ source files and `.hpp` for C++ headers; use `.cu`/`.cuh` for CUDA. Do not use `.cc`, `.cxx`, `.h`, or `.hxx` for new C++ files.
- Start source files with the NVIDIA SPDX copyright and Apache-2.0 license header.
- End every source file with a newline.
- Prefer this include order unless a local file has a stronger convention: standard C++ headers, third-party headers with angle brackets, then project headers.
- Keep public headers under natural public include roots: `include/holoscan/...` for Core and `include/holomq/...` for HoloMQ.

## Headers and ABI

- Hand-written C++ headers use `#pragma once`; do not add macro include guards.
- Generated export headers are the only macro-guard exception: `<holoscan/holoscan_core_export.h>` and `<holomq/holomq_export.h>`.
- Do not hand-write export macros or include `<holomq/export.hpp>`.
- Use `HOLOSCAN_CORE_EXPORT` for Core out-of-line public DSO ABI and `HOLOMQ_EXPORT` for HoloMQ out-of-line public DSO ABI.
- Any path segment named `detail` is internal and must not use export macros or be included by applications/examples.
  Core public headers must not include `detail/` internals. HoloMQ public headers follow the
  module-local export/detail policy because some header-only
  or template APIs may need to include HoloMQ-private implementation headers.
- Headers listed as light in `scripts/core_light_headers.txt` must not reach the listed heavy
  headers (`scripts/check_light_headers.py`). Include the vocabulary leaf that declares a name
  instead of the header that aggregates it.
- A source file that defines exported functions must include the public header that declares them.
  Otherwise the definition compiles with hidden visibility and the symbol is not exported.

## Namespaces and Ownership

- Holoscan Core code uses namespace `holoscan`.
- HoloMQ public code uses namespace `holomq`, target `holomq::holomq`, and headers under `include/holomq/`.
- Do not add `holoscan::ipc` compatibility shims unless there is an explicit shipped-compatibility requirement.

## Naming Conventions

- Use `PascalCase` for class and struct names, for example `GPUResidentOperator`.
- Use `snake_case` for functions, methods, variables, and parameters, for example `add_flow` or `min_size`.
- Use `kPascalCase` for constants and enum values, for example `kSucceeded` or `kFailed`.
- Keep acronyms and initialisms fully capitalized in `PascalCase` names, following PEP 8's rule to capitalize every letter of an acronym; for example, use `GPUResidentOperator` and `GPUOperator`, not `GpuResidentOperator` or `GpuOperator`.
- For FlatBuffers and other cross-language message schemas, use schema-idiomatic `PascalCase` names that produce natural generated APIs. Acronyms may be title-cased as words when that matches the schema ecosystem, for example `Imu`, `Uuid`, or `Gps`, instead of `IMU`, `UUID`, or `GPS`.
- Keep names descriptive enough to avoid comments that merely restate the symbol name.

## Stable Identity Names

- Use `holoscan.*` for SDK-owned identities and `holoscan.<provider>.*` for
  downstream identities invented specifically for Holoscan integration. NVIDIA
  modules use the same convention whether shipped in or outside the SDK repository.
  Place the area or provider directly after `holoscan`; do not insert `.ext`. Do not
  use alternate prefixes such as `com.nvidia.*`, `nvidia.*`, or other reverse-DNS
  spellings in those domains. Keep namespace components lowercase; a public type
  component retains its public case.
- Apply this convention to parameter schemas, custom validators, Tensor byte-span
  projectors, raw-POD identities, registry keys, and their constants, examples,
  tests, and expected output. SDK examples use `holoscan.examples.*`, tests use
  `holoscan.test.*`, and benchmarks use `holoscan.benchmark.*`.

| Domain | Example | Version |
|---|---|---|
| Operator or Graph registry key | `holoscan.vision.ResizeOp` | No version suffix |
| Parameter schema | `holoscan.vision.ResizeOpParams` | Separate positive schema version; no `.vN` suffix |
| Custom validator | `holoscan.vision.validator.finite-f64.v1` | Semantic `.vN` suffix |
| Tensor byte-span projector | `holoscan.vision.ResizeOp.output-max-byte-span.v1` | Semantic `.vN` suffix |
| Raw-POD identity | `holoscan.example.payload.FusionControl` | Separate `ChannelTypeIdentity<T>::version`; no `.vN` suffix |
| Generated FlatBuffers family | `holoscan.schema.Image` or a third-party schema's exact FQN | Separate compatibility epoch |

- Keep generated FlatBuffers family names authoritative to their `.fbs` namespace
  and root. Portable third-party schemas may use any valid namespace; never
  rewrite them or require a second hand-authored family name. Preserve existing
  built-in registry keys such as `holoscan.Tensor`.
- Keep identities explicit and readable. Do not derive them from C++ symbols,
  RTTI, paths, addresses, or random UUIDs. Generic APIs keep their existing name
  validation and exact comparison; this source convention adds no runtime prefix
  restriction or compatibility alias.
- Change a validator/projector semantic version when the same input can produce
  different accepted values, results, or failures. A spelling cleanup does not
  reset schema, POD, ABI, wire, artifact, or hash-domain versions.
- Internal hash-domain separators, diagnostic tokens, capability/policy IDs,
  endpoint names, URI schemes, and unrelated NVIDIA text follow their own
  contracts and are outside this ownership migration.
- Run `python3 scripts/check_stable_identity_names.py` from the SDK source root;
  ordinary lint runs the same check. It checks maintained C++/CUDA and Python
  string literals, including constants, multiline calls, and adjacent C++
  strings and standard `sv` suffixes. Direct schema/callback declarations and POD
  traits, including `holoscan::ChannelTypeIdentity` and its global-qualified form,
  also check ownership and version placement; simple constants are resolved
  without C++ scope analysis. It does not evaluate macros, computed expressions
  or custom literal suffixes. Keep reviewed
  exceptions in [stable_identity_exceptions.json](../scripts/stable_identity_exceptions.json)
  as an exact path/literal plus reason. Never exempt an entire test, example,
  or third-party directory. Review portable schema and intentional negative-test
  exceptions without weakening the generic runtime API.

## C++ API Documentation

- Document every exported or public C++ API with Doxygen comments. Document protected APIs when
  they are intended extension points.
- Use `///` immediately before classes, functions, type aliases, and declarations that need
  standalone or multi-line documentation. Do not introduce `/** ... */` or `/*! ... */` for new
  API documentation.
- Use `///<` only for short enum-value or simple public-data-member descriptions where trailing
  placement improves readability. Use one placement style consistently within the same enum or
  member group; put longer documentation before the declaration.
- Use `@` for standalone Doxygen commands, including `@brief`, `@tparam`, `@param`, `@return`,
  `@retval`, `@throws`, `@note`, `@warning`, `@deprecated`, and `@see`, where applicable.
- Use backslash forms for inline Doxygen markup: `\ref` for linked symbols, `\p` for parameter
  names, and `\c` for short code tokens. Do not alternate `@ref`/`\ref`, `@p`/`\p`, or
  `@c`/`\c` within a header.
- Use complete sentences. Method summaries should start with an imperative verb, for example
  "Return..." or "Add...". Put documentation immediately before the declaration it documents,
  except for the narrowly allowed trailing `///<` form.
- Treat public-header documentation as the authoritative source for planned Fern C++ Library
  Reference pages once generation is configured. Do not put Fern MDX components or MyST/Sphinx
  roles in C++ comments.
- Apply these rules to new and substantively edited documentation. Do not rewrite otherwise
  untouched declarations solely to normalize comment syntax.

## Code Style

- Prefer concise comments that explain non-obvious intent; avoid comments that restate code.
- Use `override` on overriding virtual functions.
- Use `[[maybe_unused]]` for intentionally unused parameters.
- Use `[[nodiscard]]` selectively when ignoring a return value is likely a bug, especially for
  ownership-transfer handles, `expected<T, Error>` results, status/error-checking APIs, and state
  query accessors. Do not enable it mechanically for every getter if the result is harmless to
  ignore.
- Prefer the traditional leading return type syntax for ordinary function declarations,
  especially for exported or public SDK APIs.
- Use trailing return types, `auto f(...) -> T`, only when they are required by the language or
  clearly improve readability:
  - when the return type depends on function parameters, for example `decltype(lhs + rhs)`;
  - when the return type is long or heavily templated and would obscure the function name;
  - when matching the local style of an existing class, overload set, or closely related API
    family.
- Do not rewrite existing declarations solely to normalize return-type style. Prefer consistency
  within the same class, overload set, or API family over repository-wide mechanical uniformity.
- Do not use deduced `auto` return types for exported or public SDK APIs unless the deduced type is
  intentionally part of a template or private-helper contract. `auto f(...) -> T` has an explicit
  return type; `auto f(...) { ... }` uses return type deduction.

Examples:

```cpp
// Preferred for ordinary public APIs.
[[nodiscard]] HOLOMQ_EXPORT expected<WaitResult, Error>
wait_impl(const WaitStrategy& strategy) noexcept;

// Allowed when the return type depends on parameters.
template <typename LhsT, typename RhsT>
auto add(LhsT&& lhs, RhsT&& rhs)
    -> decltype(std::forward<LhsT>(lhs) + std::forward<RhsT>(rhs));

// Allowed when a long return type would hide the function name.
auto make_subscription(const Topic& topic, const SubscriberOptions& options)
    -> expected<std::unique_ptr<Subscriber<Transport::Iceoryx2>>, Error>;
```

- Preserve const correctness.
- Use smart pointers for ownership: `std::unique_ptr` for exclusive ownership and `std::shared_ptr` for shared ownership.
- Avoid raw owning pointers; raw pointers are acceptable only for non-owning references when clearer than alternatives.
- Do not use `new`/`delete` in new code; create owned objects with `std::make_unique` or
  `std::make_shared`.
- Wrap every acquired resource in an RAII owner, including CUDA streams, events, device and
  pinned allocations, file descriptors, and other handles released by a C API.
- Use `nullptr`; do not use `NULL` or `0` for pointers.
- Validate at system boundaries: public API arguments, configuration, user input, and data read
  from files, networks, or other processes. Inside the SDK, rely on invariants already established
  instead of re-checking them, especially in runtime hot paths.
- Do not add compatibility shims, aliases, or forwarding overloads for code that has not shipped
  in a release; update the callers instead. For shipped APIs, follow the stage-specific policy in
  [docs/api_stability.mdx](../docs/api_stability.mdx).
- Treat `std::span` and `std::string_view` parameters as borrowed views. Do not store them beyond
  the call unless the data is copied into an owned container or string.
- Inline trailing comments should start lower-case, for example `return {};  // return success`.
- Standalone comments inside a method should start upper-case, for example `// Iterate over each item`.

## Error Handling and Logging

- For Holoscan Core code, use the existing `HOLOSCAN_LOG_*` macros when available.
- Keep independently linkable HoloMQ code free of a Core logging dependency;
  use HoloMQ-local logging or standard facilities.
- Prefer `expected<T, Error>` for recoverable fallible operations. In runtime hot paths, return
  typed error codes with the message unset when the code is sufficient and constructing diagnostic
  strings would add avoidable overhead.
- Provide informative error messages when failures occur during setup, configuration,
  initialization, or other pre-execution paths where diagnostics are more valuable than avoiding
  string construction.
- Include useful context in errors, such as component, operator, backend, or port names.
- Use appropriate log levels: errors for failures, debug/trace for diagnostics, and info for user-visible progress.
- Do not write diagnostics from library code with `std::cout`, `std::cerr`, `printf`, or
  `fmt::print`; use the logging facility for that code. Examples, tests, and command-line tools
  may print their own user-facing output.
- Guard logging in destructors, or functions called by destructors, with `try`/`catch` because
  formatting and sink code can throw and destructors must not let exceptions escape.

## CMake and Build

- Follow existing local CMake patterns before adding new abstractions.
- Keep shared libraries hidden by default with explicit generated export headers.
- HoloMQ is SDK-owned: its sources are under `src/holomq/`, its build is defined
  in `cmake/HoloMQ.cmake`, and its library version follows the enclosing SDK project.
- Do not enable `WINDOWS_EXPORT_ALL_SYMBOLS` or `llvm-header-guard`.
- Support both shared and static library builds where the target is intended to be reusable.
- For HoloMQ changes, use the top-level SDK CMake build and CTest paths; there is
  no module-local CMake project.

## Python

- [.ruff.toml](../.ruff.toml) is authoritative for Python style, including the Python 3.12 target,
  100-character lines, import conventions, and debugger-call checks. Fix Ruff findings rather than
  adding `noqa` comments or per-file ignores; when a suppression is unavoidable, scope it to the
  line and explain why.
- Keep C++ and Python at parity. When adding or changing a public C++ API that has Python
  bindings, update the bindings, typing stubs, and Python tests in the same change, or state why
  the Python side is deferred.
- Make the Python API Pythonic rather than a transliteration of C++: expose simple accessors as
  properties (`def_prop_ro`/`def_prop_rw`) instead of `get_x()`/`set_x()` methods, accept
  keyword arguments, and use standard protocols such as iteration and context managers where they
  fit the object's semantics.

## Python Bindings with nanobind

- Use `nanobind_add_module(... STABLE_ABI ...)` for Python extension modules so Python 3.12+ builds can produce stable ABI (`abi3`) extensions.
- Keep existing nanobind linkage flags unless there is a concrete reason to change them.
- Require Python 3.12 or newer for nanobind `STABLE_ABI` modules; supported builds must produce
  `abi3` extension suffixes rather than version-specific `cpython-3XY` suffixes.
- Keep binding code thin: expose the public C++ API and avoid duplicating business logic in `_core.cpp` or `_holomq.cpp`.
- HoloMQ bindings should link to `holomq::holomq`, not Holoscan Core targets.

## Tests and Verification

- Add focused tests for public behavior, failure cases, edge cases, and ABI-sensitive changes.
- Name test files with a `test_` prefix for both C++ and Python, for example `test_operator.cpp` or `test_holomq.py`.
- Use GTest for C++ tests and pytest for Python tests when Python bindings are involved.
- Use `./run lint` from the SDK source root for full-tree SDK/Core-wide and HoloMQ checks; at
  commit time, the installed pre-commit hook checks the staged files.
- HoloMQ does not carry module-local `run` or Dockerfile tooling. Use the SDK
  `run` script for containerized workflows and its top-level CMake project
  for host builds.
- For export/ABI changes, also run the relevant CMake targets: `check-export-macro-semantics`, `holomq-check-export-macro-semantics`, and `holomq-check-symbols`.
