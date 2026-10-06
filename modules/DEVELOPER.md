<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Holoscan Core Module development guide

This guide defines the shared development contract for SDK-owned modules under
`public/modules/`. A module extends the base Holoscan SDK with a separately selectable,
domain-specific public capability. Core-integrated subsystems, including HoloMQ,
are not modules for this guide.

The guide is contributor policy, not a behavioral design-requirements domain.
Module-specific behavior remains documented and tested with the relevant SDK
design and API conventions.

## Requirements and recommendations

**MUST** requirements apply to every new module. Existing modules may be
grandfathered: their differences are recorded below and are changed only by a
scoped, approved migration. **SHOULD** recommendations describe the preferred
layout when the module's package and API boundary make them applicable.

Every module MUST use the top-level SDK development workflow. Do not add a
module-local Dockerfile, run wrapper, CI setup, or lint configuration. Use
`public/run` from the repository root for build, test, and lint commands.

New modules MUST contain a root `CMakeLists.txt` and `README.md`, and SHOULD
also contain this `DEVELOPER.md` contract:

```text
<module>/
├── CMakeLists.txt
├── README.md
├── DEVELOPER.md
├── include/<public-include-root>/
├── src/
│   └── detail/
└── tests/
    ├── unit/
    └── installed_consumer/
```

`examples/`, `docs/`, `cmake/`, package metadata, `VERSION`, and `NOTICE` are
optional. Add them only when their stated role requires them: shipped examples,
module-local design notes, package/dependency configuration, external catalog
metadata, or independently versioned/packageable content.

### Public and private boundaries

- Module-owned public headers MUST live below the module's own `include/` root.
  New module APIs must not be installed from a broad SDK include tree or from
  `src/include`. A necessary SDK-core API addition is an explicit ownership
  exception and MUST be documented in the module boundary declaration.
- Implementation-only headers SHOULD live under `src/detail/` or another
  private source directory and must not be installed as public API.
- Public targets MUST have stable, documented aliases, include directories for
  both build and install interfaces, and installation rules appropriate to the
  chosen package boundary. Shared public libraries MUST follow the SDK's
  visibility and export-header conventions.
- Public API, CMake, ABI, ownership, documentation, and test conventions are
  governed by the repository coding conventions. This guide does not replace
  those rules.

### Build and package boundary

Each module MUST declare its boundary in its `DEVELOPER.md` or, until one is
added, in its `README.md`. The declaration must identify the build option and
default, supported build contexts, public target aliases and include roots,
package or install component, consumer `find_package` form, external
dependencies, supported platforms, and test labels.

The following table describes possible build approaches:

| Build Approach | Support Requirement | Example Invocation | Discussion |
| --- | --- | --- | --- |
| In-tree build with Holoscan SDK | REQUIRED | `cmake -S .. -B . -DHOLOSCAN_BUILD_<MODULE_NAME>:BOOL=<ON/OFF>` | Each module must be individually selectable within the Holoscan SDK build. |
| Standalone top-level build | OPTIONAL | `cmake -S path/to/modules/my-module -B build` | Each module may optionally support a build configuration as a standalone project against some pre-built Holoscan SDK core libraries installation |

Module installations must meet two required downstream CMake ingestion approaches:

| How to Use | Support Requirement | Example | Consumer boundary |
| --- | --- | --- | --- |
| Optional main-package component | REQUIRED | `find_package(holoscan COMPONENTS <MODULE_NAME> REQUIRED)` | Required compatibility form for consumers that discover the SDK as one package. |
| Separate module package | REQUIRED, recommended | `find_package(holoscan-<MODULE_NAME>)` | Recommended form for new module consumers; use the module's documented exact `_DIR` hint until prefix-only discovery is added. |

Every installable module SHOULD have an installed-consumer test that configures
only against its documented installed package boundary and links its public
target or targets. Separately installable components SHOULD additionally verify
their intended component-isolation behavior. Unit, GPU or functional, and
system or package-boundary tests SHOULD use distinct CTest labels.

### Documentation, examples, and exceptions

Each module README MUST explain its purpose, user-facing API or capability,
supported compatibility envelope, build and test entry points, and package
consumer recipe. A module that provides supported end-user API SHOULD have a
user-guide page and navigation entry. Module-local design notes are for
durable architecture decisions that do not belong in the user guide.

Shipped examples SHOULD be built, tested, and installed through the SDK's
shared example machinery. If a module has no examples, its boundary declaration
must say why and point users to the supported usage documentation instead.

Any exception to this guide MUST state the affected rule, rationale, owner, and
planned disposition: intentional permanent boundary, deferred migration, or
retirement. Do not duplicate the exception text across module documentation;
the boundary declaration is authoritative.

## New-module checklist

- Declare the build option, package archetype, targets, public headers,
  dependency ownership, supported platforms, and install component before
  adding build integration.
- Add the top-level SDK feature gate and conditional subdirectory registration.
- Add public-header installation, package export/configuration, and an
  installed-consumer test for the selected archetype.
- Add focused unit coverage; add functional or GPU coverage when the module
  performs device or runtime work; label each test category.
- Write the README and developer boundary declaration; add examples and
  user-guide documentation when the module exposes supported user-facing API.
- Run the closest SDK build and CTest path for the selected configuration, then
  run `public/run lint` before commit.

## Existing module alignment

| Module | Current archetype | Alignment direction |
| --- | --- | --- |
| Inference | Optional main-package component | Add a developer boundary declaration; make CPack inclusion an explicit package decision; normalize test labels and document the no-example policy. Migrate module-owned headers away from `src/include` and broad SDK include installation only through a compatible, scoped API migration. |
| Vision | Separate module package | Use as the compact-layout baseline. Add the shared boundary declaration and clarify its example-install policy; retain its local headers, package consumer test, and top-level SDK workflow. |
| Visualization | Optional main-package component (`find_package(holoscan COMPONENTS visualization)`), with an optional standalone build | Add a developer boundary declaration that records both public layers and build contexts; normalize unit, system, and installed-consumer labels; decide separately whether catalog metadata and Fern user-guide coverage are warranted. |

No alignment item authorizes header moves, package-boundary changes, or new
examples by itself. Those changes need their own implementation proposal and
compatibility review.
