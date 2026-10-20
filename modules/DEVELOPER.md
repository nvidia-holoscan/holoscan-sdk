<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Holoscan Core Module development guide

This guide defines the shared development contract for SDK-owned modules under
`modules/`. A module extends the base Holoscan SDK with a separately selectable,
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

Every module MUST use the top-level SDK development workflow for in-tree
builds, tests, lint, and CI. Use `./run` from the SDK repository root. Promoted
external modules may retain existing Dockerfiles, CLI requirements, wrappers,
and CI or lint configuration for optional standalone use. Do not add
module-local replacements for the SDK workflow; SDK validation MUST use the
top-level tooling.

Every module MUST contain a root `CMakeLists.txt`, `README.md`, and
`metadata.json`. New modules SHOULD also contain a `DEVELOPER.md` boundary
declaration.

### Accepted source layouts

The compact layout and the Holoscan CLI module template layout are accepted
Core Module archetypes. These source layouts are independent of the build and
package boundaries described below. The trees are representative; optional
directories are needed only when the module provides the corresponding content.

#### Compact layout

```text
<module>/
├── CMakeLists.txt
├── README.md
├── metadata.json
├── DEVELOPER.md
├── include/<public-include-root>/
├── src/
│   └── detail/
└── tests/
    ├── unit/
    └── installed_consumer/
```

`examples/`, `docs/`, `cmake/`, package metadata, and `NOTICE` are optional.
Add them only when their stated role requires them: shipped examples,
module-local design notes, package/dependency configuration, external catalog
metadata, or packageable content.

Core Modules MUST NOT carry a module-local `VERSION` file. They share the SDK
version in the top-level `VERSION` file.

### Module metadata

Each module's `metadata.json` is its catalog record. It MUST conform to
[`metadata.schema.json`](metadata.schema.json) (`urn:holoscan:module:v2`), whose
fields and value constraints match the HoloHub module schema so the same file
validates in both catalogs. Use [`vision/metadata.json`](vision/metadata.json)
as the reference. In addition to the schema:

- `$schema` MUST be `urn:holoscan:module:v2`.
- `module.name` and `module.binary_packages.debian` MUST be
  `holoscan-<module>`, where `<module>` is the module directory name.
- `module.version` and `module.holoscan_sdk.minimum_required_version` MUST
  equal the SDK version in the top-level `VERSION` file, and
  `module.holoscan_sdk.tested_versions` MUST include it.
- `module.documentation.readme` MUST name an existing file relative to
  `metadata.json`.

The `check-module-metadata` lint hook (`scripts/check_module_metadata.py`)
enforces these rules for every module directory and runs whenever a module,
the schema, or `VERSION` changes.

#### Holoscan CLI module template layout

The [Holoscan CLI module template](https://github.com/nvidia-holoscan/holoscan-cli/tree/main/src/holoscan_cli/templates/module),
formerly provided by HoloHub as the external-module template, uses this layout:

```text
<module>/
├── CMakeLists.txt
├── README.md
├── DEVELOPER.md
├── metadata.json
├── cmake/
├── operators/
│   └── <operator>/
│       ├── CMakeLists.txt
│       ├── <operator>.hpp
│       ├── <operator>.cpp
│       └── metadata.json
├── applications/
│   └── <pipeline>/cpp/
├── pkg/
│   └── holoscan-<name>/
└── tests/
    └── cpp/
```

Holoscan CLI provides the template and copies its CMake support into generated
modules; see the [CLI creation reference](https://github.com/nvidia-holoscan/holoscan-cli/blob/main/CLI_REFERENCE.md).
The scaffold also provides standalone development files such as `Dockerfile`,
`requirements-cli.txt`, `pyproject.toml`, and CI or lint configuration. Their
presence does not replace SDK build, package, or test declarations. Neither
accepted layout requires Python bindings or wheels.

Promotion from an external repository or HoloHub to a Core Module does not
require restructuring the module's internal directories or changing its
public header include paths. The promotion checklist below describes the
required SDK integration.

### Public and private boundaries

- Module-owned public headers MUST live below the module's own `include/` root
  or its `operators/<operator>/` directories in the CLI template layout.
  Installation MUST explicitly select public headers and preserve their
  documented include paths. New module APIs must not be installed from a broad
  SDK include tree or from `src/include`. A necessary SDK-core API addition is
  an explicit ownership exception and MUST be documented in the module boundary
  declaration.
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
- Add a `metadata.json` that passes the `check-module-metadata` lint hook.
- Run the closest SDK build and CTest path for the selected configuration, then
  run `./run lint` from the SDK repository root before commit.

## External-module promotion checklist

Promoted modules MUST meet the shared build, package, API, and test contract
above. Accepting the source layout does not make a generated standalone
project ready for an SDK in-tree build.

- Place the module under `modules/<name>/` and add its top-level SDK feature
  gate and conditional subdirectory registration.
- Preserve the module's internal layout, including `operators/`,
  `applications/`, `pkg/`, and existing test directories. Public headers may
  remain beside operator implementations under the boundary rules above.
- Adapt CMake to use SDK targets in-tree. Discover an installed SDK with
  `find_package(holoscan ...)` only for an optional standalone build; the
  template's standalone discovery must not be required in-tree.
- Integrate package exports, install components, and both required consumer
  forms with the selected SDK package boundary. Retained catalog metadata is
  not a substitute for these declarations.
- Use SDK shared example machinery for shipped applications, and add or adapt
  labeled tests and installed-consumer coverage without requiring directory
  moves to `examples/`, `tests/unit/`, or `tests/installed_consumer/`.
- Update the module's boundary declaration, document optional standalone
  tooling, and validate through the top-level SDK workflow.

## Existing module alignment

| Module | Current archetype | Alignment direction |
| --- | --- | --- |
| Inference | Optional main-package component | Add a developer boundary declaration; make CPack inclusion an explicit package decision; normalize test labels and document the no-example policy. Migrate module-owned headers away from `src/include` and broad SDK include installation only through a compatible, scoped API migration. |
| Vision | Separate module package | Use as the compact-layout baseline. Add the shared boundary declaration and clarify its example-install policy; retain its local headers, package consumer test, and top-level SDK workflow. |
| Visualization | Optional main-package component (`find_package(holoscan COMPONENTS visualization)`), with an optional standalone build | Add a developer boundary declaration that records both public layers and build contexts; normalize unit, system, and installed-consumer labels; decide separately whether Fern user-guide coverage is warranted. |

No alignment item authorizes header moves, package-boundary changes, or new
examples by itself. Those changes need their own implementation proposal and
compatibility review.
