# Developer Resources

This document aims to guide users with recommended and advanced workflows to build and use Holoscan SDK. This is generally not the simplest way to use the SDK, so make sure to review the [project README](./README.md) before getting started.

> [!WARNING]
> **Disclaimer**: we only recommend building the SDK from source if you are a developer of the SDK, or need to build the SDK with debug symbols or other options not used as part of the published packages.
>
> - If you want to write your own operator or application, you can use the SDK as a dependency (and contribute to [HoloHub](https://github.com/nvidia-holoscan/holohub)).
> - If you need to make other modifications to the SDK, [file a feature or bug request](https://forums.developer.nvidia.com/c/robotics-edge-computing/holoscan/757).
> - Refer to the [Holoscan SDK User Guide installation instructions](https://docs.nvidia.com/holoscan/sdk-user-guide/sdk_installation.html#install-the-sdk) for guidance on installing Holoscan SDK from published packages.

## Table of Contents

- [Building the SDK from source](#building-the-sdk-from-source)
  - [Prerequisites](#prerequisites)
  - [(Recommended) using the `run` script](#recommended-using-the-run-script)
  - [Cross-compilation](#cross-compilation)
  - [(Advanced) Docker + CMake](#advanced-docker--cmake)
  - [(Advanced) Local environment + CMake](#advanced-local-environment--cmake)
  - [Build Variants and Configuration](#build-variants-and-configuration)
- [Utilities](#utilities)
  - [Testing](#testing)
    - [Test Types and Categories](#test-types-and-categories)
    - [Test Execution Methods](#test-execution-methods)
    - [Test Environment](#test-environment)
    - [Test Configuration](#test-configuration)
    - [Reproducing Test Failures](#reproducing-test-failures)
  - [Linting](#linting)
  - [Pre-commit hooks](#pre-commit-hooks)
  - [ABI, Visibility, and Header Guards](#abi-visibility-and-header-guards)
  - [Building the User Guide](#building-the-user-guide)

## Building the SDK from source

### Prerequisites

- Prerequisites for each supported platform are documented in [the user guide](https://docs.nvidia.com/holoscan/sdk-user-guide/sdk_installation.html#prerequisites).
- To build and run the SDK in a containerized environment (recommended) you'll need:
  - the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) v1.12.2+
  - [Docker](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository), including the buildx plugin (`docker-buildx-plugin`)

### (Recommended) Using the `run` script

Call **`./run build`** within the repository to build the build container and the CMake project.

- *If you encounter errors during the CMake build, you can execute `./run clear_cache` to remove cache/build/install folders*
- *Execute `./run build --help` for more information*
- *Execute `./run build --dryrun` to see the commands that will be executed*
- *That command can be broken-up in more granular commands also:*

  ```sh
  ./run check_system_deps  # ensure the system is properly configured for building
  ./run build_image        # create the build Docker container
  ./run build              # run the CMake configuration, build, and install steps
  ```

Call the **`./run launch`** command to start and enter the build container.

The commands in this guide run from the SDK source root. `launch` starts in
the default build directory, for example `build-x86_64`. To make directory
selection explicit:

```bash
BUILD_DIR=$(./run get_build_dir)
INSTALL_DIR=$(./run get_install_dir)

# Enter the build tree; equivalent to ./run launch.
./run launch "${BUILD_DIR}"

# Or enter the install tree.
./run launch "${INSTALL_DIR}"
```

These queries return directory names relative to the directory containing `run`.
They honor `HOLOSCAN_BUILD_ARCH` and an explicit `--arch` argument, which takes
precedence; otherwise the target defaults to the host architecture. Use the same
target selection for building, querying paths, and launching. For example, set
`export HOLOSCAN_BUILD_ARCH=aarch64` before the commands above to select ARM64.
Selecting a target does not make its binaries executable on another architecture.

The queries describe the standard directory layout. If you build with a custom
`--buildpath` or `--installprefix`, pass that path explicitly instead.

- *Execute `./run launch --help` for more information*
- *Execute `./run launch --dryrun` to see the commands that will be executed*
- *Execute `./run launch --run-cmd "..."` to execute a bash command directly in the container*

Run the [**examples**](./examples) inside the container. Each example builds with the SDK and can also be built against an installed SDK using its `CMakeLists.min.txt`.

### Cross-compilation

While the Dockerfile to build the SDK does not currently support true cross-compilation, you can compile the Holoscan SDK for the developer kits (arm64) from a x86_64 host using an emulation environment.

1. [Install qemu](https://github.com/multiarch/qemu-user-static)
2. Clear your build cache: `./run clear_cache`
3. Rebuild for `linux/arm64` using `--arch|-a` or `HOLOSCAN_BUILD_ARCH`:
    - `./run build --arch arm64`
    - `HOLOSCAN_BUILD_ARCH=arm64 ./run build`

You can then copy the `install` folder generated by CMake to a developer kit with a configured environment or within a container to use for running and developing applications.

### (Advanced) Docker + CMake

The [`run`](./run) script mentioned above is helpful to understand how Docker and CMake are configured and run, as commands will be printed when running it or using `--dryrun`.
We recommend looking at those commands if you want to use Docker and CMake manually, and reading the comments inside the script for details about each parameter (specifically the `build()` and `launch()` methods).

### (Advanced) Local environment + CMake

> [!WARNING]
> **Disclaimer**: this method of building the SDK is not actively tested or maintained. Instructions below might go out of date.

#### Software Requirements

To build the Holoscan SDK on a local environment, refer to the list of dependencies installed in the top-level [Dockerfile](./Dockerfile).

For CMake to find these dependencies, install them in default system paths, or pass `CMAKE_PREFIX_PATH`, `CMAKE_LIBRARY_PATH`, and/or `CMAKE_INCLUDE_PATH` during configuration.

#### Build example

```sh
# Configure
cmake -S $source_dir -B $build_dir \
  -G Ninja \
  -D CMAKE_BUILD_TYPE=Release \
  -D CUDAToolkit_ROOT:PATH="/usr/local/cuda"

# Build
cmake --build $build_dir -j

# Install
cmake --install $build_dir --prefix $install_dir
```

The commands to run the [**examples**](./examples) are then the same as in the dockerized environment.

### Build Variants and Configuration

The SDK can be built with different configurations to match various deployment targets:

**CUDA Version**: 13 (pinned by the 5.x build image).

**Architectures**: x86_64, aarch64 (defaults to the host architecture)

```sh
./run build --arch aarch64
# or
export HOLOSCAN_BUILD_ARCH=aarch64
./run build
```

**Build Types**: Release (default), Debug, RelWithDebInfo

```sh
./run build --type debug
# or
export CMAKE_BUILD_TYPE=Debug
./run build
```

The current directory layout is `build-<arch>` and `install-<arch>`, for example
`build-x86_64` and `install-aarch64`. Use the queries above in commands that need
a path so that they follow any future naming changes.

## Utilities

Some utilities are available in the [`scripts`](./scripts) folder, others closer to the built process are listed below:

### Testing

Existing tests are using GTest for C++ and pytest for Python, and can be found under [tests](./tests/) and [python/tests](./python/tests/) respectively. The Holoscan SDK uses CTest as a framework to build and execute these tests.

#### Test Types and Categories

The SDK includes several types of tests:

1. **Core HSDK Tests**: Unit, integration, and system tests for the SDK core functionality
   - Located in `tests/` directory
   - C++ tests using GTest
   - Python tests using pytest

2. **Example Tests**: Validation that SDK examples build and run correctly
   - Tests examples from the installation tree
   - Ensures examples work with the installed SDK

#### Test Execution Methods

You can run tests using the `./run` script:

```sh
# Run all tests
./run test

# Run specific test by name (regex supported)
./run test --name <test_name>

# Run with verbose output
./run test --verbose

# Run with additional CTest options
./run test --options "-R <test_regex> --output-on-failure"
```

> [!TIP]
> Run `run test --help` to see additional options.

#### Test Environment

When using the `./run test` command, tests run inside containers, which ensures:

- Consistent environment regardless of host system
- Access to GPU via NVIDIA Container Toolkit
- Isolation from host system dependencies

The `./run` script manages the container environment automatically. For advanced scenarios, tests can also be run directly on the host system (outside containers), but this requires manual setup and configuration.

#### Test Configuration

Test configuration is controlled through:

- **Environment variables**:
  - `HOLOSCAN_INPUT_PATH`: Path to test data
  - `HOLOSCAN_TESTS_DATA_PATH`: Path to test-specific data
  - `PYTHONPATH`: Python module search path
- **Test data**: when a test needs data, point `HOLOSCAN_INPUT_PATH` / `HOLOSCAN_TESTS_DATA_PATH` at the dataset location. The 5.x skeleton tests do not require downloaded data.

#### Reproducing Test Failures

When a test fails (especially from CI), you can reproduce it locally:

1. **Identify the test**: From CI logs or CDash, note the exact test name

2. **Match the build configuration**:

   ```sh
   export HOLOSCAN_BUILD_ARCH=x86_64  # or aarch64, match CI
   export CMAKE_BUILD_TYPE=Release   # match the build being tested
   ```

3. **Run the specific test**:

   ```sh
   # Using run script
   ./run test --name <test_name> --verbose

   # Or with additional CTest options
   ./run test --options "-R <test_name> --verbose --output-on-failure"
   ```

4. **Debug in interactive container** (from build tree):

   ```sh
   ./run launch
   # Inside the container, already in the selected build directory:
   ctest -R <test_name> --verbose --output-on-failure
   ```

   Note: The container is automatically managed by the `./run` script.

5. **Run an example from the install tree**:

   ```sh
   # Launch a container with the install tree mounted
   INSTALL_DIR=$(./run get_install_dir)
   ./run launch "${INSTALL_DIR}"
   # Inside the container, configure an example against this install directory.
   # Each example ships a CMakeLists.min.txt that uses find_package(holoscan).
   cmake -S <example_src_dir> -B <example_build_dir> \
     -DCMAKE_PREFIX_PATH="${PWD}"
   cmake --build <example_build_dir> -j
   ctest --test-dir <example_build_dir> --output-on-failure
   ```

6. **Check test artifacts**: For visual tests (e.g., Holoviz), check for:
   - `*_fail.png`: Actual output that failed
   - `*_ref.png`: Expected reference image

### Linting

Linting is implemented with **[pre-commit](https://pre-commit.com)**. Hooks (Ruff, cpplint, cmakelint, codespell, copyright, clang-format, markdownlint, and standard file checks) are listed in `.pre-commit-config.yaml` at the **git repository root**; pre-commit downloads and caches each hook’s tools the first time they run.

From the build container (or any environment where you run `./run`), use:

```sh
./run lint                # runs pre-commit run --all-files from the repository root
```

`./run lint` automatically resolves `pre-commit`: it prefers [`uvx`](https://docs.astral.sh/uv/guides/tools/) when available (runs in an isolated environment without polluting your Python install), falls back to an existing `pre-commit` on PATH, or pip-installs it as a last resort. It then resolves the git top-level, checks for `.pre-commit-config.yaml` there, and runs all hooks on every tracked file. That matches a full CI-style pass. For faster, commit-oriented runs on staged files only, install hooks with `pre-commit install` and use plain `git commit`, or run `pre-commit run` from the repository root (see [Pre-commit hooks](#pre-commit-hooks)).

> [!TIP]
> For hook-specific options and filtering, see `pre-commit run --help` and [.pre-commit-config.yaml](./.pre-commit-config.yaml).

### Pre-commit hooks

Contributors should enable **[pre-commit](https://pre-commit.com)** so checks run automatically on `git commit`. Use the **git repository root** (the directory that contains [`.pre-commit-config.yaml`](./.pre-commit-config.yaml)).

**Setup** (on the host or in a shell where you commit — not only inside Docker):

```sh
# Option A: using uvx (recommended -- isolated, no pip pollution)
# Install uv first if needed: https://docs.astral.sh/uv/getting-started/installation/
uvx pre-commit install

# Option B: using pip
python3 -m pip install pre-commit
pre-commit install
```

**Run manually** (same as `./run lint` for a full-tree run):

```sh
# Option A: using uvx
uvx pre-commit run --all-files

# Option B: using pip-installed pre-commit
pre-commit run --all-files
```

Run a single hook by id (see the config file), for example:

```sh
pre-commit run ruff-check --all-files
pre-commit run clang-format --all-files
```

**What the hooks cover:**

| Area | Hooks / notes |
|------|----------------|
| Repository hygiene | `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-json`, `check-added-large-files` (standard [pre-commit-hooks](https://github.com/pre-commit/pre-commit-hooks)) |
| NVIDIA SPDX headers | `check-copyright` — runs [`scripts/check_copyright.py`](./scripts/check_copyright.py) |
| Whitespace | `remove-tabs` — tabs replaced with spaces in C++, CMake, Dockerfile, Markdown, Python, and shell sources (third-party trees excluded in config) |
| Python | `ruff-check` (with `--fix`) and `ruff-format` — rules in [`.ruff.toml`](./.ruff.toml) |
| Spelling | `codespell` — may rewrite files (`--write-changes`); settings in [`.codespell.toml`](./.codespell.toml) (`[tool.codespell]`); lines can be ignored with `// codespell-ignore` or `# codespell-ignore` |
| C/C++/CUDA style | `cpplint` and `clang-format` (clang-format version pinned in the mirror repo; binary must be available for that hook) |
| CMake | `cmakelint` |
| Markdown | `markdownlint` — paths and config file are set in `.pre-commit-config.yaml` (alongside `.markdownlint.yaml` in this tree) |

**`check-copyright`:** Implemented by [`scripts/check_copyright.py`](./scripts/check_copyright.py). On `git commit`, pre-commit passes only **staged** paths. For `pre-commit run --all-files`, the script receives a wide file list and intersects it with changes since a default baseline (`origin/main` / `main` or `origin/release/latest` / `release/latest`, chosen from your current branch). Set **`HOLOSCAN_COPYRIGHT_BASE_REF`** or pass **`--intersect-since-ref REF`** to that script to pin the baseline. Run `python3 scripts/check_copyright.py --help` for all options.

**Relationship to `./run lint`:** They run the **same** hooks from the same config. `./run lint` always executes `pre-commit run --all-files` at the git root (entire tree). After `pre-commit install`, `git commit` runs hooks on **staged** files only. Several hooks apply auto-fixes (for example Ruff and codespell); review `git diff` after a full-tree run.

### ABI, Visibility, and Header Guards

Holoscan Core and SDK-owned C++ libraries follow the policy in this section.
Hand-written headers use
`#pragma once`, shared-library targets use hidden default visibility, and public
DSO ABI declarations use CMake-generated export headers such as
`<holoscan/holoscan_core_export.h>`.

HoloMQ is standalone, so its contributor-facing policy lives under the module:
[`docs/holomq/abi_and_exports.md`](./docs/holomq/abi_and_exports.md).
Use that document when changing HoloMQ public headers or ABI.

Fast checks run through `./run lint`. The staged semantic checks use CMake's
`compile_commands.json`:

```sh
cmake --build <build-dir> --target check-export-macro-semantics
cmake --build <build-dir> --target holomq-check-export-macro-semantics
cmake --build <build-dir> --target holomq-check-symbols
```

### Building the User Guide

The source of the user guide hosted at <https://docs.nvidia.com/holoscan/sdk-user-guide> is located in [docs](./docs/). From the **holoscan-sdk repository root**, use the repository wrapper to build or preview with Fern:

```bash
./run build_docs
./run live_docs
```

With no arguments and no `FERN_TOKEN`, these commands reuse a prepared local
API tree or omit the API-reference section when no tree exists. This keeps the
local contributor workflow usable without Fern credentials. Pass
`--with-library-mdx` explicitly to generate the API reference using
`FERN_TOKEN` or `fern login`; other explicit options are forwarded unchanged.

See [docs/README.md](./docs/README.md) for authoring and publishing details.
