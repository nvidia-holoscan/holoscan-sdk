# Holoscan utility scripts

This folder contains developer utilities and source-tree validation helpers.
Documentation is provided for the following scripts:

- [`build_analysis.py`](#build_analysispy)
- [`check_copyright.py`](#check_copyrightpy)
- [`ctest_time_comparison.py`](#ctest_time_comparisonpy)
- [`download_ngc_data`](#download_ngc_data)
- [`get_cmake_cuda_archs.py`](#get_cmake_cuda_archspy)
- [`graph_surgeon.py`](#graph_surgeonpy)
- [`install-pytorch.py`](#install-pytorchpy)
- [`list_ld_dependencies.sh`](#list_ld_dependenciessh)
- [`test_pattern_validation.py`](#test_pattern_validationpy)
- [`test_run_script.py`](#test_run_scriptpy)

> Note: only the utilities selected in [`CMakeLists.txt`](CMakeLists.txt) are
> included in the SDK installation at `/opt/nvidia/holoscan/bin`. Repository
> validation helpers remain in the source tree.

____

## build_analysis.py

Measure where SDK build time goes, using an existing Ninja build directory. It reports:

- CPU and wall time by area, the slowest steps, concurrency, and the critical path, from
  `.ninja_log` and `build.ninja`;
- which objects depend on a header and the rebuild cost of editing it, from `ninja -t deps`;
- sources compiled more than once, with their flag variants, from `compile_commands.json`;
- the static include closure of public headers;
- the standalone parse cost of an include, clang `-ftime-trace` attribution, and the GCC
  front-end/back-end phase split, by invoking the recorded compilers.

The compiler-driven subcommands (`header-cost`, `ftime-trace`, `ftime-report`) run inside the SDK
build container. They accept at most 8 parallel jobs.

### Usage

A clean measurement uses a fresh build directory without a compiler cache:

```bash
./run build --sccache false --buildpath build-timing --installprefix install-timing --parallel 8
python3 scripts/build_analysis.py ninja-log build-timing
python3 scripts/build_analysis.py duplicates build-timing
python3 scripts/build_analysis.py closure holoscan/core/graph.hpp holoscan/core/diagnostic.hpp
```

Header fan-out needs the dependency log that Ninja records:

```bash
./run launch build-timing --run-cmd "ninja -t deps > deps.txt"
python3 scripts/build_analysis.py deps build-timing --deps-file build-timing/deps.txt \
    --header holoscan/core/graph.hpp
```

Compiler-driven measurements run from the build directory inside the container:

```bash
./run launch build-timing --run-cmd \
    "python3 ../scripts/build_analysis.py header-cost . holoscan/core/graph.hpp --jobs 4"
./run launch build-timing --run-cmd \
    "python3 ../scripts/build_analysis.py ftime-trace . src/core/graph.cpp --out traces"
python3 scripts/build_analysis.py trace-report build-timing/traces
```

Add `--json <file>` before the subcommand to also write machine-readable results.
Compiler-driven commands return a nonzero exit status if any measurement fails,
print the failed unit and compiler diagnostics to stderr, and retain partial
results in the JSON output. A failed unit is not a zero-cost measurement.

____

## check_copyright.py

This script adapts an NVIDIA copyright header checker from the RAFT project. It can be used to automatically update the copyright year and can also optionally run on CI and only check the files that have been modified.

The script `check_copyright.py` file checks for NVIDIA copyright headers in files matching any of the lists of regular expressions in the `files_to_check` list.
There are also regular expressions for the copyright string that is being checked. The `CheckSimple` one is for when there is only a single year and `CheckDouble` is for when the header uses a year range. The `gitutils.py` defines some utilities for interacting with Git that enables the `--git-modified-only` capabilities of the copyright checker.

### Usage

You can call the script from the repository, specifying to check all matching files within the include and src subfolders.

```bash
python scripts/check_copyright.py include src
```

Alternatively, it can be used to check only modified files in a CI script using ([example from RAFT here](https://github.com/rapidsai/raft/blob/4f22957db9a38d81d0cd6ad864507c47d243ad56/ci/checks/style.sh#L29)):

```bash
python ci/checks/copyright.py --git-modified-only
```

There is also a `--update-current-year` option to allow update of the copyright year in all files and ``-exclude`` option to allow user-specified regular expressions for exclusion from the check.

____

## check_light_headers.py

Checks that the light Core headers listed in [core_light_headers.txt](core_light_headers.txt)
(vocabulary leaves and headers that use only vocabulary from the authoring stack) do not reach any
listed heavy header (Graph and Operator authoring, the Parameter Schema and Tensor Contract DSLs,
Tensor storage, and the logger) through their Holoscan includes, directly or transitively. A
failure prints the include chain. Headers the manifest does not list are not checked.

The pre-commit hook `check-core-light-headers` runs it when a Holoscan header or the manifest
changes. To run it directly, from the SDK root:

```bash
python3 scripts/check_light_headers.py
```

## ctest_time_comparison.py

This script compares time from different tests to make sure they match expected timing comparison.

### Usage

`python3 ./scripts/ctest_time_comparison.py <filename> "TEST1" "LESS" "TEST2"`
____

## download_ngc_data

Download and unzip datasets from NGC.

## Prerequisites

- `wget` or `curl` to download datasets using curl. Note that as of Jan 2025, download full zip archives from NGC is not supported.
- [NGC CLI](https://ngc.nvidia.com/setup/installers/cli) to download datasets using the `<org>/[team]/<name>:<version>` format (useful if the above fails or for private registries)

## Usage

The example below will download and unzip the RacerX video from NGC:

```sh
./scripts/download_ngc_data \
  --url nvidia/clara-holoscan/holoscan_racerx_video:20231009 \
  --download_dir ./data \
  --download_name racerx
```

Use `--help` for options such as output directory, output name, and checksum validation.

____

## get_cmake_cuda_archs.py

Determines and formats a semicolon-separated list of CUDA architectures suitable for CMake's `CMAKE_CUDA_ARCHITECTURES` variable. It uses `nvcc -code-ls` to find supported architectures and then filters them. By default, it excludes architectures primarily for specific features (e.g., `sm_89` for FP8) or platform-specific (e.g., `sm_XXa`, `sm_XXf`) unless explicitly requested by the user with a specific flag. The filtering also considers minimum architecture requirements and platform compatibility (e.g., removing iGPU architectures on x86_64). The final list includes each selected architecture suffixed with `-real` (for SASS generation). It also adds one `-virtual` architecture entry based on the highest non-specific (no 'a' or 'f' letter suffix) architecture chosen; if no such base architecture is found, no PTX target is added.

### Usage

```sh
python3 scripts/get_cmake_cuda_archs.py <requested_archs> [options]
```

**Positional Arguments:**

- `requested_archs`: Defines the target architectures. Can be:
  - `all`: Use all nvcc supported, platform-compatible, and non-feature-specific architectures.
  - `all-major`: Use only major versions from the `all` selection (e.g., 70, 80, 90).
  - `native`: Passes the string "native" directly through (for CMake to detect).
  - A comma or space-separated list of specific architecture numbers (e.g., `'75 86 90a'`).

**Options:**

- `--nvcc-path <path>`, `-n <path>`: Path to the `nvcc` executable. Defaults to searching PATH, then `/usr/local/cuda/bin/nvcc`.
- `--min-arch <num>`, `-m <num>`: Minimum major CUDA architecture to consider (e.g., `70` for Volta and newer). Set to `0` or omit to disable.
- `--allow-specific-archs`, `-f`: Allow feature-specific architectures (e.g., `sm_89`, `sm_XXa/f`) when they are explicitly listed in `requested_archs`. This flag does not affect the `all` or `all-major` selections, which always exclude these.
- `--verbose`, `-v`: Enable verbose debug logging to stderr.

### Examples

Get all supported architectures for the current platform, with a minimum of sm_70:

```sh
python3 scripts/get_cmake_cuda_archs.py all --min-arch 70
```

Get specific architectures:

```sh
python3 scripts/get_cmake_cuda_archs.py "75 86 90a"
```

____

## graph_surgeon.py

When converting a model from PyTorch to ONNX, it is likely that the input of the model is in the form NCHW (batch, channels, height, width), and needs to be converted to NHWC (batch, height, width, channels). This script performs the conversion and generates a modified model.
Note that this script modifies the names of the output by appending `_old` to the input and output.

### Usage

```bash
python3 scripts/graph_surgeon.py input_model.onnx output_model.onnx
```

____

## install-pytorch.py

Installs PyTorch so that every CUDA library it loads is the system copy that Holoscan also
uses. The SDK build image (`Dockerfile`, `pytorch-dev` stage) installs PyTorch with it as
`holoscan-install-pytorch`. See the user guide's *Installing PyTorch* page
(`docs/pytorch_installation.mdx`) for when to use it.

PyTorch wheels declare their CUDA runtime as pip wheels (`cuda-toolkit[...]` and
`nvidia-*-cu13`). Installed next to the system CUDA stack, those wheels take precedence
through the `torch` libraries' RUNPATH, so one process loads two versions of libraries such
as cuBLAS. The helper prevents this:

- `torch` is installed with `--no-deps` from `download.pytorch.org` (or `--wheel-dir`).
- Pure-Python dependencies (`sympy`, `networkx`, `cuda-bindings`, ...) are installed with pip.
- CUDA dependencies must already be installed as system libraries, at least at the version
  the wheel pins. Otherwise the helper fails before installing anything and lists the APT
  packages to install. An unrecognized CUDA dependency is an error.
- `triton` is skipped; only `torch.compile` needs it.
- Bundled CUDA libraries are removed. Bundled NVPL libraries (aarch64) are kept unless the
  system already provides the same library, in which case the bundled copy is removed.
- Kineto references are removed from Torch's CMake configuration.

**Supported systems:** x86_64 and SBSA (aarch64) with CUDA 13.x, including Tegra iGPU
systems on the SBSA stack (JetPack 7 and later). CUDA 12 and JetPack 6 are rejected. The
default version is `torch==2.13.0`; the CUDA tag is the newest available tag that is not
newer than the system CUDA (`+cu130` on CUDA 13.0 and 13.1, `+cu132` on CUDA 13.2 and
later).

### Usage

```bash
python3 scripts/install-pytorch.py install             # install, then run "check"
python3 scripts/install-pytorch.py check --cmake       # validate an existing installation
```

- **Install options:** `--version`, `--index-url` or `--wheel-dir DIR` (install the `torch`
  wheel from a local directory; pure-Python dependencies still come from the configured pip
  index), `--skip-check`.
- **`check`:** fails if pip CUDA packages or libraries are installed, if `ldd` cannot resolve
  a `torch` library or resolves a CUDA library to a pip location, or if `import torch` maps
  a CUDA library from a pip location. With a GPU it also runs a matmul and a cuDNN
  convolution; `--require-gpu` fails when no GPU is present, and `--cmake` builds and runs a
  minimal `find_package(Torch)` consumer.

The helper installs only `torch`. Install `torchvision`, `torchaudio` and model packages
afterwards without letting pip pull CUDA wheels (for example,
`pip install --no-deps torchvision==<matching version>`), then run `check` again.

`test_install_pytorch.py` holds the helper's unit tests; the `test-install-pytorch`
pre-commit hook runs them.

____

## list_ld_dependencies.sh

This script is used to list the shared library dependencies of the Holoscan SDK and its components. It does this by iterating through a set of directories managed by the SDK and running the `ldd` command on each shared library (`.so` file) found within. It then filters out system libraries and libraries with known paths, and prints out the remaining dependencies.

The script also maintains a map of libraries to their dependencies, as well as a map of dependencies to the libraries that depend on them. This allows the script to print out a list of dependencies for each library, as well as a list of libraries that depend on each dependency.

The script can be run with no arguments to list the dependencies of all libraries in the directories managed by the SDK, or it can be run with a specific library path to list only the dependencies of that library.

### Prerequisites

- The script must be run on a Linux system with the `ldd` command available.

### Usage

baremetal:

  ```sh
  ./scripts/list_ld_dependencies.sh
  ```

container:

  ```sh
  # build container
  ./run launch . --run-cmd ./scripts/list_ld_dependencies.sh -s

  # dev container
  docker run --rm -it \
    --runtime=nvidia \
    --entrypoint=bash \
    -v $(pwd)/scripts/list_ld_dependencies.sh:/list.sh \
    nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu \
    /list.sh
  ```

____

## test_pattern_validation.py

Validates required and prohibited regular expression patterns against a log file or standard input. CI uses this helper when a single command output needs multiple pass/fail checks.

### Usage

See `python3 ./scripts/test_pattern_validation.py --help`

____

## test_run_script.py

Validates the Git-derived image tags used by `run` from a Git checkout and the
`VERSION` fallback used by source archives. The test is a source-tree
regression helper used by pre-commit; it is distributed with the public source
but is not installed with the SDK.

It also exercises the `build_image`, `build`, `test`, and `launch` command
paths with `--dryrun --standalone`, using deterministic host-command stubs so
the checks do not require Docker or an existing SDK build.

Run it from the root of a Git checkout of the public source tree (the `public/`
directory in the internal repository):

```sh
python3 scripts/test_run_script.py
```

The test creates temporary standalone and nested source-archive layouts. It
does not require a CMake configuration, build directory, or third-party Python
packages.
