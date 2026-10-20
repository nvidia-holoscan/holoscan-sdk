# Python bindings

The standalone bindings expose Tensor and DLPack types through `holoscan`.
Build the SDK once with CPython 3.12 and package its completed native artifacts
in stable-ABI wheels for regular CPython 3.12–3.14.

See [DEVELOPMENT.md](DEVELOPMENT.md) for the contributor architecture,
nanobind conventions, and source-owned stub policy.

Typing generation imports the built extension. The CUDA loader helper supports
`auto`, `native`, and `stub` modes through
`HOLOSCAN_PYTHON_STUBGEN_CUDA_MODE`. Auto mode uses a temporary, child-only
`libcuda.so.1` link to the selected toolkit stub only when the driver is missing.
Other missing dependencies fail generation. No driver is installed or packaged.

This loader support permits introspection, not GPU execution.

## Module wheels

SDK binding packages use `HOLOSCAN_PYTHON_MODULE_ROOT` as their build-tree import
root. Generated extension stubs and hand-written facade stubs are installed
with `py.typed`. Wheel creation consumes these artifacts without regenerating
them or rebuilding the SDK.

Each distribution selects an explicit native install component. That component
owns its Python package, typing artifacts, extension, and runtime libraries.
Member wheels declare dependencies on other distributions instead of copying
their files. New wheels add entries below and their own build and smoke checks;
they reuse the shared component frontend and qualification infrastructure.

| Module | Distribution | Python package | Native wheel component | Wheel dependency |
|---|---|---|---|---|
| Core, including built-in Sensor I/O schemas and operator testing | `holoscan-core-cu13` | `holoscan` (private implementation: `holoscan._core`) | `holoscan-wheel-core` | None on another SDK wheel |
| Vision | `holoscan-vision-cu13` | `holoscan_vision` | `holoscan-wheel-vision` | Exact matching `holoscan-core-cu13` version |
| Visualization | `holoscan-visualization-cu13` | `holoscan_visualization` (alias: `holoscan.visualization`) | `holoscan-wheel-visualization` | Exact matching `holoscan-core-cu13` version |
| Inference | `holoscan-inference-cu13` | `holoscan_inference` | `holoscan-wheel-inference` | Exact matching `holoscan-core-cu13` version; external TensorRT 10 runtime |

Metadata-only aggregates install those component distributions. They carry no
import package, no native payload, and no native wheel component, and are built
as `py3-none-any` without a native build tree.

| Distribution | Installs |
|---|---|
| `holoscan-core` | `holoscan-core-cu13` |
| `holoscan-vision` | `holoscan-vision-cu13` |
| `holoscan-visualization` | `holoscan-visualization-cu13` |
| `holoscan-inference` | `holoscan-inference-cu13` |
| `holoscan-cu13` | the CUDA 13 component distributions |
| `holoscan` | `holoscan-cu13` and the unqualified component metawheels |

Wheel creation requires the completed native build tree and its referenced
dependency artifacts. Modules built with the SDK share that build tree;
independently built modules supply their own. `make wheel` packages the
distributions wired into the SDK wheel workflow. A native module is not
automatically a registered wheel project.

The Core wheel includes the production `holoscan.schema.Image` and `Imu` values,
their Stable ABI codecs, and the separately linked `libholoscan_sensor_io` DSO.
Sensor I/O remains a distinct native library and selectable SDK install component;
it is not a separate Python distribution.

The same Core wheel includes `holoscan.testing`, its separate `_testing` extension
and `libholoscan_testing` when `HOLOSCAN_BUILD_TESTING_LIBRARY=ON` (the default).
The release wheel lane requires this payload. A custom testing-disabled Core
build omits it, including stale files from a previous enabled build. Ordinary
native and Python install components remain separately selectable. Importing
`holoscan` does not eagerly import testing, and Core does not link the testing
library. Pytest is a test dependency; no GPU Python array package is required.

## Wheel qualification

Install the exact selected wheel files and their declared wheel dependencies
into the same environment. With Core installed, run:

```bash
python python/smoke_wheel_bindings.py --require-typing
python python/tests/testing/smoke_wheel.py
```

Run the same artifacts under regular CPython 3.12, 3.13, and 3.14. Shared checks
cover ABI tags, typing, dependency consistency, native loading, and import
locations. Each module adds binding-specific checks and verifies that removing
its wheel leaves dependencies usable. These checks do not imply GPU workload
coverage.

## Minimal Vision bindings

With `HOLOSCAN_BUILD_VISION=ON`, the SDK also builds `holoscan_vision._vision`.
It exposes `InterpolationMode` and `ResizeOpParams`.
A private linkage probe constructs and destroys the real C++ `ResizeOp`, without
starting a session. Graph/operator authoring remains with the later binding slices.
CV-CUDA's real runtime libraries are required even for import and stub generation;
only the CUDA driver can use the temporary loader fallback.

With Core and Vision wheels installed (the alias checks require distribution metadata):

```bash
python python/smoke_wheel_bindings.py --require-typing --require-vision
python -m pytest modules/vision/python/tests/test_bindings.py
```

The native binding test also runs against a native installation. Without wheel
metadata, import its canonical `holoscan_vision` package directly.

## Inference bindings

With `HOLOSCAN_BUILD_INFERENCE=ON`, the SDK builds
`holoscan_inference._inference`. The separately owned Inference wheel includes
HoloInfer and `InferenceOp`, depends on the exact matching Core wheel, and leaves
TensorRT 10 and CUDA 13 to the deployment environment. Its discovery entry point
publishes `holoscan.inference` as an alias of the canonical
`holoscan_inference` provider package.

## Discovering independently packaged modules

Extension distributions register a package in their `pyproject.toml`:

```toml
[project.entry-points."holoscan.modules"]
example = "holoscan_example"
```

Core discovers metadata on `import holoscan` and loads the extension only when
`holoscan.example` is requested. The alias and canonical imports share the same
module objects, including nested extensions; `from holoscan import example`
also works. Targets must be module paths, not `package:object` references.
Duplicate registrations, reserved names (including `core`), and existing Core
packages are rejected with a warning. Import failures remain visible when the
extension is requested. Importing an alias from its canonical package while that
package is still initializing is rejected; use relative imports inside the
extension package. Install extensions before starting the interpreter.

This keeps Core's public `holoscan` facade unchanged. Discovery uses only distributions visible
to the interpreter; it does not bypass virtual-environment isolation or configure
the native loader. Wheel RUNPATHs still assume sibling packages in one
`site-packages`; split system/venv native installations need separate validation.
The isolated synthetic-module tests cover Python discovery across both layouts.

## Reusing the CMake helpers

The SDK installs `HoloscanStubGeneration.cmake` and its adjacent runner with the
`holoscan-core` component. Full `find_package(holoscan REQUIRED)` exposes
`holoscan_add_nanobind_stubs` and the lower-level `holoscan_add_stub_generation`.
Consumers pass their own extension targets, Python executable, and import root.
The nanobind wrapper accepts an explicit `GENERATOR_SCRIPT`; otherwise it looks
in the selected Python environment. Including the module does not discover or
install dependencies. Each generation rule owns its outputs and private loader
directory, so downstream projects can reuse it without SDK build-tree paths.

The same installation exposes `holoscan_add_python_wheel_component()` and
`holoscan_add_python_wheel_extension()` from `HoloscanPythonWheel.cmake`. A
standalone module can use them after `find_package(holoscan REQUIRED)` without
an SDK source checkout. Declare the module's own Python files, extension, and
runtime libraries; the helper stages only that payload in a wheel component.
Build extensions and generated stubs before installation. Wheel installation
requires `patchelf`; merely finding the SDK does not discover it or Python.
`holoscan_install_python_module_metadata()` installs the minimal `dist-info`
(`METADATA` and `entry_points.txt`) through which an ordinary CMake install
exposes the module's provider package as `holoscan.<module>`; wheels generate
the same entry point from their `pyproject.toml`. It also writes that metadata
next to the provider package in the build tree, so build-tree imports and tests
resolve `holoscan.<module>` the same way.

For a module added alongside the SDK source build, include `holoscan_core_python`
in `EXTENSION_TARGETS` when generating stubs imports Core. The helpers read
Core's `HOLOSCAN_PYTHON_IMPORT_DEPENDENCIES` target property and wait for its
package initialization, generated schema files, and schema codecs. This works
for modules configured after the SDK Python directory returns; consumers do not
need to read `HOLOSCAN_PYTHON_SCHEMA_IMPORT_DEPENDENCIES` from that directory.
Additional extensions can declare the same property for their own import
prerequisites. It supplies build dependencies, not Python import paths; continue
to pass the appropriate `PYTHON_PATH`. Installed SDK consumers use their own
extension targets and the already-installed Core package.

### Typing a provider's registered Operators

A provider package whose native extension registers Operators should generate
their typed proxy stub with the SDK's generator instead of writing or vendoring
one. Operators written as Python `hs.Operator` subclasses are typed by their own
source and need no generated stub. The generator
and its checks ship inside the `holoscan` package as `holoscan._tools`, so they
always match the registry they describe. `holoscan_add_operator_stubs` runs them:

```cmake
find_package(holoscan REQUIRED)
holoscan_add_operator_stubs(
    TARGET generate-example-operator-stubs
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/operators/__init__.pyi"
    PROVIDER_MODULES holoscan_example
    REGISTRY_PREFIX holoscan.example.
    HANDLE_ANNOTATIONS holoscan.example.DeviceChannel=holoscan_example.DeviceChannel
    MODULE_DOC "Registered Holoscan Example Operator proxies."
    HEADER_FILE "${CMAKE_CURRENT_SOURCE_DIR}/stub_header.txt"
    PYTHON_EXECUTABLE "${Python3_EXECUTABLE}"
    PYTHON_PATH "${example_python_root}"
    EXTENSION_TARGETS example_python
    COMMITTED_STUB "${CMAKE_CURRENT_SOURCE_DIR}/holoscan_example/operators/__init__.pyi"
    CHECK_TARGET check-example-operator-stubs)
```

- `PROVIDER_MODULES` are imported to register the rows that `REGISTRY_PREFIX` selects.
- Every constructor-handle type needs a `HANDLE_ANNOTATIONS` entry, except the SDK's built-in
  `holoscan.VoidCallback`.
- `HEADER_FILE` replaces the SDK's license header with the provider's comment lines.
- `CHECK_TARGET` fails unless the committed stub declares every generated name exactly as
  generated, including the private parameter and proxy types. It compares declarations, not
  formatting or imports, and the committed stub may add declarations and export more names. A
  provider whose build options register different Operators can therefore commit one stub that
  declares all of them. The check is not part of `ALL`, so a CI lane or test should build it.

The function finds the tools beside the `holoscan` package that `PYTHON_PATH`,
followed by the installed SDK's `holoscan_PYTHON_PACKAGE_DIR`, selects, without
importing it, and runs the generator with the same import path. Set
`HOLOSCAN_PYTHON_TOOLS_DIR` to override the tools' location. Use
`holoscan_python_tools_directory()` to locate `check_type_completeness.py` for a
type-completeness test. Outside CMake, run
`python -m holoscan._tools.generate_operator_stubs --help`. The tools' command-line
options are private and may change between SDK releases; the CMake functions are
the supported interface.

## Typed facade development

The Holoscan Python package is a typed facade over the Core authoring
extension. Public runtime modules live under `holoscan`; their committed `.pyi`
files are source-owned refinements checked against a generated nanobind
baseline. The rationale and maintenance model are documented in
[DEVELOPMENT.md](DEVELOPMENT.md#decision-source-owned-public-facade-stubs).

### Type-checking tools

Development images install exact checker versions from `requirements.dev.txt`:

- pyright is the primary checker and the source of the public type-completeness
  score;
- mypy is used only for `stubtest` runtime/stub agreement;
- ty is the second checker for focused inference fixtures.

The Python API owners review these pins once per SDK minor release. A pin bump
must rerun the capability matrix below, the type-completeness and `stubtest`
gates, and the positive and negative typing fixtures. Checker-specific
suppressions are avoided in shared code; use a narrowly scoped
`# type: ignore[code]` only when the diagnostic is understood and unavoidable.

Run the configured source check from `public/`:

```bash
pyright
```

The build-tree semantic checks run through CTest after building
`holoscan-python`. Regenerate and compare committed stubs with:

```bash
cmake --build build-cu13-x86_64 --target check-python-stubs
```

### Checker capability evaluation

The checker capability evaluation used ty 0.0.78, pyright 1.1.411, and mypy 2.3.1. The
results are part of the pin contract:

| Required API shape | ty 0.0.78 result |
|---|---|
| PEP 695 generic descriptors | Pass |
| `Unpack[TypedDict]`, including nested and optional keys | Partial: value types pass; unexpected keywords are not diagnosed |
| Generic `Graph.op` overload selection | Pass |
| Static stubs beside module `__getattr__` and `__dir__` | Pass |
| `Final`, `assert_type`, `override`, and `Self` | Pass |
| Per-path rule overrides | Pass |
| Stable machine-readable diagnostics | Pass: GitLab JSON output |
| LSP hover and completion on generated handle types | Pass |

Because the `Unpack[TypedDict]` check is incomplete, ty does not satisfy the
promotion criteria for the primary checker at this pin. Pyright is authoritative
for those keyword shapes; ty remains in the focused two-checker fixtures so a
future pin can demonstrate promotion without changing the API contract.
