# Python binding development

This document describes the contributor-facing architecture and conventions for
the Holoscan Python bindings. See [README.md](README.md) for the build, package,
and module-discovery overview.

## Architecture

Holoscan Core presents a public Python facade over a private nanobind extension:

| Area | Responsibility |
|---|---|
| `holoscan/core/*.cpp` | Bind the public C++ API into the private `holoscan._core` extension. |
| `holoscan/core/_core.cpp` | Assemble the extension by calling the domain-specific binding functions. |
| `holoscan/*.py` | Define the public runtime modules, facade behavior, and Python-only ergonomics. |
| `holoscan/_public_api.py` | Declare the fixed public names exported by each facade. |
| `holoscan/**/*.pyi` | Define the source-owned public static type contract. |
| `holoscan/cmake/` | Stage the package and declare its wheel payload. |
| `tests/core/` | Test runtime behavior, installed-package behavior, and runtime/stub agreement. |
| `tests/typing/` | Test positive and negative static inference with the supported type checkers. |

The native extension is an implementation detail. Applications import public
objects from `holoscan` or its facade modules, never from `holoscan._core`. The
facades re-export the same native objects rather than wrapping them, preserving
Python type and instance identity.

The runtime namespace, binding behavior, and static contract have different
source representations:

- the nanobind code defines runtime behavior and callable semantics;
- `_public_api.py` defines which fixed names each facade exports; and
- the adjacent `.pyi` files define the public typing contract.

Tests and generation checks keep those representations consistent.

## Decision: source-owned public facade stubs

The Core facade `.pyi` files are source-owned and reviewed like other public API
files. Nanobind stub generation remains a validation input; its output is not the
authoritative shipped facade.

### Why generation alone is insufficient

The public API is intentionally richer than a direct dump of the extension:

- Public objects live in facade modules while their storage lives in the private
  `_core` extension. Raw extension stubs can expose private module paths or form
  import cycles instead of describing the supported imports.
- APIs such as `Graph.op` and `Graph.embed` use overloads, generics, `TypedDict`,
  `Unpack`, and handle-preserving return types that native introspection cannot
  infer completely.
- Registry-backed names depend on which providers are loaded in the generating
  process. A wheel needs one provider-neutral Core contract, not a snapshot of a
  developer or test process.
- Lazy optional-module discovery and deliberate aliases need a stable static
  view that does not import optional providers.
- Generating every install artifact requires importing a newly built extension.
  That is fragile for cross builds and adds native-loader and CUDA-driver concerns
  to otherwise mechanical install and packaging steps.
- A checked-in stub makes a public signature change visible in ordinary code
  review and allows typing behavior to change in the same commit as the runtime
  API.

These properties make source ownership worth the duplication. The `.pyi` file
is not merely a cache of generated text; it is the reviewed static expression of
the Python API.

### How maintenance cost is controlled

Generation was retained to catch omissions and runtime drift:

1. `generate-python-facade-stubs` imports the built extension, runs the pinned
   nanobind generator for each public facade, and normalizes the result under
   `python/generated-stubs/` in the build tree.
2. `check-python-stubs` uses `check_stub_coverage.py` to require every generated
   public name and class member in the corresponding source-owned stub.
3. `python-core.stubtest` compares the installed package with the shipped stubs.
4. `python-core.type_completeness` requires 100 percent pyright public type
   completeness.
5. The pyright and ty fixtures exercise higher-order inference that symbol
   coverage and `stubtest` cannot prove.
6. Package and wheel tests require `py.typed` and a stub for every public facade.

The coverage check is deliberately not a byte-for-byte comparison: source-owned
stubs may contain useful overloads and generic structure that the generator
cannot produce. It also means the coverage check alone cannot prove signature
equivalence; `stubtest` and focused typing fixtures remain required.

### Updating registry-backed provider stubs

Registry-backed provider stubs are generated as complete candidates in the
build tree, but the generator does not overwrite or merge the committed
source-owned stub. For example, after adding or changing a Vision operator:

1. Rebuild the provider bindings and generate the candidate:

   ```bash
   cmake --build <build-dir> --target generate-holoscan-vision-python-stubs
   ```

2. Compare the committed stub with
   `<build-dir>/python/generated-stubs/holoscan/holoscan_vision/operators/__init__.pyi`.
   Merge only the relevant generated changes into
   `modules/vision/python/holoscan_vision/operators/__init__.pyi`; do not replace
   the committed file blindly. Preserve intentional overloads, annotations, and
   other manually refined declarations for existing operators.
3. For a new operator, merge its `__all__` entry, parameter `TypedDict`
   definitions, handle class, proxy class, and exported proxy variable. Keep
   generated declarations in their stable lexical order.
4. Check coverage and the installed runtime-to-stub contract:

   ```bash
   cmake --build <build-dir> --target check-holoscan-vision-python-stubs
   ctest --test-dir <build-dir> \
     -R '^python-vision\.(operators|stub_surface|type_completeness|stubtest)$' \
     --output-on-failure
   ```

The coverage check is asymmetric: every generated public name and class member
must appear in the committed stub, while additional source-owned refinements are
allowed. Review the candidate diff as well because coverage alone does not
compare signatures or documentation text.

Source ownership is the default for curated public facades. A module whose
runtime API is itself generated from a schema may use generated stubs when the
same schema and generator author both artifacts. Such a module must still ship
deterministic stubs and pass the installed-surface checks.

## Nanobind conventions

- Build extension modules with `nanobind_add_module(... STABLE_ABI ...)` and keep
  the Python 3.12 stable-ABI floor. Supported wheels must use an `abi3` suffix.
- Keep `_core.cpp` as assembly code. Put each API domain in its own `.cpp` and
  `.hpp` pair and expose an `init_*` function to the module initializer.
- Keep binding code thin. Validation, planning, and runtime behavior belong in
  the public C++ implementation unless the behavior is inherently Python-only.
- Bind public classes, enums, and functions with the helpers in
  `core/module_name.hpp`. They make nanobind record the canonical facade module
  while retaining the private extension as the storage module.
- Name every public parameter with `nb::arg`, and use `nb::kw_only()` and
  `.none()` where those are part of the Python call contract.
- Add `nb::sig` only when nanobind cannot infer the intended public annotation.
  Keep it consistent with the source-owned stub and test the behavior it
  describes.
- Choose return-value and lifetime policies explicitly for borrowed or nested
  objects. `nb::rv_policy::reference_internal` is appropriate only when the
  returned object's lifetime is genuinely tied to the parent.
- Register common exception translation and public error types through the
  existing error-binding path rather than translating failures independently in
  each domain.
- Before initializing a registry-backed operator catalog, explicitly initialize
  the component-owned payload catalogs used by its ports. The payload owner
  should export an idempotent registration function; a provider module consumes
  that function instead of publishing a duplicate payload row, value, or codec.
- Put longer binding docstrings in the existing `*_pydoc.hpp` pattern when that
  keeps a binding definition readable.

### Callback authority and interpreter ownership

- Treat callback-scoped ports, contexts, selections, and loans as capabilities,
  not merely borrowed pointers. Their shared activation state must validate both
  the active generation and the exact callback thread before dereferencing Core
  state or releasing the GIL.
- Invalidate callback authority on every normal and exceptional exit. Retaining
  a Python facade may retain descriptive metadata or an owning sample, but it
  must not extend access to callback-scoped Core state.
- Release the GIL around blocking Core operations only after callback authority
  and Python inputs have been validated. Revalidate a generation after a native
  call when the result creates another callback-scoped view.
- Keep selected batches lazy and callback-scoped. Length queries must not
  materialize payloads; indexing, iteration, and slicing materialize only the
  requested owning samples, which may outlive the batch view.
- Route `emit` and `forward` metadata through the Core merge and stamping paths.
  Convert publication options before releasing the GIL, and revalidate callback
  authority after a native call that returns another borrowed capability.
- Destroy Python-owned state only while the interpreter is live and the GIL is
  held. Background cleanup workers may stop and join native sessions, but must
  hand successfully retired ownership back to an interpreter-owned entry point
  for normal-time reclamation.
- Explicit CUDA Tensor copies exposed to Python must not outlive an untracked
  source read. The current Core fallback completes the copy before returning and
  still records destination readiness and deallocation streams. Do not make this
  path asynchronous without an owner that retains the source through completion.
- Project structural Tensor representations, bounds, and byte-span expressions
  as thin bindings over the public Core value model. Python convenience
  conversion must not duplicate normalization, capacity derivation, or plan
  evidence rules.
- Treat run controls and every capability derived from them as process-bound.
  Public operations must reject a different process ID, and destructors must
  abandon inherited native ownership without acquiring locks that may have been
  held by vanished threads at `fork()`. The owner process remains responsible for
  exactly-once native cleanup.
- Preserve semantic no-change behavior when a Python control updates lifecycle-
  mutable aggregates. Compare scalars through their canonical carriers, optionals
  by presence and value, nested objects recursively through their declared
  schemas, and strings or sequences through exact field equality. Equal candidates
  must not schedule lifecycle callbacks or advance the published generation.

### Runtime clock references

`LifecycleContext.resolve_clock()` accepts an optional authored `ClockKey`, and
`ExecutionContext.clock()` returns the active compute clock. Both use the same
read-only `holoscan.clocks.ClockRef` facade. Omitting the lifecycle key selects
the admitted default clock. Lifecycle resolution must always use the callback
Operator's owning Graph: equal import names in two embedded Graph instances may
be bound to different parent clocks.

```python
# Inside a reusable Graph's compose(); this key belongs to this Graph instance.
sensor_clock_import = self.import_clock("sensor")

class SensorSource(hs.Operator):
    def warm_up(self, context: hs.lifecycle.LifecycleContext):
        default_clock = context.resolve_clock()
        sensor_clock = context.resolve_clock(sensor_clock_import)
        self.clock_ids = (default_clock.id, sensor_clock.id)
        self.started_at = default_clock.now()
        return hs.lifecycle.LifecycleStatus.OK
```

The `ClockRef` does not extend its registry generation. Its `id`, immutable
`capabilities`, and `at_ns()` labelling operation remain diagnostic values after
the generation becomes stale, while `now()` then raises the translated Core
error. Keep numeric clock identifiers scoped to the plan/catalog that produced
them; do not add a bare identifier lookup without exposing the corresponding
catalog generation and provenance.

### Generated schema bindings

- Use Python 3.12 or newer for the shared schema generator, including native-only
  schema generation. Build-time Python utilities, generated Python extensions,
  and packages share the Core wheel's Python floor.
- Generate the runtime dataclass facade, registration extension, and stub from
  the same schema model. This is the schema-owned exception to source-owned Core
  facade stubs; deterministic generation and installed-surface checks remain
  required.
- Use BFBS `declaration_file` provenance and explicit external-type mappings to
  assign each reachable type to exactly one generated module. Fail generation
  when an included type has no declared owner; in particular, reuse the
  canonical `holoscan.schema.Tensor` mapping instead of regenerating it.
- Set `PYTHON_INSTALL_COMPONENT` on each production schema declaration. That
  metadata is the single inventory used for codec build ordering, native
  installation, tests, and the Core wheel; do not enumerate built-in schema
  targets again in consumer CMake files.
- Keep `PythonSchemaNativeTraits` specializations in the generated codec
  translation unit, where their symbols remain hidden. Loading a generated
  facade must replace the matching native registry row as one coherent unit;
  do not publish partial combinations of native and Python codec callbacks or
  move the specialization into a shared traits header.
- Publish production built-in schema values, including `Image` and `Imu`, below
  `holoscan.schema`. They form one explicit Core-wheel payload even when their
  native implementation remains a separately linked library and selectable SDK
  install component. Keep the wheel file list explicit and qualify the bundled
  native-loader closure whenever that payload changes.
- Keep the Core value modules for `Header`, `String`, `TensorPacket`, and `Time`,
  plus every installed component-owned value such as `Image` and `Imu`, eager
  below `import holoscan`. Importing each generated codec upgrades its
  pre-registered native row with the Python facade callbacks needed by a
  token-declared port; making these imports lazy silently breaks identity-only
  emit and receive. Discover optional modules before importing them so a
  componentized native install without Sensor I/O remains valid.
- Treat `holoscan.schema.X` as the generated Python value class and
  `holoscan.payload.schema.X` as an identity token. Class-declared inputs
  materialize the class, while token-declared schema inputs return the plain
  mapping form. Tokens must remain non-constructible and point users to the
  corresponding value namespace in their diagnostic.
- The SDK owns the registry rows for its built-in schema identities, including
  `holoscan.schema.Header`. Applications must not register another payload under
  those canonical names; duplicate registration is rejected rather than merged.
- Treat schema Tensor bundles as one publication capability. Every terminal
  path, including destruction before context-manager entry, must invalidate all
  escaped Tensor aliases before abandoning native write guards and pool loans.

## Python facade conventions

- Export fixed native names through `_public_api.py`; do not duplicate ad hoc
  export lists in individual facade modules.
- Keep implementation-only imports and variables private with a leading
  underscore. Public modules commonly use
  `from __future__ import annotations as _annotations`: the future import enables
  postponed annotation evaluation, while the private alias prevents the imported
  `_Feature` object from looking like public API.
- Avoid importing optional array or provider packages from `import holoscan`.
  Optional providers are discovered from metadata and loaded only on request.
- Keep public facade `__all__`, `_public_api.py`, runtime exports, and the
  corresponding `.pyi` in agreement.
- Do not expose `holoscan._core` in annotations, signatures, representations, or
  generated documentation.

## Changing the public API

For a new or changed binding:

1. Implement the runtime binding in the appropriate domain file and initialize
   a new domain from `_core.cpp` when necessary.
2. Assign every public native object to its canonical facade with the helpers in
   `module_name.hpp`.
3. Update `_public_api.py` and the runtime facade when the fixed export set
   changes.
4. Update the source-owned `.pyi` in the same commit. Preserve overloads,
   generics, keyword-only parameters, optionality, and return types intentionally.
5. Add runtime and typing tests. Update `stubtest_allowlist.txt` only for an
   understood limitation that cannot be represented or checked directly.
6. When adding a package file, update both the build/install lists in
   `holoscan/CMakeLists.txt` and the wheel payload in `holoscan/cmake/wheel.cmake`.

Build and run the focused conformance checks through the SDK development
container from the repository root:

```bash
HOLOSCAN_CUDA_MAJOR_VERSION=13 ./public/run launch build-cu13-x86_64 \
  --run-cmd "cmake --build . --target holoscan-python check-python-stubs && \
    ctest --test-dir python/tests/core \
      -R '^python-core\\.(type_completeness|stubtest|ty|typing)$' \
      --output-on-failure"
```

Before committing, run the repository lint in the workstation RAPIDS
environment:

```bash
conda run -n rapids-26.04-py313 ./public/run lint
```
