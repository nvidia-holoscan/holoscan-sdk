# Stub-generation loader support

`holoscan_add_nanobind_stubs()` uses the selected, unmodified nanobind generator.
The generic `holoscan_add_stub_generation()` API remains available for other
generators. Holoscan owns the CMake rule and scoped native-loader preparation;
no nanobind patch or private launcher API is required.

When available, `nanobind_sanitizer_preload_env()` supplies the sanitizer runtime
environment. The current adapter does not call `nanobind_resolve_python_path()`;
macOS sanitizer interpreter resolution remains a separate follow-up.

On Linux, `CUDA_MODE auto` retains native loading unless only `libcuda.so.1` is
missing. The selected toolkit stub is exposed through a temporary child-only
library search directory. Other missing dependencies and architecture mismatches
remain errors. Only trusted, locally built extensions may be inspected. This
supports introspection, not GPU execution; the stub is never installed or bundled
into a wheel. Use `native` mode to bypass CUDA stub selection.

This supports build-time generation with a Python interpreter that can execute
the target extension. Cross-architecture execution and install-time generation
are outside this contract.
