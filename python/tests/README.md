# Holoscan Python Tests

This directory contains Python tests registered with CTest through pytest.

Tests are grouped by module or subsystem. Add bindings for the core package
under `core/`, cross-component Python tests under `system/`, and future
module-specific tests under a matching module folder.

Use `test_<subject>.py` for test filenames so pytest discovers them naturally
and so Python and C++ test files follow the same prefix convention.

When a Python binding calls C++ code that writes directly to stdout or stderr,
prefer pytest's `capfd` fixture over `capsys` so file-descriptor output is
captured reliably.
