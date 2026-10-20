# Holoscan C++ Tests

This directory contains SDK-level C++ tests registered with CTest.

Tests are grouped by subsystem. Add core API tests under `core/`, system or
cross-component tests under `system/`, and module-owned tests under the module
tree, such as `public/modules/<module>/tests`.

Use `test_<subject>.cpp` for test filenames. The prefix keeps test files easy
to identify and matches the Python `test_<subject>.py` convention.

Each C++ test target should link only the SDK target under test and the needed
GoogleTest target. Register GoogleTest binaries through `gtest_discover_tests`
so individual test cases appear as CTest tests.
