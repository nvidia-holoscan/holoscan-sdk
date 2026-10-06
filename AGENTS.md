<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Holoscan SDK agent instructions

This file applies to the SDK source tree. Paths and commands here and in
`.ai-rules/` are relative to the directory containing this file, `run`, and
`CMakeLists.txt`. This is the repository root in the public distribution.

Write documentation and agent guidance for readers using this SDK checkout.
Use paths such as `examples/` and `docs/` from this root, and state a working
directory for commands that require another location. Keep the instructions
self-contained and validate their relative links and command paths here.

Read [.ai-rules/hsdk5-coding-conventions.md](.ai-rules/hsdk5-coding-conventions.md)
before changing C++ headers, CMake, Python bindings, FlatBuffers schemas,
HoloMQ, ABI/export behavior, or tests.
Read [.ai-rules/hsdk5-fern-documentation.md](.ai-rules/hsdk5-fern-documentation.md)
before editing the Fern guide, navigation, assets, or documentation tooling.
Public-header Doxygen follows the coding conventions instead of Fern syntax.
Read [.ai-rules/hsdk5-git-commit-rule.md](.ai-rules/hsdk5-git-commit-rule.md)
when creating, rewriting, or reviewing commit messages.

Start SDK workflows with `./run --help` and [DEVELOP.md](DEVELOP.md). Common
commands include `./run build`, `./run test`, `./run bench`, and
`./run launch --run-cmd "..."`. For a non-default build directory, use
`./run launch <build-directory> --run-cmd "ctest --output-on-failure ..."`;
`./run test` selects the default architecture build directory rather than
`CMAKE_BUILD_PATH`. See [benchmark/README.md](benchmark/README.md) for benchmarks.
Before creating a commit, run `./run lint` and fix failures.

HoloMQ is SDK-owned and uses top-level build and CTest tooling. It has no
module-local CMake project, `run` wrapper, or Dockerfile. Use the SDK-level
workflow when changing its build, backend defaults, bindings, or installation.

The guide lives under `docs/`; use
`python3 docs/scripts/build_holoscan_docs.py` and [docs/README.md](docs/README.md).
Run lint and documentation builds sequentially: the local-link scan can race
temporary documentation-build files.

See [.ai-rules/README.md](.ai-rules/README.md) before changing rule storage or
tool-specific entrypoints.
