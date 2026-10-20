# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typing tools for Holoscan provider packages.

These tools ship with the ``holoscan`` package so that a provider always uses the generator that
matches the installed registry. Their command-line options are not a stable API; CMake projects
should call ``holoscan_add_operator_stubs()`` from ``HoloscanStubGeneration.cmake`` instead.

- ``generate_operator_stubs``: write typed proxy stubs for a provider's registered Operators.
- ``check_stub_coverage``: check that a committed stub covers a generated baseline.
- ``check_type_completeness``: enforce pyright's public type-completeness score for a package.

Each tool runs by path without importing ``holoscan``, except the generator, which imports the
registry it describes.
"""
