"""
SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""  # noqa: E501

from holoscan import operators


def test_import_operator_classes():
    for o in operators._OPERATORS:
        getattr(operators, o)
        print(f"✅ Loaded {o}")


def test_import_operator_modules():
    for m in operators._OPERATOR_MODULES:
        getattr(operators, m)
        print(f"✅ Imported {m}")
