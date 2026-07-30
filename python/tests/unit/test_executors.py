"""
SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""  # noqa: E501

import pytest

from holoscan.core import Executor
from holoscan.executors import GXFExecutor


class TestGXFExecutor:
    def test_fragment(self, app):
        executor = GXFExecutor(app)
        assert executor.fragment is app

    def test_context(self, app):
        executor = GXFExecutor(app)
        assert type(executor.context).__name__ == "PyCapsule"

    def test_type(self, app):
        executor = GXFExecutor(app)
        assert isinstance(executor, Executor)

    def test_dynamic_attribute_not_allowed(self, app):
        obj = GXFExecutor(app)
        with pytest.raises(AttributeError):
            obj.custom_attribute = 5
