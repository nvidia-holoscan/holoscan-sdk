"""
 SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
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
