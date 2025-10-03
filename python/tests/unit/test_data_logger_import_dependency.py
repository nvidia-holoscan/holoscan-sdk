"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from holoscan.core import AsyncDataLoggerResource
from holoscan.data_loggers import AsyncConsoleLogger


# For purposes of this test case, do NOT import SimpleTextSerializer explicitly above
class TestAsyncConsoleLoggerImportFix:
    def test_default_initialization(self, app, capfd):
        """Test construction without explicit import of SimpleTextSerializer first."""
        data_logger = AsyncConsoleLogger(app)
        assert isinstance(data_logger, AsyncDataLoggerResource)

        # assert no errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
