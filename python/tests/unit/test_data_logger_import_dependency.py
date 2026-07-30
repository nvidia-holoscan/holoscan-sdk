"""
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
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
