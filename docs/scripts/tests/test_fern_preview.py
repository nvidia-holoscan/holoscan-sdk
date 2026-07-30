#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

SCRIPT_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from build_holoscan_docs import _write_preview_url  # noqa: E402


class FernPreviewTestCase(TestCase):
    def test_writes_preview_url_to_explicit_file(self) -> None:
        with TemporaryDirectory() as tmpdir:
            preview_url_file = Path(tmpdir) / "nested" / "fern-preview-url"

            _write_preview_url(
                preview_url_file=preview_url_file,
                url="https://nvidia-preview-branch.docs.buildwithfern.com/holoscan",
            )

            assert (
                preview_url_file.read_text(encoding="utf-8")
                == "https://nvidia-preview-branch.docs.buildwithfern.com/holoscan\n"
            )


if __name__ == "__main__":
    main()
