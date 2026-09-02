#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

SCRIPT_DIR = Path(__file__).parents[1]
DOCS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from build_holoscan_docs import (  # noqa: E402
    FERN_API_VERSION,
    _fern_generate_docs_command,
    _write_preview_url,
)


class FernPreviewTestCase(TestCase):
    def test_fern_pins_are_consistent(self) -> None:
        config_version = json.loads(
            (DOCS_ROOT / "fern" / "fern.config.json").read_text(encoding="utf-8")
        )["version"]
        dockerfile = (DOCS_ROOT / "Dockerfile").read_text(encoding="utf-8")
        docker_version_match = re.search(
            r"\bfern-api@([0-9]+\.[0-9]+\.[0-9]+)\b",
            dockerfile,
        )
        readme = (DOCS_ROOT / "README.md").read_text(encoding="utf-8")
        readme_version_match = re.search(
            r"\bnpm install -g fern-api@([0-9]+\.[0-9]+\.[0-9]+)\b",
            readme,
        )

        assert docker_version_match is not None
        assert readme_version_match is not None
        pinned_versions = {
            FERN_API_VERSION,
            config_version,
            docker_version_match.group(1),
            readme_version_match.group(1),
        }
        assert pinned_versions == {FERN_API_VERSION}

    def test_remote_preview_command(self) -> None:
        assert _fern_generate_docs_command(
            fern_exe="fern",
            preview_id="source-branch",
            force=True,
        ) == [
            "fern",
            "generate",
            "--docs",
            "--preview",
            "--id",
            "source-branch",
            "--force",
        ]

    def test_production_command(self) -> None:
        assert _fern_generate_docs_command(fern_exe="fern") == [
            "fern",
            "generate",
            "--docs",
        ]

    def test_environment_substitution_is_disabled_in_docs_config(self) -> None:
        docs_yml = (DOCS_ROOT / "fern" / "docs.yml").read_text(encoding="utf-8")
        assert re.search(r"(?m)^\s*substitute-env-vars:\s*false\s*$", docs_yml)

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
