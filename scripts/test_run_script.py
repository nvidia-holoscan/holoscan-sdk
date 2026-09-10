#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for git-derived image tags in the run script."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

PUBLIC_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = PUBLIC_ROOT / "run"
VERSION_FILE = PUBLIC_ROOT / "VERSION"


def source_run_script(run_script: Path, command: str) -> subprocess.CompletedProcess[str]:
    """Source run_script and execute one of its functions."""
    return subprocess.run(
        ["bash", "-c", f'source "$1"\n{command}', str(run_script), str(run_script)],
        check=False,
        capture_output=True,
        text=True,
    )


def copy_source_archive(destination: Path) -> Path:
    """Create the minimal source-archive layout needed to exercise run."""
    destination.mkdir()
    shutil.copy2(RUN_SCRIPT, destination / "run")
    shutil.copy2(VERSION_FILE, destination / "VERSION")
    return destination / "run"


class RunScriptGitIntegrationTestCase(unittest.TestCase):
    def assert_archive_fallback(self, run_script: Path) -> None:
        version = VERSION_FILE.read_text().strip()

        result = source_run_script(
            run_script,
            "if is_git_repo; then printf true; else printf false; fi",
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == "false"

        result = source_run_script(run_script, "get_git_sha")
        assert result.returncode != 0

        result = source_run_script(run_script, "get_build_img_tag")
        assert result.returncode == 0, result.stderr
        assert result.stdout == version

        result = source_run_script(run_script, "get_image_tag_flags holoscan-test")
        assert result.returncode == 0, result.stderr
        assert result.stdout == f"-t holoscan-test:latest -t holoscan-test:{version} "

    def test_git_checkout_uses_commit_sha(self) -> None:
        expected_sha = subprocess.run(
            ["git", "-C", str(PUBLIC_ROOT), "rev-parse", "--short=9", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        result = source_run_script(RUN_SCRIPT, "get_build_img_tag")

        assert result.returncode == 0, result.stderr
        assert result.stdout == expected_sha

    def test_image_tag_flags_deduplicate_repeated_values(self) -> None:
        result = source_run_script(
            RUN_SCRIPT,
            """
is_git_repo() { return 0; }
get_version() { printf duplicate; }
get_git_branch() { printf duplicate; }
get_git_tag() { printf duplicate; }
get_git_sha() { printf duplicate; }
get_image_tag_flags holoscan-test
""",
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout == "-t holoscan-test:latest -t holoscan-test:duplicate "

    def test_source_archive_outside_git_uses_version(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_script = copy_source_archive(Path(temporary_directory) / "archive")
            self.assert_archive_fallback(run_script)

    def test_untracked_source_archive_inside_git_uses_version(self) -> None:
        with tempfile.TemporaryDirectory(dir=PUBLIC_ROOT) as temporary_directory:
            run_script = copy_source_archive(Path(temporary_directory) / "archive")
            self.assert_archive_fallback(run_script)


if __name__ == "__main__":
    unittest.main()
