# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the shared foreign-repository environment without touching a repository."""

import os
import unittest
from unittest.mock import patch

from _git_environment import git_environment


class GitEnvironmentTest(unittest.TestCase):
    def test_removes_hook_paths_and_config_but_preserves_other_settings(self):
        overrides = {
            "GIT_DIR": "/nonexistent/hook/repository",
            "GIT_WORK_TREE": "/nonexistent/hook/worktree",
            "GIT_INDEX_FILE": "/nonexistent/hook/index",
            "GIT_COMMON_DIR": "/nonexistent/hook/common",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "user.name",
            "GIT_CONFIG_VALUE_0": "Hook user",
            "HOLOSCAN_FIXTURE_SETTING": "keep",
        }
        with patch.dict(os.environ, overrides):
            before = dict(os.environ)
            environment = git_environment()
            for name in (
                "GIT_DIR",
                "GIT_WORK_TREE",
                "GIT_INDEX_FILE",
                "GIT_COMMON_DIR",
                "GIT_CONFIG_COUNT",
            ):
                assert name not in environment
            assert environment["HOLOSCAN_FIXTURE_SETTING"] == "keep"
            assert environment.get("PATH") == before.get("PATH")
            assert dict(os.environ) == before


if __name__ == "__main__":
    unittest.main()
