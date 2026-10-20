# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# These checks run in stdlib-only docs CI.
# ruff: noqa: PT009, PT027
"""Source URLs preserve repository identity and encode revisions and paths separately."""

import sys
from pathlib import Path
from unittest import TestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source_link_target import SourceLinkError, SourceLinkTarget  # noqa: E402


class SourceLinkTargetTest(TestCase):
    def test_github_and_gitlab_repositories_have_independent_revisions(self):
        sdk = SourceLinkTarget("https://github.com/nvidia-holoscan/holoscan-sdk", "v5.0.0")
        module = SourceLinkTarget("https://gitlab.example.com/group/subgroup/module.git", "a" * 40)
        self.assertEqual(
            sdk.url("examples", directory=True),
            "https://github.com/nvidia-holoscan/holoscan-sdk/tree/v5.0.0/examples",
        )
        self.assertEqual(
            module.url("python/_camera.cpp", fragment="L12"),
            "https://gitlab.example.com/group/subgroup/module/-/blob/"
            + "a" * 40
            + "/python/_camera.cpp#L12",
        )

    def test_prefix_describes_repository_layout_independently_of_host(self):
        browser = "https://gitlab.example.com/group/repo"
        sdk = SourceLinkTarget(browser, "a" * 40, "public")
        module = SourceLinkTarget(browser, "b" * 40)
        self.assertIn("/public/include/sdk.hpp#L2", sdk.url("include/sdk.hpp", fragment="L2"))
        self.assertTrue(sdk.url("", directory=True).endswith("/public"))
        self.assertTrue(module.url("python/source.cpp").endswith("/python/source.cpp"))
        self.assertNotIn("/public/", module.url("python/source.cpp"))
        for prefix in ("/absolute", "../outside", "a/../b", "a//b", "public/", None):
            with self.subTest(prefix=prefix), self.assertRaises(SourceLinkError):
                SourceLinkTarget(browser, "main", prefix)

    def test_encodes_branch_and_path_and_preserves_query_fragment(self):
        target = SourceLinkTarget("https://gitlab.example.com/group/module/-/", "topic/docs")
        self.assertEqual(
            target.url("code/a b#c.cpp", query="plain=1", fragment="L4"),
            "https://gitlab.example.com/group/module/-/blob/topic%2Fdocs/"
            "code/a%20b%23c.cpp?plain=1#L4",
        )

    def test_rejects_gitlab_browser_segment_on_github(self):
        for browser in (
            "https://github.com/group/module/-",
            "https://github.com/group/module/-/",
            "https://github.com/group/module.git/-",
            "https://GITHUB.COM/group/module/-",
        ):
            with self.subTest(browser=browser), self.assertRaises(SourceLinkError):
                SourceLinkTarget(browser, "main")

    def test_rejects_non_repository_urls_and_credentials(self):
        for value in (
            "http://github.com/group/module",
            "https://user:secret@github.com/group/module",
            "https://github.com/group/module?token=secret",
            "https://github.com/group/module#ref",
            "https://github.com/group/../module",
            "https://github.com/group/module/blob/main",
            "https://github.com/group/module\n",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SourceLinkTarget(value, "main")

    def test_rejects_empty_or_malformed_ref_and_escaping_path(self):
        for ref in ("", "../main", "branch?query", "branch#fragment", "two words", "a//b"):
            with self.subTest(ref=ref), self.assertRaises(SourceLinkError):
                SourceLinkTarget("https://github.com/group/module", ref)
        target = SourceLinkTarget("https://github.com/group/module", "main")
        for path in ("../outside.cpp", "/absolute.cpp", "folder/../../outside.cpp"):
            with self.subTest(path=path), self.assertRaises(SourceLinkError):
                target.url(path)
