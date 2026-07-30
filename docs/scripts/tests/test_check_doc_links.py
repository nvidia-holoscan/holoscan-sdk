#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from argparse import Namespace
from pathlib import Path
from unittest import TestCase, main

SCRIPT_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from build_holoscan_docs import _should_check_doc_links  # noqa: E402
from check_doc_links import _link_targets  # noqa: E402


class CheckDocLinksTestCase(TestCase):
    def test_finds_multiline_markdown_link(self) -> None:
        lines = [
            "See [Enable Exclusive Display in",
            "Holoviz](operators/visualization#enable-exclusive-display-in-holoviz).",
        ]

        assert _link_targets(lines) == [
            (1, "operators/visualization#enable-exclusive-display-in-holoviz")
        ]

    def test_checks_links_when_replacing_remote_preview(self) -> None:
        args = Namespace(
            delete_preview_id="source-branch",
            publish_preview=True,
            skip_link_check=False,
        )

        assert _should_check_doc_links(args)

    def test_skips_links_for_standalone_remote_preview_deletion(self) -> None:
        args = Namespace(
            delete_preview_id="source-branch",
            publish_preview=False,
            skip_link_check=False,
        )

        assert not _should_check_doc_links(args)

    def test_honors_explicit_skip_during_remote_preview_publication(self) -> None:
        args = Namespace(
            delete_preview_id=None,
            publish_preview=True,
            skip_link_check=True,
        )

        assert not _should_check_doc_links(args)


if __name__ == "__main__":
    main()
