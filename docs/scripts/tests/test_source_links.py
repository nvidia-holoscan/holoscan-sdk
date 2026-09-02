#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from urllib.parse import unquote

SCRIPT_DIR = Path(__file__).parents[1]
DOCS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from render_source_links import (  # noqa: E402
    CANONICAL_SOURCE_BROWSER_URL,
    SourceLinkError,
    audit_consistent_source_links,
    audit_source_links,
    render_source_links,
    validate_target,
    write_committed_source_links,
)


def assert_source_link_error(pattern: str, callback) -> None:
    try:
        callback()
    except SourceLinkError as exc:
        assert re.search(pattern, str(exc))  # noqa: PT017
    else:
        raise AssertionError("expected SourceLinkError")


class SourceLinkTestCase(TestCase):
    def test_committed_links_use_public_github_main(self) -> None:
        assert audit_source_links(DOCS_ROOT) > 0

    def test_blob_and_tree_routes_match_checkout_objects(self) -> None:
        pattern = re.compile(
            rf"{re.escape(CANONICAL_SOURCE_BROWSER_URL)}/(blob|tree)/main/"
            r"([^\]\s)\"<>]+)"
        )
        links = []
        public_root = DOCS_ROOT.parent
        for source in sorted((*DOCS_ROOT.rglob("*.md"), *DOCS_ROOT.rglob("*.mdx"))):
            for match in pattern.finditer(source.read_text(encoding="utf-8")):
                route, raw_path = match.groups()
                path = unquote(re.split(r"[?#]", raw_path, maxsplit=1)[0]).rstrip("/")
                links.append((source, route, path))

        assert links
        for source, route, path in links:
            target = public_root / path
            with self.subTest(source=source, route=route, path=path):
                if route == "blob":
                    assert target.is_file()
                else:
                    assert target.is_dir()

    def test_renders_every_link_with_one_release_ref(self) -> None:
        with TemporaryDirectory() as tmpdir:
            docs_root = Path(tmpdir) / "docs"
            shutil.copytree(DOCS_ROOT, docs_root)

            count = render_source_links(
                docs_root,
                source_browser_url="https://gitlab.com/nvidia/holoscan/holoscan-sdk",
                source_ref="v4.5.0-ea",
            )

            assert count > 0
            assert (
                audit_source_links(
                    docs_root,
                    expected_browser_url=("https://gitlab.com/nvidia/holoscan/holoscan-sdk/-"),
                    expected_ref="v4.5.0-ea",
                )
                == count
            )

    def test_rejects_mixed_committed_refs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            docs_root = Path(tmpdir)
            (docs_root / "page.mdx").write_text(
                f"[one]({CANONICAL_SOURCE_BROWSER_URL}/blob/main/README.md)\n"
                f"[two]({CANONICAL_SOURCE_BROWSER_URL}/tree/v4.5.0/examples)\n",
                encoding="utf-8",
            )

            assert_source_link_error("expected", lambda: audit_source_links(docs_root))

    def test_write_normalizes_github_refs_and_routes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            docs_root = workspace / "docs"
            docs_root.mkdir()
            (workspace / "examples").mkdir()
            (workspace / "README.md").write_text("readme\n", encoding="utf-8")
            page = docs_root / "page.mdx"
            page.write_text(
                f"[directory]({CANONICAL_SOURCE_BROWSER_URL}/blob/v4.5.0/examples)\n"
                f"[file]({CANONICAL_SOURCE_BROWSER_URL}/tree/v4.5.0/README.md)\n",
                encoding="utf-8",
            )

            assert write_committed_source_links(docs_root, [page]) == 2
            assert page.read_text(encoding="utf-8") == (
                f"[directory]({CANONICAL_SOURCE_BROWSER_URL}/tree/main/examples)\n"
                f"[file]({CANONICAL_SOURCE_BROWSER_URL}/blob/main/README.md)\n"
            )
            assert audit_source_links(docs_root, paths=[page]) == 2

    def test_write_normalizes_ref_when_target_is_unavailable(self) -> None:
        with TemporaryDirectory() as tmpdir:
            docs_root = Path(tmpdir)
            page = docs_root / "page.mdx"
            page.write_text(
                f"[historical]({CANONICAL_SOURCE_BROWSER_URL}/blob/v1.0.3/"
                "python/holoscan/macros.hpp)\n",
                encoding="utf-8",
            )

            assert write_committed_source_links(docs_root, [page]) == 1
            assert page.read_text(encoding="utf-8") == (
                f"[historical]({CANONICAL_SOURCE_BROWSER_URL}/blob/main/"
                "python/holoscan/macros.hpp)\n"
            )

    def test_rejects_gitlab_committed_link(self) -> None:
        with TemporaryDirectory() as tmpdir:
            docs_root = Path(tmpdir)
            (docs_root / "page.mdx").write_text(
                "[source](https://gitlab.com/nvidia/holoscan/holoscan-sdk/-/blob/main/README.md)\n",
                encoding="utf-8",
            )

            assert_source_link_error("GitLab", lambda: audit_source_links(docs_root))

    def test_checks_only_selected_files(self) -> None:
        with TemporaryDirectory() as tmpdir:
            docs_root = Path(tmpdir)
            selected = docs_root / "selected.mdx"
            selected.write_text("No source links.\n", encoding="utf-8")
            (docs_root / "unselected.mdx").write_text(
                "[source](https://gitlab.com/nvidia/holoscan/holoscan-sdk/-/blob/main/README.md)\n",
                encoding="utf-8",
            )

            assert (
                audit_source_links(
                    docs_root,
                    paths=[selected],
                    allow_empty=True,
                )
                == 0
            )
            assert_source_link_error("GitLab", lambda: audit_source_links(docs_root))

    def test_accepts_consistent_archive_ref(self) -> None:
        with TemporaryDirectory() as tmpdir:
            docs_root = Path(tmpdir)
            (docs_root / "page.mdx").write_text(
                f"[one]({CANONICAL_SOURCE_BROWSER_URL}/blob/v4.5.0.post2/README.md)\n"
                f"[two]({CANONICAL_SOURCE_BROWSER_URL}/tree/v4.5.0.post2/examples)\n",
                encoding="utf-8",
            )

            assert audit_consistent_source_links(docs_root) == 2

    def test_accepts_explicit_additional_render_browser(self) -> None:
        with TemporaryDirectory() as tmpdir:
            docs_root = Path(tmpdir)
            browser_url = "https://gitlab.example.com/nvidia/holoscan-sdk"
            (docs_root / "page.mdx").write_text(
                f"[source]({browser_url}/-/blob/v4.5.0/README.md)\n",
                encoding="utf-8",
            )

            assert (
                audit_consistent_source_links(
                    docs_root,
                    allowed_source_browser_urls=[browser_url],
                )
                == 1
            )

    def test_accepts_explicit_commit_ref(self) -> None:
        with TemporaryDirectory() as tmpdir:
            docs_root = Path(tmpdir)
            source_ref = "a" * 40
            (docs_root / "page.mdx").write_text(
                f"[source]({CANONICAL_SOURCE_BROWSER_URL}/blob/{source_ref}/README.md)\n",
                encoding="utf-8",
            )

            assert (
                audit_consistent_source_links(
                    docs_root,
                    allowed_source_refs=[source_ref],
                )
                == 1
            )

    def test_renders_and_reaudits_explicit_commit_ref(self) -> None:
        with TemporaryDirectory() as tmpdir:
            docs_root = Path(tmpdir)
            source_ref = "a" * 40
            browser_url = "https://gitlab.example.com/nvidia/holoscan-sdk"
            page = docs_root / "page.mdx"
            page.write_text(
                f"[source]({CANONICAL_SOURCE_BROWSER_URL}/blob/main/README.md)\n",
                encoding="utf-8",
            )

            assert (
                render_source_links(
                    docs_root,
                    source_browser_url=browser_url,
                    source_ref=source_ref,
                    allowed_source_browser_urls=[browser_url],
                    allowed_source_refs=[source_ref],
                )
                == 1
            )
            assert f"{browser_url}/-/blob/{source_ref}/README.md" in page.read_text(
                encoding="utf-8"
            )

    def test_rejects_mixed_archive_refs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            docs_root = Path(tmpdir)
            (docs_root / "page.mdx").write_text(
                f"[one]({CANONICAL_SOURCE_BROWSER_URL}/blob/v4.5.0/README.md)\n"
                f"[two]({CANONICAL_SOURCE_BROWSER_URL}/tree/v4.4.0/examples)\n",
                encoding="utf-8",
            )

            assert_source_link_error(
                "mixed browser/ref",
                lambda: audit_consistent_source_links(docs_root),
            )

    def test_main_requires_public_github(self) -> None:
        assert_source_link_error(
            "must use",
            lambda: validate_target(
                "https://gitlab.com/nvidia/holoscan/holoscan-sdk",
                "main",
            ),
        )

    def test_non_main_ref_must_be_release_tag(self) -> None:
        assert_source_link_error(
            "release tag",
            lambda: validate_target(CANONICAL_SOURCE_BROWSER_URL, "release/latest"),
        )

    def test_release_tag_may_use_alternate_browser(self) -> None:
        assert validate_target(
            "https://gitlab.com/nvidia/holoscan/holoscan-sdk",
            "v4.5.0-ea",
        ) == (
            "https://gitlab.com/nvidia/holoscan/holoscan-sdk/-",
            "v4.5.0-ea",
        )

    def test_release_tag_rejects_unknown_browser(self) -> None:
        assert_source_link_error(
            "must be one of",
            lambda: validate_target(
                "https://example.com/nvidia/holoscan-sdk",
                "v4.5.0",
            ),
        )


if __name__ == "__main__":
    main()
