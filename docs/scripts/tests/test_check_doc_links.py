#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

SCRIPT_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from build_holoscan_docs import _should_check_doc_links  # noqa: E402
from check_doc_links import (  # noqa: E402
    _generated_pages,
    _link_targets,
    _load_nav,
    _load_redirects,
    _strip_code,
    _walk_nav,
    check,
    check_local_file_links,
)


class CheckDocLinksTestCase(TestCase):
    def test_corrects_filename_derived_4x_redirect_sources(self) -> None:
        site = "/holoscan/sdk-user-guide"
        # Fern generated these 4.x routes from navigation titles and explicit
        # slugs. They therefore do not necessarily match their MDX filenames.
        expected = {
            f"{site}/{source}": f"{site}/{destination}"
            for source, destination in (
                ("faq/faq", "resources/faq"),
                ("holo-hub/holo-hub-overview", "resources/holohub"),
                ("operators/operators-and-extensions", "resources/faq"),
                (
                    "operators/video-i-o-vendor-implementation-guide",
                    "resources/holohub",
                ),
                ("performance/gxf-job-statistics", "introduction/migrating-from-4x"),
                ("setup/additional-setup/enable-g-sync", "setup/additional-setup/gsync"),
                (
                    "setup/additional-setup/enabling-gpu-direct-rdma",
                    "setup/additional-setup/gpudirect-rdma",
                ),
                (
                    "setup/additional-setup/optimizing-jet-pack",
                    "setup/additional-setup/jetpack",
                ),
                (
                    "setup/additional-setup/use-i-gpu-with-d-gpu",
                    "setup/additional-setup/igpu-dgpu",
                ),
                ("using-the-sdk/create-a-condition", "building-operators/temporal-contracts"),
                (
                    "using-the-sdk/create-a-distributed-application",
                    "advanced/interprocess-channels",
                ),
                ("using-the-sdk/create-an-application", "getting-started/composing-a-graph"),
                ("using-the-sdk/create-an-operator", "getting-started/first-operator"),
                (
                    "using-the-sdk/create-an-operator-via-decorator",
                    "getting-started/first-operator",
                ),
                ("using-the-sdk/cuda-stream-handling", "data-and-payloads/tensors-in-depth"),
                ("using-the-sdk/data-logging", "debugging-and-testing/logging"),
                ("using-the-sdk/debugging", "debugging-and-testing/understanding-failures"),
                (
                    "using-the-sdk/dynamic-flow-control",
                    "building-operators/temporal-contracts",
                ),
                ("using-the-sdk/gpu-resident-execution", "introduction/api-stability"),
                ("using-the-sdk/logging", "debugging-and-testing/logging"),
                ("using-the-sdk/python-operator-bindings", "introduction/api-stability"),
            )
        }
        redirects = dict(_load_redirects(SCRIPT_DIR.parent / "fern" / "docs.yml"))

        for source, destination in expected.items():
            assert redirects.get(source) == destination

    def test_omits_unpublished_4x_fern_redirect_sources(self) -> None:
        site = "/holoscan/sdk-user-guide"
        unpublished = {
            f"{site}/introduction/holoscan-sdk-overview",
            f"{site}/performance/end-to-end-benchmarks",
            f"{site}/setup/third-party-hardware-setup/aja-setup",
            f"{site}/setup/third-party-hardware-setup/emergent-setup",
        }
        redirects = dict(_load_redirects(SCRIPT_DIR.parent / "fern" / "docs.yml"))

        assert unpublished.isdisjoint(redirects)

    def test_finds_multiline_markdown_link(self) -> None:
        lines = [
            "See [Enable Exclusive Display in",
            "Holoviz](operators/visualization#enable-exclusive-display-in-holoviz).",
        ]

        assert _link_targets(lines) == [
            (1, "operators/visualization#enable-exclusive-display-in-holoviz")
        ]

    def test_finds_reference_and_html_links(self) -> None:
        lines = [
            "[guide]: <docs/guide.md>",
            '<Card href="docs/card.md">',
            '<img src="images/example.png">',
            "![diagram](images/diagram.png)",
        ]

        assert _link_targets(lines) == [
            (1, "docs/guide.md"),
            (2, "docs/card.md"),
            (3, "images/example.png"),
            (4, "images/diagram.png"),
        ]

    def test_preserves_long_fence_until_matching_length_closes_it(self) -> None:
        text = "\n".join(
            [
                "before",
                "````markdown",
                "```python",
                "[ignored](missing)",
                "```",
                "```` not a closing fence",
                "[also ignored](missing)",
                "````",
                "after",
            ]
        )

        assert _strip_code(text) == ["before", "", "", "", "", "", "", "", "after"]

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

    def _write_project_with_archive(self, docs_root: Path, version: str, target: str) -> None:
        fern_dir = docs_root / "fern"
        archive_dir = fern_dir / version
        archive_dir.mkdir(parents=True)
        (docs_root / "current.mdx").write_text("# Current\n", encoding="utf-8")
        (fern_dir / "docs.yml").write_text(
            "title: Test\nversions:\n  - display-name: Current\n    path: ./index.yml\n",
            encoding="utf-8",
        )
        (fern_dir / "index.yml").write_text(
            "navigation:\n  - page: Current\n    path: ../current.mdx\n",
            encoding="utf-8",
        )
        (archive_dir / "index.yml").write_text(
            "navigation:\n"
            "  - page: Archived\n"
            "    path: ./archived.mdx\n"
            "  - page: Target\n"
            "    path: ./target.mdx\n",
            encoding="utf-8",
        )
        (archive_dir / "archived.mdx").write_text(
            f"[Historical link]({target})\n", encoding="utf-8"
        )
        (archive_dir / "target.mdx").write_text(
            '<Anchor id="historical-anchor" />\n', encoding="utf-8"
        )

    def test_validates_legacy_archives_against_their_own_navigation(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            docs_root = Path(temporary_directory)
            for version in ("v4.3", "v4.4"):
                self._write_project_with_archive(docs_root, version, "target.mdx#historical-anchor")

            assert check(docs_root, docs_root / "fern") == []

    def test_reports_missing_asset_in_legacy_archive(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            docs_root = Path(temporary_directory)
            self._write_project_with_archive(docs_root, "v4.3", "target.mdx#missing")
            archive_page = docs_root / "fern" / "v4.3" / "archived.mdx"
            archive_page.write_text(
                f'{archive_page.read_text(encoding="utf-8")}<img src="missing.png">\n',
                encoding="utf-8",
            )

            findings = check(docs_root, docs_root / "fern")

            assert any("missing asset: missing.png" in finding.message for finding in findings)

    def test_rejects_relative_site_link_in_legacy_archive(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            docs_root = Path(temporary_directory)
            self._write_project_with_archive(docs_root, "v4.4", "target")

            findings = check(docs_root, docs_root / "fern")

            assert any(
                "relative link is published unchanged: target" in finding.message
                for finding in findings
            )

    def test_reports_skipped_legacy_checks(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            docs_root = Path(temporary_directory)
            self._write_project_with_archive(docs_root, "v4.4", "target.mdx#historical-anchor")
            output = StringIO()

            with redirect_stdout(output):
                assert check(docs_root, docs_root / "fern") == []

            assert (
                "Legacy compatibility: page-file, anchor, and Sphinx migration checks skipped."
            ) in output.getvalue()

    def test_requires_versioned_site_link_in_legacy_archive(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            docs_root = Path(temporary_directory)
            self._write_project_with_archive(docs_root, "v4.4", "/holoscan/sdk-user-guide/target")

            findings = check(docs_root, docs_root / "fern")

            assert any("no such page" in finding.message for finding in findings)

    def test_accepts_versioned_site_link_in_legacy_archive(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            docs_root = Path(temporary_directory)
            self._write_project_with_archive(
                docs_root, "v4.4", "/holoscan/sdk-user-guide/4-4/target"
            )

            assert check(docs_root, docs_root / "fern") == []

    def test_accepts_generated_library_folder_route(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            library = Path(temporary_directory) / "generated"
            page = library / "namespaces" / "viz" / "typedefs" / "InstanceHandle.mdx"
            page.parent.mkdir(parents=True)
            page.write_text("# InstanceHandle\n", encoding="utf-8")

            generated, unchecked, paths = _generated_pages([("/api-reference/holoscan", library)])

            assert not unchecked
            assert paths[page.resolve()].endswith("/instancehandle")
            assert "/api-reference/holoscan/namespaces/viz" in generated

    def test_requires_generated_navigation_folder_only_when_requested(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "holoscan"
            folders = [("/api-reference/holoscan", missing)]
            assert _generated_pages(folders)[1] == ["/api-reference/holoscan"]
            for create_directory in (False, True):
                if create_directory:
                    missing.mkdir()
                with self.assertRaisesRegex(ValueError, "has no MDX pages"):  # noqa: PT027
                    _generated_pages(folders, require_generated=True)

    def test_accepts_python_library_package_layout(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "python"
            package = output / "holoscan-python" / "holoscan"
            package.mkdir(parents=True)
            (package / "index.mdx").write_text("Python API\n", encoding="utf-8")
            # A populated output root does not establish that navigation's
            # package directory exists; this is the preview failure we missed.
            with self.assertRaisesRegex(ValueError, "has no MDX pages"):  # noqa: PT027
                _generated_pages(
                    [("/api-reference/holoscan", output / "holoscan")],
                    require_generated=True,
                )
            generated, unchecked, _ = _generated_pages(
                [("/api-reference/holoscan", package)], require_generated=True
            )
            assert not unchecked
            assert "/api-reference/holoscan/index" in generated

    def test_ignores_parameterized_redirect_destination(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            docs_root = Path(temporary_directory)
            self._write_project_with_archive(docs_root, "v4.4", "target.mdx")
            docs_yml = docs_root / "fern" / "docs.yml"
            docs_yml.write_text(
                f"{docs_yml.read_text(encoding='utf-8')}"
                "redirects:\n"
                "  - source: /legacy/:slug*\n"
                "    destination: /holoscan/sdk-user-guide/unknown/:slug*\n",
                encoding="utf-8",
            )

            assert check(docs_root, docs_root / "fern") == []

    def test_accepts_link_to_redirect_source(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            docs_root = Path(temporary_directory)
            self._write_project_with_archive(
                docs_root,
                "v4.4",
                "/holoscan/sdk-user-guide/4-4/target",
            )
            (docs_root / "current.mdx").write_text(
                "[Legacy API](/holoscan/sdk-user-guide/legacy-api)\n",
                encoding="utf-8",
            )
            docs_yml = docs_root / "fern" / "docs.yml"
            docs_yml.write_text(
                f"{docs_yml.read_text(encoding='utf-8')}"
                "redirects:\n"
                "  - source: /holoscan/sdk-user-guide/legacy-api\n"
                "    destination: /holoscan/sdk-user-guide/current\n",
                encoding="utf-8",
            )

            assert check(docs_root, docs_root / "fern") == []

    def test_rejects_page_file_link_in_v4_5_archive(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            docs_root = Path(temporary_directory)
            self._write_project_with_archive(docs_root, "v4.5", "target.mdx#historical-anchor")

            findings = check(docs_root, docs_root / "fern")

            assert any("page-file link" in finding.message for finding in findings)

    def test_reports_missing_local_target_outside_docs(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            public_root = Path(temporary_directory)
            docs_root = public_root / "docs"
            docs_root.mkdir()
            (docs_root / "faq.mdx").write_text("# FAQ\n", encoding="utf-8")
            (public_root / "README.md").write_text("[FAQ](docs/faq.md)\n", encoding="utf-8")

            findings = check_local_file_links(public_root)

            assert len(findings) == 1
            assert findings[0].path == Path("README.md")
            assert findings[0].line == 1
            assert findings[0].message == "missing local target: docs/faq.md"

    def test_accepts_local_external_and_site_targets(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            public_root = Path(temporary_directory)
            docs_root = public_root / "docs"
            docs_root.mkdir()
            (docs_root / "guide.md").write_text("# Guide\n", encoding="utf-8")
            (public_root / "README.md").write_text(
                "[relative](docs/guide.md#section)\n"
                "[root](/docs/guide.md?view=1)\n"
                "[site](/holoscan/sdk-user-guide/resources/faq)\n"
                "[external](https://example.com/missing)\n"
                "[same](#section)\n"
                "```markdown\n[example](missing.md)\n```\n",
                encoding="utf-8",
            )

            assert check_local_file_links(public_root) == []

    def test_reports_nested_and_escaping_local_targets(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            public_root = workspace / "public"
            source = public_root / "examples" / "nested" / "README.md"
            source.parent.mkdir(parents=True)
            (workspace / "internal.md").write_text("internal\n", encoding="utf-8")
            source.write_text(
                "[moved](../../docs/PUBLIC_API.md)\n[internal](../../../internal.md)\n",
                encoding="utf-8",
            )

            findings = check_local_file_links(public_root)

            assert [finding.message for finding in findings] == [
                "missing local target: ../../docs/PUBLIC_API.md",
                "local link escapes the public tree: ../../../internal.md",
            ]


class PythonReferenceRoutesTestCase(TestCase):
    site = "/holoscan/sdk-user-guide"

    def test_python_api_has_one_home_and_a_shared_reference_shortcut(self):
        fern_dir = SCRIPT_DIR.parent / "fern"
        navigation = _load_nav(fern_dir / "index.yml")
        python = next(node for node in navigation if node.get("section") == "Python")
        reference = next(node for node in navigation if node.get("section") == "API reference")
        folders = []
        _walk_nav(navigation, self.site, fern_dir, {}, folders)
        python_folder = (fern_dir / "generated/api-reference/python/holoscan").resolve()
        route = f"{self.site}/python/api-reference/holoscan"
        assert [(slug, path) for slug, path in folders if path == python_folder] == [
            (route, python_folder)
        ]
        assert any(node.get("title") == "API reference" for node in python["contents"])
        shortcut = next(node for node in reference["contents"] if node.get("link") == "Python API")
        assert shortcut["href"] == route
        assert shortcut["target"] == "_self"

    def _check_reference_links(self, body: str, *, generated: bool = True):
        with TemporaryDirectory() as temporary_directory:
            docs_root = Path(temporary_directory)
            fern_dir = docs_root / "fern"
            fern_dir.mkdir()
            (fern_dir / "docs.yml").write_text(
                "instances:\n"
                "  - url: nvidia-holoscan.docs.buildwithfern.com"
                "/holoscan/sdk-user-guide\n",
                encoding="utf-8",
            )
            (fern_dir / "index.yml").write_text(
                "navigation:\n"
                "  - page: Guide\n"
                "    path: ../guide.mdx\n"
                "    slug: guide\n"
                "  - section: API reference\n"
                "    skip-slug: true\n"
                "    contents:\n"
                "      - folder: ./generated/api-reference/cpp/holoscan\n"
                "        title: C++ API\n"
                "        slug: api-reference/holoscan\n"
                "  - section: Python\n"
                "    slug: python\n"
                "    contents:\n"
                "      - folder: ./generated/api-reference/python/holoscan\n"
                "        title: API reference\n"
                "        slug: api-reference/holoscan\n",
                encoding="utf-8",
            )
            (docs_root / "guide.mdx").write_text(body, encoding="utf-8")

            if generated:
                pages = {
                    "cpp/holoscan/Allocator.mdx": (
                        "---\n"
                        "title: holoscan::Allocator\n"
                        "slug: api-reference/holoscan/allocator\n"
                        "---\nAllocator.\n"
                    ),
                    # The explicit route, not the MDX filename, identifies this page.
                    "python/holoscan/module-page.mdx": (
                        "---\n"
                        "title: holoscan.errors\n"
                        "slug: python/api-reference/holoscan/errors\n"
                        "---\nPython errors.\n"
                    ),
                }
                for relative_path, text in pages.items():
                    page = fern_dir / "generated/api-reference" / relative_path
                    page.parent.mkdir(parents=True, exist_ok=True)
                    page.write_text(text, encoding="utf-8")

            return check(
                docs_root,
                fern_dir,
                require_generated=generated,
            )

    def test_accepts_normalized_python_api_route(self):
        findings = self._check_reference_links(
            f"[Python errors]({self.site}/python/api-reference/holoscan/errors)\n"
        )
        assert not findings, "\n".join(str(finding) for finding in findings)

    def test_rejects_python_page_under_cpp_route_namespace(self):
        wrong_route = f"{self.site}/api-reference/holoscan/errors"
        findings = self._check_reference_links(f"[Python errors]({wrong_route})\n")
        assert len(findings) == 1
        assert findings[0].message == f"no such page: {wrong_route}"

    def test_preserves_cpp_api_route(self):
        findings = self._check_reference_links(
            f"[Allocator]({self.site}/api-reference/holoscan/allocator)\n"
        )
        assert not findings, "\n".join(str(finding) for finding in findings)

    def test_source_only_allows_unchecked_python_api_route(self):
        findings = self._check_reference_links(
            f"[Python errors]({self.site}/python/api-reference/holoscan/errors)\n",
            generated=False,
        )
        assert not findings, "\n".join(str(finding) for finding in findings)

    def test_source_only_preserves_explicit_cpp_folder_route(self):
        findings = self._check_reference_links(
            f"[Allocator]({self.site}/api-reference/holoscan/allocator)\n",
            generated=False,
        )
        assert not findings, "\n".join(str(finding) for finding in findings)

    def test_url_less_group_does_not_duplicate_cpp_prefix(self):
        wrong_route = f"{self.site}/api-reference/api-reference/holoscan/allocator"
        findings = self._check_reference_links(f"[Allocator]({wrong_route})\n", generated=False)
        assert [finding.message for finding in findings] == [f"no such page: {wrong_route}"]

    def test_rejects_unknown_page_when_python_output_is_present(self):
        missing_route = f"{self.site}/python/api-reference/holoscan/missing"
        findings = self._check_reference_links(f"[Missing]({missing_route})\n")
        assert [finding.message for finding in findings] == [f"no such page: {missing_route}"]

    def test_generated_slug_preserves_archive_prefix(self):
        with TemporaryDirectory() as temporary_directory:
            library = Path(temporary_directory)
            page = library / "errors.mdx"
            page.write_text(
                "---\nslug: 'python/api-reference/holoscan/errors'\n---\nErrors.\n",
                encoding="utf-8",
            )
            prefix = f"{self.site}/4-5"
            generated, unchecked, paths = _generated_pages(
                [(f"{prefix}/python/api-reference/holoscan", library)],
                site_prefix=prefix,
                require_generated=True,
            )
            assert not unchecked
            assert paths[page.resolve()] == f"{prefix}/python/api-reference/holoscan/errors"
            assert f"{prefix}/python/api-reference/holoscan/errors" in generated
            assert f"{self.site}/python/api-reference/holoscan/errors" not in generated


if __name__ == "__main__":
    main()
