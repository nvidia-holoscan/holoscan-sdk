# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# These tests also run in the stdlib-only docs CI environment.
# ruff: noqa: PT009, PT027

"""Camera page assembly validates the contract without editing source trees."""

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_doc_links import _load_nav, _walk_nav  # noqa: E402
from modules_docs import (  # noqa: E402
    MODULES,
    ModuleDocs,
    ModuleSource,
    assembled_module_docs,
    fetch_module,
    import_module,
    markdown_to_mdx,
)
from source_link_target import SourceLinkTarget  # noqa: E402


class CameraDocsTest(TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.sdk = self.root / "sdk"
        self.docs = self.sdk / "public/docs"
        (self.docs / "fern").mkdir(parents=True)
        (self.docs / "fern/index.yml").write_text(
            "navigation:\n" + MODULES["camera"].repository_link
        )
        (self.docs / "fern/docs.yml").write_text(
            "instances:\n  - url: example.com/holoscan/sdk-user-guide\n"
        )
        self.camera = self.root / "camera"
        (self.camera / "docs/fern").mkdir(parents=True)
        (self.camera / "docs/fern/index.yml").write_text(
            "navigation:\n  - page: Capture\n    path: ../capture.mdx\n    slug: capture\n"
        )
        (self.camera / "docs/capture.mdx").write_text(
            "[Self](capture.mdx#timing)\n\n## Timing\n\n```text\n[code](example.md)\n```\n"
        )

    def test_assembles_routes_and_preserves_both_sources(self):
        original = (self.docs / "fern/index.yml").read_bytes()
        page = self.camera / "docs/capture.mdx"
        authored = page.read_bytes()
        with assembled_module_docs(
            self.sdk, self.docs, {"camera": ModuleSource(self.camera)}
        ) as assembled:
            self.assertIn('section: "Camera"', (assembled / "fern/index.yml").read_text())
            text = (assembled / "camera/capture.mdx").read_text()
            self.assertIn("/holoscan/sdk-user-guide/modules/camera/capture#timing", text)
            self.assertIn("[code](example.md)", text)
            self.assertEqual((self.docs / "fern/index.yml").read_bytes(), original)
            self.assertEqual(page.read_bytes(), authored)
        self.assertFalse(assembled.exists())

    def test_source_links_use_module_target_while_pages_keep_fern_routes(self):
        (self.camera / "python").mkdir()
        (self.camera / "python/_camera.cpp").write_text("// module source\n")
        page = self.camera / "docs/capture.mdx"
        page.write_text(
            page.read_text() + "[Source](../python/_camera.cpp?plain=1#L4)\n"
            "[Directory](../python)\n"
            "[Absolute](https://github.com/nvidia-holoscan/holoscan-camera/blob/main/README.md)\n"
        )
        original = page.read_bytes()
        for browser in (
            "https://gitlab.example.com/group/camera/-",
            MODULES["camera"].public_repository,
        ):
            target = SourceLinkTarget(browser, "a" * 40)
            with assembled_module_docs(
                self.sdk, self.docs, {"camera": ModuleSource(self.camera, source_links=target)}
            ) as assembled:
                text = (assembled / "camera/capture.mdx").read_text()
                self.assertIn(
                    target.url("python/_camera.cpp", query="plain=1", fragment="L4"), text
                )
                self.assertIn(target.url("python", directory=True), text)
                self.assertIn("/holoscan/sdk-user-guide/modules/camera/capture#timing", text)
                self.assertIn("holoscan-camera/blob/main/README.md", text)
                self.assertEqual(page.read_bytes(), original)

    def test_fetched_source_links_use_resolved_commit_instead_of_requested_branch(self):
        (self.camera / "source.cpp").write_text("// source\n")
        page = self.camera / "docs/capture.mdx"
        page.write_text(page.read_text() + "[Source](../source.cpp)\n")
        repository = "https://gitlab.example.com/group/camera.git"
        with (
            patch("modules_docs.fetch_module", return_value=self.camera),
            patch("modules_docs.subprocess.check_output", side_effect=[repository, "b" * 40]),
            assembled_module_docs(
                self.sdk,
                self.docs,
                {"camera": ModuleSource(revision="topic", repository=repository)},
            ) as assembled,
        ):
            text = (assembled / "camera/capture.mdx").read_text()
            self.assertIn("group/camera/-/blob/" + "b" * 40 + "/source.cpp", text)
            self.assertNotIn("/topic/", text)

    def test_missing_migration_fails_and_cleans_temporary_tree(self):
        (self.camera / "docs/fern/index.yml").unlink()
        with (
            self.assertRaisesRegex(ValueError, "--module-source"),
            assembled_module_docs(self.sdk, self.docs, {"camera": ModuleSource(self.camera)}),
        ):
            self.fail("assembly must fail")
        self.assertEqual(list(self.docs.parent.glob(".module-docs-*")), [])

    def test_rejects_missing_and_escaping_pages(self):
        nav = self.camera / "docs/fern/index.yml"
        (self.root / "secret.mdx").write_text("secret")
        for path in ("../missing.mdx", "../../../secret.mdx"):
            nav.write_text(f"navigation:\n  - page: Capture\n    path: {path}\n    slug: capture\n")
            with self.subTest(path=path), self.assertRaises(ValueError):
                import_module(MODULES["camera"], self.camera, self.docs)

    def test_rejects_duplicate_slugs(self):
        nav = self.camera / "docs/fern/index.yml"
        nav.write_text(
            nav.read_text() + "  - page: Again\n    path: ../capture.mdx\n    slug: capture\n"
        )
        with self.assertRaisesRegex(ValueError, "duplicate Camera slug"):
            import_module(MODULES["camera"], self.camera, self.docs)

    def test_grouped_navigation_preserves_routes_and_cross_group_links(self):
        nav = self.camera / "docs/fern/index.yml"
        (self.camera / "docs/app.mdx").write_text("[Capture](capture.mdx#timing)\n")
        nav.write_text(
            "navigation:\n  - section: Operators\n    slug: operators\n"
            "    skip-slug: true\n    contents:\n"
            "      - page: Capture\n        path: ../capture.mdx\n        slug: capture\n"
            "  - section: Applications\n    slug: applications\n"
            "    skip-slug: true\n    contents:\n"
            "      - page: App\n        path: ../app.mdx\n        slug: app\n"
        )
        import_module(MODULES["camera"], self.camera, self.docs)
        nodes = _load_nav(self.docs / "fern/index.yml")
        groups = nodes[0]["contents"]
        self.assertEqual([node["section"] for node in groups], ["Operators", "Applications"])
        pages = {}
        _walk_nav(nodes, "/holoscan/sdk-user-guide/modules", self.docs / "fern", pages, [])
        self.assertEqual(
            set(pages.values()),
            {
                "/holoscan/sdk-user-guide/modules/camera/capture",
                "/holoscan/sdk-user-guide/modules/camera/app",
            },
        )
        self.assertEqual(
            (self.docs / "camera/app.mdx").read_text(),
            "[Capture](/holoscan/sdk-user-guide/modules/camera/capture#timing)\n",
        )

    def test_groups_reject_route_changes_empty_contents_and_duplicate_page_slugs(self):
        nav = self.camera / "docs/fern/index.yml"
        group = (
            "  - section: Operators\n    slug: operators\n"
            "    skip-slug: true\n    contents:\n"
            "      - page: Capture\n        path: ../capture.mdx\n        slug: capture\n"
        )
        for contents, message in (
            (group.replace("skip-slug: true", "skip-slug: false"), "skip-slug: true"),
            (group.split("      - page:")[0], "nonempty"),
            (group + group.replace("Operators", "Applications"), "duplicate Camera slug"),
        ):
            nav.write_text("navigation:\n" + contents)
            with self.subTest(contents=contents), self.assertRaisesRegex(ValueError, message):
                import_module(MODULES["camera"], self.camera, self.docs)

    def test_missing_linked_page_is_not_silently_published(self):
        (self.camera / "docs/capture.mdx").write_text("[Missing](missing.mdx)")
        with self.assertRaisesRegex(ValueError, "missing or outside"):
            import_module(MODULES["camera"], self.camera, self.docs)

    def test_copies_local_assets(self):
        (self.camera / "docs/picture.png").write_bytes(b"image")
        (self.camera / "docs/capture.mdx").write_text("![Diagram](picture.png)")
        import_module(MODULES["camera"], self.camera, self.docs)
        self.assertEqual((self.docs / "camera/assets/docs/picture.png").read_bytes(), b"image")
        self.assertIn("assets/docs/picture.png", (self.docs / "camera/capture.mdx").read_text())

    def test_readme_conversion_preserves_examples_and_inline_code(self):
        for fence in ("```", "~~~~", "````"):
            example = f"{fence}text\n# Example\n<!-- keep -->\n<br> {{x}}\n{fence}\n"
            text = "<!-- license -->\n# Title\n\nUse {x}<br>\n`<br> {x}`\n" + example
            result = markdown_to_mdx(text)
            self.assertNotIn("# Title", result)
            self.assertNotIn("license", result)
            self.assertIn("Use &#123;x&#125;<br />", result)
            self.assertIn("`<br> {x}`", result)
            self.assertIn(example, result)

    def test_root_readme_generates_overview_and_setup_links_without_internal_content(self):
        readme = self.camera / "README.md"
        readme.write_text(
            "# Camera\n\n[Capture](docs/capture.mdx#timing)\n\n## Setup\n\n"
            "<!-- module-docs:sdk-source -->\nSDK baseline text\n"
            "<!-- /module-docs:sdk-source -->\n"
            "<!-- holoscan-module-internal:begin -->\n"
            "[Private setup](internal/missing.md)\n```bash\nprivate-command\n```\n"
            "<!-- holoscan-module-internal:end -->\n"
        )
        original = readme.read_bytes()
        capture = self.camera / "docs/capture.mdx"
        capture.write_text(capture.read_text() + "[Setup](../README.md#setup)\n")
        nav = self.camera / "docs/fern/index.yml"
        nav.write_text(
            nav.read_text() + "  - page: Overview\n    path: ../../README.md\n    slug: overview\n"
        )
        import_module(MODULES["camera"], self.camera, self.docs)
        overview = (self.docs / "camera/overview.mdx").read_text()
        self.assertIn("/holoscan/sdk-user-guide/modules/camera/capture#timing", overview)
        self.assertIn("## Setup", overview)
        self.assertIn("SDK baseline text", overview)
        self.assertNotIn("# Camera", overview)
        self.assertNotIn("Private setup", overview)
        self.assertNotIn("private-command", overview)
        self.assertNotIn("module-docs:", overview)
        self.assertIn(
            "/holoscan/sdk-user-guide/modules/camera/overview#setup",
            (self.docs / "camera/capture.mdx").read_text(),
        )
        self.assertEqual(readme.read_bytes(), original)

    def test_readme_internal_markers_must_be_balanced(self):
        begin = "<!-- holoscan-module-internal:begin -->\n"
        end = "<!-- holoscan-module-internal:end -->\n"
        for text in (begin, end, begin + begin + end, "<!-- holoscan-module-internal:typo -->"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "internal"):
                markdown_to_mdx(text)

    def test_root_readme_cannot_collide_with_authored_overview(self):
        (self.camera / "README.md").write_text("# Camera\n")
        (self.camera / "docs/overview.mdx").write_text("Authored overview\n")
        (self.camera / "docs/fern/index.yml").write_text(
            "navigation:\n  - page: Overview\n    path: ../../README.md\n    slug: overview\n"
            "  - page: Old overview\n    path: ../overview.mdx\n    slug: old-overview\n"
        )
        with self.assertRaisesRegex(ValueError, "duplicate Camera output"):
            import_module(MODULES["camera"], self.camera, self.docs)

    def test_generates_readme_routes_assets_and_source_links_without_source_changes(self):
        operator = self.camera / "operators/capture/README.md"
        application = self.camera / "applications/capture/README.md"
        operator.parent.mkdir(parents=True)
        application.parent.mkdir(parents=True)
        operator.write_text("# Capture Operator\n\n## Timing\n")
        (application.parent / "diagram.png").write_bytes(b"image")
        (application.parent / "main.cpp").write_text("int main() {}")
        (self.camera / "README.md").write_text("# Setup\n")
        application.write_text(
            "# Application\n\n[Timing](../../operators/capture/README.md#timing)\n"
            "![Diagram](diagram.png)\n[Code](main.cpp)\n[Setup](../../README.md#setup)\n"
            "`[literal](missing.md)`\n~~~text\n[example](missing.md)\n~~~\n"
        )
        nav = self.camera / "docs/fern/index.yml"
        nav.write_text(
            "navigation:\n  - page: Capture Operator\n"
            "    path: ../../operators/capture/README.md\n    slug: capture\n"
            "  - page: Capture Application\n"
            "    path: ../../applications/capture/README.md\n    slug: application\n"
        )
        before = {p: p.read_bytes() for p in self.camera.rglob("*") if p.is_file()}
        with assembled_module_docs(
            self.sdk, self.docs, {"camera": ModuleSource(self.camera)}
        ) as assembled:
            text = (assembled / "camera/applications/capture.mdx").read_text()
            self.assertNotIn("# Application", text)
            self.assertIn("/holoscan/sdk-user-guide/modules/camera/capture#timing", text)
            self.assertIn("../assets/applications/capture/diagram.png", text)
            self.assertIn("/blob/main/applications/capture/main.cpp", text)
            self.assertIn("/blob/main/README.md#setup", text)
            self.assertIn("`[literal](missing.md)`", text)
            self.assertIn("[example](missing.md)", text)
            self.assertEqual(
                (assembled / "camera/assets/applications/capture/diagram.png").read_bytes(),
                b"image",
            )
        after = {p: p.read_bytes() for p in self.camera.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_rejects_readme_symlink_outside_checkout(self):
        outside = self.root / "private.md"
        outside.write_text("private")
        path = self.camera / "operators/capture/README.md"
        path.parent.mkdir(parents=True)
        path.symlink_to(outside)
        (self.camera / "docs/fern/index.yml").write_text(
            "navigation:\n  - page: Capture\n"
            "    path: ../../operators/capture/README.md\n    slug: capture\n"
        )
        with self.assertRaisesRegex(ValueError, "inside its checkout"):
            import_module(MODULES["camera"], self.camera, self.docs)

    def test_operator_benchmark_markdown_generates_stable_routes_and_preserves_sources(self):
        operator = self.camera / "operators/capture/README.md"
        operator.parent.mkdir(parents=True)
        operator.write_text("# Capture\n\n[Results](benchmarks.md#results)\n\n## Timing\n")
        report = operator.with_name("benchmarks.md")
        report.write_text(
            "# Capture benchmarks\n\n## Results\n\n[Operator](README.md#timing)\n"
            "[Source](../../Dockerfile)\n![Diagram](diagram.png)\n"
            "Use {frames}<br>\n```text\n[literal](missing.md)\n```\n"
        )
        (self.camera / "Dockerfile").write_text("FROM scratch\n")
        (operator.parent / "diagram.png").write_bytes(b"image")
        (self.camera / "docs/fern/index.yml").write_text(
            "navigation:\n  - page: Capture\n"
            "    path: ../../operators/capture/README.md\n    slug: capture\n"
            "  - page: Capture benchmarks\n"
            "    path: ../../operators/capture/benchmarks.md\n    slug: capture-benchmarks\n"
        )
        before = {p: p.read_bytes() for p in self.camera.rglob("*") if p.is_file()}
        with assembled_module_docs(
            self.sdk, self.docs, {"camera": ModuleSource(self.camera)}
        ) as assembled:
            text = (assembled / "camera/benchmarks/capture_benchmarks.mdx").read_text()
            self.assertNotIn("# Capture benchmarks", text)
            self.assertIn("## Results", text)
            self.assertIn("/holoscan/sdk-user-guide/modules/camera/capture#timing", text)
            self.assertIn("/blob/main/Dockerfile", text)
            self.assertIn("../assets/operators/capture/diagram.png", text)
            self.assertIn("Use &#123;frames&#125;<br />", text)
            self.assertIn("[literal](missing.md)", text)
            self.assertIn(
                "/holoscan/sdk-user-guide/modules/camera/capture-benchmarks#results",
                (assembled / "camera/operators/capture.mdx").read_text(),
            )
        self.assertEqual(before, {p: p.read_bytes() for p in self.camera.rglob("*") if p.is_file()})

    def test_benchmark_markdown_cannot_collide_with_authored_report(self):
        report = self.camera / "operators/capture/benchmarks.md"
        report.parent.mkdir(parents=True)
        report.write_text("# Results\n")
        authored = self.camera / "docs/benchmarks/capture_benchmarks.mdx"
        authored.parent.mkdir(parents=True)
        authored.write_text("Results\n")
        (self.camera / "docs/fern/index.yml").write_text(
            "navigation:\n  - page: Results\n"
            "    path: ../../operators/capture/benchmarks.md\n    slug: results\n"
            "  - page: Old results\n"
            "    path: ../benchmarks/capture_benchmarks.mdx\n    slug: old-results\n"
        )
        with self.assertRaisesRegex(ValueError, "duplicate Camera output"):
            import_module(MODULES["camera"], self.camera, self.docs)

    def test_manifest_imports_nested_readmes_and_python_setup_routes(self):
        pages = {
            "operators/capture/README.md": (
                "Capture",
                "capture",
                "# Capture\n[Python](../../applications/capture/python/README.md#run)\n",
            ),
            "applications/capture/python/README.md": (
                "Python application",
                "python-application",
                "# Python app\n## Run\n[Setup](../../../python/README.md#setup)\n"
                "[Operator](../../../operators/capture/README.md)\n![Layout](layout.png)\n",
            ),
            "python/README.md": (
                "Python support",
                "python-support",
                "# Python support\n## Setup\n[App](../applications/capture/python/README.md#run)\n",
            ),
        }
        navigation = "navigation:\n"
        for source, (title, slug, text) in pages.items():
            path = self.camera / source
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            navigation += f"  - page: {title}\n    path: ../../{source}\n    slug: {slug}\n"
        (self.camera / "applications/capture/python/layout.png").write_bytes(b"image")
        (self.camera / "docs/fern/index.yml").write_text(navigation)
        before = {p: p.read_bytes() for p in self.camera.rglob("*") if p.is_file()}
        import_module(MODULES["camera"], self.camera, self.docs)
        prefix = "/holoscan/sdk-user-guide/modules/camera"
        operator = (self.docs / "camera/operators/capture.mdx").read_text()
        application = (self.docs / "camera/applications/capture/python.mdx").read_text()
        setup = (self.docs / "camera/python.mdx").read_text()
        self.assertIn(f"[Python]({prefix}/python-application#run)", operator)
        self.assertIn(f"[Setup]({prefix}/python-support#setup)", application)
        self.assertIn(f"[Operator]({prefix}/capture)", application)
        self.assertIn(f"[App]({prefix}/python-application#run)", setup)
        self.assertNotIn("github.com", operator + application + setup)
        self.assertIn("../../assets/applications/capture/python/layout.png", application)
        self.assertEqual(before, {p: p.read_bytes() for p in self.camera.rglob("*") if p.is_file()})
        routes = {}
        _walk_nav(
            _load_nav(self.docs / "fern/index.yml"),
            "/holoscan/sdk-user-guide/modules",
            self.docs / "fern",
            routes,
            [],
        )
        self.assertEqual(set(routes.values()), {f"{prefix}/{page[1]}" for page in pages.values()})

    def test_manifest_accepts_arbitrary_markdown_and_mdx_but_only_listed_pages(self):
        guide = self.camera / "guides/platform.v2/setup.md"
        guide.parent.mkdir(parents=True)
        guide.write_text("# Setup\n[Reference](reference.mdx#details)\n")
        guide.with_name("reference.mdx").write_text("## Details\n<Note>Reference</Note>\n")
        guide.with_name("unlisted.md").write_text("# Not included\n")
        (self.camera / "docs/fern/index.yml").write_text(
            "navigation:\n  - page: Setup\n    path: ../../guides/platform.v2/setup.md\n"
            "    slug: setup\n"
            "  - page: Reference\n    path: ../../guides/platform.v2/reference.mdx\n"
            "    slug: reference\n"
        )
        import_module(MODULES["camera"], self.camera, self.docs)
        output = self.docs / "camera/guides/platform.v2"
        self.assertIn("/modules/camera/reference#details", (output / "setup.mdx").read_text())
        self.assertEqual(
            (output / "reference.mdx").read_text(), "## Details\n<Note>Reference</Note>\n"
        )
        self.assertFalse((output / "unlisted.mdx").exists())

    def test_manifest_rejects_non_markdown_and_colliding_generic_outputs(self):
        nav = self.camera / "docs/fern/index.yml"
        source = self.camera / "script.py"
        source.write_text("raise RuntimeError('must not execute')\n")
        nav.write_text("navigation:\n  - page: Code\n    path: ../../script.py\n    slug: code\n")
        with self.assertRaisesRegex(ValueError, "must be Markdown or MDX"):
            import_module(MODULES["camera"], self.camera, self.docs)
        (self.camera / "python").mkdir()
        (self.camera / "python/README.md").write_text("# Setup\n")
        (self.camera / "python.mdx").write_text("Other setup\n")
        nav.write_text(
            "navigation:\n  - page: Setup\n    path: ../../python/README.md\n    slug: setup\n"
            "  - page: Other\n    path: ../../python.mdx\n    slug: other\n"
        )
        with self.assertRaisesRegex(ValueError, "duplicate Camera output"):
            import_module(MODULES["camera"], self.camera, self.docs)

    def test_fetch_reuses_dependency_and_propagates_errors(self):
        with patch("modules_docs.subprocess.run") as run:
            result = fetch_module(MODULES["camera"], self.sdk, self.root, "abc123")
            command = run.call_args.args[0]
            self.assertEqual(command[:2], ["cmake", "-DHOLOSCAN_CAMERA_GIT_TAG=abc123"])
            self.assertEqual(
                command[-2:],
                ["-P", str(self.sdk / "public/cmake/external/HoloscanCameraDependency.cmake")],
            )
            self.assertEqual(result, self.root / "holoscan-camera-src")
            run.side_effect = subprocess.CalledProcessError(1, command)
            with self.assertRaises(subprocess.CalledProcessError):
                fetch_module(MODULES["camera"], self.sdk, self.root, None)

    def test_fetch_accepts_repository_and_exact_revision(self):
        repository = "ssh://git@example.com/camera.git"
        revision = "a" * 40
        with patch("modules_docs.subprocess.run") as run:
            fetch_module(MODULES["camera"], self.sdk, self.root, revision, repository)
        command = run.call_args.args[0]
        self.assertIn(f"-DHOLOSCAN_CAMERA_GIT_TAG={revision}", command)
        self.assertIn(f"-DHOLOSCAN_CAMERA_GIT_REPOSITORY={repository}", command)

    def test_two_modules_share_conversion_but_isolate_routes_assets_and_source_links(self):
        multimedia = ModuleDocs(
            "multimedia",
            "Multimedia",
            "https://example.com/modules/multimedia",
            "cmake/external/Multimedia.cmake",
            "HOLOSCAN_MULTIMEDIA",
            "multimedia-src",
        )
        other = self.root / "multimedia"
        (other / "docs/fern").mkdir(parents=True)
        (other / "docs/fern/index.yml").write_text(
            "navigation:\n  - page: Guide\n    path: ../guide.md\n    slug: capture\n"
        )
        (other / "docs/guide.md").write_text(
            "# Guide\n\nUse {mode}<br>\n[Self](guide.md#timing)\n"
            "![Image](picture.png)\n[Source](../source.cpp)\n\n## Timing\n"
        )
        (other / "docs/picture.png").write_bytes(b"multimedia image")
        (other / "source.cpp").write_text("// source\n")
        (self.camera / "docs/picture.png").write_bytes(b"camera image")
        page = self.camera / "docs/capture.mdx"
        page.write_text(page.read_text() + "![Image](picture.png)\n")
        (self.docs / "fern/index.yml").write_text(
            "navigation:\n  - section: Modules\n    slug: modules\n    contents:\n"
            + MODULES["camera"].repository_link
        )
        before = {
            p: p.read_bytes()
            for root in (self.camera, other, self.docs)
            for p in root.rglob("*")
            if p.is_file()
        }
        with (
            patch.dict(MODULES, {"multimedia": multimedia}),
            assembled_module_docs(
                self.sdk,
                self.docs,
                {"camera": ModuleSource(self.camera), "multimedia": ModuleSource(other)},
            ) as assembled,
        ):
            text = (assembled / "multimedia/guide.mdx").read_text()
            self.assertIn("Use &#123;mode&#125;<br />", text)
            self.assertIn("/modules/multimedia/capture#timing", text)
            self.assertIn("https://example.com/modules/multimedia/blob/main/source.cpp", text)
            self.assertEqual(
                (assembled / "multimedia/assets/docs/picture.png").read_bytes(), b"multimedia image"
            )
            self.assertEqual(
                (assembled / "camera/assets/docs/picture.png").read_bytes(), b"camera image"
            )
            pages = {}
            _walk_nav(
                _load_nav(assembled / "fern/index.yml"),
                "/holoscan/sdk-user-guide",
                assembled / "fern",
                pages,
                [],
            )
            self.assertEqual(
                set(pages.values()),
                {
                    "/holoscan/sdk-user-guide/modules/camera/capture",
                    "/holoscan/sdk-user-guide/modules/multimedia/capture",
                },
            )
        self.assertFalse(assembled.exists())
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        with patch("modules_docs.subprocess.run") as run:
            result = fetch_module(multimedia, self.sdk, self.root, "a" * 40)
        self.assertEqual(result, self.root / "multimedia-src")
        self.assertIn("-DHOLOSCAN_MULTIMEDIA_GIT_TAG=" + "a" * 40, run.call_args.args[0])
        self.assertEqual(
            run.call_args.args[0][-1], str(self.sdk / "public/cmake/external/Multimedia.cmake")
        )

    def test_unknown_module_fails_before_fetch_or_assembly(self):
        with (
            patch("modules_docs.fetch_module") as fetch,
            self.assertRaisesRegex(ValueError, "unsupported documentation modules"),
            assembled_module_docs(self.sdk, self.docs, {"unknown": ModuleSource()}),
        ):
            self.fail("unknown module must fail")
        fetch.assert_not_called()
        self.assertEqual(list(self.docs.parent.glob(".module-docs-*")), [])

    def test_subdirectories_keep_routes_and_cross_page_anchors(self):
        (self.camera / "docs/operators").mkdir()
        (self.camera / "docs/applications").mkdir()
        (self.camera / "docs/fern/index.yml").write_text(
            "navigation:\n  - page: Capture Operator\n"
            "    path: ../operators/capture.mdx\n    slug: capture\n"
            "  - page: Capture Application\n"
            "    path: ../applications/capture.mdx\n    slug: application\n"
        )
        (self.camera / "docs/operators/capture.mdx").write_text("## Timing\n")
        (self.camera / "docs/applications/capture.mdx").write_text(
            "[Timing](../operators/capture.mdx#timing)\n"
        )
        import_module(MODULES["camera"], self.camera, self.docs)
        self.assertEqual(
            (self.docs / "camera/applications/capture.mdx").read_text(),
            "[Timing](/holoscan/sdk-user-guide/modules/camera/capture#timing)\n",
        )

    def test_release_navigation_without_camera_placeholder(self):
        (self.docs / "fern/index.yml").write_text(
            "navigation:\n  - section: Holoscan Modules\n    slug: modules\n"
            "    path: ../modules.mdx\n    contents:\n"
            "      - page: Inference\n        path: ../inference.mdx\n        slug: inference\n"
        )
        import_module(MODULES["camera"], self.camera, self.docs)
        nav = (self.docs / "fern/index.yml").read_text()
        self.assertIn('section: "Camera"', nav)
        self.assertIn("page: Inference", nav)
