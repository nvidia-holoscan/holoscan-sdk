#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from fix_generated_library_mdx import (  # noqa: E402
    collect_holoscan_api_targets,
    fix_file_content,
    fix_python_routes,
    normalize_python_layout,
    reconcile_codeblock_holoscan_links,
)
from fix_generated_library_mdx import (  # noqa: E402
    main as postprocess,
)


class FixGeneratedLibraryMdxTestCase(TestCase):
    def test_normalizes_python_layout_and_navigation(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            python_root = root / "api-reference/python"
            package = python_root / "holoscan-python/holoscan"
            (package / "payload").mkdir(parents=True)
            body = (
                "---\nslug: holoscan-python/holoscan\n---\n"
                "[Payload](/holoscan-python/holoscan/payload)\n"
                '<CodeBlock links={{"Schema": "/holoscan-python/holoscan/payload#Schema"}} />\n'
            )
            (package / "index.mdx").write_text(body, encoding="utf-8")
            (package / "payload/index.mdx").write_text("Payload\n", encoding="utf-8")
            navigation = python_root / "_navigation.yml"
            original_navigation = (
                "- type: page\n"
                "  slug: holoscan-python/holoscan/payload\n"
                "  pageId: holoscan-python/holoscan/payload/index.mdx\n"
            )
            navigation.write_text(original_navigation, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "needs post-processing"):  # noqa: PT027
                normalize_python_layout(root, check=True)
            assert (package / "index.mdx").read_text() == body
            assert navigation.read_text() == original_navigation

            normalize_python_layout(root)
            assert not package.parent.exists()
            assert (python_root / "holoscan/index.mdx").read_text() == body
            assert (python_root / "holoscan/payload/index.mdx").read_text() == "Payload\n"
            assert navigation.read_text() == fix_python_routes(original_navigation).replace(
                "pageId: holoscan-python/", "pageId: "
            )
            normalize_python_layout(root)
            normalize_python_layout(root, check=True)

    def test_python_routes_in_reused_tree_are_checked_and_rewritten(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            page = root / "api-reference/python/holoscan/errors.mdx"
            page.parent.mkdir(parents=True)
            body = (
                "---\nslug: holoscan-python/holoscan/errors\n---\n"
                "[Root](/holoscan-python/holoscan)\n"
                "[Error](/holoscan-python/holoscan/errors#Error)\n"
                "[Prefixed](/holoscan/sdk-user-guide/holoscan-python/holoscan/errors)\n"
                '<CodeBlock links={{"Error": "/holoscan-python/holoscan/errors#Error"}} />\n'
                "[Local](#Error)\n"
                "[Other](https://example.com/holoscan-python/holoscan/errors)\n"
            )
            page.write_text(body)
            cpp = self._write_page(root, "api-reference/cpp/holoscan/Graph.mdx", "Graph")
            cpp_before = cpp.read_bytes()
            command = ["fix_generated_library_mdx.py", "--root", str(root)]
            with patch.object(sys, "argv", [*command, "--check"]):
                assert postprocess() == 1
            assert page.read_text() == body
            with patch.object(sys, "argv", command):
                assert postprocess() == 0
            expected = (
                body.replace("slug: holoscan-python/", "slug: python/api-reference/")
                .replace("](/holoscan-python/", "](/holoscan/sdk-user-guide/python/api-reference/")
                .replace(
                    '": "/holoscan-python/', '": "/holoscan/sdk-user-guide/python/api-reference/'
                )
                .replace(
                    "](/holoscan/sdk-user-guide/holoscan-python/",
                    "](/holoscan/sdk-user-guide/python/api-reference/",
                )
            )
            assert page.read_text() == expected
            with patch.object(sys, "argv", [*command, "--check"]):
                assert postprocess() == 0
            assert cpp.read_bytes() == cpp_before

    def test_moves_previous_python_routes_in_reused_output(self) -> None:
        text = (
            "---\nslug: py-api-reference/holoscan/errors\n---\n"
            "[Error](/holoscan/sdk-user-guide/py-api-reference/holoscan/errors#Error)\n"
            '<CodeBlock links={{"Error": "/holoscan/sdk-user-guide/'
            'py-api-reference/holoscan/errors#Error"}} />\n'
        )
        expected = text.replace("py-api-reference/", "python/api-reference/")
        assert fix_python_routes(text) == expected
        assert (
            fix_python_routes(text.replace("py-api-reference/", "python-api-reference/"))
            == expected
        )
        assert fix_python_routes(expected) == expected

    def test_refuses_to_overwrite_python_output(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            python_root = root / "api-reference/python"
            package = python_root / "holoscan-python/holoscan"
            package.mkdir(parents=True)
            existing = python_root / "holoscan/index.mdx"
            existing.parent.mkdir()
            existing.write_text("Existing output\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):  # noqa: PT027
                normalize_python_layout(root)
            assert existing.read_text() == "Existing output\n"
            assert package.is_dir()

    def test_leaves_cpp_only_output_unchanged(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            page = self._write_page(root, "api-reference/cpp/holoscan/Graph.mdx", "holoscan::Graph")
            original = page.read_text()
            normalize_python_layout(root)
            normalize_python_layout(root, check=True)
            assert page.read_text() == original

    @staticmethod
    def _write_page(root: Path, relative_path: str, title: str, body: str = "") -> Path:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'---\ntitle: {title}\ndescription: ""\n---\n\n{body}')
        return path

    def test_reconciles_allocator_inherited_signature(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "generated"
            self._write_page(
                root,
                "api-reference/cpp/holoscan/classes/Allocator.mdx",
                "holoscan::Allocator",
            )
            self._write_page(
                root,
                "api-reference/cpp/holoscan/enums/MemoryStorageType.mdx",
                "holoscan::MemoryStorageType",
            )
            body = (
                '<CodeBlock links={{"CudaMemoryResourceAllocator": '
                '"cudamemoryresourceallocator", "CudaStreamAllocator": '
                '"cudastreamallocator"}}>\n'
                "```cpp showLineNumbers={false}\n"
                "virtual uint64_t holoscan::Allocator::block_size(\n"
                "    MemoryStorageType storage_type\n"
                ") const\n"
                "```\n"
                "</CodeBlock>\n"
            )
            derived_allocator = self._write_page(
                root,
                "api-reference/cpp/holoscan/classes/CudaMemoryResourceAllocator.mdx",
                "holoscan::CudaMemoryResourceAllocator",
                body,
            )
            self._write_page(
                root,
                "api-reference/cpp/holoscan/classes/CudaStreamAllocator.mdx",
                "holoscan::CudaStreamAllocator",
            )
            targets = collect_holoscan_api_targets(root)

            fixed = reconcile_codeblock_holoscan_links(
                derived_allocator.read_text(), path=derived_allocator, targets=targets
            )

            assert (
                '<CodeBlock links={{"Allocator": "allocator", '
                '"MemoryStorageType": "../enums/memorystoragetype"}}>' in fixed
            )
            assert '"CudaMemoryResourceAllocator"' not in fixed
            assert '"CudaStreamAllocator"' not in fixed
            assert (
                reconcile_codeblock_holoscan_links(fixed, path=derived_allocator, targets=targets)
                == fixed
            )

    def test_omits_keys_that_would_link_substrings_of_longer_identifiers(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "generated"
            for relative_path, title in (
                ("api-reference/cpp/holoscan/classes/Arg.mdx", "holoscan::Arg"),
                ("api-reference/cpp/holoscan/classes/ArgList.mdx", "holoscan::ArgList"),
                ("api-reference/cpp/holoscan/classes/Resource.mdx", "holoscan::Resource"),
                (
                    "api-reference/cpp/holoscan/enums/ResourceType.mdx",
                    "holoscan::ResourceType",
                ),
            ):
                self._write_page(root, relative_path, title)
            body = (
                '<CodeBlock links={{"Arg": "arg", "Resource": "resource"}}>\n'
                "```cpp showLineNumbers={false}\n"
                "template <typename ArgT, typename... ArgsT>\n"
                "void holoscan::Example::configure(\n"
                "    const holoscan::Arg &arg,\n"
                "    const holoscan::ArgList &args,\n"
                "    ResourceType resource_type,\n"
                "    const holoscan::Resource &resource\n"
                ")\n"
                "```\n"
                "</CodeBlock>\n"
            )
            page = self._write_page(
                root,
                "api-reference/cpp/holoscan/classes/Example.mdx",
                "holoscan::Example",
                body,
            )
            targets = collect_holoscan_api_targets(root)

            fixed = reconcile_codeblock_holoscan_links(page.read_text(), path=page, targets=targets)

            assert '"Example": "example"' in fixed
            assert '"ArgList": "arglist"' in fixed
            assert '"ResourceType": "../enums/resourcetype"' in fixed
            assert '"Arg":' not in fixed
            assert '"Resource":' not in fixed
            assert reconcile_codeblock_holoscan_links(fixed, path=page, targets=targets) == fixed

    def test_reconciles_page_local_inner_type_anchors(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "generated"
            self._write_page(
                root,
                "api-reference/cpp/holoscan/classes/CudaMemoryResourceAllocator.mdx",
                "holoscan::CudaMemoryResourceAllocator",
            )
            self._write_page(
                root,
                "api-reference/cpp/holoscan/structs/Error.mdx",
                "holoscan::Error",
            )
            body = (
                "## Methods\n\n"
                "### stats \\[#stats]\n\n"
                '<CodeBlock links={{"Stats": "#stats", "Error": "../structs/error"}}>\n'
                "```cpp showLineNumbers={false}\n"
                "expected<Stats, Error> holoscan::CudaMemoryResourceAllocator::stats() const\n"
                "```\n"
                "</CodeBlock>\n\n"
                "### config \\[#config]\n\n"
                '<CodeBlock links={{"Config": "#config"}}>\n'
                "```cpp showLineNumbers={false}\n"
                "const Config & holoscan::CudaMemoryResourceAllocator::config() const noexcept\n"
                "```\n"
                "</CodeBlock>\n\n"
                "## Inner classes\n\n"
                "### Config\n\n"
                "### Stats\n"
            )
            page = self._write_page(
                root,
                "api-reference/cpp/holoscan/classes/Example.mdx",
                "holoscan::Example",
                body,
            )
            targets = collect_holoscan_api_targets(root)

            fixed = reconcile_codeblock_holoscan_links(page.read_text(), path=page, targets=targets)

            assert '"Stats": "#stats-1"' in fixed
            assert '"Config": "#config-1"' in fixed
            assert '"Error": "../structs/error"' in fixed
            assert reconcile_codeblock_holoscan_links(fixed, path=page, targets=targets) == fixed

    def test_reconciles_page_local_types_section(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "generated"
            body = (
                "## Methods\n\n"
                "### status \\[#status]\n\n"
                '<CodeBlock links={{"AppDriver": "appdriver"}}>\n'
                "```cpp showLineNumbers={false}\n"
                "AppStatus holoscan::AppDriver::status()\n"
                "```\n"
                "</CodeBlock>\n\n"
                "## Types\n\n"
                "### AppStatus\n"
            )
            page = self._write_page(
                root,
                "api-reference/cpp/holoscan/classes/AppDriver.mdx",
                "holoscan::AppDriver",
                body,
            )
            targets = collect_holoscan_api_targets(root)

            fixed = reconcile_codeblock_holoscan_links(page.read_text(), path=page, targets=targets)

            assert (
                '<CodeBlock links={{"AppDriver": "appdriver", "AppStatus": "#appstatus"}}>' in fixed
            )
            assert reconcile_codeblock_holoscan_links(fixed, path=page, targets=targets) == fixed

    def test_preserves_existing_link_for_ambiguous_short_token(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "generated"
            self._write_page(
                root,
                "api-reference/cpp/holoscan/namespaces/alpha/classes/Shared.mdx",
                "holoscan::alpha::Shared",
            )
            self._write_page(
                root,
                "api-reference/cpp/holoscan/namespaces/beta/classes/Shared.mdx",
                "holoscan::beta::Shared",
            )
            body = (
                '<CodeBlock links={{"Shared": "existing-shared-target"}}>'
                "\n```cpp showLineNumbers={false}\n"
                "void compare(holoscan::alpha::Shared lhs, "
                "holoscan::beta::Shared rhs)\n"
                "```\n"
                "</CodeBlock>\n"
            )
            page = self._write_page(
                root,
                "api-reference/cpp/holoscan/functions/compare.mdx",
                "holoscan::compare",
                body,
            )
            targets = collect_holoscan_api_targets(root)

            fixed = reconcile_codeblock_holoscan_links(page.read_text(), path=page, targets=targets)

            assert '<CodeBlock links={{"Shared": "existing-shared-target"}}>' in fixed
            assert reconcile_codeblock_holoscan_links(fixed, path=page, targets=targets) == fixed

    def test_does_not_link_unqualified_class_struct_collision(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "generated"
            self._write_page(
                root,
                "api-reference/cpp/holoscan/namespaces/alpha/classes/Shared.mdx",
                "holoscan::alpha::Shared",
            )
            self._write_page(
                root,
                "api-reference/cpp/holoscan/namespaces/beta/structs/Shared.mdx",
                "holoscan::beta::Shared",
            )
            body = (
                "<CodeBlock>\n"
                "```cpp showLineNumbers={false}\n"
                "void consume(Shared value)\n"
                "```\n"
                "</CodeBlock>\n"
            )
            page = self._write_page(
                root,
                "api-reference/cpp/holoscan/functions/consume.mdx",
                "holoscan::consume",
                body,
            )
            targets = collect_holoscan_api_targets(root)

            fixed = reconcile_codeblock_holoscan_links(page.read_text(), path=page, targets=targets)

            assert fixed == page.read_text()
            assert "links=" not in fixed

    def test_does_not_link_qualified_or_parameter_name_collisions(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "generated"
            for relative_path, title in (
                ("api-reference/cpp/holoscan/typedefs/expected.mdx", "holoscan::expected"),
                ("api-reference/cpp/holoscan/typedefs/type_info.mdx", "holoscan::type_info"),
                (
                    "api-reference/cpp/holoscan/namespaces/profiler/typedefs/message.mdx",
                    "holoscan::profiler::message",
                ),
                ("api-reference/cpp/holoscan/structs/codec.mdx", "holoscan::codec"),
            ):
                self._write_page(root, relative_path, title)
            body = (
                "<CodeBlock>\n"
                "```cpp showLineNumbers={false}\n"
                "expected<void, int> holoscan::Example::submit(\n"
                "    const std::type_info &service_type,\n"
                "    const char *message,\n"
                "    std::pair<int, int> &codec\n"
                ")\n"
                "```\n"
                "</CodeBlock>\n"
            )
            page = self._write_page(
                root,
                "api-reference/cpp/holoscan/classes/Example.mdx",
                "holoscan::Example",
                body,
            )
            targets = collect_holoscan_api_targets(root)

            fixed = reconcile_codeblock_holoscan_links(page.read_text(), path=page, targets=targets)

            assert '<CodeBlock links={{"Example": "example", "expected": ' in fixed
            assert '"type_info"' not in fixed
            assert '"message"' not in fixed
            assert '"codec"' not in fixed
            assert reconcile_codeblock_holoscan_links(fixed, path=page, targets=targets) == fixed

    def test_escapes_quoted_cpp_placeholder(self) -> None:
        original = (
            'The message is "<code name>: &lt;message&gt;" and '
            "[expected&lt;T, Error&gt;](../typedefs/expected) is unchanged.\n"
        )

        fixed = fix_file_content(original)

        assert fixed == (
            'The message is "&lt;code name&gt;: &lt;message&gt;" and '
            "[expected&lt;T, Error&gt;](../typedefs/expected) is unchanged.\n"
        )
        assert fix_file_content(fixed) == fixed


if __name__ == "__main__":
    main()
