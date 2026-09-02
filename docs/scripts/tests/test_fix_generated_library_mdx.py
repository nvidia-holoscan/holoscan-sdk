#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

SCRIPT_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from fix_generated_library_mdx import (  # noqa: E402
    collect_holoscan_api_targets,
    fix_file_content,
    reconcile_codeblock_holoscan_links,
)


class FixGeneratedLibraryMdxTestCase(TestCase):
    @staticmethod
    def _write_page(root: Path, relative_path: str, title: str, body: str = "") -> Path:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'---\ntitle: {title}\ndescription: ""\n---\n\n{body}')
        return path

    def test_unwraps_unpublished_gxf_api_links(self) -> None:
        original = """\
[Handle](../../../../nvidia/namespaces/gxf/classes/handle)
[Current](/holoscan/sdk-user-guide/api-reference/nvidia/namespaces/gxf/classes/handle)
[Legacy](/holoscan/sdk-user-guide/api-reference/cpp/nvidia/namespaces/gxf/classes/handle)
<a href="../../../../nvidia/namespaces/gxf/classes/handle">Handle</a>
[Component](../../../../nvidia/namespaces/gxf/classes/component)
[GXFComponent](/holoscan/sdk-user-guide/api-reference/holoscan/namespaces/gxf/classes/gxfcomponent)
"""

        fixed = fix_file_content(original)

        assert fixed == (
            "Handle\n"
            "Current\n"
            "Legacy\n"
            "Handle\n"
            "Component\n"
            "[GXFComponent](/holoscan/sdk-user-guide/api-reference/holoscan/namespaces/gxf/classes/gxfcomponent)\n"
        )
        assert fix_file_content(fixed) == fixed

    def test_removes_unpublished_gxf_api_links_from_codeblocks(self) -> None:
        original = (
            '<CodeBlock links={{"CudaStreamHandler": "cudastreamhandler", '
            '"Handle": "../../../../nvidia/namespaces/gxf/classes/handle"}}>\n'
            '<CodeBlock links={{"Handle": "/holoscan/sdk-user-guide/api-reference/'
            'nvidia/namespaces/gxf/classes/handle", "GXFComponent": '
            '"/holoscan/sdk-user-guide/api-reference/holoscan/namespaces/gxf/'
            'classes/gxfcomponent"}}>\n'
            '<CodeBlock links={{"Component": "/api-reference/cpp/nvidia/namespaces/'
            'gxf/classes/component"}}>\n'
            '<CodeBlock links={{"Valid": "valid", "Count": 2}}>\n'
        )

        fixed = fix_file_content(original)

        assert fixed == (
            '<CodeBlock links={{"CudaStreamHandler": "cudastreamhandler"}}>\n'
            '<CodeBlock links={{"GXFComponent": '
            '"/holoscan/sdk-user-guide/api-reference/holoscan/namespaces/gxf/classes/gxfcomponent"}}>\n'
            "<CodeBlock>\n"
            '<CodeBlock links={{"Valid": "valid", "Count": 2}}>\n'
        )
        assert fix_file_content(fixed) == fixed

    def test_preserves_codeblock_attributes_and_handles_multiline_links(self) -> None:
        original = """\
<CodeBlock title="Example" links={{
  "Handle": "../../../../nvidia/namespaces/gxf/classes/handle",
  "Operator": "operator"
}} collapsible>
"""

        fixed = fix_file_content(original)

        assert fixed == (
            '<CodeBlock title="Example" links={{"Operator": "operator"}} collapsible>\n'
        )
        assert fix_file_content(fixed) == fixed

    def test_reconciles_inherited_signature_links_with_generated_pages(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "generated"
            body = (
                '<CodeBlock links={{"Allocator": "allocator", '
                '"GXFResource": "../namespaces/gxf/classes/gxfresource", '
                '"Handle": "../../nvidia/namespaces/gxf/classes/handle"}}>\n'
                "```cpp showLineNumbers={false}\n"
                "nvidia::gxf::Handle<nvidia::gxf::Component> "
                "holoscan::gxf::GXFComponent::gxf_component()\n"
                "```\n"
                "</CodeBlock>\n"
            )
            allocator = self._write_page(
                root,
                "api-reference/cpp/holoscan/classes/Allocator.mdx",
                "holoscan::Allocator",
                body,
            )
            self._write_page(
                root,
                "api-reference/cpp/holoscan/namespaces/gxf/classes/GXFResource.mdx",
                "holoscan::gxf::GXFResource",
            )
            self._write_page(
                root,
                "api-reference/cpp/holoscan/namespaces/gxf/classes/GXFComponent.mdx",
                "holoscan::gxf::GXFComponent",
            )
            targets = collect_holoscan_api_targets(root)
            original = allocator.read_text()

            fixed = reconcile_codeblock_holoscan_links(
                fix_file_content(original), path=allocator, targets=targets
            )

            assert (
                '<CodeBlock links={{"GXFComponent": "../namespaces/gxf/classes/gxfcomponent"}}>'
                in fixed
            )
            assert '"GXFResource"' not in fixed
            assert '"Handle"' not in fixed
            assert (
                reconcile_codeblock_holoscan_links(
                    fix_file_content(fixed), path=allocator, targets=targets
                )
                == fixed
            )

    def test_corrects_qualified_symbol_link_to_a_different_page(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "generated"
            derived_endpoint = self._write_page(
                root,
                "api-reference/cpp/holoscan/namespaces/gxf/classes/Endpoint.mdx",
                "holoscan::gxf::Endpoint",
                """\
<CodeBlock links={{"Endpoint": "endpoint"}}>
```cpp showLineNumbers={false}
expected<size_t, RuntimeError> holoscan::Endpoint::read_trivial_type()
```
</CodeBlock>
""",
            )
            self._write_page(
                root,
                "api-reference/cpp/holoscan/classes/Endpoint.mdx",
                "holoscan::Endpoint",
            )
            targets = collect_holoscan_api_targets(root)

            fixed = reconcile_codeblock_holoscan_links(
                derived_endpoint.read_text(), path=derived_endpoint, targets=targets
            )

            assert '<CodeBlock links={{"Endpoint": "../../../classes/endpoint"}}>' in fixed

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
