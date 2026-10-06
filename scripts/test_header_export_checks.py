#!/usr/bin/env python3
"""
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import check_export_macro_semantics  # noqa: E402
import check_export_macros  # noqa: E402
import check_header_guards  # noqa: E402


class HeaderGuardCheckTest(unittest.TestCase):
    def test_accepts_pragma_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            header = Path(tmp_dir) / "ok.hpp"
            header.write_text("#pragma once\n\nvoid f();\n", encoding="utf-8")
            assert check_header_guards.check_header(header) == []

    def test_rejects_macro_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            header = Path(tmp_dir) / "bad.hpp"
            header.write_text(
                "#ifndef BAD_HPP\n#define BAD_HPP\nvoid f();\n#endif\n",
                encoding="utf-8",
            )
            errors = check_header_guards.check_header(header)
            assert any("macro include guard" in error for error in errors)

    def test_exempts_generated_export_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            header = Path(tmp_dir) / "holomq_export.h"
            header.write_text("#ifndef X\n#define X\n#endif\n", encoding="utf-8")
            assert check_header_guards.check_header(header) == []


class ExportMacroCheckTest(unittest.TestCase):
    def test_rejects_old_holomq_export_include(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            header = Path(tmp_dir) / "include" / "holomq" / "holomq.hpp"
            header.parent.mkdir(parents=True)
            header.write_text(
                "#pragma once\n#include <holomq/export.hpp>\n",
                encoding="utf-8",
            )
            errors = check_export_macros.check_file(header)
            assert any("holomq_export.h" in error for error in errors)

    def test_rejects_export_macro_in_detail_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            header = Path(tmp_dir) / "include" / "holomq" / "detail" / "impl.hpp"
            header.parent.mkdir(parents=True)
            header.write_text("#pragma once\nHOLOMQ_EXPORT void f();\n", encoding="utf-8")
            errors = check_export_macros.check_file(header)
            assert any("detail" in error for error in errors)

    def test_allows_holomq_public_header_to_include_detail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            header = Path(tmp_dir) / "public" / "include" / "holomq" / "publisher_builder.hpp"
            header.parent.mkdir(parents=True)
            header.write_text(
                "#pragma once\n#include <holomq/detail/transport/iox2/runtime.hpp>\n",
                encoding="utf-8",
            )
            assert check_export_macros.check_file(header) == []

    def test_allows_holomq_test_to_include_detail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "public" / "tests" / "holomq" / "test_backend.cpp"
            source.parent.mkdir(parents=True)
            source.write_text(
                "#include <holomq/detail/transport/iox2/runtime.hpp>\n",
                encoding="utf-8",
            )
            assert check_export_macros.check_file(source) == []

    def test_rejects_holomq_module_example_include_detail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = (
                Path(tmp_dir) / "public" / "modules" / "holomq" / "examples" / "bad_example.cpp"
            )
            source.parent.mkdir(parents=True)
            source.write_text(
                "#include <holomq/detail/transport/iox2/runtime.hpp>\n",
                encoding="utf-8",
            )
            errors = check_export_macros.check_file(source)
            assert any("examples must not include detail" in error for error in errors)


class ExportMacroSemanticCheckTest(unittest.TestCase):
    def test_inference_uses_its_own_export_boundary(self) -> None:
        for prefix, core_prefix in (
            ("public/modules/inference/operators/include", "public/include"),
            ("include", "include"),
        ):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as tmp_dir:
                root = Path(tmp_dir)
                header = root / prefix / "holoscan/operators/inference/operator_catalog.hpp"
                header.parent.mkdir(parents=True)
                header.write_text(
                    "HOLOSCAN_OP_INFERENCE_EXPORT Result register_catalog();\n",
                    encoding="utf-8",
                )
                core_header = root / core_prefix / "holoscan/core/scratch.hpp"
                core_header.parent.mkdir(parents=True)
                core_header.write_text(
                    "HOLOSCAN_CORE_EXPORT Result make_result();\n", encoding="utf-8"
                )
                headers = check_export_macro_semantics.public_headers(root)
                assert dict(headers) == {
                    header: "HOLOSCAN_OP_INFERENCE_EXPORT",
                    core_header: "HOLOSCAN_CORE_EXPORT",
                }
                for public_header in headers:
                    assert check_export_macro_semantics.check_header(*public_header) == []

    def test_sensor_io_uses_its_own_export_boundary(self) -> None:
        for prefix in ("public/include", "include"):
            with tempfile.TemporaryDirectory() as tmp_dir:
                root = Path(tmp_dir)
                header = root / prefix / "holoscan/sensor_io/scratch.hpp"
                header.parent.mkdir(parents=True)
                header.write_text(
                    "class HOLOSCAN_SENSOR_IO_EXPORT Scratch {\n public:\n  void reset();\n};\n",
                    encoding="utf-8",
                )
                headers = check_export_macro_semantics.public_headers(root)
                assert headers == [(header, "HOLOSCAN_SENSOR_IO_EXPORT")]
                assert check_export_macro_semantics.check_header(*headers[0]) == []

    def test_ignores_multiline_preprocessor_macro_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            header = Path(tmp_dir) / "macro.hpp"
            header.write_text(
                "#define ASSIGN_OR_RETURN(expression) \\\n"
                "  auto&& result = (expression);        \\\n"
                "  return result.value()\n",
                encoding="utf-8",
            )
            assert check_export_macro_semantics.check_header(header, "TEST_EXPORT") == []

    def test_still_rejects_unexported_function_after_macro(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            header = Path(tmp_dir) / "bad.hpp"
            header.write_text(
                "#define FUNCTION_BODY(expression) \\\n"
                "  auto&& result = (expression)\n"
                "Result make_result();\n",
                encoding="utf-8",
            )
            errors = check_export_macro_semantics.check_header(header, "TEST_EXPORT")
            assert any("lacks TEST_EXPORT" in error for error in errors)

    def test_exported_class_members_do_not_repeat_export_macro(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            header = Path(tmp_dir) / "exported.hpp"
            header.write_text(
                "class TEST_EXPORT Exported {\n"
                " public:\n"
                "  Result member();\n"
                " private:\n"
                "  void helper();\n"
                "};\n",
                encoding="utf-8",
            )
            assert check_export_macro_semantics.check_header(header, "TEST_EXPORT") == []

    def test_exported_class_scope_handles_nested_braces_and_literals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            header = Path(tmp_dir) / "exported.hpp"
            header.write_text(
                "class TEST_EXPORT Exported\n"
                "    : public Base {\n"
                " public:\n"
                '  const char* text() { return "}"; }  // } does not close the class\n'
                "  Result member();\n"
                "};\n"
                "Result free_function();\n",
                encoding="utf-8",
            )
            errors = check_export_macro_semantics.check_header(header, "TEST_EXPORT")
            assert len(errors) == 1
            assert "free_function" not in errors[0]
            assert "lacks TEST_EXPORT" in errors[0]

    def test_exported_forward_declaration_does_not_hide_later_declaration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            header = Path(tmp_dir) / "forward.hpp"
            header.write_text(
                "class TEST_EXPORT Forward;\nResult free_function();\n",
                encoding="utf-8",
            )
            errors = check_export_macro_semantics.check_header(header, "TEST_EXPORT")
            assert len(errors) == 1

    def test_unexported_class_member_is_still_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            header = Path(tmp_dir) / "unexported.hpp"
            header.write_text(
                "class Unexported {\n public:\n  Result member();\n};\n",
                encoding="utf-8",
            )
            errors = check_export_macro_semantics.check_header(header, "TEST_EXPORT")
            assert len(errors) == 1

    def test_inline_function_body_does_not_create_false_declarations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            header = Path(tmp_dir) / "inline_body.hpp"
            header.write_text(
                "class Unexported {\n"
                " public:\n"
                "  Result verified() const {\n"
                "    Local value(argument);\n"
                "    return verify(\n"
                "        package, header(), identity, generation, bytes());\n"
                "  }\n"
                "  Result declaration();\n"
                "};\n",
                encoding="utf-8",
            )
            errors = check_export_macro_semantics.check_header(header, "TEST_EXPORT")
            assert len(errors) == 1
            assert "declaration" in errors[0]

    def test_allowlist_is_exact_and_path_qualified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            first = Path(tmp_dir) / "first.hpp"
            second = Path(tmp_dir) / "second.hpp"
            declaration = "explicit Hidden(std::unique_ptr<Impl> impl) noexcept;"
            first.write_text(f"{declaration}\n", encoding="utf-8")
            second.write_text(f"{declaration}\n", encoding="utf-8")
            allowed = {check_export_macro_semantics.declaration_key(first, declaration)}

            assert check_export_macro_semantics.check_header(first, "TEST_EXPORT", allowed) == []
            assert (
                len(check_export_macro_semantics.check_header(second, "TEST_EXPORT", allowed)) == 1
            )

    def test_allowlist_parser_rejects_malformed_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            allowlist = Path(tmp_dir) / "allowlist.yml"
            allowlist.write_text("holoscan_core: everything\n", encoding="utf-8")
            try:
                check_export_macro_semantics.load_allowlist(allowlist)
            except RuntimeError:
                pass
            else:
                self.fail("malformed allowlist input was accepted")

    def test_canonical_path_is_stable_across_repository_roots(self) -> None:
        repository_path = Path("public/include/holoscan/core/operator.hpp")
        public_root_path = Path("include/holoscan/core/operator.hpp")
        assert check_export_macro_semantics.canonical_header_path(repository_path) == (
            check_export_macro_semantics.canonical_header_path(public_root_path)
        )

    def test_canonical_path_preserves_absolute_public_include_prefix(self) -> None:
        absolute_path = Path("/workspace/repo/public/include/holoscan/core/operator.hpp")
        assert check_export_macro_semantics.canonical_header_path(absolute_path) == (
            "public/include/holoscan/core/operator.hpp"
        )

    def test_canonical_path_preserves_absolute_public_modules_prefix(self) -> None:
        absolute_path = Path("/workspace/repo/public/include/holomq/publisher.hpp")
        assert check_export_macro_semantics.canonical_header_path(absolute_path) == (
            "public/include/holomq/publisher.hpp"
        )

    def test_canonical_path_normalizes_absolute_public_root_include(self) -> None:
        absolute_path = Path("/workspace/holoscan-sdk/include/holoscan/core/operator.hpp")
        assert check_export_macro_semantics.canonical_header_path(absolute_path) == (
            "public/include/holoscan/core/operator.hpp"
        )

    def test_canonical_path_normalizes_absolute_public_root_module(self) -> None:
        absolute_path = Path("/workspace/holoscan-sdk/include/holomq/publisher.hpp")
        assert check_export_macro_semantics.canonical_header_path(absolute_path) == (
            "public/include/holomq/publisher.hpp"
        )


class TestingExportPolicyTest(unittest.TestCase):
    def test_generated_component_includes_are_rejected_only_in_detail_headers(self) -> None:
        for component in ("testing", "config", "logger"):
            with self.subTest(component=component), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "include/holoscan" / component
                public_header = root / "fixture.hpp"
                detail_header = root / "detail/fixture.hpp"
                detail_header.parent.mkdir(parents=True)
                contents = f"#include <holoscan/holoscan_{component}_export.h>\n"
                public_header.write_text(contents)
                detail_header.write_text(contents)
                assert check_export_macros.check_file(public_header) == []
                errors = check_export_macros.check_file(detail_header)
                assert len(errors) == 1, errors
                assert "detail" in errors[0]

    def test_testing_headers_use_their_own_dso_macro(self) -> None:
        for prefix in ("public/include", "include"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                header = root / prefix / "holoscan/testing/fixture.hpp"
                header.parent.mkdir(parents=True)
                header.write_text("HOLOSCAN_TESTING_EXPORT void run_fixture();\n")
                assert check_export_macro_semantics.public_headers(root) == [
                    (header, "HOLOSCAN_TESTING_EXPORT")
                ]

    def test_testing_exports_require_generated_component_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            header = Path(directory) / "include/holoscan/testing/fixture.hpp"
            header.parent.mkdir(parents=True)
            header.write_text("HOLOSCAN_TESTING_EXPORT void run_fixture();\n")
            assert any(
                "holoscan_testing_export.h" in error
                for error in check_export_macros.check_file(header)
            )
            header.write_text(
                "#include <holoscan/holoscan_testing_export.h>\n"
                "HOLOSCAN_TESTING_EXPORT void run_fixture();\n"
            )
            assert check_export_macros.check_file(header) == []
            header.write_text(
                '#define HOLOSCAN_TESTING_EXPORT __attribute__((visibility("default")))\n'
            )
            assert any("hand-written" in error for error in check_export_macros.check_file(header))


if __name__ == "__main__":
    unittest.main()
