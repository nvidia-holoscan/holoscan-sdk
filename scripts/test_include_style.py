#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for SDK and dependency include policy and pre-commit entry points."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import check_include_style as checker  # noqa: E402


@pytest.mark.parametrize("prefix", ["holoscan", "holomq", "fmt", "iox2"])
def test_enforces_established_include_roots(prefix: str) -> None:
    header = f"{prefix}/example.hpp"
    assert checker.quoted_dependency_includes(f'#include "{header}"') == [(1, header)]
    assert checker.quoted_dependency_includes(f"#include <{header}>") == []
    assert checker.quoted_dependency_includes(f'#include "{prefix}_private/example.hpp"') == []
    assert checker.quoted_dependency_includes(f'#include "../{header}"') == []


@pytest.mark.parametrize(
    "directive",
    [
        '#include "iox2/node.hpp"',
        '# include"iox2/node.hpp"',
        '\t#\tinclude\t"iox2/node.hpp" // comment',
        '# /* comment */ include /* comment */ "iox2/node.hpp"',
        '# /* multiline\ncomment */ include /* another\ncomment */ "iox2/node.hpp"',
        '#include \\\n"iox2/node.hpp"',
        '#inc\\\nlude "io\\\nx2/node.hpp"',
        '#include \\\r\n"iox2/node.hpp"',
    ],
)
def test_rejects_quoted_dependency_headers(directive: str) -> None:
    assert checker.quoted_dependency_includes(directive) == [(1, "iox2/node.hpp")]


def test_reports_physical_lines_and_checks_inactive_branches() -> None:
    text = '#define CONTINUED \\\n1\n#if 0\n#include "iox2/node.hpp"\n#endif\n'
    assert checker.quoted_dependency_includes(text) == [(4, "iox2/node.hpp")]


@pytest.mark.parametrize(
    "source",
    [
        "#include <iox2/node.hpp>",
        '#include "../detail/private.hpp"',
        '#include "iox2_extra/private.hpp"',
        '#include "other/private.hpp"',
        "#include DEPENDENCY_HEADER",
        '// #include "iox2/node.hpp"',
        '/*\n#include "iox2/node.hpp"\n*/',
        '// continued comment \\\n#include "iox2/node.hpp"',
        'const char* text = "#include \\"iox2/node.hpp\\"";',
        'const char* text = R"(\n#include "iox2/node.hpp"\n)";',
        'const char* text = u8R"tag(\n#include "iox2/node.hpp"\n)tag";',
    ],
)
def test_allows_private_headers_and_ignores_non_directives(source: str) -> None:
    assert checker.quoted_dependency_includes(source) == []


def test_literals_do_not_hide_later_directives() -> None:
    source = "auto separator = '/'; auto wide = L'\\''; auto n = 1'000;\n"
    source += 'auto raw = R"tag(/* " //)tag";\n#include "iox2/node.hpp"\n'
    assert checker.quoted_dependency_includes(source) == [(3, "iox2/node.hpp")]


@pytest.mark.parametrize(
    "relative",
    [
        "thirdparty/test.hpp",
        "third_party/test.hpp",
        "vendor/test.hpp",
        "build-debug/test.hpp",
        "install/test.hpp",
        "_deps/test.hpp",
        "include/generated/test.hpp",
        "include/test_generated.h",
        "include/test_export.h",
        "include/test.pb.h",
        "cmake/deps/patches/test.patch",
    ],
)
def test_excludes_vendor_generated_and_patch_files(tmp_path: Path, relative: str) -> None:
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#include "iox2/node.hpp"\n')
    assert checker.check_file(path, tmp_path) == []


@pytest.mark.parametrize("layout", ["internal/public", "mirror"])
def test_cli_full_scan_and_explicit_paths(tmp_path: Path, layout: str) -> None:
    root = tmp_path / layout
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / "check_include_style.py"
    shutil.copyfile(SCRIPT_DIR / script.name, script)
    source = root / "include" / "test.hpp"
    source.parent.mkdir()
    source.write_text('#include "iox2/node.hpp"\n')
    private = source.with_name("private.hpp")
    private.write_text('#include "local.hpp"\n')
    vendor = root / "thirdparty" / "vendor.hpp"
    vendor.parent.mkdir()
    vendor.write_text(source.read_text())

    def run(*paths: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(script), *map(str, paths)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )

    assert run(private).returncode == 0
    for paths in [(), (source,)]:
        result = run(*paths)
        assert result.returncode == 1
        assert f"{source}:1: use #include <iox2/node.hpp>" in result.stdout
        assert "vendor.hpp" not in result.stdout
    source.write_text("#include <iox2/node.hpp>\n")
    assert run().returncode == 0
    assert run(source.with_name("missing.hpp")).returncode == 1
