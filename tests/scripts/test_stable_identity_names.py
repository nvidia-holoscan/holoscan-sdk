# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for SDK-owned identity spelling, not runtime admission."""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/check_stable_identity_names.py"
sys.path.insert(0, str(SCRIPT.parent))

from _git_environment import git_environment  # noqa: E402


# REQ: AUTHOR-028
def test_sdk_identity_migration():
    """The representative shipped schema must use product-scoped ownership."""
    root = Path(__file__).resolve().parents[2]
    source = (root / "modules/vision/include/holoscan/vision/resize_op.hpp").read_text()
    assert 'param_schema("holoscan.vision.ResizeOpParams", 2)' in source


def checker():
    spec = importlib.util.spec_from_file_location("stable_identity_checker", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "source",
    [
        'param_schema("com.nvidia.holoscan.Params", 1)',
        'param_schema(\n "nvidia.holoscan.Params", 1)',
        'constexpr auto kId = "com.nvidia.other.Params"; param_schema(kId, 1);',
        'static constexpr auto name = "nvidia.other.Pod";',
        '.validator<&valid>("com.nvidia.other.valid")',
        'parameter_projected_byte_span<&size>("nvidia.other.size.v1", field)',
        'REGISTER_OPERATOR(Op, "com.nvidia.other.Op")',
        'register("nvidia.other.Operator")',
        'REGISTER_OPERATOR(Op, "nvidia.Foo")',
        'REGISTER_OPERATOR(Op, "com.nvidia.Foo")',
    ],
)
def test_rejects_alternate_nvidia_roots(source):
    assert checker().identity_literals(source)


@pytest.mark.parametrize(
    "source",
    [
        'param_schema("holoscan.vision.ResizeOpParams", 2)',
        'param_schema("holoscan." "test.Params", 1)',
        'param_schema(R"(holoscan.test.Params)", 1)',
        'param_schema("holoscan.hsb.Params", 1)',
        'name = "holoscan.test.Pod"; version = 2;',
        '.validator<&valid>("holoscan.vision.validator.finite-f64.v1")',
        'parameter_projected_byte_span<&size>("holoscan.examples.Op.size.v1", field)',
        'register("holoscan.Tensor")',
        'hash.add("holoscan.schema-family/v2")',
        "namespace acme.telemetry; root_type Frame;",
        '// param_schema("nvidia.example.Params", 1)',
        '/* name = "com.nvidia.example.Pod"; */',
        'url = "https://nvidia.com"',
        'hostname = "nvidia.com"',
    ],
)
def test_preserves_valid_names_and_non_identity_text(source):
    assert checker().identity_literals(source) == []


def test_exception_is_exact_and_requires_a_reason(tmp_path):
    root = tmp_path
    file = root / "tests" / "fixture.cpp"
    file.parent.mkdir()
    file.write_text('auto policy = "nvidia.policy.old"; auto bad = "nvidia.example.New";')
    exceptions = [
        {"path": "tests/fixture.cpp", "literal": "nvidia.policy.old", "reason": "Policy ID"}
    ]
    errors = checker().check_paths(root, [file], exceptions)
    assert len(errors) == 1
    assert "nvidia.example.New" in errors[0]
    exceptions[0]["reason"] = ""
    assert any("reason" in error for error in checker().check_paths(root, [file], exceptions))


def test_stale_exception_fails(tmp_path):
    file = tmp_path / "fixture.cpp"
    file.write_text('auto name = "holoscan.test.Current";')
    errors = checker().check_paths(
        tmp_path,
        [file],
        [{"path": "fixture.cpp", "literal": "nvidia.old.Name", "reason": "Old fixture"}],
    )
    assert any("stale" in error for error in errors)


def test_lists_the_committed_files_from_a_linked_worktree_hook(tmp_path, monkeypatch):
    # A linked worktree's commit -a hook exports GIT_DIR and an index that already drops a file
    # deleted from disk. The checker lists its own tree's files from that index. This test also
    # runs from that hook, so the fixture's own git commands must not inherit its variables.
    module = checker()
    with patch.dict(os.environ, git_environment(), clear=True):
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
        root = tmp_path / "public"
        (root / "scripts").mkdir(parents=True)
        (root / "scripts" / "stable_identity_exceptions.json").write_text("[]")
        (root / "kept.py").write_text('NAME = "holoscan.test.Current"\n')
        (root / "deleted.py").write_text('NAME = "holoscan.test.Deleted"\n')
        subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
        commit_index = tmp_path / "commit-index"
        shutil.copy(tmp_path / ".git" / "index", commit_index)
        subprocess.run(
            ["git", "-C", str(tmp_path), "rm", "-q", "--cached", "public/deleted.py"],
            env={**os.environ, "GIT_INDEX_FILE": str(commit_index)},
            check=True,
        )
        (root / "deleted.py").unlink()
        monkeypatch.setattr(module, "ROOT", root)
        monkeypatch.setattr(sys, "argv", ["check_stable_identity_names.py"])
        os.environ["GIT_DIR"] = str(tmp_path / ".git")
        os.environ["GIT_INDEX_FILE"] = str(commit_index)
        assert module.main() == 0


def test_python_comments_are_not_literals():
    assert checker().identity_literals('# param_schema("nvidia.example.Old", 1)', python=True) == []


def test_adjacent_cpp_literals_cannot_hide_prefix():
    assert checker().identity_literals('param_schema("com." "nvidia.other.Params", 1)')


@pytest.mark.parametrize(
    "source",
    [
        'param_schema("com.other.Params", 1)',
        'param_schema("holoscan.test.Params.v1", 1)',
        '.validator<&valid>("holoscan.test.validator.even")',
        'parameter_projected_byte_span<&size>("holoscan.test.Op.size", field)',
        'constexpr auto kId = "com.other.Params"; param_schema(kId, 1);',
        'param_schema("com.nvidia.", 1)',
    ],
)
def test_declaration_ownership_and_versions(source):
    assert checker().identity_literals(source)


@pytest.mark.parametrize("name", ["com.other.Payload", "holoscan.test.Payload.v1"])
def test_pod_declaration_ownership_and_separate_version(name):
    source = f'struct ChannelTypeIdentity<Payload> {{ static constexpr auto name = "{name}"; }};'
    assert checker().identity_literals(source)


@pytest.mark.parametrize(
    "literal",
    ['"{name}"sv', 'R"({name})"sv', '"{first}" /* joined */ "{rest}"sv', '"{first}"sv "{rest}"'],
)
@pytest.mark.parametrize("constant", [False, True])
@pytest.mark.parametrize(
    ("declaration", "name", "invalid"),
    [
        ("param_schema({value}, 1)", "com.other.Params", True),
        ("param_schema({value}, 1)", "holoscan.hsb.Params.v1", True),
        ("param_schema({value}, 1)", "holoscan.hsb.Params", False),
        (".validator<&valid>({value})", "holoscan.hsb.validator.even", True),
        (".validator<&valid>({value})", "holoscan.hsb.validator.even.v1", False),
        ("parameter_projected_byte_span<&size>({value}, field)", "com.other.size.v1", True),
        ("parameter_projected_byte_span<&size>({value}, field)", "holoscan.hsb.Op.size", True),
        ("parameter_projected_byte_span<&size>({value}, field)", "holoscan.hsb.Op.size.v1", False),
    ],
)
def test_string_view_declarations(literal, constant, declaration, name, invalid):
    """Standard string-view suffixes preserve ownership and version checks."""
    value = literal.format(name=name, first=name[:4], rest=name[4:])
    source = "\n"
    if constant:
        source += f"constexpr auto kIdentity = {value};\n"
        value = "kIdentity"
    source += declaration.format(value=value)
    assert checker().identity_literals(source) == ([(2, name)] if invalid else [])


@pytest.mark.parametrize(
    "trait",
    ["ChannelTypeIdentity", "holoscan::ChannelTypeIdentity", "::holoscan :: ChannelTypeIdentity"],
)
@pytest.mark.parametrize("constant", [False, True])
@pytest.mark.parametrize(
    ("name", "invalid"),
    [
        ("com.other.Payload", True),
        ("holoscan.example.Payload.v1", True),
        ("holoscan.ext.example.Payload", True),
        ("holoscan.example.Payload", False),
    ],
)
def test_qualified_pod_declarations(trait, constant, name, invalid):
    """Every documented trait spelling enforces the same name/version policy."""
    source = "\n"
    value = f'"{name}"'
    if constant:
        source += f"constexpr auto kIdentity = {value}sv;\n"
        value = "kIdentity"
    source += (
        f"template <> struct {trait}<Payload> {{ static constexpr auto name = {value}; "
        "static constexpr unsigned version = 1; };"
    )
    assert checker().identity_literals(source) == ([(2, name)] if invalid else [])


@pytest.mark.parametrize(
    "source",
    [
        "struct other::ChannelTypeIdentity<Payload> {"
        ' static constexpr auto name = "com.other.Payload"; };',
        "struct holoscan::OtherTrait<Payload> {"
        ' static constexpr auto name = "com.other.Payload"; };',
        "// struct holoscan::ChannelTypeIdentity<Payload> {"
        ' static constexpr auto name = "com.other.Payload"; };',
        'const char* text = R"(struct holoscan::ChannelTypeIdentity<Payload> {'
        ' static constexpr auto name = "com.other.Payload"; };)";',
        'param_schema("com.other.Params"_custom, 1)',
    ],
)
def test_unrelated_or_uninterpreted_declarations(source):
    """Do not interpret comments, string contents, other traits or custom suffixes."""
    assert checker().identity_literals(source) == []


@pytest.mark.parametrize(
    "source",
    [
        'param_schema("holoscan.ext.hsb.Params", 1)',
        '.validator<&valid>("holoscan.ext.hsb.validator.even.v1")',
        'parameter_projected_byte_span<&size>("holoscan.ext.hsb.Op.size.v1", field)',
    ],
)
def test_rejects_obsolete_extension_layer(source):
    """Holoscan-specific provider declarations use the direct provider area."""
    assert checker().identity_literals(source)
