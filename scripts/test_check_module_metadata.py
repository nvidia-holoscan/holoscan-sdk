#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the Core Module metadata.json checker."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import check_module_metadata as checker  # noqa: E402


def _metadata(name: str = "example", version: str = "5.0.0") -> dict:
    return {
        "$schema": "urn:holoscan:module:v2",
        "module": {
            "name": f"holoscan-{name}",
            "version": version,
            "description": "Example module",
            "authors": [{"name": "Holoscan Team", "affiliation": "NVIDIA"}],
            "license": "Apache-2.0",
            "language": ["C++"],
            "namespace": {"cpp": f"holoscan::{name}"},
            "holoscan_sdk": {"minimum_required_version": version, "tested_versions": [version]},
            "platforms": ["x86_64", "aarch64"],
            "tags": ["Example"],
            "source_repository": "https://github.com/nvidia-holoscan/holoscan-sdk",
            "binary_packages": {"debian": f"holoscan-{name}"},
            "documentation": {"readme": "README.md"},
            "testing": {"cpp": "ctest"},
        },
    }


@pytest.fixture
def sdk_root(tmp_path: Path) -> Path:
    (tmp_path / "VERSION").write_text("5.0.0\n", encoding="utf-8")
    modules = tmp_path / "modules"
    modules.mkdir()
    shutil.copy(SCRIPT_DIR.parent / checker.SCHEMA_RELATIVE_PATH, modules)
    return tmp_path


def _add_module(sdk_root: Path, name: str, metadata: dict | None) -> Path:
    module_dir = sdk_root / "modules" / name
    module_dir.mkdir()
    (module_dir / "CMakeLists.txt").write_text("", encoding="utf-8")
    (module_dir / "README.md").write_text("# Example\n", encoding="utf-8")
    if metadata is not None:
        (module_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return module_dir


def test_sdk_tree_modules_pass() -> None:
    assert checker.check_all(SCRIPT_DIR.parent) == []


def test_accepts_conforming_module(sdk_root: Path) -> None:
    _add_module(sdk_root, "example", _metadata())
    assert checker.check_all(sdk_root) == []
    assert checker.main(["--root", str(sdk_root)]) == 0


def test_ignores_directories_without_cmakelists(sdk_root: Path) -> None:
    (sdk_root / "modules" / "notes").mkdir()
    assert checker.check_all(sdk_root) == []


def test_rejects_missing_metadata(sdk_root: Path) -> None:
    _add_module(sdk_root, "example", None)
    [error] = checker.check_all(sdk_root)
    assert "missing" in error
    assert checker.main(["--root", str(sdk_root)]) == 1


def test_rejects_invalid_json(sdk_root: Path) -> None:
    module_dir = _add_module(sdk_root, "example", None)
    (module_dir / "metadata.json").write_text("{", encoding="utf-8")
    [error] = checker.check_all(sdk_root)
    assert "invalid JSON" in error


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda m: m.update({"$schema": "urn:holohub:module:v2"}), "urn:holoscan:module:v2"),
        (lambda m: m["module"].pop("license"), "'license' is a required property"),
        (lambda m: m["module"].update({"platforms": ["windows"]}), "is not one of"),
        (lambda m: m["module"].update({"unknown": 1}), "Additional properties"),
        (lambda m: m["module"].update({"homepage": "not a uri"}), "is not a 'uri'"),
    ],
)
def test_rejects_schema_violations(sdk_root: Path, mutate, expected: str) -> None:
    metadata = _metadata()
    mutate(metadata)
    _add_module(sdk_root, "example", metadata)
    errors = checker.check_all(sdk_root)
    assert any(expected in error for error in errors), errors


def test_rejects_name_and_package_mismatch(sdk_root: Path) -> None:
    metadata = _metadata(name="other")
    _add_module(sdk_root, "example", metadata)
    errors = checker.check_all(sdk_root)
    assert any("module.name" in error for error in errors), errors
    assert any("binary_packages.debian" in error for error in errors), errors


def test_rejects_version_drift_from_sdk(sdk_root: Path) -> None:
    _add_module(sdk_root, "example", _metadata(version="0.1.0"))
    errors = checker.check_all(sdk_root)
    assert any("module.version" in error for error in errors), errors
    assert any("minimum_required_version" in error for error in errors), errors
    assert any("tested_versions" in error for error in errors), errors


def test_rejects_missing_readme(sdk_root: Path) -> None:
    module_dir = _add_module(sdk_root, "example", _metadata())
    (module_dir / "README.md").unlink()
    [error] = checker.check_all(sdk_root)
    assert "documentation.readme" in error
