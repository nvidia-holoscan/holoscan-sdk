# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check registry inference contracts with pyright and ty machine output."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

_MARKER = re.compile(r"# typing-error: (?P<shape>[a-z0-9-]+)\s*$")


@dataclass(frozen=True)
class Diagnostic:
    """One normalized type-checker error."""

    line: int
    rule: str
    message: str


def _run(
    command: list[str], *, environment: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True, env=environment
    )
    if completed.returncode not in (0, 1):
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise RuntimeError(f"type checker exited with status {completed.returncode}: {command[0]}")
    return completed


def _pyright(path: Path, executable: str, package_root: Path) -> list[Diagnostic]:
    config = {
        "extraPaths": [str(package_root)],
        "include": [str(path)],
        "pythonVersion": "3.12",
        "reportIncompleteStub": False,
        "reportUnsupportedDunderAll": False,
        "typeCheckingMode": "standard",
    }
    with tempfile.TemporaryDirectory(prefix="holoscan-registry-pyright-") as directory:
        config_path = Path(directory) / "pyrightconfig.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        completed = _run([executable, "--project", str(config_path), "--outputjson", str(path)])
    report = cast("dict[str, Any]", json.loads(completed.stdout))
    return [
        Diagnostic(
            line=int(item["range"]["start"]["line"]) + 1,
            rule=str(item.get("rule", "unknown")),
            message=str(item["message"]),
        )
        for item in report["generalDiagnostics"]
        if item["severity"] == "error"
    ]


def _ty(path: Path, executable: str, package_root: Path) -> list[Diagnostic]:
    completed = _run(
        [
            executable,
            "check",
            "--output-format",
            "gitlab",
            "--no-progress",
            "--extra-search-path",
            str(package_root),
            str(path),
        ]
    )
    report = cast("list[dict[str, Any]]", json.loads(completed.stdout))
    return [
        Diagnostic(
            line=int(item["location"]["positions"]["begin"]["line"]),
            rule=str(item["check_name"]),
            message=str(item["description"]),
        )
        for item in report
        if item["severity"] in {"critical", "major"}
    ]


def _markers(path: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    for line_number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if match := _MARKER.search(text):
            result[line_number] = match.group("shape")
    return result


def _require_exact(
    checker: str,
    path: Path,
    diagnostics: list[Diagnostic],
    expected: dict[int, str],
) -> None:
    actual_lines = {diagnostic.line for diagnostic in diagnostics}
    expected_lines = set(expected)
    if actual_lines == expected_lines:
        return
    missing = [expected[line] for line in sorted(expected_lines - actual_lines)]
    extra = sorted(actual_lines - expected_lines)
    details = "\n".join(
        f"  line {diagnostic.line} [{diagnostic.rule}]: {diagnostic.message}"
        for diagnostic in diagnostics
    )
    raise AssertionError(
        f"{checker} diagnostics diverged for {path.name}: missing shapes={missing}, "
        f"unexpected lines={extra}\n{details}"
    )


def _prepare_test_provider_package(
    destination: Path,
    package_root: Path,
    provider_root: Path,
    operator_generator: Path,
) -> Path:
    """Copy Core and project explicitly loaded test rows into a provider package."""

    test_root = destination / "site-packages"
    shutil.copytree(package_root / "holoscan", test_root / "holoscan")
    provider_package = test_root / "holoscan_test_operators"
    provider_package.mkdir()
    (provider_package / "__init__.py").write_text(
        '"""Typing-only test Operator provider."""\n', encoding="utf-8"
    )
    environment = os.environ.copy()
    python_path = [str(provider_root), str(package_root)]
    if inherited := environment.get("PYTHONPATH"):
        python_path.append(inherited)
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    completed = _run(
        [
            sys.executable,
            str(operator_generator),
            "--output",
            str(provider_package / "__init__.pyi"),
            "--provider",
            "holoscan_test_support",
            "--registry-prefix",
            "holoscan.tests.",
            "--module-doc",
            "Test-only registered Holoscan Operator proxies.",
        ],
        environment=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"test-provider stub generation failed:\n{completed.stdout}{completed.stderr}"
        )
    return test_root


def main() -> int:
    """Run and compare the pinned registry typing contracts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--provider-root", type=Path, required=True)
    parser.add_argument("--operator-generator", type=Path, required=True)
    parser.add_argument("--pyright", default="pyright")
    parser.add_argument("--ty", default="ty")
    args = parser.parse_args()

    fixture_root = Path(__file__).resolve().parent
    positive = fixture_root / "positive_registry.py"
    shared_negative = fixture_root / "negative_registry.py"
    pyright_negative = fixture_root / "negative_registry_pyright.py"

    with tempfile.TemporaryDirectory(prefix="holoscan-registry-typing-") as directory:
        test_package_root = _prepare_test_provider_package(
            Path(directory),
            args.package_root,
            args.provider_root,
            args.operator_generator,
        )
        for checker, diagnostics in (
            ("pyright", _pyright(positive, args.pyright, test_package_root)),
            ("ty", _ty(positive, args.ty, test_package_root)),
        ):
            _require_exact(checker, positive, diagnostics, {})

        shared_markers = _markers(shared_negative)
        pyright_shared = _pyright(shared_negative, args.pyright, test_package_root)
        ty_shared = _ty(shared_negative, args.ty, test_package_root)
        _require_exact("pyright", shared_negative, pyright_shared, shared_markers)
        _require_exact("ty", shared_negative, ty_shared, shared_markers)

        pyright_only_markers = _markers(pyright_negative)
        _require_exact(
            "pyright",
            pyright_negative,
            _pyright(pyright_negative, args.pyright, test_package_root),
            pyright_only_markers,
        )
        _require_exact(
            "ty 0.0.78 known PEP 692 gap",
            pyright_negative,
            _ty(pyright_negative, args.ty, test_package_root),
            {},
        )

    print(
        "registry typing: pyright and ty agree on "
        f"{len(shared_markers)} shared negative shapes; "
        "pyright additionally enforces the pinned unexpected-keyword contract"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
