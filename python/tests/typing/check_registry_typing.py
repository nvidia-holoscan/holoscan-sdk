# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check pinned typing contracts with pyright and ty machine output.

Without explicit fixture arguments this checks the registry inference contracts against a
generated test-provider package. With ``--positive``/``--negative``/``--pyright-negative`` it
checks the given fixture files directly against ``--package-root`` so other packages (for
example ``holoscan_visualization``) can pin their own client-facing typing contracts.
"""

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

# ``# typing-error[pyright]: shape`` pins a line only that checker reports, for example when one
# multi-line call yields a call-level error in both checkers and an argument-level error in one.
_MARKER = re.compile(r"# typing-error(?:\[(?P<checker>pyright|ty)\])?: (?P<shape>[a-z0-9-]+)\s*$")


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


def _markers(path: Path, checker: str) -> dict[int, str]:
    result: dict[int, str] = {}
    for line_number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = _MARKER.search(text)
        if match and match.group("checker") in (None, checker):
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
            "holoscan.test.",
            "--module-doc",
            "Test-only registered Holoscan Operator proxies.",
        ],
        environment=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"test-provider stub generation failed:\n{completed.stdout}{completed.stderr}"
        )

    # Constructor-handle rows type their handle keywords with the provider's own classes.
    shutil.copy(
        Path(__file__).resolve().parent / "holoscan_constructor_handle_provider.pyi",
        test_root / "holoscan_constructor_handle_provider.pyi",
    )
    handle_package = test_root / "holoscan_test_handle_operators"
    handle_package.mkdir()
    (handle_package / "__init__.py").write_text(
        '"""Typing-only constructor-handle Operator provider."""\n', encoding="utf-8"
    )
    completed = _run(
        [
            sys.executable,
            str(operator_generator),
            "--output",
            str(handle_package / "__init__.pyi"),
            "--provider",
            "holoscan_constructor_handle_provider",
            "--registry-prefix",
            "holoscan.test.py.",
            "--handle-annotation",
            "holoscan.test.py.TestChannel=holoscan_constructor_handle_provider.TestChannel",
            "--handle-annotation",
            "holoscan.test.py.TestSensor=holoscan_constructor_handle_provider.TestSensor",
            "--module-doc",
            "Test-only constructor-handle Operator proxies.",
        ],
        environment=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"handle-provider stub generation failed:\n{completed.stdout}{completed.stderr}"
        )
    return test_root


def _check_fixtures(
    package_root: Path,
    *,
    positives: list[Path],
    negatives: list[Path],
    pyright_negatives: list[Path],
    pyright: str,
    ty: str,
) -> int:
    """Require zero diagnostics on positives and exactly the marked lines on negatives."""

    shared_shapes = 0
    for positive in positives:
        _require_exact("pyright", positive, _pyright(positive, pyright, package_root), {})
        _require_exact("ty", positive, _ty(positive, ty, package_root), {})
    for negative in negatives:
        pyright_markers = _markers(negative, "pyright")
        ty_markers = _markers(negative, "ty")
        shared_shapes += len(pyright_markers.keys() & ty_markers.keys())
        _require_exact(
            "pyright", negative, _pyright(negative, pyright, package_root), pyright_markers
        )
        _require_exact("ty", negative, _ty(negative, ty, package_root), ty_markers)
    for negative in pyright_negatives:
        markers = _markers(negative, "pyright")
        _require_exact("pyright", negative, _pyright(negative, pyright, package_root), markers)
        _require_exact("ty 0.0.78 known PEP 692 gap", negative, _ty(negative, ty, package_root), {})
    return shared_shapes


def main() -> int:
    """Run and compare the pinned typing contracts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--provider-root", type=Path)
    parser.add_argument("--operator-generator", type=Path)
    parser.add_argument("--positive", type=Path, action="append", default=[])
    parser.add_argument("--negative", type=Path, action="append", default=[])
    parser.add_argument("--pyright-negative", type=Path, action="append", default=[])
    parser.add_argument("--pyright", default="pyright")
    parser.add_argument("--ty", default="ty")
    args = parser.parse_args()
    positives = cast("list[Path]", args.positive)
    negatives = cast("list[Path]", args.negative)
    pyright_negatives = cast("list[Path]", args.pyright_negative)

    if positives or negatives or pyright_negatives:
        shared = _check_fixtures(
            args.package_root,
            positives=positives,
            negatives=negatives,
            pyright_negatives=pyright_negatives,
            pyright=args.pyright,
            ty=args.ty,
        )
        print(
            f"typing fixtures: pyright and ty agree on {len(positives)} positive file(s) and "
            f"{shared} shared negative shapes"
        )
        return 0

    if args.provider_root is None or args.operator_generator is None:
        parser.error("registry mode requires --provider-root and --operator-generator")

    fixture_root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="holoscan-registry-typing-") as directory:
        test_package_root = _prepare_test_provider_package(
            Path(directory),
            args.package_root,
            args.provider_root,
            args.operator_generator,
        )
        shared = _check_fixtures(
            test_package_root,
            positives=[fixture_root / "positive_registry.py"],
            negatives=[fixture_root / "negative_registry.py"],
            pyright_negatives=[fixture_root / "negative_registry_pyright.py"],
            pyright=args.pyright,
            ty=args.ty,
        )

    print(
        "registry typing: pyright and ty agree on "
        f"{shared} shared negative shapes; "
        "pyright additionally enforces the pinned unexpected-keyword contract"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
