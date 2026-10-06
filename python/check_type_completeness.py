# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enforce pyright's public type-completeness score for an installed package."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any, cast


def main() -> int:
    """Run pyright and reject a score below the configured percentage."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package")
    parser.add_argument("--minimum", type=float, default=100.0)
    parser.add_argument("--pyright", default="pyright")
    args = parser.parse_args()

    command = [
        args.pyright,
        "--verifytypes",
        args.package,
        "--ignoreexternal",
        "--outputjson",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        return completed.returncode

    report = cast("dict[str, Any]", json.loads(completed.stdout))
    completeness = cast("dict[str, Any]", report["typeCompleteness"])
    counts = cast("dict[str, int]", completeness["exportedSymbolCounts"])
    score = float(completeness["completenessScore"]) * 100.0
    known = counts["withKnownType"]
    ambiguous = counts["withAmbiguousType"]
    unknown = counts["withUnknownType"]
    print(
        f"{args.package}: {score:.2f}% public type completeness "
        f"({known} known, {ambiguous} ambiguous, {unknown} unknown)"
    )
    if score + 1e-9 < args.minimum:
        print(f"required minimum: {args.minimum:.2f}%", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
