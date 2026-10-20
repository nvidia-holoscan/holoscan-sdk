# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay one Linux GNU-style compiler command with private output files."""

import argparse
import json
import shlex
import subprocess
import tempfile
from pathlib import Path


def isolated_command(database, target, output_dir):
    """Preserve the generated compile settings, redirecting object and diagnostic outputs."""
    marker = f"/CMakeFiles/{target}.dir/"
    entries = [entry for entry in database if marker in "/" + entry.get("output", "")]
    if len(entries) != 1:
        raise ValueError(f"Expected one compile command for {target}, found {len(entries)}")
    entry = entries[0]
    command = list(entry.get("arguments") or shlex.split(entry["command"]))
    # The supported generators expand their response files in the compile database.
    # Fail closed if that changes: do not accidentally write to a shared output.
    if any(arg.startswith("@") for arg in command):
        raise ValueError("Unexpanded compiler response file in compile database")
    if command.count("-o") != 1 or "-c" not in command:
        raise ValueError("Expected a GNU-style compile-only command with one -o output")
    for flag, filename in (
        ("-o", "probe.o"),
        ("-MF", "probe.d"),
        ("-MJ", "probe.json"),
        ("--serialize-diagnostics", "probe.dia"),
    ):
        if flag in command:
            index = command.index(flag) + 1
            if index == len(command):
                raise ValueError(f"Missing value for {flag}")
            command[index] = str(output_dir / filename)
    return command, entry["directory"]


def verify_result(result, diagnostic):
    """Reject a successful compile, a killed compiler, or the wrong diagnostic."""
    if result.returncode == 0:
        raise ValueError("Probe unexpectedly compiled successfully")
    if result.returncode < 0:
        raise ValueError(f"Compiler terminated by signal {-result.returncode}")
    if diagnostic != "HOLOSCAN_ACCEPT_ANY_DIAGNOSTIC" and diagnostic not in result.stdout:
        raise ValueError(f"Compile failed without the expected diagnostic {diagnostic!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--diagnostic", required=True)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    # A new directory also prevents a stale successful object from influencing reruns.
    with tempfile.TemporaryDirectory(dir=args.work_dir) as temporary:
        command, directory = isolated_command(
            json.loads(args.database.read_text()), args.target, Path(temporary)
        )
        (args.work_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        result = subprocess.run(
            command, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        (args.work_dir / "compiler.log").write_text(result.stdout)
        try:
            verify_result(result, args.diagnostic)
        except ValueError:
            print(result.stdout)
            raise
    print(f"{args.target}: rejected with the expected diagnostic")


if __name__ == "__main__":
    main()
