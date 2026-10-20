# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a typing generator with an isolated, optional CUDA driver-stub fallback.

Only trusted, locally built ELF extensions may be inspected with ldd. This tool
does not emulate CUDA execution or install libraries; the child must only need
the driver library for loading and introspection.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def elf_identity(path: Path) -> tuple[int, int, int]:
    """Return ELF class, byte order, and machine without loading the library."""
    with path.open("rb") as stream:
        header = stream.read(20)
    if (
        len(header) != 20
        or header[:4] != b"\x7fELF"
        or header[4] not in (1, 2)
        or header[5] not in (1, 2)
    ):
        raise ValueError(f"Not a supported ELF file: {path}")
    order = "little" if header[5] == 1 else "big"
    return header[4], header[5], int.from_bytes(header[18:20], order)


def missing_libraries(extension: Path, env: dict[str, str]) -> set[str]:
    """Inspect a trusted build artifact using the child's loader environment."""
    result = subprocess.run(
        ["ldd", str(extension)], env=env, capture_output=True, text=True, check=True
    )
    return {line.split()[0] for line in result.stdout.splitlines() if "=> not found" in line}


def run_generator(
    command: list[str],
    *,
    mode: str,
    extensions: list[Path],
    stub_library: Path | None,
    scratch_dir: Path,
) -> int:
    """Run a generator once, preserving its status and never changing os.environ."""
    env = os.environ.copy()
    if mode == "native":
        print("stubgen: native loader environment", flush=True)
        return subprocess.run(command, env=env, check=False).returncode
    if sys.platform != "linux":
        raise ValueError("CUDA stub selection requires Linux; use native mode on other platforms")
    if not extensions:
        raise ValueError("At least one extension is required for CUDA stub selection")
    identity = elf_identity(Path(sys.executable))
    for extension in extensions:
        if elf_identity(extension) != identity:
            raise ValueError(
                f"Extension architecture does not match the generator Python: {extension}"
            )

    missing = set().union(*(missing_libraries(extension, env) for extension in extensions))
    unexpected = missing - {"libcuda.so.1"}
    if unexpected:
        raise ValueError(f"Unresolved non-driver libraries: {', '.join(sorted(unexpected))}")
    if mode == "auto" and "libcuda.so.1" not in missing:
        print("stubgen: native loader environment (no missing CUDA driver library)", flush=True)
        return subprocess.run(command, env=env, check=False).returncode

    if stub_library is None or not stub_library.is_absolute() or not stub_library.is_file():
        raise ValueError("A valid absolute CUDA toolkit stub-library path is required")
    stub_library = stub_library.resolve()
    if elf_identity(stub_library) != identity:
        raise ValueError(
            f"CUDA stub architecture does not match the generator Python: {stub_library}"
        )
    scratch_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cuda-stubgen-", dir=scratch_dir) as temporary:
        (Path(temporary) / "libcuda.so.1").symlink_to(stub_library)
        env["LD_LIBRARY_PATH"] = ":".join(
            [temporary, *filter(None, env.get("LD_LIBRARY_PATH", "").split(":"))]
        )
        print(f"stubgen: CUDA stub selected: {stub_library}", flush=True)
        return subprocess.run(command, env=env, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("auto", "native", "stub"), default="auto")
    parser.add_argument("--extension", action="append", type=Path, default=[])
    parser.add_argument("--stub-library", type=Path)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a generator command is required after --")
    try:
        status = run_generator(
            command,
            mode=args.mode,
            extensions=args.extension,
            stub_library=args.stub_library,
            scratch_dir=args.scratch_dir,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"stubgen: {error}", file=sys.stderr)
        return 1
    return status if status >= 0 else 128 - status


if __name__ == "__main__":
    sys.exit(main())
