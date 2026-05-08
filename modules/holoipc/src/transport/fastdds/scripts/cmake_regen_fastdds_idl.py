#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Single CMake driver for Fast DDS IDL → C++: run fastddsgen, then a postprocess script.
#
# Subcommands:
#   ipc-transport — multiple IDLs from one directory, raw staging, split public/private outputs
#                   (holoipc transport; uses postprocess_idl_gen.py).
#   flat          — one IDL file, optional -I includes, single output directory
#                   (examples; uses postprocess_example_fastdds_idl.py).

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

_PROG = "cmake_regen_fastdds_idl"


def _route_private_hpp(name: str) -> bool:
    return name.endswith("CdrAux.hpp") or name == "control_messageTypeObjectSupport.hpp"


def cmd_ipc_transport(args: argparse.Namespace) -> int:
    idl_dir = args.idl_dir.resolve()
    out_pub = args.out_public_gen.resolve()
    out_priv = args.out_private.resolve()
    post = args.postprocess.resolve()

    if not idl_dir.is_dir():
        print(f"{_PROG} ipc-transport: idl-dir not a directory: {idl_dir}", file=sys.stderr)
        return 1
    if not post.is_file():
        print(f"{_PROG} ipc-transport: postprocess script missing: {post}", file=sys.stderr)
        return 1

    out_pub.mkdir(parents=True, exist_ok=True)
    out_priv.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="holoscan_ipc_fastdds_raw_") as raw_s:
        raw = Path(raw_s)
        for idl_name in args.idl:
            idl_path = idl_dir / idl_name
            if not idl_path.is_file():
                print(f"{_PROG} ipc-transport: missing IDL: {idl_path}", file=sys.stderr)
                return 1
            cmd = [
                args.fastddsgen,
                "-replace",
                "-d",
                str(raw),
                "-no-dependencies",
                str(idl_path),
            ]
            r = subprocess.run(cmd, cwd=idl_dir)
            if r.returncode != 0:
                return r.returncode

        for f in sorted(raw.iterdir()):
            if not f.is_file():
                continue
            suf = f.suffix
            if suf not in {".hpp", ".ipp", ".cxx"}:
                continue
            base = f.name
            if suf == ".hpp":
                dest_dir = out_priv if _route_private_hpp(base) else out_pub
            else:
                dest_dir = out_priv
            dest = dest_dir / base
            proc = subprocess.run(
                [sys.executable, str(post)],
                stdin=f.open("rb"),
                stdout=subprocess.PIPE,
                check=False,
            )
            if proc.returncode != 0:
                print(
                    f"{_PROG} ipc-transport: postprocess failed for {base}",
                    file=sys.stderr,
                )
                return proc.returncode
            dest.write_bytes(proc.stdout)

    return 0


def cmd_flat(args: argparse.Namespace) -> int:
    idl_file = args.idl_file.resolve()
    idl_dir = idl_file.parent
    out_dir = args.out_dir.resolve()
    post = args.postprocess.resolve()
    ipc_inc = args.ipc_idl_include.resolve()

    if not idl_file.is_file():
        print(f"{_PROG} flat: missing IDL {idl_file}", file=sys.stderr)
        return 1
    if not post.is_file():
        print(f"{_PROG} flat: postprocess script missing: {post}", file=sys.stderr)
        return 1
    if not (ipc_inc / "pointer_descriptor.idl").is_file():
        print(
            f"{_PROG} flat: pointer_descriptor.idl not under {ipc_inc}",
            file=sys.stderr,
        )
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.fastddsgen,
        "-replace",
        "-d",
        str(out_dir),
        "-I",
        str(ipc_inc),
        str(idl_file),
        "-no-dependencies",
    ]
    r = subprocess.run(cmd, cwd=idl_dir)
    if r.returncode != 0:
        return r.returncode

    for f in sorted(out_dir.iterdir()):
        if not f.is_file() or f.suffix not in {".hpp", ".cxx", ".ipp"}:
            continue
        raw = f.read_text(encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(post)],
            input=raw.encode("utf-8"),
            stdout=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            print(f"{_PROG} flat: postprocess failed for {f.name}", file=sys.stderr)
            return proc.returncode
        f.write_bytes(proc.stdout)

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="CMake driver: Fast DDS IDL → postprocessed C++ (ipc-transport or flat).",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ipc = sub.add_parser(
        "ipc-transport",
        help="holoipc transport: multiple IDLs, split public/private gen dirs",
    )
    p_ipc.add_argument("--fastddsgen", required=True, help="Path to fastddsgen executable")
    p_ipc.add_argument(
        "--idl-dir", required=True, type=Path, help="Directory containing .idl files"
    )
    p_ipc.add_argument(
        "--idl",
        nargs="+",
        required=True,
        help="IDL filenames (e.g. pointer_descriptor.idl control_message.idl)",
    )
    p_ipc.add_argument(
        "--out-public-gen",
        required=True,
        type=Path,
        help="Output dir for public headers (…/holoscan/ipc/transport/fastdds/gen)",
    )
    p_ipc.add_argument(
        "--out-private",
        required=True,
        type=Path,
        help="Output dir for private gen sources (*.cxx, *.ipp, private *.hpp)",
    )
    p_ipc.add_argument(
        "--postprocess", required=True, type=Path, help="postprocess_idl_gen.py path"
    )
    p_ipc.set_defaults(_run=cmd_ipc_transport)

    p_flat = sub.add_parser(
        "flat",
        help="single IDL + -I to IPC idl, all outputs in one directory (examples)",
    )
    p_flat.add_argument("--fastddsgen", required=True)
    p_flat.add_argument("--idl-file", required=True, type=Path, help="Path to e.g. Buffer.idl")
    p_flat.add_argument(
        "--ipc-idl-include",
        required=True,
        type=Path,
        help="Directory containing pointer_descriptor.idl (holoipc/idl)",
    )
    p_flat.add_argument("--out-dir", required=True, type=Path)
    p_flat.add_argument("--postprocess", required=True, type=Path)
    p_flat.set_defaults(_run=cmd_flat)

    args = ap.parse_args()
    return args._run(args)


if __name__ == "__main__":
    sys.exit(main())
