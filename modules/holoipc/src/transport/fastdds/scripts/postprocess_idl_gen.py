#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Post-process raw fastddsgen output: C++ types under holoscan::ipc::transport::fastdds::gen,
DDS wire type name strings remain holoscan::ipc::*.

Use after raw files are placed: most *.hpp -> include/.../fastdds/gen; *CdrAux.hpp,
ControlMessageTypeObjectSupport.hpp, *.ipp, *.cxx -> src/.../fastdds/gen
(PointerDescriptorTypeObjectSupport.hpp stays public).

- Main headers / PubSubTypes / TypeObject: inserts transport::fastdds::gen namespace when the
  file still has only holoscan::ipc around generated types.
- CdrAux and other files without that block: only qualify C++ type names (safe to re-run).

  python3 postprocess_idl_gen.py < pointer_descriptor.hpp > out.hpp
"""

from __future__ import annotations

import sys


def apply_qualifier_replacements(text: str) -> str:
    """Map holoscan::ipc::Type to transport::fastdds::gen in C++ (not in quoted wire strings)."""
    reps = [
        ("::holoscan::ipc::register_", "::holoscan::ipc::transport::fastdds::gen::register_"),
        (
            "::holoscan::ipc::ControlMessageType",
            "::holoscan::ipc::transport::fastdds::gen::ControlMessageType",
        ),
        (
            "::holoscan::ipc::ControlMessage",
            "::holoscan::ipc::transport::fastdds::gen::ControlMessage",
        ),
        (
            "::holoscan::ipc::PointerDescriptor",
            "::holoscan::ipc::transport::fastdds::gen::PointerDescriptor",
        ),
        ("::holoscan::ipc::HandleType", "::holoscan::ipc::transport::fastdds::gen::HandleType"),
        (
            "const holoscan::ipc::ControlMessageType",
            "const holoscan::ipc::transport::fastdds::gen::ControlMessageType",
        ),
        (
            "const holoscan::ipc::ControlMessage",
            "const holoscan::ipc::transport::fastdds::gen::ControlMessage",
        ),
        (
            "const holoscan::ipc::PointerDescriptor",
            "const holoscan::ipc::transport::fastdds::gen::PointerDescriptor",
        ),
        (
            "holoscan::ipc::ControlMessageType&",
            "holoscan::ipc::transport::fastdds::gen::ControlMessageType&",
        ),
        (
            "holoscan::ipc::ControlMessage&",
            "holoscan::ipc::transport::fastdds::gen::ControlMessage&",
        ),
        (
            "holoscan::ipc::PointerDescriptor&",
            "holoscan::ipc::transport::fastdds::gen::PointerDescriptor&",
        ),
    ]
    for a, b in reps:
        text = text.replace(a, b)
    return text


def _ipc_namespace_already_opens_transport_fastdds_gen(text: str) -> bool:
    """True if namespace ipc { is already followed by transport::fastdds::gen (wrapped once).

    Raw fastddsgen has holoscan/ipc then types; a single post-process adds transport/fastdds/gen
    under ipc. Matching only 'ipc then transport' avoids double-wrap; do not use a pattern that
    matches the first segment of an already-wrapped file as 'still raw'.
    """
    marker = "namespace ipc {"
    idx = text.find(marker)
    if idx < 0:
        return False
    window = text[idx : idx + 2500]
    return "namespace transport {" in window


def try_wrap_namespace(text: str) -> str:
    """If file is raw fastddsgen (holoscan::ipc only), wrap in transport::fastdds::gen."""
    if _ipc_namespace_already_opens_transport_fastdds_gen(text):
        return text

    pairs = [
        (
            "} // namespace ipc\n\n} // namespace holoscan",
            "} // namespace gen\n\n} // namespace fastdds\n\n} // namespace transport\n\n"
            "} // namespace ipc\n\n} // namespace holoscan",
        ),
        (
            "}  // namespace ipc\n}  // namespace holoscan",
            "}  // namespace gen\n}  // namespace fastdds\n}  // namespace transport\n"
            "}  // namespace ipc\n}  // namespace holoscan",
        ),
    ]

    old_open = "namespace holoscan {\n\nnamespace ipc {\n"
    new_open = (
        "namespace holoscan {\n\nnamespace ipc {\n\n"
        "namespace transport {\n\nnamespace fastdds {\n\nnamespace gen {\n"
    )
    if old_open in text:
        text = text.replace(old_open, new_open, 1)
        for old, new in pairs:
            if old in text:
                return text.replace(old, new, 1)
        raise ValueError("fastdds: closing pattern not found (expected blank line after holoscan)")

    old_open2 = "namespace holoscan {\nnamespace ipc {\n"
    new_open2 = (
        "namespace holoscan {\nnamespace ipc {\n"
        "namespace transport {\nnamespace fastdds {\nnamespace gen {\n"
    )
    if old_open2 in text:
        text = text.replace(old_open2, new_open2, 1)
        for old, new in pairs:
            if old in text:
                return text.replace(old, new, 1)
        raise ValueError("fastdds: closing pattern not found (tight holoscan/ipc open)")

    return text


def postprocess(text: str) -> str:
    text = try_wrap_namespace(text)
    return apply_qualifier_replacements(text)


def main() -> None:
    data = sys.stdin.read()
    sys.stdout.write(postprocess(data))


if __name__ == "__main__":
    main()
