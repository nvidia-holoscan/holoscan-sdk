#!/usr/bin/env python3
"""
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Compile a Slang shader to SPIR-V with slangc and emit a generated C++ header.

The header embeds the SPIR-V for each requested (vertex, fragment) entry-point
pair together with the shader's reflected interface, exposed as a
``make_shader_blob()`` factory returning a ready ``holoviz::ShaderBlob``.
The reflection rules (supported vertex attribute formats, binding kinds,
rejections) intentionally mirror the RHI runtime contract documented on
``holoviz::ShaderBlob``: space-0 bindings only, struct-wrapped vertex
inputs, read-only plain 2D textures / structured buffers / samplers, push
constants via a single push-constant buffer. Unknown reflection kinds are hard
errors, never silently defaulted.

Usage (from the holoscan-sdk repo root):

    python3 public/modules/visualization/cmake/holoviz_shader_compiler.py \\
        --slangc <path-to-slangc> \\
        --input public/modules/visualization/src/text/monospace.slang \\
        --emit-header out/monospace.slang.hpp \\
        --pair main=vertex_main:fragment_main

A file with several entry-point pairs nests one namespace per pair name:

    ... --input holoviz_geometry.slang --emit-header out/holoviz_geometry.slang.hpp \\
        --pair line_list=vertex_line_list:fragment_main \\
        --pair oval=vertex_oval:fragment_main

With exactly one pair the members live directly in the shader namespace
(default ``holoviz::<input file stem>``, override with --namespace).

Shared interface headers. A shader may ``#include`` a header that is also valid
C++ (see <holoviz/shader_interop.hpp>) so its push-constant struct and shared
constants have one definition. ``--include-dir`` adds slangc include paths for
such headers, ``--emit-depfile`` records every file the shader pulled in so the
build regenerates when one changes, and ``--push-constants-type`` names the C++
struct the shader's push-constant block is declared with; the generated header
then ``#include``s ``--push-constants-include`` and ``static_assert``s the
struct's size and every field offset against the reflected layout, making any
drift between the two languages a compile error:

    ... --include-dir public/modules/visualization/include \\
        --push-constants-type holoviz::imgui::detail::PushConstants \\
        --push-constants-include imgui/shader_interface.hpp \\
        --emit-depfile out/shader.slang.hpp.d
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
from dataclasses import dataclass, field
from pathlib import Path

# C++ enumerator names, keyed by (scalarType, element count) of the reflected
# vertex attribute type. Values are (Format enumerator, byte size).
_ATTRIBUTE_FORMATS = {
    ("float32", 1): ("kR32Sfloat", 4),
    ("float32", 2): ("kR32G32Sfloat", 8),
    ("float32", 3): ("kR32G32B32Sfloat", 12),
    ("float32", 4): ("kR32G32B32A32Sfloat", 16),
    ("uint32", 1): ("kR32Uint", 4),
}

# Reflected resource shape modifiers that no plain RHI Image can satisfy
# (Texture2DArray, Texture2DMS, Sampler2D, shadow/feedback variants).
_TEXTURE_EXT_SHAPE_KEYS = ("array", "multisample", "combined", "shadow", "feedback")

_BYTES_PER_LINE = 12


class ShaderCompilerError(Exception):
    """A compilation or reflection failure with a user-facing message."""


@dataclass(frozen=True)
class Attribute:
    """One reflected vertex input; mirrors holoviz::AttributeDesc."""

    name: str
    location: int
    format: str  # holoviz::Format enumerator name
    byte_size: int


@dataclass(frozen=True)
class Binding:
    """One reflected resource binding; mirrors holoviz::BindingDesc."""

    name: str
    index: int
    kind: str  # holoviz::BindingKind enumerator name
    count: int = 1


@dataclass
class ReflectedInterface:
    """The reflected interface of one (vertex, fragment) pair; mirrors ShaderBlob."""

    pair_name: str
    vertex_entry_point: str
    fragment_entry_point: str
    attributes: list[Attribute] = field(default_factory=list)
    bindings: list[Binding] = field(default_factory=list)
    push_constant_size: int = 0
    # Top-level (name, byte offset) of each push-constant block field, in
    # declaration order; empty when the shader declares no push constants.
    push_constant_fields: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class CompiledPair:
    interface: ReflectedInterface
    vertex_spirv: bytes
    fragment_spirv: bytes


@dataclass(frozen=True)
class PushConstantsSpec:
    """The C++ side of a shared push-constant struct to assert the reflected layout against."""

    cpp_type: str  # fully qualified, e.g. holoviz::imgui::detail::PushConstants
    include: str  # header spelled as the consumer includes it, e.g. imgui/shader_interface.hpp


def _run_slangc(slangc: str, args: list[str]) -> None:
    result = subprocess.run(
        [slangc, *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        diagnostics = result.stderr.decode(errors="replace").strip()
        raise ShaderCompilerError(f"slangc failed (exit {result.returncode}):\n{diagnostics}")


def _common_slangc_args(input_path: str, include_dirs: list[str]) -> list[str]:
    args = [input_path, "-target", "spirv", "-emit-spirv-directly", "-profile", "spirv_1_5"]
    for include_dir in include_dirs:
        args.extend(["-I", include_dir])
    return args


def _read_depfile_sources(depfile_path: str) -> list[str]:
    """Return the prerequisites of the single make rule slangc wrote with -depfile.

    slangc emits ``<output>: <dep> <dep...>`` on one logical line; paths with
    spaces are backslash-escaped the way make expects.
    """
    text = Path(depfile_path).read_text(encoding="utf-8").replace("\\\n", " ")
    _, sep, prerequisites = text.partition(":")
    if not sep:
        raise ShaderCompilerError(f"slangc wrote a depfile without a rule: {depfile_path}")
    return [p.replace("\\ ", " ") for p in re.split(r"(?<!\\) +", prerequisites.strip()) if p]


def compile_stage_spirv(
    slangc: str,
    input_path: str,
    entry_point: str,
    stage: str,
    work_dir: str,
    include_dirs: list[str],
    sources: set[str] | None = None,
) -> bytes:
    """Compile one entry point to SPIR-V. slangc rejects multiple -o outputs per
    target, so each stage is its own invocation. When ``sources`` is given, every
    file slangc read (the input and anything it #included) is added to it."""
    out_path = os.path.join(work_dir, f"{entry_point}.{stage}.spv")
    args = [
        *_common_slangc_args(input_path, include_dirs),
        "-entry",
        entry_point,
        "-stage",
        stage,
        "-o",
        out_path,
    ]
    depfile_path = None
    if sources is not None:
        depfile_path = out_path + ".d"
        args.extend(["-depfile", depfile_path])
    _run_slangc(slangc, args)
    spirv = Path(out_path).read_bytes()
    if len(spirv) < 4 or len(spirv) % 4 != 0 or spirv[:4] != b"\x03\x02\x23\x07":
        raise ShaderCompilerError(f"slangc produced invalid SPIR-V for entry '{entry_point}'")
    if depfile_path is not None:
        sources.update(_read_depfile_sources(depfile_path))
    return spirv


def reflect_pair_json(
    slangc: str,
    input_path: str,
    vertex_entry: str,
    fragment_entry: str,
    work_dir: str,
    include_dirs: list[str],
) -> dict:
    """Run slangc once with both entry points to get the composed program's
    reflection (matches how the RHI links VS+FS into one program). Without -o
    slangc dumps SPIR-V to stdout, which _run_slangc discards."""
    json_path = os.path.join(work_dir, f"{vertex_entry}.{fragment_entry}.reflection.json")
    _run_slangc(
        slangc,
        [
            *_common_slangc_args(input_path, include_dirs),
            "-entry",
            vertex_entry,
            "-stage",
            "vertex",
            "-entry",
            fragment_entry,
            "-stage",
            "fragment",
            "-reflection-json",
            json_path,
        ],
    )
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def _is_system_value(semantic_name: str | None) -> bool:
    return semantic_name is not None and semantic_name.startswith("SV_")


def _attribute_format(type_desc: dict, field_name: str) -> tuple[str, int]:
    kind = type_desc.get("kind")
    if kind == "scalar":
        scalar, count = type_desc.get("scalarType"), 1
    elif kind == "vector":
        scalar = type_desc.get("elementType", {}).get("scalarType")
        count = type_desc.get("elementCount", 0)
    else:
        scalar, count = None, 0
    fmt = _ATTRIBUTE_FORMATS.get((scalar, count))
    if fmt is None:
        raise ShaderCompilerError(
            f"vertex attribute '{field_name}' is not a recognized vertex attribute type "
            f"(got {json.dumps(type_desc)}); supported: float, float2..4, uint"
        )
    return fmt


def _reflect_vertex_attributes(reflection: dict, vertex_entry: str) -> list[Attribute]:
    entry = next(
        (
            e
            for e in reflection.get("entryPoints", [])
            if e.get("name") == vertex_entry and e.get("stage") == "vertex"
        ),
        None,
    )
    if entry is None:
        raise ShaderCompilerError(
            f"vertex entry point '{vertex_entry}' not present in reflection output"
        )

    attributes: list[Attribute] = []
    for param in entry.get("parameters", []):
        type_desc = param.get("type", {})
        if type_desc.get("kind") != "struct":
            # System-value inputs (e.g. `uint vid : SV_VertexID`) are legal direct
            # parameters and contribute no vertex attribute.
            if _is_system_value(param.get("semanticName")):
                continue
            # Any other direct varying input would need its own location/format
            # contract; the supported grammar is a single input struct instead.
            if param.get("binding", {}).get("kind") == "varyingInput":
                raise ShaderCompilerError(
                    f"direct vertex input '{param.get('name')}' is not supported; "
                    "wrap user vertex inputs in a struct parameter"
                )
            continue

        for struct_field in type_desc.get("fields", []):
            if _is_system_value(struct_field.get("semanticName")):
                continue
            name = struct_field.get("name", "<unnamed>")
            binding = struct_field.get("binding", {})
            if binding.get("kind") != "varyingInput":
                raise ShaderCompilerError(
                    f"vertex input struct field '{name}' has unexpected binding kind "
                    f"'{binding.get('kind')}' (expected varyingInput)"
                )
            fmt, byte_size = _attribute_format(struct_field.get("type", {}), name)
            attributes.append(
                Attribute(
                    name=name,
                    location=binding.get("index", 0),
                    format=fmt,
                    byte_size=byte_size,
                )
            )
    return attributes


def _binding_kind(type_desc: dict, param_name: str) -> str:
    kind = type_desc.get("kind")
    if kind == "samplerState":
        return "kSampler"

    if kind == "resource":
        access = type_desc.get("access", "read")
        if access != "read":
            raise ShaderCompilerError(
                f"binding '{param_name}': only read-only resources are supported "
                f"(access is '{access}')"
            )
        base_shape = type_desc.get("baseShape")
        if base_shape in ("structuredBuffer", "byteAddressBuffer"):
            return "kStorageBuffer"
        if base_shape == "texture2D":
            # The public Image API creates plain 2D images only, so arrayed/
            # multisampled/combined/shadow/feedback variants reflect an
            # interface no RHI image could satisfy.
            modifiers = [k for k in _TEXTURE_EXT_SHAPE_KEYS if type_desc.get(k)]
            if modifiers:
                raise ShaderCompilerError(
                    f"binding '{param_name}': arrayed/multisampled/combined texture "
                    f"bindings are not supported ({', '.join(modifiers)}); only plain Texture2D"
                )
            return "kSampledImage"
        if base_shape in ("texture1D", "texture3D", "textureCube"):
            raise ShaderCompilerError(
                f"binding '{param_name}': only 2D sampled images are supported (RHI images are 2D)"
            )
        raise ShaderCompilerError(
            f"binding '{param_name}': resource shape '{base_shape}' is not a recognized "
            "resource type"
        )

    raise ShaderCompilerError(
        f"binding '{param_name}': type kind '{kind}' is not a recognized resource type"
    )


def _push_constant_layout(param: dict) -> tuple[int, list[tuple[str, int]]]:
    """Return the push-constant block's uniform size and its (field, offset) list.

    The size is the element's uniform layout size (alignment padding included),
    i.e. what slang::TypeLayoutReflection::getSize(Uniform) reports on the
    element type. The field offsets are what the generated header asserts a
    shared C++ struct against.
    """
    element = param.get("type", {}).get("elementVarLayout", {})
    element_binding = element.get("binding", {})
    if element_binding.get("kind") != "uniform" or "size" not in element_binding:
        raise ShaderCompilerError(
            f"push constant buffer '{param.get('name')}': reflection lacks the element "
            "uniform layout size"
        )
    fields: list[tuple[str, int]] = []
    for struct_field in element.get("type", {}).get("fields", []):
        binding = struct_field.get("binding", {})
        if binding.get("kind") != "uniform" or "offset" not in binding:
            raise ShaderCompilerError(
                f"push constant buffer '{param.get('name')}': field "
                f"'{struct_field.get('name')}' has no uniform offset in the reflection"
            )
        fields.append((struct_field.get("name", "<unnamed>"), binding["offset"]))
    return element_binding["size"], fields


def _reflect_global_parameters(
    reflection: dict,
) -> tuple[list[Binding], int, list[tuple[str, int]]]:
    bindings: list[Binding] = []
    push_constant_size = 0
    push_constant_fields: list[tuple[str, int]] = []
    push_constant_name: str | None = None
    for param in reflection.get("parameters", []):
        name = param.get("name", "<unnamed>")
        binding = param.get("binding", {})
        category = binding.get("kind")

        if category == "pushConstantBuffer":
            # The pipeline layout exposes exactly one push-constant range, so a second block
            # cannot be represented; keeping only the last size would silently mis-size it.
            if push_constant_name is not None:
                raise ShaderCompilerError(
                    f"push constant buffer '{name}': a second push-constant buffer was found "
                    f"(first: '{push_constant_name}'); only one push-constant buffer is supported"
                )
            push_constant_name = name
            push_constant_size, push_constant_fields = _push_constant_layout(param)
            continue

        # Everything except descriptor-table slots is skipped, matching the RHI
        # reflection contract (only push constants and space-0 descriptors bind).
        if category != "descriptorTableSlot":
            continue

        # Descriptor arrays (sized or unsized) are rejected: the Descriptor API
        # only has single-resource bind operations. Checked before kind
        # classification so an array of an unsupported type reports this
        # clearer error.
        if param.get("type", {}).get("kind") == "array":
            raise ShaderCompilerError(
                f"array resource bindings are not supported yet (binding '{name}')"
            )

        kind = _binding_kind(param.get("type", {}), name)

        if binding.get("space", 0) != 0:
            raise ShaderCompilerError(
                f"binding '{name}': binding space is not 0; only space 0 is supported"
            )

        bindings.append(Binding(name=name, index=binding.get("index", 0), kind=kind))
    return bindings, push_constant_size, push_constant_fields


def reflect_interface(
    reflection: dict, pair_name: str, vertex_entry: str, fragment_entry: str
) -> ReflectedInterface:
    """Build the pair's interface from a slangc -reflection-json document."""
    attributes = _reflect_vertex_attributes(reflection, vertex_entry)
    bindings, push_constant_size, push_constant_fields = _reflect_global_parameters(reflection)
    return ReflectedInterface(
        pair_name=pair_name,
        vertex_entry_point=vertex_entry,
        fragment_entry_point=fragment_entry,
        attributes=attributes,
        bindings=bindings,
        push_constant_size=push_constant_size,
        push_constant_fields=push_constant_fields,
    )


def _format_byte_array(data: bytes, indent: str) -> str:
    lines = []
    for offset in range(0, len(data), _BYTES_PER_LINE):
        chunk = data[offset : offset + _BYTES_PER_LINE]
        lines.append(indent + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
    return "\n".join(lines)


def _emit_push_constant_asserts(
    interface: ReflectedInterface, push_constants: PushConstantsSpec, indent: str
) -> list[str]:
    """static_asserts pinning the shared C++ struct to the reflected block layout.

    The field names come from the shader's struct, which is the same definition
    the C++ type was compiled from, so `offsetof` naming a field that no longer
    exists, or an offset that moved, fails right here.
    """
    cpp_type = push_constants.cpp_type
    out = [
        f"{indent}// {cpp_type} is the struct this block was declared with (shared header "
        f"{push_constants.include}); its layout must be what the shader reflected.",
        f"{indent}static_assert(sizeof({cpp_type}) == kPushConstantSize,",
        f'{indent}              "{cpp_type} size differs from the shader push-constant block");',
    ]
    for name, offset in interface.push_constant_fields:
        out.append(f"{indent}static_assert(offsetof({cpp_type}, {name}) == {offset},")
        out.append(
            f'{indent}              "{cpp_type}::{name} offset differs from the shader '
            f'push-constant block");'
        )
    return out


def _emit_pair_members(
    pair: CompiledPair, indent: str, push_constants: PushConstantsSpec | None
) -> str:
    interface = pair.interface
    out = []
    out.append(
        f"{indent}inline constexpr uint32_t kPushConstantSize = {interface.push_constant_size};"
    )
    if push_constants is not None:
        out.extend(_emit_push_constant_asserts(interface, push_constants, indent))
    out.append("")
    out.append(f"{indent}namespace detail {{")
    out.append(f"{indent}alignas(4) inline constexpr unsigned char kVertexSpirv[] = {{")
    out.append(_format_byte_array(pair.vertex_spirv, indent + "    "))
    out.append(f"{indent}}};")
    out.append(f"{indent}alignas(4) inline constexpr unsigned char kFragmentSpirv[] = {{")
    out.append(_format_byte_array(pair.fragment_spirv, indent + "    "))
    out.append(f"{indent}}};")
    out.append(f"{indent}}}  // namespace detail")
    out.append("")
    out.append(f"{indent}[[nodiscard]] inline ::holoviz::ShaderBlob make_shader_blob() {{")
    body = indent + "  "
    out.append(f"{body}::holoviz::ShaderBlob blob;")
    out.append(f"{body}blob.ir = ::holoviz::ShaderIR::kSpirv;")
    for member, array in (("vertex_bytes", "kVertexSpirv"), ("fragment_bytes", "kFragmentSpirv")):
        out.append(
            f"{body}blob.{member}.assign(reinterpret_cast<const std::byte*>(detail::{array}),"
        )
        out.append(
            f"{body}                  "
            f"reinterpret_cast<const std::byte*>(detail::{array}) + sizeof(detail::{array}));"
        )
    out.append(f'{body}blob.vertex_entry_point = "{interface.vertex_entry_point}";')
    out.append(f'{body}blob.fragment_entry_point = "{interface.fragment_entry_point}";')
    if interface.attributes:
        out.append(f"{body}blob.attributes = {{")
        out.extend(
            f'{body}    {{"{attr.name}", {attr.location}, '
            f"::holoviz::Format::{attr.format}, {attr.byte_size}}},"
            for attr in interface.attributes
        )
        out.append(f"{body}}};")
    if interface.bindings:
        out.append(f"{body}blob.bindings = {{")
        out.extend(
            f'{body}    {{"{binding.name}", {binding.index}, '
            f"::holoviz::BindingKind::{binding.kind}, {binding.count}}},"
            for binding in interface.bindings
        )
        out.append(f"{body}}};")
    out.append(f"{body}blob.push_constant_size = kPushConstantSize;")
    out.append(f"{body}return blob;")
    out.append(f"{indent}}}")
    return "\n".join(out)


def emit_header(
    pairs: list[CompiledPair],
    namespace: str,
    input_name: str,
    push_constants: PushConstantsSpec | None = None,
) -> str:
    """Emit the generated C++ header for all compiled pairs of one input file."""
    if push_constants is not None:
        without_block = [p.interface.pair_name for p in pairs if not p.interface.push_constant_size]
        if without_block:
            raise ShaderCompilerError(
                f"--push-constants-type {push_constants.cpp_type} was given, but pair(s) "
                f"{without_block} declare no push-constant buffer"
            )
    out = []
    out.append(
        "// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
        "All rights reserved."
    )
    out.append("// SPDX-License-Identifier: Apache-2.0")
    out.append("//")
    out.append(f"// Generated by holoviz_shader_compiler.py from {input_name} — do not edit.")
    out.append("")
    out.append("#pragma once")
    out.append("")
    out.append("#include <cstddef>")
    out.append("#include <cstdint>")
    out.append("")
    out.append("#include <holoviz/pipeline.hpp>")
    if push_constants is not None:
        out.append("")
        out.append(f"#include {_include_directive_operand(push_constants.include)}")
    out.append("")
    out.append(f"namespace {namespace} {{")
    out.append("")
    if len(pairs) == 1:
        out.append(_emit_pair_members(pairs[0], "", push_constants))
        out.append("")
    else:
        for pair in pairs:
            out.append(f"namespace {pair.interface.pair_name} {{")
            out.append("")
            out.append(_emit_pair_members(pair, "", push_constants))
            out.append("")
            out.append(f"}}  // namespace {pair.interface.pair_name}")
            out.append("")
    out.append(f"}}  // namespace {namespace}")
    out.append("")
    return "\n".join(out)


def _include_directive_operand(include: str) -> str:
    """Quote a header for #include unless it is already spelled <...> or "..."."""
    if (include.startswith("<") and include.endswith(">")) or (
        include.startswith('"') and include.endswith('"')
    ):
        return include
    return f'"{include}"'


def _parse_pair(spec: str) -> tuple[str, str, str]:
    name, sep, entries = spec.partition("=")
    vs, sep2, fs = entries.partition(":")
    if not sep or not sep2 or not name or not vs or not fs:
        raise ShaderCompilerError(
            f"invalid --pair '{spec}'; expected NAME=VERTEX_ENTRY:FRAGMENT_ENTRY"
        )
    return name, vs, fs


def _resolve_slangc(explicit: str | None) -> str:
    candidate = explicit or os.environ.get("SLANGC") or "slangc"
    # A bare program name (no directory component) is looked up on PATH, so
    # `--slangc slangc` and `SLANGC=slangc` behave like the default search.
    if os.path.dirname(candidate) == "":
        resolved = shutil.which(candidate)
        if not resolved:
            raise ShaderCompilerError(
                f"no slangc found: '{candidate}' is not on PATH; pass --slangc, set $SLANGC, "
                "or add slangc to PATH"
            )
        candidate = resolved
    if not (os.path.isfile(candidate) and os.access(candidate, os.X_OK)):
        raise ShaderCompilerError(f"slangc is not an executable file: {candidate}")
    return candidate


def _default_namespace(input_path: str) -> str:
    return f"holoviz::{Path(input_path).stem}"


def compile_shader(
    slangc: str,
    input_path: str,
    pair_specs: list[tuple[str, str, str]],
    include_dirs: list[str],
    sources: set[str] | None = None,
) -> list[CompiledPair]:
    """Reflect and compile every pair; ``sources`` (if given) collects all files read."""
    pairs: list[CompiledPair] = []
    with tempfile.TemporaryDirectory(prefix="holoviz_shader_compiler.") as work_dir:
        for name, vertex_entry, fragment_entry in pair_specs:
            reflection = reflect_pair_json(
                slangc, input_path, vertex_entry, fragment_entry, work_dir, include_dirs
            )
            interface = reflect_interface(reflection, name, vertex_entry, fragment_entry)
            pairs.append(
                CompiledPair(
                    interface=interface,
                    vertex_spirv=compile_stage_spirv(
                        slangc, input_path, vertex_entry, "vertex", work_dir, include_dirs, sources
                    ),
                    fragment_spirv=compile_stage_spirv(
                        slangc,
                        input_path,
                        fragment_entry,
                        "fragment",
                        work_dir,
                        include_dirs,
                        sources,
                    ),
                )
            )
    return pairs


def _write_atomically(path: Path, text: str) -> None:
    """Write via a sibling temp file so an interrupted run never leaves a truncated
    output that a later build would consume as up to date."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _format_depfile(target: str, sources: set[str]) -> str:
    """One make rule naming every file slangc read, for add_custom_command(DEPFILE)."""

    def escape(path: str) -> str:
        return path.replace(" ", "\\ ")

    prerequisites = " ".join(escape(p) for p in sorted(sources))
    return f"{escape(target)}: {prerequisites}\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile a Slang shader and emit a C++ header with SPIR-V + reflection."
    )
    parser.add_argument(
        "--slangc",
        help="slangc executable: a path, or a program name looked up on PATH "
        "(default: $SLANGC, then 'slangc' on PATH)",
    )
    parser.add_argument("--input", required=True, help="input .slang source file")
    parser.add_argument(
        "--emit-header", required=True, help="output path of the generated C++ header"
    )
    parser.add_argument(
        "--namespace",
        help="C++ namespace for the generated members (default: holoviz::<input file stem>)",
    )
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        metavar="NAME=VS_ENTRY:FS_ENTRY",
        help="entry-point pair to compile; repeat for multiple pairs",
    )
    parser.add_argument(
        "--include-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="directory slangc searches for #include/import; repeat for several",
    )
    parser.add_argument(
        "--push-constants-type",
        metavar="CPP_TYPE",
        help="fully qualified C++ struct the shader's push-constant block is declared with "
        "(from a shared interface header); the generated header static_asserts its size and "
        "field offsets against the reflected layout",
    )
    parser.add_argument(
        "--push-constants-include",
        metavar="HEADER",
        help="the shared interface header, spelled as the consumer includes it "
        "(required with --push-constants-type)",
    )
    parser.add_argument(
        "--emit-depfile",
        metavar="PATH",
        help="write a make-style depfile listing every file slangc read, with the "
        "generated header as the target",
    )
    args = parser.parse_args(argv)

    if bool(args.push_constants_type) != bool(args.push_constants_include):
        parser.error("--push-constants-type and --push-constants-include go together")

    try:
        slangc = _resolve_slangc(args.slangc)
        pair_specs = [_parse_pair(spec) for spec in args.pair]
        names = [name for name, _, _ in pair_specs]
        if len(set(names)) != len(names):
            raise ShaderCompilerError(f"duplicate --pair names: {names}")

        sources: set[str] | None = set() if args.emit_depfile else None
        pairs = compile_shader(slangc, args.input, pair_specs, args.include_dir, sources)
        namespace = args.namespace or _default_namespace(args.input)
        push_constants = (
            PushConstantsSpec(args.push_constants_type, args.push_constants_include)
            if args.push_constants_type
            else None
        )
        header = emit_header(pairs, namespace, Path(args.input).name, push_constants)

        _write_atomically(Path(args.emit_header), header)
        if sources is not None:
            _write_atomically(Path(args.emit_depfile), _format_depfile(args.emit_header, sources))
    except ShaderCompilerError as e:
        print(f"holoviz_shader_compiler: error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
