#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate Holoscan FlatBuffers bindings from one reflected schema model."""

from __future__ import annotations

import argparse
import hashlib
import json
import keyword
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

SCALARS = {
    "Bool",
    "Byte",
    "UByte",
    "Short",
    "UShort",
    "Int",
    "UInt",
    "Long",
    "ULong",
    "Float",
    "Double",
    "UType",
}
INTEGERS = SCALARS - {"Float", "Double"}
FINITE_FLOAT = re.compile(r"^-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")
REFLECTED_NONFINITE_DEFAULT = re.compile(r'("default_real"\s*:\s*)([+-]?(?:inf|nan))(?=\s*[,}])')
VERSION = re.compile(r"([0-9]+)\.([0-9]+)\.([0-9]+)")


class GenerationError(RuntimeError):
    """A deterministic schema-generation failure."""


@dataclass(frozen=True)
class Args:
    flatc: Path
    expected_flatc_version: str
    reflection_schema: Path
    bfbs: Path
    output: Path | None
    root_generated_header: str | None
    bfbs_embed_header: str | None
    compatibility_epoch: int | None
    max_serialized_size: int | None
    descriptor_tensor_custom_codec: bool
    provenance_output: Path | None
    root_source: Path
    include_dirs: tuple[Path, ...]
    flatc_flags: tuple[str, ...]
    python_output: Path | None
    python_stub_output: Path | None
    python_registration_output: Path | None
    python_module_name: str | None
    traits_include: str | None
    python_values_only: bool
    python_external_types: tuple[tuple[str, str], ...]
    python_constants: tuple[tuple[str, int], ...]

    @property
    def root_logical_path(self) -> str:
        return logical_path(self.root_source, self.include_dirs, self.root_source.parent)


@dataclass(frozen=True)
class DependencyArgs:
    flatc: Path
    output: Path
    target: Path
    schemas: tuple[Path, ...]
    include_dirs: tuple[Path, ...]
    flatc_flags: tuple[str, ...]


@dataclass(frozen=True)
class ReflectedType:
    base: str
    element: str
    index: int
    fixed_length: int
    base_size: int
    element_size: int


@dataclass(frozen=True)
class Field:
    name: str
    field_id: int | None
    deprecated: bool
    required: bool
    optional: bool
    default_integer: int | str
    default_real: float | int | str
    type: ReflectedType
    native_inline: bool
    attributes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Object:
    name: str
    is_struct: bool
    minimum_alignment: int
    fields: tuple[Field, ...]
    declaration_file: str
    attributes: tuple[str, ...] = ()


@dataclass(frozen=True)
class EnumValue:
    name: str
    value: int
    union_type: ReflectedType | None


@dataclass(frozen=True)
class Enum:
    name: str
    values: tuple[EnumValue, ...]
    underlying_type: ReflectedType
    bit_flags: bool
    declaration_file: str


@dataclass(frozen=True)
class Model:
    root_name: str | None
    file_identifier: str
    root_index: int | None
    objects: tuple[Object, ...]
    enums: tuple[Enum, ...]

    @property
    def root(self) -> Object:
        if self.root_index is None:
            raise GenerationError("value-only schema has no payload root")
        return self.objects[self.root_index]


@dataclass
class Fragments:
    prepare: str = ""
    reset: str = ""
    validate: str = ""
    assign: str = ""


@dataclass(frozen=True)
class DescriptorTensorInventory:
    names: tuple[str, ...]
    field_ids: tuple[int, ...]
    assertions: str


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def arguments() -> Args | DependencyArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate a traits header")
    generate.add_argument("--flatc", type=Path, required=True)
    generate.add_argument("--expected-flatc-version", required=True)
    generate.add_argument("--reflection-schema", type=Path, required=True)
    generate.add_argument("--bfbs", type=Path, required=True)
    generate.add_argument("--output", type=Path)
    generate.add_argument("--root-generated-header")
    generate.add_argument("--bfbs-embed-header")
    generate.add_argument("--compatibility-epoch", type=positive_integer)
    generate.add_argument("--max-serialized-size", type=positive_integer)
    generate.add_argument("--descriptor-tensor-custom-codec", action="store_true")
    generate.add_argument("--provenance-output", type=Path)
    generate.add_argument("--root-source", type=Path, required=True)
    generate.add_argument("--include-dir", action="append", type=Path, default=[])
    generate.add_argument("--flatc-flag", action="append", default=[])
    generate.add_argument("--python-output", type=Path)
    generate.add_argument("--python-stub-output", type=Path)
    generate.add_argument("--python-registration-output", type=Path)
    generate.add_argument("--python-module-name")
    generate.add_argument("--traits-include")
    generate.add_argument("--python-values-only", action="store_true")
    generate.add_argument("--python-external-type", action="append", default=[])
    generate.add_argument("--python-constant", action="append", default=[])

    dependencies = subparsers.add_parser(
        "dependencies", help="write a depfile from flatc's parsed include graph"
    )
    dependencies.add_argument("--flatc", type=Path, required=True)
    dependencies.add_argument("--output", type=Path, required=True)
    dependencies.add_argument("--target", type=Path, required=True)
    dependencies.add_argument("--schema", action="append", type=Path, required=True)
    dependencies.add_argument("--include-dir", action="append", type=Path, default=[])
    dependencies.add_argument("--flatc-flag", action="append", required=True)

    values = parser.parse_args()
    if values.command == "dependencies":
        return DependencyArgs(
            flatc=values.flatc,
            output=values.output,
            target=values.target,
            schemas=tuple(values.schema),
            include_dirs=tuple(values.include_dir),
            flatc_flags=tuple(values.flatc_flag),
        )
    if values.python_values_only:
        if values.python_output is None or values.python_stub_output is None:
            parser.error("--python-values-only requires --python-output and --python-stub-output")
        if any(
            item is not None
            for item in (
                values.output,
                values.root_generated_header,
                values.compatibility_epoch,
                values.max_serialized_size,
                values.python_registration_output,
                values.python_module_name,
                values.traits_include,
            )
        ):
            parser.error("--python-values-only cannot generate payload traits or registration")
    else:
        required_generation_arguments = (
            values.output,
            values.root_generated_header,
            values.compatibility_epoch,
            values.max_serialized_size,
        )
        if not all(item is not None for item in required_generation_arguments):
            parser.error(
                "--output, --root-generated-header, --compatibility-epoch, and "
                "--max-serialized-size are required for payload generation"
            )
    python_arguments = (
        values.python_output,
        values.python_stub_output,
        values.python_registration_output,
        values.python_module_name,
        values.traits_include,
    )
    if (
        not values.python_values_only
        and any(item is not None for item in python_arguments)
        and not all(item is not None for item in python_arguments)
    ):
        parser.error(
            "--python-output, --python-stub-output, --python-registration-output, "
            "--python-module-name, and --traits-include must be supplied together"
        )
    external_types: list[tuple[str, str]] = []
    for specification in values.python_external_type:
        name, separator, module = specification.partition("=")
        if (
            not separator
            or not name
            or not module
            or re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", name) is None
            or re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module) is None
        ):
            parser.error("--python-external-type must use fully.qualified.Type=python.module")
        external_types.append((name, module))
    if len({name for name, _ in external_types}) != len(external_types):
        parser.error("--python-external-type names must be unique")
    python_constants: list[tuple[str, int]] = []
    for specification in values.python_constant:
        name, separator, literal = specification.partition("=")
        if (
            not separator
            or re.fullmatch(r"[A-Z][A-Z0-9_]*", name) is None
            or re.fullmatch(r"-?(?:0|[1-9][0-9]*)", literal) is None
        ):
            parser.error("--python-constant must use PUBLIC_NAME=INTEGER")
        python_constants.append((name, int(literal)))
    if len({name for name, _ in python_constants}) != len(python_constants):
        parser.error("--python-constant names must be unique")
    if python_constants and not (values.python_values_only or values.python_output is not None):
        parser.error("--python-constant requires Python value generation")
    return Args(
        flatc=values.flatc,
        expected_flatc_version=values.expected_flatc_version,
        reflection_schema=values.reflection_schema,
        bfbs=values.bfbs,
        output=values.output,
        root_generated_header=values.root_generated_header,
        bfbs_embed_header=values.bfbs_embed_header,
        compatibility_epoch=values.compatibility_epoch,
        max_serialized_size=values.max_serialized_size,
        descriptor_tensor_custom_codec=values.descriptor_tensor_custom_codec,
        provenance_output=values.provenance_output,
        root_source=values.root_source,
        include_dirs=tuple(values.include_dir),
        flatc_flags=tuple(values.flatc_flag),
        python_output=values.python_output,
        python_stub_output=values.python_stub_output,
        python_registration_output=values.python_registration_output,
        python_module_name=values.python_module_name,
        traits_include=values.traits_include,
        python_values_only=values.python_values_only,
        python_external_types=tuple(external_types),
        python_constants=tuple(python_constants),
    )


def flatc_dependencies(
    flatc: Path,
    schemas: tuple[Path, ...],
    include_dirs: tuple[Path, ...],
    flatc_flags: tuple[str, ...],
) -> list[Path]:
    command = [str(flatc), "-M", *flatc_flags]
    for include_dir in include_dirs:
        command.extend(("-I", str(include_dir.resolve())))
    command.extend(str(schema.resolve()) for schema in schemas)
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        raise GenerationError(
            f"Could not resolve FlatBuffers dependencies: {result.stderr.strip()}"
        )

    dependencies: set[Path] = set()
    make_rules = result.stdout.replace("\\\n", " ")
    for rule in make_rules.splitlines():
        if not rule.strip():
            continue
        separator = rule.find(":")
        if separator < 0:
            raise GenerationError(f"flatc emitted an invalid dependency rule: {rule}")
        for dependency in shlex.split(rule[separator + 1 :]):
            dependencies.add(Path(dependency).resolve())
    if not dependencies:
        raise GenerationError("flatc emitted an empty dependency graph")
    return sorted(dependencies, key=lambda path: path.as_posix())


def make_escape(path: Path) -> str:
    return path.resolve().as_posix().replace("$", "$$").replace("#", "\\#").replace(" ", "\\ ")


def write_dependencies(args: DependencyArgs) -> None:
    dependencies = flatc_dependencies(args.flatc, args.schemas, args.include_dirs, args.flatc_flags)
    contents = f"{make_escape(args.target)}:"
    contents += "".join(f" \\\n  {make_escape(path)}" for path in dependencies)
    publish(args.output, f"{contents}\n")


def logical_path(path: Path, include_dirs: tuple[Path, ...], fallback: Path) -> str:
    resolved = path.resolve()
    for root in (*include_dirs, fallback):
        resolved_root = root.resolve()
        if resolved.is_relative_to(resolved_root):
            return resolved.relative_to(resolved_root).as_posix()
    raise GenerationError(
        f"Schema dependency is outside every include root and the root source tree: {path}"
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_provenance(args: Args, flatc_build: str) -> None:
    if args.provenance_output is None:
        return
    dependencies = flatc_dependencies(
        args.flatc, (args.root_source,), args.include_dirs, args.flatc_flags
    )
    root = args.root_source.resolve()
    includes = [
        {
            "path": logical_path(path, args.include_dirs, root.parent),
            "sha256": sha256(path),
        }
        for path in dependencies
        if path != root
    ]
    document = {
        "format": "holoscan.flatbuffer-generation-provenance",
        "version": 1,
        "flatc_build": flatc_build,
        "flatc_flags": list(args.flatc_flags),
        "bfbs_sha256": sha256(args.bfbs),
        "root": {"path": args.root_logical_path, "sha256": sha256(root)},
        "includes": includes,
    }
    publish(args.provenance_output, f"{json.dumps(document, indent=2)}\n")


def reflect(args: Args) -> tuple[dict[str, Any], str]:
    for path, label in (
        (args.flatc, "flatc"),
        (args.reflection_schema, "FlatBuffers reflection schema"),
        (args.bfbs, "generated binary schema"),
    ):
        if not path.is_file():
            raise GenerationError(f"{label} does not exist: {path}")
    result = subprocess.run(
        [str(args.flatc), "--version"], check=False, capture_output=True, text=True
    )
    match = VERSION.search(result.stdout)
    if result.returncode or match is None:
        raise GenerationError(
            f"Could not query flatc version: {(result.stderr or result.stdout).strip()}"
        )
    actual = ".".join(match.groups())
    flatc_build = result.stdout.strip()
    if actual != args.expected_flatc_version:
        raise GenerationError(
            f"flatc version {actual} does not match configured version "
            f"{args.expected_flatc_version}"
        )
    with tempfile.TemporaryDirectory(prefix="holoscan-schema-reflection-") as temporary:
        output = Path(temporary)
        result = subprocess.run(
            [
                str(args.flatc),
                "--json",
                "--strict-json",
                "--defaults-json",
                "-o",
                str(output),
                str(args.reflection_schema),
                "--",
                str(args.bfbs),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        reflected = output / f"{args.bfbs.stem}.json"
        if result.returncode or not reflected.is_file():
            raise GenerationError(
                f"Could not reflect generated binary schema: {result.stderr.strip()}"
            )
        reflected_json = reflected.read_text(encoding="utf-8")
        # flatc emits lowercase non-finite literals in reflection JSON. Preserve them as strings
        # so schema validation can report the field-level policy error deterministically.
        reflected_json = REFLECTED_NONFINITE_DEFAULT.sub(r'\1"\2"', reflected_json)
        return json.loads(reflected_json), flatc_build


def reflected_type_from(data: dict[str, Any]) -> ReflectedType:
    return ReflectedType(
        base=data["base_type"],
        element=data["element"],
        index=int(data.get("index", -1)),
        fixed_length=int(data.get("fixed_length", 0)),
        base_size=int(data.get("base_size", 0)),
        element_size=int(data.get("element_size", 0)),
    )


def field_from(data: dict[str, Any]) -> Field:
    field_id = data.get("id")
    default_integer = data["default_integer"]
    default_real = data["default_real"]
    if not isinstance(default_integer, (int, str)):
        raise GenerationError("reflected integer default must be an integer or string")
    if not isinstance(default_real, (float, int, str)):
        raise GenerationError("reflected real default must be a number or string")
    attributes = {
        attribute["key"]: attribute.get("value", "") for attribute in (data.get("attributes") or [])
    }
    return Field(
        name=data["name"],
        field_id=field_id if isinstance(field_id, int) else None,
        deprecated=data["deprecated"],
        required=data["required"],
        optional=bool(data.get("optional", False)),
        default_integer=default_integer,
        default_real=default_real,
        type=reflected_type_from(data["type"]),
        native_inline="native_inline" in attributes,
        attributes=tuple(item["key"] for item in data.get("attributes", [])),
    )


def object_from(data: dict[str, Any]) -> Object:
    return Object(
        name=data["name"],
        is_struct=data["is_struct"],
        minimum_alignment=int(data["minalign"]),
        fields=tuple(field_from(field) for field in data["fields"]),
        declaration_file=str(data.get("declaration_file", "")),
        attributes=tuple(item["key"] for item in data.get("attributes", [])),
    )


def enum_from(data: dict[str, Any]) -> Enum:
    attributes = {
        attribute["key"]: attribute.get("value", "") for attribute in (data.get("attributes") or [])
    }
    return Enum(
        name=data["name"],
        values=tuple(
            EnumValue(
                name=value["name"],
                value=int(value["value"]),
                union_type=(
                    reflected_type_from(value["union_type"])
                    if value.get("union_type") is not None
                    else None
                ),
            )
            for value in data["values"]
        ),
        underlying_type=reflected_type_from(data["underlying_type"]),
        bit_flags="bit_flags" in attributes,
        declaration_file=str(data.get("declaration_file", "")),
    )


def model_from(data: dict[str, Any], path: str, *, allow_rootless: bool = False) -> Model:
    try:
        objects = tuple(object_from(object_) for object_ in data["objects"])
        enums = tuple(enum_from(enum) for enum in data["enums"])
        root_table = data.get("root_table")
        root_name = root_table["name"] if root_table is not None else None
        identifier = data.get("file_ident", "")
    except (KeyError, TypeError, ValueError) as error:
        raise GenerationError("Generated binary schema reflection is incomplete") from error
    if not objects:
        raise GenerationError("Generated binary schema reflection is incomplete")
    if root_name is None:
        if not allow_rootless:
            raise GenerationError(f"{path}: generated schema must declare a root_type")
        return Model(None, "", None, objects, enums)
    if len(identifier) != 4:
        raise GenerationError(
            f"{path}: generated schema root must declare an exact four-byte "
            "FlatBuffers file_identifier"
        )
    index = next((i for i, item in enumerate(objects) if item.name == root_name), None)
    if index is None:
        raise GenerationError(
            f"Generated binary schema selected traits root '{root_name}' "
            "is absent from its object inventory"
        )
    return Model(root_name, identifier, index, objects, enums)


def cpp_type(name: str) -> str:
    return name.replace(".", "::")


def package_namespace(name: str) -> str:
    namespace, separator, short = name.rpartition(".")
    return (
        f"{cpp_type(namespace)}::holoscan_generated_{short}_package"
        if separator
        else f"holoscan_generated_{short}_package"
    )


def cpp_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)[1:-1]


def check_object(model: Model, index: int, message: str) -> None:
    if index < 0 or index >= len(model.objects):
        raise GenerationError(message)


def object_references(model: Model, type_: ReflectedType) -> list[int]:
    base, element, index = type_.base, type_.element, type_.index
    if base == "Obj" or (base in {"Vector", "Array", "Vector64"} and element == "Obj"):
        check_object(model, index, "Generated binary schema has an invalid object reference")
        return [index]
    if base == "Union" or (base in {"Vector", "Array", "Vector64"} and element == "Union"):
        if index < 0 or index >= len(model.enums):
            raise GenerationError("Generated binary schema has an invalid union reference")
        references = []
        for value in model.enums[index].values:
            union_type = value.union_type
            if union_type is not None and union_type.base == "Obj":
                object_index = union_type.index
                check_object(
                    model,
                    object_index,
                    "Generated binary schema has an invalid union object reference",
                )
                references.append(object_index)
        return references
    return []


def reject_tensor(model: Model) -> None:
    pending, visited = [model.root_index], set()
    while pending:
        index = pending.pop(0)
        if index in visited:
            continue
        visited.add(index)
        object_ = model.objects[index]
        if object_.name == "holoscan.schema.Tensor":
            raise GenerationError(
                "HOLOSCAN_SCHEMA_TENSOR_REQUIRES_DIRECT_DESCRIPTOR_FIELD: generated descriptor "
                f"traits require Tensor to be a direct field of root '{model.root_name}'"
            )
        for field in object_.fields:
            pending.extend(object_references(model, field.type))


ORDINARY = Template(r"""/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Generated by holoscan_add_flatbuffer_schema(). Do not edit.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

$bfbs_api_include#include <holoscan/schema/flatbuffer_schema_package.hpp>

#include "$header"$bfbs_embed_include

namespace holoscan::schema {

template <>
struct FlatBufferSchemaBinding<$root> {
  using root_type = $root;
  using native_type = $native;
$bfbs_binding  static constexpr std::uint32_t compatibility_epoch = ${epoch}U;
  static constexpr FlatBufferFileIdentifier file_identifier =
      make_flatbuffer_file_identifier("$identifier");
  static constexpr SchemaFamilyIdentity family_identity =
      make_schema_family_identity<$root>(compatibility_epoch);
  static constexpr SchemaIdentity identity =
      make_schema_identity(family_identity, file_identifier);
  static constexpr std::size_t max_serialized_size = ${maximum}U;
  static constexpr bool descriptor_tensor = false;
};

}  // namespace holoscan::schema
""")


def substitutions(args: Args, model: Model) -> dict[str, str | int]:
    if (
        model.root_name is None
        or args.root_generated_header is None
        or args.compatibility_epoch is None
        or args.max_serialized_size is None
    ):
        raise GenerationError("payload traits generation requires a complete rooted schema")
    root = cpp_type(model.root_name)
    # flatc's --bfbs-gen-embed names the embed struct after the root type, in the schema's own
    # namespace, so the binding reaches the reflection bytes without a caller-supplied type name.
    embed = args.bfbs_embed_header is not None
    return {
        "header": cpp_string(args.root_generated_header),
        "package": package_namespace(model.root_name),
        "maximum": args.max_serialized_size,
        "root": root,
        "native": f"{root}T",
        "name": cpp_string(model.root_name),
        "epoch": args.compatibility_epoch,
        "identifier": cpp_string(model.file_identifier),
        "bfbs_api_include": "#include <holoscan/schema/binary_schema.hpp>\n" if embed else "",
        "bfbs_embed_include": (
            f'\n#include "{cpp_string(args.bfbs_embed_header)}"' if embed else ""
        ),
        "bfbs_binding": f"  using binary_schema_type = {root}BinarySchema;\n" if embed else "",
    }


def ordered_fields(args: Args, model: Model) -> list[Field]:
    if model.root.is_struct:
        raise GenerationError(
            f"{args.root_logical_path}: descriptor root '{model.root_name}' must be a table"
        )
    fields = model.root.fields
    if not fields:
        raise GenerationError(
            f"{args.root_logical_path}: descriptor root '{model.root_name}' has no fields"
        )
    valid_ids, seen = set(range(len(fields))), set()
    for field in fields:
        field_id = field.field_id
        if field_id is None or field_id not in valid_ids:
            raise GenerationError(
                f"{args.root_logical_path}:{field.name}: FlatBuffers field id '{field_id}' "
                f"is missing or outside the canonical 0..{len(fields) - 1} sequence"
            )
        if field_id in seen:
            raise GenerationError(
                f"{args.root_logical_path}:{field.name}: duplicate FlatBuffers field id {field_id}"
            )
        seen.add(field_id)
    if missing := valid_ids - seen:
        raise GenerationError(
            f"{args.root_logical_path}: missing FlatBuffers field id {min(missing)}"
        )

    def canonical_field_id(field: Field) -> int:
        assert field.field_id is not None
        return field.field_id

    return sorted(fields, key=canonical_field_id)


def check_alignment(model: Model, index: int, path: str) -> None:
    object_ = model.objects[index]
    if object_.is_struct and object_.minimum_alignment > 8:
        raise GenerationError(
            f"{path}: FlatBuffers struct alignment {object_.minimum_alignment} exceeds "
            "the descriptor metadata envelope alignment of 8"
        )


def validate_field(field: Field, path: str, *, nested: bool = False) -> None:
    base = field.type.base
    if field.deprecated:
        noun = "nested fields" if nested else "object-API fields"
        raise GenerationError(
            f"{path}: deprecated {noun} are unsupported by descriptor traits generation"
        )
    if base in {"Union", "UType"}:
        noun = "nested FlatBuffers unions" if nested else "FlatBuffers unions"
        raise GenerationError(f"{path}: {noun} are unsupported by descriptor traits generation")
    if field.required and base in {"String", "Vector", "Obj"}:
        raise GenerationError(
            f"{path}: required FlatBuffers reference fields are unsupported "
            "by descriptor traits generation"
        )


def scalar_default(field: Field, path: str) -> str:
    base = field.type.base
    if base in INTEGERS:
        return str(field.default_integer).lower()
    value = str(field.default_real)
    if not FINITE_FLOAT.fullmatch(value):
        raise GenerationError(
            f"{path}: non-finite floating-point defaults are unsupported "
            "by descriptor traits generation"
        )
    return value


def merge_fragments(target: Fragments, source: Fragments) -> None:
    target.prepare += source.prepare
    target.reset += source.reset
    target.validate += source.validate
    target.assign += source.assign


def value_field_codec(
    field: Field,
    native: str,
    accessor: str,
    path: str,
    *,
    indent: int,
    local: str,
    nested: bool,
) -> Fragments:
    """Emit the shared scalar, string, and vector native-value codec fragments."""
    prefix = " " * indent
    base, element = field.type.base, field.type.element
    result = Fragments()
    if base in SCALARS:
        default = scalar_default(field, path)
        result.reset = (
            f"{prefix}{native} =\n"
            f"{prefix}    static_cast<std::remove_cvref_t<decltype({native})>>({default});\n"
        )
        result.assign = f"{prefix}{native} = {accessor}();\n"
        return result
    if base == "String":
        result.reset = f"{prefix}{native}.clear();\n"
        result.validate = (
            f"{prefix}if (const auto* {local} = {accessor}();\n"
            f"{prefix}    {local} != nullptr && {local}->size() > kMetadataCapacity) {{\n"
            f"{prefix}  return holoscan::make_unexpected(holoscan::Error{{\n"
            f"{prefix}      holoscan::ErrorCode::kProtocolError,\n"
            f'{prefix}      "{cpp_string(path)}: string exceeds descriptor metadata capacity"}});\n'
            f"{prefix}}}\n"
        )
        result.assign = (
            f"{prefix}if (const auto* {local} = {accessor}(); {local} != nullptr) {{\n"
            f"{prefix}  {native}.assign({local}->c_str(), {local}->size());\n"
            f"{prefix}}} else {{\n"
            f"{prefix}  {native}.clear();\n"
            f"{prefix}}}\n"
        )
        return result
    if base != "Vector":
        raise AssertionError(f"{path}: value_field_codec received unsupported base type '{base}'")
    if element == "Obj":
        noun = "nested vectors" if nested else "vectors"
        raise GenerationError(
            f"{path}: {noun} of tables or structs are unsupported "
            "by descriptor traits generation"
            + ("" if nested else "; provide a hand-written standard codec")
        )
    if element in {"Union", "UType"}:
        noun = "nested vector element type" if nested else "vectors of unions"
        detail = f" '{element}'" if nested else ""
        raise GenerationError(
            f"{path}: {noun}{detail} are unsupported by descriptor traits generation"
        )
    if element != "String" and element not in SCALARS:
        noun = "nested vector element type" if nested else "vector element type"
        raise GenerationError(
            f"{path}: {noun} '{element}' is unsupported by descriptor traits generation"
        )
    result.reset = f"{prefix}{native}.clear();\n"
    result.validate = (
        f"{prefix}if (const auto* {local} = {accessor}();\n"
        f"{prefix}    {local} != nullptr && {local}->size() > kMetadataCapacity) {{\n"
        f"{prefix}  return holoscan::make_unexpected(holoscan::Error{{\n"
        f"{prefix}      holoscan::ErrorCode::kProtocolError,\n"
        f'{prefix}      "{cpp_string(path)}: vector exceeds descriptor metadata capacity"}});\n'
        f"{prefix}}}\n"
    )
    if element == "String":
        result.validate += (
            f"{prefix}if (const auto* {local} = {accessor}(); {local} != nullptr) {{\n"
            f"{prefix}  for (flatbuffers::uoffset_t index = 0U; "
            f"index < {local}->size(); ++index) {{\n"
            f"{prefix}    const auto* element = {local}->Get(index);\n"
            f"{prefix}    if (element == nullptr || element->size() > kMetadataCapacity) {{\n"
            f"{prefix}      return holoscan::make_unexpected(holoscan::Error{{\n"
            f"{prefix}          holoscan::ErrorCode::kProtocolError,\n"
            f'{prefix}          "{cpp_string(path)}: string-vector element is invalid"}});\n'
            f"{prefix}    }}\n"
            f"{prefix}  }}\n"
            f"{prefix}}}\n"
        )
        result.assign = (
            f"{prefix}if (const auto* {local} = {accessor}(); {local} != nullptr) {{\n"
            f"{prefix}  {native}.resize({local}->size());\n"
            f"{prefix}  for (flatbuffers::uoffset_t index = 0U; "
            f"index < {local}->size(); ++index) {{\n"
            f"{prefix}    const auto* element = {local}->Get(index);\n"
            f"{prefix}    {native}[index].assign(element->c_str(), element->size());\n"
            f"{prefix}  }}\n"
            f"{prefix}}} else {{\n"
            f"{prefix}  {native}.clear();\n"
            f"{prefix}}}\n"
        )
    else:
        result.assign = (
            f"{prefix}if (const auto* {local} = {accessor}(); {local} != nullptr) {{\n"
            f"{prefix}  {native}.resize({local}->size());\n"
            f"{prefix}  for (flatbuffers::uoffset_t index = 0U; "
            f"index < {local}->size(); ++index) {{\n"
            f"{prefix}    {native}[index] = static_cast<typename "
            f"std::remove_cvref_t<decltype({native})>::value_type>({local}->Get(index));\n"
            f"{prefix}  }}\n"
            f"{prefix}}} else {{\n"
            f"{prefix}  {native}.clear();\n"
            f"{prefix}}}\n"
        )
    return result


def nested_codec(
    model: Model,
    object_index: int,
    native_prefix: str,
    root_prefix: str,
    path: str,
    ancestors: tuple[int, ...],
) -> Fragments:
    if object_index in ancestors:
        raise GenerationError(
            f"{path}: recursive tables are unsupported by descriptor traits generation; "
            "provide a hand-written standard codec"
        )
    ancestors = (*ancestors, object_index)
    result = Fragments()
    for field in model.objects[object_index].fields:
        name = field.name
        member = cpp_field_name(name)
        nested_path = f"{path}.{name}"
        native = f"{native_prefix}{member}"
        type_ = field.type
        base, index = type_.base, type_.index
        validate_field(field, nested_path, nested=True)
        if base in SCALARS or base in {"String", "Vector"}:
            merge_fragments(
                result,
                value_field_codec(
                    field,
                    native,
                    f"{root_prefix}{member}",
                    nested_path,
                    indent=6,
                    local="nested_field",
                    nested=True,
                ),
            )
        elif base == "Obj":
            check_object(
                model, index, f"{nested_path}: nested object type index '{index}' is invalid"
            )
            object_ = model.objects[index]
            if object_.name == "holoscan.schema.Tensor":
                raise GenerationError(
                    f"{nested_path}: Tensor fields must be direct members of the descriptor root "
                    "so their canonical field ids remain available to Runtime carriage"
                )
            check_alignment(model, index, nested_path)
            referenced = cpp_type(object_.name)
            native_type = referenced if object_.is_struct else f"{referenced}T"
            if object_.is_struct:
                result.reset += (
                    f"      if ({native}) {{\n        *{native} = {native_type}{{}};\n      }}\n"
                )
                result.assign += (
                    f"      if (const auto* nested_field = {root_prefix}{member}(); "
                    "nested_field != nullptr) {\n"
                    f"        if (!{native}) {{\n"
                    f"          {native} = std::make_shared<{native_type}>();\n"
                    "        }\n"
                    f"        *{native} = *nested_field;\n"
                    "      } else {\n"
                    f"        {native}.reset();\n"
                    "      }\n"
                )
            else:
                variable = f"nested_object_{index}"
                child = nested_codec(
                    model, index, f"{native}->", f"{variable}->", nested_path, ancestors
                )
                result.prepare += f"    if ({native}) {{\n{child.prepare}    }}\n"
                result.reset += f"      if ({native}) {{\n{child.reset}      }}\n"
                result.validate += (
                    f"      if (const auto* {variable} = {root_prefix}{member}(); "
                    f"{variable} != nullptr) {{\n{child.validate}      }}\n"
                )
                result.assign += (
                    f"      if (const auto* {variable} = {root_prefix}{member}(); "
                    f"{variable} != nullptr) {{\n"
                    f"        if (!{native}) {{\n"
                    f"          {native} = std::make_shared<{native_type}>();\n"
                    "        }\n"
                    f"{child.assign}"
                    "      } else {\n"
                    f"        {native}.reset();\n"
                    "      }\n"
                )
        else:
            raise GenerationError(
                f"{nested_path}: nested object-API base type '{base}' is unsupported "
                "by descriptor traits generation"
            )
    return result


def vector_expression(element: str, index: int, member: str, path: str) -> str:
    if element == "String":
        return f"builder.CreateVectorOfStrings(value.{member})"
    if index < 0:
        return f"builder.CreateVector(value.{member})"
    wire_types = {
        "Byte": "std::int8_t",
        "UByte": "std::uint8_t",
        "Bool": "std::uint8_t",
        "Short": "std::int16_t",
        "UShort": "std::uint16_t",
        "Int": "std::int32_t",
        "UInt": "std::uint32_t",
        "Long": "std::int64_t",
        "ULong": "std::uint64_t",
    }
    if element not in wire_types:
        raise GenerationError(f"{path}: enum vector storage '{element}' is unsupported")
    return (
        f"builder.CreateVectorScalarCast<{wire_types[element]}>("
        f"flatbuffers::data(value.{member}), value.{member}.size())"
    )


def descriptor_tensor_inventory(args: Args, model: Model, native: str) -> DescriptorTensorInventory:
    tensor_names: list[str] = []
    tensor_ids: list[int] = []
    assertions = ""
    for field in ordered_fields(args, model):
        if field.type.base != "Obj":
            continue
        path = f"{args.root_logical_path}:{field.name}"
        check_object(model, field.type.index, f"{path}: object type index is invalid")
        if model.objects[field.type.index].name != "holoscan.schema.Tensor":
            continue
        assert field.field_id is not None
        tensor_names.append(field.name)
        tensor_ids.append(field.field_id)
        assertions += (
            f"static_assert(std::same_as<decltype({native}::{cpp_field_name(field.name)}),\n"
            "                           std::shared_ptr<holoscan::Tensor>>,\n"
            f'              "{cpp_string(path)}: generated native Tensor member must be '
            'std::shared_ptr<holoscan::Tensor>");\n'
        )
    if not tensor_names:
        raise GenerationError(
            f"{args.root_logical_path}: descriptor root '{model.root_name}' has no field "
            "of exact type holoscan.schema.Tensor"
        )
    return DescriptorTensorInventory(tuple(tensor_names), tuple(tensor_ids), assertions)


def has_direct_tensor(model: Model) -> bool:
    for field in model.root.fields:
        if field.type.base == "Obj":
            check_object(
                model, field.type.index, "Generated binary schema has an invalid object reference"
            )
            if model.objects[field.type.index].name == "holoscan.schema.Tensor":
                return True
    return False


def descriptor_fragments(
    args: Args, model: Model, native: str, inventory: DescriptorTensorInventory
) -> tuple[str, Fragments, str, str]:
    assertions, codec = "", Fragments()
    encode_offsets, encode_adds = "", ""
    tensor_indices = {name: index for index, name in enumerate(inventory.names)}
    for field in ordered_fields(args, model):
        name, field_id = field.name, field.field_id
        member = cpp_field_name(name)
        assert field_id is not None
        path = f"{args.root_logical_path}:{name}"
        type_ = field.type
        base, element, index = type_.base, type_.element, type_.index
        validate_field(field, path)
        object_ = None
        referenced, referenced_native, is_tensor = "", "", False
        if base == "Obj":
            check_object(model, index, f"{path}: object type index '{index}' is invalid")
            object_ = model.objects[index]
            check_alignment(model, index, path)
            referenced = cpp_type(object_.name)
            referenced_native = referenced if object_.is_struct else f"{referenced}T"
            is_tensor = object_.name == "holoscan.schema.Tensor"
        if is_tensor:
            tensor_index = tensor_indices[name]
            encode_offsets += f"    const auto _{name} = tensor_offsets[{tensor_index}U];\n"
        elif base in SCALARS or base in {"String", "Vector"}:
            merge_fragments(
                codec,
                value_field_codec(
                    field,
                    f"value.{member}",
                    f"root.{member}",
                    path,
                    indent=4,
                    local="field",
                    nested=False,
                ),
            )
            if base == "String":
                encode_offsets += (
                    f"    const auto _{name} = value.{member}.empty()\n"
                    "        ? flatbuffers::Offset<flatbuffers::String>{}\n"
                    f"        : builder.CreateString(value.{member});\n"
                )
            elif base == "Vector":
                expression = vector_expression(element, index, member, path)
                encode_offsets += (
                    f"    using _{name}_offset_type = decltype({expression});\n"
                    f"    const _{name}_offset_type _{name} = value.{member}.empty()\n"
                    f"        ? _{name}_offset_type{{}}\n"
                    f"        : {expression};\n"
                )
        elif base == "Obj":
            assert object_ is not None
            assertions += (
                f"static_assert(std::same_as<decltype({native}::{member}),\n"
                f"                           std::shared_ptr<{referenced_native}>>,\n"
                f'              "{cpp_string(path)}: generated native object member must use '
                'the configured std::shared_ptr form");\n'
            )
            if object_.is_struct:
                codec.reset += (
                    f"    if (value.{member}) {{\n"
                    f"      *value.{member} = {referenced_native}{{}};\n"
                    "    }\n"
                )
                encode_offsets += f"    const auto* _{name} = value.{member}.get();\n"
                codec.assign += (
                    f"    if (const auto* field = root.{member}(); field != nullptr) {{\n"
                    f"      if (!value.{member}) {{\n"
                    f"        value.{member} = std::make_shared<{referenced_native}>();\n"
                    "      }\n"
                    f"      *value.{member} = *field;\n"
                    "    } else {\n"
                    f"      value.{member}.reset();\n"
                    "    }\n"
                )
            else:
                nested = nested_codec(
                    model,
                    index,
                    f"value.{member}->",
                    "nested_root->",
                    path,
                    (model.root_index,),
                )
                codec.prepare += f"    if (value.{member}) {{\n{nested.prepare}    }}\n"
                codec.reset += f"    if (value.{member}) {{\n{nested.reset}    }}\n"
                codec.validate += (
                    f"    if (const auto* nested_root = root.{member}(); "
                    f"nested_root != nullptr) {{\n{nested.validate}    }}\n"
                )
                encode_offsets += (
                    f"    const auto _{name} = value.{member}\n"
                    f"        ? {referenced}::Pack(builder, value.{member}.get())\n"
                    f"        : flatbuffers::Offset<{referenced}>{{}};\n"
                )
                codec.assign += (
                    f"    if (const auto* nested_root = root.{member}(); "
                    "nested_root != nullptr) {\n"
                    f"      if (!value.{member}) {{\n"
                    f"        value.{member} = std::make_shared<{referenced_native}>();\n"
                    "      }\n"
                    f"{nested.assign}"
                    "    } else {\n"
                    f"      value.{member}.reset();\n"
                    "    }\n"
                )
        else:
            raise GenerationError(
                f"{path}: object-API base type '{base}' is unsupported "
                "by descriptor traits generation; provide a hand-written standard codec"
            )
        encode_adds += (
            f"    root_builder.add_{member}(value.{member});\n"
            if base in SCALARS and not is_tensor
            else f"    root_builder.add_{member}(_{name});\n"
        )
    return assertions, codec, encode_offsets, encode_adds


def qualified_cpp_name(name: str) -> tuple[str | None, str]:
    namespace, separator, short = name.rpartition("::")
    return (namespace if separator else None), short


def forwarding(root: str) -> str:
    namespace, _ = qualified_cpp_name(root)
    if namespace is None:
        return ""
    if namespace == "holoscan::schema":
        return ""
    return (
        "// FlatBuffers 25.12 emits an unqualified CreateTensor call from a cross-namespace "
        "object-API Pack\n"
        "// body. Keep this forwarding overload ahead of the generated root include until "
        "upstream qualifies\n"
        "// that call.\n"
        f"namespace {namespace} {{\n\n"
        "inline flatbuffers::Offset<holoscan::schema::Tensor> CreateTensor(\n"
        "    flatbuffers::FlatBufferBuilder& builder, const holoscan::Tensor* tensor,\n"
        "    const flatbuffers::rehasher_function_t* rehasher = nullptr) {\n"
        "  return holoscan::schema::CreateTensor(builder, tensor, rehasher);\n"
        "}\n\n"
        f"}}  // namespace {namespace}\n\n"
    )


def selector(native: str, names: list[str], ids: list[int]) -> str:
    namespace, short = qualified_cpp_name(native)
    entries = "".join(
        "  inline static constexpr "
        f"holoscan::TensorFieldSelector<{native}> {cpp_field_name(name)}"
        f"{{{field_id}U, {index}U}};\n"
        for index, (name, field_id) in enumerate(zip(names, ids, strict=True))
    )
    declaration = f"struct {short.removesuffix('T')}TensorFields {{\n{entries}}};\n"
    if namespace is None:
        return declaration + "\n"
    return f"namespace {namespace} {{\n\n{declaration}\n}}  // namespace {namespace}\n\n"


def metadata_clone_qualified(args: Args, model: Model) -> bool:
    """Admit generated owning layouts without custom native fields or hidden Tensor members."""
    for index, flag in enumerate(args.flatc_flags):
        if flag.startswith(("--cpp-str-type", "--cpp-str-flex-ctor")):
            return False
        if flag == "--cpp-ptr-type":
            if (
                index + 1 == len(args.flatc_flags)
                or args.flatc_flags[index + 1] != "std::shared_ptr"
            ):
                return False
        elif (
            flag.startswith("--cpp-ptr-type=")
            and flag.removeprefix("--cpp-ptr-type=") != "std::shared_ptr"
        ):
            return False

    def custom(attributes: tuple[str, ...]) -> bool:
        return any(name.startswith(("cpp_", "native_")) for name in attributes)

    def object_safe(index: int, ancestors: tuple[int, ...]) -> bool:
        if index in ancestors or index < 0 or index >= len(model.objects):
            return False
        obj = model.objects[index]
        if custom(obj.attributes):
            return False
        for field in obj.fields:
            if field.deprecated:
                continue
            if custom(field.attributes):
                return False
            kind = field.type
            object_field = kind.base == "Obj" or (
                kind.base in {"Vector", "Array"} and kind.element == "Obj"
            )
            if object_field:
                if kind.index < 0 or kind.index >= len(model.objects):
                    return False
                if model.objects[kind.index].name == "holoscan.schema.Tensor":
                    # Only direct root Tensor members are cleared by clone_metadata. Containers
                    # must reject independently of Tensor's native_type customization attribute.
                    if kind.base == "Obj" and index == model.root_index:
                        continue
                    return False
            if kind.base in SCALARS or kind.base == "String":
                continue
            if object_field:
                if not object_safe(kind.index, (*ancestors, index)):
                    return False
            elif kind.base in {"Vector", "Array"} and (
                kind.element in SCALARS or kind.element == "String"
            ):
                continue
            else:
                return False
        return True

    return object_safe(model.root_index, ())


def native_metadata_clone(native: str, names: list[str]) -> str:
    reset = "".join(f"      result.{cpp_field_name(name)}.reset();\n" for name in names)
    return (
        "  /// @brief Copy qualified generated ordinary fields and remove every Tensor handle.\n"
        "  /// @param source Owning generated schema payload.\n"
        "  /// @return Independently owned metadata; Tensor members are unset.\n"
        f"  [[nodiscard]] static holoscan::expected<{native}, holoscan::Error> clone_metadata(\n"
        f"      const {native}& source) {{\n"
        "    try {\n"
        "      // Generated object-API copy constructors deep-copy qualified ordinary fields.\n"
        f"      {native} result{{source}};\n"
        f"{reset}"
        "      return result;\n"
        "    } catch (const std::bad_alloc&) {\n"
        "      return holoscan::make_unexpected(holoscan::Error{\n"
        "          holoscan::ErrorCode::kResourceExhausted, "
        '"schema metadata clone allocation failed"});\n'
        "    } catch (...) {\n"
        "      return holoscan::make_unexpected(holoscan::Error{\n"
        '          holoscan::ErrorCode::kCodecError, "schema metadata clone failed"});\n'
        "    }\n"
        "  }\n"
    )


def native_accessor(native: str, names: list[str]) -> str:
    cases = "".join(
        f"      case {index}U:\n        return value.{cpp_field_name(name)}.get();\n"
        for index, name in enumerate(names)
    )
    return (
        "  [[nodiscard]] static const holoscan::Tensor* tensor_at(\n"
        f"      const {native}& value, const std::size_t index) noexcept {{\n"
        "    switch (index) {\n"
        f"{cases}"
        "      default:\n"
        "        return nullptr;\n"
        "    }\n"
        "  }\n"
    )


DESCRIPTOR = Template(r"""/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Generated by holoscan_add_flatbuffer_schema() for a descriptor-Tensor root. Do not edit.
 */

#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <span>
#include <string_view>
#include <type_traits>

#include <holoscan/core/tensor_field.hpp>
$bfbs_api_include#include <holoscan/schema/flatbuffer_schema_package.hpp>

#include "tensor_generated.h"

$forwarding#include "$header"$bfbs_embed_include

$selector
namespace holoscan::schema {

template <>
struct FlatBufferSchemaBinding<$root> {
  using root_type = $root;
  using native_type = $native;
$bfbs_binding  static constexpr std::uint32_t compatibility_epoch = ${epoch}U;
  static constexpr FlatBufferFileIdentifier file_identifier =
      make_flatbuffer_file_identifier("$identifier");
  static constexpr SchemaFamilyIdentity family_identity =
      make_schema_family_identity<$root>(compatibility_epoch);
  static constexpr SchemaIdentity identity =
      make_schema_identity(family_identity, file_identifier);
  static constexpr std::size_t metadata_capacity = ${maximum}U;
  static constexpr bool descriptor_tensor = true;
  static constexpr std::array<std::uint16_t, $tensor_count> tensor_field_ids{
$tensor_ids  };
  static constexpr std::array<std::string_view, $tensor_count> tensor_field_names{
$tensor_names  };
  static constexpr std::size_t tensor_count = ${tensor_count}U;

$native_accessor
$metadata_clone
};

template <>
struct DescriptorFlatBufferTensorViewTraits<$root>
    : DescriptorFlatBufferTensorViewAdapter<$root,
$tensor_accessors> {};

}  // namespace holoscan::schema

$assertions

namespace $package {

inline constexpr std::size_t kMetadataCapacity =
    holoscan::schema::FlatBufferSchemaBinding<$root>::metadata_capacity;

}  // namespace $package

$runtime
""")

RUNTIME = Template(r"""namespace $package::detail {

struct GeneratedDescriptorCodec {
  [[nodiscard]] holoscan::expected<void, holoscan::Error> prepare(
      $native& value) const {
$prepare    return {};
  }

  void reset($native& value) const noexcept {
$reset  }

  [[nodiscard]] holoscan::expected<flatbuffers::Offset<$root>, holoscan::Error>
  encode_root(
      flatbuffers::FlatBufferBuilder& builder, const $native& value,
      const std::span<const flatbuffers::Offset<holoscan::schema::Tensor>>
          tensor_offsets) const {
    if (tensor_offsets.size() != ${tensor_count}U) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                          "descriptor Tensor offset count does not match reflected fields"});
    }
$encode_offsets    ${root}Builder root_builder{builder};
$encode_adds    return root_builder.Finish();
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> assign_root(
      $native& value, const $root& root) const {
$validate$assign    return {};
  }
};

}  // namespace $package::detail

namespace holoscan::detail {

template <>
inline const RuntimePortTypeOps* runtime_port_type_ops<$native>() noexcept {
  static const RuntimePortTypeOps ops =
      schema::make_standard_descriptor_tensor_port_type_ops<$native,
                                                            $root,
                                                            $package::kMetadataCapacity,
$tensor_members>(
          $package::detail::GeneratedDescriptorCodec{});
  return &ops;
}

}  // namespace holoscan::detail
""")


def generate_descriptor(args: Args, model: Model) -> str:
    if args.max_serialized_size is None:
        raise GenerationError("descriptor generation requires MAX_SERIALIZED_SIZE")
    if args.max_serialized_size > 65536:
        raise GenerationError(
            "MAX_SERIALIZED_SIZE exceeds the 65536-byte descriptor metadata generation bound"
        )
    values = substitutions(args, model)
    root, native = str(values["root"]), str(values["native"])
    inventory = descriptor_tensor_inventory(args, model, native)
    names, ids = list(inventory.names), list(inventory.field_ids)
    assertions = inventory.assertions
    codec, encode_offsets, encode_adds = Fragments(), "", ""
    if not args.descriptor_tensor_custom_codec:
        object_assertions, codec, encode_offsets, encode_adds = descriptor_fragments(
            args, model, native, inventory
        )
        assertions += object_assertions
    count = len(names)
    accessors = ",\n".join(
        f"                                            &{root}::{cpp_field_name(name)}"
        for name in names
    )
    members = ",\n".join(
        "                                                            "
        f"&{native}::{cpp_field_name(name)}"
        for name in names
    )
    runtime = ""
    if not args.descriptor_tensor_custom_codec:
        runtime = RUNTIME.substitute(
            package=values["package"],
            native=native,
            root=root,
            tensor_count=count,
            prepare=codec.prepare,
            reset=codec.reset,
            encode_offsets=encode_offsets,
            encode_adds=encode_adds,
            validate=codec.validate,
            assign=codec.assign,
            tensor_members=members,
        )
    return DESCRIPTOR.substitute(
        values,
        forwarding=forwarding(root),
        tensor_count=count,
        tensor_ids="".join(f"    {field_id}U,\n" for field_id in ids),
        tensor_names="".join(f'    "{name}",\n' for name in names),
        selector=selector(native, names, ids),
        tensor_accessors=accessors,
        assertions=assertions,
        native_accessor=native_accessor(native, names),
        metadata_clone=native_metadata_clone(native, names)
        if metadata_clone_qualified(args, model)
        else "",
        runtime=runtime,
    )


def enum_at(model: Model, index: int, path: str) -> Enum | None:
    """Return a reflected enum for a scalar index, retaining -1 as no enum."""
    if index < 0:
        return None
    if index >= len(model.enums):
        raise GenerationError(f"{path}: reflected enum reference is invalid")
    return model.enums[index]


def reachable_python_objects(model: Model) -> list[int]:
    """Return the acyclic object inventory used by one generated Python adapter."""
    result: list[int] = []
    active: set[int] = set()
    visited: set[int] = set()

    def visit(index: int) -> None:
        if index in active:
            raise GenerationError("recursive tables are unsupported by Python schema generation")
        if index in visited or model.objects[index].name == "holoscan.schema.Tensor":
            return
        active.add(index)
        object_ = model.objects[index]
        for field in canonical_python_fields(object_):
            path = f"{object_.name}.{field.name}"
            if field.name in UNESCAPED_CPP20_KEYWORDS:
                raise GenerationError(
                    f"{path}: field name is a C++20 keyword that the pinned flatc does not escape"
                )
            if field.optional and field.type.base in SCALARS:
                raise GenerationError(
                    f"{path}: optional scalar fields are unsupported by Python schema generation"
                )
            base, element, child = field.type.base, field.type.element, field.type.index
            if base in SCALARS:
                enum_at(model, child, path)
            elif base == "Obj":
                check_object(model, child, "Python schema object reference is invalid")
                if (
                    index == model.root_index
                    and model.objects[child].name == "holoscan.schema.Tensor"
                    and python_field_name(field.name) != field.name
                ):
                    raise GenerationError(
                        f"{path}: Tensor field names that require Python keyword escaping "
                        "are unsupported"
                    )
                visit(child)
            elif base == "Vector":
                if element == "Obj":
                    check_object(model, child, "Python schema vector object reference is invalid")
                    if not model.objects[child].is_struct:
                        raise GenerationError(
                            f"{path}: vectors of tables are unsupported by Python schema generation"
                        )
                    visit(child)
                elif element not in SCALARS | {"String"}:
                    raise GenerationError(
                        f"{path}: unsupported Python sequence element '{element}'"
                    )
                elif element in SCALARS:
                    enum_at(model, child, path)
            elif base == "Array":
                if element not in SCALARS or field.type.fixed_length <= 0:
                    raise GenerationError(
                        f"{path}: only fixed-length scalar arrays are supported by Python schema "
                        "generation"
                    )
                enum_at(model, child, path)
            elif base != "String":
                raise GenerationError(f"{path}: Python schema field type '{base}' is unsupported")
        active.remove(index)
        visited.add(index)
        result.append(index)

    if model.root_index is None:
        for object_index in range(len(model.objects)):
            visit(object_index)
    else:
        visit(model.root_index)
    return result


def native_object_type(model: Model, index: int) -> str:
    object_ = model.objects[index]
    name = cpp_type(object_.name)
    return name if object_.is_struct else f"{name}T"


# Keep this list synchronized with flatc's CppGenerator keyword table. The generated object API
# appends an underscore to exactly these identifiers.
FLATC_CPP_KEYWORDS = {
    "alignas",
    "alignof",
    "and",
    "and_eq",
    "asm",
    "atomic_cancel",
    "atomic_commit",
    "atomic_noexcept",
    "auto",
    "bitand",
    "bitor",
    "bool",
    "break",
    "case",
    "catch",
    "char",
    "char16_t",
    "char32_t",
    "class",
    "compl",
    "concept",
    "const",
    "constexpr",
    "const_cast",
    "continue",
    "co_await",
    "co_return",
    "co_yield",
    "decltype",
    "default",
    "delete",
    "do",
    "double",
    "dynamic_cast",
    "else",
    "enum",
    "explicit",
    "export",
    "extern",
    "false",
    "float",
    "for",
    "friend",
    "goto",
    "if",
    "import",
    "inline",
    "int",
    "long",
    "module",
    "mutable",
    "namespace",
    "new",
    "noexcept",
    "not",
    "not_eq",
    "nullptr",
    "operator",
    "or",
    "or_eq",
    "private",
    "protected",
    "public",
    "register",
    "reinterpret_cast",
    "requires",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "static_assert",
    "static_cast",
    "struct",
    "switch",
    "synchronized",
    "template",
    "this",
    "thread_local",
    "throw",
    "true",
    "try",
    "typedef",
    "typeid",
    "typename",
    "union",
    "unsigned",
    "using",
    "virtual",
    "void",
    "volatile",
    "wchar_t",
    "while",
    "xor",
    "xor_eq",
}

# The pinned flatc accepts these C++20 reserved words without escaping them. Reject them before
# emitting adapters because flatc's own object API cannot be compiled as C++20 for these names.
UNESCAPED_CPP20_KEYWORDS = {"char8_t", "consteval", "constinit", "reflexpr"}

PYTHON_GENERATED_MEMBERS = {
    "_registry_name",
    "_holoscan_schema_values",
    "_from_holoscan_schema",
    "tensors",
    "_holoscan_dataclass",
    "_holoscan_field",
}

PYTHON_ENUM_RESERVED_MEMBERS = {"mro"}


def cpp_field_name(name: str) -> str:
    """Match flatc's trailing-underscore spelling for C++ keywords."""
    return f"{name}_" if name in FLATC_CPP_KEYWORDS else name


def python_field_name(name: str) -> str:
    """Keep wire names exact while avoiding Python and generated-member collisions."""
    return f"{name}_" if keyword.iskeyword(name) or name in PYTHON_GENERATED_MEMBERS else name


def python_enum_member_name(name: str) -> str:
    """Keep enum values importable while preserving the wire value separately."""
    is_sunder = len(name) > 2 and name.startswith("_") and name.endswith("_")
    if keyword.iskeyword(name) or name in PYTHON_ENUM_RESERVED_MEMBERS or is_sunder:
        return f"{name}_"
    return name


def canonical_python_fields(object_: Object) -> list[Field]:
    """Return generated fields in their stable FlatBuffers field-ID order."""
    if any(field.field_id is None for field in object_.fields):
        raise GenerationError(
            f"{object_.name}: Python schema generation requires explicit field IDs"
        )
    return sorted(
        object_.fields,
        key=lambda field: field.field_id if field.field_id is not None else -1,
    )


def foreign_scalar_expression(field: Field, expression: str) -> str:
    base = field.type.base
    if base == "Bool":
        return f"holoscan::authoring::ForeignSchemaValue::boolean({expression})"
    if base in {"Byte", "Short", "Int", "Long"}:
        return (
            "holoscan::authoring::ForeignSchemaValue::signed_integer("
            f"static_cast<std::int64_t>({expression}))"
        )
    if base in {"UByte", "UShort", "UInt", "ULong", "UType"}:
        return (
            "holoscan::authoring::ForeignSchemaValue::unsigned_integer("
            f"static_cast<std::uint64_t>({expression}))"
        )
    if base in {"Float", "Double"}:
        return (
            "holoscan::authoring::ForeignSchemaValue::floating_point("
            f"static_cast<double>({expression}))"
        )
    if base == "String":
        return f"holoscan::authoring::ForeignSchemaValue::string({expression})"
    raise AssertionError(f"unsupported scalar expression for {base}")


CPP_SCALAR_TYPES = {
    "Bool": "bool",
    "Byte": "std::int8_t",
    "UByte": "std::uint8_t",
    "Short": "std::int16_t",
    "UShort": "std::uint16_t",
    "Int": "std::int32_t",
    "UInt": "std::uint32_t",
    "Long": "std::int64_t",
    "ULong": "std::uint64_t",
    "Float": "float",
    "Double": "double",
    "UType": "std::uint8_t",
    "String": "std::string",
}


def scalar_field(field: Field, base: str | None = None) -> Field:
    """Return a scalar view of a vector/array field for shared conversion generation."""
    return Field(
        name=field.name,
        field_id=field.field_id,
        deprecated=False,
        required=False,
        optional=False,
        default_integer=field.default_integer,
        default_real=field.default_real,
        type=ReflectedType(
            base or field.type.element,
            "None",
            field.type.index,
            0,
            field.type.element_size,
            1,
        ),
        native_inline=field.type.element == "Obj",
    )


def cpp_scalar_type(model: Model, field: Field) -> str:
    enum = enum_at(model, field.type.index, field.name)
    if enum is not None:
        return cpp_type(enum.name)
    try:
        return CPP_SCALAR_TYPES[field.type.base]
    except KeyError as error:
        raise GenerationError(
            f"{field.name}: unsupported scalar type '{field.type.base}'"
        ) from error


def foreign_sequence_decode(
    *,
    field_name: str,
    field_path: str,
    target: str,
    element_type: str,
    append: str,
    exact_length: int | None = None,
) -> str:
    length_check = ""
    if exact_length is not None:
        length_check = (
            f"    if (elements->size() != {exact_length}U) {{\n"
            "      return holoscan::make_unexpected(holoscan::Error{\n"
            "          holoscan::ErrorCode::kInvalidArgument, "
            f'"PYTHON_SCHEMA_FIXED_ARRAY_LENGTH_MISMATCH; field={cpp_string(field_path)}"}});\n'
            "    }\n"
        )
    return (
        f'  if (const auto* field = value.field("{cpp_string(field_name)}")) {{\n'
        "    const auto* elements =\n"
        "        field->get<holoscan::authoring::ForeignSchemaValue::Sequence>();\n"
        "    if (elements == nullptr) {\n"
        "      return holoscan::make_unexpected(holoscan::Error{\n"
        "          holoscan::ErrorCode::kInvalidArgument, "
        f'"PYTHON_SCHEMA_FIELD_TYPE_MISMATCH; field={cpp_string(field_path)}"}});\n'
        "    }\n"
        f"{length_check}"
        f"{target}"
        "    for (const auto& element : *elements) {\n"
        "      auto converted = holoscan::authoring::detail::foreign_schema_scalar<"
        f"{element_type}>(element);\n"
        "      if (!converted) {\n"
        "        return holoscan::make_unexpected(with_schema_field(\n"
        f'            std::move(converted).error(), "{cpp_string(field_path)}"));\n'
        "      }\n"
        f"{append}"
        "    }\n"
        "  }\n"
    )


def cpp_python_adapter(model: Model) -> str:
    if model.root_index is None or model.root_name is None:
        raise GenerationError("Python native adapter generation requires a payload root")
    indices = reachable_python_objects(model)
    package = package_namespace(model.root_name)
    root_short_name = model.root_name.rsplit(".", 1)[-1]
    adapter_namespace = (
        f"{package}::holoscan_python_{root_short_name}_detail"
        if package
        else f"holoscan_python_{root_short_name}_detail"
    )
    declarations = ""
    definitions = ""
    for index in indices:
        native = native_object_type(model, index)
        declarations += (
            "[[nodiscard]] holoscan::expected<holoscan::authoring::ForeignSchemaValue, "
            f"holoscan::Error> to_foreign_{index}(const {native}& value);\n"
            f"[[nodiscard]] holoscan::expected<{native}, holoscan::Error> "
            f"from_foreign_{index}(const holoscan::authoring::ForeignSchemaValue& value);\n"
        )

    for index in indices:
        object_ = model.objects[index]
        native = native_object_type(model, index)
        to_fields, from_fields = "", ""
        struct_locals, struct_arguments = "", []
        for field in canonical_python_fields(object_):
            name = field.name
            member = cpp_field_name(name)
            base, element, child = field.type.base, field.type.element, field.type.index
            path = f"{object_.name}.{name}"
            source = f"value.{member}()" if object_.is_struct else f"value.{member}"
            if base in SCALARS or base == "String":
                to_fields += (
                    f'  fields.emplace("{cpp_string(name)}", '
                    f"{foreign_scalar_expression(field, source)});\n"
                )
                if object_.is_struct:
                    local = f"converted_{member}"
                    scalar_type = cpp_scalar_type(model, field)
                    struct_locals += f"  {scalar_type} {local}{{}};\n"
                    struct_arguments.append(local)
                    target = local
                else:
                    target = f"result.{member}"
                    scalar_type = f"std::remove_cvref_t<decltype({target})>"
                from_fields += (
                    f'  if (const auto* field = value.field("{cpp_string(name)}")) {{\n'
                    "    auto converted = holoscan::authoring::detail::foreign_schema_scalar<"
                    f"{scalar_type}>(*field);\n"
                    "    if (!converted) {\n"
                    "      return holoscan::make_unexpected(with_schema_field(\n"
                    f'          std::move(converted).error(), "{cpp_string(path)}"));\n'
                    "    }\n"
                    f"    {target} = *converted;\n"
                    "  }\n"
                )
            elif base == "Vector":
                element_field = scalar_field(field)
                to_fields += (
                    "  {\n"
                    "    holoscan::authoring::ForeignSchemaValue::Sequence elements;\n"
                    f"    elements.reserve(value.{member}.size());\n"
                    f"    for (const auto& element : value.{member}) {{\n"
                )
                if element == "Obj":
                    child_object = model.objects[child]
                    if not child_object.is_struct:
                        raise GenerationError(f"{path}: vectors of tables are unsupported")
                    to_fields += (
                        f"      auto converted = to_foreign_{child}(element);\n"
                        "      if (!converted) {\n"
                        "        return holoscan::make_unexpected(std::move(converted).error());\n"
                        "      }\n"
                        "      elements.push_back(std::move(*converted));\n"
                    )
                    from_fields += (
                        f'  if (const auto* field = value.field("{cpp_string(name)}")) {{\n'
                        "    const auto* elements =\n"
                        "        field->get<holoscan::authoring::ForeignSchemaValue::Sequence>();\n"
                        "    if (elements == nullptr) {\n"
                        "      return holoscan::make_unexpected(holoscan::Error{\n"
                        "          holoscan::ErrorCode::kInvalidArgument, "
                        f'"PYTHON_SCHEMA_FIELD_TYPE_MISMATCH; field={cpp_string(path)}"}});\n'
                        "    }\n"
                        f"    result.{member}.clear();\n"
                        f"    result.{member}.reserve(elements->size());\n"
                        "    for (const auto& element : *elements) {\n"
                        f"      auto converted = from_foreign_{child}(element);\n"
                        "      if (!converted) {\n"
                        "        return holoscan::make_unexpected(with_schema_field(\n"
                        f'            std::move(converted).error(), "{cpp_string(path)}"));\n'
                        "      }\n"
                        f"      result.{member}.push_back(std::move(*converted));\n"
                        "    }\n"
                        "  }\n"
                    )
                else:
                    foreign_element = foreign_scalar_expression(element_field, "element")
                    to_fields += f"      elements.push_back({foreign_element});\n"
                    from_fields += foreign_sequence_decode(
                        field_name=name,
                        field_path=path,
                        target=(
                            f"    result.{member}.clear();\n"
                            f"    result.{member}.reserve(elements->size());\n"
                        ),
                        element_type=(
                            f"typename std::remove_cvref_t<decltype(result.{member})>::value_type"
                        ),
                        append=f"      result.{member}.push_back(std::move(*converted));\n",
                    )
                to_fields += (
                    "    }\n"
                    f'    fields.emplace("{cpp_string(name)}", '
                    "holoscan::authoring::ForeignSchemaValue::sequence(std::move(elements)));\n"
                    "  }\n"
                )
            elif base == "Array":
                element_field = scalar_field(field)
                length = field.type.fixed_length
                element_type = cpp_scalar_type(model, element_field)
                local = f"converted_{member}"
                struct_locals += f"  std::array<{element_type}, {length}> {local}{{}};\n"
                struct_arguments.append(
                    f"flatbuffers::span<const {element_type}, {length}>"
                    f"{{{local}.data(), {length}U}}"
                )
                foreign_element = foreign_scalar_expression(element_field, "element")
                to_fields += (
                    "  {\n"
                    "    holoscan::authoring::ForeignSchemaValue::Sequence elements;\n"
                    f"    elements.reserve({length}U);\n"
                    f"    for (const auto& element : *value.{member}()) {{\n"
                    f"      elements.push_back({foreign_element});\n"
                    "    }\n"
                    f'    fields.emplace("{cpp_string(name)}", '
                    "holoscan::authoring::ForeignSchemaValue::sequence(std::move(elements)));\n"
                    "  }\n"
                )
                from_fields += foreign_sequence_decode(
                    field_name=name,
                    field_path=path,
                    target="    std::size_t element_index = 0U;\n",
                    element_type=element_type,
                    append=f"      {local}[element_index++] = *converted;\n",
                    exact_length=length,
                )
            elif base == "Obj":
                check_object(model, child, "Python schema object reference is invalid")
                if model.objects[child].name == "holoscan.schema.Tensor":
                    to_fields += (
                        f'  fields.emplace("{cpp_string(name)}", value.{member}\n'
                        "      ? holoscan::authoring::ForeignSchemaValue::tensor("
                        f"*value.{member})\n"
                        "      : holoscan::authoring::ForeignSchemaValue{});\n"
                    )
                    from_fields += (
                        f'  if (const auto* field = value.field("{cpp_string(name)}")) {{\n'
                        "    if (field->is_null()) {\n"
                        f"      result.{member}.reset();\n"
                        "    } else if (const auto* tensor = field->get<holoscan::Tensor>()) {\n"
                        f"      result.{member} = std::make_shared<holoscan::Tensor>(*tensor);\n"
                        "    } else {\n"
                        "      return holoscan::make_unexpected(holoscan::Error{\n"
                        "          holoscan::ErrorCode::kInvalidArgument, "
                        f'"PYTHON_SCHEMA_FIELD_TYPE_MISMATCH; field={cpp_string(path)}"}});\n'
                        "    }\n"
                        "  }\n"
                    )
                else:
                    child_native = native_object_type(model, child)
                    if object_.is_struct:
                        local = f"converted_{member}"
                        struct_locals += f"  {child_native} {local}{{}};\n"
                        struct_arguments.append(local)
                        to_fields += (
                            "  {\n"
                            f"    auto converted = to_foreign_{child}(value.{member}());\n"
                            "    if (!converted) {\n"
                            "      return holoscan::make_unexpected(with_schema_field(\n"
                            f'          std::move(converted).error(), "{cpp_string(path)}"));\n'
                            "    }\n"
                            f'    fields.emplace("{cpp_string(name)}", std::move(*converted));\n'
                            "  }\n"
                        )
                        from_fields += (
                            f'  if (const auto* field = value.field("{cpp_string(name)}")) {{\n'
                            "    if (field->is_null()) {\n"
                            "      return holoscan::make_unexpected(holoscan::Error{\n"
                            "          holoscan::ErrorCode::kInvalidArgument, "
                            f'"PYTHON_SCHEMA_FIELD_TYPE_MISMATCH; field={cpp_string(path)}"}});\n'
                            "    }\n"
                            f"    auto converted = from_foreign_{child}(*field);\n"
                            "    if (!converted) {\n"
                            "      return holoscan::make_unexpected(with_schema_field(\n"
                            f'          std::move(converted).error(), "{cpp_string(path)}"));\n'
                            "    }\n"
                            f"    {local} = std::move(*converted);\n"
                            "  }\n"
                        )
                    else:
                        inline = field.native_inline
                        source_value = f"value.{member}" if inline else f"*value.{member}"
                        condition = "true" if inline else f"value.{member}"
                        to_fields += (
                            f"  if ({condition}) {{\n"
                            f"    auto converted = to_foreign_{child}({source_value});\n"
                            "    if (!converted) {\n"
                            "      return holoscan::make_unexpected(with_schema_field(\n"
                            f'          std::move(converted).error(), "{cpp_string(path)}"));\n'
                            "    }\n"
                            f'    fields.emplace("{cpp_string(name)}", std::move(*converted));\n'
                            "  } else {\n"
                            f'    fields.emplace("{cpp_string(name)}", '
                            "holoscan::authoring::ForeignSchemaValue{});\n"
                            "  }\n"
                        )
                        null_assignment = (
                            f"      result.{member} = {child_native}{{}};\n"
                            if inline
                            else f"      result.{member}.reset();\n"
                        )
                        value_assignment = (
                            f"      result.{member} = std::move(*converted);\n"
                            if inline
                            else f"      result.{member} = std::make_shared<{child_native}>("
                            "std::move(*converted));\n"
                        )
                        from_fields += (
                            f'  if (const auto* field = value.field("{cpp_string(name)}")) {{\n'
                            "    if (field->is_null()) {\n"
                            f"{null_assignment}"
                            "    } else {\n"
                            f"      auto converted = from_foreign_{child}(*field);\n"
                            "      if (!converted) {\n"
                            "        return holoscan::make_unexpected(with_schema_field(\n"
                            f'            std::move(converted).error(), "{cpp_string(path)}"));\n'
                            "      }\n"
                            f"{value_assignment}"
                            "    }\n"
                            "  }\n"
                        )
        known_name_conditions = " && ".join(
            f'name != "{cpp_string(field.name)}"' for field in canonical_python_fields(object_)
        )
        unknown_fields = (
            "  for (const auto& [name, field_value] : *fields) {\n"
            "    (void)field_value;\n"
            f"    if ({known_name_conditions or 'true'}) {{\n"
            "      return holoscan::make_unexpected(holoscan::Error{\n"
            "          holoscan::ErrorCode::kInvalidArgument, "
            '"PYTHON_SCHEMA_FIELD_UNKNOWN; field=" + name});\n'
            "    }\n"
            "  }\n"
        )
        if object_.is_struct:
            result_declaration = struct_locals
            result_return = f"  return {native}{{{', '.join(struct_arguments)}}};\n"
        else:
            result_declaration = f"  {native} result{{}};\n"
            result_return = "  return result;\n"
        definitions += (
            "// Generated field dispatch stays linear to preserve exact schema-path diagnostics.\n"
            "[[nodiscard]] inline holoscan::expected<"
            "holoscan::authoring::ForeignSchemaValue, holoscan::Error>\n"
            f"to_foreign_{index}(const {native}& value) {{  "
            "// NOLINT(readability-function-cognitive-complexity)\n"
            "  holoscan::authoring::ForeignSchemaValue::Object fields;\n"
            f"{to_fields}"
            "  return holoscan::authoring::ForeignSchemaValue::object(std::move(fields));\n"
            "}\n\n"
            "// Generated field dispatch stays linear to preserve exact schema-path diagnostics.\n"
            f"[[nodiscard]] inline holoscan::expected<{native}, holoscan::Error>\n"
            f"from_foreign_{index}(const holoscan::authoring::ForeignSchemaValue& value) {{  "
            "// NOLINT(readability-function-cognitive-complexity)\n"
            "  const auto* fields =\n"
            "      value.get<holoscan::authoring::ForeignSchemaValue::Object>();\n"
            "  if (fields == nullptr) {\n"
            "    return holoscan::make_unexpected(holoscan::Error{\n"
            "        holoscan::ErrorCode::kInvalidArgument, "
            '"PYTHON_SCHEMA_PAYLOAD_TYPE_MISMATCH"});\n'
            "  }\n"
            f"{unknown_fields}"
            f"{result_declaration}"
            f"{from_fields}"
            f"{result_return}"
            "}\n\n"
        )

    root_native = native_object_type(model, model.root_index)
    return (
        "\n#include <array>\n"
        "#include <string_view>\n"
        "#include <holoscan/authoring/registry.hpp>\n\n"
        f"namespace {adapter_namespace} {{\n\n"
        "[[nodiscard]] inline holoscan::Error with_schema_field(\n"
        "    holoscan::Error error, std::string_view field) {\n"
        "  if (!error.message.empty()) {\n"
        '    error.message += "; ";\n'
        "  }\n"
        '  error.message += "field=";\n'
        "  error.message += field;\n"
        "  return error;\n"
        "}\n\n"
        f"{declarations}\n{definitions}"
        f"}}  // namespace {adapter_namespace}\n\n"
        "namespace holoscan::authoring {\n\n"
        "template <>\n"
        f"struct PythonSchemaNativeTraits<{root_native}> {{\n"
        "  [[nodiscard]] static expected<ForeignSchemaValue, Error> to_foreign(\n"
        f"      const {root_native}& value) {{\n"
        f"    return {adapter_namespace}::to_foreign_{model.root_index}(value);\n"
        "  }\n"
        f"  [[nodiscard]] static expected<{root_native}, Error> from_foreign(\n"
        "      const ForeignSchemaValue& value) {\n"
        f"    return {adapter_namespace}::from_foreign_{model.root_index}(value);\n"
        "  }\n"
        "};\n\n"
        "}  // namespace holoscan::authoring\n"
    )


def python_name(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def python_default(model: Model, field: Field) -> str:
    base = field.type.base
    if base == "Bool":
        return "True" if str(field.default_integer).lower() in {"1", "true"} else "False"
    if base in INTEGERS:
        return str(field.default_integer).lower()
    if base in {"Float", "Double"}:
        value = str(field.default_real)
        if not FINITE_FLOAT.fullmatch(value):
            raise GenerationError(
                f"{field.name}: non-finite floating-point defaults are unsupported by Python "
                "schema generation"
            )
        return value
    if base == "String":
        return '""'
    if base == "Vector":
        return "_holoscan_field(default_factory=list)"
    if base == "Array":
        enum = enum_at(model, field.type.index, field.name)
        element = (
            f"{python_name(enum.name)}(0)"
            if enum is not None
            else scalar_field_placeholder_default(field.type.element)
        )
        return f"({element},) * {field.type.fixed_length}"
    if base == "Obj":
        return "None"
    raise GenerationError(f"unsupported Python schema default for '{base}'")


def scalar_field_placeholder_default(base: str) -> str:
    if base == "Bool":
        return "False"
    if base in INTEGERS:
        return "0"
    if base in {"Float", "Double"}:
        return "0.0"
    raise GenerationError(f"unsupported fixed-array scalar type '{base}'")


def python_annotation(model: Model, field: Field, *, containing_struct: bool = False) -> str:
    base, index = field.type.base, field.type.index
    enum = enum_at(model, index, field.name) if base in SCALARS else None
    if enum is not None:
        return python_name(enum.name)
    if base == "Bool":
        return "bool"
    if base in INTEGERS:
        return "int"
    if base in {"Float", "Double"}:
        return "float"
    if base == "String":
        return "str"
    if base == "Vector":
        element_field = scalar_field(field)
        return f"list[{python_annotation(model, element_field)}]"
    if base == "Array":
        element_field = scalar_field(field)
        return f"tuple[{python_annotation(model, element_field)}, ...]"
    if base == "Obj":
        check_object(model, index, "Python schema object reference is invalid")
        if model.objects[index].name == "holoscan.schema.Tensor":
            return "_Any | None"
        suffix = "" if containing_struct or field.native_inline else " | None"
        return f"{python_name(model.objects[index].name)}{suffix}"
    raise GenerationError(f"unsupported Python schema annotation for '{base}'")


def python_local_objects(args: Args, model: Model) -> tuple[list[int], dict[str, str]]:
    external = dict(args.python_external_types)
    known = {object_.name for object_ in model.objects} | {enum.name for enum in model.enums}
    unknown = sorted(set(external) - known)
    if unknown:
        raise GenerationError(f"Python external types are absent from reflection: {unknown}")
    reachable = reachable_python_objects(model)
    root_declaration = (
        model.objects[model.root_index].declaration_file
        if model.root_index is not None
        else args.root_logical_path
    ).lstrip("/")

    def require_local_ownership(name: str, declaration_file: str) -> None:
        if (
            declaration_file
            and declaration_file.lstrip("/") != root_declaration
            and name not in external
        ):
            raise GenerationError(
                f"{name}: Python type declared in '{declaration_file}' must be mapped with "
                "--python-external-type"
            )

    if model.root_index is None:
        result = []
        for index in reachable:
            object_ = model.objects[index]
            if object_.name in external or object_.name == "holoscan.schema.Tensor":
                continue
            require_local_ownership(object_.name, object_.declaration_file)
            result.append(index)
        return result, external
    result: list[int] = []
    visited: set[int] = set()

    def visit(index: int) -> None:
        object_ = model.objects[index]
        if index in visited or object_.name in external or object_.name == "holoscan.schema.Tensor":
            return
        require_local_ownership(object_.name, object_.declaration_file)
        visited.add(index)
        for field in canonical_python_fields(object_):
            if (
                field.type.base == "Obj"
                or field.type.base == "Vector"
                and field.type.element == "Obj"
            ):
                visit(field.type.index)
        result.append(index)

    visit(model.root_index)
    return result, external


def reachable_python_enums(
    args: Args, model: Model, indices: list[int], external: dict[str, str]
) -> list[int]:
    result: set[int] = set()
    root_declaration = (
        model.objects[model.root_index].declaration_file
        if model.root_index is not None
        else args.root_logical_path
    ).lstrip("/")
    for object_index in indices:
        for field in canonical_python_fields(model.objects[object_index]):
            enum_index = field.type.index
            has_enum = field.type.base in SCALARS or (
                field.type.base in {"Vector", "Array"} and field.type.element in SCALARS
            )
            if has_enum and enum_index >= 0:
                enum = enum_at(model, enum_index, field.name)
                assert enum is not None
                if enum.name in external:
                    continue
                if enum.declaration_file and enum.declaration_file.lstrip("/") != root_declaration:
                    raise GenerationError(
                        f"{enum.name}: Python enum declared in '{enum.declaration_file}' must be "
                        "mapped with --python-external-type"
                    )
                result.add(enum_index)
    return sorted(result)


def python_enum_default(model: Model, field: Field) -> str | None:
    if field.type.base not in SCALARS:
        return None
    enum = enum_at(model, field.type.index, field.name)
    if enum is None:
        return None
    return f"{python_name(enum.name)}({field.default_integer})"


def python_fixed_array_element_type(model: Model, field: Field) -> tuple[str, bool]:
    """Return the runtime type check and whether integral values satisfy a float field."""
    element = scalar_field(field)
    enum = enum_at(model, element.type.index, field.name)
    if enum is not None:
        return python_name(enum.name), False
    if element.type.base == "Bool":
        return "bool", False
    if element.type.base in INTEGERS:
        return "int", False
    if element.type.base in {"Float", "Double"}:
        return "float", True
    raise GenerationError(
        f"{field.name}: unsupported fixed-array scalar type '{element.type.base}'"
    )


def validate_python_bound_names(names: list[str], external: dict[str, str]) -> None:
    """Reject public and imported spellings that would overwrite one another."""
    bound_names = names + [python_name(name) for name in external]
    stub_helper_names = {"Any", "ClassVar", "Final", "IntEnum", "IntFlag", "Mapping"}
    reserved_names = sorted(stub_helper_names.intersection(bound_names))
    if reserved_names:
        raise GenerationError(
            "Python schema export and imported type names conflict with stub helper imports: "
            + ", ".join(reserved_names)
        )
    if len(bound_names) != len(set(bound_names)):
        raise GenerationError("Python schema export and imported type names must be unique")


def python_schema_module(args: Args, model: Model) -> str:
    indices, external = python_local_objects(args, model)
    enum_indices = reachable_python_enums(args, model, indices, external)
    names = [python_name(model.enums[index].name) for index in enum_indices]
    names.extend(python_name(model.objects[index].name) for index in indices)
    names.extend(name for name, _ in args.python_constants)
    validate_python_bound_names(names, external)
    imports = "".join(
        f"from {module} import {python_name(name)}\n" for name, module in external.items()
    )
    enums = ""
    for index in enum_indices:
        enum = model.enums[index]
        base = "_IntFlag" if enum.bit_flags else "_HoloscanIntEnum"
        member_names = [python_enum_member_name(value.name) for value in enum.values]
        if len(member_names) != len(set(member_names)):
            raise GenerationError(f"{enum.name}: Python enum names collide after escaping")
        members = "".join(
            f"    {python_enum_member_name(value.name)} = {value.value}\n" for value in enum.values
        )
        enums += f"class {python_name(enum.name)}({base}):\n{members}\n\n"
    classes = ""
    for index in indices:
        object_ = model.objects[index]
        name = python_name(object_.name)
        field_names = [python_field_name(field.name) for field in canonical_python_fields(object_)]
        if len(field_names) != len(set(field_names)):
            raise GenerationError(
                f"{object_.name}: Python field names collide after keyword escaping"
            )
        fields = ""
        to_values = ""
        from_values = ""
        tensor_names: list[tuple[str, str]] = []
        for schema_field in canonical_python_fields(object_):
            wire_name = schema_field.name
            field_name = python_field_name(wire_name)
            default = python_enum_default(model, schema_field) or python_default(
                model, schema_field
            )
            if schema_field.type.base == "Obj" and (
                object_.is_struct or schema_field.native_inline
            ):
                child_name = python_name(model.objects[schema_field.type.index].name)
                default = f"_holoscan_field(default_factory={child_name})"
            fields += (
                f"    {field_name}: "
                f"{python_annotation(model, schema_field, containing_struct=object_.is_struct)} "
                f"= {default}\n"
            )
            base, child = schema_field.type.base, schema_field.type.index
            to_expression = f"self.{field_name}"
            from_expression = f'values["{wire_name}"]'
            if base == "Obj":
                check_object(model, child, "Python schema object reference is invalid")
                if model.objects[child].name != "holoscan.schema.Tensor":
                    child_name = python_name(model.objects[child].name)
                    if object_.is_struct or schema_field.native_inline:
                        to_expression = f"self.{field_name}._holoscan_schema_values()"
                        from_expression = (
                            f'{child_name}._from_holoscan_schema(values["{wire_name}"])'
                        )
                    else:
                        to_expression = (
                            f"None if self.{field_name} is None else "
                            f"self.{field_name}._holoscan_schema_values()"
                        )
                        from_expression = (
                            f'None if values["{wire_name}"] is None else '
                            f'{child_name}._from_holoscan_schema(values["{wire_name}"])'
                        )
                elif index == model.root_index:
                    tensor_names.append((wire_name, field_name))
            elif base == "Vector" and schema_field.type.element == "Obj":
                child_name = python_name(model.objects[child].name)
                to_expression = f"[item._holoscan_schema_values() for item in self.{field_name}]"
                from_expression = (
                    f'[{child_name}._from_holoscan_schema(item) for item in values["{wire_name}"]]'
                )
            elif base in {"Vector", "Array"} and schema_field.type.index >= 0:
                enum_name = python_name(model.enums[schema_field.type.index].name)
                constructor = "tuple" if base == "Array" else "list"
                to_expression = (
                    f'_holoscan_enum_values(self.{field_name}, "{object_.name}.{wire_name}")'
                )
                from_expression = (
                    f'{constructor}({enum_name}(item) for item in values["{wire_name}"])'
                )
            elif base in SCALARS and schema_field.type.index >= 0:
                enum_name = python_name(model.enums[schema_field.type.index].name)
                to_expression = (
                    f'_holoscan_enum_value(self.{field_name}, "{object_.name}.{wire_name}")'
                )
                from_expression = f'{enum_name}(values["{wire_name}"])'
            elif base == "Array":
                from_expression = f'tuple(values["{wire_name}"])'
            to_values += f'            "{wire_name}": {to_expression},\n'
            from_values += (
                f'        if "{wire_name}" in values:\n'
                f'            known["{field_name}"] = {from_expression}\n'
            )
        root_members = ""
        if index == model.root_index:
            root_members = f'    _registry_name: _ClassVar[str] = "{model.root_name}"\n\n'
            if tensor_names:
                entries = "".join(
                    f'            "{wire_name}": self.{field_name},\n'
                    for wire_name, field_name in tensor_names
                )
                root_members += (
                    "    @property\n"
                    "    def tensors(self) -> _Mapping[str, _Any]:\n"
                    "        return _MappingProxyType({\n"
                    f"{entries}"
                    "        })\n\n"
                )
        class_fields = fields or "    pass\n"
        post_init = ""
        for schema_field in canonical_python_fields(object_):
            if schema_field.type.base == "Array":
                field_name = python_field_name(schema_field.name)
                element_type, allow_integer = python_fixed_array_element_type(model, schema_field)
                post_init += (
                    f"        converted = _holoscan_fixed_array(self.{field_name}, "
                    f'{schema_field.type.fixed_length}, "{object_.name}.{schema_field.name}", '
                    f"{element_type}, {allow_integer})\n"
                    f'        _builtins.object.__setattr__(self, "{field_name}", converted)\n'
                )
        if post_init:
            post_init = "    def __post_init__(self) -> None:\n" + post_init + "\n"
        decorator = (
            "@_holoscan_dataclass(slots=True, frozen=True, kw_only=True)"
            if object_.is_struct
            else "@_holoscan_dataclass(slots=True, kw_only=True)"
        )
        classes += (
            f"{decorator}\n"
            f"class {name}:\n"
            f'    """Python value for ``{object_.name}``."""\n\n'
            f"{class_fields}\n"
            f"{root_members}"
            f"{post_init}"
            "    def _holoscan_schema_values(self) -> dict[str, _Any]:\n"
            "        return {\n"
            f"{to_values}"
            "        }\n\n"
            "    @classmethod\n"
            f"    def _from_holoscan_schema(cls, values: _Mapping[str, _Any]) -> {name}:\n"
            "        known: dict[str, _Any] = {}\n"
            f"{from_values}"
            "        return cls(**known)\n\n\n"
        )
    exports = ", ".join(repr(item) for item in names)
    registration_import = ""
    if not args.python_values_only:
        assert args.python_module_name is not None
        registration_import = (
            f'_importlib.import_module(f"{{__package__}}.{args.python_module_name}" '
            f'if __package__ else "{args.python_module_name}")\n\n'
        )
    registry_import = ""
    registry_call = ""
    if model.root_index is not None:
        registry_import = (
            "from holoscan.payload.schema import _register_type as _holoscan_register_type\n\n"
        )
        registry_call = (
            f"_holoscan_register_type({python_name(model.objects[model.root_index].name)})\n\n"
        )
    constants = "".join(f"{name} = {value}\n" for name, value in args.python_constants)
    if constants:
        constants += "\n"
    return (
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
        "All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n"
        "# Generated by holoscan_add_flatbuffer_schema(PYTHON). Do not edit.\n\n"
        '"""Generated Python values for a Holoscan FlatBuffers schema."""\n\n'
        "from __future__ import annotations\n\n"
        "import builtins as _builtins\n"
        "import importlib as _importlib\n"
        "from dataclasses import dataclass as _holoscan_dataclass\n"
        "from dataclasses import field as _holoscan_field\n"
        "from types import MappingProxyType as _MappingProxyType\n"
        "from typing import Any as _Any\n"
        "from typing import ClassVar as _ClassVar\n"
        "from typing import Mapping as _Mapping\n\n"
        "from enum import IntEnum as _IntEnum\n"
        "from enum import IntFlag as _IntFlag\n\n"
        f"{registry_import}"
        "class _HoloscanIntEnum(_IntEnum):\n"
        "    @classmethod\n"
        "    def _missing_(cls, value: object) -> _HoloscanIntEnum | None:\n"
        "        if not isinstance(value, int):\n"
        "            return None\n"
        "        member = int.__new__(cls, value)\n"
        '        member._name_ = f"UNKNOWN_{value}"\n'
        "        member._value_ = value\n"
        "        cls._value2member_map_[value] = member\n"
        "        return member\n\n\n"
        "def _holoscan_enum_value(value: _Any, name: str) -> int:\n"
        "    if not isinstance(value, int) or isinstance(value, bool):\n"
        '        raise TypeError(f"{name} requires an integer enum value")\n'
        "    return int(value)\n\n\n"
        "def _holoscan_enum_values(value: _Any, name: str) -> list[int]:\n"
        "    return [\n"
        '        _holoscan_enum_value(item, f"{name}[{index}]")\n'
        "        for index, item in enumerate(value)\n"
        "    ]\n\n\n"
        "def _holoscan_fixed_array(\n"
        "    value: _Any, length: int, name: str, element_type: type, allow_integer: bool\n"
        ") -> tuple[_Any, ...]:\n"
        "    converted = tuple(value)\n"
        "    if len(converted) != length:\n"
        '        raise ValueError(f"{name} requires exactly {length} elements")\n'
        "    enum_type = issubclass(element_type, (_IntEnum, _IntFlag))\n"
        "    normalized: list[_Any] = []\n"
        "    for index, element in enumerate(converted):\n"
        "        valid = isinstance(element, element_type)\n"
        "        if element_type is int and isinstance(element, bool):\n"
        "            valid = False\n"
        "        if enum_type:\n"
        "            valid = isinstance(element, int) and not isinstance(element, bool)\n"
        "        if allow_integer:\n"
        "            valid = isinstance(element, (int, float)) and not isinstance(element, bool)\n"
        "        if not valid:\n"
        '            raise TypeError(f"{name}[{index}] requires {element_type.__name__}")\n'
        "        normalized.append(element_type(element) if enum_type else element)\n"
        "    return tuple(normalized)\n\n\n"
        f"{imports}\n"
        f"{constants}"
        f"{registration_import}"
        f"{enums}"
        f"{classes}"
        f"{registry_call}"
        f"__all__ = [{exports}]\n"
    )


def python_schema_stub(args: Args, model: Model) -> str:
    indices, external = python_local_objects(args, model)
    enum_indices = reachable_python_enums(args, model, indices, external)
    exports = [python_name(model.enums[index].name) for index in enum_indices]
    exports.extend(python_name(model.objects[index].name) for index in indices)
    exports.extend(name for name, _ in args.python_constants)
    validate_python_bound_names(exports, external)
    imports = "".join(
        f"from {module} import {python_name(name)}\n" for name, module in external.items()
    )
    enums = ""
    for index in enum_indices:
        enum = model.enums[index]
        base = "IntFlag" if enum.bit_flags else "IntEnum"
        member_names = [python_enum_member_name(value.name) for value in enum.values]
        if len(member_names) != len(set(member_names)):
            raise GenerationError(f"{enum.name}: Python enum names collide after escaping")
        members = "".join(
            f"    {python_enum_member_name(value.name)} = ...\n" for value in enum.values
        )
        enums += f"class {python_name(enum.name)}({base}):\n{members}\n"
    classes = ""
    for index in indices:
        object_ = model.objects[index]
        name = python_name(object_.name)
        fields = canonical_python_fields(object_)
        attributes = "".join(
            f"    {python_field_name(field.name)}: "
            f"{python_annotation(model, field, containing_struct=object_.is_struct)}\n"
            for field in fields
        )
        parameters = ", ".join(
            f"{python_field_name(field.name)}: "
            f"{python_annotation(model, field, containing_struct=object_.is_struct)} = ..."
            for field in fields
        )
        members = "    __match_args__: ClassVar[tuple[str, ...]]\n"
        if not object_.is_struct:
            members += "    __hash__: ClassVar[None]  # type: ignore[assignment]\n"
        if index == model.root_index:
            members += "    _registry_name: ClassVar[str]\n"
            if any(
                field.type.base == "Obj"
                and model.objects[field.type.index].name == "holoscan.schema.Tensor"
                for field in fields
            ):
                members += "    @property\n    def tensors(self) -> Mapping[str, Any]: ...\n"
        if any(field.type.base == "Array" for field in fields):
            members += "    def __post_init__(self) -> None: ...\n"
        classes += (
            f"class {name}:\n"
            f"{attributes}"
            f"{members}"
            f"    def __init__(self{', *, ' + parameters if parameters else ''}) -> None: ...\n"
            "    def _holoscan_schema_values(self) -> dict[str, Any]: ...\n"
            "    @classmethod\n"
            f"    def _from_holoscan_schema(cls, values: Mapping[str, Any]) -> {name}: ...\n\n"
        )
    constants = "".join(f"{name}: Final[int] = {value}\n" for name, value in args.python_constants)
    if constants:
        constants += "\n"
    return (
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
        "All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n"
        "# Generated by holoscan_add_flatbuffer_schema(PYTHON). Do not edit.\n\n"
        "from enum import IntEnum, IntFlag\n"
        "from typing import Any, Any as _Any, ClassVar, Final, Mapping\n"
        f"{imports}\n"
        f"{constants}{enums}{classes}"
        f"__all__ = {exports!r}\n"
    )


def python_registration_source(args: Args, model: Model) -> str:
    assert args.python_module_name is not None
    assert args.traits_include is not None
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.python_module_name) is None:
        raise GenerationError("Python schema codec module name is not a valid identifier")
    root = cpp_type(model.root_name)
    native = f"{root}T"
    return f'''/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Generated by holoscan_add_flatbuffer_schema(PYTHON). Do not edit.
 */

#include <Python.h>

#include <holoscan/authoring/register_payload.hpp>

#include "{cpp_string(args.traits_include)}"

{cpp_python_adapter(model)}
namespace {{

PyModuleDef kModule{{
    PyModuleDef_HEAD_INIT,
    "{args.python_module_name}",
    "Register one generated Holoscan schema payload adapter.",
    -1,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
}};

}}  // namespace

// Python's extension-module ABI requires the reserved PyInit_ spelling.
// NOLINTNEXTLINE(bugprone-reserved-identifier)
PyMODINIT_FUNC PyInit_{args.python_module_name}() {{
  constexpr auto identity = holoscan::schema_identity<{native}>();
  auto row = holoscan::authoring::detail::make_registered_payload<{native}>(
      identity.canonical_name);
  auto& registry = holoscan::authoring::authoring_registry();
  const auto existing = registry.payload_type(identity.canonical_name);
  const bool facade_already_installed =
      existing && existing->get().generated_schema_facade &&
      holoscan::authoring::payload_identities_equal(existing->get().identity, row.identity);
  if (!facade_already_installed) {{
    auto installed = registry.install_payload_facade(std::move(row));
    if (!installed) {{
      const auto message = holoscan::runtime_error_message(installed.error());
      PyErr_SetString(PyExc_RuntimeError, message.c_str());
      return nullptr;
    }}
  }}
  return PyModule_Create(&kModule);
}}
'''


def publish(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") == contents:
        return
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    try:
        args = arguments()
        if isinstance(args, DependencyArgs):
            write_dependencies(args)
            return 0
        reflected, flatc_build = reflect(args)
        model = model_from(
            reflected, args.root_logical_path, allow_rootless=args.python_values_only
        )
        if args.python_values_only:
            assert args.python_output is not None
            assert args.python_stub_output is not None
            publish(args.python_output, python_schema_module(args, model))
            publish(args.python_stub_output, python_schema_stub(args, model))
            return 0
        if has_direct_tensor(model):
            contents = generate_descriptor(args, model)
        else:
            if args.descriptor_tensor_custom_codec:
                raise GenerationError(
                    "--descriptor-tensor-custom-codec requires a root with a direct "
                    "holoscan.schema.Tensor field"
                )
            reject_tensor(model)
            contents = ORDINARY.substitute(substitutions(args, model))
        if args.python_output is not None:
            assert args.python_registration_output is not None
            assert args.python_stub_output is not None
            publish(args.python_output, python_schema_module(args, model))
            publish(args.python_stub_output, python_schema_stub(args, model))
            publish(
                args.python_registration_output,
                python_registration_source(args, model),
            )
        assert args.output is not None
        publish(args.output, contents)
        write_provenance(args, flatc_build)
    except (GenerationError, json.JSONDecodeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
