#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate Holoscan FlatBuffers bindings from one reflected schema model."""

from __future__ import annotations

import argparse
import hashlib
import json
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
VERSION = re.compile(r"([0-9]+)\.([0-9]+)\.([0-9]+)")


class GenerationError(RuntimeError):
    """A deterministic schema-generation failure."""


@dataclass(frozen=True)
class Args:
    flatc: Path
    expected_flatc_version: str
    reflection_schema: Path
    bfbs: Path
    output: Path
    root_generated_header: str
    bfbs_embed_header: str | None
    compatibility_epoch: int
    max_serialized_size: int
    descriptor_tensor_custom_codec: bool
    provenance_output: Path | None
    root_source: Path
    include_dirs: tuple[Path, ...]
    flatc_flags: tuple[str, ...]

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


@dataclass(frozen=True)
class Field:
    name: str
    field_id: int | None
    deprecated: bool
    required: bool
    default_integer: int | str
    default_real: float | int | str
    type: ReflectedType


@dataclass(frozen=True)
class Object:
    name: str
    is_struct: bool
    minimum_alignment: int
    fields: tuple[Field, ...]


@dataclass(frozen=True)
class EnumValue:
    union_type: ReflectedType | None


@dataclass(frozen=True)
class Enum:
    values: tuple[EnumValue, ...]


@dataclass(frozen=True)
class Model:
    root_name: str
    file_identifier: str
    root_index: int
    objects: tuple[Object, ...]
    enums: tuple[Enum, ...]

    @property
    def root(self) -> Object:
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
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--root-generated-header", required=True)
    generate.add_argument("--bfbs-embed-header")
    generate.add_argument("--compatibility-epoch", type=positive_integer, required=True)
    generate.add_argument("--max-serialized-size", type=positive_integer, required=True)
    generate.add_argument("--descriptor-tensor-custom-codec", action="store_true")
    generate.add_argument("--provenance-output", type=Path)
    generate.add_argument("--root-source", type=Path, required=True)
    generate.add_argument("--include-dir", action="append", type=Path, default=[])
    generate.add_argument("--flatc-flag", action="append", default=[])

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
        return json.loads(reflected.read_text(encoding="utf-8")), flatc_build


def reflected_type_from(data: dict[str, Any]) -> ReflectedType:
    return ReflectedType(
        base=data["base_type"],
        element=data["element"],
        index=int(data.get("index", -1)),
    )


def field_from(data: dict[str, Any]) -> Field:
    field_id = data.get("id")
    default_integer = data["default_integer"]
    default_real = data["default_real"]
    if not isinstance(default_integer, (int, str)):
        raise GenerationError("reflected integer default must be an integer or string")
    if not isinstance(default_real, (float, int, str)):
        raise GenerationError("reflected real default must be a number or string")
    return Field(
        name=data["name"],
        field_id=field_id if isinstance(field_id, int) else None,
        deprecated=data["deprecated"],
        required=data["required"],
        default_integer=default_integer,
        default_real=default_real,
        type=reflected_type_from(data["type"]),
    )


def object_from(data: dict[str, Any]) -> Object:
    return Object(
        name=data["name"],
        is_struct=data["is_struct"],
        minimum_alignment=int(data["minalign"]),
        fields=tuple(field_from(field) for field in data["fields"]),
    )


def enum_from(data: dict[str, Any]) -> Enum:
    return Enum(
        values=tuple(
            EnumValue(
                union_type=(
                    reflected_type_from(value["union_type"])
                    if value.get("union_type") is not None
                    else None
                )
            )
            for value in data["values"]
        )
    )


def model_from(data: dict[str, Any], path: str) -> Model:
    try:
        root_name = data["root_table"]["name"]
        identifier = data["file_ident"]
        objects = tuple(object_from(object_) for object_ in data["objects"])
        enums = tuple(enum_from(enum) for enum in data["enums"])
    except (KeyError, TypeError, ValueError) as error:
        raise GenerationError("Generated binary schema reflection is incomplete") from error
    if not objects:
        raise GenerationError("Generated binary schema reflection is incomplete")
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
        nested_path = f"{path}.{name}"
        native = f"{native_prefix}{name}"
        type_ = field.type
        base, index = type_.base, type_.index
        validate_field(field, nested_path, nested=True)
        if base in SCALARS or base in {"String", "Vector"}:
            merge_fragments(
                result,
                value_field_codec(
                    field,
                    native,
                    f"{root_prefix}{name}",
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
                    f"      if (const auto* nested_field = {root_prefix}{name}(); "
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
                    f"      if (const auto* {variable} = {root_prefix}{name}(); "
                    f"{variable} != nullptr) {{\n{child.validate}      }}\n"
                )
                result.assign += (
                    f"      if (const auto* {variable} = {root_prefix}{name}(); "
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


def vector_expression(element: str, index: int, name: str, path: str) -> str:
    if element == "String":
        return f"builder.CreateVectorOfStrings(value.{name})"
    if index < 0:
        return f"builder.CreateVector(value.{name})"
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
        f"flatbuffers::data(value.{name}), value.{name}.size())"
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
            f"static_assert(std::same_as<decltype({native}::{field.name}),\n"
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
                    f"value.{name}",
                    f"root.{name}",
                    path,
                    indent=4,
                    local="field",
                    nested=False,
                ),
            )
            if base == "String":
                encode_offsets += (
                    f"    const auto _{name} = value.{name}.empty()\n"
                    "        ? flatbuffers::Offset<flatbuffers::String>{}\n"
                    f"        : builder.CreateString(value.{name});\n"
                )
            elif base == "Vector":
                expression = vector_expression(element, index, name, path)
                encode_offsets += (
                    f"    using _{name}_offset_type = decltype({expression});\n"
                    f"    const _{name}_offset_type _{name} = value.{name}.empty()\n"
                    f"        ? _{name}_offset_type{{}}\n"
                    f"        : {expression};\n"
                )
        elif base == "Obj":
            assert object_ is not None
            assertions += (
                f"static_assert(std::same_as<decltype({native}::{name}),\n"
                f"                           std::shared_ptr<{referenced_native}>>,\n"
                f'              "{cpp_string(path)}: generated native object member must use '
                'the configured std::shared_ptr form");\n'
            )
            if object_.is_struct:
                codec.reset += (
                    f"    if (value.{name}) {{\n"
                    f"      *value.{name} = {referenced_native}{{}};\n"
                    "    }\n"
                )
                encode_offsets += f"    const auto* _{name} = value.{name}.get();\n"
                codec.assign += (
                    f"    if (const auto* field = root.{name}(); field != nullptr) {{\n"
                    f"      if (!value.{name}) {{\n"
                    f"        value.{name} = std::make_shared<{referenced_native}>();\n"
                    "      }\n"
                    f"      *value.{name} = *field;\n"
                    "    } else {\n"
                    f"      value.{name}.reset();\n"
                    "    }\n"
                )
            else:
                nested = nested_codec(
                    model,
                    index,
                    f"value.{name}->",
                    "nested_root->",
                    path,
                    (model.root_index,),
                )
                codec.prepare += f"    if (value.{name}) {{\n{nested.prepare}    }}\n"
                codec.reset += f"    if (value.{name}) {{\n{nested.reset}    }}\n"
                codec.validate += (
                    f"    if (const auto* nested_root = root.{name}(); "
                    f"nested_root != nullptr) {{\n{nested.validate}    }}\n"
                )
                encode_offsets += (
                    f"    const auto _{name} = value.{name}\n"
                    f"        ? {referenced}::Pack(builder, value.{name}.get())\n"
                    f"        : flatbuffers::Offset<{referenced}>{{}};\n"
                )
                codec.assign += (
                    f"    if (const auto* nested_root = root.{name}(); "
                    "nested_root != nullptr) {\n"
                    f"      if (!value.{name}) {{\n"
                    f"        value.{name} = std::make_shared<{referenced_native}>();\n"
                    "      }\n"
                    f"{nested.assign}"
                    "    } else {\n"
                    f"      value.{name}.reset();\n"
                    "    }\n"
                )
        else:
            raise GenerationError(
                f"{path}: object-API base type '{base}' is unsupported "
                "by descriptor traits generation; provide a hand-written standard codec"
            )
        encode_adds += (
            f"    root_builder.add_{name}(value.{name});\n"
            if base in SCALARS and not is_tensor
            else f"    root_builder.add_{name}(_{name});\n"
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
        f"holoscan::TensorFieldSelector<{native}> {name}"
        f"{{{field_id}U, {index}U}};\n"
        for index, (name, field_id) in enumerate(zip(names, ids, strict=True))
    )
    declaration = f"struct {short.removesuffix('T')}TensorFields {{\n{entries}}};\n"
    if namespace is None:
        return declaration + "\n"
    return f"namespace {namespace} {{\n\n{declaration}\n}}  // namespace {namespace}\n\n"


def native_accessor(native: str, names: list[str]) -> str:
    cases = "".join(
        f"      case {index}U:\n        return value.{name}.get();\n"
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
  static constexpr std::size_t tensor_count = ${tensor_count}U;

$native_accessor
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
        f"                                            &{root}::{name}" for name in names
    )
    members = ",\n".join(
        f"                                                            &{native}::{name}"
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
        selector=selector(native, names, ids),
        tensor_accessors=accessors,
        assertions=assertions,
        native_accessor=native_accessor(native, names),
        runtime=runtime,
    )


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
        model = model_from(reflected, args.root_logical_path)
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
        publish(args.output, contents)
        write_provenance(args, flatc_build)
    except (GenerationError, json.JSONDecodeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
