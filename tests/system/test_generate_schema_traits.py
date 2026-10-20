#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the FlatBuffers traits host generator."""

# REQ: SCHEMA-006
# REQ: SCHEMA-007
# REQ: SCHEMA-008
# This standalone CTest uses only the standard library, not a pytest installation.
# ruff: noqa: PT027

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import typing
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

FLATC_FLAGS = (
    "--require-explicit-ids",
    "--bfbs-builtins",
    "--cpp",
    "--cpp-ptr-type",
    "std::shared_ptr",
    "--gen-object-api",
    "--gen-name-strings",
)


@contextmanager
def generated_module_environment(
    module_name: str, module: types.ModuleType, native_module_name: str
) -> Iterator[list[type[object]]]:
    """Supply and restore the imports required by one generated facade module."""
    module_names = (
        module_name,
        native_module_name,
        "holoscan",
        "holoscan.payload",
        "holoscan.payload.schema",
    )
    missing = object()
    previous_modules = {name: sys.modules.get(name, missing) for name in module_names}
    registered: list[type[object]] = []
    holoscan_module = types.ModuleType("holoscan")
    holoscan_module.__path__ = []
    payload_module = types.ModuleType("holoscan.payload")
    payload_module.__path__ = []
    schema_module = types.ModuleType("holoscan.payload.schema")
    schema_module._register_type = registered.append
    holoscan_module.payload = payload_module
    payload_module.schema = schema_module
    sys.modules.update(
        {
            module_name: module,
            native_module_name: types.ModuleType(native_module_name),
            "holoscan": holoscan_module,
            "holoscan.payload": payload_module,
            "holoscan.payload.schema": schema_module,
        }
    )
    try:
        yield registered
    finally:
        for name, previous in previous_modules.items():
            if previous is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def flatc_flag_arguments(
    flags: tuple[str, ...] = FLATC_FLAGS, *, builtins: bool = True
) -> list[str]:
    return [f"--flatc-flag={flag}" for flag in flags if builtins or flag != "--bfbs-builtins"]


class SchemaTraitsGeneratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generator = Path(os.environ["HOLOSCAN_SCHEMA_TRAITS_GENERATOR"])
        cls.flatc = Path(os.environ["HOLOSCAN_TEST_FLATC"])
        cls.flatc_version = os.environ["HOLOSCAN_TEST_FLATC_VERSION"]
        cls.reflection_schema = Path(os.environ["HOLOSCAN_TEST_REFLECTION_SCHEMA"])
        cls.sdk_schema_dir = Path(os.environ["HOLOSCAN_TEST_SCHEMA_DIR"])
        cls.diagnostics_dir = Path(os.environ["HOLOSCAN_TEST_SCHEMA_DIAGNOSTICS_DIR"])
        specification = importlib.util.spec_from_file_location(
            "schema_traits_generator", cls.generator
        )
        assert specification is not None
        assert specification.loader is not None
        cls.generator_module = importlib.util.module_from_spec(specification)
        sys.modules[specification.name] = cls.generator_module
        specification.loader.exec_module(cls.generator_module)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="holoscan-schema-generator-")
        self.directory = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write(self, name: str, contents: str) -> Path:
        path = self.directory / name
        path.write_text(contents, encoding="utf-8")
        return path

    def compile_bfbs(
        self, schema: Path, *, builtins: bool = True
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(self.flatc),
                "--binary",
                "--schema",
                *(["--bfbs-builtins"] if builtins else []),
                "--require-explicit-ids",
                "--bfbs-filenames",
                str(self.directory),
                "-o",
                str(self.directory),
                "-I",
                str(self.directory),
                "-I",
                str(self.sdk_schema_dir),
                str(schema),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def bfbs(self, schema: Path, *, builtins: bool = True) -> Path:
        result = self.compile_bfbs(schema, builtins=builtins)
        assert result.returncode == 0, result.stderr
        return self.directory / f"{schema.stem}.bfbs"

    def native_model(self, schema: Path):
        module = self.generator_module
        arguments = module.Args(
            flatc=self.flatc,
            expected_flatc_version=self.flatc_version,
            reflection_schema=self.reflection_schema,
            bfbs=self.bfbs(schema, builtins=True),
            output=self.directory / "unused.hpp",
            root_generated_header=f"{schema.stem}_generated.h",
            bfbs_embed_header=None,
            compatibility_epoch=1,
            max_serialized_size=512,
            descriptor_tensor_custom_codec=False,
            provenance_output=None,
            root_source=schema,
            include_dirs=(self.directory, self.sdk_schema_dir),
            flatc_flags=FLATC_FLAGS,
            python_output=None,
            python_stub_output=None,
            python_registration_output=None,
            python_module_name=None,
            traits_include=None,
            python_values_only=False,
            python_external_types=(),
            python_constants=(),
        )
        reflected, _ = module.reflect(arguments)
        return module.model_from(reflected, schema.name)

    def native_policy(self, model, fields, budget=65536):
        policy = {
            "version": 1,
            "schemas": {model.root_name: {"max_prepared_bytes": budget, "fields": fields}},
        }
        path = self.write("native-bounds.json", json.dumps(policy))
        return self.generator_module.load_native_preparation_policy(path, model)

    def test_native_policy_inventories_imu_inline_and_optional_structs(self) -> None:
        schema = self.generator.parent.parent / "src/sensor_io/schemas/imu.fbs"
        model = self.native_model(schema)
        policy = self.native_policy(model, {"header.frame_id": {"max_bytes": 128}})
        assert policy.max_prepared_bytes == 65536
        assert policy.tensor_fields == ()
        assert policy.optional_owners == (
            "header",
            "header.stamp",
            "orientation_covariance",
            "angular_velocity_covariance",
            "linear_acceleration_covariance",
        )
        assert [(field.path, field.kind, field.limit) for field in policy.fields] == [
            ("header.frame_id", "max_bytes", 128)
        ]
        inline = next(field for field in model.root.fields if field.name == "orientation")
        assert "native_inline" in dict(inline.attributes)

    def test_native_policy_inventories_descriptor_descendants_and_string_elements(self) -> None:
        schema = (
            self.generator.parent.parent / "benchmark/runtime/runtime_native_schema_benchmark.fbs"
        )
        model = self.native_model(schema)
        fields = {
            "label": {"max_bytes": 128},
            "readings": {"max_elements": 8},
            "tags": {"max_elements": 4},
            "tags[]": {"max_bytes": 128},
            "metadata.label": {"max_bytes": 128},
            "metadata.tags": {"max_elements": 3},
            "metadata.tags[]": {"max_bytes": 128},
            "metadata.child.label": {"max_bytes": 128},
        }
        policy = self.native_policy(model, fields)
        assert policy.optional_owners == (
            "metadata",
            "metadata.calibration",
            "metadata.child",
            "calibration",
        )
        assert policy.tensor_fields == ("first", "second")
        assert {field.path for field in policy.fields} == fields.keys()
        # Authored paths apply independently, even when the referenced object type is shared.
        reused = self.write(
            "reused.fbs",
            """namespace example;
table Child { text:string (id: 0); }
table Root { first:Child (id: 0); second:Child (id: 1); }
root_type Root;
file_identifier "REUS";
""",
        )
        repeated = self.native_policy(
            self.native_model(reused),
            {
                "first.text": {"max_bytes": 0},
                "second.text": {"max_bytes": 256},
            },
        )
        assert repeated.optional_owners == ("first", "second")
        assert [field.limit for field in repeated.fields] == [0, 256]

    def test_native_policy_scalar_optional_and_zero_container_bounds(self) -> None:
        schema = self.write(
            "optional.fbs",
            """namespace example;
table Root { counter:long = null (id: 0); text:string (id: 1); tags:[string] (id: 2); }
root_type Root;
file_identifier "OPTL";
""",
        )
        model = self.native_model(schema)
        assert next(field for field in model.root.fields if field.name == "counter").optional
        policy = self.native_policy(
            model,
            {
                "text": {"max_bytes": 0},
                "tags": {"max_elements": 0},
                "tags[]": {"max_bytes": 0},
            },
        )
        assert policy.optional_owners == ()
        assert all(field.limit == 0 for field in policy.fields)
        scalar = self.write(
            "scalar.fbs",
            """namespace example;
table Scalar { value:long (id: 0); }
root_type Scalar;
file_identifier "SCLR";
""",
        )
        assert self.native_policy(self.native_model(scalar), {}).fields == ()

    def test_native_policy_rejects_missing_unknown_wrong_and_unbounded_limits(self) -> None:
        schema = self.write(
            "bounded.fbs",
            """namespace example;
table Root { text:string (id: 0); tags:[string] (id: 1); }
root_type Root;
file_identifier "BNDS";
""",
        )
        model = self.native_model(schema)
        good = {
            "text": {"max_bytes": 128},
            "tags": {"max_elements": 4},
            "tags[]": {"max_bytes": 128},
        }
        for invalid in (-1, 1.5, True, "128", None, 1 << 64):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(self.generator_module.GenerationError, "unsigned 64-bit"),
            ):
                self.native_policy(model, {**good, "text": {"max_bytes": invalid}})
        for fields, message in (
            ({}, "missing keys"),
            ({**good, "typo": {"max_bytes": 8}}, "unknown keys"),
            ({**good, "text": {"max_elements": 128}}, "missing keys"),
            ({**good, "text": {"max_bytes": (1 << 64) - 1}}, "terminator size overflows"),
            (
                {**good, "tags": {"max_elements": (1 << 64) - 1}},
                "string-vector storage size overflows",
            ),
        ):
            with (
                self.subTest(fields=fields),
                self.assertRaisesRegex(self.generator_module.GenerationError, message),
            ):
                self.native_policy(model, fields)
        for budget in (0, -1, True, 1 << 64):
            with (
                self.subTest(budget=budget),
                self.assertRaisesRegex(self.generator_module.GenerationError, "positive unsigned"),
            ):
                self.native_policy(model, good, budget)

    def test_native_policy_rejects_malformed_duplicate_and_unknown_documents(self) -> None:
        schema = self.write(
            "plain.fbs",
            """namespace example;
table Root { value:long (id: 0); }
root_type Root;
file_identifier "PLAI";
""",
        )
        model = self.native_model(schema)
        for contents, message in (
            ("{", "invalid native bounds JSON"),
            ("[]", "expected a JSON object"),
            ('{"version":1,"version":1,"schemas":{}}', "duplicate JSON key"),
            ('{"version":true,"schemas":{}}', "unsupported native bounds version"),
            ('{"version":1,"schemas":{}}', "missing policy"),
            (
                '{"version":1,"schemas":{"example.Root":{},"example.Typo":{}}}',
                "unknown schema roots",
            ),
            ('{"version":1,"schemas":{"example.Root":{"fields":{}}}}', "missing keys"),
            (
                '{"version":1,"schemas":{"example.Root":{"fields":{},"max_prepared_bytes":8,"extra":0}}}',
                "unknown keys",
            ),
        ):
            with self.subTest(contents=contents):
                path = self.write("invalid-bounds.json", contents)
                with self.assertRaisesRegex(self.generator_module.GenerationError, message):
                    self.generator_module.load_native_preparation_policy(path, model)

    def test_native_inventory_rejects_unsupported_shapes_with_field_paths(self) -> None:
        for fixture, path, message in (
            ("recursive_table.fbs", "node.child", "recursive objects"),
            ("object_vector.fbs", "", "only scalar/string vectors"),
            ("union.fbs", "", "unions"),
            ("nested_tensor.fbs", "", "direct root members"),
            ("required_string.fbs", "", "deprecated/required"),
        ):
            with self.subTest(fixture=fixture):
                model = self.native_model(self.diagnostics_dir / fixture)
                with self.assertRaises(self.generator_module.GenerationError) as raised:
                    self.generator_module.native_preparation_inventory(model)
                assert message in str(raised.exception)
                assert model.root_name + ":" in str(raised.exception)
                assert path in str(raised.exception)

    def test_native_inventory_rejects_custom_native_owners_and_excessive_depth(self) -> None:
        schema = self.write(
            "custom.fbs",
            """namespace example;
table Foreign (native_type: "custom::Foreign") { value:int (id: 0); }
table Root { foreign:Foreign (id: 0); }
root_type Root;
file_identifier "CUST";
""",
        )
        model = self.native_model(schema)
        with self.assertRaisesRegex(
            self.generator_module.GenerationError, "Root:foreign: custom native"
        ):
            self.generator_module.native_preparation_inventory(model)
        tables = [f"table Node{i} {{ child:Node{i + 1} (id: 0); }}" for i in range(65)]
        tables.append("table Node65 { value:int (id: 0); }")
        deep = self.write(
            "deep.fbs",
            "\n".join(
                ["namespace example;", *tables, "root_type Node0;", 'file_identifier "DEEP";']
            ),
        )
        model = self.native_model(deep)
        with self.assertRaisesRegex(self.generator_module.GenerationError, "nesting exceeds 64"):
            self.generator_module.native_preparation_inventory(model)

    def generate(
        self,
        schema: Path,
        *options: str,
        flatc_flags: tuple[str, ...] = FLATC_FLAGS,
        builtins: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        output = self.directory / f"{schema.stem}_schema_traits.hpp"
        generated_options = list(options)
        if (
            "--python-output" in generated_options
            and "--python-stub-output" not in generated_options
        ):
            generated_options.extend(
                ("--python-stub-output", str(self.directory / f"{schema.stem}.pyi"))
            )
        return subprocess.run(
            [
                sys.executable,
                str(self.generator),
                "generate",
                "--flatc",
                str(self.flatc),
                "--expected-flatc-version",
                self.flatc_version,
                "--reflection-schema",
                str(self.reflection_schema),
                "--bfbs",
                str(self.bfbs(schema, builtins=builtins)),
                "--output",
                str(output),
                "--root-source",
                str(schema),
                "--root-generated-header",
                f"{schema.stem}_generated.h",
                "--compatibility-epoch",
                "1",
                "--max-serialized-size",
                "512",
                "--include-dir",
                str(self.directory),
                "--include-dir",
                str(self.sdk_schema_dir),
                *flatc_flag_arguments(flatc_flags, builtins=builtins),
                *generated_options,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def generate_values(self, schema: Path, *options: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(self.generator),
                "generate",
                "--flatc",
                str(self.flatc),
                "--expected-flatc-version",
                self.flatc_version,
                "--reflection-schema",
                str(self.reflection_schema),
                "--bfbs",
                str(self.bfbs(schema)),
                "--root-source",
                str(schema),
                "--include-dir",
                str(self.directory),
                "--include-dir",
                str(self.sdk_schema_dir),
                *flatc_flag_arguments(),
                "--python-values-only",
                "--python-output",
                str(self.directory / f"{schema.stem}.py"),
                "--python-stub-output",
                str(self.directory / f"{schema.stem}.pyi"),
                *options,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_prepared_opt_in_and_provenance_do_not_change_family_identity(self) -> None:
        schema = self.write(
            "bounded.fbs",
            "namespace example; table Root { text:string (id: 0); }\n"
            'root_type Root; file_identifier "BNDS";\n',
        )
        output = self.directory / "bounded_schema_traits.hpp"
        result = self.generate(schema, builtins=True)
        assert result.returncode == 0, result.stderr
        ordinary = output.read_text(encoding="utf-8")
        assert "PreparedSchemaTraits" not in ordinary
        policy = self.write(
            "bounds.json",
            json.dumps(
                {
                    "version": 1,
                    "schemas": {
                        "example.Root": {
                            "max_prepared_bytes": 4096,
                            "fields": {"text": {"max_bytes": 128}},
                        }
                    },
                }
            ),
        )
        provenance = self.directory / "provenance.json"
        result = self.generate(
            schema,
            "--native-bounds-file",
            str(policy),
            "--provenance-output",
            str(provenance),
            builtins=True,
        )
        assert result.returncode == 0, result.stderr
        prepared = output.read_text(encoding="utf-8")
        assert prepared.startswith(ordinary)
        assert "PreparedSchemaTraits<example::RootT>" in prepared
        assert "RuntimePreparedSchemaTraits<example::RootT>" in prepared
        assert "static bool source_fits(const example::RootT& source)" in prepared
        assert "source.text.size() > 128ULL" in prepared
        assert "Keep this specialization in the identity/native-traits header" in prepared
        evidence = json.loads(provenance.read_text(encoding="utf-8"))
        assert evidence["native_bounds"]["path"] == "bounds.json"
        assert evidence["native_bounds"]["sha256"] == self.generator_module.sha256(policy)
        assert evidence["native_bounds"]["sha256"] not in prepared
        result = self.generate(schema, "--native-bounds-file", str(policy), builtins=False)
        assert result.returncode != 0
        assert "requires --bfbs-builtins" in result.stderr

    def test_prepared_scalar_vector_types_follow_nested_native_member_paths(self) -> None:
        schema = self.write(
            "nested.fbs",
            "namespace example; enum Mode:byte { Idle, Active }\n"
            "table Child { readings:[float] (id: 0); modes:[Mode] (id: 1); }\n"
            "table Root { child:Child (id: 0); readings:[double] (id: 1); }\n"
            'root_type Root; file_identifier "BNDS";\n',
        )
        policy = self.write(
            "bounds.json",
            json.dumps(
                {
                    "version": 1,
                    "schemas": {
                        "example.Root": {
                            "max_prepared_bytes": 4096,
                            "fields": {
                                path: {"max_elements": 4}
                                for path in ("child.readings", "child.modes", "readings")
                            },
                        }
                    },
                }
            ),
        )
        result = self.generate(schema, "--native-bounds-file", str(policy), builtins=True)
        assert result.returncode == 0, result.stderr
        prepared = (self.directory / "nested_schema_traits.hpp").read_text(encoding="utf-8")
        for member in ("child->readings", "child->modes", "readings"):
            assert f"decltype(std::declval<example::RootT&>().{member})" in prepared

    def test_prepared_custom_codec_rejected(self) -> None:
        schema = self.diagnostics_dir / "file_identifier.fbs"
        result = self.generate(
            schema,
            "--native-bounds-file",
            "unused.json",
            "--descriptor-tensor-custom-codec",
            builtins=True,
        )
        assert result.returncode != 0
        assert "not a custom codec" in result.stderr

    def test_ordinary_root_and_native_share_generated_identity(self) -> None:
        # Key setup: Generate traits for an ordinary FlatBuffers root with an object API type.
        schema = self.write(
            "control.fbs",
            """namespace example;
table Control { enabled:bool (id: 0); }
root_type Control;
file_identifier \"CTL1\";
""",
        )
        result = self.generate(schema)

        # Key assertion: Wire and native types share one stable generated schema identity.
        assert result.returncode == 0, result.stderr
        generated = (self.directory / "control_schema_traits.hpp").read_text(encoding="utf-8")
        assert "make_schema_family_identity<example::Control>" in generated
        assert "FlatBufferSchemaBinding<example::Control>" in generated
        assert "using native_type = example::ControlT;" in generated
        assert "GeneratedSchemaNativeTraits<example::ControlT>" not in generated
        assert 'make_flatbuffer_file_identifier("CTL1")' in generated

    def test_descriptor_and_custom_codec_share_tensor_inventory(self) -> None:
        # Key setup: Generate standard and custom-codec traits for one descriptor Tensor schema.
        schema = self.write(
            "packet.fbs",
            """include \"tensor.fbs\";
namespace example;
table Packet {
  sequence:ulong (id: 0);
  label:string (id: 1);
  tensor:holoscan.schema.Tensor (id: 2);
}
root_type Packet;
file_identifier \"PKT1\";
""",
        )
        result = self.generate(schema)

        # Key assertion: Both modes expose the same named Tensor inventory and their chosen codec.
        assert result.returncode == 0, result.stderr
        standard = (self.directory / "packet_schema_traits.hpp").read_text(encoding="utf-8")
        assert "tensor_field_ids" in standard
        assert "tensor_field_names" in standard
        assert '"tensor"' in standard
        assert "GeneratedDescriptorCodec" in standard
        assert "static constexpr std::size_t max_serialized_size" not in standard
        assert "static constexpr std::size_t metadata_capacity = 512U" in standard
        assert "clone_metadata" in standard
        assert "#include <new>" in standard

        result = self.generate(schema, "--descriptor-tensor-custom-codec")
        assert result.returncode == 0, result.stderr
        custom = (self.directory / "packet_schema_traits.hpp").read_text(encoding="utf-8")
        assert "tensor_field_ids" in custom
        assert "tensor_field_names" in custom
        assert "GeneratedDescriptorCodec" not in custom
        assert "#if 0" not in custom
        assert "clone_metadata" in custom

    def test_descriptor_generation_escapes_cpp_keyword_member_access(self) -> None:
        # Key setup: Generate a descriptor whose root and nested object use a flatc keyword.
        schema = self.write(
            "keyword_descriptor.fbs",
            """include "tensor.fbs";
namespace example;
table Details { class:uint (id: 0); }
table KeywordDescriptor {
  class:holoscan.schema.Tensor (id: 0);
  delete:uint (id: 1);
  details:Details (id: 2);
}
root_type KeywordDescriptor;
file_identifier "KEYD";
""",
        )
        result = self.generate(schema)

        # Key assertion: Every emitted native member expression follows flatc's escaped spelling.
        assert result.returncode == 0, result.stderr
        generated = (self.directory / "keyword_descriptor_schema_traits.hpp").read_text(
            encoding="utf-8"
        )
        assert "KeywordDescriptorT::class_" in generated
        assert "result.class_.reset();" in generated
        assert "value.delete_" in generated
        assert "root.delete_" in generated
        assert "root_builder.add_delete_(value.delete_)" in generated
        assert "value.details->class_" in generated
        assert "nested_root->class_" in generated

    def test_descriptor_tensor_names_remain_paired_with_out_of_order_declared_ids(self) -> None:
        # Key setup: Declare Tensor fields in an order different from their canonical field IDs.
        schema = self.write(
            "out_of_order.fbs",
            """include "tensor.fbs";
namespace example;
table OutOfOrder {
  pixels:holoscan.schema.Tensor (id: 3);
  sequence:ulong (id: 0);
  calibration:holoscan.schema.Tensor (id: 2);
  label:string (id: 1);
}
root_type OutOfOrder;
file_identifier "OORD";
""",
        )
        result = self.generate(schema)

        # Key assertion: Names and IDs use the same canonical field-ID ordering.
        assert result.returncode == 0, result.stderr
        generated = (self.directory / "out_of_order_schema_traits.hpp").read_text(encoding="utf-8")
        assert "tensor_field_ids{\n    2U,\n    3U," in generated
        assert 'tensor_field_names{\n    "calibration",\n    "pixels",' in generated

    def test_descriptor_schema_diagnostics_are_reported_directly(self) -> None:
        # Key setup: Generate traits for each unsupported descriptor-schema shape.
        cases = {
            "object_vector.fbs": "vectors of tables or structs are",
            "union.fbs": "FlatBuffers unions are unsupported",
            "recursive_table.fbs": "recursive tables are unsupported",
            "nested_tensor.fbs": "Tensor fields must be direct members",
            "required_string.fbs": "required FlatBuffers reference fields",
            "required_vector.fbs": "required FlatBuffers reference fields",
            "required_table.fbs": "required FlatBuffers reference fields",
            "required_tensor.fbs": "required FlatBuffers reference fields",
            "force_align.fbs": "struct alignment 16 exceeds",
        }

        # Key assertion: Every fixture fails with its specific generator diagnostic.
        for fixture, expected in cases.items():
            with self.subTest(fixture=fixture):
                result = self.generate(self.diagnostics_dir / fixture)
                assert result.returncode != 0, result.stdout
                assert expected in result.stderr, result.stderr

    def test_descriptor_schema_preserves_file_identifier(self) -> None:
        # Key setup: Generate traits for a descriptor schema with a pinned file identifier.
        schema = self.diagnostics_dir / "file_identifier.fbs"
        result = self.generate(schema)
        assert result.returncode == 0, result.stderr
        generated = (self.directory / "file_identifier_schema_traits.hpp").read_text(
            encoding="utf-8"
        )

        # Key assertion: The exact four-byte identifier reaches the generated binding.
        assert 'make_flatbuffer_file_identifier("IDTD")' in generated

    def test_descriptor_mode_is_inferred_for_a_global_namespace_root(self) -> None:
        # Key setup: Generate a descriptor Tensor schema whose root has no namespace.
        schema = self.write(
            "global_packet.fbs",
            """include "tensor.fbs";
table GlobalPacket { tensor:holoscan.schema.Tensor (id: 0); }
root_type GlobalPacket;
file_identifier "GLBL";
""",
        )
        result = self.generate(schema)
        assert result.returncode == 0, result.stderr
        generated = (self.directory / "global_packet_schema_traits.hpp").read_text(encoding="utf-8")

        # Key assertion: Descriptor mode is inferred without emitting an invalid empty namespace.
        assert "FlatBufferSchemaBinding<GlobalPacket>" in generated
        assert "namespace  {" not in generated

    def test_generation_policy_bounds_are_rejected_directly(self) -> None:
        # Key setup: Request invalid ordinary and descriptor serialized-size bounds.
        ordinary = self.write(
            "ordinary.fbs",
            """table Ordinary { value:uint (id: 0); }
root_type Ordinary;
file_identifier "ORD1";
""",
        )
        for value in ("0", "-1"):
            with self.subTest(value=value):
                result = self.generate(ordinary, "--max-serialized-size", value)
                assert result.returncode != 0, result.stdout
                assert "must be a positive integer" in result.stderr

        descriptor = self.write(
            "large_descriptor.fbs",
            """include "tensor.fbs";
table LargeDescriptor { tensor:holoscan.schema.Tensor (id: 0); }
root_type LargeDescriptor;
file_identifier "DSC1";
""",
        )
        result = self.generate(descriptor, "--max-serialized-size", "65537")

        # Key assertion: Policy-bound violations fail directly with their applicable limit.
        assert result.returncode != 0, result.stdout
        assert "exceeds the 65536-byte descriptor metadata" in result.stderr

    def test_custom_codec_requires_an_inferred_descriptor_root(self) -> None:
        # Key setup: Request a descriptor custom codec for an ordinary root.
        schema = self.write(
            "ordinary_custom.fbs",
            """table OrdinaryCustom { value:uint (id: 0); }
root_type OrdinaryCustom;
file_identifier "ORD2";
""",
        )
        result = self.generate(schema, "--descriptor-tensor-custom-codec")

        # Key assertion: Custom-codec mode requires a direct descriptor Tensor field.
        assert result.returncode != 0, result.stdout
        assert "requires a root with a direct holoscan.schema.Tensor field" in result.stderr

    def test_nested_only_tensor_requires_a_direct_descriptor_field(self) -> None:
        # Key setup: Place the only Tensor field inside a nested table.
        schema = self.write(
            "nested_only_tensor.fbs",
            """include "tensor.fbs";
namespace example;
table NestedTensor { tensor:holoscan.schema.Tensor (id: 0); }
table NestedOnlyRoot { nested:NestedTensor (id: 0); }
root_type NestedOnlyRoot;
file_identifier "NST1";
""",
        )
        result = self.generate(schema)

        # Key assertion: Nested-only Tensors do not qualify the root for descriptor mode.
        assert result.returncode != 0, result.stdout
        assert "HOLOSCAN_SCHEMA_TENSOR_REQUIRES_DIRECT_DESCRIPTOR_FIELD" in result.stderr

    def test_schema_without_file_identifier_is_rejected_directly(self) -> None:
        # Key setup: Generate traits for a root without a FlatBuffers file identifier.
        schema = self.write(
            "no_identifier.fbs",
            """namespace example;
table NoIdentifier { value:uint (id: 0); }
root_type NoIdentifier;
""",
        )
        result = self.generate(schema)

        # Key assertion: The missing wire identity is rejected directly.
        assert result.returncode != 0, result.stdout
        assert "four-byte FlatBuffers file_identifier" in result.stderr

    def test_dependency_output_uses_flatc_include_graph(self) -> None:
        # Key setup: Generate a dependency file for a schema with one included schema.
        included = self.write(
            "included.fbs",
            "namespace example; table Included { value:uint (id: 0); }\n",
        )
        schema = self.write(
            "root.fbs",
            """include \"included.fbs\";
namespace example;
table Root { included:Included (id: 0); }
root_type Root;
file_identifier \"DEP1\";
""",
        )
        depfile = self.directory / "root.d"
        result = subprocess.run(
            [
                sys.executable,
                str(self.generator),
                "dependencies",
                "--flatc",
                str(self.flatc),
                "--output",
                str(depfile),
                "--target",
                str(self.directory / "root_generated.h"),
                "--include-dir",
                str(self.directory),
                *flatc_flag_arguments(),
                "--schema",
                str(schema),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        dependencies = depfile.read_text(encoding="utf-8")

        # Key assertion: The depfile records both the root and transitive include.
        assert str(schema) in dependencies
        assert str(included) in dependencies

    # REQ: OPERATOR-057
    def test_global_pointer_options_qualify_only_supported_metadata_clone(self) -> None:
        schema = self.write(
            "pointer_options.fbs",
            'include "tensor.fbs";\nnamespace example;\n'
            "table Meta { label:string (id: 0); }\n"
            "table Packet { meta:Meta (id: 0); tensor:holoscan.schema.Tensor (id: 1); }\n"
            'root_type Packet;\nfile_identifier "PTR1";\n',
        )
        ordinary_flags = tuple(
            flag for flag in FLATC_FLAGS if flag not in {"--cpp-ptr-type", "std::shared_ptr"}
        )
        cases = (
            (("--cpp-ptr-type", "std::shared_ptr"), True),
            (("--cpp-ptr-type=std::shared_ptr",), True),
            (("--cpp-ptr-type", "BorrowedPtr"), False),
            (("--cpp-ptr-type=BorrowedPtr",), False),
            (("--cpp-ptr-type", "std::unique_ptr"), False),
            (("--cpp-ptr-type=std::unique_ptr",), False),
            (("--cpp-ptr-type",), False),
            (("--cpp-ptr-type=",), False),
            (("--cpp-ptr-type", "std::shared_ptr", "--cpp-ptr-type=BorrowedPtr"), False),
            (("--cpp-ptr-type", "BorrowedPtr", "--cpp-ptr-type=std::shared_ptr"), False),
        )
        # These flags describe the Object API profile to the traits generator. Missing/custom
        # values test its fail-closed qualification, not whether flatc accepts that C++ command.
        for pointer_flags, qualified in cases:
            for codec_options in ((), ("--descriptor-tensor-custom-codec",)):
                with self.subTest(pointer_flags=pointer_flags, codec_options=codec_options):
                    result = self.generate(
                        schema, *codec_options, flatc_flags=(*ordinary_flags, *pointer_flags)
                    )
                    assert result.returncode == 0, result.stderr
                    generated = (self.directory / "pointer_options_schema_traits.hpp").read_text()
                    assert ("clone_metadata" in generated) == qualified
                    assert "tensor_at" in generated
                    assert "tensor_field_ids" in generated
                    assert ("GeneratedDescriptorCodec" in generated) == (not codec_options)

    # REQ: OPERATOR-057
    def test_global_string_options_omit_unqualified_metadata_clone(self) -> None:
        schema = self.write(
            "string_options.fbs",
            'include "tensor.fbs";\nnamespace example;\n'
            "table Packet { label:string (id: 0); tensor:holoscan.schema.Tensor (id: 1); }\n"
            'root_type Packet;\nfile_identifier "STR1";\n',
        )
        for string_flags in (
            ("--cpp-str-type", "BorrowedString"),
            ("--cpp-str-type=BorrowedString",),
            ("--cpp-str-flex-ctor",),
        ):
            with self.subTest(string_flags=string_flags):
                result = self.generate(
                    schema,
                    "--descriptor-tensor-custom-codec",
                    flatc_flags=(*FLATC_FLAGS, *string_flags),
                )
                assert result.returncode == 0, result.stderr
                generated = (self.directory / "string_options_schema_traits.hpp").read_text()
                assert "clone_metadata" not in generated
                assert "tensor_at" in generated
                assert "tensor_field_ids" in generated

    # REQ: OPERATOR-057
    def test_custom_codec_union_omits_unqualified_metadata_clone(self) -> None:
        schema = self.diagnostics_dir / "union.fbs"
        result = self.generate(schema, "--descriptor-tensor-custom-codec")
        assert result.returncode == 0, result.stderr
        generated = (self.directory / "union_schema_traits.hpp").read_text()
        assert "clone_metadata" not in generated
        assert "tensor_at" in generated
        assert "tensor_field_ids" in generated
        assert "GeneratedDescriptorCodec" not in generated

    # REQ: OPERATOR-057
    def test_custom_tensor_containers_omit_unqualified_metadata_clone(self) -> None:
        cases = {
            "tensor_vector": "table Meta { tensors:[holoscan.schema.Tensor] (id: 0); }",
            "nested_tensor_vector": (
                "table Child { tensor:holoscan.schema.Tensor (id: 0); }\n"
                "table Meta { children:[Child] (id: 0); }"
            ),
        }
        for name, metadata in cases.items():
            with self.subTest(container=name):
                schema = self.write(
                    f"{name}.fbs",
                    'include "tensor.fbs";\nnamespace example;\n'
                    f"{metadata}\n"
                    "table Packet { meta:Meta (id: 0); tensor:holoscan.schema.Tensor (id: 1); }\n"
                    'root_type Packet;\nfile_identifier "CNT1";\n',
                )
                result = self.generate(schema, "--descriptor-tensor-custom-codec")
                assert result.returncode == 0, result.stderr
                generated = (self.directory / f"{name}_schema_traits.hpp").read_text()
                assert "clone_metadata" not in generated
                assert "tensor_at" in generated
                assert "tensor_field_ids" in generated

    # REQ: OPERATOR-057
    def test_tensor_table_array_is_rejected_by_flatc_before_traits_generation(self) -> None:
        schema = self.write(
            "tensor_array.fbs",
            'include "tensor.fbs";\nnamespace example;\n'
            "struct Meta { tensors:[holoscan.schema.Tensor:2]; }\n"
            "table Packet { meta:Meta (id: 0); tensor:holoscan.schema.Tensor (id: 1); }\n"
            'root_type Packet;\nfile_identifier "ARR1";\n',
        )
        result = self.compile_bfbs(schema)
        assert result.returncode != 0, "flatc unexpectedly admitted a fixed array of Tensor tables"
        assert result.stderr

    # REQ: OPERATOR-057
    def test_custom_nested_tensor_omits_unqualified_metadata_clone(self) -> None:
        schema = self.write(
            "nested_tensor.fbs", (self.diagnostics_dir / "nested_tensor.fbs").read_text()
        )
        result = self.generate(schema, "--descriptor-tensor-custom-codec")
        assert result.returncode == 0, result.stderr
        generated = (self.directory / "nested_tensor_schema_traits.hpp").read_text()
        assert "clone_metadata" not in generated
        assert "tensor_at" in generated

    # REQ: OPERATOR-057
    def test_direct_root_tensor_field_customization_omits_metadata_clone(self) -> None:
        for custom_codec in (False, True):
            for customized in (False, True):
                with self.subTest(custom_codec=custom_codec, customized=customized):
                    attribute = ', cpp_ptr_type: "std::shared_ptr"' if customized else ""
                    schema = self.write(
                        "root_tensor.fbs",
                        'include "tensor.fbs";\nnamespace example;\n'
                        f"table Packet {{ tensor:holoscan.schema.Tensor (id: 0{attribute}); }}\n"
                        'root_type Packet;\nfile_identifier "ROOT";\n',
                    )
                    options = ("--descriptor-tensor-custom-codec",) if custom_codec else ()
                    result = self.generate(schema, *options)
                    assert result.returncode == 0, result.stderr
                    generated = (self.directory / "root_tensor_schema_traits.hpp").read_text()
                    assert ("clone_metadata" in generated) == (not customized)
                    assert "tensor_at" in generated
                    assert "tensor_field_ids" in generated

    # REQ: OPERATOR-057
    def test_custom_native_metadata_omits_unqualified_metadata_clone(self) -> None:
        cases = {
            "native_type": 'table Meta (native_type: "BorrowedMetadata") { value:int (id: 0); }',
            "cpp_str_type": 'table Meta { label:string (id: 0, cpp_str_type: "BorrowedString"); }',
            "cpp_ptr_type": (
                "table Child { value:int (id: 0); }\n"
                'table Meta { child:Child (id: 0, cpp_ptr_type: "BorrowedPtr"); }'
            ),
            "native_inline": (
                "struct InlineMeta { value:int; }\n"
                "table Meta { value:InlineMeta (id: 0, native_inline); }"
            ),
        }
        for name, metadata in cases.items():
            with self.subTest(customization=name):
                schema = self.write(
                    f"{name}.fbs",
                    'include "tensor.fbs";\nnamespace example;\n'
                    f"{metadata}\n"
                    "table Packet { meta:Meta (id: 0); tensor:holoscan.schema.Tensor (id: 1); }\n"
                    'root_type Packet;\nfile_identifier "CSTM";\n',
                )
                result = self.generate(schema, "--descriptor-tensor-custom-codec")
                assert result.returncode == 0, result.stderr
                generated = (self.directory / f"{name}_schema_traits.hpp").read_text()
                assert "clone_metadata" not in generated
                assert "tensor_at" in generated
                assert "tensor_field_ids" in generated

    def test_provenance_records_exact_bfbs_changes_outside_runtime_identity(self) -> None:
        # Key setup: Generate provenance for wire-compatible schemas with renamed fields.
        first = self.write(
            "frame_v1.fbs",
            """namespace example;
table Frame { label:string (id: 0); }
root_type Frame;
file_identifier "FRM1";
""",
        )
        renamed = self.write(
            "frame_renamed.fbs",
            """namespace example;
table Frame { title:string (id: 0); }
root_type Frame;
file_identifier "FRM1";
""",
        )
        first_provenance = self.directory / "frame_v1.provenance.json"
        renamed_provenance = self.directory / "frame_renamed.provenance.json"

        first_result = self.generate(
            first,
            "--provenance-output",
            str(first_provenance),
            "--root-source",
            str(first),
        )
        renamed_result = self.generate(
            renamed,
            "--provenance-output",
            str(renamed_provenance),
            "--root-source",
            str(renamed),
        )
        assert first_result.returncode == 0, first_result.stderr
        assert renamed_result.returncode == 0, renamed_result.stderr

        first_evidence = json.loads(first_provenance.read_text(encoding="utf-8"))
        renamed_evidence = json.loads(renamed_provenance.read_text(encoding="utf-8"))

        # Key assertion: Provenance pins generator flags and detects exact BFBS changes.
        assert first_evidence["flatc_flags"] == list(FLATC_FLAGS)
        assert first_evidence["bfbs_sha256"] != renamed_evidence["bfbs_sha256"]

    # REQ: SCHEMA-PY-001
    def test_python_generation_emits_dataclass_adapter_and_registration(self) -> None:
        # Key setup: Opt an ordinary nested schema into all Python generation artifacts.
        schema = self.write(
            "python_control.fbs",
            """namespace example;
table Header { sequence:ulong (id: 0); }
table PythonControl {
  header:Header (id: 0);
  gain:float = 1.5 (id: 1);
  labels:[string] (id: 2);
}
root_type PythonControl;
file_identifier "PYC1";
""",
        )
        python_module = self.directory / "python_control.py"
        registration = self.directory / "python_control_registration.cpp"
        result = self.generate(
            schema,
            "--python-output",
            str(python_module),
            "--python-registration-output",
            str(registration),
            "--python-module-name",
            "_python_control_holoscan",
            "--traits-include",
            "python_control_schema_traits.hpp",
        )

        # Key assertion: One opt-in emits Python values, native conversion, and a registry module.
        assert result.returncode == 0, result.stderr
        generated = (self.directory / "python_control_schema_traits.hpp").read_text(
            encoding="utf-8"
        )
        python = python_module.read_text(encoding="utf-8")
        codec = registration.read_text(encoding="utf-8")
        assert "PythonSchemaNativeTraits<example::PythonControlT>" in codec
        assert (
            "namespace example::holoscan_generated_PythonControl_package::"
            "holoscan_python_PythonControl_detail" in codec
        )
        assert "std::make_shared<example::HeaderT>" in codec
        assert "@_holoscan_dataclass(slots=True, kw_only=True)\nclass PythonControl:" in python
        assert '_registry_name: _ClassVar[str] = "example.PythonControl"' in python
        assert "def _from_holoscan_schema" in python
        assert "PYTHON_SCHEMA_FIELD_UNKNOWN" in codec
        assert "_holoscan_register_type(PythonControl)" in python
        assert "make_registered_payload<example::PythonControlT>" in codec

        repeated = self.generate(
            schema,
            "--python-output",
            str(python_module),
            "--python-registration-output",
            str(registration),
            "--python-module-name",
            "_python_control_holoscan",
            "--traits-include",
            "python_control_schema_traits.hpp",
        )
        assert repeated.returncode == 0, repeated.stderr
        assert generated == (self.directory / "python_control_schema_traits.hpp").read_text(
            encoding="utf-8"
        )
        assert python == python_module.read_text(encoding="utf-8")
        assert codec == registration.read_text(encoding="utf-8")

    # REQ: SCHEMA-PY-003
    def test_python_descriptor_generation_exposes_tensor_mapping(self) -> None:
        # Key setup: Generate the Python facade for a descriptor schema with two Tensor fields.
        schema = self.write(
            "python_stereo.fbs",
            """include "tensor.fbs";
namespace example;
table PythonStereo {
  left:holoscan.schema.Tensor (id: 0);
  right:holoscan.schema.Tensor (id: 1);
  exposure:uint (id: 2);
}
root_type PythonStereo;
file_identifier "PYS1";
""",
        )
        python_module = self.directory / "python_stereo.py"
        registration = self.directory / "python_stereo_registration.cpp"
        result = self.generate(
            schema,
            "--python-output",
            str(python_module),
            "--python-registration-output",
            str(registration),
            "--python-module-name",
            "_python_stereo_holoscan",
            "--traits-include",
            "python_stereo_schema_traits.hpp",
        )

        # Key assertion: Tensor fields stay kwargs and expose the read-only compatibility view.
        assert result.returncode == 0, result.stderr
        python = python_module.read_text(encoding="utf-8")
        assert "left: _Any | None = None" in python
        assert "right: _Any | None = None" in python
        assert "def tensors(self) -> _Mapping[str, _Any]:" in python
        assert '"left": self.left' in python
        assert '"right": self.right' in python

    # REQ: SCHEMA-PY-002
    def test_python_materialization_ignores_unknown_fields_and_uses_defaults(self) -> None:
        # Key setup: Generate defaults for fields absent in an older wire row.
        schema = self.write(
            "evolved.fbs",
            """namespace example;
table Evolved {
  sequence:ulong (id: 0);
  label:string (id: 1);
}
root_type Evolved;
file_identifier "EVL1";
""",
        )
        python_module = self.directory / "evolved.py"
        registration = self.directory / "evolved_registration.cpp"
        result = self.generate(
            schema,
            "--python-output",
            str(python_module),
            "--python-registration-output",
            str(registration),
            "--python-module-name",
            "_evolved_holoscan",
            "--traits-include",
            "evolved_schema_traits.hpp",
        )
        assert result.returncode == 0, result.stderr
        spec = importlib.util.spec_from_file_location("evolved", python_module)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        with generated_module_environment("evolved", module, "_evolved_holoscan") as registered:
            spec.loader.exec_module(module)
            value = module.Evolved._from_holoscan_schema(
                {"sequence": 7, "producer_only": "ignored"}
            )

        # Key assertion: Unknown newer fields are ignored and absent known fields keep defaults.
        assert registered == [module.Evolved]
        assert value.sequence == 7
        assert value.label == ""

    def test_python_generation_rejects_optional_scalar_fields(self) -> None:
        # Key setup: Request Python generation for an optional FlatBuffers scalar.
        schema = self.write(
            "optional_scalar.fbs",
            """namespace example;
table OptionalScalar { level:int = null (id: 0); }
root_type OptionalScalar;
file_identifier "OPT1";
""",
        )
        result = self.generate(
            schema,
            "--python-output",
            str(self.directory / "optional_scalar.py"),
            "--python-registration-output",
            str(self.directory / "optional_scalar_registration.cpp"),
            "--python-module-name",
            "_optional_scalar_holoscan",
            "--traits-include",
            "optional_scalar_schema_traits.hpp",
        )

        # Key assertion: Unsupported optional scalars fail during generation, before C++ compile.
        assert result.returncode != 0
        assert "optional scalar fields are unsupported" in result.stderr

    def test_python_generation_escapes_keyword_field_names(self) -> None:
        # Key setup: Generate a schema containing flatc and Python keyword field names.
        schema = self.write(
            "keyword_fields.fbs",
            """namespace example;
table KeywordFields {
  class:uint (id: 0);
  from:string (id: 1);
  field:[string] (id: 2);
  import:uint (id: 3);
  module:uint (id: 4);
}
root_type KeywordFields;
file_identifier "KEY1";
""",
        )
        python_module = self.directory / "keyword_fields.py"
        result = self.generate(
            schema,
            "--python-output",
            str(python_module),
            "--python-registration-output",
            str(self.directory / "keyword_fields_registration.cpp"),
            "--python-module-name",
            "_keyword_fields_holoscan",
            "--traits-include",
            "keyword_fields_schema_traits.hpp",
        )

        # Key assertion: Native and Python spellings are escaped while canonical wire keys remain.
        assert result.returncode == 0, result.stderr
        registration = (self.directory / "keyword_fields_registration.cpp").read_text(
            encoding="utf-8"
        )
        python = python_module.read_text(encoding="utf-8")
        assert "value.class_" in registration
        assert "result.class_" in registration
        assert "value.import_" in registration
        assert "result.module_" in registration
        assert "class_: int = 0" in python
        assert "from_: str" in python
        assert '"class": self.class_' in python
        compile(python, str(python_module), "exec")

    def test_python_generation_rejects_unescaped_cpp20_keyword_field_names(self) -> None:
        # Key setup: Request adapters for names the pinned flatc leaves invalid in C++20.
        for field_name in ("char8_t", "consteval", "constinit", "reflexpr"):
            with self.subTest(field_name=field_name):
                schema = self.write(
                    f"{field_name}.fbs",
                    f"""namespace example;
table Cpp20Keyword {{ {field_name}:uint (id: 0); }}
root_type Cpp20Keyword;
file_identifier "CP20";
""",
                )
                result = self.generate(
                    schema,
                    "--python-output",
                    str(self.directory / f"{field_name}.py"),
                    "--python-registration-output",
                    str(self.directory / f"{field_name}_registration.cpp"),
                    "--python-module-name",
                    f"_{field_name}_holoscan",
                    "--traits-include",
                    f"{field_name}_schema_traits.hpp",
                )

                # Key assertion: Fail during generation rather than emitting invalid C++.
                assert result.returncode != 0
                assert "C++20 keyword that the pinned flatc does not escape" in result.stderr

    def test_python_generation_rejects_keyword_tensor_field_names(self) -> None:
        # Key setup: Request a descriptor adapter whose Tensor key cannot also be a Python kwarg.
        schema = self.write(
            "keyword_tensor.fbs",
            """include "tensor.fbs";
namespace example;
table KeywordTensor { class:holoscan.schema.Tensor (id: 0); }
root_type KeywordTensor;
file_identifier "KWT1";
""",
        )
        result = self.generate(
            schema,
            "--python-output",
            str(self.directory / "keyword_tensor.py"),
            "--python-registration-output",
            str(self.directory / "keyword_tensor_registration.cpp"),
            "--python-module-name",
            "_keyword_tensor_holoscan",
            "--traits-include",
            "keyword_tensor_schema_traits.hpp",
        )

        # Key assertion: Reject rather than advertise a broken **frame.tensors round trip.
        assert result.returncode != 0
        assert "Tensor field names that require Python keyword escaping" in result.stderr

    def test_python_generation_rejects_non_finite_float_default(self) -> None:
        # Key setup: Request Python output for a schema default that is not a Python literal.
        schema = self.write(
            "non_finite.fbs",
            """namespace example;
table NonFinite { gain:float = nan (id: 0); }
root_type NonFinite;
file_identifier "NAN1";
""",
        )
        result = self.generate(
            schema,
            "--python-output",
            str(self.directory / "non_finite.py"),
            "--python-registration-output",
            str(self.directory / "non_finite_registration.cpp"),
            "--python-module-name",
            "_non_finite_holoscan",
            "--traits-include",
            "non_finite_schema_traits.hpp",
        )

        # Key assertion: Report the unsupported default instead of emitting a Python NameError.
        assert result.returncode != 0
        assert "non-finite floating-point defaults are unsupported" in result.stderr, result.stderr

    # REQ: SCHEMA-PY-007
    def test_python_raw_byte_vectors_and_builtin_name_collision(self) -> None:
        schema = self.write(
            "byte_shapes.fbs",
            """namespace example;
// The legal table name must not shadow the builtin bytes annotation.
table bytes { blob:[ubyte] (id: 0); }
enum Mode:ubyte { Zero = 0, High = 128 }
struct Fixed { blob:[ubyte:4]; }
table ByteShapes {
  blob:[ubyte] (id: 0);
  signed_values:[byte] (id: 1);
  modes:[Mode] (id: 2);
  fixed:Fixed (id: 3);
  nested:bytes (id: 4);
}
root_type ByteShapes;
file_identifier "BYS1";
""",
        )
        python_module = self.directory / "byte_shapes.py"
        registration_path = self.directory / "byte_shapes_registration.cpp"
        arguments = (
            "--python-output",
            str(python_module),
            "--python-registration-output",
            str(registration_path),
            "--python-module-name",
            "_byte_shapes_holoscan",
            "--traits-include",
            "byte_shapes_schema_traits.hpp",
        )
        result = self.generate(schema, *arguments)
        assert result.returncode == 0, result.stderr
        registration = registration_path.read_text(encoding="utf-8")
        python = python_module.read_text(encoding="utf-8")
        stub = python_module.with_suffix(".pyi").read_text(encoding="utf-8")
        assert "ForeignSchemaValue::numeric_sequence(value.blob)" in registration
        assert "for (const auto& element : value.blob)" not in registration
        assert 'blob: _builtins.bytes = b""' in python
        assert "blob: _builtins.bytes" in stub
        assert "import builtins as _builtins" in stub
        assert "signed_values: list[int]" in python
        assert "modes: list[Mode]" in python
        assert "blob: tuple[int, ...]" in python
        spec = importlib.util.spec_from_file_location("byte_shapes", python_module)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        with generated_module_environment("byte_shapes", module, "_byte_shapes_holoscan"):
            spec.loader.exec_module(module)
            value = module.ByteShapes()
            assert value.blob == b""
            received = module.ByteShapes._from_holoscan_schema({"blob": b"\x00\x80\xff"})
            assert received.blob == b"\x00\x80\xff"
            assert typing.get_type_hints(module.ByteShapes)["blob"] is bytes
        result = self.generate(schema, *arguments)
        assert result.returncode == 0, result.stderr
        assert python_module.read_text(encoding="utf-8") == python
        assert registration_path.read_text(encoding="utf-8") == registration

    # REQ: SCHEMA-PY-004
    def test_python_generation_supports_enums_structs_arrays_and_struct_vectors(self) -> None:
        # Key setup: Exercise every reflected shape used by the production Image and Imu schemas.
        schema = self.write(
            "production_shapes.fbs",
            """namespace example;
enum Mode : byte { None = 0, ON = 1 }
enum Flags : ubyte (bit_flags) { FIRST, SECOND }
struct Matrix { values:[float:3]; }
struct Plane { size:uint32; offset:uint64; }
struct Inner { value:int; }
struct Outer { inner:Inner; count:uint32; }
struct ModeArray { values:[Mode:3]; }
table ProductionShapes {
  mode:Mode (id: 0);
  flags:Flags (id: 1);
  matrix:Matrix (id: 2);
  planes:[Plane] (id: 3);
  outer:Outer (id: 4);
  modes:[Mode] (id: 5);
  mode_array:ModeArray (id: 6);
  blob:[ubyte] (id: 7);
  byte_modes:[Flags] (id: 8);
  numbers:[double] (id: 9);
  bools:[bool] (id: 10);
}
root_type ProductionShapes;
file_identifier "PSH1";
""",
        )
        python_module = self.directory / "production_shapes.py"
        result = self.generate(
            schema,
            "--python-output",
            str(python_module),
            "--python-registration-output",
            str(self.directory / "production_shapes_registration.cpp"),
            "--python-module-name",
            "_production_shapes_holoscan",
            "--traits-include",
            "production_shapes_schema_traits.hpp",
        )
        assert result.returncode == 0, result.stderr
        registration = (self.directory / "production_shapes_registration.cpp").read_text(
            encoding="utf-8"
        )
        python = python_module.read_text(encoding="utf-8")
        stub = (self.directory / "production_shapes.pyi").read_text(encoding="utf-8")
        assert "std::array<float, 3> converted_values" in registration
        assert "flatbuffers::span<const float, 3>" in registration
        assert "result.planes.push_back" in registration
        assert "to_foreign_" in registration
        assert "value.inner()" in registration
        assert "field=example.ProductionShapes.mode" in registration
        assert "class Mode(_HoloscanIntEnum):" in python
        assert "    None_ = 0" in python
        assert "class Flags(_IntFlag):" in python
        assert "class Mode(IntEnum):\n    None_ = ...\n    ON = ..." in stub
        assert "class Flags(IntFlag):\n    FIRST = ...\n    SECOND = ..." in stub
        assert 'blob: _builtins.bytes = b""' in python
        assert "blob: _builtins.bytes" in stub
        assert "byte_modes: list[Flags]" in stub
        assert "numbers: list[float]" in stub
        assert "bools: list[bool]" in stub
        assert "__hash__: ClassVar[None]  # type: ignore[assignment]" in stub
        assert (
            "@_holoscan_dataclass(slots=True, frozen=True, kw_only=True)\nclass Matrix:" in python
        )

        spec = importlib.util.spec_from_file_location("production_shapes", python_module)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        with generated_module_environment(
            "production_shapes", module, "_production_shapes_holoscan"
        ) as registered:
            spec.loader.exec_module(module)
            assert module.Mode(7).name == "UNKNOWN_7"
            assert module.Mode.None_.value == 0
            assert module.Flags(3) == module.Flags.FIRST | module.Flags.SECOND
            assert module.Matrix(values=[1.0, 2.0, 3.0]).values == (1.0, 2.0, 3.0)
            with self.assertRaises(TypeError):  # noqa: PT027
                module.Matrix([1.0, 2.0, 3.0])
            with self.assertRaisesRegex(ValueError, "exactly 3 elements"):  # noqa: PT027
                module.Matrix(values=[1.0])
            with self.assertRaisesRegex(TypeError, r"Matrix.values\[0\] requires float"):  # noqa: PT027
                module.Matrix(values="abc")
            nested = module.Outer(inner=module.Inner(value=11), count=2)
            enum_array = module.ModeArray(values=[0, module.Mode.ON, 7])
            with self.assertRaisesRegex(TypeError, r"ModeArray.values\[0\] requires Mode"):  # noqa: PT027
                module.ModeArray(values=["OFF", 1, 7])
            value = module.ProductionShapes(
                planes=[module.Plane(offset=4, size=8)],
                outer=nested,
                modes=[module.Mode.ON, module.Mode(7)],
                mode_array=enum_array,
            )
            assert value._holoscan_schema_values()["planes"] == [{"offset": 4, "size": 8}]
            assert value._holoscan_schema_values()["outer"] == {
                "inner": {"value": 11},
                "count": 2,
            }
            assert value._holoscan_schema_values()["modes"] == [1, 7]
            restored = module.ProductionShapes._from_holoscan_schema(
                value._holoscan_schema_values()
            )
            assert restored.outer.inner.__class__ is module.Inner
            assert restored.modes == [module.Mode.ON, module.Mode(7)]
            assert restored.mode_array.values == (
                module.Mode.None_,
                module.Mode.ON,
                module.Mode(7),
            )
            for invalid in (7.9, "7", True):
                with self.assertRaisesRegex(TypeError, "requires an integer enum value"):  # noqa: PT027
                    module.ProductionShapes(mode=invalid)._holoscan_schema_values()
                with self.assertRaisesRegex(TypeError, "requires an integer enum value"):  # noqa: PT027
                    module.ProductionShapes(flags=invalid)._holoscan_schema_values()
                with self.assertRaisesRegex(TypeError, "requires an integer enum value"):  # noqa: PT027
                    module.ProductionShapes(modes=[invalid])._holoscan_schema_values()
        assert registered == [module.ProductionShapes]

    # REQ: SCHEMA-PY-004
    def test_python_generation_emits_valid_empty_table_stubs(self) -> None:
        # Key setup: Generate both an empty root table and an empty nested table.
        schemas = {
            "empty_root.fbs": """namespace example;
table Empty {}
root_type Empty;
file_identifier "EMP1";
""",
            "empty_nested.fbs": """namespace example;
table Empty {}
table Wrapper { value:Empty (id: 0); }
root_type Wrapper;
file_identifier "EMP2";
""",
        }
        for filename, contents in schemas.items():
            with self.subTest(filename=filename):
                schema = self.write(filename, contents)
                result = self.generate_values(schema)
                assert result.returncode == 0, result.stderr
                runtime = self.directory / f"{schema.stem}.py"
                stub = self.directory / f"{schema.stem}.pyi"

                # Key assertion: Runtime and typing artifacts both parse, and empty tables omit the
                # invalid bare keyword-only separator from their constructor signature.
                compile(runtime.read_text(encoding="utf-8"), str(runtime), "exec")
                stub_text = stub.read_text(encoding="utf-8")
                compile(stub_text, str(stub), "exec")
                assert "def __init__(self) -> None: ..." in stub_text

    # REQ: SCHEMA-PY-004
    def test_python_value_only_generation_has_no_payload_registration(self) -> None:
        # Key setup: Generate immutable helper values from a schema with no root_type.
        schema = self.write(
            "helpers.fbs",
            """namespace example;
struct Vector3f { x:float; y:float; z:float; }
struct Matrix3f { values:[float:9]; }
""",
        )
        result = self.generate_values(schema)
        assert result.returncode == 0, result.stderr
        python = (self.directory / "helpers.py").read_text(encoding="utf-8")
        stub = (self.directory / "helpers.pyi").read_text(encoding="utf-8")
        assert "class Vector3f:" in python
        assert "class Matrix3f:" in python
        assert "_registry_name" not in python
        assert "import_module" not in python.split("class Vector3f:", 1)[0]
        assert "class Matrix3f:" in stub
        assert not (self.directory / "helpers_registration.cpp").exists()

    # REQ: SCHEMA-PY-004
    def test_python_value_only_generation_orders_nested_children_before_parents(self) -> None:
        # Key setup: Name the parent before its nested child so reflection order is not a safe
        # proxy for the dependency order required by dataclass default factories.
        schema = self.write(
            "nested_helpers.fbs",
            """namespace example;
struct ZChild { value:int; }
struct ParentValue { child:ZChild; }
""",
        )
        result = self.generate_values(schema)
        assert result.returncode == 0, result.stderr
        python = (self.directory / "nested_helpers.py").read_text(encoding="utf-8")

        # Key assertion: The generated module can evaluate the child's default factory while
        # defining the parent, independent of FlatBuffers reflection ordering.
        assert python.index("class ZChild:") < python.index("class ParentValue:")
        namespace: dict[str, object] = {}
        exec(compile(python, "nested_helpers.py", "exec"), namespace)
        parent = namespace["ParentValue"]()
        assert type(parent.child).__name__ == "ZChild"

    # REQ: SCHEMA-PY-004
    def test_python_generation_publishes_explicit_module_constants(self) -> None:
        # Key setup: Request one public integer constant alongside generated helper values.
        schema = self.write(
            "constant_helpers.fbs",
            """namespace example;
struct Helper { value:int; }
""",
        )
        result = self.generate_values(
            schema,
            "--python-constant",
            "UNKNOWN_SAMPLE_BITS=255",
        )

        # Key assertion: Runtime and typing output agree on the public spelling and value.
        assert result.returncode == 0, result.stderr
        python = (self.directory / "constant_helpers.py").read_text(encoding="utf-8")
        stub = (self.directory / "constant_helpers.pyi").read_text(encoding="utf-8")
        assert "UNKNOWN_SAMPLE_BITS = 255" in python
        assert "UNKNOWN_SAMPLE_BITS: Final[int] = 255" in stub
        assert "'UNKNOWN_SAMPLE_BITS'" in python
        assert "'UNKNOWN_SAMPLE_BITS'" in stub

    # REQ: SCHEMA-PY-004
    def test_python_generation_requires_explicit_owner_for_included_types(self) -> None:
        # Key setup: Include object and enum declarations owned by another schema without a map.
        self.write(
            "owned.fbs",
            """namespace example;
enum OwnedMode : byte { OFF = 0, ON = 1 }
table Owned { value:int (id: 0); }
""",
        )
        schema = self.write(
            "consumer.fbs",
            """include "owned.fbs";
namespace example;
table Consumer {
  owned:Owned (id: 0);
  modes:[OwnedMode] (id: 1);
}
root_type Consumer;
file_identifier "OWN1";
""",
        )

        # Key assertion: Fail closed instead of generating a competing Python owner.
        result = self.generate(
            schema,
            "--python-output",
            str(self.directory / "consumer.py"),
            "--python-registration-output",
            str(self.directory / "consumer_registration.cpp"),
            "--python-module-name",
            "_consumer_holoscan",
            "--traits-include",
            "consumer_schema_traits.hpp",
        )
        assert result.returncode != 0
        assert "must be mapped with --python-external-type" in result.stderr

    # REQ: SCHEMA-PY-004
    def test_python_value_only_generation_rejects_included_competing_types(self) -> None:
        # Key setup: Include an externally owned declaration in a rootless helper schema.
        self.write("owned_helper.fbs", "namespace example; struct OwnedHelper { value:int; }\n")
        schema = self.write(
            "local_helper.fbs",
            """include "owned_helper.fbs";
namespace example;
struct LocalHelper { value:int; }
""",
        )

        # Key assertion: Rootless mode also requires an explicit canonical owner mapping.
        result = self.generate_values(schema)
        assert result.returncode != 0
        assert "must be mapped with --python-external-type" in result.stderr

    # REQ: SCHEMA-PY-004
    def test_python_generation_reuses_external_types_and_custom_codec_row(self) -> None:
        # Key setup: Map an included type to its owner and bind Python to a custom descriptor codec.
        self.write(
            "canonical_header.fbs",
            """namespace example;
table CanonicalHeader { sequence:ulong (id: 0); }
""",
        )
        schema = self.write(
            "custom_image.fbs",
            """include "canonical_header.fbs";
include "tensor.fbs";
namespace example;
table CustomImage {
  header:CanonicalHeader (id: 0);
  data:holoscan.schema.Tensor (id: 1);
}
root_type CustomImage;
file_identifier "CIM1";
""",
        )
        python_module = self.directory / "custom_image.py"
        registration = self.directory / "custom_image_registration.cpp"
        result = self.generate(
            schema,
            "--descriptor-tensor-custom-codec",
            "--python-output",
            str(python_module),
            "--python-registration-output",
            str(registration),
            "--python-module-name",
            "_custom_image_holoscan",
            "--traits-include",
            "custom_image_schema_traits.hpp",
            "--python-external-type",
            "example.CanonicalHeader=canonical.header",
        )
        assert result.returncode == 0, result.stderr
        generated = (self.directory / "custom_image_schema_traits.hpp").read_text(encoding="utf-8")
        python = python_module.read_text(encoding="utf-8")
        codec = registration.read_text(encoding="utf-8")
        assert "GeneratedDescriptorCodec" not in generated
        assert "PythonSchemaNativeTraits<example::CustomImageT>" in codec
        assert "from canonical.header import CanonicalHeader" in python
        assert "class CanonicalHeader:" not in python
        assert "install_payload_facade(std::move(row))" in codec

    # REQ: SCHEMA-PY-004
    def test_python_generation_rejects_imported_short_name_collisions(self) -> None:
        # Key setup: Give a local class and two external owners the same unqualified Python name.
        self.write(
            "first_header.fbs",
            "namespace first; table Header { first_value:int (id: 0); }\n",
        )
        self.write(
            "second_header.fbs",
            "namespace second; table Header { second_value:int (id: 0); }\n",
        )
        local_collision = self.write(
            "local_collision.fbs",
            """include "first_header.fbs";
namespace consumer;
table Header { value:first.Header (id: 0); }
root_type Header;
file_identifier "COL1";
""",
        )
        external_collision = self.write(
            "external_collision.fbs",
            """include "first_header.fbs";
include "second_header.fbs";
namespace consumer;
table Container {
  first_value:first.Header (id: 0);
  second_value:second.Header (id: 1);
}
root_type Container;
file_identifier "COL2";
""",
        )

        cases = (
            (local_collision, ("first.Header=owner.first",)),
            (
                external_collision,
                ("first.Header=owner.first", "second.Header=owner.second"),
            ),
        )
        for schema, mappings in cases:
            with self.subTest(schema=schema.name):
                options: list[str] = []
                for mapping in mappings:
                    options.extend(("--python-external-type", mapping))
                result = self.generate_values(schema, *options)

                # Key assertion: Generation fails before an import can overwrite a local or
                # previously imported class binding.
                assert result.returncode != 0
                assert "export and imported type names must be unique" in result.stderr

    # REQ: SCHEMA-PY-004
    def test_python_generation_rejects_stub_helper_name_collisions(self) -> None:
        # Key setup: Give local schema declarations every unaliased name imported by the stub.
        for name in ("Any", "ClassVar", "Final", "IntEnum", "IntFlag", "Mapping", "_builtins"):
            with self.subTest(name=name):
                schema = self.write(
                    f"reserved_{name.lower()}.fbs",
                    f"namespace example; table {name} {{ value:int (id: 0); }}\n"
                    f"root_type {name};\n"
                    'file_identifier "RSV1";\n',
                )
                result = self.generate_values(schema)

                # Key assertion: Generation fails before the stub helper import is shadowed.
                assert result.returncode != 0
                assert "conflict with stub helper imports" in result.stderr
                assert name in result.stderr

        # Key setup: Map an included schema type whose short name collides with a stub helper.
        self.write("external_any.fbs", "namespace external; table Any { value:int (id: 0); }\n")
        schema = self.write(
            "reserved_external.fbs",
            """include "external_any.fbs";
namespace example;
table Wrapper { value:external.Any (id: 0); }
root_type Wrapper;
file_identifier "RSV2";
""",
        )
        result = self.generate_values(
            schema,
            "--python-external-type",
            "external.Any=canonical.types",
        )

        # Key assertion: External owners are subject to the same stub-helper reservation.
        assert result.returncode != 0
        assert "conflict with stub helper imports" in result.stderr
        assert "Any" in result.stderr


if __name__ == "__main__":
    unittest.main()
