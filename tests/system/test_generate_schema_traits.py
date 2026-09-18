#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the FlatBuffers traits host generator."""

# REQ: SCHEMA-006

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

FLATC_FLAGS = (
    "--require-explicit-ids",
    "--cpp",
    "--cpp-ptr-type",
    "std::shared_ptr",
    "--gen-object-api",
    "--gen-name-strings",
)


def flatc_flag_arguments() -> list[str]:
    return [f"--flatc-flag={flag}" for flag in FLATC_FLAGS]


class SchemaTraitsGeneratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generator = Path(os.environ["HOLOSCAN_SCHEMA_TRAITS_GENERATOR"])
        cls.flatc = Path(os.environ["HOLOSCAN_TEST_FLATC"])
        cls.flatc_version = os.environ["HOLOSCAN_TEST_FLATC_VERSION"]
        cls.reflection_schema = Path(os.environ["HOLOSCAN_TEST_REFLECTION_SCHEMA"])
        cls.sdk_schema_dir = Path(os.environ["HOLOSCAN_TEST_SCHEMA_DIR"])
        cls.diagnostics_dir = Path(os.environ["HOLOSCAN_TEST_SCHEMA_DIAGNOSTICS_DIR"])

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="holoscan-schema-generator-")
        self.directory = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write(self, name: str, contents: str) -> Path:
        path = self.directory / name
        path.write_text(contents, encoding="utf-8")
        return path

    def bfbs(self, schema: Path) -> Path:
        result = subprocess.run(
            [
                str(self.flatc),
                "--binary",
                "--schema",
                "--require-explicit-ids",
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
        assert result.returncode == 0, result.stderr
        return self.directory / f"{schema.stem}.bfbs"

    def generate(self, schema: Path, *options: str) -> subprocess.CompletedProcess[str]:
        output = self.directory / f"{schema.stem}_schema_traits.hpp"
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
                *flatc_flag_arguments(),
                *options,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_ordinary_root_and_native_share_generated_identity(self) -> None:
        schema = self.write(
            "control.fbs",
            """namespace example;
table Control { enabled:bool (id: 0); }
root_type Control;
file_identifier \"CTL1\";
""",
        )
        result = self.generate(schema)
        assert result.returncode == 0, result.stderr
        generated = (self.directory / "control_schema_traits.hpp").read_text(encoding="utf-8")
        assert "make_schema_family_identity<example::Control>" in generated
        assert "FlatBufferSchemaBinding<example::Control>" in generated
        assert "using native_type = example::ControlT;" in generated
        assert "GeneratedSchemaNativeTraits<example::ControlT>" not in generated
        assert 'make_flatbuffer_file_identifier("CTL1")' in generated

    def test_descriptor_and_custom_codec_share_tensor_inventory(self) -> None:
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
        assert result.returncode == 0, result.stderr
        standard = (self.directory / "packet_schema_traits.hpp").read_text(encoding="utf-8")
        assert "tensor_field_ids" in standard
        assert "GeneratedDescriptorCodec" in standard
        assert "static constexpr std::size_t max_serialized_size" not in standard
        assert "static constexpr std::size_t metadata_capacity = 512U" in standard

        result = self.generate(schema, "--descriptor-tensor-custom-codec")
        assert result.returncode == 0, result.stderr
        custom = (self.directory / "packet_schema_traits.hpp").read_text(encoding="utf-8")
        assert "tensor_field_ids" in custom
        assert "GeneratedDescriptorCodec" not in custom
        assert "#if 0" not in custom

    def test_descriptor_schema_diagnostics_are_reported_directly(self) -> None:
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
        for fixture, expected in cases.items():
            with self.subTest(fixture=fixture):
                result = self.generate(self.diagnostics_dir / fixture)
                assert result.returncode != 0, result.stdout
                assert expected in result.stderr, result.stderr

    def test_descriptor_schema_preserves_file_identifier(self) -> None:
        schema = self.diagnostics_dir / "file_identifier.fbs"
        result = self.generate(schema)
        assert result.returncode == 0, result.stderr
        generated = (self.directory / "file_identifier_schema_traits.hpp").read_text(
            encoding="utf-8"
        )
        assert 'make_flatbuffer_file_identifier("IDTD")' in generated

    def test_descriptor_mode_is_inferred_for_a_global_namespace_root(self) -> None:
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
        assert "FlatBufferSchemaBinding<GlobalPacket>" in generated
        assert "namespace  {" not in generated

    def test_generation_policy_bounds_are_rejected_directly(self) -> None:
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
        assert result.returncode != 0, result.stdout
        assert "exceeds the 65536-byte descriptor metadata" in result.stderr

    def test_custom_codec_requires_an_inferred_descriptor_root(self) -> None:
        schema = self.write(
            "ordinary_custom.fbs",
            """table OrdinaryCustom { value:uint (id: 0); }
root_type OrdinaryCustom;
file_identifier "ORD2";
""",
        )
        result = self.generate(schema, "--descriptor-tensor-custom-codec")
        assert result.returncode != 0, result.stdout
        assert "requires a root with a direct holoscan.schema.Tensor field" in result.stderr

    def test_nested_only_tensor_requires_a_direct_descriptor_field(self) -> None:
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
        assert result.returncode != 0, result.stdout
        assert "HOLOSCAN_SCHEMA_TENSOR_REQUIRES_DIRECT_DESCRIPTOR_FIELD" in result.stderr

    def test_schema_without_file_identifier_is_rejected_directly(self) -> None:
        schema = self.write(
            "no_identifier.fbs",
            """namespace example;
table NoIdentifier { value:uint (id: 0); }
root_type NoIdentifier;
""",
        )
        result = self.generate(schema)
        assert result.returncode != 0, result.stdout
        assert "four-byte FlatBuffers file_identifier" in result.stderr

    def test_dependency_output_uses_flatc_include_graph(self) -> None:
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
        assert str(schema) in dependencies
        assert str(included) in dependencies

    def test_provenance_records_exact_bfbs_changes_outside_runtime_identity(self) -> None:
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
        assert first_evidence["flatc_flags"] == list(FLATC_FLAGS)
        assert first_evidence["bfbs_sha256"] != renamed_evidence["bfbs_sha256"]


if __name__ == "__main__":
    unittest.main()
