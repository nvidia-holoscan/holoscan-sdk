# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration document and parameter override tests."""

import gc
import json
import os
import subprocess

import holoscan_test_support  # noqa: F401
import pytest

import holoscan as hs

DETECTOR = "holoscan.test.ConfigDetector"
MANUAL_WITH_DESCRIPTOR = "holoscan.test.ManualRowSourceWithDescriptor"
PROBE = "holoscan.test.ParameterProbe"
SOURCE = "holoscan.test.CounterSource"


def write_config_files(tmp_path, values):
    """Write equivalent JSON/YAML documents for config loading tests."""
    json_path = tmp_path / "app.json"
    json_path.write_text(json.dumps(values), encoding="utf-8")
    yaml_path = tmp_path / "app.yaml"
    detector = values["detector"]
    yaml_path.write_text(
        f"detector:\n  threshold: {detector['threshold']}\n  gain: {detector['gain']}\n",
        encoding="utf-8",
    )
    return json_path, yaml_path


def config_file_graph(name="config-file"):
    """Compose the source/detector pipeline used by config binding tests."""
    graph = hs.Graph(name)
    source = graph.op(SOURCE, "source")
    detector = graph.op(DETECTOR, "detector", threshold=6, gain=1.0, window=8)
    graph.add_flow(source.out, detector["in"], queue_depth=8)
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    return graph, detector


# REQ: PY-API-015
def test_config_paths_are_owning_json_pointers():
    # Key setup: Parse escaped, root, and malformed owning JSON-pointer paths.
    path = hs.config.Path.parse("/profiles/camera~1front/1")

    # Key assertion: Valid paths own decoded tokens while malformed escapes retain a stable token.
    assert path.tokens == ["profiles", "camera/front", "1"]
    assert not path.is_root
    assert str(path) == "/profiles/camera~1front/1"
    assert hs.config.Path.parse("").is_root

    with pytest.raises(hs.errors.InvalidArgumentError) as caught:
        hs.config.Path.parse("/invalid~2escape")
    assert caught.value.token == "CONFIG_PATH_SYNTAX"
    details = caught.value.config_error
    assert details.code == hs.config.ConfigErrorCode.PATH_SYNTAX
    assert details.token == "CONFIG_PATH_SYNTAX"
    assert details.path == ""
    assert details.operation_stage == hs.config.ConfigOperationStage.NONE
    assert details.destination_state == hs.config.ConfigDestinationState.NOT_APPLICABLE
    assert details.causes == ()
    assert not details.message_truncated
    assert not details.path_truncated
    assert not details.source_truncated
    assert not details.excerpt_truncated
    assert not details.cause_overflow
    assert "ConfigErrorDetails" in repr(details)


# REQ: PY-API-015
def test_config_parse_failure_preserves_structured_diagnostics():
    # Key setup: Parse malformed JSON through the GIL-released native path.
    with pytest.raises(hs.errors.InvalidArgumentError) as caught:
        hs.config.parse('{"camera":', hs.config.Format.JSON)

    # Key assertion: Translation retains typed configuration evidence after reacquiring the GIL.
    details = caught.value.config_error
    assert caught.value.token == "CONFIG_PARSE_ERROR"
    assert details.code == hs.config.ConfigErrorCode.PARSE_ERROR
    assert details.token == "CONFIG_PARSE_ERROR"
    assert details.operation_stage == hs.config.ConfigOperationStage.PARSE
    assert details.destination_state == hs.config.ConfigDestinationState.NOT_APPLICABLE
    assert isinstance(details.message, str)
    assert isinstance(details.path, str)
    assert isinstance(details.source, str)
    assert isinstance(details.causes, tuple)


# REQ: PY-API-015
def test_yaml_and_json_load_to_the_same_semantic_document(tmp_path):
    # Key setup: Load equivalent JSON/YAML plus an extensionless document.
    paths = write_config_files(
        tmp_path,
        {"detector": {"threshold": 9, "gain": 2.5}},
    )

    json_document = hs.config.load(paths[0], hs.config.Format.JSON)
    yaml_document = hs.config.load(paths[1])
    bytes_document = hs.config.load(os.fsencode(paths[0]), hs.config.Format.JSON)

    # Key assertion: Equivalent inputs share semantics and ambiguous format inference fails closed.
    assert json_document.semantic_digest == yaml_document.semantic_digest
    assert bytes_document.semantic_digest == json_document.semantic_digest
    assert json_document.semantic_digest.startswith("sha256:")
    assert json_document.source.digest.startswith("sha256:")
    assert json_document.source.byte_size == paths[0].stat().st_size
    assert json_document.base_source is None
    assert "semantic_digest='sha256:" in repr(json_document)

    extensionless = tmp_path / "app"
    extensionless.write_text("{}", encoding="utf-8")
    assert hs.config.load(extensionless, hs.config.Format.JSON).semantic_digest
    with pytest.raises(hs.errors.InvalidArgumentError) as caught:
        hs.config.load(extensionless)
    assert caught.value.token == "CONFIG_FORMAT_REQUIRED"

    missing = tmp_path / "missing.json"
    with pytest.raises(hs.errors.NotFoundError) as caught:
        hs.config.load(missing)
    assert caught.value.token == "CONFIG_PATH_NOT_FOUND"

    if os.name != "nt":
        encoded_path = os.path.join(os.fsencode(tmp_path), b"config-\xff.json")
        with open(encoded_path, "wb") as stream:
            stream.write(b'{"detector":{"threshold":9,"gain":2.5}}')
        assert (
            hs.config.load(os.fsdecode(encoded_path)).semantic_digest
            == json_document.semantic_digest
        )


# REQ: PY-API-015
def test_config_parse_accepts_owned_text_and_bytes():
    # Key setup: Parse the same in-memory document from text and raw bytes.
    source = '{"detector":{"threshold":9,"gain":2.5}}'
    text_document = hs.config.parse(source, hs.config.Format.JSON)
    bytes_document = hs.config.parse(source.encode(), hs.config.Format.JSON)

    # Key assertion: In-memory parsing preserves the same semantic identity without filesystem I/O.
    assert text_document.semantic_digest == bytes_document.semantic_digest
    with pytest.raises(TypeError, match="CONFIG_SOURCE_STRING_OR_BYTES_REQUIRED"):
        hs.config.parse(42, hs.config.Format.JSON)


# REQ: PY-API-015
def test_sparse_config_patch_attaches_to_graph_with_provenance(tmp_path):
    # Key setup: Apply two sparse detector fields to the canonical config pipeline.
    #
    #   source --> detector
    json_path, _ = write_config_files(
        tmp_path,
        {"detector": {"threshold": 9, "gain": 2.5}},
    )
    graph, detector = config_file_graph()
    document = hs.config.load(json_path)
    binding = hs.params.bind(detector, hs.config.Path.parse("/detector"))

    provenance = hs.params.set_params(graph, document, [binding])

    # Key assertion: Sparse binding retains immutable field-level provenance.
    assert provenance.mode == hs.params.RebindMode.SPARSE_CONTRIBUTIONS
    assert provenance.config_field_count == 2
    fields = provenance.config_fields
    assert fields is provenance.config_fields
    assert isinstance(fields, tuple)
    assert [field.source_path for field in fields] == [
        "/detector/threshold",
        "/detector/gain",
    ]
    assert all(field.target_operator_path == "detector" for field in fields)
    assert all(field.document.semantic_digest == document.semantic_digest for field in fields)
    assert all(field.document.source.digest == document.source.digest for field in fields)
    with pytest.raises(TypeError):
        fields[0] = fields[0]
    assert "'detector' <- '/detector'" in repr(binding)
    assert "config_fields=2" in repr(provenance)


# REQ: PY-API-015
def test_duplicate_target_is_atomic_and_invalid_values_fail_compile(tmp_path):
    # Key setup: Apply a duplicate document, retry, then compile an invalid final value.
    #
    #   source --> detector
    json_path, _ = write_config_files(
        tmp_path,
        {"detector": {"threshold": 9, "gain": 2.5}},
    )
    graph, detector = config_file_graph("atomic-config")
    document = hs.config.load(json_path)
    binding = hs.params.bind(detector, "/detector")

    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(
        '{"detector":{"threshold":9,"legacy_threshold":11}}', encoding="utf-8"
    )
    with pytest.raises(hs.errors.InvalidArgumentError) as duplicate:
        hs.params.set_params(graph, hs.config.load(duplicate_path), [binding])

    # Key assertion: The binding error retains its token, precise source path, and target operator.
    assert duplicate.value.token == "PARAM_CONFIG_FIELD_DUPLICATE"
    assert "/detector/legacy_threshold" in str(duplicate.value)
    assert "operator 'detector'" in str(duplicate.value)

    valid = hs.params.set_params(graph, document, [binding])
    assert valid.config_field_count == 2

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text('{"detector":{"threshold":65}}', encoding="utf-8")
    invalid_graph, invalid_detector = config_file_graph("invalid-config")
    hs.params.set_params(
        invalid_graph,
        hs.config.load(invalid_path),
        [hs.params.bind(invalid_detector, "/detector")],
    )
    invalid_plan = hs.compile(invalid_graph)

    # Key assertion: Rejection is atomic, while final schema validation remains a compile concern.
    assert [field.source_path for field in valid.config_fields] == [
        "/detector/threshold",
        "/detector/gain",
    ]
    assert not invalid_plan.ok
    assert "PARAMETER_OVERRIDE_VALUE_INVALID" in [error.token for error in invalid_plan.errors]


# REQ: PY-API-015
def test_config_file_plan_matches_independent_cpp_authoring(tmp_path):
    # Key setup: Apply the same sparse config to equivalent Python and typed C++ graphs.
    #
    #   source --uint32--> detector
    json_path, _ = write_config_files(
        tmp_path,
        {"detector": {"threshold": 9, "gain": 2.5}},
    )
    graph, detector = config_file_graph()
    document = hs.config.load(json_path)
    hs.params.set_params(graph, document, [hs.params.bind(detector, "/detector")])
    python_plan = hs.compile(graph)

    cpp_plan = subprocess.run(
        [os.environ["HOLOSCAN_CONFIG_PLAN_HELPER"], json_path],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    # Key assertion: Configuration produces byte-identical canonical plans across languages.
    assert python_plan.ok
    assert python_plan.json() == cpp_plan


# REQ: PY-API-015
def test_config_binding_supports_scalar_vector_and_nested_fields(tmp_path):
    # Key setup: Bind scalar, sequence, and nested fields to one registered probe.
    #
    #   probe
    config_path = tmp_path / "probe.json"
    config_path.write_text(
        json.dumps(
            {
                "probe": {
                    "gain": 1.5,
                    "levels": [-2, 4],
                    "sensor": {"channels": 3},
                }
            }
        ),
        encoding="utf-8",
    )
    graph = hs.Graph("mixed-config")
    probe = graph.op(PROBE, "probe")

    provenance = hs.params.set_params(
        graph,
        hs.config.load(config_path),
        [hs.params.bind(probe, "/probe")],
    )

    # Key assertion: Every structured field uses captured typed routing.
    assert [field.source_path for field in provenance.config_fields] == [
        "/probe/gain",
        "/probe/levels",
        "/probe/sensor",
    ]
    assert all(field.origin == hs.params.FieldOrigin.CAPTURED for field in provenance.config_fields)


# REQ: PY-API-015
def test_config_decode_options_control_unknown_and_numeric_conversion():
    # Key setup: Bind a document containing an extra field and an inexact binary32 value.
    document = hs.config.parse(
        '{"probe":{"gain":0.1,"comment":"owned by another tool"}}',
        hs.config.Format.JSON,
    )
    graph = hs.Graph("decode-options")
    probe = graph.op(PROBE, "probe")
    binding = hs.params.bind(probe, "/probe")

    with pytest.raises(hs.errors.InvalidArgumentError):
        hs.params.set_params(graph, document, [binding])

    options = hs.params.DecodeOptions()
    options.values = hs.config.ValueDecodeOptions(
        unknown_fields=hs.config.UnknownFieldPolicy.IGNORE,
        numeric_conversion=hs.config.NumericConversionPolicy.ALLOW_CHECKED_LOSSY,
    )
    provenance = hs.params.set_params(graph, document, [binding], options)

    # Key assertion: Explicit Core policies admit the checked conversion and ignore the extra key.
    assert provenance.config_field_count == 1


# REQ: PY-API-015
def test_config_binding_type_errors_and_repr_are_precise():
    # Key setup: Exercise wrong dynamic types and one graph-local operator binding.
    document = hs.config.parse('{"detector":{"threshold":9}}', hs.config.Format.JSON)
    graph, detector = config_file_graph("binding-errors")
    binding = hs.params.bind(detector, "/detector")

    with pytest.raises(TypeError, match="PARAM_CONFIG_OPERATOR_REQUIRED"):
        hs.params.bind(42, "/detector")
    with pytest.raises(TypeError, match="PARAM_CONFIG_GRAPH_REQUIRED"):
        hs.params.set_params(42, document, [binding])
    with pytest.raises(TypeError, match="PARAM_CONFIG_BINDINGS_ITERABLE_REQUIRED"):
        hs.params.set_params(graph, document, 42)
    with pytest.raises(TypeError, match="PARAM_CONFIG_BINDING_ELEMENT_REQUIRED index=1"):
        hs.params.set_params(graph, document, [binding, "/detector"])

    # Key assertion: Binding repr matches the graph-local path recorded by Core provenance.
    assert repr(binding) == "<params.Binding 'detector' <- '/detector'>"


# REQ: PY-API-015
def test_manual_registry_row_reports_config_binding_unavailable():
    # Key setup: Resolve a manual operator row without a typed config thunk.
    #
    #   manual
    graph = hs.Graph("manual-config")
    operator = graph.op(MANUAL_WITH_DESCRIPTOR, "manual", limit=4)

    # Key assertion: Binding fails with the dedicated unavailable-thunk diagnostic.
    with pytest.raises(hs.errors.NotSupportedError) as caught:
        hs.params.bind(operator, "/manual")
    assert caught.value.token == "PARAM_CONFIG_BINDING_UNAVAILABLE"

    del graph
    gc.collect()
    with pytest.raises(hs.errors.InvalidArgumentError) as caught:
        hs.params.bind(operator, "/manual")
    assert caught.value.token == "GRAPH_HANDLE_INVALID"


# REQ: PY-API-015
def test_binding_from_retired_graph_is_rejected(tmp_path):
    # Key setup: Retain a binding after retiring its source graph, then apply it elsewhere.
    #
    #   retired: source --> detector    replacement: source --> detector
    json_path, _ = write_config_files(
        tmp_path,
        {"detector": {"threshold": 9, "gain": 2.5}},
    )
    graph, detector = config_file_graph("retired-config")
    binding = hs.params.bind(detector, "/detector")
    del detector
    del graph
    gc.collect()
    replacement, _ = config_file_graph("replacement-config")

    # Key assertion: Cross-graph application rejects the stale binding.
    with pytest.raises(hs.errors.IncompatibleError) as caught:
        hs.params.set_params(replacement, hs.config.load(json_path), [binding])
    assert caught.value.token == "PARAM_CONFIG_GRAPH_MISMATCH"
