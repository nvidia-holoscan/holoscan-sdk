# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema-driven Python operator parameter tests."""

import builtins
from datetime import timedelta

import holoscan_test_support  # noqa: F401
import pytest

import holoscan as hs

PROBE = "holoscan.tests.ParameterProbe"
ParameterProbe = hs.registry.resolve(PROBE, kind="operator")


EXPECTED_PARAMETER_TYPES = {
    "enabled": "bool",
    "offset": "int",
    "mode": "Literal['disabled', 0, 'automatic', 2, 'manual', 7]",
    "gain": "float",
    "label": "str",
    "timeout": "holoscan.time.Duration | datetime.timedelta",
    "even": "int",
    "levels": "list[int] | tuple[int, ...]",
    "labels": "list[str] | tuple[str, ...]",
    "optional_count": "int | None",
    "modes": (
        "list[Literal['disabled', 0, 'automatic', 2, 'manual', 7]] | "
        "tuple[Literal['disabled', 0, 'automatic', 2, 'manual', 7], ...]"
    ),
    "sensor": "dict[str, object]",
    "backup_sensor": "dict[str, object] | None",
    "secret": "str",
    "output_path": "str",
}


# REQ: PY-API-001
def test_error_and_payload_tokens_are_stable_public_objects():
    # Key setup: Resolve the public error hierarchy and built-in payload registry rows.

    # Key assertion: Public identities and representations remain stable and intentionally scoped.
    assert not hasattr(hs, "RuntimeError")
    assert issubclass(hs.errors.InvalidArgumentError, hs.HoloscanError)
    assert hs.payload.resolve("holoscan.uint32") == hs.payload.uint32
    assert hs.payload.resolve("holoscan.Tensor").name == hs.Tensor._registry_name
    assert hs.Tensor._registry_name == "holoscan.Tensor"
    assert hs.payload.uint32.name == "holoscan.uint32"
    assert hs.payload.uint32.epoch == 1
    assert repr(hs.payload.uint32) == "<PayloadType holoscan.uint32 (epoch 1)>"
    assert hash(hs.payload.resolve("uint32")) == hash(hs.payload.uint32)
    assert {hs.payload.uint32} == {hs.payload.resolve("uint32")}
    assert hs.payload.uint32 != hs.RealtimeClock

    class ForeignRegistryObject:
        _registry_name = "holoscan.uint32"

        def __eq__(self, other):
            return False

    assert hs.payload.uint32 != ForeignRegistryObject()

    assert hs.errors.ErrorCode.ARITHMETIC_OVERFLOW.value == 40
    assert issubclass(hs.errors.ArithmeticOverflowError, hs.HoloscanError)
    assert hs.errors.HoloscanNotImplementedError is not builtins.NotImplementedError
    assert hs.errors.HoloscanTimeoutError is not builtins.TimeoutError
    assert hs.errors.HoloscanInterruptedError is not builtins.InterruptedError
    assert not hasattr(hs.errors, "NotImplementedError")
    assert not hasattr(hs.errors, "TimeoutError")
    assert not hasattr(hs.errors, "InterruptedError")


# REQ: PY-API-003
def test_kwargs_cover_schema_value_kinds_and_redact_sensitive_values():
    # Key setup: Author one ParameterProbe with every supported schema value shape.
    #
    #   probe
    graph = hs.Graph("params")
    operator = graph.op(
        PROBE,
        "probe",
        enabled=False,
        offset=-3,
        mode="manual",
        gain=1.5,
        label="camera",
        timeout=timedelta(milliseconds=5),
        even=4,
        levels=[-2, 4],
        labels=["rgb", "left"],
        optional_count=None,
        modes=["automatic", 7],
        sensor={"channels": 3},
        backup_sensor={"channels": 4},
        secret="do-not-render",
        internal_note="internal-value",
        transient_note="transient-value",
    )

    rendered = repr(operator)

    # Key assertion: Structured kwargs render canonically while sensitive data stays redacted.
    assert "mode='manual'" in rendered
    assert "backup_sensor={'channels': 4}" in rendered
    assert "secret=<redacted>" in rendered
    assert "do-not-render" not in rendered
    assert "internal_note=<redacted>" in rendered
    assert "internal-value" not in rendered
    assert "transient_note=<redacted>" in rendered
    assert "transient-value" not in rendered


# REQ: PY-API-003
def test_parameter_repr_is_evaluated_only_when_requested():
    # Key setup: Author a probe with an integer whose Python repr deliberately fails.
    #
    #   probe
    class RaisingReprInt(int):
        def __repr__(self):
            raise ValueError("repr requested")

    graph = hs.Graph("lazy-repr")
    operator = graph.op(PROBE, "probe", even=RaisingReprInt(4))

    # Key assertion: Authoring succeeds and the repr failure occurs only at display time.
    assert operator.graph_valid
    with pytest.raises(ValueError, match="repr requested"):
        repr(operator)


# REQ: PY-API-003
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("enabled", False),
        ("offset", -3),
        ("mode", "manual"),
        ("mode", 7),
        ("gain", 1.5),
        ("gain", 1),
        ("label", "camera"),
        ("timeout", timedelta(milliseconds=5)),
        ("timeout", hs.time.us(5)),
        ("even", 4),
        ("levels", [-2, 4]),
        ("levels", (-2, 4)),
        ("labels", ["rgb", "left"]),
        ("labels", ("rgb", "left")),
        ("optional_count", 4),
        ("optional_count", None),
        ("modes", ["automatic", 7]),
        ("modes", ("automatic", 7)),
        ("sensor", {"channels": 3}),
        ("backup_sensor", {"channels": 4}),
        ("backup_sensor", None),
        ("secret", "do-not-render"),
        ("output_path", "artifacts"),
    ],
)
def test_registry_annotation_kinds_match_materializer_acceptance(field, value):
    # Key setup: Construct one registered probe with a schema-compatible parameter value.
    #
    #   probe
    graph = hs.Graph(f"annotation-{field}")

    # Key assertion: Generated annotations and runtime materialization accept the same value kind.
    assert ParameterProbe.parameters[field].python_type == EXPECTED_PARAMETER_TYPES[field]
    assert ParameterProbe(graph, "probe", **{field: value}).graph_valid


# REQ: PY-API-003
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("enabled", 1),
        ("offset", 1.5),
        ("mode", "unknown"),
        ("mode", 1),
        ("gain", True),
        ("label", 7),
        ("timeout", 5),
        ("even", False),
        ("levels", [1.5]),
        ("levels", "1,2"),
        ("labels", [1]),
        ("modes", ["unknown"]),
        ("sensor", {"channels": "three"}),
        ("backup_sensor", []),
        ("secret", 7),
        ("output_path", 7),
    ],
)
def test_registry_annotation_kinds_reject_other_materializer_shapes(field, value):
    # Key setup: Construct one registered probe with an incompatible parameter value.
    #
    #   probe
    graph = hs.Graph(f"annotation-rejection-{field}")

    # Key assertion: Runtime materialization rejects values excluded by the generated annotation.
    with pytest.raises(
        hs.errors.InvalidArgumentError,
        match="AUTHORING_PARAMETER_TYPE_MISMATCH|PARAM_VALIDATION_FAILED",
    ):
        ParameterProbe(graph, "probe", **{field: value})


# REQ: PY-API-003
@pytest.mark.parametrize(
    ("unit", "value"),
    [("timedelta", 3), ("ns", 3_000), ("us", 3), ("ms", 1)],
)
def test_duration_fields_accept_explicit_units(unit, value):
    # Key setup: Express one duration through each explicit supported unit wrapper.
    #
    #   probe
    graph = hs.Graph("durations")
    duration = (
        timedelta(microseconds=value) if unit == "timedelta" else getattr(hs.time, unit)(value)
    )

    # Key assertion: Explicitly typed duration values are accepted during operator construction.
    graph.op(PROBE, "probe", timeout=duration)


# REQ: PY-API-003
def test_duration_fields_reject_bare_numbers():
    # Key setup: Pass an untyped integer to the duration field of a single probe.
    #
    #   probe
    graph = hs.Graph("durations")
    with pytest.raises(hs.errors.InvalidArgumentError) as caught:
        graph.op(PROBE, "probe", timeout=1_000_000)

    # Key assertion: Bare duration numbers fail with the stable type-mismatch token.
    assert caught.value.code == hs.errors.ErrorCode.INVALID_ARGUMENT
    assert caught.value.token == "AUTHORING_PARAMETER_TYPE_MISMATCH"


# REQ: PY-API-003
def test_filesystem_parameters_accept_the_natural_string_carrier():
    # Key setup: Supply the natural Python string carrier for a filesystem path field.
    #
    #   probe
    graph = hs.Graph("filesystem-path")
    operator = graph.op(PROBE, "probe", output_path="artifacts")

    # Key assertion: Python delegates the string to Core's existing filesystem-path materializer.
    assert "output_path='artifacts'" in repr(operator)


# REQ: PY-API-001
# REQ: PY-API-003
@pytest.mark.parametrize(
    ("params", "token"),
    [
        ({"offset": 7}, "PARAM_VALIDATION_FAILED"),
        ({"mode": "unknown"}, "AUTHORING_PARAMETER_TYPE_MISMATCH"),
        ({"even": 3}, "PARAM_VALIDATION_FAILED"),
        ({"sensor": {"channels": 7}}, "PARAM_VALIDATION_FAILED"),
        ({"missing": 1}, "AUTHORING_PARAMETER_UNKNOWN"),
    ],
)
def test_schema_rejections_preserve_stable_tokens(params, token):
    # Key setup: Apply one invalid schema assignment to a single probe.
    #
    #   probe
    graph = hs.Graph("invalid-params")
    with pytest.raises(hs.errors.InvalidArgumentError) as caught:
        graph.op(PROBE, "probe", **params)

    # Key assertion: Each schema failure retains its expected public error token.
    assert caught.value.code == hs.errors.ErrorCode.INVALID_ARGUMENT
    assert caught.value.token == token


# REQ: PY-API-003
@pytest.mark.parametrize(
    "operator_type",
    [
        "holoscan.tests.ManualRowSourceWithDescriptor",
        "holoscan.tests.ManualRowSourceWithoutDescriptor",
    ],
)
def test_manual_rows_accept_python_keyword_parameters(operator_type):
    # Key setup: Construct a source through each hand-authored registration row.
    #
    #   source
    graph = hs.Graph("manual-row")
    operator = graph.op(operator_type, "source", limit=3)

    # Key assertion: Both declared and pass-through rows return a live graph-owned handle.
    assert operator.graph_valid
