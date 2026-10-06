# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Holoscan Vision module binding tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

import holoscan as hs
from holoscan_vision import operators as vision_ops

OPERATOR_NAMES = {
    "BayerDemosaicOp": "holoscan.vision.BayerDemosaicOp",
    "ConvertToOp": "holoscan.vision.ConvertToOp",
    "CopyMakeBorderOp": "holoscan.vision.CopyMakeBorderOp",
    "CustomCropOp": "holoscan.vision.CustomCropOp",
    "NonMaximumSuppressionOp": "holoscan.vision.NonMaximumSuppressionOp",
    "NormalizeOp": "holoscan.vision.NormalizeOp",
    "ReformatOp": "holoscan.vision.ReformatOp",
    "RemapOp": "holoscan.vision.RemapOp",
    "ResizeOp": "holoscan.vision.ResizeOp",
}

OPERATOR_PORT_PAYLOADS = {
    "BayerDemosaicOp": {
        "input": "holoscan.schema.Image",
        "output": "holoscan.schema.Image",
    },
    "ConvertToOp": {"input": "holoscan.schema.Image", "output": "holoscan.schema.Image"},
    "CopyMakeBorderOp": {
        "input": "holoscan.schema.Image",
        "output": "holoscan.schema.Image",
    },
    "CustomCropOp": {"input": "holoscan.schema.Image", "output": "holoscan.schema.Image"},
    "NonMaximumSuppressionOp": {
        "boxes": "holoscan.Tensor",
        "scores": "holoscan.Tensor",
        "mask": "holoscan.Tensor",
    },
    "NormalizeOp": {"input": "holoscan.schema.Image", "output": "holoscan.schema.Image"},
    "ReformatOp": {"input": "holoscan.schema.Image", "output": "holoscan.schema.Image"},
    "RemapOp": {
        "input": "holoscan.schema.Image",
        "map": "holoscan.Tensor",
        "output": "holoscan.schema.Image",
    },
    "ResizeOp": {"input": "holoscan.schema.Image", "output": "holoscan.schema.Image"},
}


def test_fresh_import_initializes_only_the_module_owned_catalog():
    # Key setup: Import Vision in a fresh process and inspect its registered operator names.
    script = """
import holoscan
import holoscan_vision.operators
from holoscan._core import _registered_operator_names

assert set(_registered_operator_names()) == {
    'holoscan.vision.BayerDemosaicOp',
    'holoscan.vision.ConvertToOp',
    'holoscan.vision.CopyMakeBorderOp',
    'holoscan.vision.CustomCropOp',
    'holoscan.vision.NonMaximumSuppressionOp',
    'holoscan.vision.NormalizeOp',
    'holoscan.vision.ReformatOp',
    'holoscan.vision.RemapOp',
    'holoscan.vision.ResizeOp',
}
assert holoscan.payload.resolve('holoscan.schema.Image').name == 'holoscan.schema.Image'
assert holoscan.payload.resolve('holoscan.schema.Imu').name == 'holoscan.schema.Imu'
assert not hasattr(holoscan.operators, "ResizeOp")
assert not hasattr(holoscan_vision, "ResizeOp")
"""

    # Key assertion: Import initializes exactly the module-owned Vision catalog.
    subprocess.run([sys.executable, "-c", script], check=True, env=os.environ.copy())


@pytest.mark.parametrize("holoscan_first", [False, True])
# REQ: SCHEMA-PY-004
# REQ: SCHEMA-PY-006
def test_sensor_io_tokens_capture_eager_python_codecs_in_both_import_orders(holoscan_first):
    # Key setup: Construct token-backed descriptors after either Core/Vision import order,
    # before explicitly importing the Imu facade can upgrade a native-only registry row.
    script = f"""
if {holoscan_first!r}:
    import holoscan as hs
    import holoscan_vision
else:
    import holoscan_vision
    import holoscan as hs

token = hs.payload.resolve("holoscan.schema.Imu")
mapping_output = hs.Output("mapping_out", token)
mapping_input = hs.Input("in", token)
from holoscan.schema import Header, Imu

class Source(hs.Operator):
    mapping_out = mapping_output
    value_out = hs.Output("value_out", Imu)

    def setup(self, spec):
        spec.output(self.mapping_out)
        spec.output(self.value_out)

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

    def compute(self, context):
        self.mapping_out.emit({{"header": {{"frame_id": "mapping"}}, "flags": 0}})
        self.value_out.emit(Imu(header=Header(frame_id="value")))

class ValueSink(hs.Operator):
    inp = hs.Input("in", Imu)
    received = None

    def setup(self, spec):
        spec.input(self.inp)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def compute(self, context):
        type(self).received = self.inp.receive_data()

class MappingSink(hs.Operator):
    inp = mapping_input
    received = None

    def setup(self, spec):
        spec.input(self.inp)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def compute(self, context):
        type(self).received = self.inp.receive_data()

graph = hs.Graph("sensor-codec-import-order", default_clock=hs.RealtimeClock)
source = graph.op(Source, "source")
value_sink = graph.op(ValueSink, "value_sink")
mapping_sink = graph.op(MappingSink, "mapping_sink")
graph.add_flow(source.mapping_out, value_sink["in"])
graph.add_flow(source.value_out, mapping_sink["in"])
plan = hs.compile(graph)
assert plan.ok, [error.token for error in plan.errors]
session = hs.run(plan)
assert session.diagnostics == []
assert ValueSink.received.header.frame_id == "mapping"
assert MappingSink.received["header"]["frame_id"] == "value"
"""

    # Key assertion: The retained row has Python callbacks before either token-declared port is
    # authored, independent of whether Vision or Core initiated package loading.
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env=os.environ.copy(),
        timeout=15,
    )


@pytest.mark.parametrize("public_first", [False, True])
# REQ: PY-API-016
def test_entry_point_alias_preserves_vision_module_identity(tmp_path, public_first):
    # Key setup: Advertise the installed Vision provider through temporary wheel metadata.
    metadata = tmp_path / "holoscan_vision-1.0.dist-info"
    metadata.mkdir()
    (metadata / "entry_points.txt").write_text(
        "[holoscan.modules]\nvision = holoscan_vision\n", encoding="utf-8"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(tmp_path), environment.get("PYTHONPATH", "")))
    script = f"""
import importlib
import sys

import holoscan

assert "holoscan_vision" not in sys.modules
first, second = (
    ("holoscan.vision.operators", "holoscan_vision.operators")
    if {public_first!r}
    else ("holoscan_vision.operators", "holoscan.vision.operators")
)
first_module = importlib.import_module(first)
second_module = importlib.import_module(second)
import holoscan_vision

assert first_module is second_module
assert holoscan.vision is holoscan_vision
assert holoscan.vision.operators is holoscan_vision.operators
assert not hasattr(holoscan.operators, "ResizeOp")
assert not hasattr(holoscan_vision, "ResizeOp")
"""

    # Key assertion: Either import order loads one provider and aliases its operator module exactly.
    subprocess.run([sys.executable, "-c", script], check=True, env=environment)


@pytest.mark.parametrize(
    ("operator_name", "params", "ports"),
    [
        (
            "BayerDemosaicOp",
            {"bayer_grid_pos": "rggb", "generate_alpha": True, "alpha_value": 255},
            {"input", "output"},
        ),
        (
            "ConvertToOp",
            {"output_type": "uint8"},
            {"input", "output"},
        ),
        (
            "CopyMakeBorderOp",
            {"max_width": 4, "max_height": 3, "max_batch_size": 2, "border_mode": "reflect"},
            {"input", "output"},
        ),
        (
            "CustomCropOp",
            {
                "max_batch_size": 2,
                "width": 3,
                "height": 2,
            },
            {"input", "output"},
        ),
        (
            "NonMaximumSuppressionOp",
            {"max_boxes": 4, "max_batch_size": 2, "iou_threshold": 0.25},
            {"boxes", "scores", "mask"},
        ),
        (
            "NormalizeOp",
            {"base": [0.5]},
            {"input", "output"},
        ),
        (
            "ReformatOp",
            {"mode": "planar_to_interleaved"},
            {"input", "output"},
        ),
        (
            "RemapOp",
            {
                "max_batch_size": 2,
                "width": 3,
                "height": 2,
                "map_value_mode": "absolute_normalized",
            },
            {"input", "map", "output"},
        ),
        (
            "ResizeOp",
            {
                "max_batch_size": 2,
                "width": 3,
                "height": 2,
                "interpolation": "cubic",
            },
            {"input", "output"},
        ),
    ],
)
def test_every_operator_authors_from_schema_validated_kwargs(operator_name, params, ports):
    # Key setup: Author each registered Vision operator with schema-valid parameters.
    #
    #   operator
    graph = hs.Graph(f"vision-{operator_name}")
    operator_type = getattr(vision_ops, operator_name)
    handle = operator_type(graph, "operator", **params)

    # Key assertion: The typed handle preserves the registered identity and complete port set.
    assert operator_type.name == OPERATOR_NAMES[operator_name]
    assert set(handle.ports) == ports
    assert {
        name: port.payload_type for name, port in operator_type.ports.items()
    } == OPERATOR_PORT_PAYLOADS[operator_name]
    assert handle.type_name == OPERATOR_NAMES[operator_name]


@pytest.mark.parametrize(
    ("operator_name", "invalid_params"),
    [
        (
            "CopyMakeBorderOp",
            {"max_width": 0, "max_height": 3, "max_batch_size": 2},
        ),
        ("CustomCropOp", {"max_batch_size": 0}),
        ("NonMaximumSuppressionOp", {"max_boxes": 0, "max_batch_size": 2}),
        ("RemapOp", {"max_batch_size": 0}),
        ("ResizeOp", {"max_batch_size": 0}),
    ],
)
def test_every_operator_rejects_invalid_capacity(operator_name, invalid_params):
    # Key setup: Author each Vision operator with an invalid compile-time capacity.
    #
    #   invalid capacity -X-> operator

    # Key assertion: Schema validation rejects every invalid capacity with the stable token.
    with pytest.raises(hs.errors.InvalidArgumentError, match="PARAM_VALIDATION_FAILED"):
        getattr(vision_ops, operator_name)(
            hs.Graph("invalid-capacity"), "operator", **invalid_params
        )


@pytest.mark.parametrize(
    ("operator_name", "required_parameters"),
    [
        ("CopyMakeBorderOp", {"max_width", "max_height", "max_batch_size"}),
        ("CustomCropOp", {"max_batch_size"}),
        ("NonMaximumSuppressionOp", {"max_boxes", "max_batch_size"}),
        ("RemapOp", {"max_batch_size"}),
        ("ResizeOp", {"max_batch_size"}),
    ],
)
def test_capacity_derived_operators_require_structural_parameters(
    operator_name, required_parameters
):
    # Key setup: Omit the required structural-capacity keywords from a Vision operator call.
    operator_type = getattr(vision_ops, operator_name)
    assert operator_type.required_parameters == required_parameters

    # Key assertion: The proxy rejects the incomplete constructor before mutating the Graph.
    graph = hs.Graph("missing-capacity")
    with pytest.raises(TypeError, match="missing required parameters"):
        operator_type(graph, "operator")
    assert "operators=0" in repr(graph)


@pytest.mark.parametrize(
    "operator_name", ["BayerDemosaicOp", "ConvertToOp", "NormalizeOp", "ReformatOp"]
)
def test_input_derived_operators_need_no_capacity_parameters(operator_name):
    # Key setup: Author an input-derived Vision operator using its default parameters.
    operator_type = getattr(vision_ops, operator_name)
    graph = hs.Graph("input-derived-capacity")
    handle = operator_type(graph, "operator")

    # Key assertion: No obsolete capacity keyword is required or exposed.
    assert operator_type.required_parameters == set()
    assert handle.type_name == OPERATOR_NAMES[operator_name]


@pytest.mark.parametrize("bayer_grid_pos", ["rggb", 1], ids=["named", "numeric"])
# REQ: PY-API-018
def test_bayer_demosaic_accepts_named_and_numeric_phase_values(bayer_grid_pos):
    # Key setup: Author the NPP-backed operator through both stable registry enum forms.
    graph = hs.Graph("bayer-phase")
    handle = vision_ops.BayerDemosaicOp(
        graph, "bayer", bayer_grid_pos=bayer_grid_pos, generate_alpha=True, alpha_value=255
    )

    # Key assertion: Both spellings produce the same typed operator surface.
    assert handle.type_name == "holoscan.vision.BayerDemosaicOp"
    assert handle.input.payload_type.name == "holoscan.schema.Image"
    assert handle.output.payload_type.name == "holoscan.schema.Image"


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("bayer_grid_pos", "unknown"),
        ("bayer_grid_pos", 4),
        ("bayer_grid_pos", object()),
        ("alpha_value", -1),
        ("alpha_value", 65536),
    ],
)
# REQ: PY-API-018
def test_bayer_demosaic_rejects_invalid_parameters_before_graph_mutation(parameter, value):
    # Key setup: Supply one invalid phase or alpha value to a fresh graph.
    graph = hs.Graph("invalid-bayer-parameter")

    # Key assertion: Canonical schema validation fails without attaching a partial operator.
    with pytest.raises(
        hs.errors.InvalidArgumentError,
        match="AUTHORING_PARAMETER_TYPE_MISMATCH|PARAM_VALIDATION_FAILED",
    ):
        vision_ops.BayerDemosaicOp(graph, "bayer", **{parameter: value})
    assert "operators=0" in repr(graph)


# REQ: PY-API-018
def test_operator_help_is_module_owned():
    # Key setup: Inspect runtime help and enum metadata without requiring typing artifacts.

    # Key assertion: Vision owns complete runtime metadata for its operator proxies.
    assert "Parameters:" in (vision_ops.ResizeOp.__doc__ or "")
    assert vision_ops.ResizeOp.parameter_introspection == "declared"
    assert vision_ops.ResizeOp.required_parameters == {"max_batch_size"}
    assert "interpolation" in (vision_ops.ResizeOp.__doc__ or "")
    assert vision_ops.ResizeOp.interpolation.choices == (
        ("nearest", 0),
        ("linear", 1),
        ("cubic", 2),
        ("area", 3),
    )
    assert vision_ops.BayerDemosaicOp.bayer_grid_pos.choices == (
        ("bggr", 0),
        ("rggb", 1),
        ("gbrg", 2),
        ("grbg", 3),
    )


def test_python_and_cpp_resize_authoring_produce_identical_plan_bytes():
    # Key setup: Author and compile the same single-Resize graph from Python and C++.
    #
    #   resize
    graph = hs.Graph("vision-python-parity")
    resize = vision_ops.ResizeOp(
        graph,
        "resize",
        max_batch_size=2,
        width=3,
        height=2,
        interpolation="cubic",
    )
    assert resize.input.direction is hs.PortDirection.INPUT
    assert resize.output.direction is hs.PortDirection.OUTPUT
    graph.set_default_clock(graph.add_clock(hs.ManualClock, "compile-clock"))

    options = hs.CompileOptions()
    options.deployment.bind_tensor_output_device(
        operator_path="resize", output_port="output", device=hs.execution.DeviceId(0)
    )
    plan = hs.compile(graph, options)
    assert plan, plan.json()
    plan_data = json.loads(plan.json())
    resize_ports = [port for port in plan_data["ports"] if port["owner"] == 1]
    assert {
        port["local_name"]: port["payload_identity"]["schema"]["canonical_name"]
        for port in resize_ports
    } == {"input": "holoscan.schema.Image", "output": "holoscan.schema.Image"}
    assert [
        (derivation["field_id"], derivation["tensor_index"])
        for derivation in plan_data["tensor_byte_span_derivations"]
    ] == [(4, 0)]

    helper = os.environ["HOLOSCAN_VISION_PLAN_HELPER"]
    cpp_plan = subprocess.run([helper], check=True, capture_output=True, text=True).stdout

    # Key assertion: Both language surfaces produce byte-identical admitted plans.
    assert plan.json() == cpp_plan
