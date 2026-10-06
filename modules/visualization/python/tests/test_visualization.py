# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Holoscan Visualization module authoring tests."""

from __future__ import annotations

import ast
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest
from holoscan_visualization.operators import HolovizOp

import holoscan as hs
import holoscan.operators as core_operators

# Resolve the payload by its stable registry identity so this provider test does not depend on
# whether an aggregate package has published the optional ``hs.schema.Image`` convenience facade.
_IMAGE_PAYLOAD = hs.payload.resolve("holoscan.schema.Image")


def image_layer(name: str = "video") -> dict[str, object]:
    """Return one representative image declaration used throughout these tests."""

    return {
        "port_name": name,
        "kind": "image",
        "format": "rgba8_unorm",
        "max_width": 640,
        "max_height": 480,
    }


def window_sink() -> dict[str, object]:
    """Return one representative window sink declaration."""

    return {"kind": "window", "title": "Test", "width": 640, "height": 480}


def video_boundary() -> hs.tensor.TensorBoundaryContract:
    """Return the bounded host RGBA contract used by the parity graph."""

    return hs.tensor.TensorBoundaryContract(
        representation=hs.tensor.TensorRepresentation(
            memory_kind=hs.tensor.MemoryKind.HOST,
            dtype=hs.tensor.DLDataType(hs.tensor.DLDataTypeCode.DLUINT, 8, 1),
            rank=3,
            strides=hs.tensor.TensorStridePolicy.CONTIGUOUS_ONLY,
        ),
        bounds=hs.tensor.TensorBounds(max_shape=(480, 640, 4)),
    )


def image_boundary(*, bits: int, shape: tuple[int, ...]) -> hs.tensor.TensorBoundaryContract:
    """Return one bounded contiguous host image contract."""

    return hs.tensor.TensorBoundaryContract(
        representation=hs.tensor.TensorRepresentation(
            memory_kind=hs.tensor.MemoryKind.HOST,
            dtype=hs.tensor.DLDataType(hs.tensor.DLDataTypeCode.DLUINT, bits, 1),
            rank=len(shape),
            strides=hs.tensor.TensorStridePolicy.CONTIGUOUS_ONLY,
        ),
        bounds=hs.tensor.TensorBounds(max_shape=shape),
    )


# REQ: PY-API-016
def test_fresh_import_initializes_only_the_module_owned_catalog():
    # Key setup: Inspect the registry before and after importing the Visualization provider.
    script = """
import holoscan
from holoscan._core import _registered_operator_names

assert 'holoscan.visualization.HolovizOp' not in _registered_operator_names()
from holoscan_visualization.operators import HolovizOp
assert HolovizOp.name == 'holoscan.visualization.HolovizOp'
assert 'holoscan.visualization.HolovizOp' in _registered_operator_names()
assert set(_registered_operator_names()) == {'holoscan.visualization.HolovizOp'}
"""

    # Key assertion: Core stays provider-neutral and the provider installs only its own row.
    subprocess.run([sys.executable, "-c", script], check=True, env=os.environ.copy())


@pytest.mark.parametrize(
    "layer",
    [
        image_layer(),
        {
            "port_name": "geometry",
            "kind": "geometry",
            "topologies": ["line_list", "rectangle_list"],
            "max_primitives": 16,
        },
        {
            "port_name": "labels",
            "kind": "text",
            "max_glyphs": 128,
            "normalized_coordinates": True,
        },
        {
            "port_name": "segmentation",
            "kind": "lut",
            "lut_size": 256,
            "max_width": 640,
            "max_height": 480,
        },
    ],
)
def test_every_layer_kind_declares_its_instance_port(layer):
    # Key setup: Author one HolovizOp from each supported tagged layer declaration.
    graph = hs.Graph("visualization-layer")
    handle = HolovizOp(graph, "visualizer", layers=[layer], sink=window_sink())

    # Key assertion: The handle exposes the exact port name derived from that instance.
    assert set(handle.ports) == {layer["port_name"]}
    assert handle.type_name == "holoscan.visualization.HolovizOp"


# REQ: PY-API-002
@pytest.mark.parametrize("port_name", ["graph_valid", "name", "ports", "type_name"])
def test_layer_ports_colliding_with_handle_properties_remain_indexable(port_name):
    # Key setup: Use each native-valid layer name that overlaps a fixed Python handle property.
    graph = hs.Graph(f"visualization-port-property-{port_name}")
    handle = HolovizOp(
        graph,
        "visualizer",
        layers=[{"port_name": port_name, "kind": "text", "max_glyphs": 1}],
        sink=window_sink(),
    )

    # Key assertion: Fixed properties retain precedence while indexed access resolves every port.
    assert handle.graph_valid
    assert handle.name == "visualizer"
    assert set(handle.ports) == {port_name}
    assert handle.type_name == "holoscan.visualization.HolovizOp"
    assert handle[port_name].name == port_name


@pytest.mark.parametrize(
    "sink",
    [
        {
            "kind": "window",
            "title": "Test",
            "width": 640,
            "height": 480,
            "present_mode": "fifo_relaxed",
            "framebuffer_srgb": True,
        },
        {
            "kind": "exclusive_display",
            "display_name": "DP-2",
            "refresh_hz": 60,
            "framebuffer_srgb": True,
        },
        {"kind": "exclusive_display", "width": 1920},
        {"kind": "exclusive_display", "height": 1080},
        {
            "kind": "cuda",
            "width": 640,
            "height": 480,
            "format": "rgba8_srgb",
        },
    ],
)
def test_every_sink_kind_is_accepted_during_authoring(sink):
    # Key setup: Author the operator with every closed sink alternative without starting it.
    graph = hs.Graph("visualization-sink")
    handle = HolovizOp(graph, "visualizer", layers=[image_layer()], sink=sink)

    # Key assertion: Constructor replay succeeds without creating runtime GPU resources.
    assert handle.graph_valid


# REQ: PY-API-003
def test_frame_trigger_alternatives_are_authorable_and_validated():
    # Key setup: Author the default trigger plus explicit nonzero-layer and periodic alternatives.
    graph = hs.Graph("visualization-trigger")
    layer_driven = HolovizOp(graph, "layer-driven", layers=[image_layer()], sink=window_sink())
    second_layer = HolovizOp(
        graph,
        "second-layer",
        layers=[image_layer("primary"), image_layer("overlay")],
        sink=window_sink(),
        trigger={"kind": "layer", "layer": 1},
    )
    periodic = HolovizOp(
        graph,
        "periodic",
        layers=[image_layer("periodic-video")],
        sink=window_sink(),
        trigger={"kind": "clock", "period": hs.time.ms(20)},
    )

    # Key assertion: Both native alternatives are reachable, while invalid variants fail before
    # HolovizOp's terminating constructor preconditions.
    assert layer_driven.graph_valid
    assert second_layer.graph_valid
    assert periodic.graph_valid

    blank_graph = hs.Graph("visualization-empty-clock-trigger")
    blank_graph.set_default_clock(blank_graph.add_clock(hs.RealtimeClock, "clock"))
    blank = HolovizOp(
        blank_graph,
        "blank",
        layers=[],
        sink={"kind": "window"},
        trigger={"kind": "clock", "period": hs.time.ms(20)},
    )
    blank_plan = hs.compile(blank_graph)
    assert blank.graph_valid
    assert not blank.ports
    assert blank_plan, blank_plan.json()

    missing_trigger_graph = hs.Graph("visualization-empty-missing-trigger")
    with pytest.raises(
        hs.errors.InvalidArgumentError,
        match=r"PARAMETER_REQUIRED.*field_name=trigger.*clock-trigger-for-empty-layer-stack",
    ):
        HolovizOp(
            missing_trigger_graph,
            "missing-trigger",
            layers=[],
            sink={"kind": "window"},
        )
    assert "operators=0" in repr(missing_trigger_graph)

    layer_trigger_graph = hs.Graph("visualization-empty-layer-trigger")
    with pytest.raises(hs.errors.InvalidArgumentError, match=r"TRIGGER_LAYER_OUT_OF_RANGE"):
        HolovizOp(
            layer_trigger_graph,
            "invalid-empty-layer-trigger",
            layers=[],
            sink={"kind": "window"},
            trigger={"kind": "layer", "layer": 0},
        )
    assert "operators=0" in repr(layer_trigger_graph)

    with pytest.raises(hs.errors.InvalidArgumentError, match=r"TRIGGER_LAYER_OUT_OF_RANGE"):
        HolovizOp(
            graph,
            "layer-out-of-range",
            layers=[image_layer()],
            sink=window_sink(),
            trigger={"kind": "layer", "layer": 1},
        )
    with pytest.raises(hs.errors.InvalidArgumentError, match=r"VALUE_INVALID.*period_ns"):
        HolovizOp(
            graph,
            "zero-period",
            layers=[image_layer()],
            sink=window_sink(),
            trigger={"kind": "clock", "period": hs.time.ns(0)},
        )
    with pytest.raises(TypeError, match="duration must be"):
        HolovizOp(
            graph,
            "bare-period",
            layers=[image_layer()],
            sink=window_sink(),
            trigger={"kind": "clock", "period": 20_000_000},
        )
    for index, trigger in enumerate(
        (
            {"kind": "clock", "period_ns": 20_000_000},
            {"kind": "clock", "period": hs.time.ms(20), "period_ns": 20_000_000},
        )
    ):
        raw_graph = hs.Graph(f"visualization-internal-trigger-field-{index}")
        original = dict(trigger)
        with pytest.raises(TypeError, match=r"use period with"):
            HolovizOp(
                raw_graph,
                "internal-trigger-field",
                layers=[image_layer()],
                sink=window_sink(),
                trigger=trigger,
            )
        assert trigger == original
        assert "operators=0" in repr(raw_graph)
    with pytest.raises(hs.errors.InvalidArgumentError, match=r"accepted=layer,clock"):
        HolovizOp(
            graph,
            "bad-kind",
            layers=[image_layer()],
            sink=window_sink(),
            trigger={"kind": "vsync"},
        )


# REQ: PY-API-003
@pytest.mark.parametrize(
    ("image_format", "bits", "shape"),
    [
        ("r16_unorm", 16, (480, 640)),
        ("r16_unorm", 16, (480, 640, 1)),
        ("r16_unorm", 8, (480, 640, 2)),
        ("rgba16_unorm", 16, (480, 640, 4)),
        ("rgba16_unorm", 8, (480, 640, 8)),
        ("bgra8_unorm", 8, (480, 640, 4)),
    ],
)
def test_image_formats_accept_documented_tensor_layouts(image_format, bits, shape):
    # Key setup: Connect each newly exposed or newly admitted format to a matching Image topic.
    #
    #   holomq://video-input --ImageT--> visualizer
    graph = hs.Graph(f"visualization-{image_format}-{bits}-{len(shape)}")
    clock = graph.add_clock(hs.ManualClock, "compile-clock")
    graph.set_default_clock(clock)
    video = graph.import_topic(
        "holomq://video-input",
        _IMAGE_PAYLOAD,
        endpoint_name="video-input",
        capture_time=clock,
        tensor_fields={"data": image_boundary(bits=bits, shape=shape)},
    )
    handle = HolovizOp(
        graph,
        "visualizer",
        layers=[{**image_layer(), "format": image_format}],
        sink=window_sink(),
    )
    graph.add_flow(video.output, handle.video)
    plan = hs.compile(graph)

    # Key assertion: Catalog translation preserves every C++-accepted source layout in the plan.
    assert plan, plan.json()


def test_structural_errors_leave_graph_without_an_operator():
    # Key setup: Supply missing, unknown, and over-capacity structural declarations.
    graph = hs.Graph("visualization-invalid")

    # Key assertion: Facade and catalog validation reject each form before adding an operator.
    with pytest.raises(TypeError, match="required keyword-only argument"):
        HolovizOp(graph, "missing")
    with pytest.raises(hs.errors.InvalidArgumentError, match="PARAMETER_UNKNOWN"):
        HolovizOp(
            graph,
            "unknown",
            layers=[{**image_layer(), "unexpected": True}],
            sink=window_sink(),
        )
    with pytest.raises(hs.errors.InvalidArgumentError, match="LAYER_CAPACITY_EXCEEDED"):
        HolovizOp(
            graph,
            "too-many",
            layers=[image_layer(f"layer-{index}") for index in range(9)],
            sink=window_sink(),
        )
    assert "operators=0" in repr(graph)


# REQ: PY-API-003
def test_cuda_output_port_collision_is_recoverable_and_sink_specific():
    # Key setup: Isolate the native constructor boundary so a regression reports a subprocess
    # failure instead of terminating the pytest process.
    script = """
import holoscan as hs
from holoscan_visualization.operators import HolovizOp

layer = {"port_name": "output", "kind": "text", "max_glyphs": 1}
graph = hs.Graph("visualization-cuda-output-collision")
try:
    HolovizOp(
        graph,
        "cuda-visualizer",
        layers=[layer],
        sink={"kind": "cuda", "width": 16, "height": 16},
    )
except hs.errors.InvalidArgumentError as error:
    assert error.token == "VISUALIZATION_OPERATOR_PORT_NAME_RESERVED"
    assert "field_name=layers[0].port_name" in str(error)
    assert "reserved-by=cuda-sink" in str(error)
else:
    raise AssertionError("CUDA sink admitted its reserved output port as an input layer")

assert "operators=0" in repr(graph)
display = HolovizOp(graph, "display", layers=[layer], sink={"kind": "window"})
assert display.graph_valid
assert display["output"].name == "output"
assert "operators=1" in repr(graph)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        env=os.environ.copy(),
    )

    # Key assertion: The collision raises before graph mutation, while the same name remains valid
    # for display sinks and the interpreter exits normally.
    assert result.returncode == 0, result.stdout + result.stderr


# REQ: PY-API-016
def test_module_facade_owns_required_metadata_and_generic_alias_is_absent():
    # Key setup: Inspect the pass-through row through its canonical module-owned facade.

    # Key assertion: Required keys are discoverable without exposing a weaker Core leaf alias.
    assert HolovizOp.required_parameters == frozenset({"layers", "sink"})
    assert not hasattr(core_operators, "HolovizOp")


# REQ: PY-API-003
def test_numeric_and_descriptor_invariants_fail_at_authoring():
    # Key setup: Exercise integer-to-float conversion and representative invalid declarations.
    graph = hs.Graph("visualization-validation")
    valid = HolovizOp(
        graph,
        "valid",
        layers=[
            {
                "kind": "geometry",
                "port_name": "geometry",
                "topologies": ["line_list"],
                "max_primitives": 4,
                "default_line_width": 2,
            }
        ],
        sink=window_sink(),
    )

    # Key assertion: Valid numeric-tower input succeeds and bad values retain useful attribution.
    assert valid.graph_valid
    with pytest.raises(hs.errors.InvalidArgumentError, match=r"VALUE_INVALID.*value=-1"):
        HolovizOp(
            graph,
            "negative",
            layers=[{"kind": "text", "port_name": "text", "max_glyphs": -1}],
            sink=window_sink(),
        )
    with pytest.raises(
        hs.errors.InvalidArgumentError,
        match=r"TYPE_MISMATCH.*normalized_coordinates",
    ):
        HolovizOp(
            graph,
            "numeric-text-coordinates",
            layers=[
                {
                    "kind": "text",
                    "port_name": "text",
                    "max_glyphs": 128,
                    "normalized_coordinates": 1,
                }
            ],
            sink=window_sink(),
        )
    with pytest.raises(hs.errors.InvalidArgumentError, match=r"YUV_CONVERSION_REQUIRED"):
        HolovizOp(
            graph,
            "missing-yuv",
            layers=[{**image_layer(), "format": "nv12"}],
            sink=window_sink(),
        )
    with pytest.raises(hs.errors.InvalidArgumentError, match=r"YUV_CONVERSION_FORBIDDEN"):
        HolovizOp(
            graph,
            "spurious-yuv",
            layers=[{**image_layer(), "yuv_model": "bt709"}],
            sink=window_sink(),
        )
    with pytest.raises(hs.errors.InvalidArgumentError, match=r"PORT_NAME_DUPLICATE.*video"):
        HolovizOp(
            graph,
            "duplicate",
            layers=[image_layer(), image_layer()],
            sink=window_sink(),
        )
    with pytest.raises(hs.errors.InvalidArgumentError, match=r"max_width.*value=0"):
        HolovizOp(
            graph,
            "zero-image-width",
            layers=[{**image_layer(), "max_width": 0}],
            sink=window_sink(),
        )
    omitted_width = image_layer()
    del omitted_width["max_width"]
    with pytest.raises(hs.errors.InvalidArgumentError, match=r"max_width.*value=0"):
        HolovizOp(
            graph,
            "omitted-image-width",
            layers=[omitted_width],
            sink=window_sink(),
        )
    with pytest.raises(
        hs.errors.InvalidArgumentError,
        match=r"format.*value='rgba8unorm'.*accepted=",
    ):
        HolovizOp(
            graph,
            "bad-format",
            layers=[{**image_layer(), "format": "rgba8unorm"}],
            sink=window_sink(),
        )
    with pytest.raises(hs.errors.InvalidArgumentError, match=r"UNKNOWN.*display_index"):
        HolovizOp(
            graph,
            "obsolete-display-selector",
            layers=[image_layer()],
            sink={"kind": "exclusive_display", "display_index": 0},
        )
    with pytest.raises(hs.errors.InvalidArgumentError, match=r"framebuffer_srgb"):
        HolovizOp(
            graph,
            "bad-framebuffer-flag",
            layers=[image_layer()],
            sink={**window_sink(), "framebuffer_srgb": 1},
        )


# REQ: PY-API-018
def test_runtime_and_committed_stub_call_signatures_match():
    # Key setup: Read the facade's committed stub and locate its callable declaration.
    stub_path = (
        Path(__file__).resolve().parents[1]
        / "holoscan_visualization"
        / "operators"
        / "__init__.pyi"
    )
    tree = ast.parse(stub_path.read_text(encoding="utf-8"))
    facade = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_HolovizOpType"
    )
    call = next(
        node
        for node in facade.body
        if isinstance(node, ast.FunctionDef) and node.name == "__call__"
    )
    stub_names = [argument.arg for argument in call.args.args[1:]] + [
        argument.arg for argument in call.args.kwonlyargs
    ]
    stub_required = {
        argument.arg
        for argument, default in zip(call.args.kwonlyargs, call.args.kw_defaults, strict=True)
        if default is None
    }
    runtime = inspect.signature(HolovizOp)

    # Key assertion: Manual facade and manual stub expose the same names and required keywords.
    assert list(runtime.parameters) == stub_names
    assert {
        name
        for name, parameter in runtime.parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and parameter.default is inspect.Parameter.empty
    } == stub_required


# REQ: PY-API-003
def test_python_and_cpp_authoring_produce_identical_plan_bytes():
    # Key setup: Author and compile the same one-layer Holoviz graph from Python and C++.
    #
    #   holomq://video-input --ImageT--> visualizer
    graph = hs.Graph("visualization-python-parity")
    clock = graph.add_clock(hs.ManualClock, "compile-clock")
    graph.set_default_clock(clock)
    video = graph.import_topic(
        "holomq://video-input",
        _IMAGE_PAYLOAD,
        endpoint_name="video-input",
        capture_time=clock,
        tensor_fields={"data": video_boundary()},
    )
    handle = HolovizOp(
        graph,
        "visualizer",
        layers=[
            {
                **image_layer(),
                "color_space": "srgb_nonlinear",
                "view": {"scale_x": 0.5, "scale_y": 0.5},
            }
        ],
        sink={
            "kind": "window",
            "title": "Python parity",
            "width": 640,
            "height": 480,
            "present_mode": "fifo_relaxed",
            "framebuffer_srgb": True,
        },
        trigger={"kind": "clock", "period": hs.time.ms(20)},
        max_input_age=hs.time.ms(40),
    )
    assert handle.video.direction is hs.PortDirection.INPUT
    graph.add_flow(video.output, handle.video, queue_depth=8)
    plan = hs.compile(graph)
    assert plan, plan.json()

    helper = os.environ["HOLOSCAN_VISUALIZATION_PLAN_HELPER"]
    cpp_plan = subprocess.run([helper], check=True, capture_output=True, text=True).stdout

    # Key assertion: The module facade and typed C++ constructor produce identical plan bytes.
    assert plan.json() == cpp_plan


# REQ: PY-API-003
def test_window_sink_fullscreen_fields_are_authorable():
    # Key setup: Author the native independent decorated-window defaults and fullscreen controls.
    graph = hs.Graph("visualization-window-fullscreen")
    default_sinks = (
        {"kind": "window"},
        {"kind": "window", "width": 1920},
        {"kind": "window", "height": 1080},
        {"kind": "window", "width": 0, "height": 0},
    )
    default_handles = [
        HolovizOp(
            graph,
            f"window-defaults-{index}",
            layers=[image_layer(f"video-{index}")],
            sink=sink,
        )
        for index, sink in enumerate(default_sinks)
    ]
    fullscreen = HolovizOp(
        graph,
        "fullscreen",
        layers=[image_layer()],
        sink={
            "kind": "window",
            "title": "Fullscreen",
            "width": 0,
            "height": 0,
            "fullscreen": True,
            "display_name": "DP-2",
            "refresh_hz": 120,
            "close_on_escape": True,
        },
    )

    # Key assertion: Omitted dimensions retain native defaults, fullscreen fields are accepted,
    # and monitor/mode selectors are rejected on decorated windows rather than silently ignored.
    assert all(handle.graph_valid for handle in default_handles)
    assert fullscreen.graph_valid
    for field, value in (("display_name", "DP-2"), ("refresh_hz", 120)):
        with pytest.raises(
            hs.errors.InvalidArgumentError, match=r"SINK_FULLSCREEN_FIELD_FORBIDDEN"
        ):
            HolovizOp(
                graph,
                f"windowed-{field}",
                layers=[image_layer()],
                sink={**window_sink(), field: value},
            )
