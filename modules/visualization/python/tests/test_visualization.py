# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Holoscan Visualization module authoring tests."""

from __future__ import annotations

import ast
import copy
import dataclasses
import enum
import inspect
import os
import re
import subprocess
import sys
from pathlib import Path

import holoscan_visualization.operators as visualization_operators
import pytest
from holoscan_visualization.operators import (
    ClockTrigger,
    ColorSpace,
    CudaSink,
    ExclusiveDisplaySink,
    GeometryLayer,
    GeometryTopology,
    HolovizOp,
    ImageFormat,
    ImageLayer,
    LayerTrigger,
    LutLayer,
    PresentMode,
    TextLayer,
    ViewTransform,
    WindowSink,
    YuvModel,
    YuvRange,
)

import holoscan as hs
import holoscan.operators as core_operators

_STUB_PATH = (
    Path(__file__).resolve().parents[1] / "holoscan_visualization" / "operators" / "__init__.pyi"
)
_TYPED_DECLARATIONS = (
    ViewTransform,
    ImageLayer,
    GeometryLayer,
    TextLayer,
    LutLayer,
    WindowSink,
    ExclusiveDisplaySink,
    CudaSink,
    LayerTrigger,
    ClockTrigger,
)
_WIRE_ENUMS = (ImageFormat, PresentMode, ColorSpace, YuvModel, YuvRange, GeometryTopology)

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


def typed_image_layer(name: str = "video") -> ImageLayer:
    """Return the typed counterpart of :func:`image_layer`."""

    return ImageLayer(port_name=name, max_width=640, max_height=480)


def typed_window_sink() -> WindowSink:
    """Return the typed counterpart of :func:`window_sink`."""

    return WindowSink(title="Test", width=640, height=480)


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
    assert not hasattr(core_operators, "ImageLayer")


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


def parity_plan(layer: object, sink: object, trigger: object) -> str:
    """Compile the one-layer parity graph the C++ helper authors and return its plan JSON."""

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
        layers=[layer],
        sink=sink,
        trigger=trigger,
        max_input_age=hs.time.ms(40),
    )
    assert handle.video.direction is hs.PortDirection.INPUT
    graph.add_flow(video.output, handle.video, queue_depth=8)
    plan = hs.compile(graph)
    assert plan, plan.json()
    return plan.json()


def cpp_parity_plan() -> str:
    """Return the plan JSON the C++ helper produces for the same graph."""

    helper = os.environ["HOLOSCAN_VISUALIZATION_PLAN_HELPER"]
    return subprocess.run([helper], check=True, capture_output=True, text=True).stdout


# REQ: PY-API-003
def test_python_and_cpp_authoring_produce_identical_plan_bytes():
    # Key setup: Author and compile the same one-layer Holoviz graph from Python and C++.
    plan = parity_plan(
        {
            **image_layer(),
            "color_space": "srgb_nonlinear",
            "view": {"scale_x": 0.5, "scale_y": 0.5},
        },
        {
            "kind": "window",
            "title": "Python parity",
            "width": 640,
            "height": 480,
            "present_mode": "fifo_relaxed",
            "framebuffer_srgb": True,
        },
        {"kind": "clock", "period": hs.time.ms(20)},
    )

    # Key assertion: The module facade and typed C++ constructor produce identical plan bytes.
    assert plan == cpp_parity_plan()


# REQ: PY-API-003
def test_typed_and_cpp_authoring_produce_identical_plan_bytes():
    # Key setup: Author the C++ helper's graph from typed declarations, mixing enum and string
    # labels and a typed nested view.
    plan = parity_plan(
        ImageLayer(
            port_name="video",
            max_width=640,
            max_height=480,
            color_space=ColorSpace.SRGB_NONLINEAR,
            view=ViewTransform(scale_x=0.5, scale_y=0.5),
        ),
        WindowSink(
            title="Python parity",
            width=640,
            height=480,
            present_mode="fifo_relaxed",
            framebuffer_srgb=True,
        ),
        ClockTrigger(period=hs.time.ms(20)),
    )

    # Key assertion: Typed lowering reaches the same C++ descriptors, byte for byte.
    assert plan == cpp_parity_plan()


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


# REQ: PY-API-003
@pytest.mark.parametrize(
    "layer",
    [
        typed_image_layer(),
        GeometryLayer(
            port_name="geometry",
            topologies=[GeometryTopology.LINE_LIST, "rectangle_list"],
            max_primitives=16,
        ),
        TextLayer(port_name="labels", max_glyphs=128, normalized_coordinates=True),
        LutLayer(port_name="segmentation", lut_size=256, max_width=640, max_height=480),
    ],
)
def test_every_typed_layer_kind_declares_its_instance_port(layer):
    # Key setup: Author one HolovizOp from each typed layer declaration.
    graph = hs.Graph("visualization-typed-layer")
    handle = HolovizOp(graph, "visualizer", layers=[layer], sink=typed_window_sink())

    # Key assertion: The typed declaration yields the same instance port as its tagged dict.
    assert set(handle.ports) == {layer.port_name}
    assert handle.type_name == "holoscan.visualization.HolovizOp"


# REQ: PY-API-003
@pytest.mark.parametrize(
    "sink",
    [
        WindowSink(
            title="Test",
            width=640,
            height=480,
            present_mode=PresentMode.FIFO_RELAXED,
            framebuffer_srgb=True,
        ),
        ExclusiveDisplaySink(display_name="DP-2", refresh_hz=60, framebuffer_srgb=True),
        CudaSink(width=640, height=480, format="bgra8_srgb"),
        CudaSink(width=640, height=480, format=ImageFormat.RGBA8_SRGB),
    ],
)
def test_every_typed_sink_kind_is_accepted_during_authoring(sink):
    # Key setup: Author the operator with every typed sink alternative without starting it.
    graph = hs.Graph("visualization-typed-sink")
    handle = HolovizOp(graph, "visualizer", layers=[typed_image_layer()], sink=sink)

    # Key assertion: Constructor replay succeeds without creating runtime GPU resources.
    assert handle.graph_valid


# REQ: PY-API-003
def test_typed_triggers_match_dict_triggers():
    # Key setup: Author every typed trigger alternative beside its dict spelling.
    graph = hs.Graph("visualization-typed-trigger")
    second_layer = HolovizOp(
        graph,
        "second-layer",
        layers=[typed_image_layer("primary"), typed_image_layer("overlay")],
        sink=typed_window_sink(),
        trigger=LayerTrigger(layer=1),
    )
    periodic = HolovizOp(
        graph,
        "periodic",
        layers=[typed_image_layer("periodic-video")],
        sink=typed_window_sink(),
        trigger=ClockTrigger(period=hs.time.ms(20)),
    )
    blank_graph = hs.Graph("visualization-typed-default-clock")
    blank_graph.set_default_clock(blank_graph.add_clock(hs.RealtimeClock, "clock"))
    blank = HolovizOp(blank_graph, "blank", layers=[], sink=WindowSink(), trigger=ClockTrigger())

    # Key assertion: Typed triggers reach the same C++ alternatives and the same rejections.
    assert second_layer.graph_valid
    assert periodic.graph_valid
    assert blank.graph_valid
    assert hs.compile(blank_graph)
    with pytest.raises(hs.errors.InvalidArgumentError, match=r"TRIGGER_LAYER_OUT_OF_RANGE"):
        HolovizOp(
            graph,
            "layer-out-of-range",
            layers=[typed_image_layer()],
            sink=typed_window_sink(),
            trigger=LayerTrigger(layer=1),
        )
    with pytest.raises(hs.errors.InvalidArgumentError, match=r"VALUE_INVALID.*period_ns"):
        HolovizOp(
            graph,
            "zero-period",
            layers=[typed_image_layer()],
            sink=typed_window_sink(),
            trigger=ClockTrigger(period=hs.time.ns(0)),
        )
    with pytest.raises(TypeError, match="duration must be"):
        HolovizOp(
            graph,
            "bare-period",
            layers=[typed_image_layer()],
            sink=typed_window_sink(),
            trigger=ClockTrigger(period=20_000_000),
        )
    # The dict guard against caller-supplied period_ns is untouched by the typed path.
    with pytest.raises(TypeError, match=r"use period with"):
        HolovizOp(
            graph,
            "internal-trigger-field",
            layers=[typed_image_layer()],
            sink=typed_window_sink(),
            trigger={"kind": "clock", "period_ns": 20_000_000},
        )


# REQ: PY-API-003
def test_mixed_typed_and_dict_declarations_lower_uniformly():
    # Key setup: Combine typed and tagged-dict declarations in one authoring call.
    graph = hs.Graph("visualization-mixed")
    mixed = HolovizOp(
        graph,
        "mixed",
        layers=[typed_image_layer("typed"), image_layer("dict")],
        sink=window_sink(),
        trigger=LayerTrigger(layer=1),
    )
    nested = HolovizOp(
        graph,
        "nested-dict-view",
        layers=[
            ImageLayer(port_name="video", max_width=640, max_height=480, view={"scale_x": 0.5})
        ],
        sink=typed_window_sink(),
        trigger={"kind": "layer", "layer": 0},
    )

    # Key assertion: Each element lowers independently and the facade never mutates inputs.
    assert set(mixed.ports) == {"typed", "dict"}
    assert nested.graph_valid
    original_layers = [typed_image_layer("a"), image_layer("b")]
    snapshot = list(original_layers)
    HolovizOp(graph, "untouched", layers=original_layers, sink=window_sink())
    assert original_layers == snapshot


# REQ: PY-API-003
def test_typed_view_inside_dict_layer_is_rejected_with_actionable_message():
    # Key setup: Nest a typed view in the second, hand-authored image layer.
    graph = hs.Graph("visualization-dict-layer-typed-view")
    dict_layer = {**image_layer("dict"), "view": ViewTransform(scale_x=0.5)}
    before = copy.deepcopy(dict_layer)

    # Key assertion: The facade names the layer index and both supported corrections, adds no
    # operator, and leaves the caller's dictionary untouched.
    with pytest.raises(TypeError) as info:
        HolovizOp(
            graph,
            "visualizer",
            layers=[typed_image_layer("typed"), dict_layer],
            sink=typed_window_sink(),
        )
    message = str(info.value)
    assert message.startswith("layers[1].view: ViewTransform is supported inside ImageLayer")
    assert "ImageLayer(..., view=ViewTransform(...))" in message
    assert '{"scale_x": 0.5}' in message
    assert "operators=0 flows=0" in repr(graph)
    assert dict_layer == before
    assert isinstance(dict_layer["view"], ViewTransform)

    # Key assertion: Malformed view dictionaries still fail in the native catalog.
    with pytest.raises(hs.errors.InvalidArgumentError, match=r"field_name=layers\[0\]\.view"):
        HolovizOp(
            graph,
            "malformed-view",
            layers=[{**image_layer(), "view": ["not", "a", "mapping"]}],
            sink=window_sink(),
        )
    assert "operators=0" in repr(graph)


# REQ: PY-API-003
def test_enum_and_string_field_values_lower_identically():
    # Key setup: Lower the same declarations spelled with enum members and with wire strings.
    by_enum = ImageLayer(
        port_name="video",
        max_width=640,
        max_height=480,
        format=ImageFormat.NV12,
        color_space=ColorSpace.PASS_THROUGH,
        yuv_model=YuvModel.BT709,
        yuv_range=YuvRange.LIMITED,
    )._holoscan_parameters()
    by_string = ImageLayer(
        port_name="video",
        max_width=640,
        max_height=480,
        format="nv12",
        color_space="pass_through",
        yuv_model="bt709",
        yuv_range="limited",
    )._holoscan_parameters()
    geometry = GeometryLayer(
        port_name="geometry",
        topologies=(GeometryTopology.LINE_LIST, "oval_list"),
        max_primitives=4,
    )._holoscan_parameters()

    # Key assertion: Lowered values are plain wire strings with no enum instances left behind.
    assert by_enum == by_string
    assert geometry["topologies"] == ["line_list", "oval_list"]
    for lowered in (by_enum, geometry, WindowSink()._holoscan_parameters()):
        for value in lowered.values():
            assert not isinstance(value, enum.Enum)
            assert type(value) in {str, int, float, bool, dict, list}


# REQ: PY-API-003
def test_enum_members_inside_dict_declarations_reach_the_catalog_as_wire_strings():
    # Key setup: Hand-authored dictionaries carrying StrEnum members instead of wire strings. The
    # facade passes dictionaries through untouched; the native string conversion accepts ``str``
    # subclasses, so no Python-side lowering is needed for this mix.
    def plan(format, yuv_model, yuv_range, topology, present_mode) -> str:  # noqa: A002
        graph = hs.Graph("visualization-dict-enums")
        handle = HolovizOp(
            graph,
            "visualizer",
            layers=[
                {
                    **image_layer(),
                    "format": format,
                    "yuv_model": yuv_model,
                    "yuv_range": yuv_range,
                },
                {
                    "kind": "geometry",
                    "port_name": "shape",
                    "topologies": [topology],
                    "max_primitives": 4,
                },
            ],
            sink={**window_sink(), "present_mode": present_mode},
        )
        assert set(handle.ports) == {"video", "shape"}
        return hs.compile(graph).json()

    by_enum = plan(
        ImageFormat.NV12,
        YuvModel.BT709,
        YuvRange.FULL,
        GeometryTopology.OVAL_LIST,
        PresentMode.FIFO,
    )
    by_string = plan("nv12", "bt709", "full", "oval_list", "fifo")

    # Key assertion: Enum members in dictionaries produce byte-identical plans to wire strings.
    assert by_enum == by_string


# REQ: PY-API-003
@pytest.mark.parametrize(
    ("wire_enum", "declare"),
    [
        (
            ImageFormat,
            lambda label: {
                "layers": [ImageLayer(port_name="v", max_width=1, max_height=1, format=label)],
                "sink": WindowSink(),
            },
        ),
        (
            ColorSpace,
            lambda label: {
                "layers": [ImageLayer(port_name="v", max_width=1, max_height=1, color_space=label)],
                "sink": WindowSink(),
            },
        ),
        (
            YuvModel,
            lambda label: {
                "layers": [
                    ImageLayer(
                        port_name="v",
                        max_width=1,
                        max_height=1,
                        format="nv12",
                        yuv_model=label,
                        yuv_range="full",
                    )
                ],
                "sink": WindowSink(),
            },
        ),
        (
            YuvRange,
            lambda label: {
                "layers": [
                    ImageLayer(
                        port_name="v",
                        max_width=1,
                        max_height=1,
                        format="nv12",
                        yuv_model="bt709",
                        yuv_range=label,
                    )
                ],
                "sink": WindowSink(),
            },
        ),
        (
            GeometryTopology,
            lambda label: {
                "layers": [GeometryLayer(port_name="v", topologies=[label], max_primitives=1)],
                "sink": WindowSink(),
            },
        ),
        (
            PresentMode,
            lambda label: {
                "layers": [ImageLayer(port_name="v", max_width=1, max_height=1)],
                "sink": WindowSink(present_mode=label),
            },
        ),
    ],
)
def test_wire_enums_match_the_catalog_accepted_labels(wire_enum, declare):
    # Key setup: Provoke the catalog's accepted-label listing for the field the enum labels.
    graph = hs.Graph(f"visualization-enum-{wire_enum.__name__}")
    with pytest.raises(hs.errors.InvalidArgumentError, match=r"VALUE_INVALID") as info:
        HolovizOp(graph, "invalid", **declare("not-a-label"))
    accepted = re.search(r"accepted=([a-z0-9_,]+)", str(info.value))
    assert accepted, str(info.value)

    # Key assertion: The Python enum carries exactly the labels C++ accepts, in catalog order.
    assert [member.value for member in wire_enum] == accepted.group(1).split(",")
    assert all(isinstance(member, str) for member in wire_enum)


# REQ: PY-API-003
def test_none_valued_optional_fields_are_omitted_and_defaults_match_cpp():
    # Key setup: Lower declarations that leave every optional field at its default.
    image = typed_image_layer()._holoscan_parameters()
    clock = ClockTrigger()._holoscan_parameters()
    window = WindowSink()._holoscan_parameters()
    view = ViewTransform()._holoscan_parameters()

    # Key assertion: C++-optional fields are absent while C++-defaulted fields are emitted.
    assert image == {
        "port_name": "video",
        "kind": "image",
        "format": "rgba8_unorm",
        "max_width": 640,
        "max_height": 480,
        "color_space": "pass_through",
    }
    assert clock == {"kind": "clock"}
    assert LayerTrigger()._holoscan_parameters() == {"kind": "layer", "layer": 0}
    assert window == {
        "kind": "window",
        "title": "",
        "width": 0,
        "height": 0,
        "present_mode": "auto",
        "framebuffer_srgb": False,
        "fullscreen": False,
        "display_name": "",
        "refresh_hz": 0,
        "close_on_escape": False,
    }
    assert view == {"offset_x": 0.0, "offset_y": 0.0, "scale_x": 1.0, "scale_y": 1.0}
    explicit_none = ImageLayer(
        port_name="video", max_width=640, max_height=480, yuv_model=None, yuv_range=None, view=None
    )
    assert explicit_none._holoscan_parameters() == image

    # Emitted defaults must equal the C++ defaults a bare dict inherits: compare plan bytes.
    typed_plan = parity_plan(typed_image_layer(), WindowSink(), ClockTrigger())
    dict_plan = parity_plan(image_layer(), {"kind": "window"}, {"kind": "clock"})
    assert typed_plan == dict_plan


# REQ: PY-API-003
@pytest.mark.parametrize(
    ("typed", "tagged"),
    [
        (
            {"layers": [ImageLayer(port_name="video", max_width=0, max_height=480)]},
            {"layers": [{**image_layer(), "max_width": 0}]},
        ),
        (
            {"layers": [TextLayer(port_name="text", max_glyphs=-1)]},
            {"layers": [{"kind": "text", "port_name": "text", "max_glyphs": -1}]},
        ),
        (
            {
                "layers": [
                    ImageLayer(port_name="video", max_width=640, max_height=480, format="nv12")
                ]
            },
            {"layers": [{**image_layer(), "format": "nv12"}]},
        ),
        (
            {
                "layers": [
                    ImageLayer(port_name="video", max_width=640, max_height=480, yuv_model="bt709")
                ]
            },
            {"layers": [{**image_layer(), "yuv_model": "bt709"}]},
        ),
        (
            {"layers": [typed_image_layer(), typed_image_layer()]},
            {"layers": [image_layer(), image_layer()]},
        ),
        (
            {"layers": [typed_image_layer()], "sink": WindowSink(display_name="DP-2")},
            {"layers": [image_layer()], "sink": {**window_sink(), "display_name": "DP-2"}},
        ),
        (
            {
                "layers": [
                    ImageLayer(
                        port_name="video", max_width=640, max_height=480, format="rgba8unorm"
                    )
                ]
            },
            {"layers": [{**image_layer(), "format": "rgba8unorm"}]},
        ),
        (
            {"layers": [typed_image_layer(f"layer-{index}") for index in range(9)]},
            {"layers": [image_layer(f"layer-{index}") for index in range(9)]},
        ),
        (
            {"layers": [typed_image_layer()], "sink": CudaSink(width=0, height=480)},
            {"layers": [image_layer()], "sink": {"kind": "cuda", "width": 0, "height": 480}},
        ),
        # Explicit ``None`` in a field that is not optional in C++ must reach the catalog rather
        # than being dropped into the descriptor default.
        (
            {"layers": [ImageLayer(port_name="video", max_width=640, max_height=480, format=None)]},
            {"layers": [{**image_layer(), "format": None}]},
        ),
        (
            {
                "layers": [
                    ImageLayer(port_name="video", max_width=640, max_height=480, color_space=None)
                ]
            },
            {"layers": [{**image_layer(), "color_space": None}]},
        ),
        (
            {"layers": [ImageLayer(port_name="video", max_width=None, max_height=480)]},
            {"layers": [{**image_layer(), "max_width": None}]},
        ),
        (
            {"layers": [ImageLayer(port_name=None, max_width=640, max_height=480)]},
            {"layers": [{**image_layer(), "port_name": None}]},
        ),
        (
            {"layers": [typed_image_layer()], "sink": CudaSink(width=640, height=480, format=None)},
            {
                "layers": [image_layer()],
                "sink": {"kind": "cuda", "width": 640, "height": 480, "format": None},
            },
        ),
    ],
)
def test_typed_errors_carry_the_same_tokens_and_field_paths_as_dicts(typed, tagged):
    # Key setup: Author the same invalid declaration through both paths.
    graph = hs.Graph("visualization-typed-diagnostics")
    with pytest.raises(hs.errors.InvalidArgumentError) as typed_info:
        HolovizOp(graph, "typed", **{"sink": typed_window_sink(), **typed})
    with pytest.raises(hs.errors.InvalidArgumentError) as tagged_info:
        HolovizOp(graph, "tagged", **{"sink": window_sink(), **tagged})

    # Key assertion: No Python-side validation exists, so the C++ diagnostics are identical and
    # neither attempt adds an operator.
    assert typed_info.value.code is tagged_info.value.code
    assert typed_info.value.token == tagged_info.value.token
    assert str(typed_info.value) == str(tagged_info.value)
    assert "operators=0" in repr(graph)


# REQ: PY-API-003
def test_typed_declarations_are_frozen_keyword_only_value_objects():
    # Key setup: Exercise the dataclass contract every typed declaration shares.
    layer = typed_image_layer()

    # Key assertion: Declarations are immutable, hashable, keyword-only, and carry a fixed kind.
    with pytest.raises(dataclasses.FrozenInstanceError):
        layer.max_width = 1  # type: ignore[misc]
    assert layer == typed_image_layer()
    assert len({layer, typed_image_layer(), typed_image_layer("other")}) == 2
    with pytest.raises(TypeError):
        ImageLayer("video", 640, 480)  # type: ignore[misc]
    with pytest.raises(TypeError):
        ImageLayer(kind="image", port_name="video", max_width=640, max_height=480)  # type: ignore[call-arg]
    for declaration in _TYPED_DECLARATIONS:
        assert dataclasses.is_dataclass(declaration)
        params = declaration.__dataclass_params__
        assert params.frozen
        assert params.kw_only
        assert "kind" not in {field.name for field in dataclasses.fields(declaration)}
    assert ImageLayer.kind == "image"
    assert GeometryLayer.kind == "geometry"
    assert TextLayer.kind == "text"
    assert LutLayer.kind == "lut"
    assert WindowSink.kind == "window"
    assert ExclusiveDisplaySink.kind == "exclusive_display"
    assert CudaSink.kind == "cuda"
    assert LayerTrigger.kind == "layer"
    assert ClockTrigger.kind == "clock"
    assert not hasattr(ViewTransform, "kind")


# REQ: PY-API-003
def test_typed_declarations_snapshot_caller_owned_containers():
    # Key setup: Construct layers from caller-owned list, tuple, generic-sequence, and dict values.
    topologies = [GeometryTopology.LINE_LIST]
    geometry = GeometryLayer(port_name="shape", topologies=topologies, max_primitives=1)
    view = {"scale_x": 0.5}
    image = ImageLayer(port_name="video", max_width=640, max_height=480, view=view)
    topologies.append(GeometryTopology.OVAL_LIST)
    view["scale_y"] = 2.0

    # Key assertion: Later caller mutation reaches neither the stored value nor the lowering.
    assert geometry.topologies == (GeometryTopology.LINE_LIST,)
    assert geometry._holoscan_parameters()["topologies"] == ["line_list"]
    assert image.view == {"scale_x": 0.5}
    assert image._holoscan_parameters()["view"] == {"scale_x": 0.5}
    assert type(image._holoscan_parameters()["view"]) is dict
    assert GeometryLayer(
        port_name="shape", topologies=("line_list",), max_primitives=1
    ).topologies == ("line_list",)
    assert GeometryLayer(port_name="shape", topologies=range(0), max_primitives=1).topologies == ()

    # Key assertion: Stored snapshots are read-only, so well-formed declarations are hashable.
    assert isinstance(image.view, dict)
    with pytest.raises(TypeError, match="read-only"):
        image.view["scale_x"] = 1.0  # type: ignore[index]
    with pytest.raises(TypeError, match="read-only"):
        image.view.update(scale_x=1.0)  # type: ignore[union-attr]
    typed_view = ImageLayer(
        port_name="video", max_width=640, max_height=480, view=ViewTransform(scale_x=0.5)
    )
    assert typed_view.view == ViewTransform(scale_x=0.5)
    assert len({geometry, image, typed_view, dataclasses.replace(image)}) == 3

    # Key assertion: Value-object utilities keep working on the snapshots.
    assert dataclasses.asdict(image)["view"] == {"scale_x": 0.5}
    assert copy.deepcopy(image) == image
    assert dataclasses.replace(image, view={"offset_x": 1.0}).view == {"offset_x": 1.0}

    # Key assertion: Malformed values are stored unchanged for the catalog to reject.
    assert (
        GeometryLayer(
            port_name="shape",
            topologies="line_list",
            max_primitives=1,  # type: ignore[arg-type]
        ).topologies
        == "line_list"
    )

    # Key assertion: A snapshotted view dictionary compiles to the same plan as the tagged form.
    trigger = {"kind": "layer", "layer": 0}
    assert parity_plan(image, window_sink(), trigger) == parity_plan(
        {**image_layer(), "view": {"scale_x": 0.5}}, window_sink(), trigger
    )


# REQ: PY-API-018
def test_runtime_and_committed_stub_exports_and_declarations_match():
    # Key setup: Read the committed stub and index its enum and declaration classes.
    tree = ast.parse(_STUB_PATH.read_text(encoding="utf-8"))
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    stub_all = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "__all__"
    )

    # Key assertion: Exports, enum members and values, and constructor keywords do not drift.
    assert ast.literal_eval(stub_all) == visualization_operators.__all__
    assert visualization_operators.__all__ == sorted(visualization_operators.__all__)
    for wire_enum in _WIRE_ENUMS:
        stub_members = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in classes[wire_enum.__name__].body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        }
        assert stub_members == {member.name: member.value for member in wire_enum}
    for declaration in _TYPED_DECLARATIONS:
        # The stub must advertise the runtime dataclass contract (read-only attributes,
        # ``dataclasses.replace``/``asdict``), not a plain mutable class.
        decorator = next(
            node
            for node in classes[declaration.__name__].decorator_list
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "dataclass"
        )
        stub_options = {
            keyword.arg: ast.literal_eval(keyword.value) for keyword in decorator.keywords
        }
        assert stub_options == {"frozen": True, "kw_only": True, "slots": True}
        init = next(
            node
            for node in classes[declaration.__name__].body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        assert not init.args.args[1:]  # keyword-only after self
        stub_names = [argument.arg for argument in init.args.kwonlyargs]
        stub_required = {
            argument.arg
            for argument, default in zip(init.args.kwonlyargs, init.args.kw_defaults, strict=True)
            if default is None
        }
        fields = dataclasses.fields(declaration)
        assert stub_names == [field.name for field in fields]
        assert stub_required == {
            field.name for field in fields if field.default is dataclasses.MISSING
        }
        assert any(
            isinstance(node, ast.FunctionDef) and node.name == "_holoscan_parameters"
            for node in classes[declaration.__name__].body
        )
