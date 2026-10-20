# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Positive typing contracts for the typed Holoviz declarations (pyright and ty)."""

import dataclasses
from typing import Any, Literal, assert_type

from holoscan_visualization.operators import (
    ClockTrigger,
    ColorSpace,
    GeometryLayer,
    GeometryTopology,
    HolovizOp,
    ImageFormat,
    ImageLayer,
    PresentMode,
    WindowSink,
    YuvModel,
    YuvRange,
)

import holoscan as hs
from holoscan.operators import OperatorHandle

graph = hs.Graph("typing-visualization-positive")

# Frozen dataclass identity: immutable update and introspection type-check.
layer = ImageLayer(port_name="video", max_width=640, max_height=480, format=ImageFormat.R16_UNORM)
assert_type(layer.max_width, int)
assert_type(dataclasses.replace(layer, max_width=1280), ImageLayer)
assert_type(dataclasses.asdict(layer), dict[str, Any])
assert_type(ImageLayer.kind, Literal["image"])

# Topologies: preconstructed homogeneous enum lists, wire-label lists, mixed lists, and tuples are
# all accepted without casts, as are inline literals.
enum_topologies = [GeometryTopology.LINE_LIST, GeometryTopology.OVAL_LIST]
label_topologies: list[Literal["line_list", "oval_list"]] = ["line_list", "oval_list"]
mixed_topologies: list[GeometryTopology | Literal["cross_list"]] = [
    GeometryTopology.LINE_STRIP,
    "cross_list",
]
tuple_topologies = (GeometryTopology.RECTANGLE_LIST, "oval_list")
GeometryLayer(port_name="enum-list", topologies=enum_topologies, max_primitives=3)
GeometryLayer(port_name="label-list", topologies=label_topologies, max_primitives=3)
GeometryLayer(port_name="mixed-list", topologies=mixed_topologies, max_primitives=3)
GeometryLayer(port_name="tuple", topologies=tuple_topologies, max_primitives=3)
GeometryLayer(port_name="inline", topologies=["line_list", "oval_list"], max_primitives=3)

# Typed and tagged-dictionary declarations mix in one authoring call.
handle = HolovizOp(
    graph,
    "visualizer",
    layers=[
        layer,
        {"kind": "image", "port_name": "overlay", "max_width": 640, "max_height": 480},
        GeometryLayer(port_name="shape", topologies=enum_topologies, max_primitives=3),
    ],
    sink=WindowSink(title="Typing", present_mode=PresentMode.FIFO),
    trigger=ClockTrigger(period=hs.time.ms(16)),
)
assert_type(handle, OperatorHandle)

# Preconstructed homogeneous lists of typed declarations are accepted despite list invariance.
images = [ImageLayer(port_name="video", max_width=640, max_height=480)]
geometries = [GeometryLayer(port_name="shape", topologies=enum_topologies, max_primitives=3)]
HolovizOp(graph, "images", layers=images, sink=WindowSink())
HolovizOp(graph, "geometries", layers=geometries, sink=WindowSink())

# Tagged dictionaries accept the public enums wherever they accept the wire labels.
HolovizOp(
    graph,
    "enum-dictionaries",
    layers=[
        {
            "kind": "image",
            "port_name": "video",
            "format": ImageFormat.NV12,
            "max_width": 640,
            "max_height": 480,
            "color_space": ColorSpace.PASS_THROUGH,
            "yuv_model": YuvModel.BT709,
            "yuv_range": YuvRange.FULL,
        },
        {
            "kind": "geometry",
            "port_name": "shape",
            "topologies": [GeometryTopology.OVAL_LIST],
            "max_primitives": 3,
        },
    ],
    sink={"kind": "window", "width": 640, "height": 480, "present_mode": PresentMode.FIFO},
)
HolovizOp(
    graph,
    "exclusive-display-dictionary",
    layers=images,
    sink={"kind": "exclusive_display", "present_mode": PresentMode.MAILBOX},
)
HolovizOp(
    graph,
    "cuda-dictionary",
    layers=images,
    sink={"kind": "cuda", "width": 640, "height": 480, "format": ImageFormat.RGBA8_UNORM},
)

# Preconstructed topology lists inside tagged dictionaries: wire-label, enum, and mixed lists stay
# valid despite list invariance, as do tuples.
type WireTopology = Literal["line_list", "line_strip", "cross_list", "rectangle_list", "oval_list"]
wire_list: list[WireTopology] = ["line_list"]
enum_list: list[GeometryTopology] = [GeometryTopology.OVAL_LIST]
mixed_list: list[GeometryTopology | WireTopology] = [GeometryTopology.LINE_STRIP, "cross_list"]
HolovizOp(
    graph,
    "topology-dictionaries",
    layers=[
        {"kind": "geometry", "port_name": "wire", "topologies": wire_list, "max_primitives": 3},
        {"kind": "geometry", "port_name": "enum", "topologies": enum_list, "max_primitives": 3},
        {"kind": "geometry", "port_name": "mixed", "topologies": mixed_list, "max_primitives": 3},
        {
            "kind": "geometry",
            "port_name": "tuple",
            "topologies": tuple_topologies,
            "max_primitives": 3,
        },
    ],
    sink={"kind": "window"},
)
