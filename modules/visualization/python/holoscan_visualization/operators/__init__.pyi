# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from datetime import timedelta
from typing import Final, Literal, Required, TypedDict

import holoscan
from holoscan.operators import OperatorHandle as _OperatorHandle
from holoscan.registry import OperatorParameterInfo as _OperatorParameterInfo
from holoscan.registry import OperatorPortInfo as _OperatorPortInfo
from holoscan.time import Duration

type _ImageFormat = Literal[
    "r8_unorm",
    "rgb8_unorm",
    "rgba8_unorm",
    "nv12",
    "yuyv422",
    "bgra8_unorm",
    "r16_unorm",
    "rgba16_unorm",
    "rgba8_srgb",
    "bgra8_srgb",
]
type _PresentMode = Literal[
    "auto", "fifo", "fifo_latest_ready", "mailbox", "immediate", "fifo_relaxed"
]

class _ViewTransform(TypedDict, total=False):
    offset_x: float
    offset_y: float
    scale_x: float
    scale_y: float

class _ImageLayer(TypedDict, total=False):
    port_name: Required[str]
    kind: Required[Literal["image"]]
    format: _ImageFormat
    max_width: Required[int]
    max_height: Required[int]
    color_space: Literal["pass_through", "srgb_nonlinear", "bt709_linear"]
    yuv_model: Literal["bt601", "bt709", "bt2020"] | None
    yuv_range: Literal["limited", "full"] | None
    view: _ViewTransform | None

class _GeometryLayer(TypedDict, total=False):
    port_name: Required[str]
    kind: Required[Literal["geometry"]]
    topologies: Required[
        list[Literal["line_list", "line_strip", "cross_list", "rectangle_list", "oval_list"]]
        | tuple[
            Literal["line_list", "line_strip", "cross_list", "rectangle_list", "oval_list"], ...
        ]
    ]
    max_primitives: Required[int]
    default_line_width: float

class _TextLayer(TypedDict, total=False):
    port_name: Required[str]
    kind: Required[Literal["text"]]
    max_glyphs: Required[int]
    # Interpret record x/y as framebuffer fractions instead of the default pixel coordinates.
    normalized_coordinates: bool

class _LutLayer(TypedDict, total=False):
    port_name: Required[str]
    kind: Required[Literal["lut"]]
    lut_size: Required[int]
    max_width: Required[int]
    max_height: Required[int]

type _Layer = _ImageLayer | _GeometryLayer | _TextLayer | _LutLayer

class _LayerTrigger(TypedDict, total=False):
    kind: Required[Literal["layer"]]
    layer: int

class _ClockTrigger(TypedDict, total=False):
    kind: Required[Literal["clock"]]
    period: Duration | timedelta

type _Trigger = _LayerTrigger | _ClockTrigger

class _WindowSink(TypedDict, total=False):
    kind: Required[Literal["window"]]
    title: str
    width: int
    height: int
    present_mode: _PresentMode
    framebuffer_srgb: bool
    fullscreen: bool
    # Monitor and mode selectors are rejected unless fullscreen is true.
    display_name: str
    refresh_hz: int
    close_on_escape: bool

class _ExclusiveDisplaySink(TypedDict, total=False):
    kind: Required[Literal["exclusive_display"]]
    display_name: str
    width: int
    height: int
    refresh_hz: int
    present_mode: _PresentMode
    framebuffer_srgb: bool

class _CudaSink(TypedDict, total=False):
    kind: Required[Literal["cuda"]]
    width: Required[int]
    height: Required[int]
    format: _ImageFormat

type _Sink = _WindowSink | _ExclusiveDisplaySink | _CudaSink

class _HolovizOpType:
    @property
    def Handle(self) -> type[_OperatorHandle]: ...  # noqa: N802
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def parameter_introspection(self) -> str: ...
    @property
    def parameter_summary(self) -> str: ...
    @property
    def ports(self) -> Mapping[str, _OperatorPortInfo]: ...
    @property
    def parameters(self) -> Mapping[str, _OperatorParameterInfo]: ...
    @property
    def required_parameters(self) -> frozenset[str]: ...
    def __call__(
        self,
        graph: holoscan.Graph,
        instance_name: str,
        *,
        layers: list[_Layer] | tuple[_Layer, ...],
        sink: _Sink,
        trigger: _Trigger | None = None,
        max_input_age: Duration | timedelta | None = None,
    ) -> _OperatorHandle: ...
    def __repr__(self) -> str: ...

HolovizOp: Final[_HolovizOpType]

__all__: list[str] = ["HolovizOp"]
