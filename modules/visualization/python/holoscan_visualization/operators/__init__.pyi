# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import ClassVar, Final, Literal, Required, TypedDict, overload

import holoscan
from holoscan.operators import OperatorHandle as _OperatorHandle
from holoscan.registry import OperatorParameterInfo as _OperatorParameterInfo
from holoscan.registry import OperatorPortInfo as _OperatorPortInfo
from holoscan.time import Duration

# Wire labels. Member values are the exact strings the C++ operator catalog accepts.

class ImageFormat(enum.StrEnum):
    R8_UNORM = "r8_unorm"
    RGB8_UNORM = "rgb8_unorm"
    RGBA8_UNORM = "rgba8_unorm"
    NV12 = "nv12"
    YUYV422 = "yuyv422"
    BGRA8_UNORM = "bgra8_unorm"
    R16_UNORM = "r16_unorm"
    RGBA16_UNORM = "rgba16_unorm"
    RGBA8_SRGB = "rgba8_srgb"
    BGRA8_SRGB = "bgra8_srgb"

class PresentMode(enum.StrEnum):
    AUTO = "auto"
    FIFO = "fifo"
    FIFO_LATEST_READY = "fifo_latest_ready"
    MAILBOX = "mailbox"
    IMMEDIATE = "immediate"
    FIFO_RELAXED = "fifo_relaxed"

class ColorSpace(enum.StrEnum):
    PASS_THROUGH = "pass_through"
    SRGB_NONLINEAR = "srgb_nonlinear"
    BT709_LINEAR = "bt709_linear"

class YuvModel(enum.StrEnum):
    BT601 = "bt601"
    BT709 = "bt709"
    BT2020 = "bt2020"

class YuvRange(enum.StrEnum):
    LIMITED = "limited"
    FULL = "full"

class GeometryTopology(enum.StrEnum):
    LINE_LIST = "line_list"
    LINE_STRIP = "line_strip"
    CROSS_LIST = "cross_list"
    RECTANGLE_LIST = "rectangle_list"
    OVAL_LIST = "oval_list"

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
type _ColorSpace = Literal["pass_through", "srgb_nonlinear", "bt709_linear"]
type _YuvModel = Literal["bt601", "bt709", "bt2020"]
type _YuvRange = Literal["limited", "full"]
type _Topology = Literal["line_list", "line_strip", "cross_list", "rectangle_list", "oval_list"]

# Tagged dictionaries: the hand-authored declaration form.

class _ViewTransform(TypedDict, total=False):
    offset_x: float
    offset_y: float
    scale_x: float
    scale_y: float

class _ImageLayer(TypedDict, total=False):
    port_name: Required[str]
    kind: Required[Literal["image"]]
    format: ImageFormat | _ImageFormat
    max_width: Required[int]
    max_height: Required[int]
    color_space: ColorSpace | _ColorSpace
    yuv_model: YuvModel | _YuvModel | None
    yuv_range: YuvRange | _YuvRange | None
    # Only a view dictionary: a typed ``ViewTransform`` is not lowered inside a layer dictionary
    # and the facade rejects it with ``TypeError``. Use ``ImageLayer(view=ViewTransform(...))``.
    view: _ViewTransform | None

class _GeometryLayer(TypedDict, total=False):
    port_name: Required[str]
    kind: Required[Literal["geometry"]]
    # ``list`` is invariant, so each homogeneous list form is its own arm; the native dictionary
    # converter accepts only lists and tuples, unlike the typed ``GeometryLayer`` lowering.
    topologies: Required[
        list[_Topology]
        | list[GeometryTopology]
        | list[GeometryTopology | _Topology]
        | tuple[GeometryTopology | _Topology, ...]
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

class _LayerTrigger(TypedDict, total=False):
    kind: Required[Literal["layer"]]
    layer: int

class _ClockTrigger(TypedDict, total=False):
    kind: Required[Literal["clock"]]
    period: Duration | timedelta

class _WindowSink(TypedDict, total=False):
    kind: Required[Literal["window"]]
    title: str
    width: int
    height: int
    present_mode: PresentMode | _PresentMode
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
    present_mode: PresentMode | _PresentMode
    framebuffer_srgb: bool

class _CudaSink(TypedDict, total=False):
    kind: Required[Literal["cuda"]]
    width: Required[int]
    height: Required[int]
    format: ImageFormat | _ImageFormat

# Typed declarations: frozen, keyword-only, slotted dataclasses that lower to the tagged
# dictionaries. The decorator is part of the contract: attributes are read-only and
# ``dataclasses.replace``/``asdict`` work. Defaults mirror the C++ descriptor defaults; only the
# C++-optional fields (``None`` defaults) are omitted when lowering. Container fields are
# snapshotted at construction: ``topologies`` is stored as a tuple and a ``view`` dictionary as a
# read-only mapping, so later changes to the caller's list or dictionary do not reach the
# declaration and well-formed declarations are hashable.

# ``Sequence`` is covariant, so preconstructed ``list[GeometryTopology]`` or narrower literal lists
# are accepted without casts; the runtime stores any non-string sequence as a tuple.
type _Topologies = Sequence[GeometryTopology | _Topology]

@dataclass(frozen=True, kw_only=True, slots=True)
class ViewTransform:
    offset_x: float
    offset_y: float
    scale_x: float
    scale_y: float
    def __init__(
        self,
        *,
        offset_x: float = ...,
        offset_y: float = ...,
        scale_x: float = ...,
        scale_y: float = ...,
    ) -> None: ...
    def _holoscan_parameters(self) -> dict[str, object]: ...

@dataclass(frozen=True, kw_only=True, slots=True)
class ImageLayer:
    kind: ClassVar[Literal["image"]]
    port_name: str
    max_width: int
    max_height: int
    format: ImageFormat | _ImageFormat
    color_space: ColorSpace | _ColorSpace
    yuv_model: YuvModel | _YuvModel | None
    yuv_range: YuvRange | _YuvRange | None
    view: ViewTransform | Mapping[str, float] | None
    def __init__(
        self,
        *,
        port_name: str,
        max_width: int,
        max_height: int,
        format: ImageFormat | _ImageFormat = ...,  # noqa: A002
        color_space: ColorSpace | _ColorSpace = ...,
        yuv_model: YuvModel | _YuvModel | None = ...,
        yuv_range: YuvRange | _YuvRange | None = ...,
        view: ViewTransform | _ViewTransform | None = ...,
    ) -> None: ...
    def __post_init__(self) -> None: ...
    def _holoscan_parameters(self) -> dict[str, object]: ...

@dataclass(frozen=True, kw_only=True, slots=True)
class GeometryLayer:
    kind: ClassVar[Literal["geometry"]]
    port_name: str
    topologies: tuple[GeometryTopology | _Topology, ...]
    max_primitives: int
    default_line_width: float
    def __init__(
        self,
        *,
        port_name: str,
        topologies: _Topologies,
        max_primitives: int,
        default_line_width: float = ...,
    ) -> None: ...
    def __post_init__(self) -> None: ...
    def _holoscan_parameters(self) -> dict[str, object]: ...

@dataclass(frozen=True, kw_only=True, slots=True)
class TextLayer:
    kind: ClassVar[Literal["text"]]
    port_name: str
    max_glyphs: int
    # Interpret record x/y as framebuffer fractions instead of the default pixel coordinates.
    normalized_coordinates: bool
    def __init__(
        self, *, port_name: str, max_glyphs: int, normalized_coordinates: bool = ...
    ) -> None: ...
    def _holoscan_parameters(self) -> dict[str, object]: ...

@dataclass(frozen=True, kw_only=True, slots=True)
class LutLayer:
    kind: ClassVar[Literal["lut"]]
    port_name: str
    lut_size: int
    max_width: int
    max_height: int
    def __init__(
        self, *, port_name: str, lut_size: int, max_width: int, max_height: int
    ) -> None: ...
    def _holoscan_parameters(self) -> dict[str, object]: ...

@dataclass(frozen=True, kw_only=True, slots=True)
class WindowSink:
    kind: ClassVar[Literal["window"]]
    title: str
    width: int
    height: int
    present_mode: PresentMode | _PresentMode
    framebuffer_srgb: bool
    fullscreen: bool
    # Monitor and mode selectors are rejected unless fullscreen is true.
    display_name: str
    refresh_hz: int
    close_on_escape: bool
    def __init__(
        self,
        *,
        title: str = ...,
        width: int = ...,
        height: int = ...,
        present_mode: PresentMode | _PresentMode = ...,
        framebuffer_srgb: bool = ...,
        fullscreen: bool = ...,
        display_name: str = ...,
        refresh_hz: int = ...,
        close_on_escape: bool = ...,
    ) -> None: ...
    def _holoscan_parameters(self) -> dict[str, object]: ...

@dataclass(frozen=True, kw_only=True, slots=True)
class ExclusiveDisplaySink:
    kind: ClassVar[Literal["exclusive_display"]]
    display_name: str
    width: int
    height: int
    refresh_hz: int
    present_mode: PresentMode | _PresentMode
    framebuffer_srgb: bool
    def __init__(
        self,
        *,
        display_name: str = ...,
        width: int = ...,
        height: int = ...,
        refresh_hz: int = ...,
        present_mode: PresentMode | _PresentMode = ...,
        framebuffer_srgb: bool = ...,
    ) -> None: ...
    def _holoscan_parameters(self) -> dict[str, object]: ...

@dataclass(frozen=True, kw_only=True, slots=True)
class CudaSink:
    kind: ClassVar[Literal["cuda"]]
    width: int
    height: int
    format: ImageFormat | _ImageFormat
    def __init__(
        self,
        *,
        width: int,
        height: int,
        format: ImageFormat | _ImageFormat = ...,  # noqa: A002
    ) -> None: ...
    def _holoscan_parameters(self) -> dict[str, object]: ...

@dataclass(frozen=True, kw_only=True, slots=True)
class LayerTrigger:
    kind: ClassVar[Literal["layer"]]
    layer: int
    def __init__(self, *, layer: int = ...) -> None: ...
    def _holoscan_parameters(self) -> dict[str, object]: ...

@dataclass(frozen=True, kw_only=True, slots=True)
class ClockTrigger:
    kind: ClassVar[Literal["clock"]]
    period: Duration | timedelta | None
    def __init__(self, *, period: Duration | timedelta | None = ...) -> None: ...
    def _holoscan_parameters(self) -> dict[str, object]: ...

type _Layer = (
    _ImageLayer
    | _GeometryLayer
    | _TextLayer
    | _LutLayer
    | ImageLayer
    | GeometryLayer
    | TextLayer
    | LutLayer
)
type _Trigger = _LayerTrigger | _ClockTrigger | LayerTrigger | ClockTrigger
type _Sink = (
    _WindowSink | _ExclusiveDisplaySink | _CudaSink | WindowSink | ExclusiveDisplaySink | CudaSink
)

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
    # Overload 1 keeps contextual TypedDict inference for inline layer lists; overload 2 accepts
    # preconstructed homogeneous lists (e.g. ``list[ImageLayer]``), which the invariant
    # ``list[_Layer]`` would reject.
    @overload
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
    @overload
    def __call__[LayerT: _Layer](
        self,
        graph: holoscan.Graph,
        instance_name: str,
        *,
        layers: list[LayerT],
        sink: _Sink,
        trigger: _Trigger | None = None,
        max_input_age: Duration | timedelta | None = None,
    ) -> _OperatorHandle: ...
    def __repr__(self) -> str: ...

HolovizOp: Final[_HolovizOpType]

__all__: list[str] = [
    "ClockTrigger",
    "ColorSpace",
    "CudaSink",
    "ExclusiveDisplaySink",
    "GeometryLayer",
    "GeometryTopology",
    "HolovizOp",
    "ImageFormat",
    "ImageLayer",
    "LayerTrigger",
    "LutLayer",
    "PresentMode",
    "TextLayer",
    "ViewTransform",
    "WindowSink",
    "YuvModel",
    "YuvRange",
]
