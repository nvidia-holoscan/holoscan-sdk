# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Holoscan Visualization operator bindings."""

import enum as _enum
from collections.abc import Sequence as _Sequence
from dataclasses import dataclass as _dataclass
from datetime import timedelta as _timedelta
from typing import Any as _Any
from typing import ClassVar as _ClassVar
from typing import Literal as _Literal
from typing import NoReturn as _NoReturn

from holoscan.registry import resolve as _resolve
from holoscan.time import Duration as _Duration


def _duration_nanoseconds(value: _Duration | _timedelta) -> int:
    if isinstance(value, _Duration):
        return value.nanoseconds
    if not isinstance(value, _timedelta):
        raise TypeError("duration must be datetime.timedelta or holoscan.time.Duration")
    return (
        value.days * 86_400_000_000_000 + value.seconds * 1_000_000_000 + value.microseconds * 1_000
    )


def _normalize_trigger(trigger: object) -> object:
    """Copy and lower the public clock duration without mutating caller-owned input."""

    if isinstance(trigger, dict) and "period_ns" in trigger:
        raise TypeError("use period with datetime.timedelta or holoscan.time.Duration")
    if not isinstance(trigger, dict) or trigger.get("kind") != "clock" or "period" not in trigger:
        return trigger
    normalized = dict(trigger)
    normalized["period_ns"] = _duration_nanoseconds(normalized.pop("period"))
    return normalized


# ---- Wire labels ----------------------------------------------------------------------------
#
# Each member's value is the exact string the C++ operator catalog accepts. These label Holoviz
# rendering choices; they are unrelated to the payload metadata enums in ``holoscan.schema.image``.


class ImageFormat(_enum.StrEnum):
    """Texture formats accepted by ``format`` on image layers and CUDA sinks."""

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


class PresentMode(_enum.StrEnum):
    """Swapchain present modes accepted by ``present_mode`` on window and display sinks."""

    AUTO = "auto"
    FIFO = "fifo"
    FIFO_LATEST_READY = "fifo_latest_ready"
    MAILBOX = "mailbox"
    IMMEDIATE = "immediate"
    FIFO_RELAXED = "fifo_relaxed"


class ColorSpace(_enum.StrEnum):
    """Source transfer functions accepted by ``color_space`` on image layers."""

    PASS_THROUGH = "pass_through"
    SRGB_NONLINEAR = "srgb_nonlinear"
    BT709_LINEAR = "bt709_linear"


class YuvModel(_enum.StrEnum):
    """YUV model conversions accepted by ``yuv_model`` on image layers."""

    BT601 = "bt601"
    BT709 = "bt709"
    BT2020 = "bt2020"


class YuvRange(_enum.StrEnum):
    """YUV quantization ranges accepted by ``yuv_range`` on image layers."""

    LIMITED = "limited"
    FULL = "full"


class GeometryTopology(_enum.StrEnum):
    """Primitive topologies accepted in ``topologies`` on geometry layers."""

    LINE_LIST = "line_list"
    LINE_STRIP = "line_strip"
    CROSS_LIST = "cross_list"
    RECTANGLE_LIST = "rectangle_list"
    OVAL_LIST = "oval_list"


# ---- Lowering helpers -----------------------------------------------------------------------
#
# Typed declarations lower to the same tagged dictionaries callers may still author by hand.
# Lowering is pure conversion: every structural and semantic rule stays in the C++ catalog so
# both authoring paths report identical diagnostics. The one Python-side check rejects a typed
# ``ViewTransform`` nested in a hand-authored image-layer dictionary, which is deliberately not
# lowered, with an actionable ``TypeError`` instead of the generic native type mismatch.


def _enum_str(value: object) -> object:
    return value.value if isinstance(value, _enum.Enum) else value


def _enum_strs(values: object) -> object:
    # Any non-string sequence (list, tuple, or another ``Sequence``) lowers to a list of labels.
    if isinstance(values, _Sequence) and not isinstance(values, str):
        return [_enum_str(item) for item in values]
    return values


def _lower_declaration(value: object) -> object:
    """Lower one typed declaration; dictionaries and other values pass through untouched."""

    hook = getattr(value, "_holoscan_parameters", None)
    return value if hook is None else hook()


def _reject_typed_view_in_dict_layer(index: int, layer: object) -> None:
    if (
        isinstance(layer, dict)
        and layer.get("kind") == "image"
        and isinstance(layer.get("view"), ViewTransform)
    ):
        raise TypeError(
            f"layers[{index}].view: ViewTransform is supported inside ImageLayer, but not inside "
            "a layer dictionary; use ImageLayer(..., view=ViewTransform(...)) or a view "
            'dictionary such as {"scale_x": 0.5}.'
        )


def _lower_layers(layers: object) -> object:
    if isinstance(layers, list | tuple):
        for index, layer in enumerate(layers):
            _reject_typed_view_in_dict_layer(index, layer)
        return [_lower_declaration(layer) for layer in layers]
    return layers


def _lower_trigger(trigger: object) -> object:
    hook = getattr(trigger, "_holoscan_parameters", None)
    return _normalize_trigger(trigger) if hook is None else hook()


class _FrozenView(dict[str, object]):
    """Read-only, hashable snapshot of a caller's ``view`` dictionary on ``ImageLayer``.

    A ``dict`` subclass rather than ``types.MappingProxyType`` so ``dataclasses.asdict`` and
    ``copy`` keep working on the declaration.
    """

    __slots__ = ()

    def _read_only(self) -> _NoReturn:
        raise TypeError("ImageLayer.view is read-only; use dataclasses.replace to change it")

    def __setitem__(self, _key: object, _value: object, /) -> _NoReturn:
        self._read_only()

    def __delitem__(self, _key: object, /) -> _NoReturn:
        self._read_only()

    def __ior__(self, _value: object, /) -> _NoReturn:  # type: ignore[override,misc]
        self._read_only()

    def update(self, *_args: object, **_kwargs: object) -> _NoReturn:
        self._read_only()

    def pop(self, *_args: object) -> _NoReturn:
        self._read_only()

    def popitem(self) -> _NoReturn:
        self._read_only()

    def clear(self) -> _NoReturn:
        self._read_only()

    def setdefault(self, *_args: object) -> _NoReturn:
        self._read_only()

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(frozenset(self.items()))

    def __reduce__(self) -> tuple[type[dict[str, object]], tuple[dict[str, object]]]:
        return (type(self), (dict(self),))


# ---- Typed declarations ---------------------------------------------------------------------
#
# Field defaults mirror the C++ descriptor defaults. Only the fields that are optional in C++
# default to ``None`` and are omitted from the lowered dictionary when ``None``; every other value,
# including an explicit ``None``, is passed through so the catalog owns its rejection. Enum-valued
# fields accept either the ``StrEnum`` member or its wire string. Container fields are snapshotted
# at construction (``topologies`` as a tuple, a ``view`` dictionary as a read-only copy), so later
# changes to the caller's list or dictionary do not reach the declaration; malformed values are
# stored unchanged for the catalog to reject.


@_dataclass(frozen=True, kw_only=True, slots=True)
class ViewTransform:
    """Typed form of the ``view`` mapping on an image layer."""

    offset_x: float = 0.0
    offset_y: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0

    def _holoscan_parameters(self) -> dict[str, object]:
        return {
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
        }


@_dataclass(frozen=True, kw_only=True, slots=True)
class ImageLayer:
    """Typed form of the ``{"kind": "image", ...}`` layer declaration."""

    kind: _ClassVar[_Literal["image"]] = "image"
    port_name: str
    max_width: int
    max_height: int
    format: ImageFormat | str = ImageFormat.RGBA8_UNORM
    color_space: ColorSpace | str = ColorSpace.PASS_THROUGH
    yuv_model: YuvModel | str | None = None
    yuv_range: YuvRange | str | None = None
    view: ViewTransform | dict[str, float] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.view, dict):
            object.__setattr__(self, "view", _FrozenView(self.view))

    def _holoscan_parameters(self) -> dict[str, object]:
        lowered: dict[str, object] = {
            "port_name": self.port_name,
            "kind": self.kind,
            "format": _enum_str(self.format),
            "max_width": self.max_width,
            "max_height": self.max_height,
            "color_space": _enum_str(self.color_space),
        }
        # Only the C++-optional fields are omitted when ``None``.
        if self.yuv_model is not None:
            lowered["yuv_model"] = _enum_str(self.yuv_model)
        if self.yuv_range is not None:
            lowered["yuv_range"] = _enum_str(self.yuv_range)
        if isinstance(self.view, dict):
            # Hand the native converter a plain dictionary, never the read-only snapshot type.
            lowered["view"] = dict(self.view)
        elif self.view is not None:
            lowered["view"] = _lower_declaration(self.view)
        return lowered


@_dataclass(frozen=True, kw_only=True, slots=True)
class GeometryLayer:
    """Typed form of the ``{"kind": "geometry", ...}`` layer declaration."""

    kind: _ClassVar[_Literal["geometry"]] = "geometry"
    port_name: str
    topologies: _Sequence[GeometryTopology | str]
    max_primitives: int
    default_line_width: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.topologies, _Sequence) and not isinstance(self.topologies, str):
            object.__setattr__(self, "topologies", tuple(self.topologies))

    def _holoscan_parameters(self) -> dict[str, object]:
        return {
            "port_name": self.port_name,
            "kind": self.kind,
            "topologies": _enum_strs(self.topologies),
            "max_primitives": self.max_primitives,
            "default_line_width": self.default_line_width,
        }


@_dataclass(frozen=True, kw_only=True, slots=True)
class TextLayer:
    """Typed form of the ``{"kind": "text", ...}`` layer declaration."""

    kind: _ClassVar[_Literal["text"]] = "text"
    port_name: str
    max_glyphs: int
    # Interpret record x/y as framebuffer fractions instead of the default pixel coordinates.
    normalized_coordinates: bool = False

    def _holoscan_parameters(self) -> dict[str, object]:
        return {
            "port_name": self.port_name,
            "kind": self.kind,
            "max_glyphs": self.max_glyphs,
            "normalized_coordinates": self.normalized_coordinates,
        }


@_dataclass(frozen=True, kw_only=True, slots=True)
class LutLayer:
    """Typed form of the ``{"kind": "lut", ...}`` layer declaration."""

    kind: _ClassVar[_Literal["lut"]] = "lut"
    port_name: str
    lut_size: int
    max_width: int
    max_height: int

    def _holoscan_parameters(self) -> dict[str, object]:
        return {
            "port_name": self.port_name,
            "kind": self.kind,
            "lut_size": self.lut_size,
            "max_width": self.max_width,
            "max_height": self.max_height,
        }


@_dataclass(frozen=True, kw_only=True, slots=True)
class WindowSink:
    """Typed form of the ``{"kind": "window", ...}`` sink declaration."""

    kind: _ClassVar[_Literal["window"]] = "window"
    title: str = ""
    width: int = 0
    height: int = 0
    present_mode: PresentMode | str = PresentMode.AUTO
    framebuffer_srgb: bool = False
    fullscreen: bool = False
    # Monitor and mode selectors are rejected by the catalog unless fullscreen is true.
    display_name: str = ""
    refresh_hz: int = 0
    close_on_escape: bool = False

    def _holoscan_parameters(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "title": self.title,
            "width": self.width,
            "height": self.height,
            "present_mode": _enum_str(self.present_mode),
            "framebuffer_srgb": self.framebuffer_srgb,
            "fullscreen": self.fullscreen,
            "display_name": self.display_name,
            "refresh_hz": self.refresh_hz,
            "close_on_escape": self.close_on_escape,
        }


@_dataclass(frozen=True, kw_only=True, slots=True)
class ExclusiveDisplaySink:
    """Typed form of the ``{"kind": "exclusive_display", ...}`` sink declaration."""

    kind: _ClassVar[_Literal["exclusive_display"]] = "exclusive_display"
    display_name: str = ""
    width: int = 0
    height: int = 0
    refresh_hz: int = 0
    present_mode: PresentMode | str = PresentMode.AUTO
    framebuffer_srgb: bool = False

    def _holoscan_parameters(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "display_name": self.display_name,
            "width": self.width,
            "height": self.height,
            "refresh_hz": self.refresh_hz,
            "present_mode": _enum_str(self.present_mode),
            "framebuffer_srgb": self.framebuffer_srgb,
        }


@_dataclass(frozen=True, kw_only=True, slots=True)
class CudaSink:
    """Typed form of the ``{"kind": "cuda", ...}`` sink declaration."""

    kind: _ClassVar[_Literal["cuda"]] = "cuda"
    width: int
    height: int
    format: ImageFormat | str = ImageFormat.RGBA8_UNORM

    def _holoscan_parameters(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "width": self.width,
            "height": self.height,
            "format": _enum_str(self.format),
        }


@_dataclass(frozen=True, kw_only=True, slots=True)
class LayerTrigger:
    """Typed form of the ``{"kind": "layer", ...}`` frame trigger."""

    kind: _ClassVar[_Literal["layer"]] = "layer"
    layer: int = 0

    def _holoscan_parameters(self) -> dict[str, object]:
        return {"kind": self.kind, "layer": self.layer}


@_dataclass(frozen=True, kw_only=True, slots=True)
class ClockTrigger:
    """Typed form of the ``{"kind": "clock", ...}`` frame trigger.

    ``period`` is a ``holoscan.time.Duration`` or ``datetime.timedelta``; ``None`` keeps the
    C++ default period.
    """

    kind: _ClassVar[_Literal["clock"]] = "clock"
    period: _Duration | _timedelta | None = None

    def _holoscan_parameters(self) -> dict[str, object]:
        lowered: dict[str, object] = {"kind": self.kind}
        if self.period is not None:
            lowered["period_ns"] = _duration_nanoseconds(self.period)
        return lowered


# ---- Operator facade ------------------------------------------------------------------------


class _HolovizOpType:
    """Callable module facade for the registered C++ HolovizOp."""

    def __init__(self) -> None:
        self._registered = _resolve("holoscan.visualization.HolovizOp", kind="operator")

    @property
    def Handle(self) -> object:  # noqa: N802
        return self._registered.Handle

    @property
    def name(self) -> str:
        return self._registered.name

    @property
    def description(self) -> str:
        return self._registered.description

    @property
    def parameter_introspection(self) -> str:
        return self._registered.parameter_introspection

    @property
    def parameter_summary(self) -> str:
        return self._registered.parameter_summary

    @property
    def ports(self) -> object:
        return self._registered.ports

    @property
    def parameters(self) -> object:
        return self._registered.parameters

    @property
    def required_parameters(self) -> frozenset[str]:
        return self._registered.required_parameters

    def __call__(
        self,
        graph: _Any,
        instance_name: str,
        *,
        layers: object,
        sink: object,
        trigger: object | None = None,
        max_input_age: _Duration | _timedelta | None = None,
    ) -> object:
        params: dict[str, object] = {
            "layers": _lower_layers(layers),
            "sink": _lower_declaration(sink),
        }
        if trigger is not None:
            params["trigger"] = _lower_trigger(trigger)
        if max_input_age is not None:
            params["max_input_age_ns"] = _duration_nanoseconds(max_input_age)
        return self._registered(graph, instance_name, **params)

    def __repr__(self) -> str:
        return repr(self._registered)


HolovizOp = _HolovizOpType()

__all__ = [
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
