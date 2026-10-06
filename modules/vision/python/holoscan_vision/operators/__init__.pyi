# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# fmt: off

# ruff: noqa: E501 - registry metadata intentionally remains on generated lines

"""Registered Holoscan Vision Operator proxies."""

import holoscan
from holoscan.operators import OperatorHandle as _OperatorHandle
from holoscan.registry import OperatorType as _OperatorType

__all__: list[str] = ['BayerDemosaicOp', 'ConvertToOp', 'CopyMakeBorderOp', 'CustomCropOp', 'NonMaximumSuppressionOp', 'NormalizeOp', 'ReformatOp', 'RemapOp', 'ResizeOp']


# Registry-backed operator proxy types.
import datetime
from typing import Final, Literal, Never, Required, TypedDict, Unpack

from holoscan.registry import OperatorParameterInfo as _OperatorParameterInfo
from holoscan.registry import OperatorPortInfo as _OperatorPortInfo
from holoscan.time import Duration as _time_Duration

class _BayerDemosaicOpParams(TypedDict, total=False):
    bayer_grid_pos: Literal['bggr', 0, 'rggb', 1, 'gbrg', 2, 'grbg', 3]
    generate_alpha: bool
    alpha_value: int


class _BayerDemosaicOpHandle(_OperatorHandle):
    input: holoscan.Port
    output: holoscan.Port
    def __getattr__(self, name: Never) -> Never: ...  # type: ignore[misc, override]


class _BayerDemosaicOpType(_OperatorType[_BayerDemosaicOpHandle]):
    "Registered Operator holoscan.vision.BayerDemosaicOp.\n\nRegistry name: holoscan.vision.BayerDemosaicOp\n\nPorts:\n  input: Input[holoscan.schema.Image]\n  output: Output[holoscan.schema.Image]\n\nParameters:\n  bayer_grid_pos: Literal['bggr', 0, 'rggb', 1, 'gbrg', 2, 'grbg', 3], choices bggr, rggb, gbrg, grbg — Color-filter-array phase of the input mosaic's top-left 2x2 block.\n  generate_alpha: bool — Emit a fourth, constant-valued alpha output channel.\n  alpha_value: int — Constant alpha channel value used when generate_alpha is set; passed to NPP unscaled, so 8-bit output must stay within 0-255."
    Handle: Final[type[_BayerDemosaicOpHandle]]
    Params: Final[type[_BayerDemosaicOpParams]]
    alpha_value: _OperatorParameterInfo
    bayer_grid_pos: _OperatorParameterInfo
    generate_alpha: _OperatorParameterInfo
    input: _OperatorPortInfo
    output: _OperatorPortInfo
    def __call__(self, graph: holoscan.Graph, instance_name: str, **params: Unpack[_BayerDemosaicOpParams]) -> _BayerDemosaicOpHandle: ...  # type: ignore[override]

BayerDemosaicOp: Final[_BayerDemosaicOpType]


class _ConvertToOpParams(TypedDict, total=False):
    output_type: Literal['uint8', 0, 'int8', 1, 'uint16', 2, 'int16', 3, 'int32', 4, 'float32', 5, 'float64', 6]
    alpha: float
    beta: float
    rounding: Literal['nearest', 0, 'truncate', 1]


class _ConvertToOpHandle(_OperatorHandle):
    input: holoscan.Port
    output: holoscan.Port
    def __getattr__(self, name: Never) -> Never: ...  # type: ignore[misc, override]


class _ConvertToOpType(_OperatorType[_ConvertToOpHandle]):
    "Registered Operator holoscan.vision.ConvertToOp.\n\nRegistry name: holoscan.vision.ConvertToOp\n\nPorts:\n  input: Input[holoscan.schema.Image]\n  output: Output[holoscan.schema.Image]\n\nParameters:\n  output_type: Literal['uint8', 0, 'int8', 1, 'uint16', 2, 'int16', 3, 'int32', 4, 'float32', 5, 'float64', 6], choices uint8, int8, uint16, int16, int32, float32, float64 — Scalar output element type.\n  alpha: float — Multiplier applied before saturating conversion.\n  beta: float — Offset applied before saturating conversion.\n  rounding: Literal['nearest', 0, 'truncate', 1], choices nearest, truncate — Rounding mode used for integral outputs."
    Handle: Final[type[_ConvertToOpHandle]]
    Params: Final[type[_ConvertToOpParams]]
    alpha: _OperatorParameterInfo
    beta: _OperatorParameterInfo
    input: _OperatorPortInfo
    output: _OperatorPortInfo
    output_type: _OperatorParameterInfo
    rounding: _OperatorParameterInfo
    def __call__(self, graph: holoscan.Graph, instance_name: str, **params: Unpack[_ConvertToOpParams]) -> _ConvertToOpHandle: ...  # type: ignore[override]

ConvertToOp: Final[_ConvertToOpType]


class _CopyMakeBorderOpParams(TypedDict, total=False):
    max_width: Required[int]
    max_height: Required[int]
    max_batch_size: Required[int]
    top: int
    bottom: int
    left: int
    right: int
    border_mode: Literal['constant', 0, 'replicate', 1, 'reflect', 2, 'wrap', 3, 'reflect_101', 4]
    border_value: list[float] | tuple[float, ...]


class _CopyMakeBorderOpHandle(_OperatorHandle):
    input: holoscan.Port
    output: holoscan.Port
    def __getattr__(self, name: Never) -> Never: ...  # type: ignore[misc, override]


class _CopyMakeBorderOpType(_OperatorType[_CopyMakeBorderOpHandle]):
    "Registered Operator holoscan.vision.CopyMakeBorderOp.\n\nRegistry name: holoscan.vision.CopyMakeBorderOp\n\nPorts:\n  input: Input[holoscan.schema.Image]\n  output: Output[holoscan.schema.Image]\n\nParameters:\n  max_width: int, range (1, 2147483647) — Maximum bordered output width in pixels.\n  max_height: int, range (1, 2147483647) — Maximum bordered output height in pixels.\n  max_batch_size: int, range (1, 65535) — Maximum image batch size admitted by the operator.\n  top: int, range (0, 2147483647) — Top border width in pixels.\n  bottom: int, range (0, 2147483647) — Bottom border width in pixels.\n  left: int, range (0, 2147483647) — Left border width in pixels.\n  right: int, range (0, 2147483647) — Right border width in pixels.\n  border_mode: Literal['constant', 0, 'replicate', 1, 'reflect', 2, 'wrap', 3, 'reflect_101', 4], choices constant, replicate, reflect, wrap, reflect_101 — Border extrapolation mode.\n  border_value: list[float] | tuple[float, ...] — Constant fill value: one value or one value per image channel."
    Handle: Final[type[_CopyMakeBorderOpHandle]]
    Params: Final[type[_CopyMakeBorderOpParams]]
    border_mode: _OperatorParameterInfo
    border_value: _OperatorParameterInfo
    bottom: _OperatorParameterInfo
    input: _OperatorPortInfo
    left: _OperatorParameterInfo
    max_batch_size: _OperatorParameterInfo
    max_height: _OperatorParameterInfo
    max_width: _OperatorParameterInfo
    output: _OperatorPortInfo
    right: _OperatorParameterInfo
    top: _OperatorParameterInfo
    def __call__(self, graph: holoscan.Graph, instance_name: str, **params: Unpack[_CopyMakeBorderOpParams]) -> _CopyMakeBorderOpHandle: ...  # type: ignore[override]

CopyMakeBorderOp: Final[_CopyMakeBorderOpType]


class _CustomCropOpParams(TypedDict, total=False):
    x: int
    y: int
    width: int
    height: int
    max_batch_size: Required[int]


class _CustomCropOpHandle(_OperatorHandle):
    input: holoscan.Port
    output: holoscan.Port
    def __getattr__(self, name: Never) -> Never: ...  # type: ignore[misc, override]


class _CustomCropOpType(_OperatorType[_CustomCropOpHandle]):
    'Registered Operator holoscan.vision.CustomCropOp.\n\nRegistry name: holoscan.vision.CustomCropOp\n\nPorts:\n  input: Input[holoscan.schema.Image]\n  output: Output[holoscan.schema.Image]\n\nParameters:\n  x: int, range (0, 2147483647) — Left crop coordinate in pixels.\n  y: int, range (0, 2147483647) — Top crop coordinate in pixels.\n  width: int, range (1, 2147483647) — Crop width in pixels.\n  height: int, range (1, 2147483647) — Crop height in pixels.\n  max_batch_size: int, range (1, 65535) — Maximum image batch size admitted by the operator.'
    Handle: Final[type[_CustomCropOpHandle]]
    Params: Final[type[_CustomCropOpParams]]
    height: _OperatorParameterInfo
    input: _OperatorPortInfo
    max_batch_size: _OperatorParameterInfo
    output: _OperatorPortInfo
    width: _OperatorParameterInfo
    x: _OperatorParameterInfo
    y: _OperatorParameterInfo
    def __call__(self, graph: holoscan.Graph, instance_name: str, **params: Unpack[_CustomCropOpParams]) -> _CustomCropOpHandle: ...  # type: ignore[override]

CustomCropOp: Final[_CustomCropOpType]


class _NonMaximumSuppressionOpParams(TypedDict, total=False):
    max_boxes: Required[int]
    max_batch_size: Required[int]
    score_threshold: float
    iou_threshold: float


class _NonMaximumSuppressionOpHandle(_OperatorHandle):
    boxes: holoscan.Port
    mask: holoscan.Port
    scores: holoscan.Port
    def __getattr__(self, name: Never) -> Never: ...  # type: ignore[misc, override]


class _NonMaximumSuppressionOpType(_OperatorType[_NonMaximumSuppressionOpHandle]):
    'Registered Operator holoscan.vision.NonMaximumSuppressionOp.\n\nRegistry name: holoscan.vision.NonMaximumSuppressionOp\n\nPorts:\n  boxes: Input[holoscan.Tensor]\n  scores: Input[holoscan.Tensor]\n  mask: Output[holoscan.Tensor]\n\nParameters:\n  max_boxes: int, range (1, 2147483647) — Maximum boxes admitted per batch item.\n  max_batch_size: int, range (1, 65535) — Maximum detection batch size admitted by the operator.\n  score_threshold: float, range (0.0, 1.0) — Inclusive minimum score for retaining a bounding box.\n  iou_threshold: float, range (1.401298464324817e-45, 1.0) — Maximum permitted Intersection-over-Union overlap.'
    Handle: Final[type[_NonMaximumSuppressionOpHandle]]
    Params: Final[type[_NonMaximumSuppressionOpParams]]
    boxes: _OperatorPortInfo
    iou_threshold: _OperatorParameterInfo
    mask: _OperatorPortInfo
    max_batch_size: _OperatorParameterInfo
    max_boxes: _OperatorParameterInfo
    score_threshold: _OperatorParameterInfo
    scores: _OperatorPortInfo
    def __call__(self, graph: holoscan.Graph, instance_name: str, **params: Unpack[_NonMaximumSuppressionOpParams]) -> _NonMaximumSuppressionOpHandle: ...  # type: ignore[override]

NonMaximumSuppressionOp: Final[_NonMaximumSuppressionOpType]


class _NormalizeOpParams(TypedDict, total=False):
    base: list[float] | tuple[float, ...]
    scale: list[float] | tuple[float, ...]
    global_scale: float
    shift: float
    epsilon: float
    scale_is_stddev: bool


class _NormalizeOpHandle(_OperatorHandle):
    input: holoscan.Port
    output: holoscan.Port
    def __getattr__(self, name: Never) -> Never: ...  # type: ignore[misc, override]


class _NormalizeOpType(_OperatorType[_NormalizeOpHandle]):
    'Registered Operator holoscan.vision.NormalizeOp.\n\nRegistry name: holoscan.vision.NormalizeOp\n\nPorts:\n  input: Input[holoscan.schema.Image]\n  output: Output[holoscan.schema.Image]\n\nParameters:\n  base: list[float] | tuple[float, ...] — Subtracted channel values: one value or one value per image channel.\n  scale: list[float] | tuple[float, ...] — Channel scales or standard deviations: one or one per image channel.\n  global_scale: float — Additional scale applied to every output value.\n  shift: float — Bias added to every output value.\n  epsilon: float, range (0.0, 3.4028234663852886e+38) — Nonnegative variance regularizer used with standard-deviation scale.\n  scale_is_stddev: bool — Interpret scale values as standard deviations.'
    Handle: Final[type[_NormalizeOpHandle]]
    Params: Final[type[_NormalizeOpParams]]
    base: _OperatorParameterInfo
    epsilon: _OperatorParameterInfo
    global_scale: _OperatorParameterInfo
    input: _OperatorPortInfo
    output: _OperatorPortInfo
    scale: _OperatorParameterInfo
    scale_is_stddev: _OperatorParameterInfo
    shift: _OperatorParameterInfo
    def __call__(self, graph: holoscan.Graph, instance_name: str, **params: Unpack[_NormalizeOpParams]) -> _NormalizeOpHandle: ...  # type: ignore[override]

NormalizeOp: Final[_NormalizeOpType]


class _ReformatOpParams(TypedDict, total=False):
    mode: Literal['interleaved_to_planar', 0, 'planar_to_interleaved', 1]


class _ReformatOpHandle(_OperatorHandle):
    input: holoscan.Port
    output: holoscan.Port
    def __getattr__(self, name: Never) -> Never: ...  # type: ignore[misc, override]


class _ReformatOpType(_OperatorType[_ReformatOpHandle]):
    "Registered Operator holoscan.vision.ReformatOp.\n\nRegistry name: holoscan.vision.ReformatOp\n\nPorts:\n  input: Input[holoscan.schema.Image]\n  output: Output[holoscan.schema.Image]\n\nParameters:\n  mode: Literal['interleaved_to_planar', 0, 'planar_to_interleaved', 1], choices interleaved_to_planar, planar_to_interleaved — Input-to-output image layout conversion."
    Handle: Final[type[_ReformatOpHandle]]
    Params: Final[type[_ReformatOpParams]]
    input: _OperatorPortInfo
    mode: _OperatorParameterInfo
    output: _OperatorPortInfo
    def __call__(self, graph: holoscan.Graph, instance_name: str, **params: Unpack[_ReformatOpParams]) -> _ReformatOpHandle: ...  # type: ignore[override]

ReformatOp: Final[_ReformatOpType]


class _RemapOpParams(TypedDict, total=False):
    width: int
    height: int
    max_batch_size: Required[int]
    input_interpolation: Literal['nearest', 0, 'linear', 1, 'cubic', 2]
    map_interpolation: Literal['nearest', 0, 'linear', 1, 'cubic', 2]
    map_value_mode: Literal['absolute', 0, 'absolute_normalized', 1, 'relative_normalized', 2]
    align_corners: bool
    border_mode: Literal['constant', 0, 'replicate', 1, 'reflect', 2, 'wrap', 3, 'reflect_101', 4]
    border_value: list[float] | tuple[float, ...]
    map_synchronization: Literal['latest', 0, 'exact', 1]
    map_max_age: _time_Duration | datetime.timedelta


class _RemapOpHandle(_OperatorHandle):
    input: holoscan.Port
    map: holoscan.Port
    output: holoscan.Port
    def __getattr__(self, name: Never) -> Never: ...  # type: ignore[misc, override]


class _RemapOpType(_OperatorType[_RemapOpHandle]):
    "Registered Operator holoscan.vision.RemapOp.\n\nRegistry name: holoscan.vision.RemapOp\n\nPorts:\n  input: Input[holoscan.schema.Image]\n  map: Input[holoscan.Tensor]\n  output: Output[holoscan.schema.Image]\n\nParameters:\n  width: int, range (1, 2147483647) — Output image width in pixels.\n  height: int, range (1, 2147483647) — Output image height in pixels.\n  max_batch_size: int, range (1, 65535) — Maximum image batch size admitted by the operator.\n  input_interpolation: Literal['nearest', 0, 'linear', 1, 'cubic', 2], choices nearest, linear, cubic — Interpolation used to sample the input image.\n  map_interpolation: Literal['nearest', 0, 'linear', 1, 'cubic', 2], choices nearest, linear, cubic — Interpolation used to sample the coordinate map.\n  map_value_mode: Literal['absolute', 0, 'absolute_normalized', 1, 'relative_normalized', 2], choices absolute, absolute_normalized, relative_normalized — Coordinate-map value encoding.\n  align_corners: bool — Align normalized coordinates to corner pixel centers.\n  border_mode: Literal['constant', 0, 'replicate', 1, 'reflect', 2, 'wrap', 3, 'reflect_101', 4], choices constant, replicate, reflect, wrap, reflect_101 — Input-image border extrapolation mode.\n  border_value: list[float] | tuple[float, ...] — Constant fill value: one value or one value per image channel.\n  map_synchronization: Literal['latest', 0, 'exact', 1], choices latest, exact — Image-to-map capture-time synchronization policy.\n  map_max_age: holoscan.time.Duration | datetime.timedelta, range (1, 9223372036854775807) — Maximum capture-time age of a map selected or reused for an image."
    Handle: Final[type[_RemapOpHandle]]
    Params: Final[type[_RemapOpParams]]
    align_corners: _OperatorParameterInfo
    border_mode: _OperatorParameterInfo
    border_value: _OperatorParameterInfo
    height: _OperatorParameterInfo
    input: _OperatorPortInfo
    input_interpolation: _OperatorParameterInfo
    map: _OperatorPortInfo
    map_interpolation: _OperatorParameterInfo
    map_max_age: _OperatorParameterInfo
    map_synchronization: _OperatorParameterInfo
    map_value_mode: _OperatorParameterInfo
    max_batch_size: _OperatorParameterInfo
    output: _OperatorPortInfo
    width: _OperatorParameterInfo
    def __call__(self, graph: holoscan.Graph, instance_name: str, **params: Unpack[_RemapOpParams]) -> _RemapOpHandle: ...  # type: ignore[override]

RemapOp: Final[_RemapOpType]


class _ResizeOpParams(TypedDict, total=False):
    width: int
    height: int
    max_batch_size: Required[int]
    interpolation: Literal['nearest', 0, 'linear', 1, 'cubic', 2, 'area', 3]


class _ResizeOpHandle(_OperatorHandle):
    input: holoscan.Port
    output: holoscan.Port
    def __getattr__(self, name: Never) -> Never: ...  # type: ignore[misc, override]


class _ResizeOpType(_OperatorType[_ResizeOpHandle]):
    "Registered Operator holoscan.vision.ResizeOp.\n\nRegistry name: holoscan.vision.ResizeOp\n\nPorts:\n  input: Input[holoscan.schema.Image]\n  output: Output[holoscan.schema.Image]\n\nParameters:\n  width: int, range (1, 2147483647) — Output image width in pixels.\n  height: int, range (1, 2147483647) — Output image height in pixels.\n  max_batch_size: int, range (1, 65535) — Maximum image batch size admitted by the operator.\n  interpolation: Literal['nearest', 0, 'linear', 1, 'cubic', 2, 'area', 3], choices nearest, linear, cubic, area — Interpolation algorithm."
    Handle: Final[type[_ResizeOpHandle]]
    Params: Final[type[_ResizeOpParams]]
    height: _OperatorParameterInfo
    input: _OperatorPortInfo
    interpolation: _OperatorParameterInfo
    max_batch_size: _OperatorParameterInfo
    output: _OperatorPortInfo
    width: _OperatorParameterInfo
    def __call__(self, graph: holoscan.Graph, instance_name: str, **params: Unpack[_ResizeOpParams]) -> _ResizeOpHandle: ...  # type: ignore[override]

ResizeOp: Final[_ResizeOpType]
