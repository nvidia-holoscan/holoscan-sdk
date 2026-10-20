# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# fmt: off

"""Tensor protocol and conversion support types."""

from collections.abc import Mapping, Sequence
import enum
from typing import Self

import holoscan


class TensorProjection:
    """A named Tensor field projected from a descriptor-backed schema output."""

    @property
    def source(self) -> holoscan.Port: ...

    @property
    def field_name(self) -> str: ...

    @property
    def graph_valid(self) -> bool: ...

    def __repr__(self) -> str: ...


class ByteSpanExpression:
    """Immutable compile-time Tensor byte-span expression."""

    @property
    def invalid_reason(self) -> ByteSpanExpressionInvalidReason: ...

    def multiplied_by(self, factor: ByteSpanExpression) -> ByteSpanExpression: ...

    def ceiling_divided_by(self, divisor: ByteSpanExpression) -> ByteSpanExpression: ...

    def maximum_with(self, other: ByteSpanExpression) -> ByteSpanExpression: ...

    def aligned_to(self, alignment: int) -> ByteSpanExpression: ...

class ByteSpanExpressionInvalidReason(enum.Enum):
    NONE = 0
    EMPTY = 1
    NON_POSITIVE_CONSTANT = 2
    INVALID_OPERAND = 3
    NODE_LIMIT_EXCEEDED = 4
    DEPTH_LIMIT_EXCEEDED = 5
    ALIGNMENT_INVALID = 6
    PROJECTION_INVALID = 7


class DLDataType:
    """
    A DLPack data type: a scalar type ``code``, a ``bits`` width, and a ``lanes`` count.

    See the DLPack documentation:
    https://dmlc.github.io/dlpack/latest/c_api.html#_CPPv410DLDataType
    """

    def __init__(self, code: int, bits: int, lanes: int) -> None: ...

    @property
    def code(self) -> int:
        """Scalar type code (int; see ``DLDataTypeCode``)."""

    @code.setter
    def code(self, arg: int, /) -> None: ...

    @property
    def bits(self) -> int:
        """Bit width of a single scalar (int)."""

    @bits.setter
    def bits(self, arg: int, /) -> None: ...

    @property
    def lanes(self) -> int:
        """Number of lanes (int; > 1 for vector types)."""

    @lanes.setter
    def lanes(self, arg: int, /) -> None: ...

    def __repr__(self) -> str: ...

class DLDataTypeCode(enum.IntEnum):
    """
    DLPack scalar type code (``DLINT``, ``DLUINT``, ``DLFLOAT``, ``DLOPAQUEHANDLE``, ``DLBFLOAT``,
    ``DLCOMPLEX``, ``DLBOOL``, and DLPack's FP8/FP6/FP4 storage codes).
    """

    DLINT = 0

    DLUINT = 1

    DLFLOAT = 2

    DLOPAQUEHANDLE = 3

    DLBFLOAT = 4

    DLCOMPLEX = 5

    DLBOOL = 6

    DLFLOAT8_E3M4 = 7

    DLFLOAT8_E4M3 = 8

    DLFLOAT8_E4M3B11FNUZ = 9

    DLFLOAT8_E4M3FN = 10

    DLFLOAT8_E4M3FNUZ = 11

    DLFLOAT8_E5M2 = 12

    DLFLOAT8_E5M2FNUZ = 13

    DLFLOAT8_E8M0FNU = 14

    DLFLOAT6_E2M3FN = 15

    DLFLOAT6_E3M2FN = 16

    DLFLOAT4_E2M1FN = 17

class DLDevice:
    """
    A DLPack device: a ``device_type`` (``DLDeviceType``) and an integer ``device_id``.
    """

    def __init__(self, device_type: DLDeviceType, device_id: int) -> None: ...

    @property
    def device_type(self) -> DLDeviceType:
        """The device type (``DLDeviceType``)."""

    @property
    def device_id(self) -> int:
        """The device ordinal (int)."""

    def __repr__(self) -> str: ...

class DLDeviceType(enum.IntEnum):
    """
    DLPack device type.

    Members
    -------
    - ``DLCPU``: system (host) memory.
    - ``DLCUDA``: CUDA device memory.
    - ``DLCUDAHOST``: CUDA pinned host memory.
    - ``DLCUDAMANAGED``: CUDA managed (unified) memory.
    """

    DLCPU = 1

    DLCUDA = 2

    DLCUDAHOST = 3

    DLOPENCL = 4

    DLVULKAN = 7

    DLMETAL = 8

    DLVPI = 9

    DLROCM = 10

    DLROCMHOST = 11

    DLEXTDEV = 12

    DLCUDAMANAGED = 13

    DLONEAPI = 14

    DLWEBGPU = 15

    DLHEXAGON = 16

    DLMAIA = 17

    DLTRN = 18

class InterfaceTensorEmitPolicy(enum.Enum):
    REQUIRE_EXPORTABLE = 0

    COPY_IF_NEEDED = 1

    INLINE_SMALL_HOST_THEN_COPY_IF_NEEDED = 2

class MemoryConversionPolicy(enum.Enum):
    DEFAULT = 0

    DISABLE = 1

    ALLOW_IF_SAFE = 2

    REQUIRE = 3

class MemoryKind(enum.Enum):
    UNKNOWN = 0

    HOST = 1

    PINNED_HOST = 2

    CUDA_DEVICE = 3

    CUDA_MANAGED = 4

class MemoryKindExpression:
    """Immutable compiler-only parameter-to-memory-kind mapping."""

    @property
    def invalid_reason(self) -> MemoryKindExpressionInvalidReason: ...

class MemoryKindExpressionInvalidReason(enum.Enum):
    NONE = 0
    EMPTY = 1
    MAPPING_EMPTY = 2
    MAPPING_LIMIT_EXCEEDED = 3
    MAPPING_DUPLICATE = 4
    MAPPING_TARGET_INVALID = 5
    UNEXPECTED_MAPPING = 6

class InputSpec:
    """Tensor input representation, bounds, and conversion requirements."""

    def __init__(self, *, memory_kind: MemoryKind | None = None, dtype: DLDataType | object | None = None, rank: int | None = None, strides: TensorStridePolicy | None = None, bounds: TensorBounds | None = None, memory_conversion: MemoryConversionPolicy | None = None) -> None: ...

    @property
    def representation(self) -> TensorRepresentation: ...

    @property
    def bounds(self) -> TensorBounds | None: ...

    @property
    def memory_conversion(self) -> MemoryConversionPolicy | None: ...

class OutputStorage(enum.Enum):
    PASS_THROUGH = 0

    RUNTIME_POOL = 1

class OutputSpec:
    """Tensor output representation, bounds, storage, and transfer policy."""

    def __init__(self, *, storage: OutputStorage | str = 'pass_through', memory_kind: MemoryKind | None = None, dtype: DLDataType | object | None = None, rank: int | None = None, strides: TensorStridePolicy | None = None, bounds: TensorBounds | None = None, transfer: TensorTransferPolicy | None = None) -> None: ...

    @property
    def storage(self) -> OutputStorage: ...

    @property
    def representation(self) -> TensorRepresentation: ...

    @property
    def bounds(self) -> TensorBounds | None: ...

    @property
    def transfer(self) -> TensorTransferPolicy | None: ...

class TensorLoan:
    """Single-use context manager for one provider-backed writable Tensor output."""

    @property
    def tensor(self) -> holoscan.Tensor: ...

    def __enter__(self) -> Self: ...

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool: ...

class LoanRequest:
    """Shape, dtype, and optional byte strides for one schema Tensor field."""

    def __init__(self, *, shape: Sequence[int], dtype: object, strides: Sequence[int] | None = None) -> None: ...

    @property
    def shape(self) -> list[int]: ...

    @property
    def dtype(self) -> DLDataType: ...

    @property
    def strides(self) -> list[int] | None: ...

class TensorSchemaLoan:
    """Single-use context manager for one atomic descriptor-schema Tensor bundle."""

    @property
    def tensors(self) -> Mapping[str, holoscan.Tensor]: ...

    def __enter__(self) -> Self: ...

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool: ...

class TensorBoundaryContract:
    """
    Complete representation, bounds, transfer, and memory-conversion facts for a Tensor topic.
    """

    def __init__(self, *, representation: TensorRepresentation, bounds: TensorBounds, transfer: TensorTransferPolicy | None = None, memory_conversion: MemoryConversionPolicy | None = None) -> None: ...

    @property
    def representation(self) -> TensorRepresentation: ...

    @representation.setter
    def representation(self, arg: TensorRepresentation, /) -> None: ...

    @property
    def bounds(self) -> TensorBounds: ...

    @bounds.setter
    def bounds(self, arg: TensorBounds, /) -> None: ...

    @property
    def transfer(self) -> TensorTransferPolicy: ...

    @transfer.setter
    def transfer(self, arg: TensorTransferPolicy | None, /) -> None: ...

    @property
    def memory_conversion(self) -> MemoryConversionPolicy | None: ...

    @memory_conversion.setter
    def memory_conversion(self, arg: MemoryConversionPolicy | None, /) -> None: ...

class TensorBounds:
    """
    Finite logical and physical bounds for a Tensor boundary.

    Shape and stride inputs are copied into the Python value and remain valid through compile.
    """

    def __init__(self, *, max_rank: int | None = None, max_shape: Sequence[int] | None = None, max_byte_span: int | None = None, max_abs_byte_strides: Sequence[int] | None = None) -> None: ...

    @property
    def max_rank(self) -> int | None: ...

    @max_rank.setter
    def max_rank(self, arg: object | None, /) -> None: ...

    @property
    def max_shape(self) -> object: ...

    @max_shape.setter
    def max_shape(self, arg: object | None, /) -> None: ...

    @property
    def max_byte_span(self) -> int | None: ...

    @max_byte_span.setter
    def max_byte_span(self, arg: object | None, /) -> None: ...

    @property
    def max_abs_byte_strides(self) -> object: ...

    @max_abs_byte_strides.setter
    def max_abs_byte_strides(self, arg: object | None, /) -> None: ...

class TensorRepresentation:
    """
    Required memory, data-type, rank, and stride facts for a Tensor boundary.
    """

    def __init__(self, *, memory_kind: MemoryKind | None = None, dtype: DLDataType | None = None, rank: int | None = None, strides: TensorStridePolicy | None = None) -> None: ...

    @property
    def memory_kind(self) -> MemoryKind | None: ...

    @memory_kind.setter
    def memory_kind(self, arg: MemoryKind | None, /) -> None: ...

    @property
    def dtype(self) -> DLDataType | None: ...

    @dtype.setter
    def dtype(self, arg: DLDataType | None, /) -> None: ...

    @property
    def rank(self) -> int | None: ...

    @rank.setter
    def rank(self, arg: int | None, /) -> None: ...

    @property
    def strides(self) -> TensorStridePolicy | None: ...

    @strides.setter
    def strides(self, arg: TensorStridePolicy | None, /) -> None: ...

class TensorStridePolicy(enum.Enum):
    CONTIGUOUS_ONLY = 0

    BOUNDED_STRIDED = 1

class TensorTransferPolicy:
    """External Tensor export and bounded-copy policy."""

    def __init__(self, *, foreign_emit_policy: InterfaceTensorEmitPolicy = ..., max_inline_bytes: int | None = None, max_staging_byte_span: int | None = None) -> None: ...

    @property
    def foreign_emit_policy(self) -> InterfaceTensorEmitPolicy: ...

    @foreign_emit_policy.setter
    def foreign_emit_policy(self, arg: InterfaceTensorEmitPolicy, /) -> None: ...

    @property
    def max_inline_bytes(self) -> int | None: ...

    @max_inline_bytes.setter
    def max_inline_bytes(self, arg: int | None, /) -> None: ...

    @property
    def max_staging_byte_span(self) -> int | None: ...

    @max_staging_byte_span.setter
    def max_staging_byte_span(self, arg: int | None, /) -> None: ...

def is_tensor_like(obj: object) -> bool:
    """
    Whether an object is convertible to a Tensor, i.e. it exposes the DLPack protocol
    (``__dlpack__`` and ``__dlpack_device__``) or an array interface.

    Returns
    -------
    bool
    """

def byte_span(value: int) -> ByteSpanExpression: ...

def parameter_byte_span(parameter: holoscan.Param[object]) -> ByteSpanExpression: ...

def parameter_contiguous_byte_span(parameter: holoscan.Param[object]) -> ByteSpanExpression: ...

def parameter_memory_kind(parameter: holoscan.Param[bool], mapping: Mapping[bool, MemoryKind]) -> MemoryKindExpression: ...

def input_max_byte_span(input: holoscan.OperatorPort) -> ByteSpanExpression: ...

__all__: list[str] = ['ByteSpanExpression', 'ByteSpanExpressionInvalidReason', 'DLDataType', 'DLDataTypeCode', 'DLDevice', 'DLDeviceType', 'InputSpec', 'InterfaceTensorEmitPolicy', 'LoanRequest', 'MemoryConversionPolicy', 'MemoryKind', 'MemoryKindExpression', 'MemoryKindExpressionInvalidReason', 'OutputSpec', 'OutputStorage', 'TensorBoundaryContract', 'TensorBounds', 'TensorLoan', 'TensorProjection', 'TensorRepresentation', 'TensorSchemaLoan', 'TensorStridePolicy', 'TensorTransferPolicy', 'byte_span', 'input_max_byte_span', 'is_tensor_like', 'parameter_byte_span', 'parameter_contiguous_byte_span', 'parameter_memory_kind']
