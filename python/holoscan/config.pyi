# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# fmt: off

"""Configuration documents, paths, and loading."""

import enum


class ConfigDestinationState(enum.Enum):
    NOT_APPLICABLE = 0

    UNCHANGED = 1

    REPLACED = 2

class ConfigErrorCause:
    @property
    def code(self) -> ConfigErrorCode: ...

    @property
    def token(self) -> str: ...

    @property
    def message(self) -> str: ...

    @property
    def path(self) -> str: ...

    @property
    def source(self) -> str: ...

    @property
    def message_truncated(self) -> bool: ...

    @property
    def path_truncated(self) -> bool: ...

    @property
    def source_truncated(self) -> bool: ...

    @property
    def excerpt_truncated(self) -> bool: ...

    @property
    def cause_overflow(self) -> bool: ...

    @property
    def operation_stage(self) -> ConfigOperationStage: ...

    @property
    def destination_state(self) -> ConfigDestinationState: ...

    def __repr__(self) -> str: ...

class ConfigErrorCode(enum.Enum):
    INVALID_ARGUMENT = 0

    PARSE_ERROR = 1

    UNSUPPORTED_FORMAT = 2

    UNSUPPORTED_VALUE = 3

    INVALID_UTF8 = 4

    DUPLICATE_KEY = 5

    PATH_SYNTAX = 6

    PATH_NOT_FOUND = 7

    PATH_CONFLICT = 8

    TYPE_MISMATCH = 9

    OUT_OF_RANGE = 10

    LIMIT_EXCEEDED = 11

    LOSSLESS_PRESERVATION_UNAVAILABLE = 12

    SOURCE_CHANGED = 13

    GUARD_BUSY = 14

    CODEC_MISSING = 15

    CODEC_FAILURE = 16

    IO_ERROR = 17

    PERMISSION_DENIED = 18

    ATOMIC_REPLACE_UNSUPPORTED = 19

    DIGEST_MISMATCH = 20

    UNSUPPORTED_VERSION = 21

class ConfigErrorDetails:
    @property
    def code(self) -> ConfigErrorCode: ...

    @property
    def token(self) -> str: ...

    @property
    def message(self) -> str: ...

    @property
    def path(self) -> str: ...

    @property
    def source(self) -> str: ...

    @property
    def message_truncated(self) -> bool: ...

    @property
    def path_truncated(self) -> bool: ...

    @property
    def source_truncated(self) -> bool: ...

    @property
    def excerpt_truncated(self) -> bool: ...

    @property
    def cause_overflow(self) -> bool: ...

    @property
    def causes(self) -> tuple[ConfigErrorCause, ...]: ...

    @property
    def operation_stage(self) -> ConfigOperationStage: ...

    @property
    def destination_state(self) -> ConfigDestinationState: ...

    def __repr__(self) -> str: ...

class ConfigOperationStage(enum.Enum):
    NONE = 0

    PARSE = 1

    VALIDATE = 2

    SERIALIZE = 3

    WRITE_TEMPORARY = 4

    ACQUIRE_GUARD = 5

    REPLACE_DESTINATION = 6

    FLUSH_DESTINATION_METADATA = 7

class Document:
    """An immutable configuration document."""

    @property
    def semantic_digest(self) -> str: ...

    @property
    def source(self) -> RawSourceDigest | None: ...

    @property
    def base_source(self) -> RawSourceDigest | None: ...

    def __repr__(self) -> str: ...

class Format(enum.Enum):
    JSON = 0

    YAML = 1

class MissingValuePolicy(enum.Enum):
    REJECT = 0

    USE_TYPE_DEFAULTS = 1

class NumericConversionPolicy(enum.Enum):
    LOSSLESS = 0

    ALLOW_CHECKED_LOSSY = 1

class Path:
    """An owning RFC 6901 JSON Pointer."""

    @staticmethod
    def parse(value: str) -> Path: ...

    @property
    def tokens(self) -> list[str]: ...

    @property
    def is_root(self) -> bool: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...

class RawSourceDigest:
    @property
    def algorithm_version(self) -> int: ...

    @property
    def digest(self) -> str: ...

    @property
    def byte_size(self) -> int: ...

    def __repr__(self) -> str: ...

class UnknownFieldPolicy(enum.Enum):
    REJECT = 0

    IGNORE = 1

class ValueDecodeOptions:
    def __init__(self, missing_fields: MissingValuePolicy = MissingValuePolicy.REJECT, unknown_fields: UnknownFieldPolicy = UnknownFieldPolicy.REJECT, numeric_conversion: NumericConversionPolicy = NumericConversionPolicy.LOSSLESS) -> None: ...

    @property
    def missing_fields(self) -> MissingValuePolicy: ...

    @property
    def unknown_fields(self) -> UnknownFieldPolicy: ...

    @property
    def numeric_conversion(self) -> NumericConversionPolicy: ...

def load(path: object, format: Format | None = None) -> Document: ...

def parse(source: object, format: Format) -> Document: ...

__all__: list[str] = ['ConfigDestinationState', 'ConfigErrorCause', 'ConfigErrorCode', 'ConfigErrorDetails', 'ConfigOperationStage', 'Document', 'Format', 'MissingValuePolicy', 'NumericConversionPolicy', 'Path', 'RawSourceDigest', 'UnknownFieldPolicy', 'ValueDecodeOptions', 'load', 'parse']
