# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical Python values for SDK-owned FlatBuffers schemas."""

from __future__ import annotations as _annotations

import importlib as _importlib
import importlib.util as _importlib_util

from . import geometry as geometry

# These eager imports install the Core codecs before a token-only port first uses
# its registry row. Keep them eager even when a value class is not referenced.
from .header import Header
from .string import String
from .tensor_packet import TensorPacket
from .time import Time

_OPTIONAL_TYPES = {"Image": ".image", "Imu": ".imu"}


def _available_optional_types() -> list[str]:
    return [
        name
        for name, module_name in _OPTIONAL_TYPES.items()
        if _importlib_util.find_spec(f"{__name__}{module_name}") is not None
    ]


__all__ = ["Header", "String", "TensorPacket", "Time", "geometry"]
# Payload tokens retain immutable registry rows. Eagerly importing every installed
# component codec ensures a token-declared port cannot capture a native-only row
# before the Python facade upgrades it.
_installed_optional_types = _available_optional_types()
for _type_name in _installed_optional_types:
    _module = _importlib.import_module(_OPTIONAL_TYPES[_type_name], __name__)
    globals()[_type_name] = getattr(_module, _type_name)

# The shared namespace gains these values only when their component is installed.
__all__ += _installed_optional_types  # noqa: PLE0605


def __getattr__(name: str) -> object:
    """Load a component-owned schema value when that component is installed."""

    module_name = _OPTIONAL_TYPES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    qualified_name = f"{__name__}{module_name}"
    try:
        module = _importlib.import_module(module_name, __name__)
    except ModuleNotFoundError as error:
        if error.name == qualified_name:
            raise AttributeError(
                f"{__name__}.{name} requires the Sensor I/O Python component; "
                "the holoscan-core wheel includes it, while componentized native installs "
                "must install holoscan-sensor-io-python_libs"
            ) from error
        raise
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include schema values contributed by installed SDK components."""

    return sorted(set(globals()) | set(_available_optional_types()))
