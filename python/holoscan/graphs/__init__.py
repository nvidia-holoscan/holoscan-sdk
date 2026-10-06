# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Importable proxies for registered C++ reusable Graph types."""

from collections.abc import Iterator as _Iterator
from collections.abc import Sequence as _Sequence

from ..registry import GraphType as _GraphType
from ..registry import _importable_graph_aliases


def _aliases() -> dict[str, _GraphType]:
    return {
        name: graph_type
        for name, graph_type in _importable_graph_aliases().items()
        if name != "GraphType" and name not in globals()
    }


class _DynamicExports(_Sequence[str]):
    """Resolve import-star names from the current process registry."""

    def _names(self) -> tuple[str, ...]:
        return tuple(sorted(_aliases()))

    def __getitem__(self, index: int) -> str:
        return self._names()[index]

    def __len__(self) -> int:
        return len(self._names())

    def __iter__(self) -> _Iterator[str]:
        return iter(self._names())


__all__: _Sequence[str] = _DynamicExports()  # noqa: PLE0605


def __getattr__(name: str) -> _GraphType:
    try:
        return _aliases()[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_aliases()))
