# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in schema payload identities."""

from __future__ import annotations as _annotations

from .._public_api import export_public_names as _export_public_names

_schema_types: dict[str, type[object]] = {}


def _register_type(payload_type: type[object]) -> None:
    """Register one generated schema facade by its native payload identity."""
    registry_name = getattr(payload_type, "_registry_name", None)
    if not isinstance(registry_name, str) or not registry_name:
        raise TypeError("generated schema type must define a nonempty _registry_name")
    _schema_types[registry_name] = payload_type


def _resolve_type(registry_name: str) -> type[object] | None:
    """Return the imported generated facade for a native payload identity, if any."""
    return _schema_types.get(registry_name)


__all__ = _export_public_names(__name__, globals())  # noqa: PLE0605
