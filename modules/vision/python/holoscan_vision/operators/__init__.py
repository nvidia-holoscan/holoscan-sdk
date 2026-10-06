# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registered Holoscan Vision Operator proxies."""

from holoscan.registry import operators as _operators

from .. import _vision as _vision

_registry_prefix = "holoscan.vision."
_operator_types = {
    operator.name.removeprefix(_registry_prefix): operator
    for operator in _operators()
    if operator.name.startswith(_registry_prefix)
}
globals().update(_operator_types)
__all__: list[str] = []
__all__.extend(sorted(_operator_types))
