# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This module provides a Python API to underlying C++ API NetworkContexts.

.. autosummary::

    holoscan.network_contexts.FastDdsPubSubNetworkContext
    holoscan.network_contexts.UcxContext
"""

# Need to import UcxEntitySerializer before a UcxContext can be constructed
from ..resources import UcxEntitySerializer  # noqa
from . import _network_contexts

UcxContext = _network_contexts.UcxContext

__all__ = ["UcxContext"]

if hasattr(_network_contexts, "FastDdsPubSubNetworkContext"):
    FastDdsPubSubNetworkContext = _network_contexts.FastDdsPubSubNetworkContext
    __all__.append("FastDdsPubSubNetworkContext")

del UcxEntitySerializer
