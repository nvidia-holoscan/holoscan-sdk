# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This module provides a Python API to underlying C++ API Conditions.

.. autosummary::

    holoscan.conditions.AsynchronousCondition
    holoscan.conditions.BooleanCondition
    holoscan.conditions.CountCondition
    holoscan.conditions.CudaBufferAvailableCondition
    holoscan.conditions.CudaEventCondition
    holoscan.conditions.CudaStreamCondition
    holoscan.conditions.DownstreamMessageAffordableCondition
    holoscan.conditions.ExpiringMessageAvailableCondition
    holoscan.conditions.MemoryAvailableCondition
    holoscan.conditions.MessageAvailableCondition
    holoscan.conditions.MultiMessageAvailableCondition
    holoscan.conditions.MultiMessageAvailableTimeoutCondition
    holoscan.conditions.PeriodicCondition
    holoscan.conditions.PeriodicConditionPolicy
    holoscan.conditions.PublisherAvailableCondition
    holoscan.conditions.SubscriberAvailableCondition

``PendingExportCondition`` is only available when the SDK is built with
``HOLOSCAN_BUILD_IPC=ON`` and ``HOLOSCAN_IPC_TRANSPORT_FASTDDS=ON``.
"""

from . import _conditions
from ._conditions import (
    AsynchronousCondition,
    AsynchronousEventState,
    BooleanCondition,
    CountCondition,
    CudaBufferAvailableCondition,
    CudaEventCondition,
    CudaStreamCondition,
    DownstreamMessageAffordableCondition,
    ExpiringMessageAvailableCondition,
    MemoryAvailableCondition,
    MessageAvailableCondition,
    MultiMessageAvailableCondition,
    MultiMessageAvailableTimeoutCondition,
    PeriodicCondition,
    PeriodicConditionPolicy,
    PublisherAvailableCondition,
    SubscriberAvailableCondition,
)

__all__ = [
    "AsynchronousCondition",
    "AsynchronousEventState",
    "BooleanCondition",
    "CountCondition",
    "CudaBufferAvailableCondition",
    "CudaEventCondition",
    "CudaStreamCondition",
    "DownstreamMessageAffordableCondition",
    "ExpiringMessageAvailableCondition",
    "MemoryAvailableCondition",
    "MessageAvailableCondition",
    "MultiMessageAvailableCondition",
    "MultiMessageAvailableTimeoutCondition",
    "PeriodicCondition",
    "PeriodicConditionPolicy",
    "PublisherAvailableCondition",
    "SubscriberAvailableCondition",
]

if hasattr(_conditions, "PendingExportCondition"):
    PendingExportCondition = _conditions.PendingExportCondition
    __all__.append("PendingExportCondition")


# expose the SamplingMode enum from MultiMessageAvailableCondition
# (done this way instead of redefinining it in the bindings to avoids error:
#  ImportError: generic_type: type "SamplingMode" is already registered!)
MultiMessageAvailableTimeoutCondition.SamplingMode = MultiMessageAvailableCondition.SamplingMode
