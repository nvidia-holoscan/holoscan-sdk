# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This module provides a Python API to underlying C++ API Schedulers.

.. autosummary::

    holoscan.schedulers.EventBasedScheduler
    holoscan.schedulers.GreedyScheduler
    holoscan.schedulers.MultiThreadScheduler
"""

# must first import GXFClock for the std::shared_ptr<gxf::Clock> arguments in the __init__ methods
from ..resources import GXFClock  # noqa
from ._schedulers import EventBasedScheduler, GreedyScheduler, MultiThreadScheduler

__all__ = [
    "EventBasedScheduler",
    "GreedyScheduler",
    "MultiThreadScheduler",
]
