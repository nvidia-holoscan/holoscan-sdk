# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# fmt: off

"""Generic dynamic surface for registered reusable Graph proxy types."""

import holoscan as _holoscan

from ..registry import GraphType as _GraphType

def __getattr__(name: str) -> _GraphType[_holoscan.GraphHandle]: ...
