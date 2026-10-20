# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pyright-only PEP 692 unexpected-keyword contract (ty 0.0.78 gap)."""

from holoscan_test_operators import CounterSource

import holoscan as hs

graph = hs.Graph("typing-negative-pyright")

CounterSource(graph, "unknown-param", missing=1)  # typing-error: unexpected-keyword
