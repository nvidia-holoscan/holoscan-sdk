# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the separate Visualization native extension."""

from holoscan_visualization import _visualization


def test_stable_abi_extension():
    assert _visualization.__file__.endswith("/_visualization.abi3.so")
