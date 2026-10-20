# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the separate Inference native extension."""

from holoscan_inference import _inference


def test_stable_abi_extension():
    assert _inference.__file__.endswith("/_inference.abi3.so")
