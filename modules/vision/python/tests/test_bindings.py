# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise real Vision bindings without starting GPU work."""

import pytest

from holoscan_vision import InterpolationMode, ResizeOpParams, _vision


def test_resize_configuration():
    params = ResizeOpParams()
    assert (params.width, params.height, params.max_batch_size) == (1, 1, 1)
    assert params.interpolation == InterpolationMode.LINEAR
    params.width, params.height = 640, 480
    params.max_batch_size = 2
    params.interpolation = InterpolationMode.NEAREST
    assert (params.width, params.height, params.max_batch_size) == (640, 480, 2)
    assert params.interpolation == InterpolationMode.NEAREST


def test_native_linkage():
    _vision._check_native_linkage()


def test_rejects_invalid_parameter_type():
    with pytest.raises(TypeError):
        ResizeOpParams().interpolation = "nearest"


def test_stable_abi_extension():
    assert _vision.__file__.endswith("/_vision.abi3.so")
