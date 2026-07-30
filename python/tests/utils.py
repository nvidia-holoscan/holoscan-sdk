# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test utilities for holoscan Python tests."""

import functools
import logging
import re
import warnings

import pytest


@functools.cache
def is_torch_cuda_compatible() -> bool:
    """Check if torch is available with compatible CUDA support.

    Returns True if:
    - torch is installed
    - CUDA is available
    - The GPU's SM architecture is compatible with PyTorch's compiled kernels

    Returns False otherwise (with appropriate logging).
    Result is cached after first call.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return False

    if not torch.cuda.is_available():
        return False

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        torch.cuda.init()
        for w in caught_warnings:
            if re.search(r"CUDA capability sm_[^ ]+ is not compatible", str(w.message)):
                logging.warning(str(w.message))
                return False

    return True


# Pytest marker for skipping tests when torch CUDA is unavailable/incompatible.
# The check is cached after first call via @functools.cache.
requires_torch_cuda = pytest.mark.skipif(
    not is_torch_cuda_compatible(),
    reason="Torch CUDA unavailable or SM incompatible",
)
