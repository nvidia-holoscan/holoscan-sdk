# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest

from ..utils import requires_torch_cuda


def _import_torch_or_skip():
    """Import torch, skipping only when it is not installed."""
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError:
        pytest.skip("torch not installed", allow_module_level=True)


# Could not use pytest.importorskip() here because it would skip the test on ImportError
# which we want to catch to verify whether torch dependencies are missing.
torch = _import_torch_or_skip()


def test_torch_imports_without_cuda():
    """Ensure torch imports cleanly even when no GPU is visible."""
    # Check that torch was imported and exposes a version string.
    assert hasattr(torch, "__version__")


@requires_torch_cuda
def test_pytorch_cuda_inverse_sanity_check():
    """Compute a small CUDA inverse to sanity-check CUDA execution."""
    a = torch.eye(2, dtype=torch.float32, device="cuda")
    b = torch.linalg.inv(a)
    # Inverse of identity is identity, use allclose to check for equality.
    assert torch.allclose(b, a), "Matrix inverse computation failed"
    assert b.is_cuda, "Tensor is not on CUDA device"
