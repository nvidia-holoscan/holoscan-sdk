# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib

import pytest


def _import_torch_or_skip():
    """Import torch, skipping only when it is not installed."""
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError:
        pytest.skip("torch not installed")


# Could not use pytest.importorskip() here because it would skip the test on ImportError
# which we want to catch.
torch = _import_torch_or_skip()


def test_torch_imports_without_cuda():
    """Ensure torch imports cleanly even when no GPU is visible."""
    # Check that torch was imported and exposes a version string.
    assert hasattr(torch, "__version__")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA device.")
def test_pytorch_cuda_inverse_sanity_check():
    """Compute a small CUDA inverse to sanity-check CUDA execution."""
    a = torch.eye(2, dtype=torch.float32, device="cuda")
    b = torch.linalg.inv(a)
    # Inverse of identity is identity, use allclose to check for equality.
    assert torch.allclose(b, a), "Matrix inverse computation failed"
    assert b.is_cuda, "Tensor is not on CUDA device"
