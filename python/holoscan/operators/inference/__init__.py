"""
SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""  # noqa: E501

import holoscan.core  # noqa: F401
from holoscan.core import io_type_registry
from holoscan.resources import Allocator, CudaStreamPool  # noqa: F401

from ._inference import InferenceOp
from ._inference import register_types as _register_types

# register methods for receiving or emitting list[InferenceOp.ActivationSpec] and camera pose types
_register_types(io_type_registry)

__all__ = ["InferenceOp"]
