"""
SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""  # noqa: E501

import holoscan.core  # noqa: F401
from holoscan.core import io_type_registry
from holoscan.resources import Allocator, CudaStreamPool  # noqa: F401

from ._inference import InferenceOp
from ._inference import register_types as _register_types

# register methods for receiving or emitting list[InferenceOp.ActivationSpec] and camera pose types
_register_types(io_type_registry)

__all__ = ["InferenceOp"]
