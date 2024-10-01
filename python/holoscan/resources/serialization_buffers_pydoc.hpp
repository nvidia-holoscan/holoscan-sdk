/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef PYHOLOSCAN_RESOURCES_COMPONENT_SERIALIZERS_PYDOC_HPP
#define PYHOLOSCAN_RESOURCES_COMPONENT_SERIALIZERS_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace SerializationBuffer {

PYDOC(SerializationBuffer, R"doc(
Serialization Buffer.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment to assign the resource to.
allocator : holoscan.resource.Allocator
    The memory allocator for tensor components.
buffer_size : int, optional
    The size of the buffer in bytes.
name : str, optional
    The name of the serialization buffer
)doc")

}  // namespace SerializationBuffer

namespace UcxSerializationBuffer {

PYDOC(UcxSerializationBuffer, R"doc(
UCX serialization buffer.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment to assign the resource to.
allocator : holoscan.resource.Allocator
    The memory allocator for tensor components.
buffer_size : int, optional
    The size of the buffer in bytes.
name : str, optional
    The name of the serialization buffer
)doc")

}  // namespace UcxSerializationBuffer

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_RESOURCES_COMPONENT_SERIALIZERS_PYDOC_HPP
