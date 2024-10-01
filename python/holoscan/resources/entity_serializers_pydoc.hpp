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

namespace UcxEntitySerializer {

PYDOC(UcxEntitySerializer, R"doc(
UCX entity serializer.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment to assign the resource to.
component_serializer : list of holoscan.resource.Resource
    The component serializers used by the entity serializer.
verbose_warning : bool, optional
    Whether to use verbose warnings during serialization.
name : str, optional
    The name of the entity serializer.
)doc")

}  // namespace UcxEntitySerializer

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_RESOURCES_COMPONENT_SERIALIZERS_PYDOC_HPP
