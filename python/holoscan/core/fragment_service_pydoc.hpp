/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_CORE_FRAGMENT_SERVICE_PYDOC_HPP
#define PYHOLOSCAN_CORE_FRAGMENT_SERVICE_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace FragmentService {

PYDOC(FragmentService, R"doc(
Base interface for services that enable sharing of resources and functionality between
operators.
)doc")

PYDOC(FragmentService_default, R"doc(
Construct a new FragmentService object.
)doc")

PYDOC(resource, R"doc(
The resource associated with the service.
)doc")

}  // namespace FragmentService

namespace DefaultFragmentService {

PYDOC(DefaultFragmentService, R"doc(
Base class for services to be registered in the fragment service registry.

Fragment services provide a way to share resources and functionality across
operators within a fragment or application.
)doc")

PYDOC(DefaultFragmentService_default, R"doc(
Construct a new DefaultFragmentService object.
)doc")

PYDOC(DefaultFragmentService_resource, R"doc(
Construct a new DefaultFragmentService object.

Parameters
----------
resource : holoscan.core.Resource
    The resource to associate with the service.
)doc")

}  // namespace DefaultFragmentService

}  // namespace holoscan::doc

#endif /* PYHOLOSCAN_CORE_FRAGMENT_SERVICE_PYDOC_HPP */
