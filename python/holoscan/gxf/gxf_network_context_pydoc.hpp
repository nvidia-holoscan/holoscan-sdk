/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_GXF_NETWORK_CONTEXT_PYDOC_HPP
#define PYHOLOSCAN_GXF_NETWORK_CONTEXT_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace GXFNetworkContext {

// Constructor
PYDOC(GXFNetworkContext, R"doc(
Base GXF-based network context class.
)doc")

PYDOC(initialize, R"doc(
Initialize the network context.
)doc")

}  // namespace GXFNetworkContext

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_GXF_NETWORK_CONTEXT_PYDOC_HPP
