/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/resources/gxf/std_component_serializer.hpp"

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"

namespace holoscan {

void StdComponentSerializer::setup(ComponentSpec& spec) {
  spec.param(allocator_, "allocator", "Memory allocator", "Memory allocator for tensor components");
}

void StdComponentSerializer::initialize() {
  // Set up prerequisite parameters before calling GXFOperator::initialize()
  auto frag = fragment();
  auto allocator = frag->make_resource<UnboundedAllocator>("allocator");

  add_arg(Arg("allocator") = allocator);

  GXFResource::initialize();
}

}  // namespace holoscan
