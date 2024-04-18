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

#include "holoscan/core/resources/gxf/ucx_component_serializer.hpp"

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/resources/gxf/unbounded_allocator.hpp"

namespace holoscan {

void UcxComponentSerializer::setup(ComponentSpec& spec) {
  HOLOSCAN_LOG_DEBUG("UcxComponentSerializer::setup");
  spec.param(allocator_, "allocator", "Memory allocator", "Memory allocator for tensor components");
}

nvidia::gxf::UcxComponentSerializer* UcxComponentSerializer::get() const {
  return static_cast<nvidia::gxf::UcxComponentSerializer*>(gxf_cptr_);
}

void UcxComponentSerializer::initialize() {
  HOLOSCAN_LOG_DEBUG("UcxComponentSerializer::initialize");
  // Set up prerequisite parameters before calling GXFOperator::initialize()
  auto frag = fragment();

  // Find if there is an argument for 'allocator'
  auto has_allocator = std::find_if(
      args().begin(), args().end(), [](const auto& arg) { return (arg.name() == "allocator"); });
  // Create an UnboundedAllocator if no allocator was provided
  if (has_allocator == args().end()) {
    auto allocator = frag->make_resource<UnboundedAllocator>("ucx_component_allocator");
    add_arg(Arg("allocator") = allocator);
    allocator->gxf_cname(allocator->name().c_str());
    if (gxf_eid_ != 0) { allocator->gxf_eid(gxf_eid_); }
  }
  GXFResource::initialize();
}

}  // namespace holoscan
