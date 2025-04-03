/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/resources/gxf/ucx_holoscan_component_serializer.hpp"

#include <algorithm>
#include <memory>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/resources/gxf/unbounded_allocator.hpp"

namespace holoscan {

void UcxHoloscanComponentSerializer::setup(ComponentSpec& spec) {
  HOLOSCAN_LOG_DEBUG("UcxHoloscanComponentSerializer::setup");
  spec.param(allocator_, "allocator", "Memory allocator", "Memory allocator for tensor components");
}

void UcxHoloscanComponentSerializer::initialize() {
  HOLOSCAN_LOG_DEBUG("UcxHoloscanComponentSerializer::initialize");
  // Set up prerequisite parameters before calling GXFResource::initialize()
  auto frag = fragment();

  // Find if there is an argument for 'allocator'
  auto has_allocator = std::find_if(
      args().begin(), args().end(), [](const auto& arg) { return (arg.name() == "allocator"); });
  // Create an UnboundedAllocator if no allocator was provided
  // Create an UnboundedAllocator if no allocator was provided
  if (has_allocator == args().end()) {
    auto allocator = frag->make_resource<UnboundedAllocator>("ucx_holoscan_component_allocator");
    add_arg(Arg("allocator") = allocator);
    allocator->gxf_cname(allocator->name().c_str());
    if (gxf_eid_ != 0) { allocator->gxf_eid(gxf_eid_); }
  } else {
    // must set the gxf_eid for the provided allocator or GXF parameter registration will fail
    auto allocator_arg = *has_allocator;
    auto allocator = std::any_cast<std::shared_ptr<Resource>>(allocator_arg.value());
    auto gxf_allocator_resource = std::dynamic_pointer_cast<gxf::GXFResource>(allocator);
    if (gxf_eid_ != 0 && gxf_allocator_resource->gxf_eid() == 0) {
      HOLOSCAN_LOG_TRACE(
          "allocator '{}': setting gxf_eid({}) from UcxHoloscanComponentSerializer '{}'",
          allocator->name(),
          gxf_eid_,
          name());
      gxf_allocator_resource->gxf_eid(gxf_eid_);
    }
  }
  GXFResource::initialize();
}

}  // namespace holoscan
