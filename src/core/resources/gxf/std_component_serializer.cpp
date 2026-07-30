/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/resources/gxf/std_component_serializer.hpp>

#include <algorithm>
#include <memory>

#include <holoscan/core/component_spec.hpp>
#include <holoscan/core/fragment.hpp>

namespace holoscan {

void StdComponentSerializer::setup(ComponentSpec& spec) {
  spec.param(allocator_, "allocator", "Memory allocator", "Memory allocator for tensor components");
}

// nvidia::gxf::StdComponentSerializer* StdComponentSerializer::get() const {
//   return static_cast<nvidia::gxf::StdComponentSerializer*>(gxf_cptr_);
// }

void StdComponentSerializer::initialize() {
  // Add a default UnboundedAllocator if no allocator was provided
  auto has_allocator = std::find_if(
      args().begin(), args().end(), [](const auto& arg) { return (arg.name() == "allocator"); });
  if (has_allocator == args().end()) {
    HOLOSCAN_LOG_TRACE("StdComponentSerializer: allocator argument not found, using default.");
    auto frag = fragment();
    auto allocator = frag->make_resource<UnboundedAllocator>("std_component_serializer_allocator");
    allocator->gxf_cname(allocator->name().c_str());
    if (gxf_eid_ != 0) {
      allocator->gxf_eid(gxf_eid_);
    }
    add_arg(Arg("allocator") = allocator);
  } else {
    // must set the gxf_eid for the provided allocator or GXF parameter registration will fail
    auto allocator_arg = *has_allocator;
    auto allocator = std::any_cast<std::shared_ptr<Resource>>(allocator_arg.value());
    auto gxf_allocator_resource = std::dynamic_pointer_cast<gxf::GXFResource>(allocator);
    if (gxf_eid_ != 0 && gxf_allocator_resource->gxf_eid() == 0) {
      HOLOSCAN_LOG_TRACE("allocator '{}': setting gxf_eid({}) from StdComponentSerializer '{}'",
                         allocator->name(),
                         gxf_eid_,
                         name());
      gxf_allocator_resource->gxf_eid(gxf_eid_);
    }
  }

  GXFResource::initialize();
}

}  // namespace holoscan
