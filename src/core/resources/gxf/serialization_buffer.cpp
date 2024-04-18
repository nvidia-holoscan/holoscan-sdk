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

#include "holoscan/core/resources/gxf/serialization_buffer.hpp"

#include <memory>
#include <string>

#include "gxf/std/allocator.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"

namespace holoscan {

SerializationBuffer::SerializationBuffer(const std::string& name,
                                         nvidia::gxf::SerializationBuffer* component)
    : GXFResource(name, component) {
  auto maybe_buffer_size = component->getParameter<size_t>("buffer_size");
  if (!maybe_buffer_size) { throw std::runtime_error("Failed to get maybe_buffer_size"); }
  buffer_size_ = maybe_buffer_size.value();

  auto maybe_allocator =
      component->getParameter<nvidia::gxf::Handle<nvidia::gxf::Allocator>>("allocator");
  if (!maybe_allocator) { throw std::runtime_error("Failed to get allocator"); }
  auto allocator_handle = maybe_allocator.value();
  allocator_ =
      std::make_shared<Allocator>(std::string{allocator_handle->name()}, allocator_handle.get());
}

void SerializationBuffer::setup(ComponentSpec& spec) {
  HOLOSCAN_LOG_DEBUG("SerializationBuffer::setup");
  spec.param(allocator_, "allocator", "Memory allocator", "Memory allocator for tensor components");
  spec.param(buffer_size_,
             "buffer_size",
             "Buffer Size",
             "Size of the buffer in bytes (4096 by default)",
             kDefaultSerializationBufferSize);
}

nvidia::gxf::SerializationBuffer* SerializationBuffer::get() const {
  return static_cast<nvidia::gxf::SerializationBuffer*>(gxf_cptr_);
}

void SerializationBuffer::initialize() {
  HOLOSCAN_LOG_DEBUG("SerializationBuffer::initialize");
  // Set up prerequisite parameters before calling GXFOperator::initialize()
  auto frag = fragment();

  // Find if there is an argument for 'allocator'
  auto has_allocator = std::find_if(
      args().begin(), args().end(), [](const auto& arg) { return (arg.name() == "allocator"); });
  // Create an UnboundedAllocator if no allocator was provided
  if (has_allocator == args().end()) {
    auto allocator = frag->make_resource<UnboundedAllocator>("serialization_buffer_allocator");
    allocator->gxf_cname(allocator->name().c_str());
    if (gxf_eid_ != 0) { allocator->gxf_eid(gxf_eid_); }
    add_arg(Arg("allocator") = allocator);
  }
  GXFResource::initialize();
}

}  // namespace holoscan
