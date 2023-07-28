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

#include "holoscan/core/resources/gxf/serialization_buffer.hpp"

#include <memory>
#include <string>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"

namespace holoscan {

SerializationBuffer::SerializationBuffer(const std::string& name,
                                         nvidia::gxf::SerializationBuffer* component)
    : GXFResource(name, component) {

  // using GxfParameterGetUInt64 since no method specific to size_t is available
  uint64_t buffer_size = 0;
  HOLOSCAN_GXF_CALL_FATAL(
      GxfParameterGetUInt64(gxf_context_, gxf_cid_, "buffer_size", &buffer_size));
  buffer_size_ = static_cast<size_t>(buffer_size);

  // get the allocator object
  gxf_uid_t allocator_cid;
  HOLOSCAN_GXF_CALL_FATAL(
      GxfParameterGetHandle(gxf_context_, gxf_cid_, "allocator", &allocator_cid));
  gxf_tid_t allocator_tid{};
  HOLOSCAN_GXF_CALL_FATAL(
      GxfComponentTypeId(gxf_context_, "nvidia::gxf::Allocator", &allocator_tid));
  nvidia::gxf::Allocator* allocator_ptr;
  HOLOSCAN_GXF_CALL_FATAL(GxfComponentPointer(
      gxf_context_, gxf_cid_, allocator_tid, reinterpret_cast<void**>(&allocator_ptr)));
  allocator_ = std::make_shared<Allocator>(std::string{allocator_ptr->name()}, allocator_ptr);
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

void SerializationBuffer::initialize() {
  HOLOSCAN_LOG_DEBUG("SerializationBuffer::initialize");
  // Set up prerequisite parameters before calling GXFOperator::initialize()
  auto frag = fragment();

  // Find if there is an argument for 'allocator'
  auto has_allocator = std::find_if(
      args().begin(), args().end(), [](const auto& arg) { return (arg.name() == "allocator"); });
  // Create an UnboundedAllocator if no allocator was provided
  if (has_allocator == args().end()) {
    auto allocator = frag->make_resource<UnboundedAllocator>("allocator");
    add_arg(Arg("allocator") = allocator);
  }
  GXFResource::initialize();
}

}  // namespace holoscan
