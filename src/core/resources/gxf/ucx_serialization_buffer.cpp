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

#include "holoscan/core/resources/gxf/ucx_serialization_buffer.hpp"

#include <stdlib.h>  // setenv

#include <string>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"

namespace holoscan {

UcxSerializationBuffer::UcxSerializationBuffer(const std::string& name,
                                               nvidia::gxf::SerializationBuffer* component)
    : SerializationBuffer(name, component) {
  // no additional parameters to set here
}

void UcxSerializationBuffer::setup(ComponentSpec& spec) {
  std::string buffer_env_name{"HOLOSCAN_UCX_SERIALIZATION_BUFFER_SIZE"};
  const char* env_value = std::getenv(buffer_env_name.c_str());
  size_t default_buffer_size = 0;
  if (env_value) {
    try {
      default_buffer_size = std::stoull(env_value);
      HOLOSCAN_LOG_DEBUG("UcxSerializationBuffer: setting buffer size to {}", default_buffer_size);

      // Need to set corresponding underlying UCX environment variables as well or an error
      // such as the following may be seen at run time
      //     ucp_am.c:758  Fatal: RTS is too big XXXX, max YYYY
      setenv("UCX_TCP_RX_SEG_SIZE", env_value, 0);
      setenv("UCX_TCP_TX_SEG_SIZE", env_value, 0);
    } catch (std::exception& e) {
      HOLOSCAN_LOG_WARN(
          "Unable to interpret environment variable '{}': '{}'", buffer_env_name, e.what());
    }
  } else {
    default_buffer_size = kDefaultUcxSerializationBufferSize;
  }

  spec.param(allocator_, "allocator", "Memory allocator", "Memory allocator for tensor components");
  spec.param(buffer_size_,
             "buffer_size",
             "Buffer Size",
             "Size of the buffer in bytes (7168 by default unless "
             "HOLOSCAN_UCX_SERIALIZATION_BUFFER_SIZE is defined)",
             default_buffer_size);
}

void UcxSerializationBuffer::initialize() {
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
