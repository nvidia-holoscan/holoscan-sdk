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

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_SERIALIZATION_BUFFER_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_SERIALIZATION_BUFFER_HPP

#include <cstdint>
#include <memory>
#include <string>

#include <gxf/serialization/serialization_buffer.hpp>

#include "../../gxf/gxf_resource.hpp"
#include "./unbounded_allocator.hpp"

namespace holoscan {

constexpr size_t kDefaultSerializationBufferSize = 1 << 12;  // 4 kB

/**
 * @brief Base entity serialization buffer class.
 *
 * Provides a memory buffer that can be used for serialization of GXF entities. The allocator
 * parameter can be used to set the memory allocator used for this buffer.
 *
 * ==Parameters==
 *
 * - **allocator** (std::shared_ptr<holoscan::Allocator>, optional): The allocator used to
 * allocate/free the buffer memory. If no allocator is set, an `UnboundedAllocator` will be used.
 * - **buffer_size** (size_t, optional): The size of the buffer in bytes (Defaults to
 * holoscan::kDefaultSerializationBufferSize).
 */
class SerializationBuffer : public gxf::GXFResource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(SerializationBuffer, GXFResource)
  SerializationBuffer() = default;
  SerializationBuffer(const std::string& name, nvidia::gxf::SerializationBuffer* component);

  const char* gxf_typename() const override { return "nvidia::gxf::SerializationBuffer"; }

  void setup(ComponentSpec& spec) override;

  void initialize() override;

  nvidia::gxf::SerializationBuffer* get() const;

 private:
  Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
  Parameter<size_t> buffer_size_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_SERIALIZATION_BUFFER_HPP */
