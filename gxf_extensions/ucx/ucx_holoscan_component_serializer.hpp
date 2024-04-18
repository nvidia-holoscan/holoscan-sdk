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

#ifndef NVIDIA_GXF_SERIALIZATION_UCX_HOLOSCAN_COMPONENT_SERIALIZER_HPP_
#define NVIDIA_GXF_SERIALIZATION_UCX_HOLOSCAN_COMPONENT_SERIALIZER_HPP_

#include <string>

// #include "common/endian.hpp"
#include "gxf/serialization/component_serializer.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/tensor.hpp"
#include "holoscan/core/codec_registry.hpp"
#include "holoscan/core/message.hpp"

namespace nvidia {
namespace gxf {

// Serializer that supports serializaing Timestamps, Tensors, Video Buffer,
// Audio Buffer and integer components
// Valid for sharing data between devices with the same endianness
class UcxHoloscanComponentSerializer : public ComponentSerializer {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t deinitialize() override { return GXF_SUCCESS; }

 private:
  // Configures all serializer functions
  Expected<void> configureSerializers();
  // Configures all deserializer functions
  Expected<void> configureDeserializers();
  // Serializes a holoscan::Message
  Expected<size_t> serializeHoloscanMessage(const holoscan::Message& message, Endpoint* endpoint);
  // Deserializes a holoscan::Message
  Expected<holoscan::Message> deserializeHoloscanMessage(Endpoint* endpoint);

  Parameter<Handle<Allocator>> allocator_;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_SERIALIZATION_UCX_HOLOSCAN_COMPONENT_SERIALIZER_HPP_
