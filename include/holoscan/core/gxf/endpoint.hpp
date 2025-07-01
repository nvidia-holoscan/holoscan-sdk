/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_GXF_ENDPOINT_HPP
#define HOLOSCAN_CORE_GXF_ENDPOINT_HPP

#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "../endpoint.hpp"
#include "../errors.hpp"
#include "../expected.hpp"

#include "gxf/serialization/endpoint.hpp"

namespace nvidia {
namespace gxf {
enum struct MemoryStorageType;
}
}  // namespace nvidia

namespace holoscan {
namespace gxf {

/**
 * @brief GXF serialization endpoint wrapper.
 *
 * This class is a non-owning wrapper around a GXF Endpoint. It is constructed via a pointer to an
 * existing nvidia::gxf::Endpoint and its methods dispatch to that endpoint.
 */
class Endpoint : public holoscan::Endpoint {
 public:
  Endpoint() = default;
  Endpoint(Endpoint&&) = default;
  ~Endpoint() override = default;
  explicit Endpoint(nvidia::gxf::Endpoint* gxf_endpoint) {
    if (!gxf_endpoint) { throw std::invalid_argument("gxf_endpoint cannot be null"); }
    gxf_endpoint_ = gxf_endpoint;
  }

  using MemoryStorageType = nvidia::gxf::MemoryStorageType;

  // C++ API wrappers
  bool is_write_available() override;
  bool is_read_available() override;
  expected<size_t, RuntimeError> write(const void* data, size_t size) override;
  expected<size_t, RuntimeError> read(void* data, size_t size) override;
  expected<void, RuntimeError> write_ptr(const void* pointer, size_t size,
                                         holoscan::MemoryStorageType type) override;

  // Note: in GXF, writeTrivialType and readTrivialType below are not on Endpoint itself, but on
  // SerializationBuffer and UcxSerializationBuffer
  using holoscan::Endpoint::read_trivial_type;
  using holoscan::Endpoint::write_trivial_type;

 private:
  nvidia::gxf::Endpoint* gxf_endpoint_ = nullptr;
};
}  // namespace gxf
}  // namespace holoscan

#endif /* HOLOSCAN_CORE_GXF_ENDPOINT_HPP */
