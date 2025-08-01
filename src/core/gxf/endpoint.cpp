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

#include "gxf/std/allocator.hpp"  // for nvidia::gxf::MemoryStorageType

#include "holoscan/core/gxf/endpoint.hpp"

namespace holoscan {
namespace gxf {

// C++ API wrappers
bool Endpoint::is_write_available() {
  if (!gxf_endpoint_) {
    throw std::runtime_error("GXF endpoint has not been set");
  }
  return gxf_endpoint_->isWriteAvailable();
}
bool Endpoint::is_read_available() {
  if (!gxf_endpoint_) {
    throw std::runtime_error("GXF endpoint has not been set");
  }
  return gxf_endpoint_->isReadAvailable();
}

expected<size_t, RuntimeError> Endpoint::write(const void* data, size_t size) {
  if (!gxf_endpoint_) {
    return make_unexpected<RuntimeError>(
        RuntimeError(ErrorCode::kCodecError, "GXF endpoint has not been set"));
  }
  auto maybe_size = gxf_endpoint_->write(data, size);
  if (!maybe_size) {
    // converted nvidia::gxf::Unexpected to holoscan::unexpected
    auto err_msg = fmt::format("GXF endpoint read failure: {}", GxfResultStr(maybe_size.error()));
    return make_unexpected<RuntimeError>(RuntimeError(ErrorCode::kCodecError, err_msg));
  }
  return maybe_size.value();
}
expected<size_t, RuntimeError> Endpoint::read(void* data, size_t size) {
  if (!gxf_endpoint_) {
    return make_unexpected<RuntimeError>(
        RuntimeError(ErrorCode::kCodecError, "GXF endpoint has not been set"));
  }
  auto maybe_size = gxf_endpoint_->read(data, size);
  if (!maybe_size) {
    // converted nvidia::gxf::Unexpected to holoscan::unexpected
    auto err_msg = fmt::format("GXF endpoint read failure: {}", GxfResultStr(maybe_size.error()));
    return make_unexpected<RuntimeError>(RuntimeError(ErrorCode::kCodecError, err_msg));
  }
  return maybe_size.value();
}

expected<void, RuntimeError> Endpoint::write_ptr(const void* pointer, size_t size,
                                                 holoscan::MemoryStorageType type) {
  if (!gxf_endpoint_) {
    return make_unexpected<RuntimeError>(
        RuntimeError(ErrorCode::kCodecError, "GXF endpoint has not been set"));
  }
  nvidia::gxf::MemoryStorageType gxf_storage_type;
  switch (type) {
    case holoscan::MemoryStorageType::kDevice:
      gxf_storage_type = nvidia::gxf::MemoryStorageType::kDevice;
      break;
    case holoscan::MemoryStorageType::kHost:
      gxf_storage_type = nvidia::gxf::MemoryStorageType::kHost;
      break;
    case holoscan::MemoryStorageType::kSystem:
      gxf_storage_type = nvidia::gxf::MemoryStorageType::kSystem;
      break;
    default:
      throw std::runtime_error("Invalid memory storage type");
  }
  auto maybe_void = gxf_endpoint_->write_ptr(pointer, size, gxf_storage_type);
  if (!maybe_void) {
    // converted nvidia::gxf::Unexpected to holoscan::unexpected
    auto err_msg = fmt::format("GXF endpoint read failure: {}", GxfResultStr(maybe_void.error()));
    return make_unexpected<RuntimeError>(RuntimeError(ErrorCode::kCodecError, err_msg));
  }
  return expected<void, RuntimeError>();
}

}  // namespace gxf
}  // namespace holoscan
