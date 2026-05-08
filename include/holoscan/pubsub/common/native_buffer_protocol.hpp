/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_PUBSUB_COMMON_NATIVE_BUFFER_PROTOCOL_HPP
#define HOLOSCAN_PUBSUB_COMMON_NATIVE_BUFFER_PROTOCOL_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gxf/std/tensor.hpp>

namespace holoscan {

struct NativeTensorMetadata {
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  std::string dtype;
  nvidia::gxf::MemoryStorageType storage_type{nvidia::gxf::MemoryStorageType::kDevice};
  uint64_t bytes_per_element{0};
};

struct ImportedNativeTensor {
  NativeTensorMetadata metadata;
  std::shared_ptr<void> mapped_ptr;

  void* data() const { return mapped_ptr.get(); }
};

}  // namespace holoscan

#endif /* HOLOSCAN_PUBSUB_COMMON_NATIVE_BUFFER_PROTOCOL_HPP */
