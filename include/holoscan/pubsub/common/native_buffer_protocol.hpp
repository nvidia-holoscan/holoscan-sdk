/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
