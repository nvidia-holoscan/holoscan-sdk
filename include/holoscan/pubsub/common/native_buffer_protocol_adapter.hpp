/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_PUBSUB_COMMON_NATIVE_BUFFER_PROTOCOL_ADAPTER_HPP
#define HOLOSCAN_PUBSUB_COMMON_NATIVE_BUFFER_PROTOCOL_ADAPTER_HPP

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <gxf/core/expected.hpp>
#include <gxf/std/tensor.hpp>

#include <holoscan/pubsub/common/native_buffer_protocol.hpp>

namespace holoscan {

class NativeBufferProtocolAdapter {
 public:
  using PendingExportCountChangedCallback = std::function<void(size_t)>;

  virtual ~NativeBufferProtocolAdapter() = default;

  virtual bool is_initialized() const = 0;
  virtual const std::string& default_protocol_name() const = 0;
  virtual bool supports_protocol(const std::string& protocol_name) const = 0;
  virtual bool can_export_tensor(const nvidia::gxf::Tensor& tensor) const = 0;
  virtual uint8_t descriptor_format_version() const = 0;

  virtual nvidia::gxf::Expected<std::vector<uint8_t>> export_tensor(
      void* device_ptr, const std::shared_ptr<void>& device_ptr_owner,
      const NativeTensorMetadata& tensor_info, const std::string& protocol_name,
      uint32_t sequence_num) = 0;

  virtual nvidia::gxf::Expected<ImportedNativeTensor> import_tensor_generic(
      const std::vector<uint8_t>& descriptor_bytes, const std::string& protocol_name,
      std::chrono::milliseconds timeout = std::chrono::milliseconds{0}) = 0;

  virtual size_t pending_export_count() const { return 0; }

  virtual void set_on_pending_export_count_changed(PendingExportCountChangedCallback callback) {
    (void)callback;
  }
};

}  // namespace holoscan

#endif /* HOLOSCAN_PUBSUB_COMMON_NATIVE_BUFFER_PROTOCOL_ADAPTER_HPP */
