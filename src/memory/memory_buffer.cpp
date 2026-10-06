/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/memory/memory_buffer.hpp>

#include "core/diagnostic_reporting.hpp"

namespace holoscan {

void MemoryBuffer::report_release_failure(const Error& error,
                                          std::source_location location) noexcept {
  if constexpr (HOLOSCAN_LOG_ACTIVE_LEVEL <= HOLOSCAN_LOG_LEVEL_ERROR) {
    detail::report_automatic_diagnostic(LogLevel::ERROR, [&] {
      Logger::log(location.file_name(),
                  static_cast<int>(location.line()),
                  location.function_name(),
                  LogLevel::ERROR,
                  "Failed to free memory: {}",
                  runtime_error_message(error));
    });
  }
}

void MemoryBuffer::report_allocation_failure(const Error& error, uint64_t size,
                                             MemoryStorageType storage_type,
                                             std::source_location location) noexcept {
  if constexpr (HOLOSCAN_LOG_ACTIVE_LEVEL <= HOLOSCAN_LOG_LEVEL_ERROR) {
    detail::report_automatic_diagnostic(LogLevel::ERROR, [&] {
      Logger::log(location.file_name(),
                  static_cast<int>(location.line()),
                  location.function_name(),
                  LogLevel::ERROR,
                  "Native allocator failed to allocate {} bytes of memory of type {}: {}",
                  size,
                  static_cast<int32_t>(storage_type),
                  runtime_error_message(error));
    });
  }
}

}  // namespace holoscan
