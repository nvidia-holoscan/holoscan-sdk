/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/memory/memory_buffer.hpp>

// The installed header provides all declarations needed by its inline failure paths.
class RejectedAllocation final : public holoscan::Allocator {
 public:
  holoscan::expected<void, holoscan::Error> is_available(uint64_t,
                                                         holoscan::MemoryStorageType) override {
    return {};
  }
  holoscan::expected<std::byte*, holoscan::Error> allocate(uint64_t,
                                                           holoscan::MemoryStorageType) override {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kResourceExhausted});
  }
  holoscan::expected<void, holoscan::Error> free(std::byte*) override { return {}; }
};

// REQ: EXPECTED-032
int main() {
  holoscan::MemoryBuffer buffer;
  // Intentionally no <memory>: this translation unit checks the installed header in isolation.
  // NOLINTNEXTLINE(build/include_what_you_use)
  auto allocator = std::make_shared<RejectedAllocation>();
  const auto result = buffer.resize(allocator, 4096, holoscan::MemoryStorageType::kSystem);
  return !result && result.error().code == holoscan::ErrorCode::kResourceExhausted ? 0 : 1;
}
