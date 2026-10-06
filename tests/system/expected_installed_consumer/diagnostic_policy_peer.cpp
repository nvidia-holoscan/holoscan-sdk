/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstdint>
#include <memory>

#include <holoscan/core/diagnostic.hpp>
#include <holoscan/memory/memory_buffer.hpp>

#include "diagnostic_policy_peer_export.h"

// REQ: EXPECTED-032
extern "C" DIAGNOSTIC_POLICY_PEER_EXPORT bool diagnostic_policy_peer_get() {
  return holoscan::automatic_diagnostic_logging_enabled();
}

extern "C" DIAGNOSTIC_POLICY_PEER_EXPORT void diagnostic_policy_peer_set(bool enabled) {
  holoscan::set_automatic_diagnostic_logging_enabled(enabled);
}

// REQ: EXPECTED-032
extern "C" DIAGNOSTIC_POLICY_PEER_EXPORT bool diagnostic_policy_peer_memory_failures() {
  class RejectingAllocator final : public holoscan::Allocator {
   public:
    holoscan::expected<void, holoscan::Error> is_available(uint64_t,
                                                           holoscan::MemoryStorageType) override {
      return {};
    }
    holoscan::expected<std::byte*, holoscan::Error> allocate(uint64_t,
                                                             holoscan::MemoryStorageType) override {
      ++attempts;
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "allocation rejected"});
    }
    holoscan::expected<void, holoscan::Error> free(std::byte*) override { return {}; }
    int attempts{};
  };
  auto allocator = std::make_shared<RejectingAllocator>();
  holoscan::MemoryBuffer buffer;
  const auto allocation = buffer.resize(allocator, 4096, holoscan::MemoryStorageType::kSystem);
  if (allocation || allocation.error().code != holoscan::ErrorCode::kInvalidArgument ||
      allocation.error().message != "allocation rejected" || buffer.pointer() != nullptr) {
    return false;
  }
  std::byte backing{};
  bool reject_release = true;
  const auto wrapped =
      buffer.wrap_memory(&backing,
                         1,
                         holoscan::MemoryStorageType::kSystem,
                         [&](void*) -> holoscan::expected<void, holoscan::Error> {
                           if (reject_release) {
                             return holoscan::make_unexpected(holoscan::Error{
                                 holoscan::ErrorCode::kInvalidArgument, "release rejected"});
                           }
                           return {};
                         });
  if (!wrapped)
    return false;
  const auto release = buffer.resize(allocator, 4096, holoscan::MemoryStorageType::kSystem);
  const bool retained = !release && release.error().code == holoscan::ErrorCode::kInvalidArgument &&
                        release.error().message == "release rejected" &&
                        buffer.pointer() == &backing && buffer.size() == 1U &&
                        allocator->attempts == 1;
  reject_release = false;
  const auto freed = buffer.free_buffer();
  return retained && freed.has_value();
}
