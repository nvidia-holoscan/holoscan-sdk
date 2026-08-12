/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace holoscan {

/// @brief Select the bounded queue behavior for one authored graph edge.
enum class QueuePolicy : std::uint8_t {
  kBlocking = 0,  ///< Preserve every accepted sample and apply backpressure when the queue is full.
  kDropOldest = 1,  ///< Retain the newest authored queue depth, replacing the oldest when full.
};

/// @brief Select whether compilation may insert a Tensor memory conversion.
enum class MemoryConversionPolicy : std::uint8_t {
  kDefault = 0,      ///< Apply the selected profile's documented conversion policy.
  kDisable = 1,      ///< Reject a memory mismatch instead of inserting a conversion.
  kAllowIfSafe = 2,  ///< Permit a conversion only with admitted bounded capability evidence.
  kRequire = 3,      ///< Require a visible conversion even if direct compatibility is available.
};

/// @brief Carry cold edge-local queue, alias, and memory-conversion requests.
///
/// The string view is a call-lifetime authoring input.
/// `Graph::add_flow` copies it before returning. These options never configure an
/// external endpoint or resize an active runtime queue.
struct ConnectionOptions {
  QueuePolicy queue_policy{QueuePolicy::kBlocking};  ///< Requested edge queue behavior.
  std::optional<std::uint32_t> queue_depth;          ///< Positive destination history depth.
  std::optional<std::string_view> topic_name;        ///< Optional logical flow alias to copy.
  std::optional<MemoryConversionPolicy> memory_conversion;  ///< Optional edge conversion rule.
  std::optional<std::uint32_t> max_inflight_samples;  ///< Positive diagnostic/retention bound.
};

}  // namespace holoscan
