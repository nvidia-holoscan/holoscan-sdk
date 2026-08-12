/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/source_id.hpp>
#include <holoscan/core/tag.hpp>
#include <holoscan/time/time_point.hpp>

namespace holoscan::detail {

/// @brief Identify Runtime-managed callback contexts that prohibit blocking/control ingress.
enum class RuntimeCallbackContext : std::uint8_t {
  kNone = 0,
  kCompute = 1,
  kLifecycle = 2,
  kDispatcher = 3,
};

/// @brief Retain stable bounded identity for one currently executing Runtime callback.
///
/// The record contains no borrowed payload, string, or callback address. Lifecycle callbacks set
/// operator, stage, and transition phase. Dispatcher/compute and predicate integration can add the
/// already frozen source, Tag, and state generation without changing this private contract.
struct RuntimeCallbackIdentity {
  RuntimeCallbackContext context{RuntimeCallbackContext::kNone};
  OperatorInstanceId operator_id{};
  LifecycleStage lifecycle_stage{LifecycleStage::kConfigure};
  LifecycleTransitionPhase lifecycle_phase{LifecycleTransitionPhase::kNone};
  SourceId source{};
  std::uint64_t source_sequence{};
  Tag tag{};
  TagTime tag_time{};
  TimePoint reference_time{};
  std::uint64_t state_generation{};
};

/// @brief Mark one exact Runtime callback scope on the executing thread.
class HOLOSCAN_CORE_EXPORT RuntimeCallbackContextScope {
 public:
  explicit RuntimeCallbackContextScope(RuntimeCallbackContext context) noexcept;
  explicit RuntimeCallbackContextScope(RuntimeCallbackIdentity identity) noexcept;

  RuntimeCallbackContextScope(const RuntimeCallbackContextScope&) = delete;
  RuntimeCallbackContextScope& operator=(const RuntimeCallbackContextScope&) = delete;

  ~RuntimeCallbackContextScope() noexcept;

 private:
  RuntimeCallbackIdentity previous_;
};

/// @brief Return whether the current thread is executing restricted Runtime callback code.
[[nodiscard]] HOLOSCAN_CORE_EXPORT bool runtime_callback_context_restricted() noexcept;

/// @brief Return the exact stable identity installed for the current Runtime callback scope.
[[nodiscard]] HOLOSCAN_CORE_EXPORT RuntimeCallbackIdentity runtime_callback_identity() noexcept;

}  // namespace holoscan::detail
