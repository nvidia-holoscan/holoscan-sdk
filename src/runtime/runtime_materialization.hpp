/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/expected.hpp>

#include "runtime_package_capability.hpp"

namespace holoscan::detail {

class ExecutionLaneSet;
struct ExecutionPlanState;

inline constexpr std::size_t kRuntimeMaterializationMaximumObligations = 256U;
inline constexpr std::size_t kRuntimeMaterializationMaximumTokenBytes = 512U;

/// @brief Classify one immutable Runtime materialization result.
enum class RuntimeMaterializationResult : std::uint8_t {
  kPending = 0,
  kCompleted = 1,
  kCleaned = 2,
  kFailed = 3,
};

/// @brief Preserve bounded evidence for one plan-owned materialization obligation.
struct RuntimeMaterializationRecord {
  RuntimePackageCapabilityMatchContext context;
  std::string package_id;
  std::string package_manifest_digest;
  std::string object_id;
  std::string strategy_id;
  LifecycleStage complete_before_stage{LifecycleStage::kConfigure};
  std::uint8_t cleanup_stage_mask{};
  std::uint32_t max_cleanup_actions{};
  std::string evidence_id;
  RuntimeMaterializationResult result{RuntimeMaterializationResult::kPending};
  std::uint32_t cleanup_actions{};
  bool cleanup_required{};

  friend bool operator==(const RuntimeMaterializationRecord&,
                         const RuntimeMaterializationRecord&) = default;
};

/// @brief Execute typed materialization barriers as lifecycle-owned transactions.
///
/// Construction validates and copies the canonical obligation set before user
/// callbacks. Forward callbacks verify the exact plan-owned resource at the
/// obligation's deadline. Teardown performs at most the compiled cleanup bound
/// and never resumes completed work after failure.
class RuntimeMaterializationState {
 public:
  RuntimeMaterializationState(const RuntimeMaterializationState&) = delete;
  RuntimeMaterializationState& operator=(const RuntimeMaterializationState&) = delete;
  RuntimeMaterializationState(RuntimeMaterializationState&&) noexcept = default;
  RuntimeMaterializationState& operator=(RuntimeMaterializationState&&) noexcept = default;

  /// @brief Validate and own the complete immutable obligation set.
  [[nodiscard]] static expected<RuntimeMaterializationState, Error> create(
      const ExecutionPlanState& plan, const RuntimePackageCapabilityMatchReport& package,
      const ExecutionLaneSet* lanes);

  /// @brief Return the number of canonical materialization records.
  [[nodiscard]] std::size_t size() const noexcept { return records_.size(); }

  /// @brief Return one canonical materialization record.
  [[nodiscard]] const RuntimeMaterializationRecord& record(std::size_t index) const noexcept {
    return records_[index];
  }

  /// @brief Run one lifecycle-owned forward or cleanup transition.
  [[nodiscard]] LifecycleStatus transition(std::size_t index, LifecycleStage stage,
                                           LifecycleTransitionPhase phase) noexcept;

  /// @brief Return whether a materialization callback has failed.
  [[nodiscard]] bool failed() const noexcept { return failed_; }

  /// @brief Return canonical record indexes in their completed cleanup order.
  [[nodiscard]] std::span<const std::size_t> cleanup_order() const noexcept {
    return cleanup_order_;
  }

  /// @brief Release completed records not reached by lifecycle rollback.
  void cleanup_remaining() noexcept;

 private:
  RuntimeMaterializationState(const ExecutionPlanState& plan, const ExecutionLaneSet* lanes,
                              std::vector<RuntimeMaterializationRecord> records) noexcept;

  [[nodiscard]] bool verify_resource(const RuntimeMaterializationRecord& record) const noexcept;

  const ExecutionPlanState* plan_{};
  const ExecutionLaneSet* lanes_{};
  std::vector<RuntimeMaterializationRecord> records_;
  std::vector<std::size_t> cleanup_order_;
  bool failed_{};
};

}  // namespace holoscan::detail
