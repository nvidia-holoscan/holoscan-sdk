/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>

#if defined(_WIN32)
#include <windows.h>
#endif

#include <holoscan/core/execution_plan.hpp>

#include "../support/optional_assertions.hpp"
#include "compiler/execution_plan_state.hpp"

namespace holoscan::test::partition_parity {

inline constexpr const char* kEnvironmentName = "HOLOSCAN_EXPERIMENTAL_PARTITION_ALGORITHM";

[[nodiscard]] inline std::optional<std::string> read_environment() {
  const char* const value = std::getenv(kEnvironmentName);
  return value != nullptr ? std::optional<std::string>{value} : std::nullopt;
}

[[nodiscard]] inline bool set_environment(std::optional<std::string_view> value) {
#if defined(_WIN32)
  return _putenv_s(kEnvironmentName, value.has_value() ? std::string{*value}.c_str() : "") == 0;
#else
  return value.has_value() ? setenv(kEnvironmentName, std::string(*value).c_str(), 1) == 0
                           : unsetenv(kEnvironmentName) == 0;
#endif
}

class ScopedEnvironmentMutation {
 public:
  explicit ScopedEnvironmentMutation(std::string_view value) : previous_(read_environment()) {
    EXPECT_TRUE(set_environment(value));
  }

  ScopedEnvironmentMutation(const ScopedEnvironmentMutation&) = delete;
  ScopedEnvironmentMutation& operator=(const ScopedEnvironmentMutation&) = delete;
  ScopedEnvironmentMutation(ScopedEnvironmentMutation&&) = delete;
  ScopedEnvironmentMutation& operator=(ScopedEnvironmentMutation&&) = delete;

  ~ScopedEnvironmentMutation() {
    const std::optional<std::string_view> previous =
        previous_.has_value() ? std::optional<std::string_view>{*previous_} : std::nullopt;
    static_cast<void>(set_environment(previous));
  }

 private:
  std::optional<std::string> previous_;
};

inline void expect_accepted_cover(const ExecutionPlan& plan) {
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto& requested_algorithm = plan.partitioning_plan().requested_algorithm;
  const auto& accepted_algorithm = plan.partitioning_plan().accepted_algorithm;
  HOLOSCAN_ASSERT_HAS_VALUE(requested_algorithm) << "Missing requested partition algorithm";
  HOLOSCAN_ASSERT_HAS_VALUE(accepted_algorithm) << "Missing accepted partition algorithm";
  const std::string_view expected_algorithm =
      plan.partitioning_plan().mode == PartitioningMode::kManual
          ? "holoscan.partition.serial_baseline"
          : "holoscan.partition.source_chain";
  EXPECT_EQ(requested_algorithm->id, expected_algorithm);
  EXPECT_EQ(accepted_algorithm->id, expected_algorithm);
  EXPECT_EQ(accepted_algorithm->version, 1U);
  EXPECT_EQ(plan.scheduler_plan().ready_policy.id, "holoscan.ready.deterministic_serial");
  EXPECT_EQ(plan.scheduler_plan().ready_policy.version, 1U);
  const auto* const state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  const auto callback_runner_count = static_cast<std::uint32_t>(state->partition_runners.size());
  EXPECT_EQ(plan.scheduler_plan().worker_count, callback_runner_count);
  EXPECT_EQ(plan.scheduler_plan().max_concurrent_activations, callback_runner_count);
  EXPECT_EQ(plan.scheduler_plan().activation_overlap, callback_runner_count > 1U);
}

}  // namespace holoscan::test::partition_parity
