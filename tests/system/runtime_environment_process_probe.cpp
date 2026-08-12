/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Subprocess helper for ea1_runtime_environment_smoke.
// The parent launches this binary with HOLOSCAN_EXPERIMENTAL_PARTITION_ALGORITHM
// set to an intentionally invalid value ("definitely-not-an-algorithm").
// This probe verifies that the runtime ignores the current environment variable
// when executing a pre-compiled plan: it briefly overrides the variable to a
// valid selector only during compilation, then restores the conflict before
// calling run().  A correct implementation must use only the partition records
// baked into the immutable ExecutionPlan, not re-read the environment at runtime.

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include <holoscan/core/run.hpp>

#include "runtime_test_graph.hpp"
#include "support/direct_port_binding.hpp"

namespace {

constexpr const char* kPartitionEnvironmentName = "HOLOSCAN_EXPERIMENTAL_PARTITION_ALGORITHM";

[[nodiscard]] std::optional<std::string> read_partition_environment() {
#if defined(_WIN32)
  SetLastError(ERROR_SUCCESS);
  const DWORD required = GetEnvironmentVariableA(kPartitionEnvironmentName, nullptr, 0U);
  if (required == 0U) {
    return GetLastError() == ERROR_ENVVAR_NOT_FOUND ? std::nullopt
                                                    : std::optional<std::string>{std::string{}};
  }
  std::string value(required, '\0');
  const DWORD written = GetEnvironmentVariableA(kPartitionEnvironmentName, value.data(), required);
  value.resize(written);
  return value;
#else
  const char* const value = std::getenv(kPartitionEnvironmentName);
  return value != nullptr ? std::optional<std::string>{value} : std::nullopt;
#endif
}

[[nodiscard]] bool set_partition_environment(std::optional<std::string_view> value) noexcept {
#if defined(_WIN32)
  if (!value.has_value()) {
    return SetEnvironmentVariableA(kPartitionEnvironmentName, nullptr) != 0;
  }
  const std::string owned{*value};
  return SetEnvironmentVariableA(kPartitionEnvironmentName, owned.c_str()) != 0;
#else
  return value.has_value() ? setenv(kPartitionEnvironmentName, std::string(*value).c_str(), 1) == 0
                           : unsetenv(kPartitionEnvironmentName) == 0;
#endif
}

// RAII guard that sets HOLOSCAN_EXPERIMENTAL_PARTITION_ALGORITHM to a given
// value on construction and restores the previous value (or clears it) on
// destruction.  Used to confine the valid selector to compilation only.
class ScopedPartitionEnvironment {
 public:
  explicit ScopedPartitionEnvironment(std::optional<std::string_view> value)
      : previous_(read_partition_environment()) {
    valid_ = set_partition_environment(value);
  }

  ScopedPartitionEnvironment(const ScopedPartitionEnvironment&) = delete;
  ScopedPartitionEnvironment& operator=(const ScopedPartitionEnvironment&) = delete;

  ~ScopedPartitionEnvironment() {
    const std::optional<std::string_view> previous =
        previous_.has_value() ? std::optional<std::string_view>{*previous_} : std::nullopt;
    static_cast<void>(set_partition_environment(previous));
  }

  [[nodiscard]] bool valid() const noexcept { return valid_; }

 private:
  std::optional<std::string> previous_;
  bool valid_{};
};

}  // namespace

// This probe starts in a fresh process with a deliberately conflicting local
// selector. It temporarily clears the selector while compiling an explicitly
// partitioned graph, restores the conflict, and then executes the already
// accepted plan. Runtime behavior must depend only on the immutable partition
// records in that plan.
// REQ: RUNTIME-002
// REQ: RUNTIME-004
int main() {
  if (read_partition_environment() != std::optional<std::string>{"definitely-not-an-algorithm"}) {
    return 1;  // exit 1: parent must have set the conflicting env var before spawning this process
  }

  holoscan::ExecutionPlan plan;
  {
    const ScopedPartitionEnvironment compile_selector{std::nullopt};
    if (!compile_selector.valid()) {
      return 2;  // exit 2: must be able to temporarily override the env var for compilation
    }
    plan = holoscan::test::enable_direct_runtime(holoscan::test::make_runtime_test_plan());
  }

  if (read_partition_environment() != std::optional<std::string>{"definitely-not-an-algorithm"} ||
      !plan.ok()) {
    // Exit 3: env var must be restored to the conflicting value after scope exit,
    //         and the plan must have compiled successfully inside the scope
    return 3;
  }

  holoscan::test::RuntimeProbeState::reset();
  if (holoscan::run(plan) != 0 ||
      holoscan::test::RuntimeProbeState::snapshot() != (std::vector<std::int32_t>{11, 12, 13})) {
    // Exit 4: run() must succeed and the probe state must match the expected sequence —
    //         proving the runtime used the plan's partition records, not the env var
    return 4;
  }
  return 0;
}
