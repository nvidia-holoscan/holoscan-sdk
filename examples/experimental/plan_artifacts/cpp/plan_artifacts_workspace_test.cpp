/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

#include "plan_artifacts_workspace.hpp"

namespace {

constexpr std::string_view kNonce = "collision-regression";

}  // namespace

int main() {
  std::atomic<std::uint64_t> root_sequence{};
  const std::string root_nonce =
      "workspace-collision-test-" +
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  const std::filesystem::path test_root = plan_artifacts_detail::create_temporary_workspace(
      std::filesystem::temp_directory_path(), root_nonce, root_sequence);
  if (test_root.empty()) {
    std::cerr << "failed to create test root\n";
    return 1;
  }
  const plan_artifacts_detail::TemporaryWorkspaceCleanup test_root_cleanup{test_root};

  const std::filesystem::path occupied =
      test_root / "holoscan-plan-artifacts-collision-regression-0";
  if (!std::filesystem::create_directory(occupied)) {
    std::cerr << "failed to create occupied peer workspace\n";
    return 2;
  }
  const std::filesystem::path sentinel = occupied / "peer-sentinel";
  {
    std::ofstream file{sentinel};
    file << "owned by another run\n";
    if (!file) {
      std::cerr << "failed to create peer sentinel\n";
      return 2;
    }
  }

  std::atomic<std::uint64_t> next_sequence{};
  {
    const std::filesystem::path workspace =
        plan_artifacts_detail::create_temporary_workspace(test_root, kNonce, next_sequence, 2U);
    if (workspace != test_root / "holoscan-plan-artifacts-collision-regression-1") {
      std::cerr << "workspace collision did not select the next bounded candidate\n";
      return 3;
    }
    const plan_artifacts_detail::TemporaryWorkspaceCleanup workspace_cleanup{workspace};
  }

  if (!std::filesystem::exists(sentinel)) {
    std::cerr << "workspace cleanup removed another run's sentinel\n";
    return 4;
  }

  std::cout << "plan_artifacts workspace collision isolation verified\n";
  return 0;
}
