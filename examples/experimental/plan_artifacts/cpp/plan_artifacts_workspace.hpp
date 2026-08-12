/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace plan_artifacts_detail {

[[nodiscard]] inline std::filesystem::path create_temporary_workspace(
    const std::filesystem::path& temporary_root, std::string_view nonce,
    std::atomic<std::uint64_t>& next_sequence, std::size_t max_attempts = 128U) {
  for (std::size_t attempt = 0U; attempt < max_attempts; ++attempt) {
    const std::uint64_t sequence = next_sequence.fetch_add(1U, std::memory_order_relaxed);
    const std::filesystem::path candidate =
        temporary_root /
        ("holoscan-plan-artifacts-" + std::string(nonce) + "-" + std::to_string(sequence));
    std::error_code create_error;
    if (std::filesystem::create_directory(candidate, create_error)) {
      return candidate;
    }
    if (create_error && create_error != std::make_error_code(std::errc::file_exists)) {
      return {};
    }
  }
  return {};
}

class TemporaryWorkspaceCleanup {
 public:
  explicit TemporaryWorkspaceCleanup(std::filesystem::path workspace)
      : workspace_(std::move(workspace)) {}

  ~TemporaryWorkspaceCleanup() {
    std::error_code cleanup_error;
    std::filesystem::remove_all(workspace_, cleanup_error);
  }

  TemporaryWorkspaceCleanup(const TemporaryWorkspaceCleanup&) = delete;
  TemporaryWorkspaceCleanup& operator=(const TemporaryWorkspaceCleanup&) = delete;

 private:
  std::filesystem::path workspace_;
};

}  // namespace plan_artifacts_detail
