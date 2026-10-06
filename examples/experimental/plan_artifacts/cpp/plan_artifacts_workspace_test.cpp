/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>

#include "../../../common/temporary_workspace.hpp"

namespace {

constexpr std::string_view kNonce = "collision-regression";

#ifndef _WIN32
class UmaskGuard {
 public:
  explicit UmaskGuard(mode_t mask) : previous_mask_(::umask(mask)) {}
  ~UmaskGuard() { ::umask(previous_mask_); }

  UmaskGuard(const UmaskGuard&) = delete;
  UmaskGuard& operator=(const UmaskGuard&) = delete;

 private:
  mode_t previous_mask_;
};
#endif

}  // namespace

int main() {
#ifndef _WIN32
  // Keep peer fixtures usable independently of the caller's process mask.
  const UmaskGuard fixture_mask{0};
#endif
  const std::filesystem::path test_root = holoscan_examples::create_temporary_workspace();
  if (test_root.empty()) {
    std::cerr << "failed to create test root\n";
    return 1;
  }
  const holoscan_examples::TemporaryWorkspaceCleanup test_root_cleanup{test_root};

  const std::filesystem::path occupied = test_root / "holoscan-example-collision-regression-0";
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

  const std::filesystem::path occupied_file = test_root / "holoscan-example-collision-regression-1";
  {
    std::ofstream file{occupied_file};
    file << "another run's file\n";
    if (!file) {
      return 2;
    }
  }
  const std::filesystem::path occupied_link = test_root / "holoscan-example-collision-regression-2";
  std::error_code link_error;
  std::filesystem::create_directory_symlink(occupied, occupied_link, link_error);
#ifdef _WIN32
  // Windows symlink creation may require privileges or Developer Mode. Keep the other checks.
  if (link_error) {
    std::cout << "symlink collision fixture unavailable: " << link_error.message() << '\n';
  }
#else
  if (link_error) {
    std::cerr << "failed to create symlink collision fixture: " << link_error.message() << '\n';
    return 2;
  }
#endif
  const std::size_t occupied_count = link_error ? 2U : 3U;

  std::atomic<std::uint64_t> next_sequence{};
  const std::filesystem::path expected =
      test_root / ("holoscan-example-collision-regression-" + std::to_string(occupied_count));
  const auto check_workspace = [&]() {
    next_sequence = 0U;
    const std::filesystem::path workspace = holoscan_examples::create_temporary_workspace(
        test_root, kNonce, next_sequence, occupied_count + 1U);
    if (workspace != expected) {
      std::cerr << "workspace collision did not select the next bounded candidate\n";
      return false;
    }
    const holoscan_examples::TemporaryWorkspaceCleanup workspace_cleanup{workspace};
#ifndef _WIN32
    if (std::filesystem::status(workspace).permissions() != std::filesystem::perms::owner_all) {
      std::cerr << "workspace was not created with owner-only permissions\n";
      return false;
    }
#endif
    std::ofstream artifact{workspace / "artifact"};
    artifact << "workspace is usable\n";
    if (!artifact) {
      std::cerr << "owner could not write an artifact in the workspace\n";
      return false;
    }
    return true;
  };
#ifdef _WIN32
  if (!check_workspace()) {
    return 3;
  }
#else
  for (const mode_t mask : {mode_t{0}, mode_t{S_IXUSR}, mode_t{S_IRWXU}}) {
    // Neither a permissive mask nor masked owner bits may change workspace permissions.
    const UmaskGuard workspace_mask{mask};
    if (!check_workspace()) {
      return 3;
    }
  }
#endif

  if (std::filesystem::exists(expected) || !std::filesystem::exists(sentinel) ||
      !std::filesystem::is_regular_file(occupied_file) ||
      (!link_error && !std::filesystem::is_symlink(occupied_link))) {
    std::cerr << "workspace cleanup failed to preserve peer objects or remove its own directory\n";
    return 4;
  }

  next_sequence = 0U;
  if (!holoscan_examples::create_temporary_workspace(
           test_root, kNonce, next_sequence, occupied_count)
           .empty() ||
      !holoscan_examples::create_temporary_workspace(occupied_file, kNonce, next_sequence)
           .empty()) {
    std::cerr << "workspace creation did not fail safely for occupied or invalid roots\n";
    return 5;
  }

  next_sequence = 0U;
  for (const auto nonce : {"../escape", "..\\escape", "name:stream"}) {
    if (!holoscan_examples::create_temporary_workspace(test_root, nonce, next_sequence).empty() ||
        next_sequence != 0U) {
      std::cerr << "workspace nonce was allowed to introduce path components\n";
      return 6;
    }
  }

  if (!holoscan_examples::create_temporary_workspace(".", kNonce, next_sequence).empty() ||
      next_sequence != 0U) {
    std::cerr << "relative workspace root was accepted\n";
    return 6;
  }

#ifndef _WIN32
  const std::filesystem::path shared_parent = test_root / "shared-parent";
  const std::filesystem::path private_child = shared_parent / "private-child";
  const std::filesystem::path root_alias = test_root / "root-alias";
  std::filesystem::create_directory(shared_parent);
  std::filesystem::create_directory(private_child);
  std::filesystem::create_directory_symlink(private_child, root_alias);
  if (::chmod(private_child.c_str(), S_IRWXU) != 0) {
    return 6;
  }

  for (const mode_t mode : {mode_t{0777}, mode_t{0770}}) {
    if (::chmod(shared_parent.c_str(), mode) != 0) {
      return 6;
    }
    for (const auto& root : {shared_parent, private_child, root_alias}) {
      next_sequence = 0U;
      if (!holoscan_examples::create_temporary_workspace(root, kNonce, next_sequence).empty() ||
          next_sequence != 0U) {
        std::cerr << "unsafe temporary root or ancestor was accepted\n";
        return 6;
      }
    }
  }

  // A sticky ancestor protects its children, but a shared writable root is no longer accepted.
  if (::chmod(shared_parent.c_str(), S_ISVTX | 0777) != 0) {
    return 7;
  }
  next_sequence = 0U;
  if (!holoscan_examples::create_temporary_workspace(shared_parent, kNonce, next_sequence)
           .empty() ||
      next_sequence != 0U) {
    std::cerr << "shared writable root was accepted despite the private-root policy\n";
    return 7;
  }
  for (const auto& root : {private_child, root_alias}) {
    next_sequence = 0U;
    const std::filesystem::path workspace =
        holoscan_examples::create_temporary_workspace(root, kNonce, next_sequence);
    if (workspace.empty()) {
      std::cerr << "protected temporary root was rejected\n";
      return 7;
    }
    const holoscan_examples::TemporaryWorkspaceCleanup cleanup{workspace};
    if (workspace.parent_path() != std::filesystem::canonical(root)) {
      std::cerr << "workspace retained a mutable temporary-root alias\n";
      return 7;
    }
    std::ofstream artifact{workspace / "artifact"};
    artifact << "protected parent\n";
    if (!artifact) {
      return 7;
    }
  }

  // Root-run container tests can also exercise untrusted directory ownership.
  if (::geteuid() == 0) {
    if (::chown(shared_parent.c_str(), 1, static_cast<gid_t>(-1)) != 0) {
      return 8;
    }
    for (const auto& root : {shared_parent, private_child, root_alias}) {
      next_sequence = 0U;
      if (!holoscan_examples::create_temporary_workspace(root, kNonce, next_sequence).empty() ||
          next_sequence != 0U) {
        std::cerr << "foreign-owned temporary root or ancestor was accepted\n";
        return 8;
      }
    }
  }

#endif

  std::cout << "plan_artifacts workspace collision isolation verified\n";
  return 0;
}
