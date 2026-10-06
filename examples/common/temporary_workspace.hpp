/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#ifdef _WIN32
#pragma message("Warning: Windows workspaces inherit ACLs; owner-only access is not enforced.")
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace holoscan_examples {

[[nodiscard]] inline std::filesystem::path trusted_temporary_root(
    const std::filesystem::path& temporary_root) {
  if (!temporary_root.is_absolute()) {
    return {};
  }
  std::error_code error;
  const std::filesystem::path resolved = std::filesystem::canonical(temporary_root, error);
  if (error) {
    return {};
  }

#ifdef _WIN32
  // Compatibility fallback only: callers must choose a trusted root; ACLs are not validated.
  if (!std::filesystem::is_directory(resolved, error) || error) {
    return {};
  }
#else
  // Validate from the filesystem root down: a private leaf is not protected if a peer can
  // replace one of its ancestors. Trust this effective user and root, not arbitrary owners.
  // Shared directories require the sticky bit; their children must also have trusted owners.
  // This assumes enforced POSIX modes/sticky bits without additional peer access via ACLs.
  const uid_t owner = ::geteuid();
  std::filesystem::path ancestor;
  for (const auto& component : resolved) {
    ancestor /= component;
    struct stat metadata{};
    if (::lstat(ancestor.c_str(), &metadata) != 0 || !S_ISDIR(metadata.st_mode) ||
        (metadata.st_uid != owner && metadata.st_uid != 0) ||
        ((metadata.st_mode & (S_IWGRP | S_IWOTH)) != 0 && (metadata.st_mode & S_ISVTX) == 0)) {
      return {};
    }
  }
  struct stat root_metadata{};
  if (::lstat(resolved.c_str(), &root_metadata) != 0 || root_metadata.st_uid != owner ||
      (root_metadata.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
    return {};
  }
#endif
  // Use the validated absolute path for creation, artifact access, and cleanup, not an alias.
  return resolved;
}

[[nodiscard]] inline std::filesystem::path create_temporary_workspace(
    const std::filesystem::path& temporary_root, std::string_view nonce,
    std::atomic<std::uint64_t>& next_sequence, std::size_t max_attempts = 128U) {
  const std::filesystem::path root = trusted_temporary_root(temporary_root);
  if (root.empty() || nonce.find_first_of("/\\:") != std::string_view::npos) {
    return {};
  }
  for (std::size_t attempt = 0U; attempt < max_attempts; ++attempt) {
    const std::uint64_t sequence = next_sequence.fetch_add(1U, std::memory_order_relaxed);
    const std::filesystem::path candidate =
        root / ("holoscan-example-" + std::string(nonce) + "-" + std::to_string(sequence));
#ifdef _WIN32
    // Only own a newly created directory. Existing candidates belong to another run.
    // Permissions are inherited from the parent; this does not provide POSIX-equivalent privacy.
    std::error_code error;
    if (std::filesystem::create_directory(candidate, error)) {
      return candidate;
    }
    if (error && error != std::errc::file_exists) {
      return {};
    }
#else
    // Create an owner-only directory atomically on POSIX:
    // do not expose artifacts to peers while tightening initially broader permissions.
    // Exclusive mkdir also rejects pre-existing files, directories, and symlinks.
    if (::mkdir(candidate.c_str(), S_IRWXU) == 0) {
      // mkdir applies umask even to owner bits. Restore them without ever granting peer access.
      if (::chmod(candidate.c_str(), S_IRWXU) != 0) {
        ::rmdir(candidate.c_str());
        return {};
      }
      return candidate;
    }
    if (errno != EEXIST) {
      return {};
    }
#endif
  }
  return {};
}

[[nodiscard]] inline std::filesystem::path create_temporary_workspace() {
  static std::atomic<std::uint64_t> next_sequence{};
  // Use the user's home by default, never the shared system temporary area. An explicit
  // override that fails validation is an error, not a reason to silently fall back elsewhere.
  const char* workspace_root = std::getenv("HOLOSCAN_EXAMPLE_WORKSPACE_ROOT");
  if (workspace_root == nullptr) {
#ifdef _WIN32
    workspace_root = std::getenv("USERPROFILE");
#else
    workspace_root = std::getenv("HOME");
#endif
  }
  if (workspace_root == nullptr || *workspace_root == '\0') {
    return {};
  }
  const std::string nonce =
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  return create_temporary_workspace(workspace_root, nonce, next_sequence);
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

}  // namespace holoscan_examples
