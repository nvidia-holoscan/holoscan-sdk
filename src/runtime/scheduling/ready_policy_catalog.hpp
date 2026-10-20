/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include "ready_policy.hpp"

namespace holoscan::detail::scheduling {

using ReadyPolicyFactory = std::unique_ptr<ReadyPolicy> (*)();

/// Bind one immutable private policy descriptor to an explicit factory.
struct ReadyPolicyRegistration {
  ReadyPolicyDescriptor descriptor;
  ReadyPolicyFactory create{};
};

inline constexpr std::size_t kMaxReadyPolicyRegistrations = 4U;

/// Own a bounded constructor-injected catalog of private READY policies.
///
/// The baseline built-in catalog contains exactly one production entry. The larger fixed
/// registration capacity exists only for controlled tests and later private implementations; it
/// creates no installed extension ABI, self-registration path, dynamic loader, or runtime selector.
class ReadyPolicyCatalog {
 public:
  /// Validate and copy an explicit bounded registration table.
  [[nodiscard]] static expected<ReadyPolicyCatalog, Error> create(
      std::span<const ReadyPolicyRegistration> registrations,
      ReadyPolicyCapabilities available_capabilities = kAllReadyPolicyCapabilities);

  /// Create the exact one-entry baseline deterministic-serial catalog.
  [[nodiscard]] static expected<ReadyPolicyCatalog, Error> create_default_builtins();

  /// Construct a fresh policy for one exact algorithm identifier and version.
  [[nodiscard]] expected<std::unique_ptr<ReadyPolicy>, Error> make(
      const AlgorithmKey& algorithm) const;

  /// Return the number of validated entries in this catalog.
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

 private:
  struct Entry {
    std::string algorithm_id;
    std::uint32_t algorithm_version{};
    std::uint32_t interface_version{};
    AlgorithmMaturity maturity{AlgorithmMaturity::kProduction};
    ReadyPolicyCapabilities required_capabilities;
    ReadyPolicyFactory create{};
  };

  ReadyPolicyCatalog(std::array<Entry, kMaxReadyPolicyRegistrations> entries, std::size_t size)
      : entries_(std::move(entries)), size_(size) {}

  std::array<Entry, kMaxReadyPolicyRegistrations> entries_{};
  std::size_t size_{};
};

}  // namespace holoscan::detail::scheduling
