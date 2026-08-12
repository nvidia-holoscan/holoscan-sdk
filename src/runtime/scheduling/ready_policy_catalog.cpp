/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ready_policy_catalog.hpp"

#include <algorithm>
#include <array>
#include <string>
#include <string_view>
#include <utility>

#include "deterministic_serial_ready_policy.hpp"

namespace holoscan::detail::scheduling {
namespace {

[[nodiscard]] unexpected<Error> catalog_error(ErrorCode code, std::string_view token) {
  return make_unexpected(Error{code, std::string{token}});
}

[[nodiscard]] bool same_algorithm(const AlgorithmKey& lhs, const AlgorithmKey& rhs) noexcept {
  return lhs.id == rhs.id && lhs.version == rhs.version;
}

[[nodiscard]] bool valid_maturity(AlgorithmMaturity maturity) noexcept {
  return maturity == AlgorithmMaturity::kProduction || maturity == AlgorithmMaturity::kExperimental;
}

[[nodiscard]] bool descriptor_matches(const ReadyPolicyDescriptor& lhs,
                                      const ReadyPolicyDescriptor& rhs) noexcept {
  return same_algorithm(lhs.algorithm, rhs.algorithm) &&
         lhs.interface_version == rhs.interface_version && lhs.maturity == rhs.maturity &&
         lhs.required_capabilities == rhs.required_capabilities;
}

}  // namespace

expected<ReadyPolicyCatalog, Error> ReadyPolicyCatalog::create(
    std::span<const ReadyPolicyRegistration> registrations,
    ReadyPolicyCapabilities available_capabilities) {
  if (registrations.empty() || registrations.size() > kMaxReadyPolicyRegistrations) {
    return catalog_error(ErrorCode::kInvalidArgument, "READY_POLICY_REGISTRATION_INVALID");
  }

  std::array<Entry, kMaxReadyPolicyRegistrations> entries{};
  for (std::size_t index = 0U; index < registrations.size(); ++index) {
    const auto& registration = registrations[index];
    const auto& descriptor = registration.descriptor;
    if (descriptor.algorithm.id.empty() || descriptor.algorithm.version == 0U ||
        !valid_maturity(descriptor.maturity) || registration.create == nullptr ||
        !kAllReadyPolicyCapabilities.contains(descriptor.required_capabilities)) {
      return catalog_error(ErrorCode::kInvalidArgument, "READY_POLICY_REGISTRATION_INVALID");
    }
    if (descriptor.interface_version != kReadyPolicyInterfaceVersion) {
      return catalog_error(ErrorCode::kNotSupported, "READY_POLICY_INTERFACE_VERSION_UNSUPPORTED");
    }
    if (!available_capabilities.contains(descriptor.required_capabilities)) {
      return catalog_error(ErrorCode::kNotSupported, "READY_POLICY_PLAN_INVALID");
    }

    for (std::size_t prior = 0U; prior < index; ++prior) {
      const auto& other = registrations[prior];
      if (same_algorithm(descriptor.algorithm, other.descriptor.algorithm) ||
          registration.create == other.create) {
        return catalog_error(ErrorCode::kInvalidArgument, "READY_POLICY_REGISTRATION_INVALID");
      }
    }

    auto implementation = registration.create();
    if (!implementation || !descriptor_matches(descriptor, implementation->descriptor())) {
      return catalog_error(ErrorCode::kInvalidArgument, "READY_POLICY_REGISTRATION_INVALID");
    }

    entries[index] = Entry{
        .algorithm_id = descriptor.algorithm.id,
        .algorithm_version = descriptor.algorithm.version,
        .interface_version = descriptor.interface_version,
        .maturity = descriptor.maturity,
        .required_capabilities = descriptor.required_capabilities,
        .create = registration.create,
    };
  }

  std::sort(entries.begin(),
            entries.begin() + registrations.size(),
            [](const Entry& lhs, const Entry& rhs) {
              if (lhs.algorithm_id != rhs.algorithm_id) {
                return lhs.algorithm_id < rhs.algorithm_id;
              }
              return lhs.algorithm_version < rhs.algorithm_version;
            });
  return ReadyPolicyCatalog{std::move(entries), registrations.size()};
}

expected<ReadyPolicyCatalog, Error> ReadyPolicyCatalog::create_ea1_builtins() {
  const DeterministicSerialReadyPolicy implementation;
  const std::array registrations{
      ReadyPolicyRegistration{implementation.descriptor(), &make_deterministic_serial_ready_policy},
  };
  return create(registrations);
}

expected<std::unique_ptr<ReadyPolicy>, Error> ReadyPolicyCatalog::make(
    const AlgorithmKey& algorithm) const {
  bool known_algorithm_id = false;
  for (std::size_t index = 0U; index < size_; ++index) {
    const auto& entry = entries_[index];
    if (entry.algorithm_id != algorithm.id) {
      continue;
    }
    known_algorithm_id = true;
    if (entry.algorithm_version != algorithm.version) {
      continue;
    }

    auto policy = entry.create();
    const ReadyPolicyDescriptor expected_descriptor{
        .algorithm = AlgorithmKey{entry.algorithm_id, entry.algorithm_version},
        .interface_version = entry.interface_version,
        .maturity = entry.maturity,
        .required_capabilities = entry.required_capabilities,
    };
    if (!policy || !descriptor_matches(expected_descriptor, policy->descriptor())) {
      return catalog_error(ErrorCode::kFailure, "READY_POLICY_REGISTRATION_INVALID");
    }
    return policy;
  }

  if (known_algorithm_id) {
    return catalog_error(ErrorCode::kNotSupported, "READY_POLICY_VERSION_UNSUPPORTED");
  }
  return catalog_error(ErrorCode::kNotFound, "READY_POLICY_NOT_FOUND");
}

}  // namespace holoscan::detail::scheduling
