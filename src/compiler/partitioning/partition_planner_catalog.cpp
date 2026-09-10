/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "partition_planner_catalog.hpp"

#include <algorithm>
#include <array>
#include <string>
#include <utility>
#include <vector>

#include "serial_baseline_partition_planner.hpp"
#include "source_chain_partition_planner.hpp"

namespace holoscan::detail::partitioning {
namespace {

[[nodiscard]] unexpected<Error> catalog_error(ErrorCode code, std::string_view token) {
  return make_unexpected(Error{code, std::string(token)});
}

[[nodiscard]] bool same_algorithm(const AlgorithmKey& lhs, const AlgorithmKey& rhs) noexcept {
  return lhs.id == rhs.id && lhs.version == rhs.version;
}

[[nodiscard]] bool descriptor_matches(const PartitionPlannerDescriptor& lhs,
                                      const PartitionPlannerDescriptor& rhs) noexcept {
  return same_algorithm(lhs.algorithm, rhs.algorithm) &&
         lhs.interface_version == rhs.interface_version && lhs.maturity == rhs.maturity &&
         lhs.required_capabilities == rhs.required_capabilities;
}

}  // namespace

expected<PartitionPlannerCatalog, Error> PartitionPlannerCatalog::create(
    std::span<const PartitionPlannerRegistration> registrations,
    PartitionPlannerCapabilities available_capabilities) {
  std::vector<Entry> entries;
  entries.reserve(registrations.size());

  for (std::size_t index = 0; index < registrations.size(); ++index) {
    const auto& registration = registrations[index];
    const auto& descriptor = registration.descriptor;
    if (descriptor.algorithm.id.empty() || descriptor.algorithm.version == 0U ||
        registration.create == nullptr) {
      return catalog_error(ErrorCode::kInvalidArgument, "PARTITION_PLANNER_REGISTRATION_INVALID");
    }
    if (descriptor.interface_version != kPartitionPlannerInterfaceVersion) {
      return catalog_error(ErrorCode::kNotSupported,
                           "PARTITION_PLANNER_INTERFACE_VERSION_UNSUPPORTED");
    }
    if (!available_capabilities.contains(descriptor.required_capabilities)) {
      return catalog_error(ErrorCode::kNotSupported, "PARTITION_PLANNER_CAPABILITY_UNSUPPORTED");
    }

    for (std::size_t prior = 0; prior < index; ++prior) {
      const auto& other = registrations[prior];
      if (same_algorithm(descriptor.algorithm, other.descriptor.algorithm) ||
          registration.create == other.create) {
        return catalog_error(ErrorCode::kInvalidArgument, "PARTITION_PLANNER_REGISTRATION_INVALID");
      }
    }

    auto planner = registration.create();
    if (!planner || !descriptor_matches(descriptor, planner->descriptor())) {
      return catalog_error(ErrorCode::kInvalidArgument, "PARTITION_PLANNER_REGISTRATION_INVALID");
    }

    entries.push_back(Entry{std::string(descriptor.algorithm.id),
                            descriptor.algorithm.version,
                            descriptor.interface_version,
                            descriptor.maturity,
                            descriptor.required_capabilities,
                            registration.create});
  }

  std::sort(entries.begin(), entries.end(), [](const Entry& lhs, const Entry& rhs) {
    if (lhs.algorithm_id != rhs.algorithm_id) {
      return lhs.algorithm_id < rhs.algorithm_id;
    }
    return lhs.algorithm_version < rhs.algorithm_version;
  });
  return PartitionPlannerCatalog{std::move(entries)};
}

expected<PartitionPlannerCatalog, Error> PartitionPlannerCatalog::create_default_builtins() {
  const std::array registrations{
      PartitionPlannerRegistration{SerialBaselinePartitionPlanner{}.descriptor(),
                                   &make_serial_baseline_partition_planner},
      PartitionPlannerRegistration{SourceChainPartitionPlanner{}.descriptor(),
                                   &make_source_chain_partition_planner},
  };
  return create(registrations);
}

expected<std::unique_ptr<PartitionPlanner>, Error> PartitionPlannerCatalog::make(
    const AlgorithmKey& algorithm) const {
  bool known_algorithm_id = false;
  for (const auto& entry : entries_) {
    if (entry.algorithm_id != algorithm.id) {
      continue;
    }
    known_algorithm_id = true;
    if (entry.algorithm_version != algorithm.version) {
      continue;
    }

    auto planner = entry.create();
    if (!planner) {
      return catalog_error(ErrorCode::kFailure, "PARTITION_PLANNER_REGISTRATION_INVALID");
    }
    const PartitionPlannerDescriptor expected_descriptor{
        AlgorithmKey{entry.algorithm_id, entry.algorithm_version},
        entry.interface_version,
        entry.maturity,
        entry.required_capabilities};
    if (!descriptor_matches(expected_descriptor, planner->descriptor())) {
      return catalog_error(ErrorCode::kFailure, "PARTITION_PLANNER_REGISTRATION_INVALID");
    }
    return std::move(planner);
  }

  if (known_algorithm_id) {
    return catalog_error(ErrorCode::kNotSupported,
                         "PARTITION_PLANNER_ALGORITHM_VERSION_UNSUPPORTED");
  }
  return catalog_error(ErrorCode::kNotFound, "PARTITION_PLANNER_NOT_FOUND");
}

}  // namespace holoscan::detail::partitioning
