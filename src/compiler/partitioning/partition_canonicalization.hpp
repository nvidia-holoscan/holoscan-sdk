/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <vector>

#include "partition_types.hpp"

namespace holoscan::detail::partitioning {

/// Append mandatory boundary reasons in their canonical decision order.
inline void append_boundary_reasons(PartitionBoundaryMask mask,
                                    std::vector<PartitionDecisionReason>& reasons) {
  if (mask.contains(PartitionBoundary::kProcess)) {
    reasons.push_back(PartitionDecisionReason::kProcessBoundary);
  }
  if (mask.contains(PartitionBoundary::kExternalEndpoint)) {
    reasons.push_back(PartitionDecisionReason::kExternalEndpointBoundary);
  }
  if (mask.contains(PartitionBoundary::kDeviceTransfer)) {
    reasons.push_back(PartitionDecisionReason::kDeviceTransferIncompatibility);
  }
  if (mask.contains(PartitionBoundary::kRegionBackend)) {
    reasons.push_back(PartitionDecisionReason::kRegionBackendAdmission);
  }
  if (mask.contains(PartitionBoundary::kExplicitThreadPlacement)) {
    reasons.push_back(PartitionDecisionReason::kExplicitThreadPlacement);
  }
  if (mask.contains(PartitionBoundary::kLifecycle)) {
    reasons.push_back(PartitionDecisionReason::kLifecycleIsolation);
  }
  if (mask.contains(PartitionBoundary::kReadiness)) {
    reasons.push_back(PartitionDecisionReason::kReadinessIsolation);
  }
}

/// Sort values with the supplied canonical comparator and remove duplicates.
template <typename T, typename Compare>
void sort_and_unique(std::vector<T>& values, Compare compare) {
  std::sort(values.begin(), values.end(), compare);
  values.erase(std::unique(values.begin(), values.end()), values.end());
}

}  // namespace holoscan::detail::partitioning
