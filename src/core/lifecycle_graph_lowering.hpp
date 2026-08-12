/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/core/expected.hpp>

#include "graph_snapshot.hpp"
#include "lifecycle_kernel.hpp"

namespace holoscan::detail {

/// Exact private lifecycle-normalizer input and its derived finite minimum capacity.
struct LifecycleGraphLowering {
  LifecycleStageGraphInput input;
  LifecycleKernelCapacity minimum_capacity;
};

/// Lower one canonical Graph snapshot into pointer-free lifecycle normalization records.
///
/// The caller must pass the result of `GraphCompilerAccess::canonical_snapshot`. The translator
/// preserves Graph-instance scope hierarchy, assigns dense checked lifecycle node identities, and
/// never derives identity from an authoring address or unordered-container traversal.
[[nodiscard]] expected<LifecycleGraphLowering, LifecycleKernelError>
lower_lifecycle_stage_graph_input(const GraphSnapshotData& snapshot);

}  // namespace holoscan::detail
