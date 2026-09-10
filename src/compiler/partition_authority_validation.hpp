/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace holoscan::detail {

struct ExecutionPlanState;

/// Validate the complete canonical authority for one accepted execution-partition cover.
[[nodiscard]] bool valid_accepted_partition_authority(const ExecutionPlanState& state);

}  // namespace holoscan::detail
