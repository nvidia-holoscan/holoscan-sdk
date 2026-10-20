/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace holoscan::detail {

struct ExecutionPlanState;

/// Validate the frozen execution-partition and placement relationships.
[[nodiscard]] bool partition_records_supported(const ExecutionPlanState& plan);

}  // namespace holoscan::detail
