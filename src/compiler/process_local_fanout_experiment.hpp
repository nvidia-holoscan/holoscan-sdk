/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace holoscan::detail {

/// Source-tree experiment only; never selected by ordinary compilation.
enum class ProcessLocalFanoutExperiment : std::uint8_t {
  kDisabled,
  kPostComputeCut,
  kEmitRecipients,
};

}  // namespace holoscan::detail
