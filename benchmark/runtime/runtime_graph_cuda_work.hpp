/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace holoscan::benchmarking {

/// Enqueue bounded synthetic device work without synchronizing the calling host thread.
[[nodiscard]] cudaError_t launch_runtime_graph_cuda_work(cudaStream_t stream, void* scratch,
                                                         std::uint64_t duration_cycles,
                                                         std::uint32_t block_count) noexcept;

}  // namespace holoscan::benchmarking
