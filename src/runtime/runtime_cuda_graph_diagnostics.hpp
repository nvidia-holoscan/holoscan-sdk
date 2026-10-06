/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string_view>

namespace holoscan::detail {

inline constexpr std::string_view kCUDAGraphRuntimeInitializationFailed =
    "CUDA_GRAPH_RUNTIME_INITIALIZATION_FAILED";
inline constexpr std::string_view kCUDAGraphRuntimeExecutionFailed =
    "CUDA_GRAPH_RUNTIME_EXECUTION_FAILED";
inline constexpr std::string_view kCUDAGraphRuntimeQuiesceFailed =
    "CUDA_GRAPH_RUNTIME_QUIESCE_FAILED";
inline constexpr std::string_view kCUDAGraphRuntimeShutdownFailed =
    "CUDA_GRAPH_RUNTIME_SHUTDOWN_FAILED";
inline constexpr std::string_view kCUDAGraphRuntimeIngressMismatch =
    "CUDA_GRAPH_RUNTIME_INGRESS_MISMATCH";

}  // namespace holoscan::detail
