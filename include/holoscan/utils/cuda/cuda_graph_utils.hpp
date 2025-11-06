/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HOLOSCAN_UTILS_CUDA_CUDA_GRAPH_UTILS_HPP
#define HOLOSCAN_UTILS_CUDA_CUDA_GRAPH_UTILS_HPP

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "holoscan/logger/logger.hpp"
#include "holoscan/utils/cuda_macros.hpp"

namespace holoscan::utils::cuda {

/**
 * @brief Get CUDA runtime version as major and minor version numbers
 * @return std::pair<int, int> where first is major version, second is minor version
 */
inline std::pair<int, int> get_cuda_runtime_version() {
  int runtime_version = 0;
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaRuntimeGetVersion(&runtime_version),
                                 "Failed to get CUDA runtime version");

  int major = runtime_version / 1000;
  int minor = (runtime_version % 1000) / 10;

  return {major, minor};
}

/**
 * @brief Version-aware wrapper for cudaGraphAddNode that maintains backward compatibility.
 *
 * This function automatically selects the appropriate CUDA Graph API based on the runtime version:
 * - CUDA 12.x: Uses cudaGraphAddNode_v2
 * - CUDA 13.x+: Uses cudaGraphAddNode
 *
 * This function is thread-safe and uses std::call_once to ensure the symbol is resolved only once.
 *
 * @return cudaError_t CUDA error code
 */
inline cudaError_t cudaGraphAddNodeCompat(cudaGraphNode_t* pGraphNode, cudaGraph_t graph,
                                          const cudaGraphNode_t* pDependencies,
                                          const cudaGraphEdgeData* dependencyData,
                                          size_t numDependencies, cudaGraphNodeParams* nodeParams) {
  using AddNodeFn = cudaError_t (*)(cudaGraphNode_t*,
                                    cudaGraph_t,
                                    const cudaGraphNode_t*,
                                    const cudaGraphEdgeData*,
                                    size_t,
                                    cudaGraphNodeParams*);

  static std::once_flag init_flag;
  static AddNodeFn cached_fn = nullptr;
  static cudaError_t init_error = cudaErrorNotSupported;

  std::call_once(init_flag, []() {
    auto [major_version, minor_version] = get_cuda_runtime_version();

    const char* symbol_name = nullptr;
    switch (major_version) {
      case 13:
        symbol_name = "cudaGraphAddNode";
        break;
      case 12:
        if (minor_version < 6) {
          HOLOSCAN_LOG_ERROR("Unsupported CUDA version: {}.{}. Supported: 12.6+, 13.x",
                             major_version,
                             minor_version);
          init_error = cudaErrorNotSupported;
          return;
        }
        symbol_name = "cudaGraphAddNode_v2";
        break;
      default:
        HOLOSCAN_LOG_ERROR("Unsupported CUDA version: {}.{}. Supported: 12.6+, 13.x",
                           major_version,
                           minor_version);
        init_error = cudaErrorNotSupported;
        return;
    }

    void* symbol = dlsym(RTLD_DEFAULT, symbol_name);
    if (!symbol) {
      HOLOSCAN_LOG_ERROR("Failed to resolve symbol: {}", symbol_name);
      init_error = cudaErrorNotSupported;
      return;
    }

    cached_fn = reinterpret_cast<AddNodeFn>(symbol);
    init_error = cudaSuccess;
  });

  if (init_error != cudaSuccess) {
    return init_error;
  }

  return cached_fn(pGraphNode, graph, pDependencies, dependencyData, numDependencies, nodeParams);
}

}  // namespace holoscan::utils::cuda

#endif /* HOLOSCAN_UTILS_CUDA_CUDA_GRAPH_UTILS_HPP */
