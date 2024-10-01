/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_UTILS_CUDA_MACROS_HPP
#define HOLOSCAN_UTILS_CUDA_MACROS_HPP

#include <cuda_runtime.h>

#include <common/logger.hpp>  // GXF_LOG_ERROR, GXF_LOG_WARNING

// Note: ({ ... }) here is a GNU statement expression and not standard C++
// see: https://gcc.gnu.org/onlinedocs/gcc/Statement-Exprs.html
// It should be supported by both clang and gcc, but maybe not by MSVC
#define HOLOSCAN_CUDA_CALL(stmt)                                                           \
  ({                                                                                       \
    cudaError_t _holoscan_cuda_err = stmt;                                                 \
    if (cudaSuccess != _holoscan_cuda_err) {                                               \
      GXF_LOG_ERROR("CUDA Runtime call %s in line %d of file %s failed with '%s' (%d).\n", \
                    #stmt,                                                                 \
                    __LINE__,                                                              \
                    __FILE__,                                                              \
                    cudaGetErrorString(_holoscan_cuda_err),                                \
                    static_cast<int>(_holoscan_cuda_err));                                 \
    }                                                                                      \
    _holoscan_cuda_err;                                                                    \
  })

#define HOLOSCAN_CUDA_CALL_WARN(stmt)                                                        \
  ({                                                                                         \
    cudaError_t _holoscan_cuda_err = stmt;                                                   \
    if (cudaSuccess != _holoscan_cuda_err) {                                                 \
      GXF_LOG_WARNING("CUDA Runtime call %s in line %d of file %s failed with '%s' (%d).\n", \
                      #stmt,                                                                 \
                      __LINE__,                                                              \
                      __FILE__,                                                              \
                      cudaGetErrorString(_holoscan_cuda_err),                                \
                      static_cast<int>(_holoscan_cuda_err));                                 \
    }                                                                                        \
    _holoscan_cuda_err;                                                                      \
  })

#define HOLOSCAN_CUDA_CALL_WARN_MSG(stmt, ...)                                 \
  ({                                                                           \
    cudaError_t _holoscan_cuda_err = HOLOSCAN_CUDA_CALL_WARN(stmt);            \
    if (_holoscan_cuda_err != cudaSuccess) { HOLOSCAN_LOG_WARN(__VA_ARGS__); } \
    _holoscan_cuda_err;                                                        \
  })

#define HOLOSCAN_CUDA_CALL_ERR_MSG(stmt, ...)                                   \
  ({                                                                            \
    cudaError_t _holoscan_cuda_err = HOLOSCAN_CUDA_CALL(stmt);                  \
    if (_holoscan_cuda_err != cudaSuccess) { HOLOSCAN_LOG_ERROR(__VA_ARGS__); } \
    _holoscan_cuda_err;                                                         \
  })

#define HOLOSCAN_CUDA_CALL_THROW_ERROR(stmt, ...)                                           \
  do {                                                                                      \
    if (HOLOSCAN_CUDA_CALL(stmt) != cudaSuccess) { throw std::runtime_error(__VA_ARGS__); } \
  } while (0)

#endif /* HOLOSCAN_UTILS_CUDA_MACROS_HPP */
