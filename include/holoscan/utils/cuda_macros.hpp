/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_UTILS_CUDA_MACROS_HPP
#define HOLOSCAN_UTILS_CUDA_MACROS_HPP

#include <cuda_runtime.h>

#include <holoscan/logger/logger.hpp>  // HOLOSCAN_LOG_ERROR, HOLOSCAN_LOG_WARN

// Note: ({ ... }) here is a GNU statement expression and not standard C++
// see: https://gcc.gnu.org/onlinedocs/gcc/Statement-Exprs.html
// It should be supported by both clang and gcc, but maybe not by MSVC
#define HOLOSCAN_CUDA_CALL(stmt)                                                              \
  ({                                                                                          \
    cudaError_t _holoscan_cuda_err = stmt;                                                    \
    if (cudaSuccess != _holoscan_cuda_err) {                                                  \
      HOLOSCAN_LOG_ERROR("CUDA Runtime call {} in line {} of file {} failed with '{}' ({}).", \
                         #stmt,                                                               \
                         __LINE__,                                                            \
                         __FILE__,                                                            \
                         cudaGetErrorString(_holoscan_cuda_err),                              \
                         static_cast<int>(_holoscan_cuda_err));                               \
    }                                                                                         \
    _holoscan_cuda_err;                                                                       \
  })

#define HOLOSCAN_CUDA_CALL_WARN(stmt)                                                        \
  ({                                                                                         \
    cudaError_t _holoscan_cuda_err = stmt;                                                   \
    if (cudaSuccess != _holoscan_cuda_err) {                                                 \
      HOLOSCAN_LOG_WARN("CUDA Runtime call {} in line {} of file {} failed with '{}' ({}).", \
                        #stmt,                                                               \
                        __LINE__,                                                            \
                        __FILE__,                                                            \
                        cudaGetErrorString(_holoscan_cuda_err),                              \
                        static_cast<int>(_holoscan_cuda_err));                               \
    }                                                                                        \
    _holoscan_cuda_err;                                                                      \
  })

#define HOLOSCAN_CUDA_CALL_DEBUG(stmt)                                                        \
  ({                                                                                          \
    cudaError_t _holoscan_cuda_err = stmt;                                                    \
    if (cudaSuccess != _holoscan_cuda_err) {                                                  \
      HOLOSCAN_LOG_DEBUG("CUDA Runtime call {} in line {} of file {} failed with '{}' ({}).", \
                         #stmt,                                                               \
                         __LINE__,                                                            \
                         __FILE__,                                                            \
                         cudaGetErrorString(_holoscan_cuda_err),                              \
                         static_cast<int>(_holoscan_cuda_err));                               \
    }                                                                                         \
    _holoscan_cuda_err;                                                                       \
  })

#define HOLOSCAN_CUDA_CALL_DEBUG_MSG(stmt, ...)                      \
  ({                                                                 \
    cudaError_t _holoscan_cuda_err = HOLOSCAN_CUDA_CALL_DEBUG(stmt); \
    if (_holoscan_cuda_err != cudaSuccess) {                         \
      HOLOSCAN_LOG_DEBUG(__VA_ARGS__);                               \
    }                                                                \
    _holoscan_cuda_err;                                              \
  })

#define HOLOSCAN_CUDA_CALL_WARN_MSG(stmt, ...)                      \
  ({                                                                \
    cudaError_t _holoscan_cuda_err = HOLOSCAN_CUDA_CALL_WARN(stmt); \
    if (_holoscan_cuda_err != cudaSuccess) {                        \
      HOLOSCAN_LOG_WARN(__VA_ARGS__);                               \
    }                                                               \
    _holoscan_cuda_err;                                             \
  })

#define HOLOSCAN_CUDA_CALL_ERR_MSG(stmt, ...)                  \
  ({                                                           \
    cudaError_t _holoscan_cuda_err = HOLOSCAN_CUDA_CALL(stmt); \
    if (_holoscan_cuda_err != cudaSuccess) {                   \
      HOLOSCAN_LOG_ERROR(__VA_ARGS__);                         \
    }                                                          \
    _holoscan_cuda_err;                                        \
  })

#define HOLOSCAN_CUDA_CALL_THROW_ERROR(stmt, ...)  \
  do {                                             \
    if (HOLOSCAN_CUDA_CALL(stmt) != cudaSuccess) { \
      throw std::runtime_error(__VA_ARGS__);       \
    }                                              \
  } while (0)

#endif /* HOLOSCAN_UTILS_CUDA_MACROS_HPP */
