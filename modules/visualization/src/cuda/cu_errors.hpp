/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoviz/holoviz_export.h>
#include <holoviz/errors.hpp>
#include <holoviz/expected.hpp>

#include <cuda.h>

#include <cstdint>
#include <string>

namespace holoviz::cuda {

/**
 * @brief Check the result of a CUDA function call and log an error if it indicates a failure.
 * @param result The CUDA result to check.
 * @param file The file where the CUDA call was made.
 * @param line The line number where the CUDA call was made.
 * @return true if the CUDA call failed, false on success.
 */
[[nodiscard]] HOLOVIZ_EXPORT bool checkCuResult(CUresult result, const char* file, int32_t line);

HOLOVIZ_EXPORT std::string cu_error_string(CUresult result, const char* file, int32_t line);

#ifndef VIZ_CU_CHECK
#define VIZ_CU_CHECK(result) holoviz::cuda::checkCuResult(result, __FILE__, __LINE__)
#endif

#ifndef VIZ_CU_FORWARD_ERROR
#define VIZ_CU_FORWARD_ERROR(result)                                                   \
  {                                                                                    \
    const CUresult _cu_result = result;                                                \
    if (VIZ_CU_CHECK(_cu_result))                                                      \
      return holoviz::make_unexpected(holoviz::Error{                                  \
          .code = ErrorCode::kCudaError,                                               \
          .message = holoviz::cuda::cu_error_string(_cu_result, __FILE__, __LINE__)}); \
  }
#endif

// Best-effort teardown helper: logs a failed CUDA call (via VIZ_CU_CHECK) and, if
// result_var still holds success, records the error there instead of returning,
// so destroy() keeps freeing the remaining resources and reports the first error.
#ifndef VIZ_CU_ACCUMULATE_ERROR
#define VIZ_CU_ACCUMULATE_ERROR(result_var, call)                                      \
  {                                                                                    \
    const CUresult _cu_result = (call);                                                \
    if (VIZ_CU_CHECK(_cu_result) && (result_var))                                      \
      (result_var) = holoviz::make_unexpected(holoviz::Error{                          \
          .code = ErrorCode::kCudaError,                                               \
          .message = holoviz::cuda::cu_error_string(_cu_result, __FILE__, __LINE__)}); \
  }
#endif

}  // namespace holoviz::cuda
