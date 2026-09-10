/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cu_errors.hpp"
#include <holoviz/logger.hpp>
#include <magic_enum/magic_enum.hpp>

#include <string>

namespace holoviz::rhi::cuda {

bool checkCuResult(CUresult result, const char* file, int32_t line) {
  if (result == CUDA_SUCCESS) {
    return false;
  }

  HOLOVIZ_LOG_ERROR("CUDA error at {}:{} - {} ({})",
                    file,
                    line,
                    magic_enum::enum_name(result),
                    static_cast<int>(result));

  return true;
}

std::string cu_error_string(CUresult result, const char* file, int32_t line) {
  std::string error_string = "CUDA error at " + std::string(file) + ":" + std::to_string(line) +
                             " - " + std::string(magic_enum::enum_name(result)) + " (" +
                             std::to_string(static_cast<int>(result)) + ")";
  return error_string;
}

}  // namespace holoviz::rhi::cuda
