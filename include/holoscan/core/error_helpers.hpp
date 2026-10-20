/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

#include <fmt/format.h>

#include <string>
#include <string_view>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

namespace holoscan::error_detail {

[[nodiscard]] inline Error runtime_error(ErrorCode code, std::string_view message) {
  return Error{code, std::string(message)};
}

[[nodiscard]] inline Error cuda_runtime_error(std::string_view message, cudaError_t code) {
  return Error{
      ErrorCode::kFailure,
      fmt::format("{}: {} ({})", message, cudaGetErrorString(code), static_cast<int>(code))};
}

[[nodiscard]] inline unexpected<Error> unexpected_runtime_error(ErrorCode code,
                                                                std::string_view message) {
  return make_unexpected(runtime_error(code, message));
}

[[nodiscard]] inline unexpected<Error> unexpected_cuda_runtime_error(std::string_view message,
                                                                     cudaError_t code) {
  return make_unexpected(cuda_runtime_error(message, code));
}

[[nodiscard]] inline expected<void, Error> make_cuda_error(cudaError_t code,
                                                           std::string_view operation) {
  if (code == cudaSuccess) {
    return {};
  }
  return unexpected_cuda_runtime_error(operation, code);
}

}  // namespace holoscan::error_detail
