/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/cuda_execution/cuda_green_context.hpp>

#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE

#include <string>
#include <utility>

namespace holoscan {

CudaGreenContext::CudaGreenContext(const CudaGreenContextPool* pool, uint32_t index,
                                   CUgreenCtx green_context, CUcontext cuda_context,
                                   std::string nvtx_identifier)
    : pool_(pool),
      index_(index),
      green_context_(green_context),
      cuda_context_(cuda_context),
      nvtx_identifier_(std::move(nvtx_identifier)) {}

expected<CudaGreenContext, Error> CudaGreenContext::create(const CudaGreenContextPool& pool,
                                                           std::optional<uint32_t> index,
                                                           std::string nvtx_identifier) {
  uint32_t resolved_index = 0;
  if (index) {
    resolved_index = *index;
  } else {
    auto default_index = pool.default_context_index();
    if (!default_index) {
      return forward_error(default_index);
    }
    resolved_index = default_index.value();
  }

  auto green_context = pool.green_context(resolved_index);
  if (!green_context) {
    return forward_error(green_context);
  }
  auto cuda_context = pool.cuda_context(resolved_index);
  if (!cuda_context) {
    return forward_error(cuda_context);
  }

  return CudaGreenContext(&pool,
                          resolved_index,
                          green_context.value(),
                          cuda_context.value(),
                          std::move(nvtx_identifier));
}

expected<CUgreenCtx, Error> CudaGreenContext::green_context() const {
  if (pool_ == nullptr || green_context_ == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "CudaGreenContext is not bound to a pool"});
  }
  return green_context_;
}

expected<CUcontext, Error> CudaGreenContext::cuda_context() const {
  if (pool_ == nullptr || cuda_context_ == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "CudaGreenContext is not bound to a pool"});
  }
  return cuda_context_;
}

}  // namespace holoscan

#endif  // HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
