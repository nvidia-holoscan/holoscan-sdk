/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/cuda_execution/cuda_green_context_pool.hpp>

namespace holoscan {

#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE

/**
 * @brief Native experimental counterpart of nvidia::gxf::CudaGreenContext.
 *
 * A lightweight, non-owning handle to one partition of a CudaGreenContextPool. The pool owns the
 * underlying CUgreenCtx / CUcontext; this type just resolves a partition index (defaulting to the
 * pool's default context) and exposes the borrowed handles. It has no GXF or holoscan::core
 * dependency and reports failures through expected<T, Error>.
 *
 * The referenced pool must outlive the CudaGreenContext.
 */
class CudaGreenContext {
 public:
  CudaGreenContext() = default;

  /**
   * @brief Bind to a partition of @p pool.
   *
   * @param pool             an initialized green context pool (must outlive this handle)
   * @param index            partition index; if unset, the pool's default context index is used
   * @param nvtx_identifier  name retained for profiling/diagnostics
   */
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<CudaGreenContext, Error> create(
      const CudaGreenContextPool& pool, std::optional<uint32_t> index = std::nullopt,
      std::string nvtx_identifier = "cuda_green_context");

  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<CUgreenCtx, Error> green_context() const;
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<CUcontext, Error> cuda_context() const;

  [[nodiscard]] uint32_t index() const noexcept { return index_; }
  [[nodiscard]] const std::string& nvtx_identifier() const noexcept { return nvtx_identifier_; }
  [[nodiscard]] explicit operator bool() const noexcept { return pool_ != nullptr; }

 private:
  CudaGreenContext(const CudaGreenContextPool* pool, uint32_t index, CUgreenCtx green_context,
                   CUcontext cuda_context, std::string nvtx_identifier);

  const CudaGreenContextPool* pool_ = nullptr;
  uint32_t index_ = 0;
  CUgreenCtx green_context_ = nullptr;
  CUcontext cuda_context_ = nullptr;
  std::string nvtx_identifier_;
};

#endif  // HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE

}  // namespace holoscan
