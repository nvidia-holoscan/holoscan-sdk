/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#ifndef HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12050)
#define HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE 1
#else
#define HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE 0
#endif
#endif

namespace holoscan::cuda_execution_detail {

#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE

struct CudaGreenContextDriver {
  struct SmResourceGroupParams {
    unsigned int sm_count;
    unsigned int coscheduled_sm_count;
    unsigned int preferred_coscheduled_sm_count;
    unsigned int flags;
    // Mirror the CUDA driver ABI exactly; std::array is not part of that C contract.
    unsigned int reserved[12];  // NOLINT(cppcoreguidelines-avoid-c-arrays)
  };

  CUresult (*device_get)(CUdevice*, int) = nullptr;
  CUresult (*device_get_uuid)(CUuuid*, CUdevice) = nullptr;
  CUresult (*device_get_attribute)(int*, CUdevice_attribute, CUdevice) = nullptr;
  CUresult (*device_get_resource)(CUdevice, CUdevResource*, CUdevResourceType) = nullptr;
  CUresult (*sm_resource_split_by_count)(CUdevResource*, unsigned int*, const CUdevResource*,
                                         CUdevResource*, unsigned int, unsigned int) = nullptr;
  CUresult (*sm_resource_split)(CUdevResource*, unsigned int, const CUdevResource*, CUdevResource*,
                                unsigned int, SmResourceGroupParams*) = nullptr;
  CUresult (*resource_generate_desc)(CUdevResourceDesc*, CUdevResource*, unsigned int) = nullptr;
  CUresult (*green_context_create)(CUgreenCtx*, CUdevResourceDesc, CUdevice,
                                   unsigned int) = nullptr;
  CUresult (*green_context_destroy)(CUgreenCtx) = nullptr;
  CUresult (*green_context_get_resource)(CUgreenCtx, CUdevResource*, CUdevResourceType) = nullptr;
  CUresult (*context_from_green_context)(CUcontext*, CUgreenCtx) = nullptr;
  CUresult (*green_context_stream_create)(CUstream*, CUgreenCtx, unsigned int, int) = nullptr;
  CUresult (*stream_synchronize)(CUstream) = nullptr;
  CUresult (*stream_query)(CUstream) = nullptr;
  CUresult (*stream_destroy)(CUstream) = nullptr;
  CUresult (*stream_get_green_context)(CUstream, CUgreenCtx*) = nullptr;
  CUresult (*get_error_name)(CUresult, const char**) = nullptr;
  CUresult (*get_error_string)(CUresult, const char**) = nullptr;
};

struct CudaGreenContextDeviceCapability {
  std::int32_t device_id{-1};
  std::string deployment_device_id;
  std::uint32_t device_sm_count{};
  std::uint32_t min_sm_partition_size{};
  std::uint32_t sm_coscheduled_alignment{};
  std::uint32_t structured_sm_quantum{};
  std::uint32_t max_structured_context_count{};
};

enum class CudaGreenContextStreamIdentityKind : std::uint8_t {
  kNullStream = 0,
  kOrdinaryContext = 1,
  kExpectedGreenContext = 2,
  kWrongGreenContext = 3,
  kUnsupported = 4,
};

struct CudaGreenContextStreamIdentity {
  CudaGreenContextStreamIdentityKind kind{CudaGreenContextStreamIdentityKind::kNullStream};
  CUgreenCtx observed_context{};
};

[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<std::shared_ptr<const CudaGreenContextDriver>, Error>
cuda_green_context_driver();

[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<CudaGreenContextDeviceCapability, Error>
query_cuda_green_context_device_capability(const CudaGreenContextDriver& driver,
                                           std::int32_t device_id);

[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<CudaGreenContextStreamIdentity, Error>
query_cuda_green_context_stream_identity(const CudaGreenContextDriver& driver, CUstream stream,
                                         CUgreenCtx expected_context);

[[nodiscard]] HOLOSCAN_CORE_EXPORT std::shared_ptr<const CudaGreenContextDriver>
set_cuda_green_context_driver_for_testing(
    std::shared_ptr<const CudaGreenContextDriver> driver) noexcept;

#endif  // HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE

}  // namespace holoscan::cuda_execution_detail
