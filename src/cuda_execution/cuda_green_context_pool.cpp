/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/cuda_execution/cuda_green_context_pool.hpp>

#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE

#include <holoscan/core/error_helpers.hpp>

#include <fmt/format.h>

#include <array>
#include <memory>
#include <numeric>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/logger/logger.hpp>

#include "cuda_driver_entry_points.hpp"

namespace holoscan {

// CUDA driver entry points loaded lazily via the runtime's driver-entry-point query. Keeping these
// out of the header avoids exposing the driver function-pointer surface to includers.
struct CudaDriverFunctionTable {
  CUresult (*cuDeviceGet)(CUdevice*, int) = nullptr;
  CUresult (*cuGreenCtxGetDevResource)(CUgreenCtx, CUdevResource*, CUdevResourceType) = nullptr;
  CUresult (*cuGreenCtxCreate)(CUgreenCtx*, CUdevResourceDesc, CUdevice, unsigned int) = nullptr;
  CUresult (*cuGreenCtxDestroy)(CUgreenCtx) = nullptr;
  CUresult (*cuDevSmResourceSplitByCount)(CUdevResource*, unsigned int*, const CUdevResource*,
                                          CUdevResource*, unsigned int, unsigned int) = nullptr;
  CUresult (*cuCtxFromGreenCtx)(CUcontext*, CUgreenCtx) = nullptr;
  CUresult (*cuDevResourceGenerateDesc)(CUdevResourceDesc*, CUdevResource*, unsigned int) = nullptr;
  CUresult (*cuDeviceGetDevResource)(CUdevice, CUdevResource*, CUdevResourceType) = nullptr;
  CUresult (*cuGetErrorName)(CUresult, const char**) = nullptr;
  CUresult (*cuGetErrorString)(CUresult, const char**) = nullptr;
};

namespace {

// SM-split co-scheduling can require a minimum SMs-per-partition; below these compute-capability
// thresholds we relax the constraint so finer partitions are still possible (mirrors GXF).
constexpr uint32_t kDefault7xMinSMsPerPartition = 2;
constexpr uint32_t kDefault8xMinSMsPerPartition = 4;
constexpr uint32_t kDefault9xMinSMsPerPartition = 4;
constexpr uint32_t kDefault10xMinSMsPerPartition = 8;

using error_detail::cuda_runtime_error;
using error_detail::runtime_error;
using error_detail::unexpected_runtime_error;

}  // namespace

CudaGreenContextPool::CudaGreenContextPool() = default;

CudaGreenContextPool::CudaGreenContextPool(Config config) : config_(std::move(config)) {}

CudaGreenContextPool::~CudaGreenContextPool() {
  auto result = deinitialize_pool();
  if (!result) {
    HOLOSCAN_LOG_ERROR("Failed to deinitialize CUDA green context pool: {}",
                       runtime_error_message(result.error()));
  }
}

void CudaGreenContextPool::initialize() {
  auto result = initialize_pool();
  if (!result) {
    throw RuntimeError(result.error());
  }
}

expected<void, Error> CudaGreenContextPool::load_cuda_driver() {
  if (driver_ == nullptr) {
    driver_ = std::make_unique<CudaDriverFunctionTable>();
  }

  const std::array<std::pair<void**, const char*>, 10> symbols{{
      {reinterpret_cast<void**>(&driver_->cuDeviceGet), "cuDeviceGet"},
      {reinterpret_cast<void**>(&driver_->cuGreenCtxGetDevResource), "cuGreenCtxGetDevResource"},
      {reinterpret_cast<void**>(&driver_->cuGreenCtxCreate), "cuGreenCtxCreate"},
      {reinterpret_cast<void**>(&driver_->cuGreenCtxDestroy), "cuGreenCtxDestroy"},
      {reinterpret_cast<void**>(&driver_->cuDevSmResourceSplitByCount),
       "cuDevSmResourceSplitByCount"},
      {reinterpret_cast<void**>(&driver_->cuCtxFromGreenCtx), "cuCtxFromGreenCtx"},
      {reinterpret_cast<void**>(&driver_->cuDevResourceGenerateDesc), "cuDevResourceGenerateDesc"},
      {reinterpret_cast<void**>(&driver_->cuDeviceGetDevResource), "cuDeviceGetDevResource"},
      {reinterpret_cast<void**>(&driver_->cuGetErrorName), "cuGetErrorName"},
      {reinterpret_cast<void**>(&driver_->cuGetErrorString), "cuGetErrorString"},
  }};
  for (const auto& [func, name] : symbols) {
    if (auto r = cuda_execution_detail::load_driver_entry_point(func, name); !r) {
      return r;
    }
  }
  return {};
}

namespace {

// Build an Error from a failed CUresult using the loaded driver error-string helpers.
Error driver_error(const CudaDriverFunctionTable& driver, std::string_view message,
                   CUresult result) {
  const char* name = nullptr;
  const char* str = nullptr;
  driver.cuGetErrorName(result, &name);
  driver.cuGetErrorString(result, &str);
  return runtime_error(
      ErrorCode::kFailure,
      fmt::format(
          "{}: {} - {}", message, name != nullptr ? name : "?", str != nullptr ? str : "?"));
}

}  // namespace

// Mirrors CUDA/GXF setup sequencing; keep the linear validation and resource creation flow visible.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> CudaGreenContextPool::reserve_green_contexts(
    int32_t compute_capability_major) {
  CUdevice device = 0;
  if (auto result = driver_->cuDeviceGet(&device, dev_id_); result != CUDA_SUCCESS) {
    return make_unexpected(driver_error(*driver_, "failed to get device", result));
  }

  CUdevResource device_resource = {};
  if (auto result =
          driver_->cuDeviceGetDevResource(device, &device_resource, CU_DEV_RESOURCE_TYPE_SM);
      result != CUDA_SUCCESS) {
    return make_unexpected(driver_error(*driver_, "failed to get device SM resource", result));
  }

  const uint32_t min_sm_count = config_.min_sm_count;
  if (min_sm_count == 0) {
    return unexpected_runtime_error(ErrorCode::kInvalidArgument, "min_sm_count must be > 0");
  }
  uint32_t flags = config_.green_context_flags;
  const bool relax =
      (compute_capability_major == 7 && min_sm_count < kDefault7xMinSMsPerPartition) ||
      (compute_capability_major == 8 && min_sm_count < kDefault8xMinSMsPerPartition) ||
      (compute_capability_major == 9 && min_sm_count < kDefault9xMinSMsPerPartition) ||
      (compute_capability_major >= 10 && min_sm_count < kDefault10xMinSMsPerPartition);
  if (relax) {
    flags |= CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING;
  }

  // Query how many SM groups can be created at the requested granularity.
  uint32_t nb_groups = max_sm_count_ / min_sm_count;
  CUdevResource remaining_resource = {};
  std::vector<CUdevResource> split_result(nb_groups);
  HOLOSCAN_LOG_INFO(
      "CudaGreenContextPool splitting SMs: max_sm_count={}, min_sm_count={}, expected_groups={}, "
      "flags={}",
      max_sm_count_,
      min_sm_count,
      nb_groups,
      flags);
  if (auto result = driver_->cuDevSmResourceSplitByCount(split_result.data(),
                                                         &nb_groups,
                                                         &device_resource,
                                                         &remaining_resource,
                                                         flags,
                                                         min_sm_count);
      result != CUDA_SUCCESS) {
    return make_unexpected(
        driver_error(*driver_, "failed to split SMs for green contexts", result));
  }
  if (nb_groups < 1 || nb_groups != max_sm_count_ / min_sm_count) {
    return unexpected_runtime_error(ErrorCode::kFailure,
                                    fmt::format("unexpected SM split group count {}", nb_groups));
  }

  green_contexts_.assign(resolved_num_partitions_, nullptr);
  cuda_contexts_.assign(resolved_num_partitions_, nullptr);

  uint32_t split_idx = 0;
  for (uint32_t idx = 0; idx < resolved_num_partitions_; ++idx) {
    const auto requested_sms = static_cast<uint32_t>(resolved_sms_per_partition_[idx]);
    if (requested_sms < min_sm_count || requested_sms % min_sm_count != 0) {
      return unexpected_runtime_error(
          ErrorCode::kInvalidArgument,
          fmt::format("partition {} SMs ({}) must be a positive multiple of min_sm_count ({})",
                      idx,
                      requested_sms,
                      min_sm_count));
    }
    const uint32_t num_resources = requested_sms / min_sm_count;

    CUdevResourceDesc out_desc{};
    if (auto result =
            driver_->cuDevResourceGenerateDesc(&out_desc, &split_result[split_idx], num_resources);
        result != CUDA_SUCCESS) {
      return make_unexpected(driver_error(*driver_, "cuDevResourceGenerateDesc failed", result));
    }
    split_idx += num_resources;

    if (auto result = driver_->cuGreenCtxCreate(
            &green_contexts_[idx], out_desc, device, CU_GREEN_CTX_DEFAULT_STREAM);
        result != CUDA_SUCCESS) {
      return make_unexpected(driver_error(*driver_, "cuGreenCtxCreate failed", result));
    }

    CUdevResource green_ctx_resource{};
    if (auto result = driver_->cuGreenCtxGetDevResource(
            green_contexts_[idx], &green_ctx_resource, CU_DEV_RESOURCE_TYPE_SM);
        result != CUDA_SUCCESS) {
      return make_unexpected(driver_error(*driver_, "cuGreenCtxGetDevResource failed", result));
    }
    if (green_ctx_resource.sm.smCount != requested_sms) {
      return unexpected_runtime_error(ErrorCode::kFailure,
                                      fmt::format("green context SM count {} != requested {}",
                                                  green_ctx_resource.sm.smCount,
                                                  requested_sms));
    }

    if (auto result = driver_->cuCtxFromGreenCtx(&cuda_contexts_[idx], green_contexts_[idx]);
        result != CUDA_SUCCESS) {
      return make_unexpected(driver_error(*driver_, "cuCtxFromGreenCtx failed", result));
    }
    if (cuda_contexts_[idx] == nullptr) {
      return unexpected_runtime_error(ErrorCode::kFailure,
                                      fmt::format("null CUcontext for partition {}", idx));
    }
  }

  // Resolve the default partition index.
  if (config_.default_context && *config_.default_context >= 0) {
    default_index_ = static_cast<uint32_t>(*config_.default_context);
  } else {
    default_index_ = resolved_num_partitions_ - 1;
  }
  if (default_index_ >= resolved_num_partitions_) {
    return unexpected_runtime_error(
        ErrorCode::kInvalidArgument,
        fmt::format("default_context index {} is out of range ({} partitions)",
                    default_index_,
                    resolved_num_partitions_));
  }
  return {};
}

expected<void, Error> CudaGreenContextPool::initialize_pool() {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (stage_ == AllocatorStage::kInitialized) {
    return {};
  }
  stage_ = AllocatorStage::kInitializationInProgress;

  auto fail = [&](auto&& err) {
    stage_ = AllocatorStage::kUninitialized;
    return err;
  };

  if (auto loaded = load_cuda_driver(); !loaded) {
    return fail(forward_error(loaded));
  }

  if (auto status = cudaSetDevice(config_.device_id); status != cudaSuccess) {
    return fail(make_unexpected(cuda_runtime_error("failed to set device", status)));
  }
  dev_id_ = config_.device_id;

  cudaDeviceProp dev_prop{};
  if (auto status = cudaGetDeviceProperties(&dev_prop, dev_id_); status != cudaSuccess) {
    return fail(make_unexpected(cuda_runtime_error("failed to get device properties", status)));
  }
  max_sm_count_ = static_cast<uint32_t>(dev_prop.multiProcessorCount);

  // Resolve the requested partitioning (mirrors nvidia::gxf::CudaGreenContextPool::initialize).
  resolved_num_partitions_ = config_.num_partitions;
  resolved_sms_per_partition_ = config_.sms_per_partition;
  if (resolved_num_partitions_ == 0) {
    resolved_sms_per_partition_ = {static_cast<int32_t>(max_sm_count_)};
    resolved_num_partitions_ = 1;
  }
  if (resolved_sms_per_partition_.size() < resolved_num_partitions_) {
    return fail(
        unexpected_runtime_error(ErrorCode::kInvalidArgument,
                                 "sms_per_partition must have at least num_partitions entries"));
  }

  int64_t total_sms = 0;
  for (uint32_t i = 0; i < resolved_num_partitions_; ++i) {
    total_sms += resolved_sms_per_partition_[i];
  }
  if (total_sms > static_cast<int64_t>(max_sm_count_)) {
    return fail(unexpected_runtime_error(
        ErrorCode::kInvalidArgument,
        fmt::format(
            "total requested SMs ({}) exceeds device SM count ({})", total_sms, max_sm_count_)));
  }
  if (total_sms < static_cast<int64_t>(max_sm_count_)) {
    // Add a partition to cover the remaining SMs so the whole device is represented.
    resolved_sms_per_partition_.resize(resolved_num_partitions_);
    resolved_sms_per_partition_.push_back(static_cast<int32_t>(max_sm_count_ - total_sms));
    ++resolved_num_partitions_;
  }

  if (auto reserved = reserve_green_contexts(dev_prop.major); !reserved) {
    auto cleanup = destroy_green_contexts();
    if (!cleanup) {
      HOLOSCAN_LOG_ERROR("Failed to clean up CUDA green contexts after reserve failure: {}",
                         runtime_error_message(cleanup.error()));
    }
    return fail(forward_error(reserved));
  }

  stage_ = AllocatorStage::kInitialized;
  return {};
}

expected<void, Error> CudaGreenContextPool::destroy_green_contexts() {
  expected<void, Error> result;
  if (driver_ != nullptr && driver_->cuGreenCtxDestroy != nullptr) {
    for (CUgreenCtx ctx : green_contexts_) {
      if (ctx != nullptr) {
        if (auto status = driver_->cuGreenCtxDestroy(ctx); status != CUDA_SUCCESS && result) {
          result = make_unexpected(driver_error(*driver_, "cuGreenCtxDestroy failed", status));
        }
      }
    }
  }
  green_contexts_.clear();
  cuda_contexts_.clear();
  return result;
}

expected<void, Error> CudaGreenContextPool::deinitialize_pool() {
  const std::lock_guard<std::mutex> lock(mutex_);
  stage_ = AllocatorStage::kDeinitializationInProgress;
  auto result = destroy_green_contexts();
  resolved_sms_per_partition_.clear();
  resolved_num_partitions_ = 0;
  max_sm_count_ = 0;
  dev_id_ = -1;
  default_index_ = 0;
  stage_ = AllocatorStage::kUninitialized;
  return result;
}

expected<CUgreenCtx, Error> CudaGreenContextPool::green_context(uint32_t index) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (stage_ != AllocatorStage::kInitialized) {
    return unexpected_runtime_error(ErrorCode::kFailure, "pool is not initialized");
  }
  if (index >= green_contexts_.size()) {
    return unexpected_runtime_error(ErrorCode::kInvalidArgument,
                                    fmt::format("invalid partition index {}", index));
  }
  return green_contexts_[index];
}

expected<CUcontext, Error> CudaGreenContextPool::cuda_context(uint32_t index) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (stage_ != AllocatorStage::kInitialized) {
    return unexpected_runtime_error(ErrorCode::kFailure, "pool is not initialized");
  }
  if (index >= cuda_contexts_.size()) {
    return unexpected_runtime_error(ErrorCode::kInvalidArgument,
                                    fmt::format("invalid partition index {}", index));
  }
  return cuda_contexts_[index];
}

expected<uint32_t, Error> CudaGreenContextPool::partition_sms(uint32_t index) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (stage_ != AllocatorStage::kInitialized) {
    return unexpected_runtime_error(ErrorCode::kFailure, "pool is not initialized");
  }
  if (index >= resolved_sms_per_partition_.size()) {
    return unexpected_runtime_error(ErrorCode::kInvalidArgument,
                                    fmt::format("invalid partition index {}", index));
  }
  return static_cast<uint32_t>(resolved_sms_per_partition_[index]);
}

expected<uint32_t, Error> CudaGreenContextPool::default_context_index() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (stage_ != AllocatorStage::kInitialized) {
    return unexpected_runtime_error(ErrorCode::kFailure, "pool is not initialized");
  }
  return default_index_;
}

expected<CUgreenCtx, Error> CudaGreenContextPool::default_context() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (stage_ != AllocatorStage::kInitialized) {
    return unexpected_runtime_error(ErrorCode::kFailure, "pool is not initialized");
  }
  if (default_index_ >= green_contexts_.size()) {
    return unexpected_runtime_error(ErrorCode::kInvalidArgument, "invalid default context index");
  }
  return green_contexts_[default_index_];
}

expected<uint32_t, Error> CudaGreenContextPool::device_total_sms() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (stage_ != AllocatorStage::kInitialized) {
    return unexpected_runtime_error(ErrorCode::kFailure, "pool is not initialized");
  }
  return max_sm_count_;
}

size_t CudaGreenContextPool::num_partitions() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return green_contexts_.size();
}

}  // namespace holoscan

#endif  // HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
