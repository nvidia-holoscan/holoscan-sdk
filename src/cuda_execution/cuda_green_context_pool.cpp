/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/cuda_execution/cuda_green_context_pool.hpp>

#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE

#include <holoscan/core/error_helpers.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <memory>
#include <numeric>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/logger/logger.hpp>

#include "cuda_green_context_driver.hpp"

namespace holoscan {

namespace {

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
  auto driver = cuda_execution_detail::cuda_green_context_driver();
  if (!driver) {
    return forward_error(driver);
  }
  driver_ = std::move(*driver);
  return {};
}

namespace {

// Build an Error from a failed CUresult using the loaded driver error-string helpers.
Error driver_error(const cuda_execution_detail::CudaGreenContextDriver& driver,
                   std::string_view message, CUresult result) {
  const char* name = nullptr;
  const char* str = nullptr;
  (void)driver.get_error_name(result, &name);
  (void)driver.get_error_string(result, &str);
  return runtime_error(
      ErrorCode::kFailure,
      fmt::format(
          "{}: {} - {}", message, name != nullptr ? name : "?", str != nullptr ? str : "?"));
}

}  // namespace

// Keep the linear validation and resource creation flow visible.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> CudaGreenContextPool::reserve_green_contexts() {
  CUdevice device = 0;
  if (auto result = driver_->device_get(&device, dev_id_); result != CUDA_SUCCESS) {
    return make_unexpected(driver_error(*driver_, "failed to get device", result));
  }

  CUdevResource device_resource = {};
  if (auto result = driver_->device_get_resource(device, &device_resource, CU_DEV_RESOURCE_TYPE_SM);
      result != CUDA_SUCCESS) {
    return make_unexpected(driver_error(*driver_, "failed to get device SM resource", result));
  }
  if (device_resource.type != CU_DEV_RESOURCE_TYPE_SM ||
      device_resource.sm.smCount != max_sm_count_ ||
      device_resource.sm.minSmPartitionSize != min_sm_partition_size_ ||
      device_resource.sm.smCoscheduledAlignment != sm_coscheduled_alignment_) {
    return unexpected_runtime_error(ErrorCode::kFailure,
                                    "CUDA Green Context capability changed during initialization");
  }

  CUdevResource weak_remainder{};
  std::vector<CUdevResource> split_result;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> group_resource_ranges;
  HOLOSCAN_LOG_INFO(
      "CudaGreenContextPool splitting SMs: max_sm_count={}, min_partition_size={}, "
      "coscheduled_alignment={}, structured_groups={}, split_flags=0",
      max_sm_count_,
      min_sm_partition_size_,
      sm_coscheduled_alignment_,
      resolved_num_partitions_);
  const bool whole_device_group =
      resolved_num_partitions_ == 1U &&
      resolved_sms_per_partition_.front() == static_cast<std::int32_t>(max_sm_count_);
  if (whole_device_group) {
    split_result.push_back(device_resource);
    group_resource_ranges.emplace_back(0U, 1U);
  } else if (driver_->sm_resource_split != nullptr) {
    // CUDA 13.1+ provides the newer, more expressive cuDevSmResourceSplit.
    std::vector<cuda_execution_detail::CudaGreenContextDriver::SmResourceGroupParams> group_params(
        resolved_num_partitions_);
    for (std::uint32_t index = 0U; index < resolved_num_partitions_; ++index) {
      group_params[index].sm_count = static_cast<std::uint32_t>(resolved_sms_per_partition_[index]);
      group_params[index].coscheduled_sm_count = sm_coscheduled_alignment_;
    }
    split_result.resize(resolved_num_partitions_);
    if (auto result = driver_->sm_resource_split(split_result.data(),
                                                 resolved_num_partitions_,
                                                 &device_resource,
                                                 &weak_remainder,
                                                 0U,
                                                 group_params.data());
        result != CUDA_SUCCESS) {
      return make_unexpected(
          driver_error(*driver_, "failed to split SMs for green contexts", result));
    }
    for (std::uint32_t index = 0U; index < resolved_num_partitions_; ++index) {
      group_resource_ranges.emplace_back(index, 1U);
    }
  } else {
    // CUDA 13.0 can form the same heterogeneous groups by producing minimum-size structured
    // resources in one split and aggregating adjacent outputs into each context descriptor.
    unsigned int available_units = 0U;
    if (auto result = driver_->sm_resource_split_by_count(
            nullptr, &available_units, &device_resource, nullptr, 0U, structured_sm_quantum_);
        result != CUDA_SUCCESS) {
      return make_unexpected(driver_error(*driver_, "failed to query structured SM split", result));
    }
    const auto required_units = static_cast<std::uint32_t>(
        std::accumulate(resolved_sms_per_partition_.begin(),
                        resolved_sms_per_partition_.begin() + resolved_num_partitions_,
                        std::int64_t{0}) /
        structured_sm_quantum_);
    if (available_units < required_units) {
      return unexpected_runtime_error(
          ErrorCode::kFailure,
          fmt::format("CUDA driver can form only {} structured units ({} SMs at quantum {}); "
                      "{} units ({} SMs) are required",
                      available_units,
                      static_cast<std::uint64_t>(available_units) * structured_sm_quantum_,
                      structured_sm_quantum_,
                      required_units,
                      static_cast<std::uint64_t>(required_units) * structured_sm_quantum_));
    }
    // Request every available structured unit in one cold split. The complete array lets the
    // validated group ranges consume only their exact prefixes and leaves the tail unprovisioned.
    split_result.resize(available_units);
    unsigned int created_units = available_units;
    if (auto result = driver_->sm_resource_split_by_count(split_result.data(),
                                                          &created_units,
                                                          &device_resource,
                                                          &weak_remainder,
                                                          0U,
                                                          structured_sm_quantum_);
        result != CUDA_SUCCESS) {
      return make_unexpected(driver_error(*driver_, "failed to split structured SM units", result));
    }
    split_result.resize(created_units);
    if (created_units < required_units) {
      return unexpected_runtime_error(
          ErrorCode::kFailure,
          fmt::format("CUDA driver created only {} structured units ({} SMs at quantum {}); "
                      "{} units ({} SMs) are required",
                      created_units,
                      static_cast<std::uint64_t>(created_units) * structured_sm_quantum_,
                      structured_sm_quantum_,
                      required_units,
                      static_cast<std::uint64_t>(required_units) * structured_sm_quantum_));
    }
    std::uint32_t unit_offset = 0U;
    for (std::uint32_t index = 0U; index < resolved_num_partitions_; ++index) {
      const auto unit_count =
          static_cast<std::uint32_t>(resolved_sms_per_partition_[index]) / structured_sm_quantum_;
      group_resource_ranges.emplace_back(unit_offset, unit_count);
      unit_offset += unit_count;
    }
  }

  for (std::uint32_t index = 0U; index < resolved_num_partitions_; ++index) {
    const auto requested_sms = static_cast<std::uint32_t>(resolved_sms_per_partition_[index]);
    const auto [resource_offset, resource_count] = group_resource_ranges[index];
    std::uint32_t returned_sms = 0U;
    for (std::uint32_t resource_index = 0U; resource_index < resource_count; ++resource_index) {
      const auto& resource = split_result[resource_offset + resource_index];
      if (resource.type != CU_DEV_RESOURCE_TYPE_SM) {
        return unexpected_runtime_error(ErrorCode::kFailure,
                                        fmt::format("group {} returned a non-SM resource", index));
      }
      returned_sms += resource.sm.smCount;
    }
    if (returned_sms != requested_sms) {
      return unexpected_runtime_error(
          ErrorCode::kFailure,
          fmt::format("structured group {} returned {} SMs instead of requested {}",
                      index,
                      returned_sms,
                      requested_sms));
    }
  }

  green_contexts_.assign(resolved_num_partitions_, nullptr);
  cuda_contexts_.assign(resolved_num_partitions_, nullptr);

  for (uint32_t idx = 0; idx < resolved_num_partitions_; ++idx) {
    const auto requested_sms = static_cast<uint32_t>(resolved_sms_per_partition_[idx]);
    const auto [resource_offset, resource_count] = group_resource_ranges[idx];
    CUdevResourceDesc out_desc{};
    if (auto result = driver_->resource_generate_desc(
            &out_desc, split_result.data() + resource_offset, resource_count);
        result != CUDA_SUCCESS) {
      return make_unexpected(driver_error(*driver_, "cuDevResourceGenerateDesc failed", result));
    }

    if (auto result = driver_->green_context_create(
            &green_contexts_[idx], out_desc, device, CU_GREEN_CTX_DEFAULT_STREAM);
        result != CUDA_SUCCESS) {
      return make_unexpected(driver_error(*driver_, "cuGreenCtxCreate failed", result));
    }

    CUdevResource green_ctx_resource{};
    if (auto result = driver_->green_context_get_resource(
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

    if (auto result =
            driver_->context_from_green_context(&cuda_contexts_[idx], green_contexts_[idx]);
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
  dev_id_ = config_.device_id;
  auto capability =
      cuda_execution_detail::query_cuda_green_context_device_capability(*driver_, dev_id_);
  if (!capability) {
    return fail(forward_error(capability));
  }
  max_sm_count_ = capability->device_sm_count;
  min_sm_partition_size_ = capability->min_sm_partition_size;
  sm_coscheduled_alignment_ = capability->sm_coscheduled_alignment;
  structured_sm_quantum_ = capability->structured_sm_quantum;
  max_structured_context_count_ = capability->max_structured_context_count;

  // Resolve the caller's exact structured groups. A legal leftover becomes one final structured
  // group; an ineligible tail remains deliberately unprovisioned and is never used for HSDK work.
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
    const std::int32_t requested = resolved_sms_per_partition_[i];
    const auto requested_sms = static_cast<std::uint32_t>(requested);
    const bool whole_device_request =
        resolved_num_partitions_ == 1U && requested_sms == max_sm_count_;
    if (requested <= 0 ||
        (!whole_device_request && (requested_sms < min_sm_partition_size_ ||
                                   requested_sms % structured_sm_quantum_ != 0U))) {
      return fail(unexpected_runtime_error(
          ErrorCode::kInvalidArgument,
          fmt::format("partition {} SM count {} must be at least {} and a multiple of {}",
                      i,
                      requested,
                      min_sm_partition_size_,
                      structured_sm_quantum_)));
    }
    total_sms += resolved_sms_per_partition_[i];
  }
  const std::uint64_t structured_sm_capacity =
      static_cast<std::uint64_t>(max_structured_context_count_) * structured_sm_quantum_;
  const bool whole_device_request =
      resolved_num_partitions_ == 1U && total_sms == static_cast<std::int64_t>(max_sm_count_);
  const std::uint64_t usable_sm_capacity =
      whole_device_request ? max_sm_count_ : structured_sm_capacity;
  if (total_sms < 0 || static_cast<std::uint64_t>(total_sms) > usable_sm_capacity) {
    return fail(unexpected_runtime_error(
        ErrorCode::kInvalidArgument,
        fmt::format("total requested SMs ({}) exceeds usable structured SM count ({})",
                    total_sms,
                    usable_sm_capacity)));
  }
  if (!whole_device_request && static_cast<std::uint64_t>(total_sms) < structured_sm_capacity) {
    const std::uint32_t remaining = static_cast<std::uint32_t>(structured_sm_capacity - total_sms);
    if (remaining >= min_sm_partition_size_ && remaining % structured_sm_quantum_ == 0U) {
      resolved_sms_per_partition_.resize(resolved_num_partitions_);
      resolved_sms_per_partition_.push_back(static_cast<int32_t>(remaining));
      ++resolved_num_partitions_;
    }
  }
  if (resolved_num_partitions_ > max_structured_context_count_) {
    return fail(unexpected_runtime_error(
        ErrorCode::kInvalidArgument,
        fmt::format("requested {} structured groups exceeds device maximum {}",
                    resolved_num_partitions_,
                    max_structured_context_count_)));
  }

  if (auto reserved = reserve_green_contexts(); !reserved) {
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
  if (driver_ != nullptr && driver_->green_context_destroy != nullptr) {
    for (auto context = green_contexts_.rbegin(); context != green_contexts_.rend(); ++context) {
      const CUgreenCtx ctx = *context;
      if (ctx != nullptr) {
        if (auto status = driver_->green_context_destroy(ctx); status != CUDA_SUCCESS && result) {
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
  min_sm_partition_size_ = 0;
  sm_coscheduled_alignment_ = 0;
  structured_sm_quantum_ = 0;
  max_structured_context_count_ = 0;
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

expected<uint32_t, Error> CudaGreenContextPool::min_sm_partition_size() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (stage_ != AllocatorStage::kInitialized) {
    return unexpected_runtime_error(ErrorCode::kFailure, "pool is not initialized");
  }
  return min_sm_partition_size_;
}

expected<uint32_t, Error> CudaGreenContextPool::sm_coscheduled_alignment() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (stage_ != AllocatorStage::kInitialized) {
    return unexpected_runtime_error(ErrorCode::kFailure, "pool is not initialized");
  }
  return sm_coscheduled_alignment_;
}

expected<uint32_t, Error> CudaGreenContextPool::max_structured_context_count() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (stage_ != AllocatorStage::kInitialized) {
    return unexpected_runtime_error(ErrorCode::kFailure, "pool is not initialized");
  }
  return max_structured_context_count_;
}

size_t CudaGreenContextPool::num_partitions() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return green_contexts_.size();
}

}  // namespace holoscan

#endif  // HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
