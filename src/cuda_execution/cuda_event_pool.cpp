/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/cuda_execution/cuda_event_pool.hpp>

#include <holoscan/core/error_helpers.hpp>

#include <cuda/devices>

#include <fmt/format.h>

#include <algorithm>
#include <exception>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/logger/logger.hpp>

#include "core/diagnostic_reporting.hpp"
#include "cuda_driver_entry_points.hpp"

namespace holoscan {

namespace {

Error cuda_pool_error(std::string_view message, const std::exception& error) {
  return Error{ErrorCode::kFailure, fmt::format("{}: {}", message, error.what())};
}

using error_detail::unexpected_cuda_runtime_error;
using error_detail::unexpected_runtime_error;

unexpected<Error> unexpected_invalid_lease_error() {
  return unexpected_runtime_error(ErrorCode::kInvalidArgument, "CUDA event lease is invalid");
}

#if HOLOSCAN_CUDA_DRIVER_ENTRY_POINTS_AVAILABLE
using cuda_execution_detail::ctx_pop_current_fn;
using cuda_execution_detail::ctx_push_current_fn;
#endif

}  // namespace

CudaEventPool::EventLease::EventLease(CudaEventPool* pool, std::shared_ptr<EventEntry> entry,
                                      std::weak_ptr<bool> pool_alive)
    : pool_(pool), entry_(std::move(entry)), pool_alive_(std::move(pool_alive)) {}

CudaEventPool::EventLease::EventLease(EventLease&& other) noexcept {
  *this = std::move(other);
}

CudaEventPool::EventLease& CudaEventPool::EventLease::operator=(
    CudaEventPool::EventLease&& other) noexcept {
  if (this != &other) {
    (void)release();
    pool_ = other.pool_;
    entry_ = std::move(other.entry_);
    pool_alive_ = std::move(other.pool_alive_);
    other.pool_ = nullptr;
    other.pool_alive_.reset();
  }
  return *this;
}

CudaEventPool::EventLease::~EventLease() {
  auto result = release();
  if (!result) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR("Failed to release CUDA event lease: {}",
                                 runtime_error_message(result.error()));
  }
}

expected<void, Error> CudaEventPool::EventLease::release() {
  if (entry_ == nullptr) {
    return {};
  }
  if (pool_alive_.expired()) {
    entry_.reset();
    pool_ = nullptr;
    pool_alive_.reset();
    return {};
  }
  if (pool_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kFailure, "CUDA event lease has no owning pool"});
  }

  auto entry = std::move(entry_);
  auto* pool = pool_;
  pool_ = nullptr;
  pool_alive_.reset();
  return pool->release(std::move(entry));
}

cuda::event_ref CudaEventPool::EventLease::event() const {
  cudaEvent_t native_event = native_handle();
  if (native_event == nullptr) {
    return cuda::event_ref(cudaEvent_t{});
  }
  return cuda::event_ref(native_event);
}

cudaEvent_t CudaEventPool::EventLease::native_handle() const {
  if (entry_ == nullptr || entry_->event.get() == nullptr) {
    return nullptr;
  }
  return entry_->event.get();
}

expected<void, Error> CudaEventPool::EventLease::record(::cuda::stream_ref stream) const {
  if (!entry_) {
    return unexpected_invalid_lease_error();
  }
  try {
    event().record(stream);
    return {};
  } catch (const std::exception& e) {
    return make_unexpected(cuda_pool_error("Failed to record CUDA event", e));
  }
}

expected<void, Error> CudaEventPool::EventLease::wait(::cuda::stream_ref stream) const {
  if (!entry_) {
    return unexpected_invalid_lease_error();
  }
  try {
    stream.wait(event());
    return {};
  } catch (const std::exception& e) {
    return make_unexpected(cuda_pool_error("Failed to make CUDA stream wait on event", e));
  }
}

expected<void, Error> CudaEventPool::EventLease::synchronize() const {
  if (!entry_) {
    return unexpected_invalid_lease_error();
  }
  try {
    event().sync();
    return {};
  } catch (const std::exception& e) {
    return make_unexpected(cuda_pool_error("Failed to synchronize CUDA event", e));
  }
}

expected<bool, Error> CudaEventPool::EventLease::is_done() const {
  if (!entry_) {
    return unexpected_invalid_lease_error();
  }
  try {
    return event().is_done();
  } catch (const std::exception& e) {
    return make_unexpected<Error>(cuda_pool_error("Failed to query CUDA event", e));
  }
}

CudaEventPool::CudaEventPool(Config config) : config_(std::move(config)) {}

CudaEventPool::CudaEventPool(int32_t device_id, cuda::event_flags event_flags,
                             uint32_t reserved_size, uint32_t max_size, std::string nvtx_identifier)
    : config_{device_id, event_flags, reserved_size, max_size, std::move(nvtx_identifier)} {}

CudaEventPool::~CudaEventPool() {
  auto result = deinitialize_pool();
  if (!result) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR("Failed to deinitialize CUDA event pool: {}",
                                 runtime_error_message(result.error()));
  }
}

void CudaEventPool::initialize() {
  auto result = initialize_pool();
  if (!result) {
    throw RuntimeError(result.error());
  }
}

expected<void, Error> CudaEventPool::initialize_pool() {
  auto result = reserve_events();
  if (!result) {
    stage_ = AllocatorStage::kUninitialized;
    return forward_error(result);
  }
  return {};
}

expected<void, Error> CudaEventPool::deinitialize_pool() {
  const std::lock_guard<std::mutex> lock(mutex_);
  stage_ = AllocatorStage::kDeinitializationInProgress;

  if (!allocated_events_.empty()) {
    HOLOSCAN_LOG_AUTOMATIC_WARN("CudaEventPool still has leased events");
  }
  alive_token_.reset();
  allocated_events_.clear();
  reserved_events_.clear();
  next_event_index_ = 0;
  stage_ = AllocatorStage::kUninitialized;
  return {};
}

expected<CudaEventPool::EventLease, Error> CudaEventPool::acquire() {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (stage_ != AllocatorStage::kInitialized) {
    return make_unexpected(Error{ErrorCode::kFailure, "CudaEventPool is not initialized"});
  }

  if (config_.max_size != 0 && allocated_events_.size() >= config_.max_size) {
    return make_unexpected(Error{ErrorCode::kFailure, "CudaEventPool has reached capacity"});
  }

  std::shared_ptr<EventEntry> entry;
  if (!reserved_events_.empty()) {
    entry = std::move(reserved_events_.back());
    reserved_events_.pop_back();
  } else {
    auto event_name = fmt::format("{}:{}", config_.nvtx_identifier, next_event_index_++);
    auto maybe_entry = create_event_entry(std::move(event_name));
    if (!maybe_entry) {
      return forward_error(maybe_entry);
    }
    entry = std::move(maybe_entry.value());
  }

  allocated_events_.push_back(entry.get());
  return EventLease(this, std::move(entry), alive_token_);
}

expected<void, Error> CudaEventPool::release(std::shared_ptr<EventEntry> entry) {
  if (entry == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "Cannot release a null CUDA event entry"});
  }

  const std::lock_guard<std::mutex> lock(mutex_);
  const auto iter = std::find(allocated_events_.begin(), allocated_events_.end(), entry.get());
  if (iter == allocated_events_.end()) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "CUDA event entry was not allocated by pool"});
  }

  *iter = allocated_events_.back();
  allocated_events_.pop_back();
  reserved_events_.push_back(std::move(entry));
  return {};
}

size_t CudaEventPool::available() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return reserved_events_.size();
}

size_t CudaEventPool::allocated() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return allocated_events_.size();
}

expected<std::shared_ptr<CudaEventPool::EventEntry>, Error> CudaEventPool::create_event_entry(
    std::string name) {
  try {
    int device_count = 0;
    const cudaError_t get_device_count_result = cudaGetDeviceCount(&device_count);
    if (get_device_count_result != cudaSuccess) {
      return unexpected_cuda_runtime_error("Failed to query CUDA device count",
                                           get_device_count_result);
    }
    if (config_.device_id < 0 || config_.device_id >= device_count) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "Invalid CUDA event pool device id"});
    }

    if (config_.cuda_context != nullptr) {
#if HOLOSCAN_CUDA_DRIVER_ENTRY_POINTS_AVAILABLE
      const auto& push_current = ctx_push_current_fn();
      const auto& pop_current = ctx_pop_current_fn();
      if (!push_current) {
        return forward_error(push_current);
      }
      if (!pop_current) {
        return forward_error(pop_current);
      }

      const CUresult push_result = push_current.value()(config_.cuda_context);
      if (push_result != CUDA_SUCCESS) {
        return unexpected_runtime_error(
            ErrorCode::kFailure,
            fmt::format("Failed to push provided CUDA context (CUresult {})",
                        static_cast<int>(push_result)));
      }

      // Verify the provided context is on the configured device while it is current.
      auto device_check = cuda_execution_detail::verify_current_context_device(config_.device_id);

      // Untimed events disable timing, matching cuda::event's device constructor.
      const unsigned int flags =
          static_cast<unsigned int>(config_.event_flags) | cudaEventDisableTiming;
      cudaEvent_t native_event = nullptr;
      cudaError_t create_result = cudaSuccess;
      if (device_check) {
        create_result = cudaEventCreateWithFlags(&native_event, flags);
      }

      // Always pop to restore the caller's context stack, even if the check/creation failed.
      CUcontext previous_context = nullptr;
      const CUresult pop_result = pop_current.value()(&previous_context);
      (void)previous_context;

      if (pop_result != CUDA_SUCCESS) {
        if (native_event != nullptr) {
          (void)cudaEventDestroy(native_event);
        }
        return unexpected_runtime_error(
            ErrorCode::kFailure,
            fmt::format("Failed to restore CUDA context after event creation (CUresult {})",
                        static_cast<int>(pop_result)));
      }
      if (!device_check) {
        return forward_error(device_check);
      }
      if (create_result != cudaSuccess) {
        return unexpected_cuda_runtime_error("Failed to create CUDA event in provided context",
                                             create_result);
      }
      auto event = cuda::event::from_native_handle(native_event);
      return std::make_shared<EventEntry>(std::move(event), name);
#else
      return unexpected_runtime_error(ErrorCode::kFailure,
                                      "Config::cuda_context requires CUDA 12.3 or newer");
#endif
    }

    auto event = cuda::event(cuda::device_ref(config_.device_id), config_.event_flags);
    return std::make_shared<EventEntry>(std::move(event), std::move(name));
  } catch (const std::exception& e) {
    return make_unexpected<Error>(cuda_pool_error("Failed to create CUDA event", e));
  }
}

expected<void, Error> CudaEventPool::reserve_events() {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (stage_ == AllocatorStage::kInitialized) {
    return {};
  }
  stage_ = AllocatorStage::kInitializationInProgress;
  if (!alive_token_) {
    alive_token_ = std::make_shared<bool>(true);
  }

  if (config_.max_size != 0 && config_.max_size < config_.reserved_size) {
    stage_ = AllocatorStage::kUninitialized;
    return unexpected_runtime_error(ErrorCode::kInvalidArgument,
                                    "CudaEventPool max_size must be zero or at least "
                                    "reserved_size");
  }

  const auto bounded_capacity = config_.max_size == 0 ? config_.reserved_size : config_.max_size;
  reserved_events_.reserve(bounded_capacity);
  allocated_events_.reserve(bounded_capacity);
  for (uint32_t i = 0; i < config_.reserved_size; ++i) {
    auto event_name = fmt::format("{}:{}", config_.nvtx_identifier, next_event_index_++);
    auto maybe_entry = create_event_entry(std::move(event_name));
    if (!maybe_entry) {
      alive_token_.reset();
      reserved_events_.clear();
      stage_ = AllocatorStage::kUninitialized;
      return forward_error(maybe_entry);
    }
    reserved_events_.push_back(std::move(maybe_entry.value()));
  }

  stage_ = AllocatorStage::kInitialized;
  return {};
}

}  // namespace holoscan
