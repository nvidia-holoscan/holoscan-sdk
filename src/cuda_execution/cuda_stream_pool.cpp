/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/cuda_execution/cuda_stream_pool.hpp>

#include <holoscan/core/error_helpers.hpp>

#include <fmt/format.h>

#include <nvtx3/nvToolsExtCudaRt.h>

#include <algorithm>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/diagnostic_reporting.hpp"
#include "cuda_driver_entry_points.hpp"

namespace holoscan {

namespace {

using error_detail::cuda_runtime_error;
using error_detail::unexpected_cuda_runtime_error;
using error_detail::unexpected_runtime_error;

#if HOLOSCAN_CUDA_DRIVER_ENTRY_POINTS_AVAILABLE
using cuda_execution_detail::ctx_pop_current_fn;
using cuda_execution_detail::ctx_push_current_fn;
using cuda_execution_detail::resolve_driver_entry_point;
#endif

#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
using GreenCtxStreamCreateFn = CUresult (*)(CUstream*, CUgreenCtx, unsigned int, int);

// Resolve cuGreenCtxStreamCreate once and cache the result (pointer or diagnostic) in a thread-safe
// magic static.
const expected<GreenCtxStreamCreateFn, Error>& green_ctx_stream_create_fn() {
  static const expected<GreenCtxStreamCreateFn, Error> fn =
      resolve_driver_entry_point<GreenCtxStreamCreateFn>(
          "cuGreenCtxStreamCreate", cuda_execution_detail::kCUDADriverABIVersion12050);
  return fn;
}
#endif

}  // namespace

CudaStreamPool::StreamLease::StreamLease(CudaStreamPool* pool, std::shared_ptr<StreamEntry> entry,
                                         std::weak_ptr<bool> pool_alive)
    : pool_(pool), entry_(std::move(entry)), pool_alive_(std::move(pool_alive)) {}

CudaStreamPool::StreamLease::StreamLease(StreamLease&& other) noexcept {
  *this = std::move(other);
}

CudaStreamPool::StreamLease& CudaStreamPool::StreamLease::operator=(
    CudaStreamPool::StreamLease&& other) noexcept {
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

CudaStreamPool::StreamLease::~StreamLease() {
  auto result = release();
  if (!result) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR("Failed to release CUDA stream lease: {}",
                                 runtime_error_message(result.error()));
  }
}

expected<void, Error> CudaStreamPool::StreamLease::release() {
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
    return make_unexpected(Error{ErrorCode::kFailure, "CUDA stream lease has no owning pool"});
  }

  auto entry = std::move(entry_);
  auto* pool = pool_;
  pool_ = nullptr;
  pool_alive_.reset();
  return pool->release(std::move(entry));
}

::cuda::stream_ref CudaStreamPool::StreamLease::stream() const {
  if (entry_ == nullptr) {
    throw std::runtime_error("accessing moved-from or default-constructed StreamLease");
  }
  return entry_->stream;
}

cudaStream_t CudaStreamPool::StreamLease::native_handle() const {
  if (entry_ == nullptr) {
    throw std::runtime_error("accessing moved-from or default-constructed StreamLease");
  }
  return entry_->stream.get();
}

CudaStreamPool::CudaStreamPool(Config config) : config_(std::move(config)) {}

CudaStreamPool::CudaStreamPool(int32_t device_id, uint32_t stream_flags, int32_t stream_priority,
                               uint32_t reserved_size, uint32_t max_size,
                               std::string nvtx_identifier)
    : config_{device_id,
              stream_flags,
              stream_priority,
              reserved_size,
              max_size,
              std::move(nvtx_identifier)} {}

CudaStreamPool::~CudaStreamPool() {
  auto result = deinitialize_pool();
  if (!result) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR("Failed to deinitialize CUDA stream pool: {}",
                                 runtime_error_message(result.error()));
  }
}

void CudaStreamPool::initialize() {
  auto result = initialize_pool();
  if (!result) {
    throw RuntimeError(result.error());
  }
}

expected<void, Error> CudaStreamPool::initialize_pool() {
  auto result = reserve_streams();
  if (!result) {
    return forward_error(result);
  }
  return {};
}

expected<void, Error> CudaStreamPool::deinitialize_pool() {
  const std::lock_guard<std::mutex> lock(mutex_);
  stage_ = AllocatorStage::kDeinitializationInProgress;

  if (!allocated_streams_.empty()) {
    HOLOSCAN_LOG_AUTOMATIC_WARN("CudaStreamPool still has leased streams");
  }
  alive_token_.reset();
  allocated_streams_.clear();
  reserved_streams_.clear();
  next_stream_index_ = 0;
  stage_ = AllocatorStage::kUninitialized;
  return {};
}

expected<CudaStreamPool::StreamLease, Error> CudaStreamPool::acquire() {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (stage_ != AllocatorStage::kInitialized) {
    return make_unexpected(Error{ErrorCode::kFailure, "CudaStreamPool is not initialized"});
  }

  if (config_.max_size != 0 && allocated_streams_.size() >= config_.max_size) {
    return make_unexpected(Error{ErrorCode::kFailure, "CudaStreamPool has reached capacity"});
  }

  std::shared_ptr<StreamEntry> entry;
  if (!reserved_streams_.empty()) {
    entry = std::move(reserved_streams_.back());
    reserved_streams_.pop_back();
  } else {
    auto stream_name = fmt::format("{}:{}", config_.nvtx_identifier, next_stream_index_++);
    auto maybe_entry = create_stream_entry(std::move(stream_name));
    if (!maybe_entry) {
      return forward_error(maybe_entry);
    }
    entry = std::move(maybe_entry.value());
  }

  allocated_streams_.push_back(entry.get());
  return StreamLease(this, std::move(entry), alive_token_);
}

expected<void, Error> CudaStreamPool::release(std::shared_ptr<StreamEntry> entry) {
  if (entry == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "Cannot release a null CUDA stream entry"});
  }

  const std::lock_guard<std::mutex> lock(mutex_);
  const auto iter = std::find(allocated_streams_.begin(), allocated_streams_.end(), entry.get());
  if (iter == allocated_streams_.end()) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "CUDA stream entry was not allocated by pool"});
  }

  *iter = allocated_streams_.back();
  allocated_streams_.pop_back();
  reserved_streams_.push_back(std::move(entry));
  return {};
}

size_t CudaStreamPool::available() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return reserved_streams_.size();
}

size_t CudaStreamPool::allocated() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return allocated_streams_.size();
}

// Dispatches across three stream-creation paths (green context, custom context, default). The
// branching reads linearly but trips clang-tidy's cognitive-complexity heuristic; suppress it here
// rather than fragmenting a straightforward function into single-use helpers.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<std::shared_ptr<CudaStreamPool::StreamEntry>, Error> CudaStreamPool::create_stream_entry(
    std::string name) {
  try {
    int previous_device = 0;
    const cudaError_t get_device_result = cudaGetDevice(&previous_device);
    if (get_device_result != cudaSuccess) {
      return unexpected_cuda_runtime_error("Failed to query current CUDA device",
                                           get_device_result);
    }

    const cudaError_t set_device_result = cudaSetDevice(config_.device_id);
    if (set_device_result != cudaSuccess) {
      return unexpected_cuda_runtime_error("Failed to set CUDA stream pool device",
                                           set_device_result);
    }

    cudaStream_t native_stream = nullptr;
    expected<void, Error> create_status;  // default-constructed = success
#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
    if (config_.green_context != nullptr) {
      const auto& green_ctx_stream_create = green_ctx_stream_create_fn();
      if (!green_ctx_stream_create) {
        create_status = forward_error(green_ctx_stream_create);
      } else {
        const unsigned int cu_flags = (config_.stream_flags & cudaStreamNonBlocking) != 0u
                                          ? CU_STREAM_NON_BLOCKING
                                          : CU_STREAM_DEFAULT;
        const CUresult cu_result = green_ctx_stream_create.value()(
            &native_stream, config_.green_context, cu_flags, config_.stream_priority);
        if (cu_result != CUDA_SUCCESS) {
          create_status = unexpected_runtime_error(
              ErrorCode::kFailure,
              fmt::format("Failed to create CUDA stream in green context (CUresult {})",
                          static_cast<int>(cu_result)));
        }
      }
    } else  // NOLINT(readability/braces)
#endif
        if (config_.cuda_context != nullptr) {
#if HOLOSCAN_CUDA_DRIVER_ENTRY_POINTS_AVAILABLE
      const auto& push_current = ctx_push_current_fn();
      const auto& pop_current = ctx_pop_current_fn();
      if (!push_current) {
        create_status = forward_error(push_current);
      } else if (!pop_current) {
        create_status = forward_error(pop_current);
      } else {
        const CUresult push_result = push_current.value()(config_.cuda_context);
        if (push_result != CUDA_SUCCESS) {
          create_status = unexpected_runtime_error(
              ErrorCode::kFailure,
              fmt::format("Failed to push provided CUDA context (CUresult {})",
                          static_cast<int>(push_result)));
        } else {
          // Verify the provided context is on the configured device while it is current.
          auto device_check =
              cuda_execution_detail::verify_current_context_device(config_.device_id);
          if (device_check) {
            const cudaError_t create_result = cudaStreamCreateWithPriority(
                &native_stream, config_.stream_flags, config_.stream_priority);
            if (create_result != cudaSuccess) {
              create_status = make_unexpected(cuda_runtime_error(
                  "Failed to create CUDA stream in provided context", create_result));
            }
          }
          // Always pop to restore the caller's context stack, even if the check/creation failed.
          CUcontext previous_context = nullptr;
          const CUresult pop_result = pop_current.value()(&previous_context);
          (void)previous_context;
          if (pop_result != CUDA_SUCCESS) {
            if (native_stream != nullptr) {
              (void)cudaStreamDestroy(native_stream);
              native_stream = nullptr;
            }
            create_status = unexpected_runtime_error(
                ErrorCode::kFailure,
                fmt::format("Failed to restore CUDA context after stream creation (CUresult {})",
                            static_cast<int>(pop_result)));
          } else if (!device_check) {
            create_status = forward_error(device_check);
          }
        }
      }
#else
      create_status = unexpected_runtime_error(ErrorCode::kFailure,
                                               "Config::cuda_context requires CUDA 12.3 or newer");
#endif
    } else {
      const cudaError_t create_result = cudaStreamCreateWithPriority(
          &native_stream, config_.stream_flags, config_.stream_priority);
      if (create_result != cudaSuccess) {
        create_status =
            make_unexpected(cuda_runtime_error("Failed to create CUDA stream", create_result));
      }
    }

    const cudaError_t restore_result = cudaSetDevice(previous_device);
    if (restore_result != cudaSuccess) {
      if (native_stream != nullptr) {
        (void)cudaStreamDestroy(native_stream);
      }
      return unexpected_cuda_runtime_error("Failed to restore CUDA device", restore_result);
    }
    if (!create_status) {
      return forward_error(create_status);
    }

    nvtxNameCudaStreamA(native_stream, name.c_str());
    auto entry = std::make_shared<StreamEntry>(
        StreamEntry{::cuda::stream::from_native_handle(native_stream), std::move(name)});
    return entry;
  } catch (const std::exception& e) {
    return make_unexpected(
        Error{ErrorCode::kFailure, fmt::format("Failed to create CUDA stream: {}", e.what())});
  }
}

expected<void, Error> CudaStreamPool::reserve_streams() {
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
                                    "CudaStreamPool max_size must be zero or at least "
                                    "reserved_size");
  }

#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
  if (config_.green_context != nullptr && config_.cuda_context != nullptr) {
    stage_ = AllocatorStage::kUninitialized;
    return unexpected_runtime_error(
        ErrorCode::kInvalidArgument,
        "CudaStreamPool cannot set both green_context and cuda_context");
  }
#endif

  const auto bounded_capacity = config_.max_size == 0 ? config_.reserved_size : config_.max_size;
  reserved_streams_.reserve(bounded_capacity);
  allocated_streams_.reserve(bounded_capacity);
  for (uint32_t i = 0; i < config_.reserved_size; ++i) {
    auto stream_name = fmt::format("{}:{}", config_.nvtx_identifier, next_stream_index_++);
    auto maybe_entry = create_stream_entry(std::move(stream_name));
    if (!maybe_entry) {
      alive_token_.reset();
      reserved_streams_.clear();
      stage_ = AllocatorStage::kUninitialized;
      return forward_error(maybe_entry);
    }
    reserved_streams_.push_back(std::move(maybe_entry.value()));
  }

  stage_ = AllocatorStage::kInitialized;
  return {};
}

}  // namespace holoscan
