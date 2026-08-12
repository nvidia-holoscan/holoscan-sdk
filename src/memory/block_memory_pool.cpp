/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/memory/block_memory_pool.hpp>

#include <holoscan/core/error_helpers.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#include <holoscan/logger/logger.hpp>

namespace holoscan {
namespace {

using error_detail::unexpected_cuda_runtime_error;
using error_detail::unexpected_runtime_error;

}  // namespace

class BlockMemoryPool::FixedPoolUint64 {
 public:
  expected<void, Error> allocate(uint64_t size) {
    size_ = size;
    next_ = 0;
    available_.resize(size_);
    indices_.resize(size_);
    for (uint64_t i = 0; i < size_; ++i) {
      available_[i] = i;
      indices_[i] = i;
    }
    return {};
  }

  size_t capacity() const { return size_; }
  size_t size() const { return size_ - next_; }
  bool is_available(uint64_t item) const {
    if (item >= size_) {
      return false;
    }
    return indices_[item] >= next_;
  }

  expected<uint64_t, Error> pop() {
    if (next_ == size_) {
      return unexpected_runtime_error(ErrorCode::kFailure, "Block memory pool is empty");
    }
    return available_[next_++];
  }

  expected<void, Error> push(uint64_t item) {
    if (item >= size_) {
      return unexpected_runtime_error(ErrorCode::kInvalidArgument,
                                      "Block index is outside the pool");
    }

    const uint64_t item_index = indices_[item];
    if (item_index >= next_) {
      return unexpected_runtime_error(ErrorCode::kInvalidArgument,
                                      "Block is already available in the pool");
    }

    --next_;
    const uint64_t other = available_[next_];
    std::swap(available_[item_index], available_[next_]);
    std::swap(indices_[item], indices_[other]);
    return {};
  }

 private:
  uint64_t size_ = 0;
  uint64_t next_ = 0;
  std::vector<uint64_t> available_;
  std::vector<uint64_t> indices_;
};

BlockMemoryPool::BlockMemoryPool(Config config) : config_(config) {}

BlockMemoryPool::BlockMemoryPool(MemoryStorageType storage_type, uint64_t block_size,
                                 uint64_t num_blocks, int32_t device_id)
    : config_{storage_type, block_size, num_blocks, device_id, false} {}

BlockMemoryPool::~BlockMemoryPool() {
  auto result = deinitialize_pool();
  if (!result) {
    HOLOSCAN_LOG_ERROR("Failed to deinitialize BlockMemoryPool: {}",
                       runtime_error_message(result.error()));
  }
}

expected<void, Error> BlockMemoryPool::initialize_pool() {
  const std::lock_guard<std::mutex> lock(stack_mutex_);
  if (stage_ == AllocatorStage::kInitialized) {
    return {};
  }
  stage_ = AllocatorStage::kInitializationInProgress;

  if (config_.block_size == 0 || config_.num_blocks == 0) {
    stage_ = AllocatorStage::kUninitialized;
    return unexpected_runtime_error(ErrorCode::kInvalidArgument,
                                    "Block size and number of blocks must be positive");
  }
  if (config_.block_size > std::numeric_limits<uint64_t>::max() / config_.num_blocks) {
    stage_ = AllocatorStage::kUninitialized;
    return unexpected_runtime_error(ErrorCode::kInvalidArgument,
                                    "Block memory pool total size overflows uint64_t");
  }

  const uint64_t total_size = config_.num_blocks * config_.block_size;
  void* raw_pointer = nullptr;

  switch (config_.storage_type) {
    case MemoryStorageType::kHost: {
      const cudaError_t set_device_result = cudaSetDevice(config_.device_id);
      if (set_device_result != cudaSuccess) {
        stage_ = AllocatorStage::kUninitialized;
        return unexpected_cuda_runtime_error("Failed to set CUDA device", set_device_result);
      }
      const cudaError_t error = cudaMallocHost(&raw_pointer, total_size);
      if (error != cudaSuccess) {
        stage_ = AllocatorStage::kUninitialized;
        HOLOSCAN_LOG_ERROR("Failure in cudaMallocHost: {}", cudaGetErrorString(error));
        return unexpected_cuda_runtime_error("cudaMallocHost failed", error);
      }
      break;
    }
    case MemoryStorageType::kDevice: {
      const cudaError_t set_device_result = cudaSetDevice(config_.device_id);
      if (set_device_result != cudaSuccess) {
        stage_ = AllocatorStage::kUninitialized;
        return unexpected_cuda_runtime_error("Failed to set CUDA device", set_device_result);
      }
      const cudaError_t error = cudaMalloc(&raw_pointer, total_size);
      if (error != cudaSuccess) {
        stage_ = AllocatorStage::kUninitialized;
        HOLOSCAN_LOG_ERROR("Failure in cudaMalloc: {}", cudaGetErrorString(error));
        return unexpected_cuda_runtime_error("cudaMalloc failed", error);
      }
      break;
    }
    case MemoryStorageType::kCudaManaged: {
      const cudaError_t set_device_result = cudaSetDevice(config_.device_id);
      if (set_device_result != cudaSuccess) {
        stage_ = AllocatorStage::kUninitialized;
        return unexpected_cuda_runtime_error("Failed to set CUDA device", set_device_result);
      }
      const cudaError_t error = cudaMallocManaged(&raw_pointer, total_size, cudaMemAttachGlobal);
      if (error != cudaSuccess) {
        stage_ = AllocatorStage::kUninitialized;
        HOLOSCAN_LOG_ERROR("Failure in cudaMallocManaged: {}", cudaGetErrorString(error));
        return unexpected_cuda_runtime_error("cudaMallocManaged failed", error);
      }
      break;
    }
    case MemoryStorageType::kSystem:
      raw_pointer = new (std::nothrow) std::byte[total_size];
      if (raw_pointer == nullptr) {
        stage_ = AllocatorStage::kUninitialized;
        return unexpected_runtime_error(ErrorCode::kFailure, "System memory allocation failed");
      }
      break;
    default:
      stage_ = AllocatorStage::kUninitialized;
      return unexpected_runtime_error(ErrorCode::kInvalidArgument, "Invalid memory storage type");
  }

  pointer_ = static_cast<std::byte*>(raw_pointer);
  stack_ = std::make_unique<FixedPoolUint64>();
  auto result = stack_->allocate(config_.num_blocks);
  if (!result) {
    stage_ = AllocatorStage::kUninitialized;
    return forward_error(result);
  }
  deferred_blocks_.clear();
  deferred_blocks_.reserve(config_.num_blocks);
  block_deferred_.assign(config_.num_blocks, 0);
  auto event_result = initialize_block_events();
  if (!event_result) {
    destroy_block_events();
    stack_.reset();
    deferred_blocks_.clear();
    block_deferred_.clear();
    std::byte* allocated_pointer = pointer_;
    pointer_ = nullptr;
    stage_ = AllocatorStage::kUninitialized;

    switch (config_.storage_type) {
      case MemoryStorageType::kHost:
        (void)cudaFreeHost(allocated_pointer);
        break;
      case MemoryStorageType::kDevice:
      case MemoryStorageType::kCudaManaged:
        (void)cudaFree(allocated_pointer);
        break;
      case MemoryStorageType::kSystem:
        delete[] allocated_pointer;
        break;
      default:
        break;
    }
    return forward_error(event_result);
  }

  high_water_used_.store(0, std::memory_order_relaxed);
  failed_acquisitions_.store(0, std::memory_order_relaxed);
  stage_ = AllocatorStage::kInitialized;
  return {};
}

expected<void, Error> BlockMemoryPool::deinitialize_pool() {
  std::unique_ptr<FixedPoolUint64> stack;
  std::byte* pointer = nullptr;
  std::vector<cudaEvent_t> block_events;
  {
    const std::lock_guard<std::mutex> lock(stack_mutex_);
    if (stage_ == AllocatorStage::kUninitialized) {
      return {};
    }
    stage_ = AllocatorStage::kDeinitializationInProgress;

    for (auto& deferred : deferred_blocks_) {
      if (deferred.index < block_events_.size() && block_events_[deferred.index] != nullptr) {
        (void)cudaEventSynchronize(block_events_[deferred.index]);
      }
      if (stack_ != nullptr) {
        auto result = stack_->push(deferred.index);
        if (!result) {
          HOLOSCAN_LOG_ERROR("Failed to return deferred block to pool: {}",
                             runtime_error_message(result.error()));
        }
      }
    }
    deferred_blocks_.clear();
    block_deferred_.clear();

    if (stack_ != nullptr && stack_->size() != config_.num_blocks) {
      HOLOSCAN_LOG_WARN("BlockMemoryPool still has unreleased memory");
    }
    stack = std::move(stack_);
    block_events = std::move(block_events_);
    pointer = pointer_;
    pointer_ = nullptr;
  }

  for (auto event : block_events) {
    if (event != nullptr) {
      (void)cudaEventDestroy(event);
    }
  }

  if (pointer != nullptr) {
    switch (config_.storage_type) {
      case MemoryStorageType::kHost: {
        const cudaError_t error = cudaFreeHost(pointer);
        if (error != cudaSuccess) {
          stage_ = AllocatorStage::kUninitialized;
          HOLOSCAN_LOG_ERROR("Failure in cudaFreeHost: {}", cudaGetErrorString(error));
          return unexpected_cuda_runtime_error("cudaFreeHost failed", error);
        }
        break;
      }
      case MemoryStorageType::kDevice:
      case MemoryStorageType::kCudaManaged: {
        const cudaError_t error = cudaFree(pointer);
        if (error != cudaSuccess) {
          stage_ = AllocatorStage::kUninitialized;
          HOLOSCAN_LOG_ERROR("Failure in cudaFree: {}", cudaGetErrorString(error));
          return unexpected_cuda_runtime_error("cudaFree failed", error);
        }
        break;
      }
      case MemoryStorageType::kSystem:
        delete[] pointer;
        break;
      default:
        stage_ = AllocatorStage::kUninitialized;
        return unexpected_runtime_error(ErrorCode::kInvalidArgument, "Invalid memory storage type");
    }
  }

  stage_ = AllocatorStage::kUninitialized;
  return {};
}

expected<void, Error> BlockMemoryPool::is_available(uint64_t size, MemoryStorageType storage_type) {
  if (stage_ != AllocatorStage::kInitialized) {
    return unexpected_runtime_error(
        ErrorCode::kFailure,
        fmt::format("Allocator must be initialized. Current state is {}",
                    allocator_stage_str(stage_.load())));
  }
  if (storage_type != config_.storage_type) {
    return unexpected_runtime_error(ErrorCode::kInvalidArgument,
                                    "Requested storage type does not match pool storage type");
  }

  const std::lock_guard<std::mutex> lock(stack_mutex_);
  reclaim_completed_deferred_blocks();
  if (stack_ == nullptr || stack_->size() == 0 || size > config_.block_size) {
    return unexpected_runtime_error(ErrorCode::kFailure, "Requested memory is not available");
  }
  return {};
}

expected<std::byte*, Error> BlockMemoryPool::allocate(uint64_t size,
                                                      MemoryStorageType storage_type) {
  if (stage_ != AllocatorStage::kInitialized) {
    record_failed_acquisition();
    return unexpected_runtime_error(
        ErrorCode::kFailure,
        fmt::format("Allocator must be initialized. Current state is {}",
                    allocator_stage_str(stage_.load())));
  }
  if (storage_type != config_.storage_type) {
    record_failed_acquisition();
    return unexpected_runtime_error(ErrorCode::kInvalidArgument,
                                    "Requested storage type does not match pool storage type");
  }
  if (size > config_.block_size) {
    record_failed_acquisition();
    HOLOSCAN_LOG_ERROR(
        "Requested {} bytes of memory in a pool with block size {}", size, config_.block_size);
    return unexpected_runtime_error(ErrorCode::kInvalidArgument,
                                    "Requested allocation exceeds block size");
  }

  const std::lock_guard<std::mutex> lock(stack_mutex_);
  if (stack_ == nullptr) {
    record_failed_acquisition();
    return unexpected_runtime_error(ErrorCode::kFailure, "Block memory pool is not allocated");
  }
  reclaim_completed_deferred_blocks();

  auto maybe_index = stack_->pop();
  if (!maybe_index) {
    record_failed_acquisition();
    HOLOSCAN_LOG_ERROR("Too many chunks allocated, memory of size {} not available", size);
    return forward_error(maybe_index);
  }
  if (config_.enable_stats) {
    update_high_water_locked();
  }
  return pointer_ + maybe_index.value() * config_.block_size;
}

int64_t BlockMemoryPool::pointer_to_index(std::byte* pointer) const {
  if (pointer_ == nullptr || pointer < pointer_ || config_.block_size == 0 ||
      config_.num_blocks == 0) {
    return -1;
  }

  const uint64_t delta = static_cast<uint64_t>(pointer - pointer_);
  const uint64_t index = delta / config_.block_size;
  if (index * config_.block_size != delta || index >= config_.num_blocks) {
    return -1;
  }
  return static_cast<int64_t>(index);
}

expected<void, Error> BlockMemoryPool::initialize_block_events() {
  block_events_.clear();
  if (config_.storage_type == MemoryStorageType::kSystem) {
    return {};
  }

  block_events_.resize(config_.num_blocks, nullptr);
  for (uint64_t index = 0; index < config_.num_blocks; ++index) {
    cudaEvent_t event = nullptr;
    const cudaError_t error = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    if (error != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("cudaEventCreateWithFlags failed: {}", cudaGetErrorString(error));
      return unexpected_cuda_runtime_error("cudaEventCreateWithFlags failed", error);
    }
    block_events_[index] = event;
  }
  return {};
}

void BlockMemoryPool::destroy_block_events() {
  for (auto event : block_events_) {
    if (event != nullptr) {
      (void)cudaEventDestroy(event);
    }
  }
  block_events_.clear();
}

void BlockMemoryPool::reclaim_completed_deferred_blocks() {
  size_t i = 0;
  while (i < deferred_blocks_.size()) {
    const uint64_t index = deferred_blocks_[i].index;
    if (index >= block_events_.size() || block_events_[index] == nullptr) {
      HOLOSCAN_LOG_ERROR("Missing CUDA event for deferred block {}", index);
      deferred_blocks_[i] = std::move(deferred_blocks_.back());
      deferred_blocks_.pop_back();
      continue;
    }

    const cudaError_t status = cudaEventQuery(block_events_[index]);
    if (status == cudaSuccess) {
      auto result = stack_->push(index);
      if (!result) {
        HOLOSCAN_LOG_ERROR("Failed to return deferred block to pool: {}",
                           runtime_error_message(result.error()));
      }
      if (index < block_deferred_.size()) {
        block_deferred_[index] = 0;
      }
      deferred_blocks_[i] = std::move(deferred_blocks_.back());
      deferred_blocks_.pop_back();
    } else if (status == cudaErrorNotReady) {
      ++i;
    } else {
      HOLOSCAN_LOG_ERROR(
          "cudaEventQuery failed for deferred block {}: {}", index, cudaGetErrorString(status));
      if (index < block_deferred_.size()) {
        block_deferred_[index] = 0;
      }
      deferred_blocks_[i] = std::move(deferred_blocks_.back());
      deferred_blocks_.pop_back();
    }
  }
}

uint64_t BlockMemoryPool::checked_out_blocks_locked() const {
  if (stack_ == nullptr) {
    return 0;
  }
  const uint64_t available = static_cast<uint64_t>(stack_->size());
  const uint64_t deferred = static_cast<uint64_t>(deferred_blocks_.size());
  if (available + deferred >= config_.num_blocks) {
    return 0;
  }
  return config_.num_blocks - available - deferred;
}

void BlockMemoryPool::update_high_water_locked() {
  const uint64_t used = checked_out_blocks_locked();
  uint64_t current = high_water_used_.load(std::memory_order_relaxed);
  while (used > current &&
         !high_water_used_.compare_exchange_weak(
             current, used, std::memory_order_relaxed, std::memory_order_relaxed)) {
  }
}

void BlockMemoryPool::record_failed_acquisition() {
  if (!config_.enable_stats) {
    return;
  }
  failed_acquisitions_.fetch_add(1, std::memory_order_relaxed);
}

BlockMemoryPool::Stats BlockMemoryPool::stats() {
  const std::lock_guard<std::mutex> lock(stack_mutex_);
  if (stack_ != nullptr) {
    reclaim_completed_deferred_blocks();
  }

  Stats stats;
  stats.capacity = config_.num_blocks;
  stats.available = stack_ != nullptr ? static_cast<uint64_t>(stack_->size()) : 0;
  stats.deferred = static_cast<uint64_t>(deferred_blocks_.size());
  stats.used = checked_out_blocks_locked();
  stats.high_water_used = high_water_used_.load(std::memory_order_relaxed);
  stats.failed_acquisitions = failed_acquisitions_.load(std::memory_order_relaxed);
  stats.initialized = stage_.load() == AllocatorStage::kInitialized;
  stats.counters_enabled = config_.enable_stats;
  return stats;
}

expected<void, Error> BlockMemoryPool::free(std::byte* pointer) {
  const std::lock_guard<std::mutex> lock(stack_mutex_);
  const int64_t index = pointer_to_index(pointer);
  if (index < 0) {
    return unexpected_runtime_error(ErrorCode::kInvalidArgument,
                                    "Pointer does not belong to this block memory pool");
  }
  if (stack_ == nullptr) {
    return unexpected_runtime_error(ErrorCode::kFailure, "Block memory pool is not allocated");
  }
  return stack_->push(static_cast<uint64_t>(index));
}

expected<std::byte*, Error> BlockMemoryPool::allocate_async_impl(uint64_t size,
                                                                 MemoryStorageType storage_type,
                                                                 cudaStream_t stream) {
  (void)stream;
  return allocate(size, storage_type);
}

expected<void, Error> BlockMemoryPool::free_async_impl(std::byte* pointer, cudaStream_t stream) {
  if (stream == nullptr || config_.storage_type == MemoryStorageType::kSystem) {
    return free(pointer);
  }

  const int64_t signed_index = pointer_to_index(pointer);
  if (signed_index < 0) {
    return unexpected_runtime_error(ErrorCode::kInvalidArgument,
                                    "Pointer does not belong to this block memory pool");
  }
  const uint64_t index = static_cast<uint64_t>(signed_index);

  const std::lock_guard<std::mutex> lock(stack_mutex_);
  if (stack_ == nullptr) {
    return unexpected_runtime_error(ErrorCode::kFailure, "Block memory pool is not allocated");
  }
  if (index >= block_events_.size() || block_events_[index] == nullptr) {
    return unexpected_runtime_error(ErrorCode::kFailure, "Block memory pool has no CUDA event");
  }
  if (stack_->is_available(index)) {
    return unexpected_runtime_error(ErrorCode::kInvalidArgument,
                                    "Block is already available in the pool");
  }
  if (index < block_deferred_.size() && block_deferred_[index] != 0) {
    return unexpected_runtime_error(ErrorCode::kInvalidArgument,
                                    "Block is already pending deferred release");
  }

  const cudaError_t error =
      cudaEventRecord(block_events_[index], static_cast<cudaStream_t>(stream));
  if (error != cudaSuccess) {
    HOLOSCAN_LOG_ERROR("cudaEventRecord failed: {}", cudaGetErrorString(error));
    return unexpected_cuda_runtime_error("cudaEventRecord failed", error);
  }

  if (index < block_deferred_.size()) {
    block_deferred_[index] = 1;
  }
  deferred_blocks_.push_back(DeferredBlock(index));
  return {};
}

uint64_t BlockMemoryPool::block_size(MemoryStorageType storage_type) const {
  if (storage_type != config_.storage_type) {
    return 0;
  }
  return config_.block_size;
}

const char* BlockMemoryPool::allocator_stage_str(AllocatorStage stage) {
  switch (stage) {
    case AllocatorStage::kUninitialized:
      return "Uninitialized";
    case AllocatorStage::kInitializationInProgress:
      return "InitializationInProgress";
    case AllocatorStage::kInitialized:
      return "Initialized";
    case AllocatorStage::kDeinitializationInProgress:
      return "DeinitializationInProgress";
    default:
      return "Unknown";
  }
}

}  // namespace holoscan
