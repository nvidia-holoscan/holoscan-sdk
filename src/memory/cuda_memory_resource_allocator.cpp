/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/memory/cuda_memory_resource_allocator.hpp>

#include <holoscan/core/error_helpers.hpp>

#include <cuda_runtime.h>

#include <fmt/format.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/logger/logger.hpp>

namespace holoscan {
namespace {

#ifdef __aarch64__
constexpr const char* kPoolInitialSize = "8MB";
constexpr const char* kPoolMaxSize = "16MB";
#else
constexpr const char* kPoolInitialSize = "16MB";
constexpr const char* kPoolMaxSize = "32MB";
#endif

constexpr size_t kKb = 1024;
constexpr size_t kMb = 1024 * kKb;
constexpr size_t kGb = 1024 * kMb;
constexpr size_t kTb = 1024 * kGb;

using error_detail::cuda_runtime_error;
using error_detail::unexpected_cuda_runtime_error;
using error_detail::unexpected_runtime_error;

expected<size_t, Error> align_up_256(size_t value, std::string_view label) {
  constexpr size_t kAlignment = 256;
  if (value > std::numeric_limits<size_t>::max() - (kAlignment - 1)) {
    return unexpected_runtime_error(ErrorCode::kInvalidArgument,
                                    fmt::format("{} overflows 256-byte alignment", label));
  }
  return ((value + kAlignment - 1) / kAlignment) * kAlignment;
}

}  // namespace

const char* CudaMemoryResourceAllocator::allocator_stage_str(AllocatorStage stage) {
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

CudaMemoryResourceAllocator::CudaMemoryResourceAllocator()
    : config_{kPoolInitialSize, kPoolMaxSize, kPoolInitialSize, kPoolMaxSize, 0} {}

CudaMemoryResourceAllocator::CudaMemoryResourceAllocator(Config config)
    : config_(std::move(config)) {
  if (config_.device_memory_initial_size.empty()) {
    config_.device_memory_initial_size = kPoolInitialSize;
  }
  if (config_.device_memory_max_size.empty()) {
    config_.device_memory_max_size = kPoolMaxSize;
  }
  if (config_.host_memory_initial_size.empty()) {
    config_.host_memory_initial_size = kPoolInitialSize;
  }
  if (config_.host_memory_max_size.empty()) {
    config_.host_memory_max_size = kPoolMaxSize;
  }
}

CudaMemoryResourceAllocator::CudaMemoryResourceAllocator(
    std::string_view device_memory_initial_size, std::string_view device_memory_max_size,
    std::string_view host_memory_initial_size, std::string_view host_memory_max_size,
    int32_t device_id)
    : CudaMemoryResourceAllocator(Config{std::string(device_memory_initial_size),
                                         std::string(device_memory_max_size),
                                         std::string(host_memory_initial_size),
                                         std::string(host_memory_max_size),
                                         device_id}) {}

CudaMemoryResourceAllocator::~CudaMemoryResourceAllocator() {
  auto result = deinitialize_pool();
  if (!result) {
    HOLOSCAN_LOG_ERROR("Failed to deinitialize CUDA memory resource allocator: {}",
                       runtime_error_message(result.error()));
  }
}

void CudaMemoryResourceAllocator::initialize() {
  auto result = initialize_pool();
  if (!result) {
    throw RuntimeError(result.error());
  }
}

expected<size_t, Error> CudaMemoryResourceAllocator::parse_storage_size(
    std::string_view text) const {
  text.remove_prefix(std::min(text.find_first_not_of(' '), text.size()));

  double parsed_value = 0.0;
  const auto* const begin = text.data();
  const auto* const end = text.data() + text.size();
  const auto [suffix_pointer, error_code] = std::from_chars(begin, end, parsed_value);
  if (error_code == std::errc::invalid_argument || suffix_pointer == begin) {
    return unexpected_runtime_error(ErrorCode::kInvalidArgument,
                                    fmt::format("Storage size value is not a number: {}", text));
  }
  if (error_code == std::errc::result_out_of_range || !std::isfinite(parsed_value)) {
    return unexpected_runtime_error(ErrorCode::kInvalidArgument,
                                    fmt::format("Storage size value is too large: {}", text));
  }
  if (parsed_value <= 0.0) {
    return unexpected_runtime_error(ErrorCode::kInvalidArgument,
                                    fmt::format("Storage size must be positive: {}", text));
  }
  if (parsed_value > static_cast<double>(std::numeric_limits<size_t>::max())) {
    return unexpected_runtime_error(ErrorCode::kInvalidArgument,
                                    fmt::format("Storage size value is too large: {}", text));
  }

  std::string_view suffix_view{suffix_pointer, static_cast<size_t>(end - suffix_pointer)};
  if (suffix_view.starts_with(' ')) {
    suffix_view.remove_prefix(1);
  }
  std::string suffix{suffix_view};
  if (suffix.starts_with(' ')) {
    suffix.erase(0, 1);
  }
  std::transform(suffix.begin(), suffix.end(), suffix.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });

  size_t multiplier = kMb;
  if (suffix == "B") {
    multiplier = 1;
  } else if (suffix == "KB") {
    multiplier = kKb;
  } else if (suffix == "MB" || suffix.empty()) {
    multiplier = kMb;
  } else if (suffix == "GB") {
    multiplier = kGb;
  } else if (suffix == "TB") {
    multiplier = kTb;
  } else {
    return unexpected_runtime_error(ErrorCode::kInvalidArgument,
                                    fmt::format("Unexpected storage size suffix: {}", suffix));
  }

  const double bytes = parsed_value * static_cast<double>(multiplier);
  if (bytes > static_cast<double>(std::numeric_limits<size_t>::max())) {
    return unexpected_runtime_error(ErrorCode::kInvalidArgument,
                                    fmt::format("Storage size overflows size_t: {}", text));
  }
  const size_t byte_count = static_cast<size_t>(bytes);
  if (byte_count == 0) {
    return unexpected_runtime_error(
        ErrorCode::kInvalidArgument,
        fmt::format("Storage size rounds down to zero bytes: {}", text));
  }
  return byte_count;
}

// Keep allocator setup linear so validation, CUDA stream creation, and cleanup state transitions
// remain easy to audit.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> CudaMemoryResourceAllocator::initialize_pool() {
  if (stage_ == AllocatorStage::kInitialized) {
    return {};
  }
  stage_ = AllocatorStage::kInitializationInProgress;

  auto maybe_device_initial_size = parse_storage_size(config_.device_memory_initial_size);
  auto maybe_host_initial_size = parse_storage_size(config_.host_memory_initial_size);
  auto maybe_default_initial_size = parse_storage_size(kPoolInitialSize);
  auto maybe_default_max_size = parse_storage_size(kPoolMaxSize);
  auto maybe_device_max_size = parse_storage_size(config_.device_memory_max_size);
  auto maybe_host_max_size = parse_storage_size(config_.host_memory_max_size);
  if (!maybe_device_initial_size) {
    stage_ = AllocatorStage::kUninitialized;
    return forward_error(maybe_device_initial_size);
  }
  if (!maybe_host_initial_size) {
    stage_ = AllocatorStage::kUninitialized;
    return forward_error(maybe_host_initial_size);
  }
  if (!maybe_default_initial_size) {
    stage_ = AllocatorStage::kUninitialized;
    return forward_error(maybe_default_initial_size);
  }
  if (!maybe_default_max_size) {
    stage_ = AllocatorStage::kUninitialized;
    return forward_error(maybe_default_max_size);
  }
  if (!maybe_device_max_size) {
    stage_ = AllocatorStage::kUninitialized;
    return forward_error(maybe_device_max_size);
  }
  if (!maybe_host_max_size) {
    stage_ = AllocatorStage::kUninitialized;
    return forward_error(maybe_host_max_size);
  }

  auto maybe_device_initial_aligned =
      align_up_256(maybe_device_initial_size.value(), "Device memory initial size");
  auto maybe_host_initial_aligned =
      align_up_256(maybe_host_initial_size.value(), "Host memory initial size");
  auto maybe_device_max_aligned =
      align_up_256(maybe_device_max_size.value(), "Device memory max size");
  auto maybe_host_max_aligned = align_up_256(maybe_host_max_size.value(), "Host memory max size");
  if (!maybe_device_initial_aligned) {
    stage_ = AllocatorStage::kUninitialized;
    return forward_error(maybe_device_initial_aligned);
  }
  if (!maybe_host_initial_aligned) {
    stage_ = AllocatorStage::kUninitialized;
    return forward_error(maybe_host_initial_aligned);
  }
  if (!maybe_device_max_aligned) {
    stage_ = AllocatorStage::kUninitialized;
    return forward_error(maybe_device_max_aligned);
  }
  if (!maybe_host_max_aligned) {
    stage_ = AllocatorStage::kUninitialized;
    return forward_error(maybe_host_max_aligned);
  }

  const size_t device_initial_size = maybe_device_initial_aligned.value();
  const size_t host_initial_size = maybe_host_initial_aligned.value();
  const size_t default_initial_size = maybe_default_initial_size.value();
  const size_t default_max_size = maybe_default_max_size.value();

  if (device_initial_size != default_initial_size &&
      maybe_device_max_size.value() == default_max_size) {
    if (device_initial_size > std::numeric_limits<size_t>::max() / 2) {
      stage_ = AllocatorStage::kUninitialized;
      return unexpected_runtime_error(ErrorCode::kInvalidArgument,
                                      "Device memory auto max size overflows size_t");
    }
    auto maybe_auto_max = align_up_256(device_initial_size * 2, "Device memory auto max size");
    if (!maybe_auto_max) {
      stage_ = AllocatorStage::kUninitialized;
      return forward_error(maybe_auto_max);
    }
    device_max_memory_pool_size_ = maybe_auto_max.value();
  } else {
    device_max_memory_pool_size_ = maybe_device_max_aligned.value();
  }

  if (host_initial_size != default_initial_size &&
      maybe_host_max_size.value() == default_max_size) {
    if (host_initial_size > std::numeric_limits<size_t>::max() / 2) {
      stage_ = AllocatorStage::kUninitialized;
      return unexpected_runtime_error(ErrorCode::kInvalidArgument,
                                      "Host memory auto max size overflows size_t");
    }
    auto maybe_auto_max = align_up_256(host_initial_size * 2, "Host memory auto max size");
    if (!maybe_auto_max) {
      stage_ = AllocatorStage::kUninitialized;
      return forward_error(maybe_auto_max);
    }
    host_max_memory_pool_size_ = maybe_auto_max.value();
  } else {
    host_max_memory_pool_size_ = maybe_host_max_aligned.value();
  }

  if (device_max_memory_pool_size_ < device_initial_size) {
    stage_ = AllocatorStage::kUninitialized;
    return unexpected_runtime_error(
        ErrorCode::kInvalidArgument,
        "Device memory max size must be greater than or equal to initial size");
  }
  if (host_max_memory_pool_size_ < host_initial_size) {
    stage_ = AllocatorStage::kUninitialized;
    return unexpected_runtime_error(
        ErrorCode::kInvalidArgument,
        "Host memory max size must be greater than or equal to initial size");
  }

  cudaError_t error = cudaSetDevice(config_.device_id);
  if (error != cudaSuccess) {
    stage_ = AllocatorStage::kUninitialized;
    return unexpected_cuda_runtime_error("Failed to set CUDA device", error);
  }

  error = cudaStreamCreate(&stream_);
  if (error != cudaSuccess) {
    stage_ = AllocatorStage::kUninitialized;
    return unexpected_cuda_runtime_error("Failed to create CUDA stream", error);
  }

  try {
    cuda::memory_pool_properties device_properties;
    device_properties.initial_pool_size = device_initial_size;
    device_properties.max_pool_size = device_max_memory_pool_size_;
    device_pool_ = std::make_unique<cuda::device_memory_pool>(cuda::device_ref{config_.device_id},
                                                              device_properties);

    // host_initial_size is parsed, validated, and aligned, but host_resource_ uses
    // legacy_pinned_memory_resource(cuda::device_ref{config_.device_id}) which allocates on
    // demand rather than preallocating a pool. host_max_memory_pool_size_ enforces the host limit.
    host_resource_ = std::make_unique<cuda::mr::legacy_pinned_memory_resource>(
        cuda::device_ref{config_.device_id});
  } catch (const std::exception& e) {
    stage_ = AllocatorStage::kUninitialized;
    if (stream_ != nullptr) {
      (void)cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
    return unexpected_runtime_error(
        ErrorCode::kFailure, fmt::format("Failed to initialize CCCL allocator: {}", e.what()));
  }

  stage_ = AllocatorStage::kInitialized;
  return {};
}

// Teardown deliberately keeps the cleanup and first-error-preservation flow in one place.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> CudaMemoryResourceAllocator::deinitialize_pool() {
  if (stage_ == AllocatorStage::kUninitialized) {
    return {};
  }

  stage_ = AllocatorStage::kDeinitializationInProgress;
  std::string cleanup_error_message;
  auto remember_cleanup_error = [&cleanup_error_message](const std::string& message) {
    HOLOSCAN_LOG_ERROR("{}", message);
    if (cleanup_error_message.empty()) {
      cleanup_error_message = message;
    }
  };

  if (stream_ != nullptr || device_pool_ != nullptr || host_resource_ != nullptr) {
    const cudaError_t device_error = cudaSetDevice(config_.device_id);
    if (device_error != cudaSuccess) {
      remember_cleanup_error(runtime_error_message(
          cuda_runtime_error("Failed to set CUDA device during teardown", device_error)));
    }
  }

  bool released_device_allocations = false;
  {
    const std::unique_lock<std::shared_mutex> lock(shared_mutex_);
    if (!pool_map_.empty()) {
      HOLOSCAN_LOG_WARN("CCCL allocator still has unreleased memory");
    }

    for (const auto& [pointer, info] : pool_map_) {
      try {
        if (info.storage_type == MemoryStorageType::kDevice) {
          if (device_pool_ == nullptr || stream_ == nullptr) {
            remember_cleanup_error(
                "Cannot release outstanding device allocation because the device pool or stream "
                "is not initialized");
            continue;
          }
          // This is a shutdown fallback for leaked allocations. Normal use should free on the
          // stream that carries the allocation's ordering constraints before deinitialization.
          device_pool_->deallocate(::cuda::stream_ref(stream_), pointer, info.size);
          released_device_allocations = true;
        } else if (info.storage_type == MemoryStorageType::kHost) {
          if (host_resource_ == nullptr) {
            remember_cleanup_error(
                "Cannot release outstanding pinned-host allocation because the host resource is "
                "not initialized");
            continue;
          }
          host_resource_->deallocate_sync(pointer, info.size);
        } else {
          remember_cleanup_error(
              "Cannot release outstanding allocation with unsupported storage type");
        }
      } catch (const std::exception& e) {
        remember_cleanup_error(
            fmt::format("Failed to release outstanding CCCL allocation: {}", e.what()));
      }
    }

    pool_map_.clear();
    host_used_bytes_ = 0;
  }

  if (released_device_allocations && stream_ != nullptr) {
    const cudaError_t sync_error = cudaStreamSynchronize(stream_);
    if (sync_error != cudaSuccess) {
      remember_cleanup_error(runtime_error_message(
          cuda_runtime_error("Failed to synchronize CUDA stream during teardown", sync_error)));
    }
  }

  host_resource_.reset();
  device_pool_.reset();
  if (stream_ != nullptr) {
    const cudaError_t error = cudaStreamDestroy(stream_);
    stream_ = nullptr;
    if (error != cudaSuccess) {
      stage_ = AllocatorStage::kUninitialized;
      return unexpected_cuda_runtime_error("Failed to destroy CUDA stream", error);
    }
  }
  stage_ = AllocatorStage::kUninitialized;
  if (!cleanup_error_message.empty()) {
    return unexpected_runtime_error(ErrorCode::kFailure, cleanup_error_message);
  }
  return {};
}

expected<void, Error> CudaMemoryResourceAllocator::is_available(uint64_t size) {
  return is_available(size, MemoryStorageType::kDevice);
}

expected<void, Error> CudaMemoryResourceAllocator::is_available(uint64_t size,
                                                                MemoryStorageType storage_type) {
  if (stage_ != AllocatorStage::kInitialized) {
    return unexpected_runtime_error(
        ErrorCode::kFailure,
        fmt::format("Allocator must be initialized. Current state is {}",
                    allocator_stage_str(stage_)));
  }
  if (storage_type == MemoryStorageType::kSystem ||
      storage_type == MemoryStorageType::kCudaManaged) {
    return unexpected_runtime_error(ErrorCode::kInvalidArgument,
                                    "CCCL allocator supports only device and pinned host memory");
  }

  return ensure_available(size, storage_type);
}

expected<void, Error> CudaMemoryResourceAllocator::ensure_available(
    uint64_t size, MemoryStorageType storage_type) const {
  auto maybe_available = storage_type == MemoryStorageType::kDevice ? device_available_bytes()
                                                                    : host_available_bytes();
  if (!maybe_available) {
    return forward_error(maybe_available);
  }
  if (size > maybe_available.value()) {
    return unexpected_runtime_error(ErrorCode::kFailure, "Requested memory is not available");
  }
  return {};
}

expected<size_t, Error> CudaMemoryResourceAllocator::device_available_bytes() const {
  if (device_pool_ == nullptr) {
    return unexpected_runtime_error(ErrorCode::kFailure, "Device memory pool is not initialized");
  }
  const size_t used = device_pool_->attribute(cuda::memory_pool_attributes::used_mem_current);
  return used >= device_max_memory_pool_size_ ? 0 : device_max_memory_pool_size_ - used;
}

expected<size_t, Error> CudaMemoryResourceAllocator::host_available_bytes() const {
  if (host_resource_ == nullptr) {
    return unexpected_runtime_error(ErrorCode::kFailure, "Host memory resource is not initialized");
  }
  const std::shared_lock<std::shared_mutex> lock(shared_mutex_);
  return host_used_bytes_ >= host_max_memory_pool_size_
             ? 0
             : host_max_memory_pool_size_ - host_used_bytes_;
}

expected<std::byte*, Error> CudaMemoryResourceAllocator::allocate(uint64_t size,
                                                                  MemoryStorageType storage_type) {
  if (stage_ != AllocatorStage::kInitialized) {
    return unexpected_runtime_error(
        ErrorCode::kFailure,
        fmt::format("Allocator must be initialized. Current state is {}",
                    allocator_stage_str(stage_)));
  }
  if (size == 0) {
    size = 1;
  }
  if (storage_type == MemoryStorageType::kSystem ||
      storage_type == MemoryStorageType::kCudaManaged) {
    return unexpected_runtime_error(ErrorCode::kInvalidArgument,
                                    "CCCL allocator supports only device and pinned host memory");
  }
  auto available = ensure_available(size, storage_type);
  if (!available) {
    return forward_error(available);
  }

  try {
    void* pointer = nullptr;
    if (storage_type == MemoryStorageType::kDevice) {
      pointer = device_pool_->allocate(::cuda::stream_ref(stream_), static_cast<size_t>(size));
    } else {
      const std::unique_lock<std::shared_mutex> lock(shared_mutex_);
      if (static_cast<size_t>(size) > host_max_memory_pool_size_ - host_used_bytes_) {
        return unexpected_runtime_error(ErrorCode::kFailure,
                                        "Requested host memory is not available");
      }
      pointer = host_resource_->allocate_sync(static_cast<size_t>(size));
      auto* byte_pointer = static_cast<std::byte*>(pointer);
      host_used_bytes_ += static_cast<size_t>(size);
      pool_map_.emplace(byte_pointer, AllocationInfo{static_cast<size_t>(size), storage_type});
      return byte_pointer;
    }
    auto* byte_pointer = static_cast<std::byte*>(pointer);
    const std::unique_lock<std::shared_mutex> lock(shared_mutex_);
    pool_map_.emplace(byte_pointer, AllocationInfo{static_cast<size_t>(size), storage_type});
    return byte_pointer;
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Unexpected error while allocating memory: {}", e.what());
    return unexpected_runtime_error(ErrorCode::kFailure,
                                    fmt::format("CCCL allocation failed: {}", e.what()));
  }
}

expected<std::byte*, Error> CudaMemoryResourceAllocator::allocate_async_impl(
    uint64_t size, MemoryStorageType storage_type, cudaStream_t stream) {
  if (storage_type != MemoryStorageType::kDevice) {
    return allocate(size, storage_type);
  }
  if (stage_ != AllocatorStage::kInitialized) {
    return unexpected_runtime_error(
        ErrorCode::kFailure,
        fmt::format("Allocator must be initialized. Current state is {}",
                    allocator_stage_str(stage_)));
  }
  if (size == 0) {
    size = 1;
  }
  auto available = ensure_available(size, MemoryStorageType::kDevice);
  if (!available) {
    return forward_error(available);
  }

  try {
    void* pointer = device_pool_->allocate(::cuda::stream_ref(stream), static_cast<size_t>(size));
    auto* byte_pointer = static_cast<std::byte*>(pointer);
    const std::unique_lock<std::shared_mutex> lock(shared_mutex_);
    pool_map_.emplace(byte_pointer,
                      AllocationInfo{static_cast<size_t>(size), MemoryStorageType::kDevice});
    return byte_pointer;
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Unexpected error while allocating async memory: {}", e.what());
    return unexpected_runtime_error(ErrorCode::kFailure,
                                    fmt::format("CCCL async allocation failed: {}", e.what()));
  }
}

expected<void, Error> CudaMemoryResourceAllocator::free_async_impl(std::byte* pointer,
                                                                   cudaStream_t stream) {
  try {
    const std::unique_lock<std::shared_mutex> lock(shared_mutex_);
    const auto it = pool_map_.find(pointer);
    if (it == pool_map_.end()) {
      return unexpected_runtime_error(ErrorCode::kInvalidArgument,
                                      "Pointer is not defined within this memory pool");
    }
    if (it->second.storage_type == MemoryStorageType::kDevice) {
      device_pool_->deallocate(::cuda::stream_ref(stream), pointer, it->second.size);
    } else {
      host_resource_->deallocate_sync(pointer, it->second.size);
      host_used_bytes_ -= it->second.size;
    }
    pool_map_.erase(it);
    return {};
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Unexpected error during async deallocation: {}", e.what());
    return unexpected_runtime_error(ErrorCode::kFailure,
                                    fmt::format("CCCL async deallocation failed: {}", e.what()));
  }
}

expected<void, Error> CudaMemoryResourceAllocator::free(std::byte* pointer) {
  try {
    const std::unique_lock<std::shared_mutex> lock(shared_mutex_);
    const auto it = pool_map_.find(pointer);
    if (it == pool_map_.end()) {
      return unexpected_runtime_error(ErrorCode::kInvalidArgument,
                                      "Pointer is not defined within this memory pool");
    }
    if (it->second.storage_type == MemoryStorageType::kDevice) {
      device_pool_->deallocate(::cuda::stream_ref(stream_), pointer, it->second.size);
    } else {
      host_resource_->deallocate_sync(pointer, it->second.size);
      host_used_bytes_ -= it->second.size;
    }
    pool_map_.erase(it);
    return {};
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Unexpected error during deallocation: {}", e.what());
    return unexpected_runtime_error(ErrorCode::kFailure,
                                    fmt::format("CCCL deallocation failed: {}", e.what()));
  }
}

expected<size_t, Error> CudaMemoryResourceAllocator::pool_size(
    MemoryStorageType storage_type) const {
  const std::shared_lock<std::shared_mutex> lock(shared_mutex_);
  if (storage_type == MemoryStorageType::kDevice) {
    if (device_pool_ == nullptr) {
      return unexpected_runtime_error(ErrorCode::kFailure, "Device memory pool is not initialized");
    }
    return device_pool_->attribute(cuda::memory_pool_attributes::reserved_mem_current);
  }
  if (storage_type == MemoryStorageType::kHost) {
    if (host_resource_ == nullptr) {
      return unexpected_runtime_error(ErrorCode::kFailure,
                                      "Host memory resource is not initialized");
    }
    return host_used_bytes_;
  }
  return unexpected_runtime_error(ErrorCode::kInvalidArgument,
                                  "CCCL allocator supports only device and pinned host memory");
}

expected<CudaMemoryResourceAllocator::Stats, Error> CudaMemoryResourceAllocator::stats() const {
  const std::shared_lock<std::shared_mutex> lock(shared_mutex_);
  Stats stats;
  stats.initialized = stage_ == AllocatorStage::kInitialized;
  stats.device_pool_configured_max_bytes = device_max_memory_pool_size_;
  stats.host_configured_max_bytes = host_max_memory_pool_size_;
  stats.host_used_bytes = host_used_bytes_;
  stats.host_available_bytes = host_used_bytes_ >= host_max_memory_pool_size_
                                   ? 0
                                   : host_max_memory_pool_size_ - host_used_bytes_;
  stats.allocation_count = pool_map_.size();

  if (!stats.initialized) {
    return stats;
  }
  if (device_pool_ == nullptr) {
    return unexpected_runtime_error(ErrorCode::kFailure, "Device memory pool is not initialized");
  }
  try {
    stats.device_pool_reserved_bytes =
        device_pool_->attribute(cuda::memory_pool_attributes::reserved_mem_current);
    stats.device_pool_used_bytes =
        device_pool_->attribute(cuda::memory_pool_attributes::used_mem_current);
    stats.device_pool_available_bytes =
        stats.device_pool_used_bytes >= device_max_memory_pool_size_
            ? 0
            : device_max_memory_pool_size_ - stats.device_pool_used_bytes;
    return stats;
  } catch (const std::exception& e) {
    return unexpected_runtime_error(
        ErrorCode::kFailure,
        fmt::format("Failed to query CCCL allocator statistics: {}", e.what()));
  }
}

}  // namespace holoscan
