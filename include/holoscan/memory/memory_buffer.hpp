/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda/stream>

#include <fmt/format.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <source_location>
#include <utility>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/error_helpers.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/memory/allocator.hpp>
#include <holoscan/memory/cuda_stream_allocator.hpp>

namespace holoscan {

class MemoryBuffer {
 public:
  /// Maximum number of distinct CUDA consumer streams retained by one backing allocation.
  static constexpr std::size_t kMaximumFanoutReleaseStreams = 8;
  /// Default consumer-stream capacity for allocations without a plan-derived bound.
  static constexpr std::size_t kDefaultMaxFanoutReleaseStreams = kMaximumFanoutReleaseStreams;

  struct Options {
    std::size_t max_fanout_release_streams = kDefaultMaxFanoutReleaseStreams;
  };

  MemoryBuffer() = default;
  explicit MemoryBuffer(Options options)
      : max_fanout_release_streams_(options.max_fanout_release_streams) {}
  MemoryBuffer(const MemoryBuffer&) = delete;
  MemoryBuffer& operator=(const MemoryBuffer&) = delete;

  MemoryBuffer(MemoryBuffer&& other) noexcept { *this = std::move(other); }

  MemoryBuffer& operator=(MemoryBuffer&& other) noexcept {
    if (this != &other) {
      (void)free_buffer();
      max_fanout_release_streams_ = other.max_fanout_release_streams_;
      size_ = other.size_;
      storage_type_ = other.storage_type_;
      device_id_ = other.device_id_;
      pointer_ = other.pointer_;
      release_func_ = std::move(other.release_func_);
      stream_state_ = std::move(other.stream_state_);

      other.pointer_ = nullptr;
      other.device_id_ = Allocator::kUnknownDeviceId;
      other.release_func_ = nullptr;
      other.stream_state_ = nullptr;
    }

    return *this;
  }

  // Type of the callback function to release memory passed to the MemoryBuffer using wrap_memory.
  using release_function_t = std::function<expected<void, Error>(void* pointer)>;

  [[nodiscard]] expected<void, Error> set_max_fanout_release_streams(std::size_t capacity) {
    if (pointer_ != nullptr || stream_state_ != nullptr) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument,
                "MemoryBuffer fanout release-stream capacity must be configured before storage is "
                "attached"});
    }

    auto valid = validate_fanout_release_stream_capacity(capacity);
    if (!valid) {
      return forward_error(valid);
    }
    max_fanout_release_streams_ = capacity;
    return {};
  }

  [[nodiscard]] std::size_t max_fanout_release_streams() const noexcept {
    return max_fanout_release_streams_;
  }

  [[nodiscard]] expected<void, Error> free_buffer() {
    if (pointer_ != nullptr) {
      if (release_func_ != nullptr) {
        const expected<void, Error> result = release_func_(pointer_);
        if (!result) {
          return forward_error(result);
        }
        release_func_ = nullptr;
      }

      pointer_ = nullptr;
      size_ = 0;
      stream_state_ = nullptr;
      device_id_ = Allocator::kUnknownDeviceId;
    }

    return {};
  }

  virtual ~MemoryBuffer() { (void)free_buffer(); }

  /// @brief Resize the buffer using a native allocator.
  /// @param allocator Allocator that owns the new backing storage.
  /// @param size Requested size in bytes.
  /// @param storage_type Requested storage location.
  /// @param stream Optional CUDA stream for stream-aware allocation and release.
  /// @return Success or the original release/allocation Error. Failed release retains the backing
  /// storage and prevents a new allocation. Reporting follows the automatic diagnostic policy and
  /// cannot replace the returned failure. Use set_stream() to record later consumer work.
  [[nodiscard]] expected<void, Error> resize(std::shared_ptr<Allocator> allocator, uint64_t size,
                                             MemoryStorageType storage_type,
                                             void* stream = nullptr) {
    const auto capacity_valid =
        validate_fanout_release_stream_capacity(max_fanout_release_streams_);
    if (!capacity_valid) {
      return forward_error(capacity_valid);
    }

    const auto result = free_buffer();
    if (!result) {
#if HOLOSCAN_LOG_ACTIVE_LEVEL <= HOLOSCAN_LOG_LEVEL_ERROR
      report_release_failure(result.error());
#endif
      return forward_error(result);
    }

    if (allocator == nullptr) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "MemoryBuffer::resize allocator is null"});
    }

    auto* stream_allocator = dynamic_cast<CudaStreamAllocator*>(allocator.get());
    const auto maybe = stream != nullptr && stream_allocator != nullptr
                           ? stream_allocator->allocate_async(
                                 size, storage_type, static_cast<cudaStream_t>(stream))
                           : allocator->allocate(size, storage_type);
    if (!maybe) {
#if HOLOSCAN_LOG_ACTIVE_LEVEL <= HOLOSCAN_LOG_LEVEL_ERROR
      report_allocation_failure(maybe.error(), size, storage_type);
#endif
      return forward_error(maybe);
    }

    auto stream_state = std::make_shared<StreamState>(max_fanout_release_streams_);
    stream_state->initial_stream.store(stream, std::memory_order_release);
    auto initialize_sync = stream_state->initialize_sync_resources(storage_type);
    if (!initialize_sync) {
      if (stream != nullptr && stream_allocator != nullptr) {
        (void)stream_allocator->free_async(maybe.value(), static_cast<cudaStream_t>(stream));
      } else {
        (void)allocator->free(maybe.value());
      }
      return forward_error(initialize_sync);
    }

    storage_type_ = storage_type;
    pointer_ = maybe.value();
    size_ = size;
    stream_state_ = std::move(stream_state);
    device_id_ = allocator->device_id(storage_type);

    release_func_ = [allocator, storage_type, stream_state = stream_state_](
                        void* ptr) -> expected<void, Error> {
      auto* release_stream_allocator = dynamic_cast<CudaStreamAllocator*>(allocator.get());
      const auto release_with_stream = [&](void* release_stream) -> expected<void, Error> {
        if (release_stream != nullptr && release_stream_allocator != nullptr) {
          return release_stream_allocator->free_async(static_cast<std::byte*>(ptr),
                                                      static_cast<cudaStream_t>(release_stream));
        }
        return allocator->free(static_cast<std::byte*>(ptr));
      };

      if (stream_state == nullptr) {
        return allocator->free(static_cast<std::byte*>(ptr));
      }

      const auto release_streams = stream_state->release_streams_snapshot();
      if (release_streams.count == 0) {
        return release_with_stream(stream_state->initial_stream_snapshot());
      }
      if (release_streams.count == 1) {
        return release_with_stream(release_streams.streams.front());
      }
      if (storage_type == MemoryStorageType::kSystem) {
        return release_with_stream(release_streams.streams[release_streams.count - 1]);
      }

      auto join_result = stream_state->join_release_streams(release_streams);
      if (!join_result) {
        return forward_error(join_result);
      }
      return release_with_stream(stream_state->fanout_join_stream());
    };

    return {};
  }

  [[nodiscard]] expected<void, Error> resize(std::shared_ptr<Allocator> allocator, uint64_t size,
                                             MemoryStorageType storage_type,
                                             ::cuda::stream_ref stream) {
    return resize(std::move(allocator), size, storage_type, static_cast<void*>(stream.get()));
  }

  /** @brief Wrap externally allocated memory.
   * @param pointer Backing memory pointer.
   * @param size Backing memory size in bytes.
   * @param storage_type Memory storage type.
   * @param release_func Optional ownership-release callback.
   * @param device_id CUDA ordinal or Allocator::kUnknownDeviceId.
   * @return Empty success or Error when existing storage cannot be released or initialized. */
  [[nodiscard]] expected<void, Error> wrap_memory(void* pointer, uint64_t size,
                                                  MemoryStorageType storage_type,
                                                  release_function_t release_func,
                                                  int32_t device_id = Allocator::kUnknownDeviceId) {
    const auto capacity_valid =
        validate_fanout_release_stream_capacity(max_fanout_release_streams_);
    if (!capacity_valid) {
      return forward_error(capacity_valid);
    }

    if (pointer != this->pointer()) {
      const auto result = free_buffer();
      if (!result) {
        return forward_error(result);
      }

      pointer_ = static_cast<std::byte*>(pointer);
    }
    storage_type_ = storage_type;
    size_ = size;
    release_func_ = std::move(release_func);
    // Record the caller-provided device ordinal (defaults to unknown). Set unconditionally so the
    // same-pointer path does not retain a stale hint left by a prior resize().
    device_id_ = device_id;

    if (!stream_state_) {
      stream_state_ = std::make_shared<StreamState>(max_fanout_release_streams_);
    }
    auto initialize_sync = stream_state_->initialize_sync_resources(storage_type);
    if (!initialize_sync) {
      return forward_error(initialize_sync);
    }

    return {};
  }

  // The type of memory where the data is stored.
  [[nodiscard]] MemoryStorageType storage_type() const noexcept { return storage_type_; }

  // Raw pointer to the first byte of elements stored in the buffer.
  [[nodiscard]] std::byte* pointer() const noexcept { return pointer_; }

  // Size of buffer contents in bytes.
  [[nodiscard]] uint64_t size() const noexcept { return size_; }

  // CUDA device ordinal the allocator bound this buffer to, for storage types that carry a
  // meaningful DLPack ordinal (device memory). It is Allocator::kUnknownDeviceId for pinned host
  // memory (DLPack kDLCUDAHost uses device_id 0), managed memory (DLPack kDLCUDAManaged uses
  // device_id 0), system memory, and wrapped/external memory unless the wrapper supplied an
  // explicit ordinal. Tensor construction uses a nonnegative value as trusted metadata to set
  // DLPack device metadata without a per-attach pointer query.
  [[nodiscard]] int32_t device_id() const noexcept { return device_id_; }

  // Sets the CUDA stream that last operated on this buffer's data.
  [[nodiscard]] bool set_stream(void* stream) {
    if (stream_state_) {
      return stream_state_->add_release_stream(stream);
    }
    return false;
  }

  [[nodiscard]] bool set_stream(::cuda::stream_ref stream) {
    return set_stream(static_cast<void*>(stream.get()));
  }

  // Gets the CUDA stream that last operated on this buffer's data.
  [[nodiscard]] void* stream() const noexcept {
    return stream_state_ ? stream_state_->last_stream_snapshot() : nullptr;
  }

  // Shared state for CUDA release streams. Fanout copies may record different consumer streams;
  // final buffer release joins all recorded streams before returning memory to the allocator.
  // The current implementation uses a conservative fixed fanout limit. Production setup should
  // size this from static graph fanout and pair each slot with a stable per-subscriber access
  // guard.
  struct StreamState {
    explicit StreamState(std::size_t max_release_streams = kDefaultMaxFanoutReleaseStreams)
        : max_release_streams(max_release_streams) {}

    struct StreamSnapshot {
      // Fixed-size arrays keep snapshots allocation-free. release_streams_snapshot() and
      // join_release_streams() only use entries below max_release_streams.
      std::array<void*, kMaximumFanoutReleaseStreams> streams{};
      std::size_t count = 0;
    };

    const std::size_t max_release_streams = kDefaultMaxFanoutReleaseStreams;
    std::atomic<void*> initial_stream{nullptr};
    // Arrays are sized to kMaximumFanoutReleaseStreams by design; initialize_sync_resources()
    // creates events only for max_release_streams entries and destroy_sync_resources() tolerates
    // the remaining nullptr slots.
    std::array<std::atomic<void*>, kMaximumFanoutReleaseStreams> release_streams{};

    ~StreamState() { destroy_sync_resources(); }

    [[nodiscard]] expected<void, Error> initialize_sync_resources(MemoryStorageType storage_type) {
      if (storage_type == MemoryStorageType::kSystem || sync_resources_initialized) {
        return {};
      }

      cudaError_t error = cudaStreamCreateWithFlags(&join_stream, cudaStreamNonBlocking);
      if (error != cudaSuccess) {
        return error_detail::unexpected_cuda_runtime_error("cudaStreamCreateWithFlags failed",
                                                           error);
      }

      for (std::size_t i = 0; i < max_release_streams; ++i) {
        error = cudaEventCreateWithFlags(&release_events[i], cudaEventDisableTiming);
        if (error != cudaSuccess) {
          destroy_sync_resources();
          return error_detail::unexpected_cuda_runtime_error("cudaEventCreateWithFlags failed",
                                                             error);
        }
      }
      sync_resources_initialized = true;
      return {};
    }

    void destroy_sync_resources() {
      for (auto& event : release_events) {
        if (event != nullptr) {
          (void)cudaEventDestroy(event);
          event = nullptr;
        }
      }
      if (join_stream != nullptr) {
        (void)cudaStreamDestroy(join_stream);
        join_stream = nullptr;
      }
      sync_resources_initialized = false;
    }

    [[nodiscard]] expected<void, Error> join_release_streams(const StreamSnapshot& snapshot) const {
      if (!sync_resources_initialized || join_stream == nullptr) {
        return make_unexpected(
            Error{ErrorCode::kFailure, "MemoryBuffer CUDA release resources unavailable"});
      }

      for (std::size_t i = 0; i < snapshot.count; ++i) {
        cudaEvent_t event = release_events[i];
        cudaError_t error = cudaEventRecord(event, static_cast<cudaStream_t>(snapshot.streams[i]));
        if (error != cudaSuccess) {
          return error_detail::unexpected_cuda_runtime_error("cudaEventRecord failed", error);
        }

        error = cudaStreamWaitEvent(join_stream, event, 0);
        if (error != cudaSuccess) {
          return error_detail::unexpected_cuda_runtime_error("cudaStreamWaitEvent failed", error);
        }
      }

      return {};
    }

    [[nodiscard]] cudaStream_t fanout_join_stream() const noexcept { return join_stream; }
    [[nodiscard]] bool fanout_sync_resources_ready() const noexcept {
      return sync_resources_initialized;
    }

    [[nodiscard]] bool add_release_stream(void* stream) {
      if (stream == nullptr) {
        return true;
      }

      for (std::size_t i = 0; i < max_release_streams; ++i) {
        void* existing = release_streams[i].load(std::memory_order_acquire);
        if (existing == stream) {
          return true;
        }
      }

      for (std::size_t i = 0; i < max_release_streams; ++i) {
        void* expected = nullptr;
        if (release_streams[i].compare_exchange_strong(
                expected, stream, std::memory_order_acq_rel, std::memory_order_acquire)) {
          return true;
        }
        if (expected == stream) {
          return true;
        }
      }
      return false;
    }

    [[nodiscard]] void* initial_stream_snapshot() const noexcept {
      return initial_stream.load(std::memory_order_acquire);
    }

    [[nodiscard]] StreamSnapshot release_streams_snapshot() const {
      StreamSnapshot snapshot;
      for (std::size_t i = 0; i < max_release_streams; ++i) {
        void* stream = release_streams[i].load(std::memory_order_acquire);
        if (stream != nullptr) {
          snapshot.streams[snapshot.count++] = stream;
        }
      }
      return snapshot;
    }

    [[nodiscard]] void* last_stream_snapshot() const noexcept {
      void* last_stream = nullptr;
      for (std::size_t i = 0; i < max_release_streams; ++i) {
        void* stream = release_streams[i].load(std::memory_order_acquire);
        if (stream != nullptr) {
          last_stream = stream;
        }
      }
      if (last_stream != nullptr) {
        return last_stream;
      }
      return initial_stream_snapshot();
    }

    cudaStream_t join_stream = nullptr;
    std::array<cudaEvent_t, kMaximumFanoutReleaseStreams> release_events{};
    bool sync_resources_initialized = false;
  };

  // Returns the shared stream state for release functions that need stream updates.
  [[nodiscard]] std::shared_ptr<StreamState> stream_state() const { return stream_state_; }

  [[nodiscard]] static expected<void, Error> validate_fanout_release_stream_capacity(
      std::size_t capacity) {
    if (capacity == 0 || capacity > kMaximumFanoutReleaseStreams) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument,
                fmt::format("MemoryBuffer fanout release-stream capacity must be in [1, {}]",
                            kMaximumFanoutReleaseStreams)});
    }
    return {};
  }

 protected:
  std::size_t max_fanout_release_streams_ = kDefaultMaxFanoutReleaseStreams;
  uint64_t size_ = 0;
  std::byte* pointer_ = nullptr;
  MemoryStorageType storage_type_ = MemoryStorageType::kHost;
  int32_t device_id_ = Allocator::kUnknownDeviceId;
  release_function_t release_func_ = nullptr;
  std::shared_ptr<StreamState> stream_state_;

 private:
  /// @brief Present a release failure through the shared native automatic-report policy.
  /// @param error Original failure, borrowed without rendering until presentation is enabled.
  /// @param location Inline resize call site retained for the normal logger.
  static HOLOSCAN_CORE_EXPORT void report_release_failure(
      const Error& error, std::source_location location = std::source_location::current()) noexcept;

  /// @brief Present an allocation rejection without changing the allocator's Error.
  /// @param error Original failure, borrowed until this synchronous call returns.
  /// @param size Requested size in bytes.
  /// @param storage_type Requested storage location.
  /// @param location Inline resize call site retained for the normal logger.
  static HOLOSCAN_CORE_EXPORT void report_allocation_failure(
      const Error& error, uint64_t size, MemoryStorageType storage_type,
      std::source_location location = std::source_location::current()) noexcept;
};

/**
 * @brief Move-only ownership token for a memory allocation.
 *
 * `BufferAllocation` is the production-direction handle that should be passed
 * between allocation code and Tensor/message construction. The underlying
 * `MemoryBuffer` control block can be shared with a Tensor so that dropping the
 * move-only token does not prematurely return storage while Tensor or DLPack
 * handles still exist.
 */
class BufferAllocation {
 public:
  BufferAllocation() = default;
  explicit BufferAllocation(std::shared_ptr<MemoryBuffer> memory_buffer)
      : memory_buffer_(std::move(memory_buffer)) {}

  BufferAllocation(const BufferAllocation&) = delete;
  BufferAllocation& operator=(const BufferAllocation&) = delete;
  BufferAllocation(BufferAllocation&&) noexcept = default;
  BufferAllocation& operator=(BufferAllocation&&) noexcept = default;
  ~BufferAllocation() = default;

  [[nodiscard]] static expected<BufferAllocation, Error> allocate(
      std::shared_ptr<Allocator> allocator, uint64_t size, MemoryStorageType storage_type,
      void* stream = nullptr) {
    return allocate_with_options(
        std::move(allocator), size, storage_type, MemoryBuffer::Options{}, stream);
  }

  [[nodiscard]] static expected<BufferAllocation, Error> allocate_with_options(
      std::shared_ptr<Allocator> allocator, uint64_t size, MemoryStorageType storage_type,
      MemoryBuffer::Options options, void* stream = nullptr) {
    auto memory_buffer = std::make_shared<MemoryBuffer>(options);
    auto result = memory_buffer->resize(std::move(allocator), size, storage_type, stream);
    if (!result) {
      return forward_error(result);
    }
    return BufferAllocation(std::move(memory_buffer));
  }

  [[nodiscard]] static expected<BufferAllocation, Error> allocate(
      std::shared_ptr<Allocator> allocator, uint64_t size, MemoryStorageType storage_type,
      ::cuda::stream_ref stream) {
    return allocate(std::move(allocator), size, storage_type, static_cast<void*>(stream.get()));
  }

  [[nodiscard]] static expected<BufferAllocation, Error> allocate_with_options(
      std::shared_ptr<Allocator> allocator, uint64_t size, MemoryStorageType storage_type,
      MemoryBuffer::Options options, ::cuda::stream_ref stream) {
    return allocate_with_options(
        std::move(allocator), size, storage_type, options, static_cast<void*>(stream.get()));
  }

  [[nodiscard]] static expected<BufferAllocation, Error> wrap_memory(
      void* pointer, uint64_t size, MemoryStorageType storage_type,
      MemoryBuffer::release_function_t release_func = nullptr,
      int32_t device_id = Allocator::kUnknownDeviceId) {
    auto memory_buffer = std::make_shared<MemoryBuffer>();
    auto result =
        memory_buffer->wrap_memory(pointer, size, storage_type, std::move(release_func), device_id);
    if (!result) {
      return forward_error(result);
    }
    return BufferAllocation(std::move(memory_buffer));
  }

  [[nodiscard]] explicit operator bool() const noexcept {
    return memory_buffer_ != nullptr && memory_buffer_->pointer() != nullptr;
  }

  [[nodiscard]] std::byte* data() const noexcept {
    return memory_buffer_ ? memory_buffer_->pointer() : nullptr;
  }
  [[nodiscard]] uint64_t size_bytes() const noexcept {
    return memory_buffer_ ? memory_buffer_->size() : 0;
  }
  [[nodiscard]] MemoryStorageType storage_type() const noexcept {
    return memory_buffer_ ? memory_buffer_->storage_type() : MemoryStorageType::kHost;
  }

  [[nodiscard]] bool record_release_stream(void* stream) {
    return memory_buffer_ ? memory_buffer_->set_stream(stream) : false;
  }

  [[nodiscard]] bool record_release_stream(::cuda::stream_ref stream) {
    return record_release_stream(static_cast<void*>(stream.get()));
  }

  [[nodiscard]] void* release_stream() const noexcept {
    return memory_buffer_ ? memory_buffer_->stream() : nullptr;
  }

  [[nodiscard]] MemoryBuffer* memory_buffer() const noexcept { return memory_buffer_.get(); }
  [[nodiscard]] std::shared_ptr<MemoryBuffer> share_memory_buffer() const noexcept {
    return memory_buffer_;
  }

  [[nodiscard]] expected<void, Error> release() {
    if (!memory_buffer_) {
      return {};
    }

    if (memory_buffer_.use_count() == 1) {
      auto memory_buffer = std::move(memory_buffer_);
      return memory_buffer->free_buffer();
    }

    memory_buffer_.reset();
    return {};
  }

 private:
  std::shared_ptr<MemoryBuffer> memory_buffer_;
};

}  // namespace holoscan
