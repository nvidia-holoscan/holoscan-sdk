/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_UTILS_MATX_ALLOCATOR_HPP
#define HOLOSCAN_UTILS_MATX_ALLOCATOR_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>

#include <holoscan/core/resources/gxf/allocator.hpp>
#include <holoscan/core/resources/gxf/cuda_allocator.hpp>
#include <holoscan/logger/logger.hpp>

namespace holoscan {

/**
 * @brief Wrap a holoscan::Allocator for use with MatX's custom allocator interface.
 *
 * MatX (v0.9.3+) detects custom allocators via SFINAE: any type providing
 * `allocate(size_t) -> void*` and `deallocate(void*, size_t) -> void` is accepted.
 * This class bridges the holoscan::Allocator API to satisfy that interface.
 *
 * The adapter supports stream-aware allocation when the underlying allocator is a
 * CudaAllocator (e.g., RMMAllocator, StreamOrderedAllocator). For non-CudaAllocator
 * types (e.g., BlockMemoryPool), synchronous allocation is used, but stream-aware
 * deallocation is still leveraged via the GXF-level `free(ptr, stream)` when a
 * stream is bound.
 *
 * Allocator behavior matrix (behavior depends on whether a CUDA stream is
 * passed to MatXAllocator, not on the allocator class itself):
 *
 * | Allocator                            | Stream | Allocation     | Deallocation     |
 * |--------------------------------------|--------|----------------|------------------|
 * | RMMAllocator / StreamOrderedAllocator | Yes   | Async          | Async            |
 * | RMMAllocator / StreamOrderedAllocator | No    | Sync           | Sync             |
 * | BlockMemoryPool                       | Yes   | Sync           | Deferred (event) |
 * | BlockMemoryPool                       | No    | Sync           | Sync             |
 * | UnboundedAllocator                    | Any   | Sync           | Sync             |
 *
 * "Stream" = whether a non-null cudaStream_t is passed to MatXAllocator.
 *
 * @note Rows 1-2 refer to the same allocator type; the distinction is whether
 *       MatXAllocator is constructed with a non-null stream.
 *
 * @note "Sync" in the Allocation/Deallocation columns means *not stream-ordered*
 *       (no cudaMallocAsync/cudaFreeAsync). It does NOT mean that each
 *       allocation forces a GPU sync. For BlockMemoryPool, allocation from
 *       the preallocated pool is CPU bookkeeping only (mutex + stack).
 *
 * @note This class does NOT own the Allocator. The caller must ensure the
 *       Allocator outlives the MatXAllocator and any tensors allocated through it.
 *
 * @note Async allocation (CudaAllocator + stream) only supports device memory
 *       (`MemoryStorageType::kDevice`). Constructing with a non-kDevice storage
 *       type and a CudaAllocator + stream throws `std::invalid_argument`.
 *
 * Example usage:
 * ```cpp
 * // Inside an operator's compute() method, where allocator_ is a
 * // Parameter<std::shared_ptr<Allocator>> registered in setup():
 * holoscan::MatXAllocator matx_alloc(allocator_.get(), cuda_stream);
 * auto tensor = matx::make_tensor<float>({1024, 1024}, matx_alloc);
 *
 * // Direct construction via MetaParameter's implicit conversion also works:
 * holoscan::MatXAllocator matx_alloc2(allocator_, cuda_stream);
 * ```
 *
 * @note MatX's `make_tensor` with a custom allocator does NOT accept a CUDA
 *       stream parameter. To enable stream-ordered allocation, bind the stream
 *       when constructing the MatXAllocator. Use `with_stream()` to create
 *       allocators for different streams without reconstructing from scratch.
 *
 * @since 4.0.0
 */
class MatXAllocator {
 public:
  /**
   * @brief Construct a MatXAllocator with full control over memory type and stream.
   *
   * @param allocator     Pointer to a holoscan::Allocator (must not be null).
   * @param storage_type  Memory type for allocations (default: kDevice). When
   *                      using a CudaAllocator with a stream, only kDevice is
   *                      supported.
   * @param stream        Optional CUDA stream for async allocation/deallocation.
   *                      When non-null and the allocator supports it, async APIs
   *                      are used.
   * @throws std::invalid_argument if @p allocator is null.
   * @throws std::invalid_argument if a CudaAllocator + stream is used with a
   *         non-kDevice storage type (async allocation is device-only).
   */
  explicit MatXAllocator(Allocator* allocator,
                         MemoryStorageType storage_type = MemoryStorageType::kDevice,
                         cudaStream_t stream = nullptr)
      : allocator_(allocator),
        storage_type_(storage_type),
        stream_(stream),
        cuda_allocator_(dynamic_cast<CudaAllocator*>(allocator)) {
    if (!allocator_) {
      throw std::invalid_argument("MatXAllocator: allocator must not be null");
    }
    // Async allocation (CudaAllocator + stream) only supports device memory.
    // The GXF CudaAllocator::allocate_async_abi has no storage-type parameter
    // and always allocates device memory. Fail loudly to prevent silent misuse.
    if (cuda_allocator_ && stream_ && storage_type_ != MemoryStorageType::kDevice) {
      throw std::invalid_argument(
          "MatXAllocator: async allocation (CudaAllocator + stream) only "
          "supports kDevice storage type; use the synchronous path (no "
          "stream) for other memory types");
    }
  }

  /**
   * @brief Construct with device memory and a CUDA stream (convenience overload).
   *
   * Equivalent to `MatXAllocator(allocator, MemoryStorageType::kDevice, stream)`.
   *
   * @param allocator  Pointer to a holoscan::Allocator (must not be null).
   * @param stream     CUDA stream for async allocation/deallocation.
   */
  MatXAllocator(Allocator* allocator, cudaStream_t stream)
      : MatXAllocator(allocator, MemoryStorageType::kDevice, stream) {}

  /**
   * @brief Construct from a `std::shared_ptr<Allocator>`.
   *
   * Extract the raw pointer via `shared_ptr::get()` and delegate to the
   * `Allocator*` constructor. The MatXAllocator does NOT retain or extend
   * the lifetime of the shared_ptr — only the raw pointer is stored.
   * The caller must ensure the Allocator outlives the MatXAllocator.
   *
   * This overload enables ergonomic construction from
   * `Parameter<std::shared_ptr<Allocator>>`:
   * @code
   * holoscan::MatXAllocator alloc(allocator_.get());   // shared_ptr
   * holoscan::MatXAllocator alloc(allocator_);          // implicit conversion
   * @endcode
   *
   * @param allocator     Shared pointer to a holoscan::Allocator (must not
   *                      be null).
   * @param storage_type  Memory type for allocations (default: kDevice).
   * @param stream        Optional CUDA stream for async operations.
   * @throws std::invalid_argument if the underlying pointer is null.
   * @throws std::invalid_argument if a CudaAllocator + stream is used with
   *         a non-kDevice storage type.
   */
  explicit MatXAllocator(const std::shared_ptr<Allocator>& allocator,
                         MemoryStorageType storage_type = MemoryStorageType::kDevice,
                         cudaStream_t stream = nullptr)
      : MatXAllocator(allocator.get(), storage_type, stream) {}

  /**
   * @brief Construct from a `std::shared_ptr<Allocator>` with a CUDA stream
   *        (convenience overload).
   *
   * Equivalent to
   * `MatXAllocator(allocator.get(), MemoryStorageType::kDevice, stream)`.
   *
   * @param allocator  Shared pointer to a holoscan::Allocator (must not
   *                   be null).
   * @param stream     CUDA stream for async allocation/deallocation.
   */
  MatXAllocator(const std::shared_ptr<Allocator>& allocator, cudaStream_t stream)
      : MatXAllocator(allocator.get(), MemoryStorageType::kDevice, stream) {}

  // Copyable and movable (lightweight, non-owning).
  MatXAllocator(const MatXAllocator&) = default;
  MatXAllocator& operator=(const MatXAllocator&) = default;
  MatXAllocator(MatXAllocator&&) = default;
  MatXAllocator& operator=(MatXAllocator&&) = default;

  /**
   * @brief Allocate memory (satisfy MatX's allocator interface).
   *
   * Dispatch strategy:
   *   1. If size is 0, return nullptr (no allocation needed).
   *   2. If the underlying allocator is a CudaAllocator and a stream is bound,
   *      use allocate_async(size, stream).
   *   3. Otherwise, use allocate(size, storage_type) (synchronous).
   *
   * @param size  Number of bytes to allocate. Zero returns nullptr without error.
   * @return Pointer to allocated memory.
   * @throws std::bad_alloc if allocation fails.
   */
  [[nodiscard]] void* allocate(size_t size) {
    if (size == 0) {
      return nullptr;
    }

    void* ptr = nullptr;
    if (cuda_allocator_ && stream_) {
      // Path 1: CudaAllocator with stream — use async allocation.
      ptr =
          static_cast<void*>(cuda_allocator_->allocate_async(static_cast<uint64_t>(size), stream_));
    } else {
      // Path 2: Synchronous allocation with explicit memory type.
      ptr = static_cast<void*>(allocator_->allocate(static_cast<uint64_t>(size), storage_type_));
    }
    if (!ptr) {
      HOLOSCAN_LOG_ERROR("MatXAllocator: failed to allocate {} bytes", size);
      throw std::bad_alloc();
    }
    return ptr;
  }

  /**
   * @brief Deallocate memory (satisfy MatX's allocator interface).
   *
   * This method is `noexcept` to be safe when called from destructors
   * (e.g., when a MatX tensor is destroyed). Any internal errors are logged
   * but not propagated.
   *
   * Dispatch strategy:
   *   1. If the underlying allocator is a CudaAllocator and a stream is bound,
   *      call the GXF-level free_async(ptr, stream) for status checking, with
   *      fallback to synchronous free(ptr) on failure.
   *   2. Else if a stream is bound (e.g., BlockMemoryPool), call GXF-level
   *      free(ptr, stream) for stream-aware deferred deallocation. Fall back
   *      to synchronous free(ptr) if unavailable.
   *   3. Otherwise, use free(ptr) (synchronous).
   *
   * @note The fallback from stream-aware free to synchronous free is safe for
   *       all current Holoscan allocators: BlockMemoryPool uses CUDA-event-based
   *       deferred free and UnboundedAllocator's default free_abi(ptr, stream)
   *       delegates to free_abi(ptr). If a future allocator differentiates these,
   *       revisit this logic.
   *
   * @note The underlying allocator's `free()` implementation must not throw.
   *       If it does, the exception is caught and logged but the pointer may
   *       be leaked. All current Holoscan allocators satisfy this requirement.
   *
   * @note If the GXF allocator handle is unavailable, deallocation falls back
   *       to `allocator_->free()` as a best-effort path. In this case, status
   *       cannot be queried from GXF.
   *
   * @param ptr   Pointer to memory to deallocate (null is a no-op).
   * @param size  Size of allocation (required by MatX interface, unused).
   */
  void deallocate(void* ptr, [[maybe_unused]] size_t size) noexcept {
    if (!ptr) {
      return;
    }

    try {
      if (cuda_allocator_ && stream_) {
        // Path 1: CudaAllocator with stream — use GXF-level free_async for
        // status. The Holoscan CudaAllocator::free_async wrapper is void and
        // silently swallows errors. Call the GXF API directly to get
        // Expected<void> for error detection.
        auto* gxf_cuda_alloc = cuda_allocator_->get();
        if (gxf_cuda_alloc) {
          auto result = gxf_cuda_alloc->free_async(static_cast<nvidia::byte*>(ptr), stream_);
          if (result) {
            return;
          }
          HOLOSCAN_LOG_WARN(
              "MatXAllocator: free_async failed, "
              "synchronizing stream before sync free fallback");
        } else {
          HOLOSCAN_LOG_WARN(
              "MatXAllocator: GXF CudaAllocator unavailable, "
              "synchronizing stream before sync free fallback");
        }
        // Synchronize the stream before falling back to synchronous free
        // to prevent use-after-free if GPU operations are still in flight.
        cudaStreamSynchronize(stream_);
        allocator_->free(static_cast<nvidia::byte*>(ptr));
      } else if (stream_) {
        // Path 2: Non-CudaAllocator with stream (e.g., BlockMemoryPool).
        // Access GXF-level allocator for stream-aware free(ptr, stream).
        auto* gxf_allocator = allocator_->get();
        if (gxf_allocator) {
          auto maybe_result =
              gxf_allocator->free(static_cast<nvidia::byte*>(ptr), static_cast<void*>(stream_));
          if (maybe_result) {
            return;
          }
          HOLOSCAN_LOG_WARN(
              "MatXAllocator: stream-aware free failed, "
              "synchronizing stream before sync free fallback");
        } else {
          HOLOSCAN_LOG_WARN(
              "MatXAllocator: GXF allocator unavailable, "
              "synchronizing stream before sync free fallback");
        }
        // Synchronize the stream before falling back to synchronous free
        // to prevent use-after-free if GPU operations are still in flight.
        cudaStreamSynchronize(stream_);
        allocator_->free(static_cast<nvidia::byte*>(ptr));
      } else {
        // Path 3: No stream — synchronous free.
        allocator_->free(static_cast<nvidia::byte*>(ptr));
      }
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("MatXAllocator: deallocate failed: {}", e.what());
    } catch (...) {
      HOLOSCAN_LOG_ERROR("MatXAllocator: deallocate failed with unknown exception");
    }
  }

  /// @brief Return the underlying Holoscan allocator.
  Allocator* allocator() const noexcept { return allocator_; }

  /// @brief Return the configured memory storage type.
  MemoryStorageType storage_type() const noexcept { return storage_type_; }

  /// @brief Return the bound CUDA stream (nullptr if none).
  cudaStream_t stream() const noexcept { return stream_; }

  /**
   * @brief Create a copy of this allocator bound to a different CUDA stream.
   *
   * Return a new MatXAllocator sharing the same underlying Allocator and
   * storage type, but associated with a different stream. Useful in
   * multi-stream pipelines where the same allocator serves multiple streams.
   *
   * @param stream  The CUDA stream to bind to the new allocator.
   * @return A new MatXAllocator bound to the given stream.
   * @throws std::invalid_argument if the new stream + storage_type
   *         combination is invalid (see primary constructor).
   */
  MatXAllocator with_stream(cudaStream_t stream) const {
    return MatXAllocator(allocator_, storage_type_, stream);
  }

 private:
  Allocator* allocator_;            ///< non-owning pointer to the Holoscan allocator
  MemoryStorageType storage_type_;  ///< memory type for synchronous allocations
  cudaStream_t stream_;             ///< optional CUDA stream for async operations
  CudaAllocator* cuda_allocator_;   ///< cached dynamic_cast (nullptr if not CudaAllocator)
};

}  // namespace holoscan

#endif  // HOLOSCAN_UTILS_MATX_ALLOCATOR_HPP
