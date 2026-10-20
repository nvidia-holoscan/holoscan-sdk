/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime_api.h>
#include <dlpack/dlpack.h>

#include <cstdint>
#include <memory>
#include <vector>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

namespace holoscan {

class Tensor;

namespace detail {
class TensorAccess;
class TensorAccessState;
}  // namespace detail

/**
 * @brief Native base class for Tensor-owned memory/lifetime references.
 *
 * This is the production-facing hook for Holoscan-owned Tensor storage, pool allocation
 * references, imported DLPack ownership, and CUDA stream/event release state. A
 * DLManagedTensorContext holds one of these to keep the backing memory alive and to optionally
 * order producer readiness against consumer streams.
 *
 * The default implementation is intentionally non-capable: it owns nothing stream-aware and
 * reports no readiness contract. Imported DLPack wrappers (DLManagedMemoryBuffer /
 * DLManagedMemoryBufferVersioned) use these defaults because they only know the producer's
 * deleter, not the stream that made the data ready. Holoscan-owned references that can record and
 * wait on CUDA events should override the relevant methods. Implementations used by concurrently
 * copied or retired Tensor handles must synchronize their mutable state internally.
 */
class TensorMemoryReference {
 public:
  virtual ~TensorMemoryReference() = default;

  /// @brief Retain one live Tensor-handle claim on this backing allocation.
  ///
  /// Pool-backed references may override this no-throw hook to prevent reuse while a Tensor handle
  /// or exported DLPack owner still names the allocation. The default is a no-op for references
  /// whose shared ownership already supplies the complete lifetime contract. Overrides must be
  /// thread-safe, allocation-free, and non-blocking after resource warm-up.
  virtual void retain_tensor_handle() noexcept {}

  /// @brief Release one live Tensor-handle claim on this backing allocation.
  ///
  /// Every successful retain is paired with exactly one release. Implementations must remain
  /// no-throw because Tensor destruction and move assignment invoke this hook. Overrides must be
  /// thread-safe, allocation-free, and non-blocking after resource warm-up.
  virtual void release_tensor_handle() noexcept {}

  /// Record the CUDA stream after which the backing memory may be safely reused by an allocator.
  virtual bool set_deallocation_stream(cudaStream_t stream) {
    (void)stream;
    return false;
  }

  /// @brief Return whether this memory reference tracks CUDA use before storage reuse.
  [[nodiscard]] virtual bool tracks_deallocation_stream() const noexcept { return false; }

  /**
   * @brief Record the CUDA stream that last made this Tensor's data ready to read.
   *
   * The readiness stream is independent from the deallocation stream: readiness drives
   * producer-to-consumer synchronization for DLPack export, while deallocation drives allocator
   * reuse after the final owner releases the memory.
   */
  virtual bool set_readiness_stream(cudaStream_t stream) {
    (void)stream;
    return false;
  }

  /**
   * @brief Return whether this memory reference can order producer readiness against consumers.
   *
   * Imported DLPack ownership wrappers generally do not know the producer stream that made the
   * tensor data ready, so they return false; Tensor then reports unsupported readiness
   * synchronization distinctly instead of treating it as a successful no-op.
   */
  [[nodiscard]] virtual bool has_readiness_stream_contract() const noexcept { return false; }

  /// @brief Return the retained producer-readiness stream when its identity is locally meaningful.
  ///
  /// Runtime-owned references may expose this source-private ordering coordinate so a bounded
  /// materialization path can enqueue work after the producer without inventing another stream.
  /// Imported cross-process references return null because CUDA stream handles are process-local.
  ///
  /// @return Borrowed producer stream, or null when no process-local stream identity is available.
  [[nodiscard]] virtual cudaStream_t readiness_stream() const noexcept { return nullptr; }

  /**
   * @brief Make this Tensor's data safe to consume on the provided CUDA stream.
   *
   * References that know their producer/ready stream should record an event on it and make
   * `consumer_stream` wait. References that do not own stream state leave the default no-op.
   */
  [[nodiscard]] virtual expected<void, Error> synchronize_for_consumer(
      cudaStream_t consumer_stream) {
    (void)consumer_stream;
    return {};
  }

  /**
   * @brief Block the calling host thread until this Tensor's producer stream is ready.
   *
   * This is an ordering primitive only; it does not make device memory CPU-addressable.
   */
  [[nodiscard]] virtual expected<void, Error> synchronize_for_host() { return {}; }
};

/**
 * @brief Context object holding a versioned DLPack tensor and a reference to its backing memory.
 *
 * The Tensor class wraps this struct. The wrapped tensor is a DLManagedTensorVersioned so the
 * Tensor can carry DLPack version and flags (e.g. read-only / copied). `memory_reference` keeps
 * the data alive for as long as the context exists; it typically holds a DLManagedMemoryBuffer /
 * DLManagedMemoryBufferVersioned whose destructor invokes the source tensor's DLPack deleter, or a
 * Holoscan-owned stream-aware reference. `dl_shape` / `dl_strides` own the shape/stride arrays the
 * wrapped DLTensor points at. A private control block makes authority and transient read-lease
 * state common to every Tensor wrapper over this context.
 *
 * Vendored from nvidia::gxf::DLManagedTensorContext to remove the GXF dependency.
 */
struct DLManagedTensorContext {
  /// @brief Create a context with a private local-writable access-authority control block.
  HOLOSCAN_CORE_EXPORT DLManagedTensorContext();

  DLManagedTensorVersioned tensor{DLPackVersion{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION},
                                  nullptr,
                                  nullptr,
                                  0,
                                  DLTensor{}};  ///< The versioned tensor to wrap.
  /// Native Holoscan lifetime/stream hook keeping the backing memory alive.
  std::shared_ptr<TensorMemoryReference> memory_reference;

  std::vector<int64_t> dl_shape;    ///< Shape of the DLTensor (owned).
  std::vector<int64_t> dl_strides;  ///< Strides of the DLTensor in elements (owned).

 private:
  friend class Tensor;
  friend class detail::TensorAccess;
  std::shared_ptr<detail::TensorAccessState> access_state_;
};

/**
 * @brief Wraps the deleter of a DLManagedTensor.
 *
 * Used as a DLManagedTensorContext::memory_reference. When the last reference to the context is
 * released, the buffer is destroyed and the wrapped DLManagedTensor's deleter is called. Imported
 * DLPack tensors only know the producer deleter, so this keeps the default (non-capable)
 * TensorMemoryReference readiness behavior.
 *
 * Vendored from nvidia::gxf::DLManagedMemoryBuffer to remove the GXF dependency.
 */
class DLManagedMemoryBuffer : public TensorMemoryReference {
 public:
  HOLOSCAN_CORE_EXPORT explicit DLManagedMemoryBuffer(DLManagedTensor* self);
  ~DLManagedMemoryBuffer() override;

 private:
  DLManagedTensor* self_ = nullptr;
};

/**
 * @brief Wraps the deleter of a DLManagedTensorVersioned.
 *
 * Versioned (DLPack >= 1.0) counterpart of DLManagedMemoryBuffer.
 */
class DLManagedMemoryBufferVersioned : public TensorMemoryReference {
 public:
  HOLOSCAN_CORE_EXPORT explicit DLManagedMemoryBufferVersioned(DLManagedTensorVersioned* self);
  ~DLManagedMemoryBufferVersioned() override;

 private:
  DLManagedTensorVersioned* self_ = nullptr;
};

}  // namespace holoscan
