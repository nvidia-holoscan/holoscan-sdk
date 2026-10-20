/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include <holoscan/core/domain/tensor.hpp>

#include <holomq/cuda/stream_lifetime.hpp>

namespace holoscan::detail {

/// @brief Provide an optional in-tree extension for borrowed consumer-stream snapshots.
/// Keep this separate from the installed TensorMemoryReference ABI while retirement evolves.
class HOLOSCAN_CORE_EXPORT TensorConsumerCompletion {
 public:
  /// @brief Destroy this optional completion extension.
  virtual ~TensorConsumerCompletion();
  /// @brief Snapshot submitted work without waiting for GPU completion.
  /// @param stream Borrowed consumer stream that is still alive.
  virtual void record_consumer_completion(cudaStream_t stream) noexcept = 0;
};

/// Optional private stream-lifetime registration, without changing TensorMemoryReference ABI.
class HOLOSCAN_CORE_EXPORT TensorStreamLifetimeConsumer {
 public:
  virtual ~TensorStreamLifetimeConsumer();
  virtual bool register_consumer_stream(
      cudaStream_t stream,
      const std::shared_ptr<const holomq::cuda::StreamLifetime>& lifetime) noexcept = 0;
};

/// @brief Identify a live native or Python callback whose stream can still be fenced.
/// Lifecycle callbacks have no assigned stream and require conservative retirement.
class HOLOSCAN_CORE_EXPORT TensorConsumerScope {
 public:
  /// @brief Enter a callback on this thread.
  /// @param stream Live callback stream, or no stream for lifecycle callbacks.
  explicit TensorConsumerScope(
      std::optional<cudaStream_t> stream = std::nullopt,
      const std::shared_ptr<const holomq::cuda::StreamLifetime>* lifetime = nullptr,
      bool capturing = false) noexcept;
  /// @brief Restore the enclosing callback scope.
  ~TensorConsumerScope();
  TensorConsumerScope(const TensorConsumerScope&) = delete;
  TensorConsumerScope& operator=(const TensorConsumerScope&) = delete;
  /// @brief Return the live stream of the current callback, if assigned.
  [[nodiscard]] static std::optional<cudaStream_t> stream() noexcept;
  /// @brief Return whether this thread is executing a scoped callback.
  [[nodiscard]] static bool active() noexcept;
  [[nodiscard]] static bool capturing() noexcept;
  [[nodiscard]] static const std::shared_ptr<const holomq::cuda::StreamLifetime>& lifetime(
      cudaStream_t stream) noexcept;

 private:
  TensorConsumerScope* previous_{};
  std::optional<cudaStream_t> stream_;
  const std::shared_ptr<const holomq::cuda::StreamLifetime>* lifetime_{};
  bool capturing_{};
};

/// Initial mutation authority assigned by trusted Tensor construction paths.
enum class TensorAuthority : std::uint8_t { kLocalWritable = 0, kReadOnly = 1 };

/// Allocation-free metadata snapshot used by trusted runtime validation paths.
struct TensorAccessSnapshot {
  bool initialized = false;
  bool valid = false;
  const void* data = nullptr;
  std::uint64_t byte_offset = 0;
  std::uint64_t byte_size = 0;
  std::uint64_t byte_span = 0;
  DLDevice device{kDLCPU, 0};
};

/**
 * @brief Move-only internal claim that temporarily protects Tensor backing from mutable access.
 *
 * The lease owns one explicit TensorMemoryReference claim in addition to its shared metadata
 * context. This prevents a pool-backed allocation from being recycled while the runtime still
 * holds an activation-scoped read view. This type is an implementation boundary for runtime
 * selection and transport code. It is not an installed application API.
 */
class TensorReadLease {
 public:
  TensorReadLease() = default;
  TensorReadLease(const TensorReadLease&) = delete;
  TensorReadLease& operator=(const TensorReadLease&) = delete;
  HOLOSCAN_CORE_EXPORT TensorReadLease(TensorReadLease&& other) noexcept;
  HOLOSCAN_CORE_EXPORT TensorReadLease& operator=(TensorReadLease&& other) noexcept;
  HOLOSCAN_CORE_EXPORT ~TensorReadLease();

  [[nodiscard]] explicit operator bool() const noexcept { return state_ != nullptr; }

  // The runtime arms this only after successful CUDA input admission. Release snapshots the
  // submitted use while the consumer stream remains alive, before dropping the memory claim.
  void set_consumer_stream(cudaStream_t stream) noexcept { consumer_stream_ = stream; }

 private:
  friend class TensorAccess;
  TensorReadLease(std::shared_ptr<TensorAccessState> state,
                  std::shared_ptr<DLManagedTensorContext> context) noexcept
      : state_(std::move(state)), context_(std::move(context)) {
    if (context_ != nullptr && context_->memory_reference != nullptr) {
      context_->memory_reference->retain_tensor_handle();
    }
  }
  void release() noexcept;

  std::optional<cudaStream_t> consumer_stream_;
  std::shared_ptr<TensorAccessState> state_;
  std::shared_ptr<DLManagedTensorContext> context_;
};

/**
 * @brief Internal friend adapter for trusted Tensor materialization and read-lease operations.
 *
 * Application code must use Tensor's public metadata and access methods. This adapter replaces
 * the former public mutable `dl_ctx()` escape hatch without exposing an authority-upgrade path.
 */
class TensorAccess {
 public:
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static const std::shared_ptr<DLManagedTensorContext>& context(
      const Tensor& tensor) noexcept;
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static bool register_consumer_stream(
      const Tensor& tensor, cudaStream_t stream,
      const std::shared_ptr<const holomq::cuda::StreamLifetime>& lifetime) noexcept;
  /// @brief Snapshot a borrowed consumer stream through the optional in-tree extension.
  /// @param tensor Tensor whose backing may implement the extension.
  /// @param stream Consumer stream on which work has already been submitted.
  HOLOSCAN_CORE_EXPORT static void record_consumer_completion(const Tensor& tensor,
                                                              cudaStream_t stream) noexcept;
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static TensorAccessSnapshot snapshot(
      const Tensor& tensor) noexcept;
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static cudaStream_t readiness_stream(
      const Tensor& tensor) noexcept;
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static bool tracks_deallocation_stream(
      const Tensor& tensor) noexcept;
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static Tensor from_context(
      std::shared_ptr<DLManagedTensorContext> context, TensorAuthority authority);
  HOLOSCAN_CORE_EXPORT static void replace_context(Tensor& tensor,
                                                   std::shared_ptr<DLManagedTensorContext> context,
                                                   TensorAuthority authority);
  HOLOSCAN_CORE_EXPORT static void narrow_read_only(Tensor& tensor) noexcept;
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<TensorReadLease, Error> acquire_read_lease(
      const Tensor& tensor);
};

}  // namespace holoscan::detail
