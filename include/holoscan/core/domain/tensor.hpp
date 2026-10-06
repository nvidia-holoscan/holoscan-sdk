/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime_api.h>
#include <dlpack/dlpack.h>
#include <cuda/stream>

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/domain/dlpack_utils.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

namespace holoscan {

/// @brief Describe one bounded Tensor view carried by schema metadata.
///
/// The schema is authoritative for shape, dtype, strides, and logical byte size. A transport
/// descriptor separately identifies the backing storage and its memory kind.
struct TensorViewMetadata {
  /// Maximum Tensor rank represented inline.
  static constexpr std::size_t kMaxRank = 8U;

  std::uint32_t rank{};                        ///< Tensor rank.
  DLDataType dtype{};                          ///< DLPack element type.
  DLDevice device{};                           ///< DLPack device encoded by the schema.
  bool has_explicit_byte_strides{};            ///< Whether \ref byte_strides is explicit.
  std::uint64_t byte_size{};                   ///< Logical Tensor byte size.
  std::array<std::int64_t, kMaxRank> shape{};  ///< Valid entries are in `[0, rank)`.
  /// Valid entries are in `[0, rank)` when \ref has_explicit_byte_strides is true.
  std::array<std::int64_t, kMaxRank> byte_strides{};
};

static_assert(std::is_standard_layout_v<TensorViewMetadata>);
static_assert(std::is_trivially_copyable_v<TensorViewMetadata>);

namespace detail {
class TensorAccess;

// Keep the typed Tensor access table closed over the scalar types that the public API admits.
// Unsupported C++ types use the zero-lane sentinel and therefore cannot be accepted merely because
// their size or signedness happens to match one of these DLPack encodings.
template <typename T>
struct TensorDTypeTraits {
  static constexpr DLDataType kDType{0U, 0U, 0U};
};

template <>
struct TensorDTypeTraits<std::int8_t> {
  static constexpr DLDataType kDType{kDLInt, 8U, 1U};
};

template <>
struct TensorDTypeTraits<std::uint8_t> {
  static constexpr DLDataType kDType{kDLUInt, 8U, 1U};
};

template <>
struct TensorDTypeTraits<std::int16_t> {
  static constexpr DLDataType kDType{kDLInt, 16U, 1U};
};

template <>
struct TensorDTypeTraits<std::uint16_t> {
  static constexpr DLDataType kDType{kDLUInt, 16U, 1U};
};

template <>
struct TensorDTypeTraits<std::int32_t> {
  static constexpr DLDataType kDType{kDLInt, 32U, 1U};
};

template <>
struct TensorDTypeTraits<std::uint32_t> {
  static constexpr DLDataType kDType{kDLUInt, 32U, 1U};
};

template <>
struct TensorDTypeTraits<std::int64_t> {
  static constexpr DLDataType kDType{kDLInt, 64U, 1U};
};

template <>
struct TensorDTypeTraits<std::uint64_t> {
  static constexpr DLDataType kDType{kDLUInt, 64U, 1U};
};

template <>
struct TensorDTypeTraits<float> {
  static constexpr DLDataType kDType{kDLFloat, 32U, 1U};
};

template <>
struct TensorDTypeTraits<double> {
  static constexpr DLDataType kDType{kDLFloat, 64U, 1U};
};

template <>
struct TensorDTypeTraits<bool> {
  static constexpr DLDataType kDType{kDLBool, static_cast<std::uint8_t>(sizeof(bool) * 8U), 1U};
};

template <>
struct TensorDTypeTraits<std::complex<float>> {
  static constexpr DLDataType kDType{kDLComplex, 64U, 1U};
};

template <>
struct TensorDTypeTraits<std::complex<double>> {
  static constexpr DLDataType kDType{kDLComplex, 128U, 1U};
};
}  // namespace detail

// The DLPack version the Tensor implementation produces on versioned export. Tracks the DLPack
// header version (a versioned DLManagedTensorContext carries this by default).
constexpr uint32_t HOLOSCAN_DLPACK_IMPL_VERSION_MAJOR{DLPACK_MAJOR_VERSION};
constexpr uint32_t HOLOSCAN_DLPACK_IMPL_VERSION_MINOR{DLPACK_MINOR_VERSION};

/**
 * @brief Tensor class.
 *
 * A Tensor is a multi-dimensional array of elements of a single data type.
 *
 * The Tensor class is a wrapper around the DLManagedTensorContext struct that holds the
 * DLManagedTensor object.
 * (https://dmlc.github.io/dlpack/latest/c_api.html#c.DLManagedTensor).
 *
 * This class provides a primary interface to access Tensor data and is interoperable with other
 * frameworks that support DLManagedTensor. Copies share backing storage and its monotonic access
 * authority; moves transfer one backing claim. Concurrent const metadata/read access is permitted
 * while the backing remains alive. Mutable access is a momentary authorization check and does not
 * itself provide synchronization or a writer lease; callers must not race it with other access.
 */
class Tensor {
 public:
  /// @brief Create an empty Tensor handle with no backing allocation.
  Tensor() = default;

  /// @brief Copy this handle while retaining one claim on the shared backing allocation.
  /// @param other Tensor whose context and backing allocation are shared.
  HOLOSCAN_CORE_EXPORT Tensor(const Tensor& other) noexcept;

  /// @brief Replace this handle with a retained claim on another Tensor's shared backing
  /// allocation.
  /// @param other Tensor whose context and backing allocation are shared.
  /// @return This Tensor after releasing its previous claim and retaining \p other.
  HOLOSCAN_CORE_EXPORT Tensor& operator=(const Tensor& other) noexcept;

  /// @brief Move this handle without changing the backing allocation's live-handle count.
  /// @param other Tensor whose claim is transferred; it becomes empty after the move.
  HOLOSCAN_CORE_EXPORT Tensor(Tensor&& other) noexcept;

  /// @brief Replace this handle by transferring another handle's backing claim.
  /// @param other Tensor whose claim is transferred; it becomes empty after the move.
  /// @return This Tensor after releasing its previous claim and adopting \p other.
  HOLOSCAN_CORE_EXPORT Tensor& operator=(Tensor&& other) noexcept;

  /**
   * @brief Construct a new Tensor from an existing DLManagedTensorContext.
   *
   * @param ctx A shared pointer to the DLManagedTensorContext to be used in Tensor construction.
   *        Passed by value so callers can hand off rvalue handles without an extra retain/release.
   * @warning This low-level constructor treats an unflagged context as locally owned writable
   *          storage. Use the DLPack import factories for externally owned storage; they establish
   *          permanent read-only authority and producer-deleter ownership.
   */
  HOLOSCAN_CORE_EXPORT explicit Tensor(std::shared_ptr<DLManagedTensorContext> ctx);

  /**
   * @brief Construct a new Tensor from an existing DLManagedTensor pointer.
   *
   * Equivalent to make_tensor_from_dlpack(), but throws on an invalid import instead of returning
   * an error. Imported storage is permanently read-only. Prefer the factory at call sites that can
   * handle failure without exceptions.
   *
   * @param dl_managed_tensor_ptr A pointer to the DLManagedTensor to be used in Tensor
   * construction.
   * @throws RuntimeError if the pointer is null.
   */
  HOLOSCAN_CORE_EXPORT explicit Tensor(DLManagedTensor* dl_managed_tensor_ptr);

  /**
   * @brief Construct a new Tensor from an existing DLManagedTensorVersioned pointer.
   *
   * The DLPack version and flags (e.g. read-only / copied) carried by the versioned tensor are
   * preserved, and imported storage is permanently narrowed to read-only authority. Equivalent to
   * make_tensor_from_dlpack_versioned(), but throws on an invalid import instead of returning an
   * error.
   *
   * @param dl_managed_tensor_ver_ptr A pointer to the DLManagedTensorVersioned to be used in Tensor
   * construction.
   * @throws RuntimeError if the pointer is null or the DLPack major version is newer than
   * supported.
   */
  HOLOSCAN_CORE_EXPORT explicit Tensor(DLManagedTensorVersioned* dl_managed_tensor_ver_ptr);

  /// @brief Release this handle's claim on the backing allocation.
  virtual ~Tensor() { release_handle(); }

  /**
   * @brief Return a read-only pointer to the underlying data.
   *
   * The returned pointer includes the DLPack byte offset and therefore names the first logical
   * Tensor byte, not necessarily the backing allocation base. The pointer is a momentary borrowed
   * view; it does not extend the Tensor lifetime and must not be used to mutate the backing
   * allocation. Empty, null-data, and overflowing base-plus-offset representations return null.
   *
   * @return The read-only data pointer, or null if no data is present.
   */
  [[nodiscard]] HOLOSCAN_CORE_EXPORT const void* data() const noexcept;

  /// @brief Return a typed read-only pointer when the Tensor dtype is exactly compatible.
  ///
  /// Compatibility requires a scalar, single-lane DLPack dtype that exactly matches \p U and a
  /// pointer aligned for \p U. This method performs no conversion and returns null for empty,
  /// null-data, incompatible-dtype, vector-lane, or misaligned tensors.
  ///
  /// @tparam U Scalar element type requested by the caller.
  /// @return A typed read-only pointer, or null when the view is not structurally valid.
  template <typename U>
  [[nodiscard]] const U* data_as() const noexcept {
    using Value = std::remove_cv_t<U>;
    static_assert(!std::is_void_v<Value>, "Tensor::data_as<void>() is not supported");
    static_assert(!std::is_volatile_v<U>, "Tensor::data_as<volatile U>() is not supported");
    const void* pointer = data();
    if (pointer == nullptr || !dtype_matches<Value>() ||
        reinterpret_cast<std::uintptr_t>(pointer) % alignof(Value) != 0U) {
      return nullptr;
    }
    return static_cast<const U*>(pointer);
  }

  /**
   * @brief Get a mutable pointer to the underlying data.
   *
   * @return The data pointer, or an Error if the Tensor is empty, has a null data pointer,
   *         or is read-only (per its DLPack flags).
   */
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void*, Error> mutable_data() &;

  /// @brief Return a typed mutable pointer when dtype, alignment, and access authority permit it.
  ///
  /// This lvalue-only operation does not copy or convert data. Structural failures return
  /// `kInvalidArgument`; valid data whose backing is permanently read-only or currently protected
  /// by a read lease returns `kPermissionDenied`. The pointer is a momentary borrow and does not
  /// extend the Tensor lifetime.
  ///
  /// @tparam U Scalar element type requested by the caller.
  /// @return A typed mutable pointer, or an Error describing structural or authority failure.
  template <typename U>
  [[nodiscard]] expected<U*, Error> mutable_data_as() & {
    static_assert(!std::is_void_v<U>, "Tensor::mutable_data_as<void>() is not supported");
    static_assert(!std::is_const_v<U> && !std::is_volatile_v<U>,
                  "Tensor::mutable_data_as<U>() requires an unqualified element type");
    if (data_as<U>() == nullptr) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                   "Tensor dtype, data pointer, or alignment is invalid"});
    }
    auto pointer = mutable_data();
    if (!pointer) {
      return make_unexpected(pointer.error());
    }
    return static_cast<U*>(pointer.value());
  }

  /**
   * @brief Get the device information of the Tensor.
   *
   * @return The device information of the Tensor, or the CPU device sentinel for an empty Tensor.
   */
  DLDevice device() const {
    return dl_ctx_ == nullptr ? DLDevice{kDLCPU, 0} : dl_ctx_->tensor.dl_tensor.device;
  }

  /**
   * @brief Get the Tensor's data type information.
   *
   * For details of the DLDataType struct see the DLPack documentation:
   * https://dmlc.github.io/dlpack/latest/c_api.html#_CPPv410DLDataType
   *
   * @return The DLPack dtype, or a zero-initialized dtype for an empty Tensor.
   */
  DLDataType dtype() const {
    return dl_ctx_ == nullptr ? DLDataType{} : dl_ctx_->tensor.dl_tensor.dtype;
  }

  /**
   * @brief Get the shape of the Tensor data.
   *
   * @return The vector containing the Tensor's shape.
   */
  HOLOSCAN_CORE_EXPORT std::vector<int64_t> shape() const;

  /**
   * @brief Get an allocation-free view of the Tensor's shape.
   *
   * The returned span is valid while the Tensor's internal DLPack context remains alive.
   * Unlike shape(), this method does not materialize a std::vector.
   *
   * @return A span containing the Tensor's shape.
   */
  HOLOSCAN_CORE_EXPORT std::span<const int64_t> shape_span() const;

  /**
   * @brief Get the strides of the Tensor data, in bytes.
   *
   * Note that, unlike `DLTensor.strides` (which are in elements), the strides this method returns
   * are in number of bytes, not elements (to be consistent with NumPy/CuPy's strides).
   *
   * @return The vector containing the Tensor's strides, in bytes.
   */
  HOLOSCAN_CORE_EXPORT std::vector<int64_t> strides() const;

  /**
   * @brief Get an allocation-free view of the Tensor's DLPack strides, in elements.
   *
   * DLPack strides are in elements, not bytes (in contrast to strides(), which returns byte
   * strides). For contiguous tensors whose DLPack metadata uses implicit strides
   * (`DLTensor.strides == nullptr`), this returns an empty span. Use strides() when byte strides
   * or implicit contiguous strides need to be materialized.
   *
   * @return A span containing explicit DLPack strides in elements, or an empty span for implicit
   *         contiguous strides.
   */
  HOLOSCAN_CORE_EXPORT std::span<const int64_t> dlpack_strides_span() const;

  /**
   * @brief Check if the tensor a has contiguous, row-major memory layout.
   *
   * @return True if the tensor is contiguous, False otherwise.
   */
  HOLOSCAN_CORE_EXPORT bool is_contiguous() const;

  /// @brief Return the number of logical elements in the Tensor.
  ///
  /// Vector lanes form one logical element; for example, a `float4` value counts as one element.
  /// Empty or malformed metadata, negative extents, and an element-count overflow return zero. A
  /// rank-zero DLPack scalar contains one element.
  ///
  /// @return The element count, or zero when it cannot be represented.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT int64_t size() const noexcept;

  /**
   * @brief Get the number of dimensions of the Tensor.
   *
   * @return The number of dimensions, or zero for an empty Tensor.
   */
  int32_t ndim() const { return dl_ctx_ == nullptr ? 0 : dl_ctx_->tensor.dl_tensor.ndim; }

  /**
   * @brief Get the byte offset from the Tensor's data pointer.
   *
   * This is the DLPack `DLTensor::byte_offset` value.
   *
   * @return The byte offset from the Tensor's data pointer, or zero for an empty Tensor.
   */
  uint64_t byte_offset() const {
    return dl_ctx_ == nullptr ? 0U : dl_ctx_->tensor.dl_tensor.byte_offset;
  }

  /**
   * @brief Get the DLPack flags of the Tensor.
   *
   * @return The DLPack `DLManagedTensorVersioned::flags` value (0 for an empty Tensor).
   */
  uint64_t dlpack_flags() const { return dl_ctx_ == nullptr ? 0U : dl_ctx_->tensor.flags; }

  /**
   * @brief Return whether the shared backing authority is permanently read-only.
   *
   * The private authority control block is authoritative; the DLPack flag is its external
   * projection and cannot independently upgrade storage after an alias has been narrowed.
   *
   * @return True if the Tensor's effective maximum authority is read-only.
   */
  [[nodiscard]] HOLOSCAN_CORE_EXPORT bool is_read_only() const noexcept;

  /**
   * @brief Whether the Tensor's data is a DLPack-reported copy (per its DLPack flags).
   *
   * @return True if the DLPACK_FLAG_BITMASK_IS_COPIED bit is set.
   */
  bool is_dlpack_copied() const { return (dlpack_flags() & DLPACK_FLAG_BITMASK_IS_COPIED) != 0U; }

  /// @brief Return the byte size of one logical Tensor element.
  ///
  /// Vector lanes contribute to the item size; for example, a `float4` element occupies 16 bytes.
  /// Empty Tensors and dtypes with zero bits or lanes return zero. The 64-bit result preserves the
  /// complete DLPack lane product instead of truncating wide vectorized element sizes.
  ///
  /// @return The item size in bytes, or zero for empty or invalid dtype metadata.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT uint64_t itemsize() const noexcept;

  /// @brief Return the total byte size of the logical Tensor view.
  ///
  /// Empty or malformed metadata and a byte-count overflow return zero. This value excludes the
  /// `byte_offset` bytes preceding the first logical element.
  ///
  /// @return The logical Tensor size in bytes, or zero when it cannot be represented.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT int64_t nbytes() const noexcept;

  /**
   * @brief Export this Tensor through the legacy, unversioned DLPack protocol.
   *
   * The returned object retains the backing allocation until its DLPack deleter runs. Because
   * legacy DLPack cannot represent read-only authority, exporting a permanently read-only Tensor
   * or a Tensor protected by an active read lease fails rather than silently widening access.
   *
   * @return The exported DLPack object, `kInvalidArgument` for an empty Tensor,
   *         `kNotSupported` for read-only authority, or `kResourceExhausted` if export metadata
   *         allocation fails.
   */
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<DLManagedTensor*, Error> to_dlpack() const;

  /**
   * @brief Export this Tensor through the versioned DLPack protocol.
   *
   * The returned object retains the backing allocation until its DLPack deleter runs. Permanent
   * read-only authority and a finite read lease active at export time are represented by
   * `DLPACK_FLAG_BITMASK_READ_ONLY` in the export flags. The flag remains set on that export even
   * after a finite local read lease ends because the exported owner may outlive the lease.
   *
   * @return The exported DLPack object, `kInvalidArgument` for an empty Tensor, or
   *         `kResourceExhausted` if export metadata allocation fails.
   */
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<DLManagedTensorVersioned*, Error>
  to_dlpack_versioned() const;

  /**
   * @brief Set the CUDA stream that last made this Tensor's data ready to read.
   *
   * Producers should call this with the stream that produced or last wrote the Tensor data. DLPack
   * export uses this stream for producer-to-consumer synchronization when a consumer provides a
   * stream via `__dlpack__(stream=...)`. This is intentionally separate from
   * set_deallocation_stream(), which controls when memory can be returned to an allocator.
   *
   * @param stream The CUDA stream that last made this Tensor's data readable.
   * @return true if the stream was recorded, false if the tensor's memory reference does not track
   *         readiness streams (e.g. a plain imported DLPack tensor).
   * @warning The caller must keep a non-null stream alive until the Tensor has been exported or
   *          the readiness stream has been replaced or cleared. CUDA stream handles cannot be
   *          safely queried after destruction.
   */
  HOLOSCAN_CORE_EXPORT bool set_readiness_stream(cudaStream_t stream);
  bool set_readiness_stream(::cuda::stream_ref stream) {
    return set_readiness_stream(stream.get());
  }

  /**
   * @brief Make this Tensor's data safe for a DLPack consumer stream.
   *
   * The DLPack object itself does not carry CUDA stream state. Python `__dlpack__(stream=...)`
   * callers provide the stream they will use to consume the tensor; this asks the Tensor's memory
   * reference to insert any needed event/wait dependency from the stream that last made the data
   * ready. Imported DLPack tensors only have this contract if the importer or memory owner recorded
   * readiness information; otherwise synchronization remains the producer's responsibility.
   *
   * @param consumer_stream The CUDA stream that will consume the Tensor data.
   * @return Success, or an Error if readiness synchronization is unsupported (kNotImplemented) or
   *         CUDA event synchronization fails.
   * @warning The caller must keep a non-null consumer stream alive until its enqueued Tensor work
   *          has completed.
   */
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> synchronize_for_consumer_stream(
      cudaStream_t consumer_stream) const;
  [[nodiscard]] expected<void, Error> synchronize_for_consumer_stream(
      ::cuda::stream_ref consumer_stream) const {
    return synchronize_for_consumer_stream(consumer_stream.get());
  }

  /**
   * @brief Return whether this Tensor can synchronize producer readiness with consumer streams.
   *
   * This is false for plain imported DLPack tensors backed only by a producer-owned deleter,
   * because Holoscan does not know which producer stream must be waited on.
   */
  [[nodiscard]] HOLOSCAN_CORE_EXPORT bool has_readiness_stream_contract() const noexcept;

  /**
   * @brief Block the calling host thread until the Tensor's readiness stream completes.
   *
   * This only orders prior CUDA work against the CPU. It does not copy device memory to host memory
   * and does not make device pointers CPU-addressable.
   *
   * @return Success, or an Error if CUDA event synchronization fails.
   */
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> synchronize_for_host() const;

  /**
   * @brief Enqueue a copy of this contiguous Tensor into caller-owned host memory.
   *
   * This method does not allocate. For CUDA-backed tensors it first makes `stream` wait for the
   * Tensor readiness stream, then enqueues a device-to-host copy on `stream`. Stream-aware backing
   * storage records that stream as its retirement boundary; imported storage without that
   * capability completes the copy before return so its producer deleter cannot run early. For
   * CPU/host-backed tensors it synchronizes producer readiness on the host and performs a host
   * memcpy. Only contiguous Tensor layouts are supported.
   *
   * @param destination Host-visible destination pointer.
   * @param destination_bytes Number of bytes available at `destination`.
   * @param stream CUDA stream used for CUDA-backed Tensor copies.
   * @return Success, or an Error if arguments are invalid or CUDA enqueue fails.
   */
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> copy_to_host_async(
      void* destination, std::size_t destination_bytes, cudaStream_t stream) const;
  [[nodiscard]] expected<void, Error> copy_to_host_async(void* destination,
                                                         std::size_t destination_bytes,
                                                         ::cuda::stream_ref stream) const {
    return copy_to_host_async(destination, destination_bytes, stream.get());
  }

  /**
   * @brief Copy this contiguous Tensor into caller-owned host memory and wait for completion.
   *
   * Blocking counterpart to copy_to_host_async(). It does not allocate destination storage.
   */
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> copy_to_host(
      void* destination, std::size_t destination_bytes, cudaStream_t stream = nullptr) const;
  [[nodiscard]] expected<void, Error> copy_to_host(void* destination, std::size_t destination_bytes,
                                                   ::cuda::stream_ref stream) const {
    return copy_to_host(destination, destination_bytes, stream.get());
  }

  /// @brief Allocate an owning contiguous copy on the Tensor's current device.
  ///
  /// CUDA copies are ordered on \p stream and the returned Tensor records that stream as its
  /// readiness and deallocation coordinate. Host copies complete before return. The returned
  /// Tensor has local writable authority and carries the DLPack copied flag.
  ///
  /// @param stream CUDA stream used for CUDA-backed copies.
  /// @return An owning copy, or an Error for invalid metadata, unsupported storage, allocation, or
  /// CUDA copy failure.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<Tensor, Error> copy(
      cudaStream_t stream = nullptr) const;
  [[nodiscard]] expected<Tensor, Error> copy(::cuda::stream_ref stream) const {
    return copy(stream.get());
  }

  /// @brief Allocate an owning contiguous copy on a requested DLPack device.
  ///
  /// The first implementation accepts CPU, CUDA device, CUDA pinned-host, and CUDA managed
  /// destinations. CUDA work is ordered on \p stream; host destinations are ready before return.
  /// Cross-device CUDA copies use peer-copy semantics and report the CUDA failure when peer access
  /// is unavailable.
  ///
  /// @param target Destination DLPack device and ordinal.
  /// @param stream CUDA stream used for CUDA-backed copies.
  /// @return An owning copy, or an Error for invalid metadata/device, unsupported storage,
  /// allocation, or CUDA copy failure.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<Tensor, Error> copy_to(
      DLDevice target, cudaStream_t stream = nullptr) const;
  [[nodiscard]] expected<Tensor, Error> copy_to(DLDevice target, ::cuda::stream_ref stream) const {
    return copy_to(target, stream.get());
  }

  /**
   * @brief Set the CUDA stream for stream-aware memory deallocation.
   *
   * For sink operators that don't emit data, this should be called with the operator's working
   * CUDA stream so allocators (like BlockMemoryPool) defer memory reuse until GPU operations on the
   * stream complete. Only works for tensors whose memory is managed by a native Holoscan
   * TensorMemoryReference; for tensors imported via DLPack (e.g. from CuPy or PyTorch) this returns
   * false unless the imported tensor was wrapped with a native memory reference.
   *
   * @param stream The CUDA stream that last accessed this tensor's data.
   * @return true if the stream was set successfully, false otherwise.
   * @warning The caller must keep a non-null stream alive until the last Tensor handle releases
   *          the backing storage and its stream-ordered deallocation has been enqueued.
   */
  HOLOSCAN_CORE_EXPORT bool set_deallocation_stream(cudaStream_t stream);
  bool set_deallocation_stream(::cuda::stream_ref stream) {
    return set_deallocation_stream(stream.get());
  }

 private:
  template <typename U>
  [[nodiscard]] bool dtype_matches() const noexcept {
    using Value = std::remove_cv_t<U>;
    if (dl_ctx_ == nullptr) {
      return false;
    }
    constexpr DLDataType expected = detail::TensorDTypeTraits<Value>::kDType;
    if constexpr (expected.lanes == 0U) {
      return false;
    }
    const DLDataType actual = dl_ctx_->tensor.dl_tensor.dtype;
    return actual.code == expected.code && actual.bits == expected.bits &&
           actual.lanes == expected.lanes;
  }

  HOLOSCAN_CORE_EXPORT void retain_handle() const noexcept;
  HOLOSCAN_CORE_EXPORT void release_handle() const noexcept;

  friend class detail::TensorAccess;
  std::shared_ptr<DLManagedTensorContext> dl_ctx_;  ///< The DLManagedTensorContext object.
};

/**
 * @brief Check whether a Tensor exactly matches descriptor-backed schema view metadata.
 *
 * The comparison is allocation-free and covers rank, shape, dtype, device, logical byte size,
 * and the presence and values of explicit byte strides. Empty or structurally invalid Tensors and
 * metadata with an unsupported rank do not match.
 *
 * @param tensor Tensor whose current read-only metadata is inspected.
 * @param metadata Schema-derived Tensor view metadata to compare.
 * @return True only when every represented Tensor view field matches.
 */
[[nodiscard]] HOLOSCAN_CORE_EXPORT bool tensor_matches(const Tensor& tensor,
                                                       const TensorViewMetadata& metadata) noexcept;

/**
 * @brief Create a Tensor from an unversioned DLManagedTensor, validating the producer handle.
 *
 * The unversioned source is treated as a current-version tensor and permanently narrowed to
 * read-only authority because imported ownership does not establish a local writer claim. On
 * success the returned Tensor owns the producer's DLPack deleter (invoked when the last reference
 * is released).
 *
 * @param dl_managed_tensor_ptr A pointer to the DLManagedTensor to import.
 * @return The imported Tensor, or an Error (kInvalidArgument) if the pointer is null.
 */
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<Tensor, Error> make_tensor_from_dlpack(
    DLManagedTensor* dl_managed_tensor_ptr);

/**
 * @brief Create a Tensor from a versioned DLManagedTensorVersioned, validating the producer handle.
 *
 * The DLPack version and flags are preserved and imported storage is permanently narrowed to
 * read-only authority. Per the DLPack contract a consumer must reject a newer major version (the
 * struct layout may have changed); rejection leaves ownership with the producer. On success the
 * returned Tensor owns the producer's DLPack deleter.
 *
 * @param dl_managed_tensor_ver_ptr A pointer to the DLManagedTensorVersioned to import.
 * @return The imported Tensor, or an Error (kInvalidArgument) if the pointer is null or the
 * DLPack major version is newer than supported.
 */
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<Tensor, Error> make_tensor_from_dlpack_versioned(
    DLManagedTensorVersioned* dl_managed_tensor_ver_ptr);

}  // namespace holoscan
