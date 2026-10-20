/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <cuda/stream>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/tensor_field.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/memory/tensor_shape.hpp>
#include <holoscan/schema/schema_package.hpp>

namespace holoscan {

/// Maximum number of statically declared Tensor fields in one descriptor-backed schema.
inline constexpr std::size_t kMaximumTensorSchemaLoanFields = 64U;
static_assert(kMaximumTensorSchemaLoanFields <= std::numeric_limits<std::uint64_t>::digits,
              "Tensor schema loan fields must fit the descriptor presence mask");

/// @brief Request one field in a transactional descriptor-schema Tensor allocation.
/// @tparam SchemaT Owning generated schema type exposed on the output port.
template <typename SchemaT>
struct TensorFieldLoanRequest {
  TensorFieldSelector<SchemaT> field;  ///< Generated schema-typed field selector.
  TensorLoanRequest tensor;            ///< Shape, dtype, and optional byte strides.
};

/// @brief Request one field shape while deriving dtype from the admitted output spec.
///
/// Shape and stride views are borrowed only for the duration of \c allocate_tensors. This request
/// is valid only when the selected output field has one exact planned dtype.
///
/// @tparam SchemaT Owning generated schema type exposed on the output port.
template <typename SchemaT>
struct TensorFieldShapeRequest {
  TensorFieldSelector<SchemaT> field;                ///< Generated schema-typed field selector.
  TensorShapeView shape;                             ///< Logical extents for this allocation.
  TensorByteStridesView byte_strides{std::nullopt};  ///< Optional explicit byte strides.
};

/// @brief Create one shapes-only Tensor field allocation request.
/// @tparam SchemaT Owning generated descriptor-backed schema type.
/// @param field Generated schema-typed field selector.
/// @param shape Logical Tensor extents borrowed for the allocation call.
/// @param byte_strides Optional explicit byte strides borrowed for the allocation call.
/// @return Request whose dtype is resolved from the admitted output spec.
template <typename SchemaT>
[[nodiscard]] TensorFieldShapeRequest<SchemaT> tensor_shape(
    TensorFieldSelector<SchemaT> field, TensorShapeView shape,
    TensorByteStridesView byte_strides = std::nullopt) noexcept {
  return {.field = field, .shape = shape, .byte_strides = byte_strides};
}

/// @brief Describe one CUDA-stream-ordered field callback in a batch write.
/// @tparam SchemaT Owning generated descriptor-backed schema type.
/// @tparam Fill Concrete callback type retained without type erasure.
template <typename SchemaT, typename Fill>
struct TensorDeviceFieldWrite {
  using schema_type = SchemaT;  ///< Owning schema type used for compile-time matching.

  TensorFieldSelector<SchemaT> field;  ///< Generated schema-typed field selector.
  ::cuda::stream_ref stream;           ///< Producer stream ordering this device write.
  [[no_unique_address]] Fill fill;     ///< Concrete borrowed-guard callback.
};

/// @brief Describe one host-accessible field callback in a batch write.
/// @tparam SchemaT Owning generated descriptor-backed schema type.
/// @tparam Fill Concrete callback type retained without type erasure.
template <typename SchemaT, typename Fill>
struct TensorHostFieldWrite {
  using schema_type = SchemaT;  ///< Owning schema type used for compile-time matching.

  TensorFieldSelector<SchemaT> field;  ///< Generated schema-typed field selector.
  [[no_unique_address]] Fill fill;     ///< Concrete borrowed-guard callback.
};

/// @brief Create one CUDA-stream-ordered field callback for \c write_all.
/// @tparam SchemaT Owning generated descriptor-backed schema type.
/// @tparam Fill Callback accepting a borrowed Tensor write guard.
/// @param field Generated schema-typed field selector.
/// @param stream Producer stream ordering the device write.
/// @param fill Concrete callback retained without heap allocation or type erasure.
/// @return Device field-write entry suitable for \c TensorSchemaOutputLoan::write_all.
template <typename SchemaT, typename Fill>
  requires detail::TensorLoanFill<std::decay_t<Fill>&, TensorOutputWriteGuard>
[[nodiscard]] TensorDeviceFieldWrite<SchemaT, std::decay_t<Fill>> device_field(
    TensorFieldSelector<SchemaT> field, ::cuda::stream_ref stream, Fill&& fill) {
  return {.field = field, .stream = stream, .fill = std::forward<Fill>(fill)};
}

/// @brief Create one host-accessible field callback for \c write_all.
/// @tparam SchemaT Owning generated descriptor-backed schema type.
/// @tparam Fill Callback accepting a borrowed Tensor write guard.
/// @param field Generated schema-typed field selector.
/// @param fill Concrete callback retained without heap allocation or type erasure.
/// @return Host field-write entry suitable for \c TensorSchemaOutputLoan::write_all.
template <typename SchemaT, typename Fill>
  requires detail::TensorLoanFill<std::decay_t<Fill>&, TensorOutputWriteGuard>
[[nodiscard]] TensorHostFieldWrite<SchemaT, std::decay_t<Fill>> host_field(
    TensorFieldSelector<SchemaT> field, Fill&& fill) {
  return {.field = field, .fill = std::forward<Fill>(fill)};
}

namespace detail {

template <typename Entry, typename SchemaT>
inline constexpr bool kTensorSchemaFieldWriteFor = false;

template <typename SchemaT, typename Fill>
inline constexpr bool kTensorSchemaFieldWriteFor<TensorDeviceFieldWrite<SchemaT, Fill>, SchemaT> =
    TensorLoanFill<Fill&, TensorOutputWriteGuard>;

template <typename SchemaT, typename Fill>
inline constexpr bool kTensorSchemaFieldWriteFor<TensorHostFieldWrite<SchemaT, Fill>, SchemaT> =
    TensorLoanFill<Fill&, TensorOutputWriteGuard>;

template <typename Entry>
inline constexpr bool kTensorSchemaDeviceFieldWrite = false;

template <typename SchemaT, typename Fill>
inline constexpr bool kTensorSchemaDeviceFieldWrite<TensorDeviceFieldWrite<SchemaT, Fill>> = true;

template <typename Entry, typename SchemaT>
concept TensorSchemaFieldWriteFor = kTensorSchemaFieldWriteFor<std::remove_cvref_t<Entry>, SchemaT>;

struct TensorSchemaLoanRequest {
  std::uint16_t field_id{};
  std::uint16_t tensor_index{};
  TensorLoanRequest tensor;
};

struct TensorSchemaLoanRequestList {
  std::array<TensorSchemaLoanRequest, kMaximumTensorSchemaLoanFields> fields;
  std::uint32_t field_count{};
};

struct TensorSchemaLoanBundleStorage {
  std::array<std::optional<TensorOutputLoan>, kMaximumTensorSchemaLoanFields> fields;
  std::array<std::uint16_t, kMaximumTensorSchemaLoanFields> field_ids{};
  std::uint32_t field_count{};
};

class TensorSchemaOutputLoanAccess;

}  // namespace detail

/// @brief Hold one atomic set of unpublished provider-backed Tensor field allocations.
///
/// The bundle is move-constructible but neither copyable nor assignable. Field write operations
/// borrow the corresponding private loan without allowing individual authority to escape.
/// Destruction abandons every still-live field as one unpublished transaction.
///
/// @tparam SchemaT Owning descriptor-backed schema type.
template <typename SchemaT>
class TensorSchemaOutputLoan {
 public:
  /// @brief Transfer the complete unpublished field transaction.
  /// @param other Bundle whose authority becomes moved-from.
  TensorSchemaOutputLoan(TensorSchemaOutputLoan&& other) noexcept
      : storage_(std::move(other.storage_)) {
    other.storage_.field_count = 0U;
  }

  /// @brief Reject copying a sole multi-field publication transaction.
  TensorSchemaOutputLoan(const TensorSchemaOutputLoan&) = delete;

  /// @brief Reject copy assignment of a sole multi-field publication transaction.
  TensorSchemaOutputLoan& operator=(const TensorSchemaOutputLoan&) = delete;

  /// @brief Reject move assignment so live field authority cannot be replaced implicitly.
  TensorSchemaOutputLoan& operator=(TensorSchemaOutputLoan&&) = delete;

  /// @brief Destroy the bundle and abandon any unpublished field allocations.
  ~TensorSchemaOutputLoan() noexcept = default;

  /// @brief Return the number of fields in this transaction.
  /// @return Static schema field count, or zero after the bundle has been moved from.
  [[nodiscard]] std::size_t size() const noexcept { return storage_.field_count; }

  /// @brief Return whether this bundle has no live field inventory.
  /// @return \c true after move or for an invalid empty bundle.
  [[nodiscard]] bool empty() const noexcept { return size() == 0U; }

  /// @brief Return one field's allocation shape.
  /// @param field Generated selector for this schema.
  /// @return Borrowed shape view, or an empty view when the selector is not in this bundle.
  [[nodiscard]] TensorShapeView shape(TensorFieldSelector<SchemaT> field) const noexcept {
    const TensorOutputLoan* loan = loan_at(field);
    return loan != nullptr ? loan->shape() : TensorShapeView{};
  }

  /// @brief Return one field's concrete memory kind.
  /// @param field Generated selector for this schema.
  /// @return Planned memory kind, or \c kUnknown for an invalid selector.
  [[nodiscard]] MemoryKind memory_kind(TensorFieldSelector<SchemaT> field) const noexcept {
    const TensorOutputLoan* loan = loan_at(field);
    return loan != nullptr ? loan->memory_kind() : MemoryKind::kUnknown;
  }

  /// @brief Return one field's resolved CUDA device.
  /// @param field Generated selector for this schema.
  /// @return Planned device, or an invalid identity for an invalid selector or host field.
  [[nodiscard]] DeviceId device_id(TensorFieldSelector<SchemaT> field) const noexcept {
    const TensorOutputLoan* loan = loan_at(field);
    return loan != nullptr ? loan->device_id() : DeviceId{};
  }

  /// @brief Run one CUDA-stream-ordered field write and commit it.
  /// @tparam Fill Callable accepted by \ref TensorOutputLoan::write_and_commit.
  /// @param field Generated selector for this schema.
  /// @param producer_stream Stream on which producer writes are ordered.
  /// @param fill Callback receiving the borrowed field write guard.
  /// @return Success, or a typed selector, state, callback, or commit error.
  template <typename Fill>
    requires detail::TensorLoanFill<Fill&&, TensorOutputWriteGuard>
  [[nodiscard]] expected<void, Error> write_and_commit(TensorFieldSelector<SchemaT> field,
                                                       ::cuda::stream_ref producer_stream,
                                                       Fill&& fill) & {
    TensorOutputLoan* loan = loan_at(field);
    if (loan == nullptr) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    return loan->write_and_commit(producer_stream, std::forward<Fill>(fill));
  }

  /// @brief Run one host field write and commit it.
  /// @tparam Fill Callable accepted by \ref TensorOutputLoan::write_host_and_commit.
  /// @param field Generated selector for this schema.
  /// @param fill Callback receiving the borrowed field write guard.
  /// @return Success, or a typed selector, state, callback, or commit error.
  template <typename Fill>
    requires detail::TensorLoanFill<Fill&&, TensorOutputWriteGuard>
  [[nodiscard]] expected<void, Error> write_host_and_commit(TensorFieldSelector<SchemaT> field,
                                                            Fill&& fill) & {
    TensorOutputLoan* loan = loan_at(field);
    if (loan == nullptr) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    return loan->write_host_and_commit(std::forward<Fill>(fill));
  }

  /// @brief Preflight, run, and commit one callback for every Tensor field in this bundle.
  /// @tparam Writes Typed host or device field-write entry types for this schema.
  /// @param writes Complete generated field inventory in callback execution order.
  /// @return Success after every callback and commit, or the first selector, kind, state,
  /// callback, or commit error.
  /// @throws Any exception propagated by a fill callback. Active writes are abandoned during
  /// unwinding.
  /// @note Selector and inventory rejection occurs before write acquisition and leaves the bundle
  /// reusable. Once acquisition begins, an acquisition, callback, or commit failure abandons the
  /// transaction: acquired guards and eventual bundle destruction release its unpublished fields,
  /// and the bundle cannot be published or retried.
  /// @note All selectors, placement kinds, and write phases are checked before the first callback.
  /// The operation performs no heap allocation or type-erased dispatch on the success path.
  template <typename... Writes>
    requires(detail::TensorSchemaFieldWriteFor<Writes, SchemaT> && ...)
  // Forwarding references preserve noncopyable lvalue callbacks, but each entry participates in
  // validation, acquisition, and invocation and therefore cannot be forwarded more than once.
  [[nodiscard]] expected<void, Error> write_all(
      Writes&&... writes) & {  // NOLINT(cppcoreguidelines-missing-std-forward)
    constexpr std::size_t kFieldCount = sizeof...(Writes);
    static_assert(kFieldCount == detail::descriptor_tensor_schema_tensor_count_v<SchemaT>,
                  "write_all requires exactly one entry per schema Tensor field");
    if (storage_.field_count != kFieldCount) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "TENSOR_SCHEMA_BATCH_WRITE_BUNDLE_INVALID"});
    }

    auto selected = std::array<bool, kMaximumTensorSchemaLoanFields>{};
    std::optional<Error> validation_error;
    const auto validate = [&](const auto& write) {
      if (validation_error) {
        return;
      }
      const auto index = write.field.tensor_index;
      TensorOutputLoan* const loan = loan_at(write.field);
      if (loan == nullptr) {
        validation_error.emplace(
            batch_write_field_error(ErrorCode::kInvalidArgument,
                                    "TENSOR_SCHEMA_BATCH_WRITE_FIELD_SELECTOR_INVALID",
                                    write.field));
        return;
      }
      if (selected[index]) {
        validation_error.emplace(batch_write_field_error(
            ErrorCode::kInvalidArgument, "TENSOR_SCHEMA_BATCH_WRITE_DUPLICATE_FIELD", write.field));
        return;
      }
      selected[index] = true;
    };
    (validate(writes), ...);
    if (validation_error) {
      return make_unexpected(std::move(*validation_error));
    }

    std::array<std::optional<TensorOutputWriteGuard>, kFieldCount> guards;
    std::size_t field_index = 0U;
    expected<void, Error> result;
    const auto acquire = [&](auto& write) {
      if (!result) {
        return;
      }
      TensorOutputLoan* const loan = loan_at(write.field);
      auto writer = [&]() -> expected<TensorOutputWriteGuard, Error> {
        if constexpr (detail::kTensorSchemaDeviceFieldWrite<std::remove_cvref_t<decltype(write)>>) {
          return loan->write(write.stream);
        }
        return loan->write_host();
      }();
      if (!writer) {
        result = make_unexpected(
            batch_write_field_error(writer.error().code,
                                    "TENSOR_SCHEMA_BATCH_WRITE_FIELD_ACQUIRE_FAILED",
                                    write.field,
                                    writer.error().message));
        return;
      }
      guards[field_index].emplace(std::move(*writer));
      ++field_index;
    };
    (acquire(writes), ...);
    if (!result) {
      return result;
    }

    field_index = 0U;
    const auto fill_and_commit = [&](auto& write) {
      if (!result) {
        return;
      }
      auto& guard = *guards[field_index];
      using FillResult =
          std::invoke_result_t<decltype((write.fill)), const TensorOutputWriteGuard&>;
      if constexpr (std::same_as<FillResult, expected<void, Error>>) {
        auto filled = std::invoke(write.fill, std::as_const(guard));
        if (!filled) {
          result = make_unexpected(std::move(filled).error());
          return;
        }
      } else {
        std::invoke(write.fill, std::as_const(guard));
      }
      result = std::move(guard).commit();
      ++field_index;
    };
    (fill_and_commit(writes), ...);
    return result;
  }

 private:
  [[nodiscard]] static Error batch_write_field_error(ErrorCode code, std::string_view token,
                                                     TensorFieldSelector<SchemaT> field,
                                                     std::string_view detail = {}) {
    std::string message{token};
    message += " (field_id=" + std::to_string(field.field_id) +
               ", tensor_index=" + std::to_string(field.tensor_index) + ")";
    if (!detail.empty()) {
      message += ": ";
      message += detail;
    }
    return Error{code, std::move(message)};
  }

  explicit TensorSchemaOutputLoan(detail::TensorSchemaLoanBundleStorage storage) noexcept
      : storage_(std::move(storage)) {}

  [[nodiscard]] TensorOutputLoan* loan_at(TensorFieldSelector<SchemaT> field) noexcept {
    if (field.tensor_index >= storage_.field_count ||
        storage_.field_ids[field.tensor_index] != field.field_id) {
      return nullptr;
    }
    auto& loan = storage_.fields[field.tensor_index];
    return loan ? &*loan : nullptr;
  }

  [[nodiscard]] const TensorOutputLoan* loan_at(TensorFieldSelector<SchemaT> field) const noexcept {
    if (field.tensor_index >= storage_.field_count ||
        storage_.field_ids[field.tensor_index] != field.field_id) {
      return nullptr;
    }
    const auto& loan = storage_.fields[field.tensor_index];
    return loan ? &*loan : nullptr;
  }

  detail::TensorSchemaLoanBundleStorage storage_;

  friend class detail::TensorSchemaOutputLoanAccess;
};

namespace detail {

class TensorSchemaOutputLoanAccess {
 public:
  template <typename SchemaT>
  [[nodiscard]] static TensorSchemaOutputLoan<SchemaT> make(
      TensorSchemaLoanBundleStorage storage) noexcept {
    return TensorSchemaOutputLoan<SchemaT>{std::move(storage)};
  }

  template <typename SchemaT>
  [[nodiscard]] static TensorSchemaLoanBundleStorage release(
      TensorSchemaOutputLoan<SchemaT> loan) noexcept {
    TensorSchemaLoanBundleStorage storage = std::move(loan.storage_);
    loan.storage_.field_count = 0U;
    return storage;
  }
};

}  // namespace detail

}  // namespace holoscan
