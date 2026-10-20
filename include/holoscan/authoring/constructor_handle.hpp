/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/authoring/parameter_materializer.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

namespace holoscan::authoring {

/// @file
/// @brief Typed, owning, process-local constructor handles for registered Operator rows.
///
/// The declarations in this header have Prototype API stability. A constructor handle carries one
/// provider-owned C++ object into a hand-authored registry construction thunk. It is not a
/// Parameter: it never enters a Parameter schema, configuration binding, parameter update, or
/// portable plan artifact.

/// @brief Identify one constructor-handle interface type (a C++ type).
///
/// Each provider defines exactly one descriptor object per interface type, in the library that owns
/// that type, and exposes it only through a typed accessor returning \ref ConstructorHandleType.
/// Identity is the descriptor's address; the stable name is diagnostic and lookup metadata and
/// never authorizes a conversion by itself. The object must have process lifetime.
struct ConstructorHandleDescriptor {
  /// Stable provider-facing type name, for example `holoscan.<provider>.<Type>`.
  std::string_view name;
  /// Short human-readable summary for discovery and generated documentation.
  std::string_view description;
};

/// @brief Bind one exact C++ interface type to its provider-owned descriptor.
/// @tparam T Exact interface type carried by handles of this type. `const T` and `T` are distinct.
///
/// Define the descriptor and this binding together in one provider source file, and return the
/// binding from an exported accessor. Consumers then name the interface type and the descriptor in
/// a single declaration, so a handle created for one type cannot be labeled as another.
template <typename T>
class ConstructorHandleType {
 public:
  static_assert(std::is_object_v<T> && !std::is_array_v<T>,
                "a constructor handle type must name a non-array object type");

  /// @brief Bind \p descriptor to \p T.
  /// @param descriptor Provider-owned descriptor with process lifetime.
  explicit constexpr ConstructorHandleType(const ConstructorHandleDescriptor& descriptor) noexcept
      : descriptor_(&descriptor) {}

  /// @brief Return the bound descriptor.
  [[nodiscard]] constexpr const ConstructorHandleDescriptor& descriptor() const noexcept {
    return *descriptor_;
  }

 private:
  const ConstructorHandleDescriptor* descriptor_;
};

namespace detail {
/// @brief Return the calling process's identifier for constructor-handle origin checks.
[[nodiscard]] HOLOSCAN_CORE_EXPORT std::int64_t constructor_handle_process_id() noexcept;

/// @brief Build a constructor-handle error as `<token> <key>=<value>`.
[[nodiscard]] inline Error constructor_handle_error(ErrorCode code, std::string_view token,
                                                    std::string_view key, std::string_view value) {
  std::string message(token);
  message += ' ';
  message += key;
  message += '=';
  message += value;
  return Error{code, std::move(message)};
}
}  // namespace detail

/// @brief Own one typed C++ object for a registered Operator construction call.
///
/// Copies share ownership of the original object and preserve its control block; recovery never
/// creates a second owner from a raw pointer. A handle also records the process that created it so
/// that an object inherited through `fork()` is rejected rather than admitted by address.
class ConstructorHandle {
 public:
  /// @brief Create an empty handle, which no construction call accepts.
  ConstructorHandle() = default;

  /// @brief Wrap one shared owner under its exact interface type.
  /// @param type Typed descriptor binding for \p T.
  /// @param owner Non-null shared owner; its control block is retained unchanged.
  /// @return The handle, or `AUTHORING_CONSTRUCTOR_HANDLE_NULL` for a null owner.
  template <typename T>
  [[nodiscard]] static expected<ConstructorHandle, Error> from_shared(
      const ConstructorHandleType<T>& type, std::shared_ptr<T> owner) {
    if (!owner) {
      return make_unexpected(detail::constructor_handle_error(ErrorCode::kInvalidArgument,
                                                              "AUTHORING_CONSTRUCTOR_HANDLE_NULL",
                                                              "type",
                                                              type.descriptor().name));
    }
    ConstructorHandle handle;
    handle.owner_ = std::shared_ptr<const void>(std::move(owner));
    handle.type_ = &type.descriptor();
    handle.origin_process_ = detail::constructor_handle_process_id();
    return handle;
  }

  /// @brief Recover an independent shared owner after checking the exact descriptor.
  /// @param type Typed descriptor binding expected by the caller.
  /// @return A shared owner of the original object, or a type-mismatch or empty-handle error.
  template <typename T>
  [[nodiscard]] expected<std::shared_ptr<T>, Error> as_shared(
      const ConstructorHandleType<T>& type) const {
    if (!owner_) {
      return make_unexpected(detail::constructor_handle_error(ErrorCode::kInvalidArgument,
                                                              "AUTHORING_CONSTRUCTOR_HANDLE_NULL",
                                                              "type",
                                                              type.descriptor().name));
    }
    if (type_ != &type.descriptor()) {
      return make_unexpected(
          detail::constructor_handle_error(ErrorCode::kIncompatible,
                                           "AUTHORING_CONSTRUCTOR_HANDLE_TYPE_MISMATCH",
                                           "type",
                                           type.descriptor().name));
    }
    // The descriptor binds exactly one T, so this restores the type and constness it was wrapped
    // as.
    return std::const_pointer_cast<T>(std::static_pointer_cast<const std::remove_cv_t<T>>(owner_));
  }

  /// @brief Return the descriptor this handle was created with, or `nullptr` when empty.
  [[nodiscard]] const ConstructorHandleDescriptor* type() const noexcept { return type_; }

  /// @brief Return whether this handle owns an object.
  [[nodiscard]] explicit operator bool() const noexcept { return owner_ != nullptr; }

  /// @brief Return whether this handle was created in the calling process.
  [[nodiscard]] bool created_in_current_process() const noexcept {
    return origin_process_ == detail::constructor_handle_process_id();
  }

  /// @brief Return the type-erased owner, preserving the original control block.
  [[nodiscard]] const std::shared_ptr<const void>& owner() const noexcept { return owner_; }

 private:
  std::shared_ptr<const void> owner_;
  const ConstructorHandleDescriptor* type_{};
  std::int64_t origin_process_{};
};

/// @brief Copy one handle into an opaque box for a language-binding exchange.
///
/// A box is how a provider's language binding hands a handle to Holoscan's binding without sharing
/// binding-library internals. The box owns one independent handle copy; destroy it with
/// \ref release_constructor_handle_box. Its address is valid only in this process.
/// @param handle Non-empty handle to copy.
/// @return Owning box pointer.
/// @throws std::bad_alloc if the box cannot be allocated.
[[nodiscard]] HOLOSCAN_CORE_EXPORT void* box_constructor_handle(const ConstructorHandle& handle);

/// @brief Destroy one box created by \ref box_constructor_handle, releasing its handle copy.
/// @param box Box pointer, or `nullptr`.
HOLOSCAN_CORE_EXPORT void release_constructor_handle_box(void* box) noexcept;

/// @brief Return the identity of this Holoscan Core instance's box format.
///
/// Exchanges record this value next to each box so that a receiver can reject a box created by a
/// different, independently loaded copy of Holoscan Core.
[[nodiscard]] HOLOSCAN_CORE_EXPORT const void* constructor_handle_box_identity() noexcept;

/// @brief Lifecycle callback carried by a built-in callback constructor handle.
using ConstructorCallback = std::function<void()>;

/// @brief Return the built-in handle type for `void()` callbacks, named `holoscan.VoidCallback`.
///
/// Rows declare callback inputs, such as a device's start and stop actions, with this type and
/// read them with \ref OperatorConstructionInputs::callback. C++ callers supply a
/// `std::shared_ptr<const ConstructorCallback>`. Holoscan's Python binding adapts any Python
/// callable passed for such an input; that callback acquires the GIL when it runs and fails with
/// `CONSTRUCTOR_CALLBACK_INTERPRETER_UNAVAILABLE` once the interpreter has shut down.
[[nodiscard]] HOLOSCAN_CORE_EXPORT const ConstructorHandleType<const ConstructorCallback>&
callback_handle_type() noexcept;

/// @brief Supply one named constructor handle to a registered Operator construction call.
struct ConstructorHandleAssignment {
  std::string_view name;    ///< Canonical handle-input name declared by the row.
  ConstructorHandle value;  ///< Non-empty typed owner.
};

/// @brief Declare one named constructor-handle input accepted by a registered Operator row.
struct ConstructorHandleInput {
  std::string_view name;         ///< Canonical input name; distinct from every value field name.
  std::string_view description;  ///< Short human-readable summary.
  const ConstructorHandleDescriptor* type{};  ///< Exact expected interface descriptor.
  bool required{true};                        ///< Whether a call must supply this input.
};

/// @brief Borrow the complete named inputs of one registered Operator construction call.
///
/// Views are valid only for the duration of the construction thunk. Thunks must copy any owner they
/// keep, for example by capturing the `std::shared_ptr` returned by \ref require_shared by value.
struct OperatorConstructionInputs {
  std::span<const ParameterAssignment> values;           ///< Ordinary named value assignments.
  std::span<const ConstructorHandleAssignment> handles;  ///< Validated named constructor handles.

  /// @brief Return the handle supplied under \p name, or `nullptr` when it was omitted.
  [[nodiscard]] const ConstructorHandle* find(std::string_view name) const noexcept {
    for (const ConstructorHandleAssignment& assignment : handles) {
      if (assignment.name == name) {
        return &assignment.value;
      }
    }
    return nullptr;
  }

  /// @brief Recover the owner supplied under \p name.
  /// @param name Canonical handle-input name.
  /// @param type Typed descriptor binding expected for \p name.
  /// @return The owner, or `AUTHORING_CONSTRUCTOR_HANDLE_REQUIRED` or a type-mismatch error.
  template <typename T>
  [[nodiscard]] expected<std::shared_ptr<T>, Error> require_shared(
      std::string_view name, const ConstructorHandleType<T>& type) const {
    const ConstructorHandle* handle = find(name);
    if (handle == nullptr) {
      return make_unexpected(detail::constructor_handle_error(
          ErrorCode::kInvalidArgument, "AUTHORING_CONSTRUCTOR_HANDLE_REQUIRED", "field", name));
    }
    return handle->as_shared(type);
  }

  /// @brief Return the callback supplied under \p name, or an empty function when it was omitted.
  ///
  /// The returned function shares ownership of the supplied callback, so the caller may store and
  /// copy it freely; copies never duplicate the underlying callable.
  /// @param name Canonical callback-input name declared with \ref callback_handle_type.
  /// @return The callback, an empty function, or a type-mismatch error.
  [[nodiscard]] expected<ConstructorCallback, Error> callback(std::string_view name) const {
    auto owner = optional_shared(name, callback_handle_type());
    if (!owner) {
      return make_unexpected(std::move(owner).error());
    }
    if (!*owner) {
      return ConstructorCallback{};
    }
    return ConstructorCallback{[callback = std::move(*owner)] { (*callback)(); }};
  }

  /// @brief Recover the owner supplied under \p name, or `nullptr` when it was omitted.
  /// @param name Canonical optional handle-input name.
  /// @param type Typed descriptor binding expected for \p name.
  /// @return The owner or `nullptr`, or a type-mismatch error.
  template <typename T>
  [[nodiscard]] expected<std::shared_ptr<T>, Error> optional_shared(
      std::string_view name, const ConstructorHandleType<T>& type) const {
    const ConstructorHandle* handle = find(name);
    if (handle == nullptr) {
      return std::shared_ptr<T>{};
    }
    return handle->as_shared(type);
  }
};

}  // namespace holoscan::authoring
