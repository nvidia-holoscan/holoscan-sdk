/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <dlpack/dlpack.h>

#include <optional>
#include <span>
#include <string>
#include <string_view>

#include <holoscan/core/port_types.hpp>

namespace holoscan {

class OperatorSpec;

namespace detail {
class SyntheticOperatorBinding;
class TensorOutputPortBindingAccess;
class GraphPortOwnershipAccess;
struct TensorOutputBindingOps;

struct TensorOutputBinding {
  void* state{};
  const TensorOutputBindingOps* ops{};
  std::optional<DLDataType> tensor_dtype;
  std::span<const std::optional<DLDataType>> tensor_field_dtypes;
};

}  // namespace detail

/// @brief Provide the non-templated authoring identity shared by typed inputs.
///
/// A port member is stable, noncopyable, and nonmovable. Its name is cold authoring metadata and
/// may allocate during construction. It does not own a queue, backend endpoint, or execution
/// session, and it exposes no testing load operation.
class InputBase {
 public:
  /// @brief Reject copying a stable input declaration.
  /// @param other Declaration that would otherwise be copied.
  InputBase(const InputBase& other) = delete;

  /// @brief Reject copy assignment of a stable input declaration.
  /// @param other Declaration that would otherwise replace this one.
  InputBase& operator=(const InputBase& other) = delete;

  /// @brief Reject moving a stable input declaration.
  /// @param other Declaration that would otherwise be moved.
  InputBase(InputBase&& other) = delete;

  /// @brief Reject move assignment of a stable input declaration.
  /// @param other Declaration that would otherwise replace this one.
  InputBase& operator=(InputBase&& other) = delete;

  /// @brief Destroy an unbound authoring input declaration.
  virtual ~InputBase() = default;

  /// @brief Return the cold declaration name.
  /// @return View valid until this port is destroyed.
  [[nodiscard]] std::string_view name() const noexcept { return name_; }

 protected:
  /// @brief Construct an unnamed input declaration.
  InputBase() = default;

  /// @brief Construct an input declaration that owns its name.
  /// @param name Declaration name copied into cold authoring storage.
  explicit InputBase(std::string_view name) : name_(name) {}

 private:
  std::string name_;
  mutable const void* graph_owner_token_{};

  friend class PortRef;
  friend class OperatorSpec;
  friend class detail::SyntheticOperatorBinding;
  friend class detail::GraphPortOwnershipAccess;
};

/// @brief Provide the non-templated authoring identity shared by typed outputs.
///
/// An output declaration is a stable member slot rather than an endpoint owner. It carries no
/// queue, scheduler, or testing retrieval API. Its cold name may allocate during construction.
class OutputBase {
 public:
  /// @brief Reject copying a stable output declaration.
  /// @param other Declaration that would otherwise be copied.
  OutputBase(const OutputBase& other) = delete;

  /// @brief Reject copy assignment of a stable output declaration.
  /// @param other Declaration that would otherwise replace this one.
  OutputBase& operator=(const OutputBase& other) = delete;

  /// @brief Reject moving a stable output declaration.
  /// @param other Declaration that would otherwise be moved.
  OutputBase(OutputBase&& other) = delete;

  /// @brief Reject move assignment of a stable output declaration.
  /// @param other Declaration that would otherwise replace this one.
  OutputBase& operator=(OutputBase&& other) = delete;

  /// @brief Destroy an unbound authoring output declaration.
  virtual ~OutputBase() = default;

  /// @brief Return the cold declaration name.
  /// @return View valid until this port is destroyed.
  [[nodiscard]] std::string_view name() const noexcept { return name_; }

 protected:
  /// @brief Construct an unnamed output declaration.
  OutputBase() = default;

  /// @brief Construct an output declaration that owns its name.
  /// @param name Declaration name copied into cold authoring storage.
  explicit OutputBase(std::string_view name) : name_(name) {}

  /// @brief Return the private shaped-allocation binding for a Tensor-backed output.
  /// @return Mutable binding used only by exact typed output specializations.
  [[nodiscard]] detail::TensorOutputBinding& tensor_output_binding() noexcept {
    return tensor_binding_;
  }

 private:
  std::string name_;
  mutable const void* graph_owner_token_{};
  detail::TensorOutputBinding tensor_binding_{};

  friend class PortRef;
  friend class OperatorSpec;
  friend class detail::SyntheticOperatorBinding;
  friend class detail::TensorOutputPortBindingAccess;
  friend class detail::GraphPortOwnershipAccess;
};

/// @brief Borrow a typed authoring port declaration for cold contract construction.
///
/// The reference does not own or bind a runtime endpoint. It is valid only while the nonmovable
/// declaration outlives it. Authoring addresses never enter execution plans, schema identities,
/// serialized records, or transport metadata.
class PortRef {
 public:
  /// @brief Borrow an input declaration.
  /// @param port Stable input member that outlives this reference.
  PortRef(const InputBase& port) noexcept  // NOLINT(runtime/explicit)
      : port_(&port), direction_(PortDirection::kInput) {}

  /// @brief Borrow an output declaration.
  /// @param port Stable output member that outlives this reference.
  PortRef(const OutputBase& port) noexcept  // NOLINT(runtime/explicit)
      : port_(&port), direction_(PortDirection::kOutput) {}

  /// @brief Reject borrowing a temporary input declaration.
  /// @param port Temporary declaration that cannot outlive this reference.
  // NOLINTNEXTLINE(runtime/explicit)
  PortRef(InputBase&& port) = delete;

  /// @brief Reject borrowing a const temporary input declaration.
  /// @param port Temporary declaration that cannot outlive this reference.
  // NOLINTNEXTLINE(runtime/explicit)
  PortRef(const InputBase&& port) = delete;

  /// @brief Reject borrowing a temporary output declaration.
  /// @param port Temporary declaration that cannot outlive this reference.
  // NOLINTNEXTLINE(runtime/explicit)
  PortRef(OutputBase&& port) = delete;

  /// @brief Reject borrowing a const temporary output declaration.
  /// @param port Temporary declaration that cannot outlive this reference.
  // NOLINTNEXTLINE(runtime/explicit)
  PortRef(const OutputBase&& port) = delete;

  /// @brief Return the direction of the borrowed declaration.
  /// @return Input or output direction.
  [[nodiscard]] PortDirection direction() const noexcept { return direction_; }

  /// @brief Return the borrowed declaration name.
  /// @return Name view, or an empty view for an invalid internal state.
  [[nodiscard]] std::string_view name() const noexcept {
    if (port_ == nullptr) {
      return {};
    }
    return direction_ == PortDirection::kInput ? static_cast<const InputBase*>(port_)->name()
                                               : static_cast<const OutputBase*>(port_)->name();
  }

  /// @brief Return whether this reference borrows a declaration.
  /// @return \c true when bound to an input or output member.
  [[nodiscard]] explicit operator bool() const noexcept { return port_ != nullptr; }

 private:
  const void* port_{};
  PortDirection direction_{PortDirection::kInput};
};

}  // namespace holoscan
