/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <concepts>
#include <type_traits>
#include <utility>

#include <holoscan/core/temporal_contract.hpp>

namespace holoscan {

/// @brief Own a pointer-free self-member method descriptor for Runtime-Operator rebinding.
/// @tparam Object Declaring Operator type.
/// @tparam Method Unambiguous member-function pointer type.
///
/// This cold value stores the method identity but never the address of the authoring object passed
/// to \ref bind_predicate. Plan compilation validates the declaring type and each RunSession
/// rebinds the method to its fresh Runtime Operator instance.
template <typename Object, typename Method>
class BoundPredicate {
 public:
  /// @brief Name the declaring runtime Operator type used for rebinding.
  using object_type = Object;

  /// @brief Construct one pointer-free method descriptor.
  /// @param method Unambiguous self-member method identity to copy.
  explicit BoundPredicate(Method method) noexcept : method_(method) {}

  /// @brief Return the copied member-function identity.
  /// @return Method pointer with no receiver address.
  [[nodiscard]] Method method() const noexcept { return method_; }

 private:
  Method method_;
};

namespace detail {

template <typename Object, typename Method>
inline constexpr bool is_bound_predicate_v<BoundPredicate<Object, Method>> = true;

template <typename Object, typename Method>
struct PredicateBoundObject<BoundPredicate<Object, Method>> {
  using type = Object;
};

template <typename Object, typename Method>
struct PredicateCallableSignature<BoundPredicate<Object, Method>, void>
    : PredicateCallableSignature<Method> {};

template <typename Method>
struct PredicateMemberOwner;

template <typename Owner, typename Result, typename... Args>
struct PredicateMemberOwner<Result (Owner::*)(Args...)> {
  using type = Owner;
};

template <typename Owner, typename Result, typename... Args>
struct PredicateMemberOwner<Result (Owner::*)(Args...) const> {
  using type = Owner;
};

template <typename Owner, typename Result, typename... Args>
struct PredicateMemberOwner<Result (Owner::*)(Args...) noexcept> {
  using type = Owner;
};

template <typename Owner, typename Result, typename... Args>
struct PredicateMemberOwner<Result (Owner::*)(Args...) const noexcept> {
  using type = Owner;
};

template <typename Object, typename Method>
struct PredicateEvaluatorInvoker<BoundPredicate<Object, Method>> {
  template <typename... Args>
  [[nodiscard]] static decltype(auto) invoke(BoundPredicate<Object, Method>& descriptor,
                                             void* self_receiver, Args&&... args) {
    auto& object = *static_cast<Object*>(self_receiver);
    return std::invoke(descriptor.method(), object, std::forward<Args>(args)...);
  }
};

}  // namespace detail

/// @brief Create a pointer-free self-member predicate descriptor.
/// @tparam Object Declaring Operator type used for each fresh Runtime instance.
/// @tparam Method Unambiguous member-function pointer type.
/// @param object Authoring self object used only for type validation; its address is not retained.
/// @param method Method identity copied for later RunSession rebinding.
/// @return Copyable cold descriptor accepted by \ref predicate.
template <typename Object, typename Method>
  requires std::is_member_function_pointer_v<Method> &&
           std::derived_from<std::remove_cvref_t<Object>, OperatorBase> &&
           std::same_as<std::remove_cvref_t<Object>,
                        typename detail::PredicateMemberOwner<Method>::type>
[[nodiscard]] BoundPredicate<std::remove_cvref_t<Object>, Method> bind_predicate(Object& object,
                                                                                 Method method) {
  static_cast<void>(object);
  return BoundPredicate<std::remove_cvref_t<Object>, Method>{method};
}

}  // namespace holoscan
