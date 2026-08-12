/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <tl/expected.hpp>

#include <type_traits>
#include <utility>

namespace holoscan {

/// @brief Represent either a value of type \c T or an error of type \c E.
///
/// This alias remains backed by \c tl::expected as the HSDK 5 EA1 public source and type contract.
///
/// @tparam T Value type, or `void` for success-without-a-value operations.
/// @tparam E Error type.
template <class T, class E>
using expected = tl::expected<T, E>;

/// @brief Represent an unexpected error value.
///
/// @tparam E Error type.
template <class E>
using unexpected = tl::unexpected<E>;

/// @brief Report invalid value access on an expected result.
///
/// @tparam E Error type.
template <class E>
using bad_expected_access = tl::bad_expected_access<E>;

/// @brief Select construction of an expected result from an error value.
using unexpect_t = tl::unexpect_t;

/// Tag used to construct an expected result from an error value.
static constexpr unexpect_t unexpect{};  // codespell-ignore

/// @brief Construct an owning unexpected error value.
///
/// Decay the argument type so an lvalue never produces a reference-bearing `unexpected`.
///
/// @tparam E Deduced error argument type.
/// @param error Error value to copy or move into the result.
/// @return The owning unexpected error value.
template <class E>
[[nodiscard]] constexpr unexpected<std::decay_t<E>> make_unexpected(E&& error) {
  return unexpected<std::decay_t<E>>{std::forward<E>(error)};
}

/// @brief Copy an existing failure into an unexpected result.
///
/// @tparam T Source value type.
/// @tparam E Error type.
/// @param result Failed result whose error is copied.
/// @return An unexpected result containing a copy of the original error.
/// @pre `result` does not contain a value.
template <class T, class E>
[[nodiscard]] unexpected<E> forward_error(const expected<T, E>& result) {
  return unexpected<E>{result.error()};
}

/// @brief Move an existing failure into an unexpected result.
///
/// @tparam T Source value type.
/// @tparam E Error type.
/// @param result Failed result whose concrete error is moved.
/// @return An unexpected result containing the original error.
/// @pre `result` does not contain a value.
/// @note The error held by `result` is moved from.
template <class T, class E>
[[nodiscard]] unexpected<E> forward_error(expected<T, E>&& result) {
  return make_unexpected(std::move(result).error());
}

}  // namespace holoscan

// Keep the macro machinery isolated while preserving expected.hpp as the supported consumer
// include. The helper header deliberately follows the aliases and functions that its expansions
// reference.
#include <holoscan/core/expected_macros.hpp>
