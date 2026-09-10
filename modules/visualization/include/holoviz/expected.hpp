/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <tl/expected.hpp>

#include <holoviz/errors.hpp>

#include <type_traits>
#include <utility>

namespace holoviz {

/// HoloViz expected type; the error type defaults to holoviz::Error (see errors.hpp).
template <class T, class E = Error>
using expected = tl::expected<T, E>;

template <class E>
using unexpected = tl::unexpected<E>;

template <class E>
using bad_expected_access = tl::bad_expected_access<E>;

using unexpect_t = tl::unexpect_t;

static constexpr unexpect_t unexpect{};  // codespell-ignore

template <class E>
[[nodiscard]] constexpr unexpected<std::decay_t<E>> make_unexpected(E&& error) {
  return unexpected<std::decay_t<E>>{std::forward<E>(error)};
}

template <class T, class E>
[[nodiscard]] unexpected<E> forward_error(const expected<T, E>& expected_value) {
  return unexpected<E>{expected_value.error()};
}

template <class T, class E>
[[nodiscard]] unexpected<E> forward_error(expected<T, E>&& expected_value) {
  return make_unexpected(std::move(expected_value).error());
}

}  // namespace holoviz

#define VIZ_UNWRAP(TYPE, VAR_NAME, FUNC_CALL)             \
  TYPE VAR_NAME{};                                        \
  {                                                       \
    auto VAR_NAME##_expected = FUNC_CALL;                 \
    if (!VAR_NAME##_expected) {                           \
      return holoviz::forward_error(VAR_NAME##_expected); \
    }                                                     \
    VAR_NAME = std::move(VAR_NAME##_expected).value();    \
  }

#define VIZ_UNWRAP_NODECL(TYPE, VAR_NAME, FUNC_CALL)         \
  {                                                          \
    auto VAR_NAME##_expected = FUNC_CALL;                    \
    if (!VAR_NAME##_expected) {                              \
      return holoviz::forward_error(VAR_NAME##_expected);    \
    }                                                        \
    VAR_NAME = (TYPE)std::move(VAR_NAME##_expected).value(); \
  }

// Evaluate an expected-returning expression, forward the error on failure, and otherwise do
// nothing. For expected<void> calls and calls whose value the caller does not need.
#define VIZ_FORWARD_ERROR(EXPR)                                \
  {                                                            \
    auto _holoviz_assert_expected = (EXPR);                    \
    if (!_holoviz_assert_expected) {                           \
      return holoviz::forward_error(_holoviz_assert_expected); \
    }                                                          \
  }
