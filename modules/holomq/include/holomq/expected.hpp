/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <tl/expected.hpp>

#include <type_traits>
#include <utility>

namespace holomq {

template <class T, class E>
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

}  // namespace holomq
