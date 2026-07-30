/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_EXPECTED_HPP
#define HOLOSCAN_CORE_EXPECTED_HPP

#include <tl/expected.hpp>
#include <utility>

namespace holoscan {

template <class T, class E>
using expected = tl::expected<T, E>;

template <class E>
using unexpected = tl::unexpected<E>;

template <class E>
using bad_expected_access = tl::bad_expected_access<E>;

using unexpect_t = tl::unexpect_t;

static constexpr unexpect_t unexpect{};  // codespell-ignore

/**
 * @brief Construct a new unexpected object
 *
 * @tparam E The type of the error class
 * @param e The object of the error class
 * @return The unexpected object
 */
template <class E>
static inline constexpr unexpected<E> make_unexpected(E&& e) {
  return unexpected<E>{std::forward<E>(e)};
}

// Extracts the error code as an unexpected.
template <class T, class E>
unexpected<E> forward_error(const expected<T, E>& expected) {
  return unexpected<E>{expected.error()};
}

// Extracts the error code as an unexpected.
template <class T, class E>
unexpected<E> forward_error(expected<T, E>&& expected) {
  return make_unexpected(expected.error());
}

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_EXPECTED_HPP */
