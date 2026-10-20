/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <tl/expected.hpp>

#include <type_traits>
#include <utility>

namespace holoscan::detail {

template <typename T>
struct IsExpected : std::false_type {};

template <typename T, typename E>
struct IsExpected<tl::expected<T, E>> : std::true_type {
  using value_type = T;
};

template <typename T>
inline constexpr bool is_expected_v = IsExpected<std::remove_cvref_t<T>>::value;

// Static assertions live in templates so malformed macro operands always include stable,
// SDK-owned diagnostics even when a frontend also reports secondary expression errors.
template <typename Result>
constexpr void validate_expected_operand() {
  static_assert(is_expected_v<Result>, "HOLOSCAN_TRY requires a holoscan::expected operand");
}

template <typename Result>
constexpr void validate_void_expected_operand() {
  validate_expected_operand<Result>();
  if constexpr (is_expected_v<Result>) {
    static_assert(std::is_void_v<typename IsExpected<std::remove_cvref_t<Result>>::value_type>,
                  "HOLOSCAN_RETURN_IF_ERROR requires holoscan::expected<void, E>");
  }
}

template <typename Result>
constexpr void validate_value_expected_operand() {
  validate_expected_operand<Result>();
  if constexpr (is_expected_v<Result>) {
    static_assert(!std::is_void_v<typename IsExpected<std::remove_cvref_t<Result>>::value_type>,
                  "HOLOSCAN_ASSIGN_OR_RETURN requires holoscan::expected<T, E> with non-void T");
  }
}

}  // namespace holoscan::detail

#define HOLOSCAN_DETAIL_CONCATENATE_INNER(left, right) left##right
#define HOLOSCAN_DETAIL_CONCATENATE(left, right) HOLOSCAN_DETAIL_CONCATENATE_INNER(left, right)

#if defined(__COUNTER__)
#define HOLOSCAN_DETAIL_UNIQUE_ID __COUNTER__
#else
// All supported GCC, Clang, clang-cl, and MSVC lanes provide __COUNTER__. The line fallback keeps
// the header usable on another conforming frontend, except for two invocations on one source line.
#define HOLOSCAN_DETAIL_UNIQUE_ID __LINE__
#endif

#define HOLOSCAN_DETAIL_PARENTHESIS_PROBE() ~, 1
#define HOLOSCAN_DETAIL_CHECK_PARENTHESIS_INNER(_ignored, result, ...) result
#define HOLOSCAN_DETAIL_CHECK_PARENTHESIS(...) \
  HOLOSCAN_DETAIL_CHECK_PARENTHESIS_INNER(__VA_ARGS__, 0)
#define HOLOSCAN_DETAIL_PARENTHESIS_EAT(...) HOLOSCAN_DETAIL_PARENTHESIS_END
#define HOLOSCAN_DETAIL_PARENTHESIS_END(...) HOLOSCAN_DETAIL_PARENTHESIS_PROBE()
// Invoke the terminal probe only when one outer pair encloses the complete token sequence. An
// assignable expression such as `(holder).value` therefore remains untouched, while a protected
// comma declaration such as `(std::pair<int, int> value)` loses its one syntactic wrapper.
#define HOLOSCAN_DETAIL_IS_FULLY_PARENTHESIZED(value) \
  HOLOSCAN_DETAIL_CHECK_PARENTHESIS(HOLOSCAN_DETAIL_PARENTHESIS_EAT value())
#define HOLOSCAN_DETAIL_REMOVE_PARENS_INNER(...) __VA_ARGS__
#define HOLOSCAN_DETAIL_REMOVE_PARENS_0(value) value
#define HOLOSCAN_DETAIL_REMOVE_PARENS_1(value) HOLOSCAN_DETAIL_REMOVE_PARENS_INNER value
#define HOLOSCAN_DETAIL_REMOVE_OPTIONAL_PARENS(value)                        \
  HOLOSCAN_DETAIL_CONCATENATE(HOLOSCAN_DETAIL_REMOVE_PARENS_,                \
                              HOLOSCAN_DETAIL_IS_FULLY_PARENTHESIZED(value)) \
  (value)

#define HOLOSCAN_DETAIL_RETURN_IF_ERROR_IMPL(expression, unique_id)                               \
  do {                                                                                            \
    auto&& HOLOSCAN_DETAIL_CONCATENATE(_holoscan_expected_result_, unique_id) = (expression);     \
    ::holoscan::detail::validate_void_expected_operand<decltype(HOLOSCAN_DETAIL_CONCATENATE(      \
        _holoscan_expected_result_, unique_id))>();                                               \
    if (!HOLOSCAN_DETAIL_CONCATENATE(_holoscan_expected_result_, unique_id)) {                    \
      return ::tl::make_unexpected(                                                               \
          std::move(HOLOSCAN_DETAIL_CONCATENATE(_holoscan_expected_result_, unique_id)).error()); \
    }                                                                                             \
  } while (false)

#define HOLOSCAN_DETAIL_ASSIGN_OR_RETURN_IMPL(lhs, expression, unique_id)                       \
  auto&& HOLOSCAN_DETAIL_CONCATENATE(_holoscan_expected_result_, unique_id) = (expression);     \
  ::holoscan::detail::validate_value_expected_operand<decltype(HOLOSCAN_DETAIL_CONCATENATE(     \
      _holoscan_expected_result_, unique_id))>();                                               \
  if (!HOLOSCAN_DETAIL_CONCATENATE(_holoscan_expected_result_, unique_id)) {                    \
    return ::tl::make_unexpected(                                                               \
        std::move(HOLOSCAN_DETAIL_CONCATENATE(_holoscan_expected_result_, unique_id)).error()); \
  }                                                                                             \
  HOLOSCAN_DETAIL_REMOVE_OPTIONAL_PARENS(lhs) =                                                 \
      *std::move(HOLOSCAN_DETAIL_CONCATENATE(_holoscan_expected_result_, unique_id))

/// @brief Return the error from a failed `expected<void, E>` expression.
///
/// Evaluate `expression` exactly once. On failure, move its concrete error object into an
/// unexpected result returned from the enclosing function. On success, continue execution.
/// The macro is a single statement and is safe in an unbraced control-flow body.
/// Exceptions raised while evaluating `expression` or moving its error are ordinary C++
/// exceptions; this macro does not catch or convert them.
///
/// @param expression An expression yielding `holoscan::expected<void, E>` whose error type can be
/// moved into the enclosing function's unexpected return value.
#define HOLOSCAN_RETURN_IF_ERROR(expression) \
  HOLOSCAN_DETAIL_RETURN_IF_ERROR_IMPL(expression, HOLOSCAN_DETAIL_UNIQUE_ID)

/// @brief Assign an expected value or return its concrete error.
///
/// Evaluate `expression` exactly once. On success, move its value into `lhs`; `lhs` may be an
/// existing assignable expression or a declaration. Parenthesize a declaration that contains a
/// comma or a structured binding, for example `(std::pair<int, int> value)` or
/// `(auto [left, right])`. On failure, `lhs` is not evaluated and the original error is moved into
/// an unexpected result returned from the enclosing function. A declaration remains visible in
/// the surrounding compound scope.
/// Exceptions raised while evaluating `expression`, assigning or moving its successful value, or
/// moving its error are ordinary C++ exceptions; this macro does not catch or convert them.
///
/// @param lhs An existing assignable expression or declaration that receives the successful value.
/// Parenthesize declarations containing preprocessing commas or structured bindings.
/// @param expression An expression yielding a non-void `holoscan::expected<T, E>` whose value is
/// assigned to `lhs` and whose error can be moved into the enclosing function's unexpected return
/// value.
#define HOLOSCAN_ASSIGN_OR_RETURN(lhs, expression) \
  HOLOSCAN_DETAIL_ASSIGN_OR_RETURN_IMPL(lhs, expression, HOLOSCAN_DETAIL_UNIQUE_ID)

#define HOLOSCAN_DETAIL_TRY_ONE(expression) HOLOSCAN_RETURN_IF_ERROR(expression)
#define HOLOSCAN_DETAIL_TRY_TWO(lhs, expression) HOLOSCAN_ASSIGN_OR_RETURN(lhs, expression)
#define HOLOSCAN_DETAIL_TRY_INVALID(...) \
  static_assert(false, "HOLOSCAN_TRY accepts exactly one or two arguments")
#define HOLOSCAN_DETAIL_ARGUMENT_COUNT_IMPL(_0,    \
                                            _1,    \
                                            _2,    \
                                            _3,    \
                                            _4,    \
                                            _5,    \
                                            _6,    \
                                            _7,    \
                                            _8,    \
                                            _9,    \
                                            _10,   \
                                            _11,   \
                                            _12,   \
                                            _13,   \
                                            _14,   \
                                            _15,   \
                                            _16,   \
                                            _17,   \
                                            _18,   \
                                            _19,   \
                                            _20,   \
                                            _21,   \
                                            _22,   \
                                            _23,   \
                                            _24,   \
                                            _25,   \
                                            _26,   \
                                            _27,   \
                                            _28,   \
                                            _29,   \
                                            _30,   \
                                            _31,   \
                                            _32,   \
                                            _33,   \
                                            _34,   \
                                            _35,   \
                                            _36,   \
                                            _37,   \
                                            _38,   \
                                            _39,   \
                                            _40,   \
                                            _41,   \
                                            _42,   \
                                            _43,   \
                                            _44,   \
                                            _45,   \
                                            _46,   \
                                            _47,   \
                                            _48,   \
                                            _49,   \
                                            _50,   \
                                            _51,   \
                                            _52,   \
                                            _53,   \
                                            _54,   \
                                            _55,   \
                                            _56,   \
                                            _57,   \
                                            _58,   \
                                            _59,   \
                                            _60,   \
                                            _61,   \
                                            _62,   \
                                            _63,   \
                                            _64,   \
                                            count, \
                                            ...)   \
  count
#define HOLOSCAN_DETAIL_ARGUMENT_COUNT_EXPAND(...) HOLOSCAN_DETAIL_ARGUMENT_COUNT_IMPL(__VA_ARGS__)
#define HOLOSCAN_DETAIL_ARGUMENT_COUNT(...)                                     \
  HOLOSCAN_DETAIL_ARGUMENT_COUNT_EXPAND(__VA_OPT__(, /* NOLINT */) __VA_ARGS__, \
                                        64,                                     \
                                        63,                                     \
                                        62,                                     \
                                        61,                                     \
                                        60,                                     \
                                        59,                                     \
                                        58,                                     \
                                        57,                                     \
                                        56,                                     \
                                        55,                                     \
                                        54,                                     \
                                        53,                                     \
                                        52,                                     \
                                        51,                                     \
                                        50,                                     \
                                        49,                                     \
                                        48,                                     \
                                        47,                                     \
                                        46,                                     \
                                        45,                                     \
                                        44,                                     \
                                        43,                                     \
                                        42,                                     \
                                        41,                                     \
                                        40,                                     \
                                        39,                                     \
                                        38,                                     \
                                        37,                                     \
                                        36,                                     \
                                        35,                                     \
                                        34,                                     \
                                        33,                                     \
                                        32,                                     \
                                        31,                                     \
                                        30,                                     \
                                        29,                                     \
                                        28,                                     \
                                        27,                                     \
                                        26,                                     \
                                        25,                                     \
                                        24,                                     \
                                        23,                                     \
                                        22,                                     \
                                        21,                                     \
                                        20,                                     \
                                        19,                                     \
                                        18,                                     \
                                        17,                                     \
                                        16,                                     \
                                        15,                                     \
                                        14,                                     \
                                        13,                                     \
                                        12,                                     \
                                        11,                                     \
                                        10,                                     \
                                        9,                                      \
                                        8,                                      \
                                        7,                                      \
                                        6,                                      \
                                        5,                                      \
                                        4,                                      \
                                        3,                                      \
                                        2,                                      \
                                        1,                                      \
                                        0)
#define HOLOSCAN_DETAIL_TRY_COUNT_0 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_1 HOLOSCAN_DETAIL_TRY_ONE
#define HOLOSCAN_DETAIL_TRY_COUNT_2 HOLOSCAN_DETAIL_TRY_TWO
#define HOLOSCAN_DETAIL_TRY_COUNT_3 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_4 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_5 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_6 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_7 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_8 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_9 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_10 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_11 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_12 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_13 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_14 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_15 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_16 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_17 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_18 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_19 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_20 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_21 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_22 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_23 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_24 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_25 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_26 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_27 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_28 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_29 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_30 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_31 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_32 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_33 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_34 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_35 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_36 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_37 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_38 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_39 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_40 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_41 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_42 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_43 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_44 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_45 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_46 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_47 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_48 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_49 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_50 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_51 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_52 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_53 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_54 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_55 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_56 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_57 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_58 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_59 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_60 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_61 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_62 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_63 HOLOSCAN_DETAIL_TRY_INVALID
#define HOLOSCAN_DETAIL_TRY_COUNT_64 HOLOSCAN_DETAIL_TRY_INVALID

/// @brief Propagate a failed expected expression using the portable statement form.
///
/// `HOLOSCAN_TRY(expression)` is equivalent to `HOLOSCAN_RETURN_IF_ERROR(expression)`.
/// `HOLOSCAN_TRY(lhs, expression)` is equivalent to
/// `HOLOSCAN_ASSIGN_OR_RETURN(lhs, expression)`.
/// Calls with zero or 3 through 64 preprocessing arguments produce a stable SDK diagnostic.
/// Calls with more than 64 preprocessing arguments are outside the supported diagnostic bound.
/// Exception behavior is inherited from the selected `HOLOSCAN_RETURN_IF_ERROR` or
/// `HOLOSCAN_ASSIGN_OR_RETURN` form; exceptions are not caught or converted.
///
/// @param ... Pass one `holoscan::expected<void, E>` expression, or pass an assignment target and
/// one non-void `holoscan::expected<T, E>` expression. Parenthesize a first argument that contains
/// preprocessing commas as documented by `HOLOSCAN_ASSIGN_OR_RETURN`.
#define HOLOSCAN_TRY(...)                                                  \
  HOLOSCAN_DETAIL_CONCATENATE(HOLOSCAN_DETAIL_TRY_COUNT_,                  \
                              HOLOSCAN_DETAIL_ARGUMENT_COUNT(__VA_ARGS__)) \
  (__VA_ARGS__)

/// @brief Indicate whether the optional expression-form propagation extension is available.
///
/// Expression-form propagation is intentionally unavailable until every supported frontend has
/// a validated implementation. In particular, every native MSVC and clang-cl mode remains `0`.
#define HOLOSCAN_HAS_TRY_EXPR 0

/// @brief Diagnose use of the unavailable expression-form propagation extension.
///
/// @param ... Arguments reserved for a future expression-form propagation contract. Every current
/// invocation is rejected with a stable diagnostic; use `HOLOSCAN_TRY` instead.
#define HOLOSCAN_TRYX(...)                                                                   \
  ([]<bool holoscan_tryx_is_available = false>() {                                           \
    static_assert(holoscan_tryx_is_available,                                                \
                  "HOLOSCAN_TRYX is unavailable; use portable HOLOSCAN_TRY statement form"); \
  }())
