/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <string>
#include <type_traits>

#include <holomq/errors.hpp>
#include <holomq/expected.hpp>

namespace holomq {

TEST(ExpectedTest, MakeUnexpectedDecaysErrorTypeForRvalueAndLvalue) {
  auto from_rvalue = make_unexpected(Error{ErrorCode::kInvalidArgument, "rvalue"});
  static_assert(std::is_same_v<decltype(from_rvalue), unexpected<Error>>,
                "make_unexpected must decay an rvalue error to a value type");

  const Error error{ErrorCode::kNotReady, "lvalue"};
  auto from_lvalue = make_unexpected(error);
  static_assert(std::is_same_v<decltype(from_lvalue), unexpected<Error>>,
                "make_unexpected must decay an lvalue error to a value type, not a reference");

  const Error const_error{ErrorCode::kClosed, "const lvalue"};
  auto from_const_lvalue = make_unexpected(const_error);
  static_assert(std::is_same_v<decltype(from_const_lvalue), unexpected<Error>>,
                "make_unexpected must decay a const lvalue error to a value type");

  const expected<int, Error> rvalue_result = from_rvalue;
  const expected<int, Error> lvalue_result = from_lvalue;
  ASSERT_FALSE(rvalue_result);
  ASSERT_FALSE(lvalue_result);
  EXPECT_EQ(rvalue_result.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(rvalue_result.error().message, "rvalue");
  EXPECT_EQ(lvalue_result.error().code, ErrorCode::kNotReady);
  EXPECT_EQ(lvalue_result.error().message, "lvalue");
}

TEST(ExpectedTest, ForwardErrorPreservesErrorAcrossValueTypes) {
  const expected<int, Error> lvalue_source =
      make_unexpected(Error{ErrorCode::kNotFound, "missing"});
  static_assert(std::is_same_v<decltype(forward_error(lvalue_source)), unexpected<Error>>,
                "forward_error must return unexpected<Error>");
  const expected<double, Error> forwarded_from_lvalue = forward_error(lvalue_source);
  ASSERT_FALSE(forwarded_from_lvalue);
  EXPECT_EQ(forwarded_from_lvalue.error().code, ErrorCode::kNotFound);
  EXPECT_EQ(forwarded_from_lvalue.error().message, "missing");

  static_assert(std::is_same_v<decltype(forward_error(expected<int, Error>{
                                   make_unexpected(Error{ErrorCode::kClosed, "closed"})})),
                               unexpected<Error>>,
                "forward_error must return unexpected<Error>");
  const expected<double, Error> forwarded_from_rvalue =
      forward_error(expected<int, Error>{make_unexpected(Error{ErrorCode::kClosed, "closed"})});
  ASSERT_FALSE(forwarded_from_rvalue);
  EXPECT_EQ(forwarded_from_rvalue.error().code, ErrorCode::kClosed);
  EXPECT_EQ(forwarded_from_rvalue.error().message, "closed");
}

TEST(ExpectedTest, MakeUnexpectedNothrowPreservesDiagnostic) {
  static_assert(noexcept(make_unexpected_nothrow<int>(ErrorCode::kInvalidArgument, "bad value")));

  const expected<int, Error> result =
      make_unexpected_nothrow<int>(ErrorCode::kInvalidArgument, "bad value");
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(result.error().message, "bad value");
}

TEST(ErrorsTest, ErrorValueIsCheapForCodeOnly) {
  static_assert(std::is_nothrow_move_constructible_v<Error>,
                "Error should be cheap to move (no throwing members)");

  const Error default_error;
  EXPECT_EQ(default_error.code, ErrorCode::kFailure);
  EXPECT_TRUE(default_error.message.empty());

  const Error code_only{ErrorCode::kBackpressured};
  EXPECT_EQ(code_only.code, ErrorCode::kBackpressured);
  EXPECT_TRUE(code_only.message.empty());
}

TEST(ErrorsTest, ErrorMessageRendersCodeAndOptionalDetail) {
  const Error code_only{ErrorCode::kNotReady};
  EXPECT_EQ(error_message(code_only), "kNotReady");

  const Error with_detail{ErrorCode::kInvalidArgument, "bad value"};
  EXPECT_EQ(error_message(with_detail), "kInvalidArgument: bad value");
}

TEST(ErrorsTest, ErrorCodeNameReturnsEnumeratorName) {
  EXPECT_EQ(std::string(error_code_name(ErrorCode::kInvalidArgument)), "kInvalidArgument");
  EXPECT_EQ(std::string(error_code_name(ErrorCode::kNotReady)), "kNotReady");
  EXPECT_EQ(std::string(error_code_name(ErrorCode::kStaleEndpoint)), "kStaleEndpoint");
  EXPECT_EQ(std::string(error_code_name(ErrorCode::kTimeout)), "kTimeout");
  EXPECT_EQ(std::string(error_code_name(ErrorCode::kClosed)), "kClosed");
  EXPECT_EQ(std::string(error_code_name(ErrorCode::kResourceExhausted)), "kResourceExhausted");
  EXPECT_EQ(std::string(error_code_name(ErrorCode::kPermissionDenied)), "kPermissionDenied");
  EXPECT_EQ(std::string(error_code_name(ErrorCode::kInterrupted)), "kInterrupted");
  EXPECT_EQ(std::string(error_code_name(ErrorCode::kProtocolError)), "kProtocolError");
  EXPECT_EQ(std::string(error_code_name(ErrorCode::kIncompatible)), "kIncompatible");
  EXPECT_EQ(std::string(error_code_name(ErrorCode::kIncompleteFanout)), "kIncompleteFanout");
}

}  // namespace holomq
