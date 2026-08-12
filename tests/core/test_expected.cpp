/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <type_traits>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

namespace holoscan {

// holoscan::Error is the value type returned via expected<T, Error>. make_unexpected applies
// std::decay_t to the deduced error type, so it returns an unexpected that holds a value regardless
// of the argument's value category. Without decay_t, an lvalue argument deduces E = Error& and
// yields the malformed unexpected<Error&>.
TEST(Expected, MakeUnexpectedDecaysErrorTypeForRvalueAndLvalue) {
  // Key setup: three argument categories (rvalue, lvalue, const lvalue) each passed to
  // make_unexpected; static_assert on each result confirms the decayed value type.
  // rvalue argument
  auto from_rvalue = make_unexpected(Error{ErrorCode::kInvalidArgument, "rvalue"});
  static_assert(std::is_same_v<decltype(from_rvalue), unexpected<Error>>,
                "make_unexpected must decay an rvalue error to a value type");

  // lvalue argument (the case that regresses without decay_t)
  const Error error{ErrorCode::kNotReady, "lvalue"};
  auto from_lvalue = make_unexpected(error);
  static_assert(std::is_same_v<decltype(from_lvalue), unexpected<Error>>,
                "make_unexpected must decay an lvalue error to a value type, not a reference");

  // const lvalue argument
  const Error const_error{ErrorCode::kClosed, "const lvalue"};
  auto from_const_lvalue = make_unexpected(const_error);
  static_assert(std::is_same_v<decltype(from_const_lvalue), unexpected<Error>>,
                "make_unexpected must decay a const lvalue error to a value type");

  // Key assertion: the decayed unexpected holds an independent value copy, so the
  // original error's code and message survive conversion to expected<int, Error>.
  const expected<int, Error> rvalue_result = from_rvalue;
  const expected<int, Error> lvalue_result = from_lvalue;
  ASSERT_FALSE(rvalue_result);
  ASSERT_FALSE(lvalue_result);
  EXPECT_EQ(rvalue_result.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(rvalue_result.error().message, "rvalue");
  EXPECT_EQ(lvalue_result.error().code, ErrorCode::kNotReady);
  EXPECT_EQ(lvalue_result.error().message, "lvalue");
}

// forward_error re-wraps the error of one expected so it can be returned from a function with a
// different value type. Both the const-ref (copy) and rvalue (move) overloads preserve the error.
TEST(Expected, ForwardErrorPreservesErrorAcrossValueTypes) {
  // Key setup: one error in expected<int,Error> forwarded to expected<double,Error>
  // via lvalue (copy) and rvalue (move) to confirm both overloads preserve code+message.
  const expected<int, Error> lvalue_source =
      make_unexpected(Error{ErrorCode::kNotFound, "missing"});
  const expected<double, Error> forwarded_from_lvalue = forward_error(lvalue_source);
  ASSERT_FALSE(forwarded_from_lvalue);
  EXPECT_EQ(forwarded_from_lvalue.error().code, ErrorCode::kNotFound);
  EXPECT_EQ(forwarded_from_lvalue.error().message, "missing");

  const expected<double, Error> forwarded_from_rvalue =
      forward_error(expected<int, Error>{make_unexpected(Error{ErrorCode::kClosed, "closed"})});
  ASSERT_FALSE(forwarded_from_rvalue);
  EXPECT_EQ(forwarded_from_rvalue.error().code, ErrorCode::kClosed);
  EXPECT_EQ(forwarded_from_rvalue.error().message, "closed");
}

// Error is cheap to move and the common code-only case carries no message (and therefore performs
// no allocation).
TEST(Errors, ErrorValueIsCheapForCodeOnly) {
  static_assert(std::is_nothrow_move_constructible_v<Error>,
                "Error should be cheap to move (no throwing members)");

  const Error default_error;
  EXPECT_EQ(default_error.code, ErrorCode::kFailure);
  EXPECT_TRUE(default_error.message.empty());

  const Error code_only{ErrorCode::kBackpressured};
  EXPECT_EQ(code_only.code, ErrorCode::kBackpressured);
  EXPECT_TRUE(code_only.message.empty());
}

TEST(Errors, ErrorBridgeRendersEmptyMessageFromCode) {
  // Key setup: a code-only Error bridged to RuntimeError and then round-tripped back to
  // Error via make_error; a second case adds a context string to verify it is preserved.
  const Error code_only{ErrorCode::kNotReady};
  EXPECT_TRUE(code_only.message.empty());

  const RuntimeError runtime_error(code_only);
  EXPECT_EQ(runtime_error.code(), ErrorCode::kNotReady);
  // Key assertion: code-only what() is the bare enumerator name with no trailing separator.
  EXPECT_STREQ(runtime_error.what(), "kNotReady");

  const Error round_trip = make_error(runtime_error);
  EXPECT_EQ(round_trip.code, ErrorCode::kNotReady);
  EXPECT_TRUE(round_trip.message.empty());

  const RuntimeError with_context(ErrorCode::kInvalidArgument, "bad sleep target");
  const Error context_round_trip = make_error(with_context);
  EXPECT_EQ(context_round_trip.code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(context_round_trip.message, "bad sleep target");
}

// RuntimeError carries an ErrorCode; the constructors inherited from std::runtime_error (message
// only) default it to kFailure.
TEST(Errors, RuntimeErrorCarriesCode) {
  const RuntimeError coded(ErrorCode::kInvalidArgument, "bad argument");
  EXPECT_EQ(coded.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(std::string(coded.what()).find("bad argument"), std::string::npos);

  const RuntimeError code_only(ErrorCode::kNotReady);
  EXPECT_EQ(code_only.code(), ErrorCode::kNotReady);

  // An inherited std::runtime_error constructor (message only) defaults the code to kFailure and
  // leaves the message verbatim.
  const RuntimeError message_only("just a message");
  EXPECT_EQ(message_only.code(), ErrorCode::kFailure);
  EXPECT_STREQ(message_only.what(), "just a message");
}

// RuntimeError owns a copy of the message, so a borrowed string_view sub-range is safe
// to pass as long as RuntimeError's constructor reads it synchronously.
TEST(Errors, RuntimeErrorAcceptsBorrowedMessageView) {
  // Key setup: message is a sub-range of a local string_view, not a null-terminated string.
  constexpr std::string_view storage = "prefix: borrowed diagnostic :suffix";
  constexpr std::string_view message = storage.substr(8, 19);

  const RuntimeError error(ErrorCode::kInvalidArgument, message);

  EXPECT_EQ(error.code(), ErrorCode::kInvalidArgument);
  EXPECT_STREQ(error.what(), "kInvalidArgument: borrowed diagnostic");
}

// RuntimeError is a std::runtime_error, so it can be thrown and caught as std::runtime_error /
// std::exception while still exposing its ErrorCode when recovered as a RuntimeError.
TEST(Errors, RuntimeErrorIsCompatibleWithStdRuntimeError) {
  static_assert(std::is_base_of_v<std::runtime_error, RuntimeError>,
                "RuntimeError must derive from std::runtime_error");

  try {
    throw RuntimeError(ErrorCode::kInvalidArgument, "boom");
  } catch (const std::exception& error) {
    EXPECT_NE(std::string(error.what()).find("boom"), std::string::npos);
  }

  try {
    throw RuntimeError(ErrorCode::kNotReady, "later");
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("later"), std::string::npos);
    const auto* runtime_error = dynamic_cast<const RuntimeError*>(&error);
    ASSERT_NE(runtime_error, nullptr);
    EXPECT_EQ(runtime_error->code(), ErrorCode::kNotReady);
  }
}

// error_code_name returns the enumerator name via magic_enum, e.g. ErrorCode::kInvalidArgument ->
// "kInvalidArgument".
TEST(Errors, ErrorCodeNameReturnsEnumeratorName) {
  EXPECT_EQ(std::string(error_code_name(ErrorCode::kInvalidArgument)), "kInvalidArgument");
  EXPECT_EQ(std::string(error_code_name(ErrorCode::kNotReady)), "kNotReady");
  EXPECT_EQ(std::string(error_code_name(ErrorCode::kStaleEndpoint)), "kStaleEndpoint");
  EXPECT_EQ(std::string(error_code_name(ErrorCode::kClosed)), "kClosed");
  EXPECT_EQ(std::string(error_code_name(ErrorCode::kTimeout)), "kTimeout");
  EXPECT_EQ(std::string(error_code_name(ErrorCode::kPermissionDenied)), "kPermissionDenied");
  EXPECT_EQ(std::string(error_code_name(ErrorCode::kInterrupted)), "kInterrupted");
  EXPECT_EQ(std::string(error_code_name(ErrorCode::kProtocolError)), "kProtocolError");
  EXPECT_EQ(std::string(error_code_name(ErrorCode::kIncompatible)), "kIncompatible");
}

// RuntimeError can be constructed from an Error, bridging the expected<T, Error> return path to the
// throwing path. The code is preserved and the message mirrors the (ErrorCode, message) ctor.
TEST(Errors, RuntimeErrorFromErrorValue) {
  // Key setup: three bridging cases — code-only, message-bearing, and the common one-liner
  // escalation pattern (throw RuntimeError(result.error())).
  // A code-only Error yields the same what() as the code-only RuntimeError ctor (no trailing
  // separator from an empty message).
  const RuntimeError from_code_only(Error{ErrorCode::kNotReady});
  EXPECT_EQ(from_code_only.code(), ErrorCode::kNotReady);
  EXPECT_STREQ(from_code_only.what(), RuntimeError(ErrorCode::kNotReady).what());

  // A message-bearing Error produces a composite what() containing the message.
  const RuntimeError from_message(Error{ErrorCode::kInvalidArgument, "bad value"});
  EXPECT_EQ(from_message.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(std::string(from_message.what()).find("bad value"), std::string::npos);

  // The one-liner escalation from a returned Error to a throw preserves code and message.
  const expected<int, Error> result = make_unexpected(Error{ErrorCode::kClosed, "gone"});
  ASSERT_FALSE(result);
  try {
    throw RuntimeError(result.error());
  } catch (const std::runtime_error& error) {
    const auto* runtime_error = dynamic_cast<const RuntimeError*>(&error);
    ASSERT_NE(runtime_error, nullptr);
    EXPECT_EQ(runtime_error->code(), ErrorCode::kClosed);
    EXPECT_NE(std::string(error.what()).find("gone"), std::string::npos);
  }
}

// make_error is the reverse bridge (RuntimeError -> Error). It recovers the code and strips the
// generated "<error string>: " prefix so an Error -> RuntimeError -> Error round-trip is stable.
TEST(Errors, MakeErrorFromRuntimeErrorRoundTrips) {
  // Key setup: three round-trip cases confirming the prefix-stripping logic in make_error:
  // code-only, message-bearing (prefix stripped), and inherited ctor (kept verbatim).
  // A code-only RuntimeError maps to an Error with an empty message.
  const Error from_code_only = make_error(RuntimeError(ErrorCode::kClosed));
  EXPECT_EQ(from_code_only.code, ErrorCode::kClosed);
  EXPECT_TRUE(from_code_only.message.empty());

  // A message-bearing RuntimeError strips the generated prefix, so re-bridging does not duplicate
  // it: Error -> RuntimeError -> Error reproduces the original what().
  const Error from_message = make_error(RuntimeError(ErrorCode::kInvalidArgument, "bad value"));
  EXPECT_EQ(from_message.code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(from_message.message, "bad value");
  EXPECT_STREQ(RuntimeError(from_message).what(),
               RuntimeError(ErrorCode::kInvalidArgument, "bad value").what());

  // A message that lacks the generated prefix (inherited std::runtime_error ctor) is kept verbatim
  // and defaults to kFailure.
  const Error from_inherited = make_error(RuntimeError("plain message"));
  EXPECT_EQ(from_inherited.code, ErrorCode::kFailure);
  EXPECT_EQ(from_inherited.message, "plain message");
}

}  // namespace holoscan
