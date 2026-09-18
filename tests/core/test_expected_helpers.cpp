/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include "core/diagnostic_registry.hpp"

namespace holoscan {
namespace {

// Returns expected<void, Error> with a side-effect counter so callers can verify the
// expression is evaluated exactly once. Used by HOLOSCAN_TRY void-form tests.
expected<void, Error> maybe_void(bool fail, int& evaluations) {
  ++evaluations;
  if (fail) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "void failure"});
  }
  return {};
}

// Returns expected<int, Error> with a side-effect counter. Yields 41 on success so
// callers can detect value threading through propagation helpers (e.g. +1 → 42).
expected<int, Error> maybe_value(bool fail, int& evaluations) {
  ++evaluations;
  if (fail) {
    return make_unexpected(Error{ErrorCode::kNotFound, "value failure"});
  }
  return 41;
}

// Exercises HOLOSCAN_TRY in its void form: the macro must propagate an error without
// declaring or unwrapping a value variable. See VoidFormPreservesSuccessFailureAndSingleEvaluation.
expected<void, Error> propagate_void(bool fail, int& evaluations) {
  HOLOSCAN_TRY(maybe_void(fail, evaluations));
  return {};
}

// Exercises the HOLOSCAN_TRY(auto value, expr) declaration form: the unwrapped value
// must be visible in the enclosing scope after the macro expands.
expected<int, Error> propagate_declared_value(bool fail, int& evaluations) {
  HOLOSCAN_TRY(auto value, maybe_value(fail, evaluations));
  return value + 1;
}

// Exercises HOLOSCAN_ASSIGN_OR_RETURN with a pre-declared variable: the macro must
// assign into an existing lvalue and propagate the error unchanged on failure.
expected<int, Error> propagate_assigned_value(bool fail, int& evaluations) {
  int value = 0;
  HOLOSCAN_ASSIGN_OR_RETURN(value, maybe_value(fail, evaluations));
  return value + 1;
}

// Simple aggregate whose member can serve as an HOLOSCAN_ASSIGN_OR_RETURN target
// via parenthesized sub-expressions such as (holder).value or (*ptr).value.
struct AssignableHolder {
  int value{};
};

// Verifies that HOLOSCAN_ASSIGN_OR_RETURN handles LHS targets that begin with a
// parenthesized sub-expression, which would otherwise break naive comma-splitting
// macro implementations. Tests both (holder).value and (*ptr).value forms.
expected<int, Error> propagate_parenthesized_assignable_targets() {
  int evaluations = 0;
  AssignableHolder holder{};
  AssignableHolder* holder_pointer = &holder;
  HOLOSCAN_ASSIGN_OR_RETURN((holder).value, maybe_value(false, evaluations));
  HOLOSCAN_ASSIGN_OR_RETURN((*holder_pointer).value, maybe_value(false, evaluations));
  return holder.value + evaluations;
}

// REQ: EXPECTED-001
// REQ: EXPECTED-003
TEST(ExpectedHelpers, VoidFormPreservesSuccessFailureAndSingleEvaluation) {
  int evaluations = 0;
  const expected<void, Error> success = propagate_void(false, evaluations);
  EXPECT_TRUE(success);
  EXPECT_EQ(evaluations, 1);

  const expected<void, Error> failure = propagate_void(true, evaluations);
  ASSERT_FALSE(failure);
  EXPECT_EQ(evaluations, 2);
  EXPECT_EQ(failure.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(failure.error().message, "void failure");
}

// REQ: EXPECTED-002
// REQ: EXPECTED-003
TEST(ExpectedHelpers, ValueDeclarationAndAssignmentPreserveResult) {
  int evaluations = 0;
  const expected<int, Error> declared = propagate_declared_value(false, evaluations);
  ASSERT_TRUE(declared);
  EXPECT_EQ(*declared, 42);

  const expected<int, Error> assigned = propagate_assigned_value(false, evaluations);
  ASSERT_TRUE(assigned);
  EXPECT_EQ(*assigned, 42);

  const expected<int, Error> failure = propagate_declared_value(true, evaluations);
  ASSERT_FALSE(failure);
  EXPECT_EQ(failure.error().code, ErrorCode::kNotFound);
  EXPECT_EQ(failure.error().message, "value failure");
  EXPECT_EQ(evaluations, 3);
}

// REQ: EXPECTED-002
TEST(ExpectedHelpers, SupportsAssignableTargetsThatBeginWithParenthesizedSubexpressions) {
  const auto result = propagate_parenthesized_assignable_targets();
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, 43);
}

// A non-copyable error type (unique_ptr member) that verifies HOLOSCAN_TRY never
// attempts to copy the error when propagating it out of a function. Both the success
// path (unique_ptr<int> value) and the failure path must be move-only safe.
struct MoveOnlyError {
  explicit MoveOnlyError(int input) : value(std::make_unique<int>(input)) {}
  MoveOnlyError(MoveOnlyError&&) noexcept = default;
  MoveOnlyError& operator=(MoveOnlyError&&) noexcept = default;
  MoveOnlyError(const MoveOnlyError&) = delete;
  MoveOnlyError& operator=(const MoveOnlyError&) = delete;

  std::unique_ptr<int> value;
};

// Source function: returns a move-only value (7) or a move-only error (9).
expected<std::unique_ptr<int>, MoveOnlyError> move_only_result(bool fail) {
  if (fail) {
    return make_unexpected(MoveOnlyError{9});
  }
  return std::make_unique<int>(7);
}

// Propagates a move-only value/error through HOLOSCAN_TRY; confirms no copy is made.
expected<std::unique_ptr<int>, MoveOnlyError> propagate_move_only(bool fail) {
  HOLOSCAN_TRY(auto value, move_only_result(fail));
  return value;
}

// REQ: EXPECTED-004
TEST(ExpectedHelpers, MovesValuesAndConcreteErrorsWithoutCopying) {
  const auto success = propagate_move_only(false);
  ASSERT_TRUE(success);
  ASSERT_NE(success->get(), nullptr);
  EXPECT_EQ(**success, 7);

  const auto failure = propagate_move_only(true);
  ASSERT_FALSE(failure);
  ASSERT_NE(failure.error().value, nullptr);
  EXPECT_EQ(*failure.error().value, 9);
}

// Four helpers exercising every supported spelling of explicit error construction so
// that SupportsPredicateResultsAndExplicitErrorForwarding can verify each path:
//   make_unexpected(), explicit unexpected<E>{}, forward_error(const&), forward_error(&&).
expected<bool, Error> predicate_failure_from_make_unexpected() {
  return make_unexpected(Error{ErrorCode::kProtocolError, "predicate make_unexpected failure"});
}

expected<bool, Error> predicate_failure_from_explicit_unexpected() {
  return unexpected<Error>{Error{ErrorCode::kInterrupted, "predicate explicit failure"}};
}

expected<bool, Error> predicate_failure_from_const_forward() {
  const expected<int, Error> source =
      make_unexpected(Error{ErrorCode::kNotFound, "predicate copied failure"});
  return forward_error(source);
}

expected<bool, Error> predicate_failure_from_move_forward() {
  expected<int, Error> source =
      make_unexpected(Error{ErrorCode::kPermissionDenied, "predicate moved failure"});
  return forward_error(std::move(source));
}

// REQ: EXPECTED-017
TEST(ExpectedHelpers, SupportsPredicateResultsAndExplicitErrorForwarding) {
  // Key setup: expected<bool,Error> holds true/false as ordinary successes; all four
  //   error-construction helpers use a distinct ErrorCode so each path is unambiguous.
  const expected<bool, Error> true_result = true;
  const expected<bool, Error> false_result = false;
  ASSERT_TRUE(true_result);
  ASSERT_TRUE(false_result);
  EXPECT_TRUE(*true_result);
  EXPECT_FALSE(*false_result);

  const auto made = predicate_failure_from_make_unexpected();
  ASSERT_FALSE(made);
  EXPECT_EQ(made.error().code, ErrorCode::kProtocolError);
  EXPECT_EQ(made.error().message, "predicate make_unexpected failure");

  const auto explicit_result = predicate_failure_from_explicit_unexpected();
  ASSERT_FALSE(explicit_result);
  EXPECT_EQ(explicit_result.error().code, ErrorCode::kInterrupted);
  EXPECT_EQ(explicit_result.error().message, "predicate explicit failure");

  const auto copied = predicate_failure_from_const_forward();
  ASSERT_FALSE(copied);
  EXPECT_EQ(copied.error().code, ErrorCode::kNotFound);
  EXPECT_EQ(copied.error().message, "predicate copied failure");

  const auto moved = predicate_failure_from_move_forward();
  // Key assertion: move-forwarded error retains its original code and message.
  ASSERT_FALSE(moved);
  EXPECT_EQ(moved.error().code, ErrorCode::kPermissionDenied);
  EXPECT_EQ(moved.error().message, "predicate moved failure");
}

// Returns a pair so callers can drive HOLOSCAN_TRY with declarations that contain an
// internal comma (structured bindings, parenthesized pair types).
expected<std::pair<int, int>, Error> pair_result() {
  return std::pair<int, int>{20, 22};
}

// Exercises three syntactic forms that embed internal commas in HOLOSCAN_TRY:
//   (const std::pair<int,int> pair), (const auto [left, right]),
//   and a parenthesized RHS expression (expected<std::pair<int,int>, Error>{...}).
expected<int, Error> propagate_parenthesized_declarations() {
  HOLOSCAN_TRY((const std::pair<int, int> pair), pair_result());
  HOLOSCAN_TRY((const auto [left, right]), pair_result());
  HOLOSCAN_TRY(const auto wrapped_expression,
               (expected<std::pair<int, int>, Error>{std::pair<int, int>{1, 2}}));
  return pair.first + pair.second + left + right + wrapped_expression.first +
         wrapped_expression.second;
}

// REQ: EXPECTED-005
TEST(ExpectedHelpers, SupportsParenthesizedCommaDeclarationsAndStructuredBindings) {
  const auto result = propagate_parenthesized_declarations();
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, 87);
}

// Template version of pair_result: the comma in the template argument list means the
// RHS of HOLOSCAN_TRY must be parenthesized to prevent macro argument splitting.
template <typename Left, typename Right>
expected<std::pair<Left, Right>, Error> templated_pair_result() {
  return std::pair<Left, Right>{20, 22};
}

// Wrapping alias macro: verifies HOLOSCAN_TRY expands correctly when invoked from
// inside another variadic macro (double __VA_ARGS__ forwarding round-trip).
#define HOLOSCAN_TEST_NESTED_TRY(...) HOLOSCAN_TRY(__VA_ARGS__)

// Uses HOLOSCAN_TRY inside a braced for-loop body; the macro must expand to a single
// statement and must not shadow variables declared in previous loop iterations.
expected<int, Error> propagate_in_braced_loop(const int iterations, int& evaluations) {
  int total = 0;
  for (int index = 0; index < iterations; ++index) {
    HOLOSCAN_TRY(auto value, maybe_value(false, evaluations));
    total += value;
  }
  return total;
}

// Uses HOLOSCAN_TRY in two distinct braced switch-case bodies (one success path,
// one failure path) to confirm the macro is safe across control-flow boundaries.
expected<int, Error> propagate_in_braced_switch(const int branch, int& evaluations) {
  switch (branch) {
    case 0: {
      HOLOSCAN_TRY(auto value, maybe_value(false, evaluations));
      return value;
    }
    case 1: {
      HOLOSCAN_TRY(auto value, maybe_value(true, evaluations));
      return value;
    }
    default:
      return -1;
  }
}

// Calls HOLOSCAN_TRY through the wrapper alias macro inside a nested inner scope;
// the result variable must not leak into the enclosing outer scope.
expected<int, Error> propagate_through_nested_macro_and_scope(int& evaluations) {
  int total = 1;
  {
    HOLOSCAN_TEST_NESTED_TRY(auto nested_value, maybe_value(false, evaluations));
    total += nested_value;
  }
  return total;
}

// Uses HOLOSCAN_TRY with a structured-binding LHS and a parenthesized templated RHS.
// The outer parens around templated_pair_result<int, int>() protect the template
// argument comma from the preprocessor's macro argument parser.
expected<int, Error> propagate_protected_rhs_comma() {
  HOLOSCAN_TRY((const auto [left, right]), (templated_pair_result<int, int>()));
  return left + right;
}

#undef HOLOSCAN_TEST_NESTED_TRY

// REQ: EXPECTED-019
TEST(ExpectedHelpers, SupportsNestedMacrosBracedControlFlowAndNestedScopes) {
  // Key setup: four call sites exercise distinct macro expansion contexts — for-loop,
  //   switch success, switch failure, and a nested inner scope via alias macro.
  // Key assertion: the evaluations counter increments by exactly 1 per call site,
  //   confirming the expression is not duplicated by any expansion strategy.
  int evaluations = 0;

  const auto loop_result = propagate_in_braced_loop(3, evaluations);
  ASSERT_TRUE(loop_result);
  EXPECT_EQ(*loop_result, 123);
  EXPECT_EQ(evaluations, 3);

  const auto switch_success = propagate_in_braced_switch(0, evaluations);
  ASSERT_TRUE(switch_success);
  EXPECT_EQ(*switch_success, 41);
  EXPECT_EQ(evaluations, 4);

  const auto switch_failure = propagate_in_braced_switch(1, evaluations);
  ASSERT_FALSE(switch_failure);
  EXPECT_EQ(switch_failure.error().code, ErrorCode::kNotFound);
  EXPECT_EQ(evaluations, 5);

  const auto nested_scope = propagate_through_nested_macro_and_scope(evaluations);
  ASSERT_TRUE(nested_scope);
  EXPECT_EQ(*nested_scope, 42);
  EXPECT_EQ(evaluations, 6);

  const auto protected_comma = propagate_protected_rhs_comma();
  ASSERT_TRUE(protected_comma);
  EXPECT_EQ(*protected_comma, 42);
}

// Invokes HOLOSCAN_TRY twice on the same physical source line (semicolon-separated).
// The macro must generate distinct temporary names to avoid symbol redefinition.
expected<int, Error> propagate_two_calls_on_one_line() {
  int evaluations = 0;
  // clang-format off
  HOLOSCAN_TRY(auto first, maybe_value(false, evaluations)); HOLOSCAN_TRY(auto second, maybe_value(false, evaluations));  // NOLINT
  // clang-format on
  return first + second + evaluations;
}

// REQ: EXPECTED-006
TEST(ExpectedHelpers, UsesCollisionResistantTemporaryNames) {
  const auto result = propagate_two_calls_on_one_line();
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, 84);
}

// Places HOLOSCAN_RETURN_IF_ERROR in an unbraced if-branch. The macro must expand to
// a single statement (typically do-while(false)) to prevent dangling-else bugs.
expected<void, Error> use_void_form_in_unbraced_control_flow(bool run, int& evaluations) {
  if (run)
    HOLOSCAN_RETURN_IF_ERROR(maybe_void(false, evaluations));
  else
    ++evaluations;
  return {};
}

// Places HOLOSCAN_RETURN_IF_ERROR as the sole unbraced body of a while-loop.
expected<void, Error> use_void_form_in_unbraced_loop(int iterations, int& evaluations) {
  while (iterations-- > 0)
    HOLOSCAN_RETURN_IF_ERROR(maybe_void(false, evaluations));
  return {};
}

// Places HOLOSCAN_RETURN_IF_ERROR in an unbraced else-branch.
expected<void, Error> use_void_form_in_unbraced_else(bool use_else, int& evaluations) {
  if (!use_else)
    ++evaluations;
  else
    HOLOSCAN_RETURN_IF_ERROR(maybe_void(false, evaluations));
  return {};
}

// REQ: EXPECTED-007
TEST(ExpectedHelpers, VoidFormIsSafeInUnbracedControlFlow) {
  int evaluations = 0;
  EXPECT_TRUE(use_void_form_in_unbraced_control_flow(true, evaluations));
  EXPECT_EQ(evaluations, 1);
  EXPECT_TRUE(use_void_form_in_unbraced_control_flow(false, evaluations));
  EXPECT_EQ(evaluations, 2);
  EXPECT_TRUE(use_void_form_in_unbraced_loop(3, evaluations));
  EXPECT_EQ(evaluations, 5);
  EXPECT_TRUE(use_void_form_in_unbraced_else(true, evaluations));
  EXPECT_EQ(evaluations, 6);
}

// Verifies the LHS target expression is NOT evaluated when the RHS returns an error.
// target_evaluations must remain 0 because values.at(target_evaluations++) is the
// assignment target and must be skipped entirely when maybe_value() fails.
expected<int, Error> assignment_target_is_not_evaluated_on_failure(int& target_evaluations) {
  std::array<int, 1> values{0};
  int result_evaluations = 0;
  HOLOSCAN_ASSIGN_OR_RETURN(values.at(target_evaluations++), maybe_value(true, result_evaluations));
  return values.front();
}

// REQ: EXPECTED-003
TEST(ExpectedHelpers, DoesNotEvaluateAssignmentTargetOnFailure) {
  int target_evaluations = 0;
  const auto result = assignment_target_is_not_evaluated_on_failure(target_evaluations);
  ASSERT_FALSE(result);
  EXPECT_EQ(target_evaluations, 0);
}

// REQ: EXPECTED-014
// REQ: EXPECTED-029
TEST(Errors, StableCodesUseFixedAbiAndDistinctMeanings) {
  // Key assertion: each ErrorCode ordinal is a compile-time constant pinned to its
  //   ABI value; kNotSupported != kNotImplemented guards against accidental aliasing.
  static_assert(std::is_same_v<std::underlying_type_t<ErrorCode>, std::uint32_t>);
  static_assert(sizeof(ErrorCode) == sizeof(std::uint32_t));
  static_assert(static_cast<std::uint32_t>(ErrorCode::kSuccess) == 0U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kFailure) == 1U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kReceiveError) == 2U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kCodecError) == 3U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kInvalidArgument) == 4U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kNotFound) == 5U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kDuplicateName) == 6U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kNotImplemented) == 7U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kBackpressured) == 8U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kNotReady) == 9U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kStaleEndpoint) == 10U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kClosed) == 11U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kResourceExhausted) == 12U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kTimeout) == 13U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kPermissionDenied) == 14U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kInterrupted) == 15U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kProtocolError) == 16U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kIncompatible) == 17U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kInputEmpty) == 18U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kInputNotSelected) == 19U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kInputUndeclared) == 20U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kInputSelectionCardinalityMismatch) == 21U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kPublicationIndeterminate) == 22U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kNotSupported) == 23U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kLifecycleInvalidRestartStage) == 24U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kLifecycleTransactionsDisabled) == 25U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kStaleOperatorRef) == 26U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kCrossSessionOperatorRef) == 27U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kLifecycleRestartUnauthorized) == 28U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kLifecycleRestartQueueFull) == 29U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kLifecycleRestartUnsupportedInProfile) ==
                30U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kLifecycleInvalidCallingContext) == 31U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kLifecycleRunState) == 32U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kLifecycleQuiesceFailed) == 33U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kLifecycleStageFailed) == 34U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kLifecycleRollbackFailed) == 35U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kPublicationPartial) == 36U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kRateBindingInvalid) == 37U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kRateBindingAboveMaximum) == 38U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kRateBindingConflict) == 39U);
  static_assert(static_cast<std::uint32_t>(ErrorCode::kArithmeticOverflow) == 40U);

  EXPECT_EQ(error_code_name(ErrorCode::kInputEmpty), "kInputEmpty");
  EXPECT_EQ(error_code_name(ErrorCode::kInputNotSelected), "kInputNotSelected");
  EXPECT_EQ(error_code_name(ErrorCode::kInputUndeclared), "kInputUndeclared");
  EXPECT_EQ(error_code_name(ErrorCode::kInputSelectionCardinalityMismatch),
            "kInputSelectionCardinalityMismatch");
  EXPECT_EQ(error_code_name(ErrorCode::kPublicationIndeterminate), "kPublicationIndeterminate");
  EXPECT_EQ(error_code_name(ErrorCode::kNotSupported), "kNotSupported");
  EXPECT_EQ(error_code_name(ErrorCode::kLifecycleInvalidRestartStage),
            "kLifecycleInvalidRestartStage");
  EXPECT_EQ(error_code_name(ErrorCode::kLifecycleRollbackFailed), "kLifecycleRollbackFailed");
  EXPECT_EQ(error_code_name(ErrorCode::kPublicationPartial), "kPublicationPartial");
  EXPECT_EQ(error_code_name(ErrorCode::kRateBindingInvalid), "kRateBindingInvalid");
  EXPECT_EQ(error_code_name(ErrorCode::kRateBindingAboveMaximum), "kRateBindingAboveMaximum");
  EXPECT_EQ(error_code_name(ErrorCode::kRateBindingConflict), "kRateBindingConflict");
  EXPECT_EQ(error_code_name(ErrorCode::kArithmeticOverflow), "kArithmeticOverflow");
  EXPECT_NE(ErrorCode::kNotSupported, ErrorCode::kNotImplemented);
  EXPECT_TRUE(publication_error_consumes_authority(ErrorCode::kPublicationIndeterminate));
  EXPECT_TRUE(publication_error_consumes_authority(ErrorCode::kPublicationPartial));
  EXPECT_FALSE(publication_error_consumes_authority(ErrorCode::kBackpressured));
}

// REQ: EXPECTED-008
static_assert(HOLOSCAN_HAS_TRY_EXPR == 0 || HOLOSCAN_HAS_TRY_EXPR == 1,
              "HOLOSCAN_HAS_TRY_EXPR must be a numeric feature macro");

#if defined(_MSC_VER)
static_assert(HOLOSCAN_HAS_TRY_EXPR == 0,
              "expression-form propagation is unavailable on every MSVC frontend mode");
#endif

// REQ: EXPECTED-015
// REQ: EXPECTED-026
// REQ: EXPECTED-027
// REQ: EXPECTED-028
// REQ: COMPILE-051
// REQ: COMPILE-068
// REQ: COMPILE-069
// REQ: COMPILE-073
// REQ: COMPILE-074
// REQ: COMPILE-080
// REQ: RUNTIME-027
// REQ: RUNTIME-028
// REQ: RUNTIME-047
// REQ: RUNTIME-049
// REQ: COMPILE-071
// REQ: COMPILE-072
TEST(StructuralDiagnosticRegistry, OwnsUniqueStableTokensAndTypedSchemas) {
  using detail::DiagnosticPayloadField;
  using detail::DiagnosticSeverity;
  using detail::StructuralDiagnosticId;

  using detail::diagnostic_fields_v;
  using detail::StructuralDiagnosticDescriptor;

  // Key setup: expected[] mirrors the compile-time registry in declaration order with
  //   matching id, token string, severity, and required_fields bitmask per entry.
  // Key assertion: structural_diagnostic_registry_is_unique() is a compile-time
  //   uniqueness check; the runtime loop verifies round-trip lookup via
  //   find_structural_diagnostic() and diagnostic_id().
  using ExpectedStructuralDiagnosticRegistry =
      std::array<StructuralDiagnosticDescriptor, detail::kStructuralDiagnosticCount>;
  constexpr ExpectedStructuralDiagnosticRegistry expected{{
      {StructuralDiagnosticId::kProfileValueUnknown,
       "PROFILE_VALUE_UNKNOWN",
       DiagnosticSeverity::kError,
       detail::kProfileCommonFields | diagnostic_fields_v<DiagnosticPayloadField::kObservedValue>},
      {StructuralDiagnosticId::kProfileRegionLoosensGraph,
       "PROFILE_REGION_LOOSENS_GRAPH",
       DiagnosticSeverity::kError,
       detail::kProfileCommonFields | diagnostic_fields_v<DiagnosticPayloadField::kRequestedProfile,
                                                          DiagnosticPayloadField::kAffectedRegion>},
      {StructuralDiagnosticId::kProfilePackageManifestMissing,
       "PROFILE_PACKAGE_MANIFEST_MISSING",
       DiagnosticSeverity::kError,
       detail::kProfileCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredEvidence>},
      {StructuralDiagnosticId::kProfilePackageManifestMalformed,
       "PROFILE_PACKAGE_MANIFEST_MALFORMED",
       DiagnosticSeverity::kError,
       detail::kProfileCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kObservedEvidence>},
      {StructuralDiagnosticId::kProfilePackageManifestAmbiguous,
       "PROFILE_PACKAGE_MANIFEST_AMBIGUOUS",
       DiagnosticSeverity::kError,
       detail::kProfileCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredEvidence,
                               DiagnosticPayloadField::kObservedEvidence>},
      {StructuralDiagnosticId::kProfilePlanSchemaUnsupported,
       "PROFILE_PLAN_SCHEMA_UNSUPPORTED",
       DiagnosticSeverity::kError,
       detail::kProfileCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredSchemaIdentity,
                               DiagnosticPayloadField::kObservedSchemaIdentity>},
      {StructuralDiagnosticId::kProfilePolicyUnsupported,
       "PROFILE_POLICY_UNSUPPORTED",
       DiagnosticSeverity::kError,
       detail::kProfileCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredProfilePolicyId,
                               DiagnosticPayloadField::kObservedProfilePolicyId>},
      {StructuralDiagnosticId::kProfileRuntimeAbiIncompatible,
       "PROFILE_RUNTIME_ABI_INCOMPATIBLE",
       DiagnosticSeverity::kError,
       detail::kProfileCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredRuntimeAbi,
                               DiagnosticPayloadField::kObservedRuntimeAbi>},
      {StructuralDiagnosticId::kProfileTargetIncompatible,
       "PROFILE_TARGET_INCOMPATIBLE",
       DiagnosticSeverity::kError,
       detail::kProfileCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredTargetIdentity,
                               DiagnosticPayloadField::kObservedTargetIdentity>},
      {StructuralDiagnosticId::kProfileCapabilityUnsatisfied,
       "PROFILE_CAPABILITY_UNSATISFIED",
       DiagnosticSeverity::kError,
       detail::kProfileCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredCapabilityIdentity,
                               DiagnosticPayloadField::kObservedCapabilityIdentity>},
      {StructuralDiagnosticId::kProfileEvidenceUnsatisfied,
       "PROFILE_EVIDENCE_UNSATISFIED",
       DiagnosticSeverity::kError,
       detail::kProfileCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredEvidence,
                               DiagnosticPayloadField::kObservedEvidence>},
      {StructuralDiagnosticId::kProfileMaterializationFailed,
       "PROFILE_MATERIALIZATION_FAILED",
       DiagnosticSeverity::kError,
       detail::kProfileCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kMaterializationStrategy,
                               DiagnosticPayloadField::kCompleteBeforeStage,
                               DiagnosticPayloadField::kTypedCause>},
      {StructuralDiagnosticId::kLifecycleTransactionCapacityInvalid,
       "LIFECYCLE_TRANSACTION_CAPACITY_INVALID",
       DiagnosticSeverity::kError,
       diagnostic_fields_v<DiagnosticPayloadField::kCapacityField,
                           DiagnosticPayloadField::kProvidedCapacity>},
      {StructuralDiagnosticId::kLifecycleTransactionCapacityInsufficient,
       "LIFECYCLE_TRANSACTION_CAPACITY_INSUFFICIENT",
       DiagnosticSeverity::kError,
       diagnostic_fields_v<DiagnosticPayloadField::kCapacityField,
                           DiagnosticPayloadField::kRequiredCapacity,
                           DiagnosticPayloadField::kProvidedCapacity>},
      {StructuralDiagnosticId::kLifecycleTransactionCapacityOverflow,
       "LIFECYCLE_TRANSACTION_CAPACITY_OVERFLOW",
       DiagnosticSeverity::kError,
       diagnostic_fields_v<DiagnosticPayloadField::kCapacityField,
                           DiagnosticPayloadField::kRequiredCapacity,
                           DiagnosticPayloadField::kProvidedCapacity>},
      {StructuralDiagnosticId::kLifecycleTransactionUnsupportedProfile,
       "LIFECYCLE_TRANSACTION_UNSUPPORTED_PROFILE",
       DiagnosticSeverity::kError,
       diagnostic_fields_v<DiagnosticPayloadField::kScope,
                           DiagnosticPayloadField::kSelectedProfile>},
      {StructuralDiagnosticId::kLifecycleRestartTargetUnsupported,
       "LIFECYCLE_RESTART_TARGET_UNSUPPORTED",
       DiagnosticSeverity::kError,
       diagnostic_fields_v<DiagnosticPayloadField::kOrigin,
                           DiagnosticPayloadField::kTargetAuthority,
                           DiagnosticPayloadField::kDisposition>},
      {StructuralDiagnosticId::kLifecycleRestartStageMaskInvalid,
       "LIFECYCLE_RESTART_STAGE_MASK_INVALID",
       DiagnosticSeverity::kError,
       diagnostic_fields_v<DiagnosticPayloadField::kOrigin,
                           DiagnosticPayloadField::kTargetAuthority,
                           DiagnosticPayloadField::kStageMask>},
      {StructuralDiagnosticId::kPartitionSelectionEnvironmentValueInvalid,
       "PARTITION_SELECTION_ENV_VALUE_INVALID",
       DiagnosticSeverity::kError,
       detail::kPartitionCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kObservedValue>},
      {StructuralDiagnosticId::kPartitionSelectionConflict,
       "PARTITION_SELECTION_CONFLICT",
       DiagnosticSeverity::kError,
       detail::kPartitionCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kNormalizedValue,
                               DiagnosticPayloadField::kRequestedAlgorithm>},
      {StructuralDiagnosticId::kPartitionSelectionProfileForbidden,
       "PARTITION_SELECTION_PROFILE_FORBIDDEN",
       DiagnosticSeverity::kError,
       detail::kPartitionCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kNormalizedValue,
                               DiagnosticPayloadField::kRequestedAlgorithm>},
      {StructuralDiagnosticId::kPartitionPlannerExperimentalSelected,
       "PARTITION_PLANNER_EXPERIMENTAL_SELECTED",
       DiagnosticSeverity::kWarning,
       detail::kPartitionCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kNormalizedValue,
                               DiagnosticPayloadField::kRequestedAlgorithm,
                               DiagnosticPayloadField::kAcceptedAlgorithm>},
      {StructuralDiagnosticId::kPlatformCapabilityUnavailable,
       "PLATFORM_CAPABILITY_UNAVAILABLE",
       DiagnosticSeverity::kError,
       diagnostic_fields_v<DiagnosticPayloadField::kRequestedCapability,
                           DiagnosticPayloadField::kPlatformAdapter,
                           DiagnosticPayloadField::kMissingPrerequisite,
                           DiagnosticPayloadField::kAuthoredAlternative,
                           DiagnosticPayloadField::kDisposition>},
      {StructuralDiagnosticId::kPublicationIndeterminate,
       "PUBLICATION_INDETERMINATE",
       DiagnosticSeverity::kError,
       diagnostic_fields_v<DiagnosticPayloadField::kOperationPhase,
                           DiagnosticPayloadField::kEndpointGeneration,
                           DiagnosticPayloadField::kExpectedRecipients,
                           DiagnosticPayloadField::kObservedRecipients,
                           DiagnosticPayloadField::kBackendCode,
                           DiagnosticPayloadField::kSequence,
                           DiagnosticPayloadField::kPublicationAuthority>},
      {StructuralDiagnosticId::kHoloMqErrorSuccessInFailure,
       "HOLOMQ_ERROR_SUCCESS_IN_FAILURE",
       DiagnosticSeverity::kError,
       diagnostic_fields_v<DiagnosticPayloadField::kOperationPhase,
                           DiagnosticPayloadField::kBackendCode>},
      {StructuralDiagnosticId::kHoloMqErrorUnmapped,
       "HOLOMQ_ERROR_UNMAPPED",
       DiagnosticSeverity::kError,
       diagnostic_fields_v<DiagnosticPayloadField::kOperationPhase,
                           DiagnosticPayloadField::kBackendCode>},
      {StructuralDiagnosticId::kLifecycleInvalidRestartStage,
       "LIFECYCLE_INVALID_RESTART_STAGE",
       DiagnosticSeverity::kError,
       diagnostic_fields_v<DiagnosticPayloadField::kRequestOrigin,
                           DiagnosticPayloadField::kRequesterRef,
                           DiagnosticPayloadField::kTargetRef,
                           DiagnosticPayloadField::kRequestedStage,
                           DiagnosticPayloadField::kTargetState>},
      {StructuralDiagnosticId::kLifecycleTransactionsDisabled,
       "LIFECYCLE_TRANSACTIONS_DISABLED",
       DiagnosticSeverity::kError,
       diagnostic_fields_v<DiagnosticPayloadField::kRequestOrigin,
                           DiagnosticPayloadField::kTargetRef,
                           DiagnosticPayloadField::kRequestedStage,
                           DiagnosticPayloadField::kSelectedProfile,
                           DiagnosticPayloadField::kDisposition>},
      {StructuralDiagnosticId::kStaleOperatorRef,
       "STALE_OPERATOR_REF",
       DiagnosticSeverity::kError,
       diagnostic_fields_v<DiagnosticPayloadField::kRequestOrigin,
                           DiagnosticPayloadField::kTargetRef,
                           DiagnosticPayloadField::kTargetPlanGeneration,
                           DiagnosticPayloadField::kActivePlanGeneration>},
      {StructuralDiagnosticId::kCrossSessionOperatorRef,
       "CROSS_SESSION_OPERATOR_REF",
       DiagnosticSeverity::kError,
       diagnostic_fields_v<DiagnosticPayloadField::kRequestOrigin,
                           DiagnosticPayloadField::kTargetRef,
                           DiagnosticPayloadField::kTargetRunGeneration,
                           DiagnosticPayloadField::kActiveRunGeneration>},
      {StructuralDiagnosticId::kLifecycleRestartUnauthorized,
       "LIFECYCLE_RESTART_UNAUTHORIZED",
       DiagnosticSeverity::kError,
       diagnostic_fields_v<DiagnosticPayloadField::kRequestOrigin,
                           DiagnosticPayloadField::kRequesterRef,
                           DiagnosticPayloadField::kTargetRef,
                           DiagnosticPayloadField::kRequestedStage,
                           DiagnosticPayloadField::kTargetAuthority,
                           DiagnosticPayloadField::kDisposition>},
      {StructuralDiagnosticId::kLifecycleRestartQueueFull,
       "LIFECYCLE_RESTART_QUEUE_FULL",
       DiagnosticSeverity::kError,
       diagnostic_fields_v<DiagnosticPayloadField::kRequestOrigin,
                           DiagnosticPayloadField::kTargetRef,
                           DiagnosticPayloadField::kCapacityField,
                           DiagnosticPayloadField::kRequiredCapacity,
                           DiagnosticPayloadField::kProvidedCapacity,
                           DiagnosticPayloadField::kQueueCapacity>},
      {StructuralDiagnosticId::kLifecycleRestartUnsupportedInProfile,
       "LIFECYCLE_RESTART_UNSUPPORTED_IN_PROFILE",
       DiagnosticSeverity::kError,
       diagnostic_fields_v<DiagnosticPayloadField::kRequestOrigin,
                           DiagnosticPayloadField::kTargetRef,
                           DiagnosticPayloadField::kScope,
                           DiagnosticPayloadField::kSelectedProfile,
                           DiagnosticPayloadField::kDisposition>},
      {StructuralDiagnosticId::kLifecycleInvalidCallingContext,
       "LIFECYCLE_INVALID_CALLING_CONTEXT",
       DiagnosticSeverity::kError,
       diagnostic_fields_v<DiagnosticPayloadField::kRequestOrigin,
                           DiagnosticPayloadField::kRequesterRef,
                           DiagnosticPayloadField::kTargetRef,
                           DiagnosticPayloadField::kCallingContext>},
      {StructuralDiagnosticId::kLifecycleRunState,
       "LIFECYCLE_RUN_STATE",
       DiagnosticSeverity::kError,
       diagnostic_fields_v<DiagnosticPayloadField::kRequestOrigin,
                           DiagnosticPayloadField::kTargetRef,
                           DiagnosticPayloadField::kRunState,
                           DiagnosticPayloadField::kTargetState>},
      {StructuralDiagnosticId::kLifecycleQuiesceFailed,
       "LIFECYCLE_QUIESCE_FAILED",
       DiagnosticSeverity::kError,
       diagnostic_fields_v<DiagnosticPayloadField::kTransactionId,
                           DiagnosticPayloadField::kTargetRef,
                           DiagnosticPayloadField::kEffectiveStage,
                           DiagnosticPayloadField::kClosureId,
                           DiagnosticPayloadField::kActivationFenceSequence,
                           DiagnosticPayloadField::kActivationRetirementSequence,
                           DiagnosticPayloadField::kQuiesceObligation,
                           DiagnosticPayloadField::kTypedCause>},
      {StructuralDiagnosticId::kLifecycleStageFailed,
       "LIFECYCLE_STAGE_FAILED",
       DiagnosticSeverity::kError,
       diagnostic_fields_v<DiagnosticPayloadField::kTransactionId,
                           DiagnosticPayloadField::kSubmissionId,
                           DiagnosticPayloadField::kReceiptId,
                           DiagnosticPayloadField::kTargetRef,
                           DiagnosticPayloadField::kRequestedStage,
                           DiagnosticPayloadField::kEffectiveStage,
                           DiagnosticPayloadField::kCandidateGeneration,
                           DiagnosticPayloadField::kEffectiveGeneration,
                           DiagnosticPayloadField::kLifecycleEpoch,
                           DiagnosticPayloadField::kPrimaryCause>},
      {StructuralDiagnosticId::kLifecycleRollbackFailed,
       "LIFECYCLE_ROLLBACK_FAILED",
       DiagnosticSeverity::kError,
       diagnostic_fields_v<DiagnosticPayloadField::kTransactionId,
                           DiagnosticPayloadField::kTargetRef,
                           DiagnosticPayloadField::kEffectiveStage,
                           DiagnosticPayloadField::kCandidateGeneration,
                           DiagnosticPayloadField::kEffectiveGeneration,
                           DiagnosticPayloadField::kLifecycleEpoch,
                           DiagnosticPayloadField::kPrimaryCause,
                           DiagnosticPayloadField::kRollbackCause>},
      {StructuralDiagnosticId::kCompilerDiagnosticTokenUnregistered,
       "COMPILER_DIAGNOSTIC_TOKEN_UNREGISTERED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kObservedValue>},
      {StructuralDiagnosticId::kClockCatalogDuplicate,
       "CLOCK_CATALOG_DUPLICATE",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kClockConversionInvalid,
       "CLOCK_CONVERSION_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kClockConversionUndeclared,
       "CLOCK_CONVERSION_UNDECLARED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kClockDeclarationInvalid,
       "CLOCK_DECLARATION_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kClockDefaultDuplicate,
       "CLOCK_DEFAULT_DUPLICATE",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kDeploymentThreadRoleInvalid,
       "DEPLOYMENT_THREAD_ROLE_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kDeploymentThreadRoleDuplicate,
       "DEPLOYMENT_THREAD_ROLE_DUPLICATE",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kDeploymentThreadRoleUnused,
       "DEPLOYMENT_THREAD_ROLE_UNUSED",
       DiagnosticSeverity::kWarning,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kEndpointCatalogMismatch,
       "ENDPOINT_CATALOG_MISMATCH",
       DiagnosticSeverity::kError,
       detail::kEndpointAdmissionCommonFields},
      {StructuralDiagnosticId::kEndpointNameDuplicate,
       "ENDPOINT_NAME_DUPLICATE",
       DiagnosticSeverity::kError,
       detail::kEndpointAdmissionCommonFields},
      {StructuralDiagnosticId::kEndpointNameGeneratedPinDuplicate,
       "ENDPOINT_NAME_GENERATED_PIN_DUPLICATE",
       DiagnosticSeverity::kError,
       detail::kEndpointAdmissionCommonFields},
      {StructuralDiagnosticId::kEndpointNameGeneratedPinInvalid,
       "ENDPOINT_NAME_GENERATED_PIN_INVALID",
       DiagnosticSeverity::kError,
       detail::kEndpointAdmissionCommonFields},
      {StructuralDiagnosticId::kEndpointNameGeneratedPinMismatch,
       "ENDPOINT_NAME_GENERATED_PIN_MISMATCH",
       DiagnosticSeverity::kError,
       detail::kEndpointAdmissionCommonFields},
      {StructuralDiagnosticId::kEndpointNameGeneratedPinRequired,
       "ENDPOINT_NAME_GENERATED_PIN_REQUIRED",
       DiagnosticSeverity::kError,
       detail::kEndpointAdmissionCommonFields},
      {StructuralDiagnosticId::kEndpointNameInvalid,
       "ENDPOINT_NAME_INVALID",
       DiagnosticSeverity::kError,
       detail::kEndpointAdmissionCommonFields},
      {StructuralDiagnosticId::kEndpointNameReservedPrefix,
       "ENDPOINT_NAME_RESERVED_PREFIX",
       DiagnosticSeverity::kError,
       detail::kEndpointAdmissionCommonFields},
      {StructuralDiagnosticId::kPartitionBoundaryCrossed,
       "PARTITION_BOUNDARY_CROSSED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPartitionCollapseIllegal,
       "PARTITION_COLLAPSE_ILLEGAL",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPartitionCoverInvalid,
       "PARTITION_COVER_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPartitionDecisionReasonInvalid,
       "PARTITION_DECISION_REASON_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPartitionLimitExceeded,
       "PARTITION_LIMIT_EXCEEDED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPartitionPlannerAlgorithmVersionUnsupported,
       "PARTITION_PLANNER_ALGORITHM_VERSION_UNSUPPORTED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequestedAlgorithm>},
      {StructuralDiagnosticId::kPartitionPlannerCapabilityUnsupported,
       "PARTITION_PLANNER_CAPABILITY_UNSUPPORTED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPartitionPlannerInterfaceVersionUnsupported,
       "PARTITION_PLANNER_INTERFACE_VERSION_UNSUPPORTED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPartitionPlannerNotFound,
       "PARTITION_PLANNER_NOT_FOUND",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequestedAlgorithm>},
      {StructuralDiagnosticId::kPartitionPlannerRegistrationInvalid,
       "PARTITION_PLANNER_REGISTRATION_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPartitionProposalAlgorithmMismatch,
       "PARTITION_PROPOSAL_ALGORITHM_MISMATCH",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPartitionProposalContractInvalid,
       "PARTITION_PROPOSAL_CONTRACT_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPartitionQuotientCycle,
       "PARTITION_QUOTIENT_CYCLE",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPlanArtifactProvenanceInvalid,
       "PLAN_ARTIFACT_PROVENANCE_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kPlanReference>},
      {StructuralDiagnosticId::kPlanGenerationInvalid,
       "PLAN_GENERATION_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kPlanReference>},
      {StructuralDiagnosticId::kReadinessCapacityInvalid,
       "READINESS_CAPACITY_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kCapacityField,
                               DiagnosticPayloadField::kProvidedCapacity>},
      {StructuralDiagnosticId::kReadinessClockUndeclared,
       "READINESS_CLOCK_UNDECLARED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kReadinessDefaultClockUndeclared,
       "READINESS_DEFAULT_CLOCK_UNDECLARED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kReadinessOwnerUndeclared,
       "READINESS_OWNER_UNDECLARED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kReadinessSourceIdExhausted,
       "READINESS_SOURCE_ID_EXHAUSTED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredCapacity>},
      {StructuralDiagnosticId::kReadinessSourceUnused,
       "READINESS_SOURCE_UNUSED",
       DiagnosticSeverity::kWarning,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kParameterCompileOptionsUnsupported,
       "PARAMETER_COMPILE_OPTIONS_UNSUPPORTED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kConverterAutoInsertionDisableUnsupported,
       "CONVERTER_AUTO_INSERTION_DISABLE_UNSUPPORTED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kEndpointNameGeneratedPinRecommended,
       "ENDPOINT_NAME_GENERATED_PIN_RECOMMENDED",
       DiagnosticSeverity::kWarning,
       detail::kEndpointAdmissionCommonFields},
      {StructuralDiagnosticId::kTemporalPlanGenerationInvalid,
       "TEMPORAL_PLAN_GENERATION_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kTemporalContractFactoryInvalid,
       "TEMPORAL_CONTRACT_FACTORY_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kTemporalContractOwnerUnresolved,
       "TEMPORAL_CONTRACT_OWNER_UNRESOLVED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kTemporalContractFactoryReplayFailed,
       "TEMPORAL_CONTRACT_FACTORY_REPLAY_FAILED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kTemporalContractSetupFailed,
       "TEMPORAL_CONTRACT_SETUP_FAILED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kTemporalContractPortUnresolved,
       "TEMPORAL_CONTRACT_PORT_UNRESOLVED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kTemporalTriggerSourceUnresolved,
       "TEMPORAL_TRIGGER_SOURCE_UNRESOLVED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kTemporalContractReplayFailed,
       "TEMPORAL_CONTRACT_REPLAY_FAILED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kTemporalContractInvalid,
       "TEMPORAL_CONTRACT_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kTemporalProfileUnresolved,
       "TEMPORAL_PROFILE_UNRESOLVED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kTemporalClockUndeclared,
       "TEMPORAL_CLOCK_UNDECLARED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kTemporalInputPortInvalid,
       "TEMPORAL_INPUT_PORT_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kTemporalInputUnconnected,
       "TEMPORAL_INPUT_UNCONNECTED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kTemporalRetentionUnbounded,
       "TEMPORAL_RETENTION_UNBOUNDED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kCapacityField>},
      {StructuralDiagnosticId::kTemporalWindowQueuePolicyDropOldest,
       "TEMPORAL_WINDOW_QUEUE_POLICY_DROP_OLDEST",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kTemporalRetentionInsufficient,
       "TEMPORAL_RETENTION_INSUFFICIENT",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredCapacity,
                               DiagnosticPayloadField::kProvidedCapacity>},
      {StructuralDiagnosticId::kPredicateCallableProfileUnsupported,
       "PREDICATE_CALLABLE_PROFILE_UNSUPPORTED",
       DiagnosticSeverity::kError,
       detail::kProfileCommonFields},
      {StructuralDiagnosticId::kPredicateCallableTypeErased,
       "PREDICATE_CALLABLE_TYPE_ERASED",
       DiagnosticSeverity::kWarning,
       detail::kProfileCommonFields},
      {StructuralDiagnosticId::kPredicateDriverInvalid,
       "PREDICATE_DRIVER_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPredicateInputInvalid,
       "PREDICATE_INPUT_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPredicateConfigIdentityUnavailable,
       "PREDICATE_CONFIG_IDENTITY_UNAVAILABLE",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kSchemaIdentity>},
      {StructuralDiagnosticId::kTemporalWindowWaitClockCapabilityUnavailable,
       "TEMPORAL_WINDOW_WAIT_CLOCK_CAPABILITY_UNAVAILABLE",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kCapabilityIdentity>},
      {StructuralDiagnosticId::kPlacementRoleUnbound,
       "PLACEMENT_ROLE_UNBOUND",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kThreadPlacementRejected,
       "THREAD_PLACEMENT_REJECTED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kSpinUnplaced,
       "SPIN_UNPLACED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPartitionResourceRealizationFailed,
       "PARTITION_RESOURCE_REALIZATION_FAILED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kProfileManifestDuplicateKey,
       "PROFILE_MANIFEST_DUPLICATE_KEY",
       DiagnosticSeverity::kError,
       detail::kProfileCommonFields},
      {StructuralDiagnosticId::kParameterPolicyInvalid,
       "PARAMETER_POLICY_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kParameterOverrideReconstructionUnavailable,
       "PARAMETER_OVERRIDE_RECONSTRUCTION_UNAVAILABLE",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kParameterExplicitFieldPolicyUnproven,
       "PARAMETER_EXPLICIT_FIELD_POLICY_UNPROVEN",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kParameterUpdateStorageUnsupported,
       "PARAMETER_UPDATE_STORAGE_UNSUPPORTED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kParameterFactoryInventoryInvalid,
       "PARAMETER_FACTORY_INVENTORY_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kParameterUpdateProfileUnsupported,
       "PARAMETER_UPDATE_PROFILE_UNSUPPORTED",
       DiagnosticSeverity::kError,
       detail::kProfileCommonFields},
      {StructuralDiagnosticId::kParameterSchemaIncomplete,
       "PARAMETER_SCHEMA_INCOMPLETE",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kSchemaIdentity>},
      {StructuralDiagnosticId::kParameterFieldIdCollision,
       "PARAMETER_FIELD_ID_COLLISION",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kSchemaIdentity>},
      {StructuralDiagnosticId::kParameterUpdateCapacityInsufficient,
       "PARAMETER_UPDATE_CAPACITY_INSUFFICIENT",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kCapacityField,
                               DiagnosticPayloadField::kRequiredCapacity,
                               DiagnosticPayloadField::kProvidedCapacity>},
      {StructuralDiagnosticId::kParameterLifecycleCapacityInsufficient,
       "PARAMETER_LIFECYCLE_CAPACITY_INSUFFICIENT",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kCapacityField,
                               DiagnosticPayloadField::kRequiredCapacity,
                               DiagnosticPayloadField::kProvidedCapacity>},
      {StructuralDiagnosticId::kEndpointClockHealthContractInvalid,
       "ENDPOINT_CLOCK_HEALTH_CONTRACT_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequestedCapability,
                               DiagnosticPayloadField::kSelectedProfile>},
      {StructuralDiagnosticId::kEndpointCatalogNormalizationFailed,
       "ENDPOINT_CATALOG_NORMALIZATION_FAILED",
       DiagnosticSeverity::kError,
       detail::kEndpointAdmissionCommonFields},
      {StructuralDiagnosticId::kEndpointCatalogInlineCapacityExceeded,
       "ENDPOINT_CATALOG_INLINE_CAPACITY_EXCEEDED",
       DiagnosticSeverity::kError,
       detail::kEndpointAdmissionCommonFields},
      {StructuralDiagnosticId::kTemporalClockConversionUnavailable,
       "TEMPORAL_CLOCK_CONVERSION_UNAVAILABLE",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredEvidence>},
      {StructuralDiagnosticId::kParameterDerivedTermInvalid,
       "PARAMETER_DERIVED_TERM_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kSchemaIdentity>},
      {StructuralDiagnosticId::kParameterInstanceIdCollision,
       "PARAMETER_INSTANCE_ID_COLLISION",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kSchemaIdentity>},
      {StructuralDiagnosticId::kParameterOverrideBaselineMismatch,
       "PARAMETER_OVERRIDE_BASELINE_MISMATCH",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredEvidence,
                               DiagnosticPayloadField::kObservedEvidence>},
      {StructuralDiagnosticId::kParameterOverrideDeclarationMismatch,
       "PARAMETER_OVERRIDE_DECLARATION_MISMATCH",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredSchemaIdentity,
                               DiagnosticPayloadField::kObservedSchemaIdentity>},
      {StructuralDiagnosticId::kParameterOverrideDuplicateTarget,
       "PARAMETER_OVERRIDE_DUPLICATE_TARGET",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kParameterOverrideFieldInvalid,
       "PARAMETER_OVERRIDE_FIELD_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kSchemaIdentity>},
      {StructuralDiagnosticId::kParameterOverrideSchemaMismatch,
       "PARAMETER_OVERRIDE_SCHEMA_MISMATCH",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredSchemaIdentity,
                               DiagnosticPayloadField::kObservedSchemaIdentity>},
      {StructuralDiagnosticId::kParameterOverrideTargetMismatch,
       "PARAMETER_OVERRIDE_TARGET_MISMATCH",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kParameterOverrideTargetUnknown,
       "PARAMETER_OVERRIDE_TARGET_UNKNOWN",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kParameterOverrideUndeclaredChange,
       "PARAMETER_OVERRIDE_UNDECLARED_CHANGE",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kSchemaIdentity>},
      {StructuralDiagnosticId::kParameterOverrideValueInvalid,
       "PARAMETER_OVERRIDE_VALUE_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kSchemaIdentity>},
      {StructuralDiagnosticId::kParameterOverrideAdmissionContextMismatch,
       "PARAMETER_OVERRIDE_ADMISSION_CONTEXT_MISMATCH",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredEvidence,
                               DiagnosticPayloadField::kObservedEvidence>},
      {StructuralDiagnosticId::kParameterOverrideProvenanceMismatch,
       "PARAMETER_OVERRIDE_PROVENANCE_MISMATCH",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPhysicalPortContractInvalid,
       "PHYSICAL_PORT_CONTRACT_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPhysicalEdgeContractInvalid,
       "PHYSICAL_EDGE_CONTRACT_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPhysicalInterfaceBindingInvalid,
       "PHYSICAL_INTERFACE_BINDING_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPhysicalQueueCapacityInvalid,
       "PHYSICAL_QUEUE_CAPACITY_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kCapacityField,
                               DiagnosticPayloadField::kProvidedCapacity>},
      {StructuralDiagnosticId::kPhysicalTensorContractInvalid,
       "PHYSICAL_TENSOR_CONTRACT_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPhysicalTensorCapacityOverflow,
       "PHYSICAL_TENSOR_CAPACITY_OVERFLOW",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kCapacityField>},
      {StructuralDiagnosticId::kPhysicalTensorFanoutCapacityInsufficient,
       "PHYSICAL_TENSOR_FANOUT_CAPACITY_INSUFFICIENT",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPhysicalTensorAllocationWidening,
       "PHYSICAL_TENSOR_ALLOCATION_WIDENING",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPhysicalTensorConversionUnsupported,
       "PHYSICAL_TENSOR_CONVERSION_UNSUPPORTED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPhysicalServiceContractInvalid,
       "PHYSICAL_SERVICE_CONTRACT_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPhysicalReadinessCapacityInsufficient,
       "PHYSICAL_READINESS_CAPACITY_INSUFFICIENT",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kCapacityField,
                               DiagnosticPayloadField::kRequiredCapacity,
                               DiagnosticPayloadField::kProvidedCapacity>},
      {StructuralDiagnosticId::kTemporalAlignedMemberInvalid,
       "TEMPORAL_ALIGNED_MEMBER_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kTemporalClockTriggerInvalid,
       "TEMPORAL_CLOCK_TRIGGER_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kTemporalReadableSourceInvalid,
       "TEMPORAL_READABLE_SOURCE_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kImplicitDeviceZeroRejected,
       "IMPLICIT_DEVICE_ZERO_REJECTED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kProcessPlacementRejected,
       "PROCESS_PLACEMENT_REJECTED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kTransportLocalityRejected,
       "TRANSPORT_LOCALITY_REJECTED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kTemporalCaptureTimeRequired,
       "TEMPORAL_CAPTURE_TIME_REQUIRED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredEvidence>},
      {StructuralDiagnosticId::kEndpointAttachBoundRequired,
       "ENDPOINT_ATTACH_BOUND_REQUIRED",
       DiagnosticSeverity::kError,
       detail::kEndpointAdmissionCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kSelectedProfile,
                               DiagnosticPayloadField::kRequiredCapacity>},
      {StructuralDiagnosticId::kGenericLoanTraitEvidenceUnavailable,
       "GENERIC_LOAN_TRAIT_EVIDENCE_UNAVAILABLE",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredEvidence>},
      {StructuralDiagnosticId::kLifecycleDependencyMissing,
       "LIFECYCLE_DEPENDENCY_MISSING",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kLifecycleDependencyDuplicate,
       "LIFECYCLE_DEPENDENCY_DUPLICATE",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kLifecycleDependencyScopeInvalid,
       "LIFECYCLE_DEPENDENCY_SCOPE_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kLifecycleDependencyStageInvalid,
       "LIFECYCLE_DEPENDENCY_STAGE_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kLifecycleDependencyCycle,
       "LIFECYCLE_DEPENDENCY_CYCLE",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kLifecycleCallbackDuplicate,
       "LIFECYCLE_CALLBACK_DUPLICATE",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kLifecycleNodeIdentityAmbiguous,
       "LIFECYCLE_NODE_IDENTITY_AMBIGUOUS",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kLifecycleStageGraphInvalid,
       "LIFECYCLE_STAGE_GRAPH_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPhysicalTensorInlineProviderUnsupported,
       "PHYSICAL_TENSOR_INLINE_PROVIDER_UNSUPPORTED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredEvidence>},
      {StructuralDiagnosticId::kTemporalDeadlineSourceAmbiguous,
       "TEMPORAL_DEADLINE_SOURCE_AMBIGUOUS",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredEvidence>},
      {StructuralDiagnosticId::kTemporalDeadlineSourceUnresolved,
       "TEMPORAL_DEADLINE_SOURCE_UNRESOLVED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredEvidence>},
      {StructuralDiagnosticId::kReadyPolicyRegistrationInvalid,
       "READY_POLICY_REGISTRATION_INVALID",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kReadyPolicyNotFound,
       "READY_POLICY_NOT_FOUND",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kReadyPolicyVersionUnsupported,
       "READY_POLICY_VERSION_UNSUPPORTED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kReadyPolicyInterfaceVersionUnsupported,
       "READY_POLICY_INTERFACE_VERSION_UNSUPPORTED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kReadyPolicyPlanInvalid,
       "READY_POLICY_PLAN_INVALID",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeExecutionPlanEmpty,
       "RUNTIME_EXECUTION_PLAN_EMPTY",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeOperatorInventoryMismatch,
       "RUNTIME_OPERATOR_INVENTORY_MISMATCH",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeOperatorFactoryInvalid,
       "RUNTIME_OPERATOR_FACTORY_INVALID",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeOperatorReconstructionFailed,
       "RUNTIME_OPERATOR_RECONSTRUCTION_FAILED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeOperatorSetupFailed,
       "RUNTIME_OPERATOR_SETUP_FAILED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeOperatorReplayMismatch,
       "RUNTIME_OPERATOR_REPLAY_MISMATCH",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimePortBindingUnavailable,
       "RUNTIME_PORT_BINDING_UNAVAILABLE",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeReadinessBindingUnavailable,
       "RUNTIME_READINESS_BINDING_UNAVAILABLE",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeAdvancedLifecycleBindingUnavailable,
       "RUNTIME_ADVANCED_LIFECYCLE_BINDING_UNAVAILABLE",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeClockInventoryMismatch,
       "RUNTIME_CLOCK_INVENTORY_MISMATCH",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeClockReconstructionFailed,
       "RUNTIME_CLOCK_RECONSTRUCTION_FAILED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeClockFactoryInvalid,
       "RUNTIME_CLOCK_FACTORY_INVALID",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeRunGenerationExhausted,
       "RUNTIME_RUN_GENERATION_EXHAUSTED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeWorkerStartFailed,
       "RUNTIME_WORKER_START_FAILED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeStartupFailed,
       "RUNTIME_STARTUP_FAILED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeLifecycleCallbackFailed,
       "RUNTIME_LIFECYCLE_CALLBACK_FAILED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeTagCapacityExhausted,
       "RUNTIME_TAG_CAPACITY_EXHAUSTED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeOperatorCallbackFailed,
       "RUNTIME_OPERATOR_CALLBACK_FAILED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeSelfJoinForbidden,
       "RUNTIME_SELF_JOIN_FORBIDDEN",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeProfileCompatibilityFailed,
       "RUNTIME_PROFILE_COMPATIBILITY_FAILED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimePlacementBindingUnavailable,
       "RUNTIME_PLACEMENT_BINDING_UNAVAILABLE",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeExternalBindingUnavailable,
       "RUNTIME_EXTERNAL_BINDING_UNAVAILABLE",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeClockBindingUnavailable,
       "RUNTIME_CLOCK_BINDING_UNAVAILABLE",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeClockReadFailed,
       "RUNTIME_CLOCK_READ_FAILED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeParameterBindingUnavailable,
       "RUNTIME_PARAMETER_BINDING_UNAVAILABLE",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeExecutionLaneCountExhausted,
       "RUNTIME_EXECUTION_LANE_COUNT_EXHAUSTED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeExecutionLanePlanInvalid,
       "RUNTIME_EXECUTION_LANE_PLAN_INVALID",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeExecutionLaneRoleDuplicate,
       "RUNTIME_EXECUTION_LANE_ROLE_DUPLICATE",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeLanesNotReady,
       "RUNTIME_LANES_NOT_READY",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeLaneAffinityReadbackMismatch,
       "RUNTIME_LANE_AFFINITY_READBACK_MISMATCH",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeLaneDispatchInvalid,
       "RUNTIME_LANE_DISPATCH_INVALID",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeLaneJoinFailed,
       "RUNTIME_LANE_JOIN_FAILED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeLaneOsPlacementUnavailable,
       "RUNTIME_LANE_OS_PLACEMENT_UNAVAILABLE",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeLanePlacementAdapterUnavailable,
       "RUNTIME_LANE_PLACEMENT_ADAPTER_UNAVAILABLE",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeLaneRecursiveDispatchForbidden,
       "RUNTIME_LANE_RECURSIVE_DISPATCH_FORBIDDEN",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeLaneResetOnForkReadbackMismatch,
       "RUNTIME_LANE_RESET_ON_FORK_READBACK_MISMATCH",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeLaneSchedulingReadbackMismatch,
       "RUNTIME_LANE_SCHEDULING_READBACK_MISMATCH",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeLaneSelfJoinForbidden,
       "RUNTIME_LANE_SELF_JOIN_FORBIDDEN",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeLaneSetAlreadyStarted,
       "RUNTIME_LANE_SET_ALREADY_STARTED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeLaneStartupCancelled,
       "RUNTIME_LANE_STARTUP_CANCELLED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeLaneStartupFailed,
       "RUNTIME_LANE_STARTUP_FAILED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeLaneStopped,
       "RUNTIME_LANE_STOPPED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeLaneTaskCancelled,
       "RUNTIME_LANE_TASK_CANCELLED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeLaneWorkerException,
       "RUNTIME_LANE_WORKER_EXCEPTION",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeLaneWorkerStartFailed,
       "RUNTIME_LANE_WORKER_START_FAILED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeLifecyclePlanInvalid,
       "RUNTIME_LIFECYCLE_PLAN_INVALID",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kEndpointClockHealthAgeClockFaultInvalid,
       "ENDPOINT_CLOCK_HEALTH_AGE_CLOCK_FAULT_INVALID",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kEndpointClockHealthDeadlineOverflow,
       "ENDPOINT_CLOCK_HEALTH_DEADLINE_OVERFLOW",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kEndpointClockHealthGenerationStale,
       "ENDPOINT_CLOCK_HEALTH_GENERATION_STALE",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kEndpointClockHealthIngressFull,
       "ENDPOINT_CLOCK_HEALTH_INGRESS_FULL",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kEndpointClockHealthPreallocationFailed,
       "ENDPOINT_CLOCK_HEALTH_PREALLOCATION_FAILED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kEndpointClockHealthReasonUnsupported,
       "ENDPOINT_CLOCK_HEALTH_REASON_UNSUPPORTED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kEndpointClockHealthReportInvalid,
       "ENDPOINT_CLOCK_HEALTH_REPORT_INVALID",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kEndpointClockHealthResetInvalid,
       "ENDPOINT_CLOCK_HEALTH_RESET_INVALID",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kEndpointClockHealthResetRequiresQuiescence,
       "ENDPOINT_CLOCK_HEALTH_RESET_REQUIRES_QUIESCENCE",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kEndpointClockHealthRuntimePlanInvalid,
       "ENDPOINT_CLOCK_HEALTH_RUNTIME_PLAN_INVALID",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kEndpointClockHealthSequenceInvalid,
       "ENDPOINT_CLOCK_HEALTH_SEQUENCE_INVALID",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kEndpointClockHealthUnspecifiedRejected,
       "ENDPOINT_CLOCK_HEALTH_UNSPECIFIED_REJECTED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kEndpointClockHealthUpdateCapacityExhausted,
       "ENDPOINT_CLOCK_HEALTH_UPDATE_CAPACITY_EXHAUSTED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeDiagnosticCapacityExhausted,
       "RUNTIME_DIAGNOSTIC_CAPACITY_EXHAUSTED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kTemporalSchedulingDomainUnsupported,
       "TEMPORAL_SCHEDULING_DOMAIN_UNSUPPORTED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kRuntimeSchedulingDomainMismatch,
       "RUNTIME_SCHEDULING_DOMAIN_MISMATCH",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kPhysicalTensorDeviceBindingInvalid,
       "PHYSICAL_TENSOR_DEVICE_BINDING_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredEvidence,
                               DiagnosticPayloadField::kObservedEvidence>},
      {StructuralDiagnosticId::kPhysicalTensorDeviceBindingUnexpected,
       "PHYSICAL_TENSOR_DEVICE_BINDING_UNEXPECTED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredEvidence,
                               DiagnosticPayloadField::kObservedEvidence>},
      {StructuralDiagnosticId::kPhysicalTensorDeviceBindingUnused,
       "PHYSICAL_TENSOR_DEVICE_BINDING_UNUSED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredEvidence,
                               DiagnosticPayloadField::kObservedEvidence>},
      {StructuralDiagnosticId::kPhysicalTensorPoolBoundsInvalid,
       "PHYSICAL_TENSOR_ALLOCATION_BOUNDS_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredEvidence,
                               DiagnosticPayloadField::kObservedEvidence>},
      {StructuralDiagnosticId::kPhysicalTensorMemoryKindUnresolved,
       "PHYSICAL_TENSOR_MEMORY_KIND_UNRESOLVED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredEvidence,
                               DiagnosticPayloadField::kObservedEvidence>},
      {StructuralDiagnosticId::kPhysicalTensorProcessGroupUnresolved,
       "PHYSICAL_TENSOR_PROCESS_GROUP_UNRESOLVED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredEvidence,
                               DiagnosticPayloadField::kObservedEvidence>},
      {StructuralDiagnosticId::kPhysicalTensorLogicalRequirementsInvalid,
       "PHYSICAL_TENSOR_LOGICAL_REQUIREMENTS_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredEvidence,
                               DiagnosticPayloadField::kObservedEvidence>},
      {StructuralDiagnosticId::kPhysicalTensorInterfaceTopologyInvalid,
       "PHYSICAL_TENSOR_INTERFACE_TOPOLOGY_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredEvidence,
                               DiagnosticPayloadField::kObservedEvidence>},
      {StructuralDiagnosticId::kPhysicalTensorLayoutMismatch,
       "PHYSICAL_TENSOR_LAYOUT_MISMATCH",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredEvidence,
                               DiagnosticPayloadField::kObservedEvidence>},
      {StructuralDiagnosticId::kPhysicalTensorProviderUnsupported,
       "PHYSICAL_TENSOR_PROVIDER_UNSUPPORTED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredEvidence,
                               DiagnosticPayloadField::kObservedEvidence>},
      {StructuralDiagnosticId::kPhysicalTensorRankUnsupported,
       "PHYSICAL_TENSOR_RANK_UNSUPPORTED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredEvidence,
                               DiagnosticPayloadField::kObservedEvidence>},
      {StructuralDiagnosticId::kDeadlineMiss, "DEADLINE_MISS", DiagnosticSeverity::kWarning, {}},
      {StructuralDiagnosticId::kClockImportDeclarationInvalid,
       "CLOCK_IMPORT_DECLARATION_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kClockImportDuplicate,
       "CLOCK_IMPORT_DUPLICATE",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kClockImportUnbound,
       "CLOCK_IMPORT_UNBOUND",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kClockImportBindingInvalid,
       "CLOCK_IMPORT_BINDING_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kParameterOverrideTargetSelectionMismatch,
       "PARAMETER_OVERRIDE_TARGET_SELECTION_MISMATCH",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kRuntimeDeadlineEvaluationFailed,
       "RUNTIME_DEADLINE_EVALUATION_FAILED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeComputeCudaDeviceUnavailable,
       "RUNTIME_COMPUTE_CUDA_DEVICE_UNAVAILABLE",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeOperatorReplayFingerprintMismatch,
       "RUNTIME_OPERATOR_REPLAY_FINGERPRINT_MISMATCH",
       DiagnosticSeverity::kError,
       detail::diagnostic_fields_v<detail::DiagnosticPayloadField::kRequiredEvidence,
                                   detail::DiagnosticPayloadField::kObservedEvidence>},
      {StructuralDiagnosticId::kRuntimeDeadlineFailRun,
       "RUNTIME_DEADLINE_FAIL_RUN",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeProcessGroupSelectionRequired,
       "RUNTIME_PROCESS_GROUP_SELECTION_REQUIRED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeParameterControlBindingFailed,
       "RUNTIME_PARAMETER_CONTROL_BINDING_FAILED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kGraphPartitionNameInvalid,
       "GRAPH_PARTITION_NAME_INVALID",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kGraphPartitionNameDuplicate,
       "GRAPH_PARTITION_NAME_DUPLICATE",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kGraphPartitionHandleInvalid,
       "GRAPH_PARTITION_HANDLE_INVALID",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kGraphPartitionMemberDuplicate,
       "GRAPH_PARTITION_MEMBER_DUPLICATE",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kPartitionManualAuthoringRequired,
       "PARTITION_MANUAL_AUTHORING_REQUIRED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPartitionManualEmbeddedGraphUnsupported,
       "PARTITION_MANUAL_EMBEDDED_GRAPH_UNSUPPORTED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPartitionPlacementUnknownOrDuplicate,
       "PARTITION_PLACEMENT_UNKNOWN_OR_DUPLICATE",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kGraphPartitionMemberForeign,
       "GRAPH_PARTITION_MEMBER_FOREIGN",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kChannelSchemaTensorFieldUnbounded,
       "CHANNEL_SCHEMA_TENSOR_FIELD_UNBOUNDED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredEvidence>},
      {StructuralDiagnosticId::kChannelSchemaTensorFieldUnsupportedMemory,
       "CHANNEL_SCHEMA_TENSOR_FIELD_UNSUPPORTED_MEMORY",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredEvidence>},
      {StructuralDiagnosticId::kChannelSchemaTensorBufferCountExceeded,
       "CHANNEL_SCHEMA_TENSOR_BUFFER_COUNT_EXCEEDED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields |
           diagnostic_fields_v<DiagnosticPayloadField::kRequiredCapacity>},
      {StructuralDiagnosticId::kCUDAGreenContextUnknownOrDuplicate,
       "CUDA_GREEN_CONTEXT_UNKNOWN_OR_DUPLICATE",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kCUDAGreenContextInvalidPlacement,
       "CUDA_GREEN_CONTEXT_INVALID_PLACEMENT",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kCUDAGreenContextPartitionMultiDevice,
       "CUDA_GREEN_CONTEXT_PARTITION_MULTI_DEVICE",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kCUDAGreenContextDeviceConflict,
       "CUDA_GREEN_CONTEXT_DEVICE_CONFLICT",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kCUDAGreenContextSplitGranularity,
       "CUDA_GREEN_CONTEXT_SPLIT_GRANULARITY",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kCUDAGreenContextSplitOvercommitted,
       "CUDA_GREEN_CONTEXT_SPLIT_OVERCOMMITTED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kCUDAGreenContextCapabilityUnsupported,
       "CUDA_GREEN_CONTEXT_CAPABILITY_UNSUPPORTED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kCUDAGreenContextGPUBackendConflict,
       "CUDA_GREEN_CONTEXT_GPU_BACKEND_CONFLICT",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kCUDAGreenContextDeviceOwnershipUnproven,
       "CUDA_GREEN_CONTEXT_DEVICE_OWNERSHIP_UNPROVEN",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kCUDAGreenContextPlanInvalid,
       "CUDA_GREEN_CONTEXT_PLAN_INVALID",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kCUDAGreenContextActiveOwnerConflict,
       "CUDA_GREEN_CONTEXT_ACTIVE_OWNER_CONFLICT",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kCUDAGreenContextCapabilityDrift,
       "CUDA_GREEN_CONTEXT_CAPABILITY_DRIFT",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kCUDAGreenContextMaterializationFailed,
       "CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kCUDAGreenContextStreamMismatch,
       "CUDA_GREEN_CONTEXT_STREAM_MISMATCH",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kCUDAGreenContextInactiveReservation,
       "CUDA_GREEN_CONTEXT_INACTIVE_RESERVATION",
       DiagnosticSeverity::kWarning,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kCUDAExecutionPlanInvalid,
       "CUDA_EXECUTION_PLAN_INVALID",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kPhysicalDescriptorTensorExternalTopicLoweringUnavailable,
       "PHYSICAL_DESCRIPTOR_TENSOR_EXTERNAL_TOPIC_LOWERING_UNAVAILABLE",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kClockConversionConflict,
       "CLOCK_CONVERSION_CONFLICT",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPhysicalTensorByteSpanDerivationInvalid,
       "PHYSICAL_TENSOR_BYTE_SPAN_DERIVATION_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPhysicalTensorByteSpanParameterTypeInvalid,
       "PHYSICAL_TENSOR_BYTE_SPAN_PARAMETER_TYPE_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPhysicalTensorByteSpanParameterMutabilityInvalid,
       "PHYSICAL_TENSOR_BYTE_SPAN_PARAMETER_MUTABILITY_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPhysicalTensorByteSpanValueInvalid,
       "PHYSICAL_TENSOR_BYTE_SPAN_VALUE_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPhysicalTensorByteSpanInputUnbounded,
       "PHYSICAL_TENSOR_BYTE_SPAN_INPUT_UNBOUNDED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPhysicalTensorByteSpanDependencyInvalid,
       "PHYSICAL_TENSOR_BYTE_SPAN_DEPENDENCY_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPhysicalTensorByteSpanOverflow,
       "PHYSICAL_TENSOR_BYTE_SPAN_OVERFLOW",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPhysicalTensorByteSpanContractConflict,
       "PHYSICAL_TENSOR_BYTE_SPAN_CONTRACT_CONFLICT",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kRuntimeManagedSignalReceived,
       "RUNTIME_MANAGED_SIGNAL_RECEIVED",
       DiagnosticSeverity::kWarning,
       diagnostic_fields_v<DiagnosticPayloadField::kSignal,
                           DiagnosticPayloadField::kOccurrence,
                           DiagnosticPayloadField::kSignalPhase,
                           DiagnosticPayloadField::kPriorDisposition,
                           DiagnosticPayloadField::kOriginallyBlocked,
                           DiagnosticPayloadField::kOwnerGeneration>},
      {StructuralDiagnosticId::kRuntimeManagedSignalOwnerBusy,
       "RUNTIME_MANAGED_SIGNAL_OWNER_BUSY",
       DiagnosticSeverity::kError,
       diagnostic_fields_v<DiagnosticPayloadField::kSignalPhase,
                           DiagnosticPayloadField::kOwnerGeneration>},
      {StructuralDiagnosticId::kRuntimeManagedSignalDomainRequired,
       "RUNTIME_MANAGED_SIGNAL_DOMAIN_REQUIRED",
       DiagnosticSeverity::kError,
       diagnostic_fields_v<DiagnosticPayloadField::kSignalPhase,
                           DiagnosticPayloadField::kSignalCensusReason>},
      {StructuralDiagnosticId::kRuntimeManagedSignalInstallFailed,
       "RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED",
       DiagnosticSeverity::kError,
       diagnostic_fields_v<DiagnosticPayloadField::kSignalPhase, DiagnosticPayloadField::kOsError>},
      {StructuralDiagnosticId::kRuntimeManagedSignalWaitFailed,
       "RUNTIME_MANAGED_SIGNAL_WAIT_FAILED",
       DiagnosticSeverity::kError,
       diagnostic_fields_v<DiagnosticPayloadField::kSignalPhase,
                           DiagnosticPayloadField::kOsError,
                           DiagnosticPayloadField::kOwnerGeneration>},
      {StructuralDiagnosticId::kRuntimeManagedSignalRestoreFailed,
       "RUNTIME_MANAGED_SIGNAL_RESTORE_FAILED",
       DiagnosticSeverity::kError,
       diagnostic_fields_v<DiagnosticPayloadField::kSignalPhase,
                           DiagnosticPayloadField::kOsError,
                           DiagnosticPayloadField::kOwnerGeneration>},
      {StructuralDiagnosticId::kRuntimeManagedSignalEscalated,
       "RUNTIME_MANAGED_SIGNAL_ESCALATED",
       DiagnosticSeverity::kWarning,
       diagnostic_fields_v<DiagnosticPayloadField::kSignal,
                           DiagnosticPayloadField::kOccurrence,
                           DiagnosticPayloadField::kSignalPhase,
                           DiagnosticPayloadField::kPriorDisposition,
                           DiagnosticPayloadField::kOriginallyBlocked,
                           DiagnosticPayloadField::kOwnerGeneration>},
      // REQ: CONTRACT-020
      {StructuralDiagnosticId::kOperatorCompletionPlacementUnsupported,
       "OPERATOR_COMPLETION_PLACEMENT_UNSUPPORTED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPartitionEnvironmentSelectorRemoved,
       "PARTITION_ENVIRONMENT_SELECTOR_REMOVED",
       DiagnosticSeverity::kError,
       detail::kPartitionCommonFields},
      {StructuralDiagnosticId::kGraphPartitionAutomaticRemainderEmpty,
       "GRAPH_PARTITION_AUTOMATIC_REMAINDER_EMPTY",
       DiagnosticSeverity::kError,
       detail::kPartitionCommonFields},
      {StructuralDiagnosticId::kGraphPartitionAutomaticRemainderConflict,
       "GRAPH_PARTITION_AUTOMATIC_REMAINDER_CONFLICT",
       DiagnosticSeverity::kError,
       detail::kPartitionCommonFields},
      {StructuralDiagnosticId::kGraphPartitionIncludeAllConflict,
       "GRAPH_PARTITION_INCLUDE_ALL_CONFLICT",
       DiagnosticSeverity::kError,
       detail::kPartitionCommonFields},
      {StructuralDiagnosticId::kPartitionAnchorNonConvex,
       "PARTITION_ANCHOR_NON_CONVEX",
       DiagnosticSeverity::kError,
       detail::kPartitionCommonFields},
      {StructuralDiagnosticId::kPartitionAnchorDomainIncompatible,
       "PARTITION_ANCHOR_DOMAIN_INCOMPATIBLE",
       DiagnosticSeverity::kError,
       detail::kPartitionCommonFields},
      {StructuralDiagnosticId::kPartitionProposalConstraintViolation,
       "PARTITION_PROPOSAL_CONSTRAINT_VIOLATION",
       DiagnosticSeverity::kError,
       detail::kPartitionCommonFields},
      {StructuralDiagnosticId::kExternalChannelPlanInvalid,
       "EXTERNAL_CHANNEL_PLAN_INVALID",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kPartitionInspectionPlanInvalid,
       "PARTITION_INSPECTION_PLAN_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kRateHeadroomInvalid,
       "RATE_HEADROOM_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kRateDeclarationInvalid,
       "RATE_DECLARATION_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kRateFollowsInvalid,
       "RATE_FOLLOWS_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kRateDependencyCycle,
       "RATE_DEPENDENCY_CYCLE",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kRateArithmeticOverflow,
       "RATE_ARITHMETIC_OVERFLOW",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kRateCapacityOverflow,
       "RATE_CAPACITY_OVERFLOW",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kRateExpectationMismatch,
       "RATE_EXPECTATION_MISMATCH",
       DiagnosticSeverity::kWarning,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kRateQueueDepthBelowHardFloor,
       "RATE_QUEUE_DEPTH_BELOW_HARD_FLOOR",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kRateQueueDepthBelowRecommendation,
       "RATE_QUEUE_DEPTH_BELOW_RECOMMENDATION",
       DiagnosticSeverity::kWarning,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kRateBoundedEvidenceRequired,
       "RATE_BOUNDED_EVIDENCE_REQUIRED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kRateSustainedBackpressure,
       "RATE_SUSTAINED_BACKPRESSURE",
       DiagnosticSeverity::kWarning,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kRatePredictedReplacement,
       "RATE_PREDICTED_REPLACEMENT",
       DiagnosticSeverity::kWarning,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kQueueActiveDepthNotEnforceable,
       "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kQueuePreallocationBoundRequired,
       "QUEUE_PREALLOCATION_BOUND_REQUIRED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kRatePlanRecordsInvalid,
       "RATE_PLAN_RECORDS_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPoolCapacityPlanInvalid,
       "POOL_CAPACITY_PLAN_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPhysicalTensorFieldAllocationOrderInvalid,
       "PHYSICAL_TENSOR_FIELD_ALLOCATION_ORDER_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kRateGenerationRejected,
       "RATE_GENERATION_REJECTED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kRateProviderEvidenceMissing,
       "RATE_PROVIDER_EVIDENCE_MISSING",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kQueueReplacementNotAdmitted,
       "QUEUE_REPLACEMENT_NOT_ADMITTED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPortRateBindingMissing,
       "PORT_RATE_BINDING_MISSING",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPortRateBindingConflict,
       "PORT_RATE_BINDING_CONFLICT",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPortRateBindingAboveMaximum,
       "PORT_RATE_BINDING_ABOVE_MAXIMUM",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kQueueRateMaterializationInvalid,
       "QUEUE_RATE_MATERIALIZATION_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPortRateArithmeticOverflow,
       "PORT_RATE_ARITHMETIC_OVERFLOW",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPortRateBindingInvalid,
       "PORT_RATE_BINDING_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPortRateBindNotAuthorized,
       "PORT_RATE_BIND_NOT_AUTHORIZED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPortRateBindStageInvalid,
       "PORT_RATE_BIND_STAGE_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPortRateCandidateActive,
       "PORT_RATE_CANDIDATE_ACTIVE",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPortRateCandidateMissing,
       "PORT_RATE_CANDIDATE_MISSING",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPortRateConflict,
       "PORT_RATE_CONFLICT",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPortRateDeclarationAmbiguous,
       "PORT_RATE_DECLARATION_AMBIGUOUS",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPortRateGenerationExhausted,
       "PORT_RATE_GENERATION_EXHAUSTED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPortRateGenerationSlotsExhausted,
       "PORT_RATE_GENERATION_SLOTS_EXHAUSTED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPortRateIndexExhausted,
       "PORT_RATE_INDEX_EXHAUSTED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPortRateInvalid,
       "PORT_RATE_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPortRatePlanInvalid,
       "PORT_RATE_PLAN_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPortRatePublicationMissing,
       "PORT_RATE_PUBLICATION_MISSING",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPortRateReferenceInvalid,
       "PORT_RATE_REFERENCE_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPortRateResolutionCycle,
       "PORT_RATE_RESOLUTION_CYCLE",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPortRateViewInvalid,
       "PORT_RATE_VIEW_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kQueueAllocationGenerationInvalid,
       "QUEUE_ALLOCATION_GENERATION_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kQueueCapacityOverflow,
       "QUEUE_CAPACITY_OVERFLOW",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kQueueDepthBelowHardMinimum,
       "QUEUE_DEPTH_BELOW_HARD_MINIMUM",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kQueuePoolCapacityUnbounded,
       "QUEUE_POOL_CAPACITY_UNBOUNDED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kQueuePoolCapacityUnresolved,
       "QUEUE_POOL_CAPACITY_UNRESOLVED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kQueuePoolDependencyInvalid,
       "QUEUE_POOL_DEPENDENCY_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kQueueRateUnresolvedFallback,
       "QUEUE_RATE_UNRESOLVED_FALLBACK",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kQueueSizingRecipeMissing,
       "QUEUE_SIZING_RECIPE_MISSING",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kRateGenerationSnapshotAllocationFailed,
       "RATE_GENERATION_SNAPSHOT_ALLOCATION_FAILED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kRateGenerationSnapshotFailed,
       "RATE_GENERATION_SNAPSHOT_FAILED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kRateQueueInspectionPlanMismatch,
       "RATE_QUEUE_INSPECTION_PLAN_MISMATCH",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kRateQueueInspectionGenerationUnknown,
       "RATE_QUEUE_INSPECTION_GENERATION_UNKNOWN",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kRateQueueInspectionEdgeUnknown,
       "RATE_QUEUE_INSPECTION_EDGE_UNKNOWN",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kRateQueueInspectionJoinInvalid,
       "RATE_QUEUE_INSPECTION_JOIN_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kRateQueueInspectionAllocationFailed,
       "RATE_QUEUE_INSPECTION_ALLOCATION_FAILED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kRateQueueInspectionFailed,
       "RATE_QUEUE_INSPECTION_FAILED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kRateLifecycleCrossProcessUnsupported,
       "RATE_LIFECYCLE_CROSS_PROCESS_UNSUPPORTED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kParameterInputRouteConflict,
       "PARAMETER_INPUT_ROUTE_CONFLICT",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kParameterRebindScopeIncomplete,
       "PARAMETER_REBIND_SCOPE_INCOMPLETE",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kParameterRebindScopeInvalid,
       "PARAMETER_REBIND_SCOPE_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kParameterRebindProfileMismatch,
       "PARAMETER_REBIND_PROFILE_MISMATCH",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kGraphPartitionGraphInvalid,
       "GRAPH_PARTITION_GRAPH_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kGraphPartitionGraphEmpty,
       "GRAPH_PARTITION_GRAPH_EMPTY",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kGraphPartitionInternalDeclaration,
       "GRAPH_PARTITION_INTERNAL_DECLARATION",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPartitionEmbeddedInstanceUnassigned,
       "PARTITION_EMBEDDED_INSTANCE_UNASSIGNED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      // REQ: COMPILE-076
      {StructuralDiagnosticId::kPhysicalTensorMemoryKindDerivationInvalid,
       "PHYSICAL_TENSOR_MEMORY_KIND_DERIVATION_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPhysicalTensorMemoryKindDependencyInvalid,
       "PHYSICAL_TENSOR_MEMORY_KIND_DEPENDENCY_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPhysicalTensorMemoryKindParameterMutabilityInvalid,
       "PHYSICAL_TENSOR_MEMORY_KIND_PARAMETER_MUTABILITY_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPhysicalTensorMemoryKindParameterTypeInvalid,
       "PHYSICAL_TENSOR_MEMORY_KIND_PARAMETER_TYPE_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPhysicalTensorMemoryKindMappingInvalid,
       "PHYSICAL_TENSOR_MEMORY_KIND_MAPPING_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPhysicalTensorMemoryKindValueInvalid,
       "PHYSICAL_TENSOR_MEMORY_KIND_VALUE_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kPhysicalTensorMemoryKindContractConflict,
       "PHYSICAL_TENSOR_MEMORY_KIND_CONTRACT_CONFLICT",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kTensorMemoryKindDerivationPlanInvalid,
       "TENSOR_MEMORY_KIND_DERIVATION_PLAN_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      // REQ: COMPILE-077
      {StructuralDiagnosticId::kPhysicalTensorByteSpanProjectionFailed,
       "PHYSICAL_TENSOR_BYTE_SPAN_PROJECTION_FAILED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kTensorByteSpanProjectionPlanInvalid,
       "TENSOR_BYTE_SPAN_PROJECTION_PLAN_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kCudaGraphRegionAnalysisUnavailable,
       "CUDA_GRAPH_REGION_ANALYSIS_UNAVAILABLE",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kRuntimeComputeCudaDeviceRestoreFailed,
       "RUNTIME_COMPUTE_CUDA_DEVICE_RESTORE_FAILED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kRuntimeComputeDispatchFailed,
       "RUNTIME_COMPUTE_DISPATCH_FAILED",
       DiagnosticSeverity::kError,
       {}},
      {StructuralDiagnosticId::kCUDAGraphRegionInvalid,
       "CUDA_GRAPH_REGION_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kCUDAGraphExecutionLoweringUnavailable,
       "CUDA_GRAPH_EXECUTION_LOWERING_UNAVAILABLE",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kTemporalInputUnconnectedSafeDefault,
       "TEMPORAL_INPUT_UNCONNECTED_SAFE_DEFAULT",
       DiagnosticSeverity::kWarning,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kChannelLocalEnumCrossProcessUnsupported,
       "CHANNEL_LOCAL_ENUM_CROSS_PROCESS_UNSUPPORTED",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
      {StructuralDiagnosticId::kTemporalInputOptionalInvalid,
       "TEMPORAL_INPUT_OPTIONAL_INVALID",
       DiagnosticSeverity::kError,
       detail::kCompileLoweringCommonFields},
  }};

  static_assert(detail::structural_diagnostic_registry_is_unique());
  static_assert(expected.size() == detail::kStructuralDiagnosticRegistry.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    SCOPED_TRACE(index);
    const auto& actual = detail::kStructuralDiagnosticRegistry[index];
    EXPECT_EQ(actual.id, expected[index].id);
    EXPECT_EQ(actual.token, expected[index].token);
    EXPECT_EQ(actual.severity, expected[index].severity);
    EXPECT_EQ(actual.required_fields, expected[index].required_fields);
    EXPECT_EQ(detail::find_structural_diagnostic(expected[index].id), &actual);
    EXPECT_EQ(detail::diagnostic_id(expected[index].token),
              DiagnosticId{static_cast<std::uint32_t>(expected[index].id)});
  }
  EXPECT_EQ(detail::diagnostic_id("UNREGISTERED_DIAGNOSTIC"), DiagnosticId{});
  const auto* generic =
      detail::find_structural_diagnostic(StructuralDiagnosticId::kRuntimeOperatorReplayMismatch);
  const auto* fingerprint = detail::find_structural_diagnostic(
      StructuralDiagnosticId::kRuntimeOperatorReplayFingerprintMismatch);
  ASSERT_NE(generic, nullptr);
  ASSERT_NE(fingerprint, nullptr);
  EXPECT_FALSE(detail::requires_field(*generic, DiagnosticPayloadField::kRequiredEvidence));
  EXPECT_FALSE(detail::requires_field(*generic, DiagnosticPayloadField::kObservedEvidence));
  EXPECT_TRUE(detail::requires_field(*fingerprint, DiagnosticPayloadField::kRequiredEvidence));
  EXPECT_TRUE(detail::requires_field(*fingerprint, DiagnosticPayloadField::kObservedEvidence));

  const auto* received =
      detail::find_structural_diagnostic(StructuralDiagnosticId::kRuntimeManagedSignalReceived);
  const auto* domain_required = detail::find_structural_diagnostic(
      StructuralDiagnosticId::kRuntimeManagedSignalDomainRequired);
  const auto* install_failed = detail::find_structural_diagnostic(
      StructuralDiagnosticId::kRuntimeManagedSignalInstallFailed);
  const auto* escalated =
      detail::find_structural_diagnostic(StructuralDiagnosticId::kRuntimeManagedSignalEscalated);
  ASSERT_NE(received, nullptr);
  ASSERT_NE(domain_required, nullptr);
  ASSERT_NE(install_failed, nullptr);
  ASSERT_NE(escalated, nullptr);
  EXPECT_TRUE(detail::requires_field(*received, DiagnosticPayloadField::kSignal));
  EXPECT_TRUE(detail::requires_field(*received, DiagnosticPayloadField::kOccurrence));
  EXPECT_TRUE(detail::requires_field(*received, DiagnosticPayloadField::kPriorDisposition));
  EXPECT_TRUE(detail::requires_field(*received, DiagnosticPayloadField::kOriginallyBlocked));
  EXPECT_TRUE(detail::requires_field(*received, DiagnosticPayloadField::kOwnerGeneration));
  EXPECT_TRUE(
      detail::requires_field(*domain_required, DiagnosticPayloadField::kSignalCensusReason));
  EXPECT_FALSE(detail::requires_field(*domain_required, DiagnosticPayloadField::kOwnerGeneration));
  EXPECT_TRUE(detail::requires_field(*install_failed, DiagnosticPayloadField::kOsError));
  EXPECT_FALSE(detail::requires_field(*install_failed, DiagnosticPayloadField::kSignal));
  EXPECT_TRUE(detail::requires_field(*escalated, DiagnosticPayloadField::kSignal));
  EXPECT_TRUE(detail::requires_field(*escalated, DiagnosticPayloadField::kOccurrence));
}

}  // namespace
}  // namespace holoscan
