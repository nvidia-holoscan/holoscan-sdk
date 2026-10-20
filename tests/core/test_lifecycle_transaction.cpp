/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include "../common/allocation_counter.hpp"
#include "core/lifecycle_transaction_kernel.hpp"

namespace holoscan::test {
namespace {

using detail::LifecycleTransactionCapacityKind;
using detail::LifecycleTransactionCompatibilityKey;
using detail::LifecycleTransactionGate;
using detail::LifecycleTransactionKernel;
using detail::LifecycleTransactionKernelCapacity;
using detail::LifecycleTransactionKernelErrorCode;
using detail::LifecycleTransactionRequest;
using detail::LifecycleTransactionRequestOrigin;
using detail::LifecycleTransactionTargetState;

constexpr std::size_t kStructuralEventsPerRequest = 8U;

// Builds a LifecycleTransactionKernelCapacity with every limit set to `value` (default 16).
// Tests that probe a specific capacity constraint override one field after calling this.
// Representative use: RejectsDeltaCapacityWithoutPartialFenceOrReceipt.
[[nodiscard]] LifecycleTransactionKernelCapacity capacity(std::size_t value = 16U) {
  return {
      .max_pending_requests = value,
      .max_pending_transactions = value,
      .max_receipts = value,
      .max_fenced_operators = value,
      .max_candidate_generations = value,
      .max_events = value * kStructuralEventsPerRequest,
      .max_audit_records = value,
      .max_quiesce_records = value,
      .max_rollback_records = value,
  };
}

// Builds a LifecycleTransactionRequest targeting `target` with the given closure and stage.
// The compatibility key fields are fixed constants so all requests with the same
// `compatibility` argument will coalesce into the same transaction unless explicitly
// differed. See IncompatibleClosureKeysDoNotCoalesce.
[[nodiscard]] LifecycleTransactionRequest request(
    LifecycleTransactionRequestOrigin origin, OperatorInstanceId target,
    std::span<const OperatorInstanceId> closure, LifecycleStage requested = LifecycleStage::kArm,
    LifecycleStage effective = LifecycleStage::kArm, ParamGeneration candidate = 0U,
    bool equivalent = false, std::uint64_t compatibility = 1U) {
  return {
      .origin = origin,
      .target = {.id = target, .plan_generation = 7U, .run_generation = 9U},
      .requested_stage = requested,
      .effective_stage = effective,
      .compatibility =
          LifecycleTransactionCompatibilityKey{
              .authorization = compatibility,
              .profile = 2U,
              .rollback = 3U,
              .deadline = 4U,
              .closure = 5U,
          },
      .closure = closure,
      .candidate_generation = candidate,
      .parameter_candidate_equivalent = equivalent,
      .target_state = LifecycleTransactionTargetState::kIdle,
      .structural_event_delta = kStructuralEventsPerRequest,
  };
}

// Extracts the LifecycleTransactionId from a receipt via inspect(). Asserts that
// inspect() succeeds so failures surface at the call site rather than inside helpers.
[[nodiscard]] LifecycleTransactionId transaction_id(const LifecycleTransactionKernel& kernel,
                                                    LifecycleRestartReceipt receipt) {
  const auto view = kernel.inspect(receipt);
  EXPECT_TRUE(view.has_value());
  return view ? view->transaction_id : 0U;
}

// REQ: LIFECYCLE-008
TEST(LifecycleTransaction, NoChangeIsTerminalBeforeFenceButExplicitIntentReconstructsEpoch) {
  LifecycleTransactionKernel kernel(capacity());
  constexpr std::array<OperatorInstanceId, 1> closure{1U};

  const auto no_change = kernel.submit(request(LifecycleTransactionRequestOrigin::kParameter,
                                               1U,
                                               closure,
                                               LifecycleStage::kConfigure,
                                               LifecycleStage::kConfigure,
                                               11U,
                                               true));
  ASSERT_TRUE(no_change);
  const auto no_change_view = kernel.inspect(*no_change);
  ASSERT_TRUE(no_change_view);
  EXPECT_EQ(no_change_view->state, LifecycleTransactionState::kAcceptedNoChange);
  EXPECT_EQ(no_change_view->transaction_id, 0U);
  EXPECT_EQ(no_change_view->fence_id, 0U);
  EXPECT_FALSE(kernel.is_fenced(1U));
  EXPECT_EQ(kernel.pending_request_count(), 0U);
  EXPECT_EQ(kernel.pending_transaction_count(), 0U);

  const auto explicit_restart =
      kernel.submit(request(LifecycleTransactionRequestOrigin::kComputeSelf,
                            1U,
                            closure,
                            LifecycleStage::kConfigure,
                            LifecycleStage::kConfigure));
  ASSERT_TRUE(explicit_restart);
  const auto explicit_view = kernel.inspect(*explicit_restart);
  ASSERT_TRUE(explicit_view);
  EXPECT_EQ(explicit_view->state, LifecycleTransactionState::kQueued);
  EXPECT_NE(explicit_view->transaction_id, 0U);
  EXPECT_NE(explicit_view->fence_id, 0U);
  EXPECT_TRUE(kernel.is_fenced(1U));

  auto invalid = request(LifecycleTransactionRequestOrigin::kComputeSelf, 1U, closure);
  invalid.requested_stage = LifecycleStage::kStop;
  const auto rejected = kernel.submit(invalid);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, LifecycleTransactionKernelErrorCode::kInvalidRestartStage);
  EXPECT_EQ(kernel.pending_request_count(), 1U);
}

// REQ: LIFECYCLE-008
TEST(LifecycleTransaction, EquivalentCandidateSupersedesQueuedCandidateWithoutStalePublication) {
  LifecycleTransactionKernel kernel(capacity());
  constexpr std::array<OperatorInstanceId, 1> closure{1U};

  const auto queued = kernel.submit(request(LifecycleTransactionRequestOrigin::kParameter,
                                            1U,
                                            closure,
                                            LifecycleStage::kArm,
                                            LifecycleStage::kArm,
                                            10U));
  ASSERT_TRUE(queued);
  const LifecycleTransactionId transaction = transaction_id(kernel, *queued);
  EXPECT_TRUE(kernel.is_fenced(1U));

  const auto equivalent = kernel.submit(request(LifecycleTransactionRequestOrigin::kParameter,
                                                1U,
                                                closure,
                                                LifecycleStage::kArm,
                                                LifecycleStage::kArm,
                                                11U,
                                                true));
  ASSERT_TRUE(equivalent);
  const auto queued_view = kernel.inspect(*queued);
  const auto equivalent_view = kernel.inspect(*equivalent);
  ASSERT_TRUE(queued_view);
  ASSERT_TRUE(equivalent_view);
  EXPECT_EQ(queued_view->state, LifecycleTransactionState::kSuperseded);
  EXPECT_EQ(queued_view->coalesced_into_transaction, transaction);
  EXPECT_EQ(equivalent_view->state, LifecycleTransactionState::kAcceptedNoChange);
  EXPECT_EQ(equivalent_view->transaction_id, 0U);
  EXPECT_EQ(equivalent_view->fence_id, 0U);
  EXPECT_FALSE(kernel.transaction_state(transaction));
  EXPECT_FALSE(kernel.is_fenced(1U));
  EXPECT_EQ(kernel.pending_request_count(), 0U);
  EXPECT_EQ(kernel.pending_transaction_count(), 0U);
}

// REQ: LIFECYCLE-008
TEST(LifecycleTransaction, CoalescesLatestCandidateEarliestStageAndAtomicClosureExpansion) {
  LifecycleTransactionKernel kernel(capacity());
  constexpr std::array<OperatorInstanceId, 2> first_closure{1U, 2U};
  constexpr std::array<OperatorInstanceId, 3> expanded_closure{1U, 2U, 3U};

  // Key setup: two requests targeting operator 1 are coalesced into one transaction.
  // The second request expands the closure from {1,2} to {1,2,3} and promotes the
  // effective stage from kArm to kAllocate. Both receipts share the same transaction_id.
  const auto first = kernel.submit(request(LifecycleTransactionRequestOrigin::kParameter,
                                           1U,
                                           first_closure,
                                           LifecycleStage::kArm,
                                           LifecycleStage::kArm,
                                           10U));
  ASSERT_TRUE(first);
  const LifecycleTransactionId transaction = transaction_id(kernel, *first);
  const auto initial_view = kernel.inspect(*first);
  ASSERT_TRUE(initial_view);
  EXPECT_EQ(initial_view->effective_stage, LifecycleStage::kArm);

  const auto second = kernel.submit(request(LifecycleTransactionRequestOrigin::kParameter,
                                            1U,
                                            expanded_closure,
                                            LifecycleStage::kAllocate,
                                            LifecycleStage::kAllocate,
                                            11U));
  ASSERT_TRUE(second);
  const auto first_view = kernel.inspect(*first);
  const auto second_view = kernel.inspect(*second);
  ASSERT_TRUE(first_view);
  ASSERT_TRUE(second_view);
  EXPECT_EQ(first_view->state, LifecycleTransactionState::kSuperseded);
  EXPECT_EQ(first_view->coalesced_into_transaction, transaction);
  EXPECT_EQ(second_view->transaction_id, transaction);
  EXPECT_EQ(second_view->coalesced_into_transaction, transaction);
  EXPECT_EQ(first_view->effective_stage, LifecycleStage::kAllocate);
  EXPECT_EQ(second_view->effective_stage, LifecycleStage::kAllocate);
  EXPECT_EQ(kernel.candidate_generation(transaction, 1U), 11U);
  EXPECT_EQ(kernel.effective_stage(transaction), LifecycleStage::kAllocate);
  EXPECT_TRUE(kernel.is_fenced(1U));
  EXPECT_TRUE(kernel.is_fenced(2U));
  EXPECT_TRUE(kernel.is_fenced(3U));
  EXPECT_EQ(kernel.pending_request_count(), 1U);
  EXPECT_EQ(kernel.pending_transaction_count(), 1U);

  ASSERT_TRUE(
      kernel.acknowledge_gate(transaction, 1U, LifecycleTransactionGate::kControlSafePoint));
  ASSERT_TRUE(kernel.begin_quiescing(transaction));
  ASSERT_TRUE(kernel.begin_restarting(transaction));
  const auto transaction_members = kernel.transaction_members(transaction);
  ASSERT_TRUE(transaction_members);
  EXPECT_TRUE(std::equal(transaction_members->begin(),
                         transaction_members->end(),
                         expanded_closure.begin(),
                         expanded_closure.end()));
  ASSERT_TRUE(kernel.publish(transaction));
  // Key assertion: after publish both receipts report the promoted effective stage.
  const auto first_terminal_view = kernel.inspect(*first);
  const auto second_terminal_view = kernel.inspect(*second);
  ASSERT_TRUE(first_terminal_view);
  ASSERT_TRUE(second_terminal_view);
  EXPECT_EQ(first_terminal_view->effective_stage, LifecycleStage::kAllocate);
  EXPECT_EQ(second_terminal_view->effective_stage, LifecycleStage::kAllocate);
}

// REQ: LIFECYCLE-008
TEST(LifecycleTransaction, CoalescedReceiptReportsExistingEarlierEffectiveStage) {
  LifecycleTransactionKernel kernel(capacity());
  constexpr std::array<OperatorInstanceId, 1> first_closure{1U};
  constexpr std::array<OperatorInstanceId, 1> second_closure{2U};

  const auto first = kernel.submit(request(LifecycleTransactionRequestOrigin::kExternalControl,
                                           1U,
                                           first_closure,
                                           LifecycleStage::kConfigure,
                                           LifecycleStage::kConfigure));
  ASSERT_TRUE(first);
  const LifecycleTransactionId transaction = transaction_id(kernel, *first);
  const auto second = kernel.submit(request(LifecycleTransactionRequestOrigin::kExternalControl,
                                            2U,
                                            second_closure,
                                            LifecycleStage::kArm,
                                            LifecycleStage::kArm));
  ASSERT_TRUE(second);

  const auto first_view = kernel.inspect(*first);
  const auto second_view = kernel.inspect(*second);
  ASSERT_TRUE(first_view);
  ASSERT_TRUE(second_view);
  EXPECT_EQ(second_view->transaction_id, transaction);
  EXPECT_EQ(kernel.effective_stage(transaction), LifecycleStage::kConfigure);
  EXPECT_EQ(first_view->effective_stage, LifecycleStage::kConfigure);
  EXPECT_EQ(second_view->effective_stage, LifecycleStage::kConfigure);
}

// REQ: LIFECYCLE-008
TEST(LifecycleTransaction, IncompatibleClosureKeysDoNotCoalesce) {
  LifecycleTransactionKernel kernel(capacity());
  constexpr std::array<OperatorInstanceId, 1> first_closure{1U};
  constexpr std::array<OperatorInstanceId, 1> second_closure{2U};

  const auto first =
      kernel.submit(request(LifecycleTransactionRequestOrigin::kComputeSelf, 1U, first_closure));
  auto incompatible = request(LifecycleTransactionRequestOrigin::kComputeSelf, 2U, second_closure);
  incompatible.compatibility.closure = 6U;
  const auto second = kernel.submit(incompatible);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  const auto first_view = kernel.inspect(*first);
  const auto second_view = kernel.inspect(*second);
  ASSERT_TRUE(first_view);
  ASSERT_TRUE(second_view);
  EXPECT_NE(first_view->transaction_id, second_view->transaction_id);
  EXPECT_EQ(first_view->coalesced_into_transaction, 0U);
  EXPECT_EQ(second_view->coalesced_into_transaction, 0U);
  EXPECT_EQ(kernel.pending_transaction_count(), 2U);
}

// REQ: LIFECYCLE-008
TEST(LifecycleTransaction, QuiescingCutsOffCoalescingAndOnlyOneFifoSuccessorIsAllowed) {
  LifecycleTransactionKernel kernel(capacity());
  constexpr std::array<OperatorInstanceId, 1> first_closure{1U};
  constexpr std::array<OperatorInstanceId, 1> successor_closure{2U};
  constexpr std::array<OperatorInstanceId, 1> third_closure{3U};

  // Key setup: once the first transaction enters quiescing, new requests form a successor
  // transaction rather than coalescing. The successor cannot quiesce until the first
  // publishes (FIFO ordering). A third incompatible request is rejected outright because
  // only one successor generation is allowed at a time.
  const auto first =
      kernel.submit(request(LifecycleTransactionRequestOrigin::kComputeSelf, 1U, first_closure));
  ASSERT_TRUE(first);
  const LifecycleTransactionId first_id = transaction_id(kernel, *first);
  ASSERT_TRUE(kernel.acknowledge_gate(first_id, 1U, LifecycleTransactionGate::kControlSafePoint));
  ASSERT_TRUE(kernel.begin_quiescing(first_id));

  // A matching compatibility key cannot join a transaction after it starts quiescing.
  const auto successor = kernel.submit(
      request(LifecycleTransactionRequestOrigin::kComputeSelf, 2U, successor_closure));
  ASSERT_TRUE(successor);
  const LifecycleTransactionId successor_id = transaction_id(kernel, *successor);
  EXPECT_NE(successor_id, first_id);
  ASSERT_TRUE(
      kernel.acknowledge_gate(successor_id, 2U, LifecycleTransactionGate::kControlSafePoint));
  const auto out_of_order = kernel.begin_quiescing(successor_id);
  ASSERT_FALSE(out_of_order);
  EXPECT_EQ(out_of_order.error().code, LifecycleTransactionKernelErrorCode::kInvalidTransition);

  // A different closure contract cannot create a third retained transaction generation.
  auto incompatible = request(LifecycleTransactionRequestOrigin::kComputeSelf, 3U, third_closure);
  incompatible.compatibility.closure = 6U;
  const auto third = kernel.submit(incompatible);
  ASSERT_FALSE(third);
  EXPECT_EQ(third.error().code, LifecycleTransactionKernelErrorCode::kSuccessorUnavailable);
  EXPECT_EQ(kernel.pending_transaction_count(), 2U);

  ASSERT_TRUE(kernel.begin_restarting(first_id));
  ASSERT_TRUE(kernel.publish(first_id));
  EXPECT_TRUE(kernel.begin_quiescing(successor_id));
}

// REQ: LIFECYCLE-008
TEST(LifecycleTransaction, LaterEffectiveStageRejectsWithoutConsumingAnyIdentity) {
  LifecycleTransactionKernel kernel(capacity());
  constexpr std::array<OperatorInstanceId, 1> closure{1U};

  const auto invalid = request(LifecycleTransactionRequestOrigin::kComputeSelf,
                               1U,
                               closure,
                               LifecycleStage::kAllocate,
                               LifecycleStage::kArm);
  const auto rejected_first = kernel.submit(invalid);
  ASSERT_FALSE(rejected_first);
  EXPECT_EQ(rejected_first.error().code, LifecycleTransactionKernelErrorCode::kInvalidRequest);

  const auto first =
      kernel.submit(request(LifecycleTransactionRequestOrigin::kComputeSelf, 1U, closure));
  ASSERT_TRUE(first);
  const auto first_view = kernel.inspect(*first);
  ASSERT_TRUE(first_view);
  EXPECT_EQ(first->receipt_id, 1U);
  EXPECT_EQ(first->acceptance_sequence, 1U);
  EXPECT_EQ(first_view->transaction_id, 1U);
  EXPECT_EQ(first_view->fence_id, 1U);
  ASSERT_TRUE(kernel.acknowledge_gate(
      first_view->transaction_id, 1U, LifecycleTransactionGate::kControlSafePoint));
  ASSERT_TRUE(kernel.begin_quiescing(first_view->transaction_id));
  ASSERT_TRUE(kernel.begin_restarting(first_view->transaction_id));
  ASSERT_TRUE(kernel.publish(first_view->transaction_id));

  const auto rejected_second = kernel.submit(invalid);
  ASSERT_FALSE(rejected_second);
  EXPECT_EQ(rejected_second.error().code, LifecycleTransactionKernelErrorCode::kInvalidRequest);

  const auto second =
      kernel.submit(request(LifecycleTransactionRequestOrigin::kComputeSelf, 1U, closure));
  ASSERT_TRUE(second);
  const auto second_view = kernel.inspect(*second);
  ASSERT_TRUE(second_view);
  EXPECT_EQ(second->receipt_id, 2U);
  EXPECT_EQ(second->acceptance_sequence, 2U);
  EXPECT_EQ(second_view->transaction_id, 2U);
  EXPECT_EQ(second_view->fence_id, 2U);
}

// REQ: LIFECYCLE-008
TEST(LifecycleTransaction, RejectsDeltaCapacityWithoutPartialFenceOrReceipt) {
  auto limits = capacity();
  limits.max_fenced_operators = 2U;
  limits.max_quiesce_records = 3U;
  limits.max_rollback_records = 3U;
  LifecycleTransactionKernel kernel(limits);
  constexpr std::array<OperatorInstanceId, 2> first_closure{1U, 2U};
  constexpr std::array<OperatorInstanceId, 3> expanded_closure{1U, 2U, 3U};
  const auto first = kernel.submit(request(LifecycleTransactionRequestOrigin::kParameter,
                                           1U,
                                           first_closure,
                                           LifecycleStage::kArm,
                                           LifecycleStage::kArm,
                                           10U));
  ASSERT_TRUE(first);

  const auto rejected = kernel.submit(request(LifecycleTransactionRequestOrigin::kParameter,
                                              1U,
                                              expanded_closure,
                                              LifecycleStage::kAllocate,
                                              LifecycleStage::kAllocate,
                                              11U));
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, LifecycleTransactionKernelErrorCode::kCapacityExceeded);
  EXPECT_EQ(rejected.error().capacity, LifecycleTransactionCapacityKind::kFencedOperators);
  const auto first_view = kernel.inspect(*first);
  ASSERT_TRUE(first_view);
  EXPECT_EQ(first_view->state, LifecycleTransactionState::kQueued);
  EXPECT_EQ(kernel.candidate_generation(first_view->transaction_id, 1U), 10U);
  EXPECT_TRUE(kernel.is_fenced(1U));
  EXPECT_TRUE(kernel.is_fenced(2U));
  EXPECT_FALSE(kernel.is_fenced(3U));
  EXPECT_EQ(kernel.pending_request_count(), 1U);
}

// REQ: LIFECYCLE-008
TEST(LifecycleTransaction, InheritsOverlapFenceAcrossOneBoundedSuccessor) {
  LifecycleTransactionKernel kernel(capacity());
  constexpr std::array<OperatorInstanceId, 2> current_closure{1U, 2U};
  constexpr std::array<OperatorInstanceId, 2> successor_closure{2U, 3U};
  const auto current =
      kernel.submit(request(LifecycleTransactionRequestOrigin::kComputeSelf, 1U, current_closure));
  ASSERT_TRUE(current);
  const LifecycleTransactionId current_id = transaction_id(kernel, *current);
  ASSERT_TRUE(kernel.acknowledge_gate(current_id, 1U, LifecycleTransactionGate::kControlSafePoint));
  ASSERT_TRUE(kernel.begin_quiescing(current_id));
  ASSERT_TRUE(kernel.begin_restarting(current_id));

  const auto successor = kernel.submit(
      request(LifecycleTransactionRequestOrigin::kExternalControl, 2U, successor_closure));
  ASSERT_TRUE(successor);
  const LifecycleTransactionId successor_id = transaction_id(kernel, *successor);
  EXPECT_NE(successor_id, current_id);
  EXPECT_EQ(kernel.pending_transaction_count(), 2U);

  ASSERT_TRUE(kernel.publish(current_id));
  EXPECT_FALSE(kernel.is_fenced(1U));
  EXPECT_TRUE(kernel.is_fenced(2U));
  EXPECT_TRUE(kernel.is_fenced(3U));
  EXPECT_EQ(kernel.inspect(*current)->state, LifecycleTransactionState::kPublished);
  EXPECT_EQ(kernel.inspect(*successor)->state, LifecycleTransactionState::kQueued);

  ASSERT_TRUE(
      kernel.acknowledge_gate(successor_id, 2U, LifecycleTransactionGate::kControlSafePoint));
  ASSERT_TRUE(kernel.begin_quiescing(successor_id));
  ASSERT_TRUE(kernel.begin_restarting(successor_id));
  ASSERT_TRUE(kernel.publish(successor_id));
  EXPECT_FALSE(kernel.is_fenced(2U));
  EXPECT_FALSE(kernel.is_fenced(3U));
  EXPECT_EQ(kernel.inspect(*successor)->state, LifecycleTransactionState::kPublished);
  EXPECT_EQ(kernel.pending_transaction_count(), 0U);
}

// REQ: LIFECYCLE-004
// REQ: LIFECYCLE-008
TEST(LifecycleTransaction, RollbackRestoresOrRetainsFailedStoppedFenceUntilRecovery) {
  LifecycleTransactionKernel kernel(capacity());
  constexpr std::array<OperatorInstanceId, 1> closure{1U};
  // Key setup: three sequential transactions exercise two rollback outcomes:
  //   rollback(true)  → kRolledBack: fence released, operator can re-submit normally.
  //   rollback(false) → kFailedStopped: fence retained until kExternalControl recovery.
  // Event reservations accumulate until retire_receipt() releases each contribution.
  const auto restored =
      kernel.submit(request(LifecycleTransactionRequestOrigin::kComputeSelf, 1U, closure));
  ASSERT_TRUE(restored);
  EXPECT_EQ(kernel.reserved_event_count(), kStructuralEventsPerRequest);
  const auto restored_id = transaction_id(kernel, *restored);
  ASSERT_TRUE(
      kernel.acknowledge_gate(restored_id, 1U, LifecycleTransactionGate::kControlSafePoint));
  ASSERT_TRUE(kernel.begin_quiescing(restored_id));
  ASSERT_TRUE(kernel.begin_restarting(restored_id));
  ASSERT_TRUE(kernel.rollback(restored_id, true));
  EXPECT_EQ(kernel.inspect(*restored)->state, LifecycleTransactionState::kRolledBack);
  EXPECT_FALSE(kernel.is_fenced(1U));
  EXPECT_EQ(kernel.reserved_event_count(), kStructuralEventsPerRequest);
  ASSERT_TRUE(kernel.retire_receipt(*restored));
  EXPECT_EQ(kernel.reserved_event_count(), 0U);

  const auto failed =
      kernel.submit(request(LifecycleTransactionRequestOrigin::kComputeSelf, 1U, closure));
  ASSERT_TRUE(failed);
  const auto failed_id = transaction_id(kernel, *failed);
  ASSERT_TRUE(kernel.acknowledge_gate(failed_id, 1U, LifecycleTransactionGate::kControlSafePoint));
  ASSERT_TRUE(kernel.begin_quiescing(failed_id));
  ASSERT_TRUE(kernel.begin_restarting(failed_id));
  ASSERT_TRUE(kernel.rollback(failed_id, false));
  EXPECT_EQ(kernel.inspect(*failed)->state, LifecycleTransactionState::kFailedStopped);
  EXPECT_TRUE(kernel.is_fenced(1U));
  EXPECT_EQ(kernel.reserved_event_count(), kStructuralEventsPerRequest);

  auto recovery_request = request(LifecycleTransactionRequestOrigin::kExternalControl, 1U, closure);
  recovery_request.target_state = LifecycleTransactionTargetState::kFailedStopped;
  const auto recovery = kernel.submit(recovery_request);
  ASSERT_TRUE(recovery);
  const auto recovery_id = transaction_id(kernel, *recovery);
  ASSERT_TRUE(
      kernel.acknowledge_gate(recovery_id, 1U, LifecycleTransactionGate::kFailedStoppedNormalized));
  ASSERT_TRUE(kernel.begin_quiescing(recovery_id));
  ASSERT_TRUE(kernel.begin_restarting(recovery_id));
  ASSERT_TRUE(kernel.publish(recovery_id));
  EXPECT_EQ(kernel.inspect(*recovery)->state, LifecycleTransactionState::kPublished);
  EXPECT_FALSE(kernel.is_fenced(1U));
  EXPECT_EQ(kernel.reserved_event_count(), 2U * kStructuralEventsPerRequest);
  ASSERT_TRUE(kernel.retire_receipt(*failed));
  EXPECT_EQ(kernel.reserved_event_count(), kStructuralEventsPerRequest);
  // Key assertion: retire_receipt() releases exactly the structural_event_delta for each
  // receipt; after retiring all three, reserved_event_count() returns 0.
  ASSERT_TRUE(kernel.retire_receipt(*recovery));
  EXPECT_EQ(kernel.reserved_event_count(), 0U);
}

// REQ: LIFECYCLE-008
TEST(LifecycleTransaction, StopCancelsCurrentAndSuccessorAndClosesIngress) {
  LifecycleTransactionKernel kernel(capacity());
  constexpr std::array<OperatorInstanceId, 1> first_closure{1U};
  constexpr std::array<OperatorInstanceId, 2> second_closure{1U, 2U};
  // Key setup: first transaction is quiescing, second is queued — two in-flight
  // transactions when request_stop() is called. Both must be cancelled atomically.
  const auto first =
      kernel.submit(request(LifecycleTransactionRequestOrigin::kComputeSelf, 1U, first_closure));
  ASSERT_TRUE(first);
  const auto first_id = transaction_id(kernel, *first);
  ASSERT_TRUE(kernel.acknowledge_gate(first_id, 1U, LifecycleTransactionGate::kControlSafePoint));
  ASSERT_TRUE(kernel.begin_quiescing(first_id));
  const auto second = kernel.submit(
      request(LifecycleTransactionRequestOrigin::kExternalControl, 2U, second_closure));
  ASSERT_TRUE(second);
  EXPECT_EQ(kernel.reserved_event_count(), 2U * kStructuralEventsPerRequest);

  // Key assertion: stop closes ingress, cancels both transactions, retains fences until
  // complete_stop_cleanup(); event budget is freed only after retire_receipt() per receipt.
  kernel.request_stop(first_id);
  EXPECT_FALSE(kernel.ingress_open());
  EXPECT_TRUE(kernel.stop_requested());
  EXPECT_EQ(kernel.inspect(*first)->state, LifecycleTransactionState::kCanceled);
  EXPECT_EQ(kernel.inspect(*second)->state, LifecycleTransactionState::kCanceled);
  EXPECT_EQ(kernel.inspect(*first)->related_transaction_id, first_id);
  EXPECT_TRUE(kernel.is_fenced(1U));
  EXPECT_TRUE(kernel.is_fenced(2U));
  EXPECT_EQ(kernel.pending_request_count(), 0U);
  EXPECT_EQ(kernel.pending_transaction_count(), 2U);
  EXPECT_EQ(kernel.reserved_event_count(), 2U * kStructuralEventsPerRequest);

  const auto rejected =
      kernel.submit(request(LifecycleTransactionRequestOrigin::kComputeSelf, 1U, first_closure));
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, LifecycleTransactionKernelErrorCode::kIngressClosed);

  ASSERT_TRUE(kernel.complete_stop_cleanup());
  EXPECT_FALSE(kernel.is_fenced(1U));
  EXPECT_FALSE(kernel.is_fenced(2U));
  EXPECT_EQ(kernel.pending_transaction_count(), 0U);
  EXPECT_EQ(kernel.reserved_event_count(), 2U * kStructuralEventsPerRequest);
  ASSERT_TRUE(kernel.retire_receipt(*first));
  ASSERT_TRUE(kernel.retire_receipt(*second));
  EXPECT_EQ(kernel.reserved_event_count(), 0U);
}

// REQ: LIFECYCLE-008
// Each LifecycleTransactionTargetState maps to exactly one gate type; begin_quiescing()
// must fail until that gate is acknowledged. Transient states (Starting, Stopping, etc.)
// are rejected at submit time so no fence is ever placed.
TEST(LifecycleTransaction, TargetTimingRequiresExactExecutorGateBeforeQuiescing) {
  struct AcceptedCase {
    LifecycleTransactionTargetState state;
    LifecycleTransactionGate gate;
  };
  constexpr std::array accepted_cases{
      AcceptedCase{LifecycleTransactionTargetState::kRunning,
                   LifecycleTransactionGate::kActivationRetired},
      AcceptedCase{LifecycleTransactionTargetState::kReady,
                   LifecycleTransactionGate::kReadyActivationCanceled},
      AcceptedCase{LifecycleTransactionTargetState::kIdle,
                   LifecycleTransactionGate::kControlSafePoint},
      AcceptedCase{LifecycleTransactionTargetState::kFailedStopped,
                   LifecycleTransactionGate::kFailedStoppedNormalized},
  };
  constexpr std::array<OperatorInstanceId, 1> closure{1U};
  for (const auto& test_case : accepted_cases) {
    LifecycleTransactionKernel kernel(capacity());
    auto value = request(LifecycleTransactionRequestOrigin::kExternalControl, 1U, closure);
    value.target_state = test_case.state;
    const auto receipt = kernel.submit(value);
    ASSERT_TRUE(receipt);
    const auto id = transaction_id(kernel, *receipt);
    const auto too_early = kernel.begin_quiescing(id);
    ASSERT_FALSE(too_early);
    EXPECT_EQ(too_early.error().code, LifecycleTransactionKernelErrorCode::kGateUnsatisfied);
    ASSERT_TRUE(kernel.acknowledge_gate(id, 1U, test_case.gate));
    EXPECT_TRUE(kernel.begin_quiescing(id));
  }

  constexpr std::array rejected_states{
      LifecycleTransactionTargetState::kStarting,
      LifecycleTransactionTargetState::kStopping,
      LifecycleTransactionTargetState::kStopped,
      LifecycleTransactionTargetState::kTerminal,
  };
  for (const auto state : rejected_states) {
    LifecycleTransactionKernel kernel(capacity());
    auto value = request(LifecycleTransactionRequestOrigin::kExternalControl, 1U, closure);
    value.target_state = state;
    const auto rejected = kernel.submit(value);
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, LifecycleTransactionKernelErrorCode::kTargetStateRejected);
    EXPECT_FALSE(kernel.is_fenced(1U));
  }

  LifecycleTransactionKernel kernel(capacity());
  auto parameter_recovery = request(LifecycleTransactionRequestOrigin::kParameter,
                                    1U,
                                    closure,
                                    LifecycleStage::kArm,
                                    LifecycleStage::kArm,
                                    11U);
  parameter_recovery.target_state = LifecycleTransactionTargetState::kFailedStopped;
  const auto rejected = kernel.submit(parameter_recovery);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, LifecycleTransactionKernelErrorCode::kTargetStateRejected);
}

// REQ: LIFECYCLE-008
TEST(LifecycleTransaction, CoalescedTargetsRequireIndependentGateAcknowledgements) {
  LifecycleTransactionKernel kernel(capacity());
  constexpr std::array<OperatorInstanceId, 1> first_closure{1U};
  constexpr std::array<OperatorInstanceId, 1> second_closure{2U};

  auto first_request =
      request(LifecycleTransactionRequestOrigin::kExternalControl, 1U, first_closure);
  first_request.target_state = LifecycleTransactionTargetState::kRunning;
  const auto first = kernel.submit(first_request);
  ASSERT_TRUE(first);
  const LifecycleTransactionId transaction = transaction_id(kernel, *first);

  auto second_request =
      request(LifecycleTransactionRequestOrigin::kExternalControl, 2U, second_closure);
  second_request.target_state = LifecycleTransactionTargetState::kRunning;
  const auto second = kernel.submit(second_request);
  ASSERT_TRUE(second);
  EXPECT_EQ(transaction_id(kernel, *second), transaction);

  ASSERT_TRUE(
      kernel.acknowledge_gate(transaction, 1U, LifecycleTransactionGate::kActivationRetired));
  const auto one_target_remaining = kernel.begin_quiescing(transaction);
  ASSERT_FALSE(one_target_remaining);
  EXPECT_EQ(one_target_remaining.error().code,
            LifecycleTransactionKernelErrorCode::kGateUnsatisfied);

  ASSERT_TRUE(
      kernel.acknowledge_gate(transaction, 2U, LifecycleTransactionGate::kActivationRetired));
  EXPECT_TRUE(kernel.begin_quiescing(transaction));
}

// REQ: LIFECYCLE-008
// Receipt storage is finite and fixed at construction. Retiring a terminal receipt
// (kAcceptedNoChange) frees its slot, allowing a subsequent submission to reuse it.
TEST(LifecycleTransaction, TerminalReceiptRetirementReusesFiniteObservationStorage) {
  auto limits = capacity();
  limits.max_receipts = 1U;
  limits.max_events = kStructuralEventsPerRequest;
  limits.max_audit_records = 1U;
  LifecycleTransactionKernel kernel(limits);
  constexpr std::array<OperatorInstanceId, 1> closure{1U};
  const auto first = kernel.submit(request(LifecycleTransactionRequestOrigin::kParameter,
                                           1U,
                                           closure,
                                           LifecycleStage::kArm,
                                           LifecycleStage::kArm,
                                           1U,
                                           true));
  ASSERT_TRUE(first);
  const auto full = kernel.submit(request(LifecycleTransactionRequestOrigin::kParameter,
                                          1U,
                                          closure,
                                          LifecycleStage::kArm,
                                          LifecycleStage::kArm,
                                          2U,
                                          true));
  ASSERT_FALSE(full);
  EXPECT_EQ(full.error().capacity, LifecycleTransactionCapacityKind::kReceipts);
  ASSERT_TRUE(kernel.retire_receipt(*first));
  EXPECT_FALSE(kernel.inspect(*first));
  const auto reused = kernel.submit(request(LifecycleTransactionRequestOrigin::kParameter,
                                            1U,
                                            closure,
                                            LifecycleStage::kArm,
                                            LifecycleStage::kArm,
                                            2U,
                                            true));
  EXPECT_TRUE(reused);
}

// REQ: LIFECYCLE-008
// structural_event_delta is submitted by the compiler to claim space in the event log.
// The reservation check is atomic with coalescence: expanding a transaction that would
// exceed max_events fails without placing any additional fences.
TEST(LifecycleTransaction, ReservesCompilerDerivedStructuralEventsBeforeAcceptance) {
  constexpr std::array<OperatorInstanceId, 1> first_closure{1U};
  constexpr std::array<OperatorInstanceId, 2> expanded_closure{1U, 2U};

  auto no_change_limits = capacity();
  no_change_limits.max_events = 1U;
  LifecycleTransactionKernel insufficient_no_change(no_change_limits);
  auto no_change_request = request(LifecycleTransactionRequestOrigin::kParameter,
                                   1U,
                                   first_closure,
                                   LifecycleStage::kArm,
                                   LifecycleStage::kArm,
                                   1U,
                                   true);
  no_change_request.structural_event_delta = 2U;
  const auto rejected_no_change = insufficient_no_change.submit(no_change_request);
  ASSERT_FALSE(rejected_no_change);
  EXPECT_EQ(rejected_no_change.error().capacity, LifecycleTransactionCapacityKind::kEvents);
  EXPECT_EQ(insufficient_no_change.reserved_event_count(), 0U);

  auto limits = capacity();
  limits.max_events = 11U;
  LifecycleTransactionKernel kernel(limits);
  auto first_request =
      request(LifecycleTransactionRequestOrigin::kExternalControl, 1U, first_closure);
  first_request.structural_event_delta = 7U;
  const auto first = kernel.submit(first_request);
  ASSERT_TRUE(first);
  const LifecycleTransactionId transaction = transaction_id(kernel, *first);
  EXPECT_EQ(kernel.reserved_event_count(), 7U);

  auto expansion_request =
      request(LifecycleTransactionRequestOrigin::kExternalControl, 2U, expanded_closure);
  expansion_request.structural_event_delta = 5U;
  const auto rejected_expansion = kernel.submit(expansion_request);
  ASSERT_FALSE(rejected_expansion);
  EXPECT_EQ(rejected_expansion.error().capacity, LifecycleTransactionCapacityKind::kEvents);
  EXPECT_EQ(kernel.reserved_event_count(), 7U);
  EXPECT_FALSE(kernel.is_fenced(2U));
  EXPECT_EQ(kernel.pending_request_count(), 1U);

  expansion_request.structural_event_delta = 4U;
  const auto expansion = kernel.submit(expansion_request);
  ASSERT_TRUE(expansion);
  EXPECT_EQ(transaction_id(kernel, *expansion), transaction);
  EXPECT_EQ(kernel.reserved_event_count(), 11U);
  EXPECT_TRUE(kernel.is_fenced(2U));

  ASSERT_TRUE(
      kernel.acknowledge_gate(transaction, 1U, LifecycleTransactionGate::kControlSafePoint));
  ASSERT_TRUE(
      kernel.acknowledge_gate(transaction, 2U, LifecycleTransactionGate::kControlSafePoint));
  ASSERT_TRUE(kernel.begin_quiescing(transaction));
  ASSERT_TRUE(kernel.begin_restarting(transaction));
  ASSERT_TRUE(kernel.publish(transaction));
  EXPECT_EQ(kernel.reserved_event_count(), 11U);
  ASSERT_TRUE(kernel.retire_receipt(*first));
  EXPECT_EQ(kernel.reserved_event_count(), 4U);
  ASSERT_TRUE(kernel.retire_receipt(*expansion));
  EXPECT_EQ(kernel.reserved_event_count(), 0U);
}

// REQ: LIFECYCLE-008
// Quiesce and rollback record slots are reserved at coalescence time. A closure expansion
// that exhausts either pool must fail atomically without placing any new fence. Slots freed
// by a publish or rollback(true) become immediately available for the next submission.
TEST(LifecycleTransaction, QuiesceAndRollbackReservationsRejectAtomicallyAndAreReusable) {
  constexpr std::array<OperatorInstanceId, 1> first_closure{1U};
  constexpr std::array<OperatorInstanceId, 2> expanded_closure{1U, 2U};

  struct CapacityCase {
    LifecycleTransactionCapacityKind expected;
    std::size_t quiesce_records;
    std::size_t rollback_records;
  };
  constexpr std::array cases{
      CapacityCase{LifecycleTransactionCapacityKind::kQuiesceRecords, 1U, 2U},
      CapacityCase{LifecycleTransactionCapacityKind::kRollbackRecords, 2U, 1U},
  };
  for (const auto& test_case : cases) {
    auto limits = capacity();
    limits.max_quiesce_records = test_case.quiesce_records;
    limits.max_rollback_records = test_case.rollback_records;
    LifecycleTransactionKernel kernel(limits);
    const auto first = kernel.submit(
        request(LifecycleTransactionRequestOrigin::kExternalControl, 1U, first_closure));
    ASSERT_TRUE(first);
    const LifecycleTransactionId transaction = transaction_id(kernel, *first);

    const auto rejected = kernel.submit(
        request(LifecycleTransactionRequestOrigin::kExternalControl, 2U, expanded_closure));
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().capacity, test_case.expected);
    EXPECT_EQ(kernel.pending_request_count(), 1U);
    EXPECT_EQ(kernel.pending_transaction_count(), 1U);
    EXPECT_TRUE(kernel.is_fenced(1U));
    EXPECT_FALSE(kernel.is_fenced(2U));
    const auto first_view = kernel.inspect(*first);
    ASSERT_TRUE(first_view);
    EXPECT_EQ(first_view->transaction_id, transaction);
  }

  auto limits = capacity();
  limits.max_quiesce_records = 2U;
  limits.max_rollback_records = 2U;
  LifecycleTransactionKernel kernel(limits);
  constexpr std::array<OperatorInstanceId, 2> published_closure{1U, 2U};
  constexpr std::array<OperatorInstanceId, 2> rolled_back_closure{3U, 4U};
  constexpr std::array<OperatorInstanceId, 2> reused_closure{5U, 6U};

  const auto published = kernel.submit(
      request(LifecycleTransactionRequestOrigin::kExternalControl, 1U, published_closure));
  ASSERT_TRUE(published);
  const LifecycleTransactionId published_id = transaction_id(kernel, *published);
  ASSERT_TRUE(
      kernel.acknowledge_gate(published_id, 1U, LifecycleTransactionGate::kControlSafePoint));
  ASSERT_TRUE(kernel.begin_quiescing(published_id));
  ASSERT_TRUE(kernel.begin_restarting(published_id));
  ASSERT_TRUE(kernel.publish(published_id));

  const auto rolled_back = kernel.submit(
      request(LifecycleTransactionRequestOrigin::kExternalControl, 3U, rolled_back_closure));
  ASSERT_TRUE(rolled_back);
  const LifecycleTransactionId rolled_back_id = transaction_id(kernel, *rolled_back);
  ASSERT_TRUE(
      kernel.acknowledge_gate(rolled_back_id, 3U, LifecycleTransactionGate::kControlSafePoint));
  ASSERT_TRUE(kernel.begin_quiescing(rolled_back_id));
  ASSERT_TRUE(kernel.begin_restarting(rolled_back_id));
  ASSERT_TRUE(kernel.rollback(rolled_back_id, true));

  const auto reused = kernel.submit(
      request(LifecycleTransactionRequestOrigin::kExternalControl, 5U, reused_closure));
  ASSERT_TRUE(reused);
  EXPECT_TRUE(kernel.is_fenced(5U));
  EXPECT_TRUE(kernel.is_fenced(6U));
  EXPECT_EQ(kernel.pending_transaction_count(), 1U);
}

// REQ: LIFECYCLE-008
// REQ: PARAM-021
// REQ: PARAM-023
// Each internal pool (pending requests, transactions, receipts, fenced operators, etc.)
// is checked before any mutation. Zeroing any single limit makes every submit() fail
// at that check, leaving fences, request count, and transaction count all at zero.
TEST(LifecycleTransaction, EveryFiniteReservationRejectsBeforeMutation) {
  struct Case {
    LifecycleTransactionCapacityKind expected;
    void (*set_zero)(LifecycleTransactionKernelCapacity&);
  };
  constexpr std::array cases{
      Case{LifecycleTransactionCapacityKind::kPendingRequests,
           [](LifecycleTransactionKernelCapacity& value) { value.max_pending_requests = 0U; }},
      Case{LifecycleTransactionCapacityKind::kPendingTransactions,
           [](LifecycleTransactionKernelCapacity& value) { value.max_pending_transactions = 0U; }},
      Case{LifecycleTransactionCapacityKind::kReceipts,
           [](LifecycleTransactionKernelCapacity& value) { value.max_receipts = 0U; }},
      Case{LifecycleTransactionCapacityKind::kFencedOperators,
           [](LifecycleTransactionKernelCapacity& value) { value.max_fenced_operators = 0U; }},
      Case{LifecycleTransactionCapacityKind::kCandidateGenerations,
           [](LifecycleTransactionKernelCapacity& value) { value.max_candidate_generations = 0U; }},
      Case{LifecycleTransactionCapacityKind::kEvents,
           [](LifecycleTransactionKernelCapacity& value) { value.max_events = 0U; }},
      Case{LifecycleTransactionCapacityKind::kAuditRecords,
           [](LifecycleTransactionKernelCapacity& value) { value.max_audit_records = 0U; }},
      Case{LifecycleTransactionCapacityKind::kQuiesceRecords,
           [](LifecycleTransactionKernelCapacity& value) { value.max_quiesce_records = 0U; }},
      Case{LifecycleTransactionCapacityKind::kRollbackRecords,
           [](LifecycleTransactionKernelCapacity& value) { value.max_rollback_records = 0U; }},
  };
  constexpr std::array<OperatorInstanceId, 1> closure{1U};
  for (const auto& test_case : cases) {
    auto limits = capacity();
    test_case.set_zero(limits);
    LifecycleTransactionKernel kernel(limits);
    const auto result = kernel.submit(request(LifecycleTransactionRequestOrigin::kParameter,
                                              1U,
                                              closure,
                                              LifecycleStage::kArm,
                                              LifecycleStage::kArm,
                                              8U));
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, LifecycleTransactionKernelErrorCode::kCapacityExceeded);
    EXPECT_EQ(result.error().capacity, test_case.expected);
    EXPECT_FALSE(kernel.is_fenced(1U));
    EXPECT_EQ(kernel.pending_request_count(), 0U);
    EXPECT_EQ(kernel.pending_transaction_count(), 0U);
  }
}

// REQ: LIFECYCLE-008
TEST(LifecycleTransaction, PostConstructionTransitionsAllocateNoMemory) {
  LifecycleTransactionKernel kernel(capacity());
  constexpr std::array<OperatorInstanceId, 2> closure{1U, 2U};

  // Key setup: capacity() pre-allocates all internal pools; the monitored scope covers
  // only the post-construction sequence (submit → inspect → gate → quiesce → restart → rollback).
  std::size_t allocations = 0U;
  bool accepted = false;
  bool quiesced = false;
  bool restarted = false;
  bool rolled_back = false;
  {
    const ScopedAllocationCounter counter;
    const auto receipt = kernel.submit(request(LifecycleTransactionRequestOrigin::kParameter,
                                               1U,
                                               closure,
                                               LifecycleStage::kAllocate,
                                               LifecycleStage::kAllocate,
                                               8U));
    accepted = receipt.has_value();
    if (receipt) {
      const auto view = kernel.inspect(*receipt);
      if (view) {
        const bool acknowledged =
            kernel
                .acknowledge_gate(
                    view->transaction_id, 1U, LifecycleTransactionGate::kControlSafePoint)
                .has_value();
        if (acknowledged) {
          quiesced = kernel.begin_quiescing(view->transaction_id).has_value();
          restarted = kernel.begin_restarting(view->transaction_id).has_value();
          rolled_back = kernel.rollback(view->transaction_id, true).has_value();
        }
      }
    }
    allocations = counter.count();
  }

  // Key assertion: the complete lifecycle transition sequence allocates zero heap memory.
  EXPECT_TRUE(accepted);
  EXPECT_TRUE(quiesced);
  EXPECT_TRUE(restarted);
  EXPECT_TRUE(rolled_back);
  EXPECT_EQ(allocations, 0U);
}

}  // namespace
}  // namespace holoscan::test
