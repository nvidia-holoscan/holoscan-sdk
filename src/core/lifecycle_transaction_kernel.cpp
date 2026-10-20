/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "core/lifecycle_transaction_kernel.hpp"

#include <algorithm>

namespace holoscan::detail {
namespace {

[[nodiscard]] constexpr bool is_terminal(LifecycleTransactionState state) noexcept {
  switch (state) {
    case LifecycleTransactionState::kAcceptedNoChange:
    case LifecycleTransactionState::kPublished:
    case LifecycleTransactionState::kSuperseded:
    case LifecycleTransactionState::kCanceled:
    case LifecycleTransactionState::kRolledBack:
    case LifecycleTransactionState::kFailedStopped:
      return true;
    case LifecycleTransactionState::kRejected:
    case LifecycleTransactionState::kQueued:
    case LifecycleTransactionState::kQuiescing:
    case LifecycleTransactionState::kRestarting:
      return false;
  }
  return false;
}

[[nodiscard]] constexpr bool is_parameter_origin(
    LifecycleTransactionRequestOrigin origin) noexcept {
  return origin == LifecycleTransactionRequestOrigin::kParameter;
}

[[nodiscard]] constexpr LifecycleTransactionKernelError error(
    LifecycleTransactionKernelErrorCode code,
    LifecycleTransactionCapacityKind capacity = LifecycleTransactionCapacityKind::kNone) noexcept {
  return {.code = code, .capacity = capacity};
}

[[nodiscard]] constexpr std::size_t stage_index(LifecycleStage stage) noexcept {
  return static_cast<std::size_t>(stage);
}

[[nodiscard]] constexpr std::uint8_t gate_bit(LifecycleTransactionGate gate) noexcept {
  return static_cast<std::uint8_t>(1U << static_cast<std::uint8_t>(gate));
}

[[nodiscard]] constexpr bool target_state_rejected(LifecycleTransactionTargetState state) noexcept {
  return state == LifecycleTransactionTargetState::kStarting ||
         state == LifecycleTransactionTargetState::kStopping ||
         state == LifecycleTransactionTargetState::kStopped ||
         state == LifecycleTransactionTargetState::kTerminal;
}

[[nodiscard]] constexpr LifecycleTransactionGate required_gate(
    LifecycleTransactionTargetState state) noexcept {
  switch (state) {
    case LifecycleTransactionTargetState::kRunning:
      return LifecycleTransactionGate::kActivationRetired;
    case LifecycleTransactionTargetState::kReady:
      return LifecycleTransactionGate::kReadyActivationCanceled;
    case LifecycleTransactionTargetState::kIdle:
      return LifecycleTransactionGate::kControlSafePoint;
    case LifecycleTransactionTargetState::kFailedStopped:
      return LifecycleTransactionGate::kFailedStoppedNormalized;
    case LifecycleTransactionTargetState::kStarting:
    case LifecycleTransactionTargetState::kStopping:
    case LifecycleTransactionTargetState::kStopped:
    case LifecycleTransactionTargetState::kTerminal:
      return LifecycleTransactionGate::kControlSafePoint;
  }
  return LifecycleTransactionGate::kControlSafePoint;
}

}  // namespace

LifecycleTransactionKernel::LifecycleTransactionKernel(
    const LifecycleTransactionKernelCapacity& capacity)
    : capacity_(capacity) {
  transactions_.reserve(capacity_.max_pending_transactions);
  for (std::size_t index = 0U; index < capacity_.max_pending_transactions; ++index) {
    transactions_.emplace_back();
    transactions_.back().members.reserve(capacity_.max_fenced_operators);
    transactions_.back().candidates.reserve(capacity_.max_candidate_generations);
    transactions_.back().gates.reserve(capacity_.max_fenced_operators);
  }
  fences_.reserve(capacity_.max_fenced_operators);
  receipts_.reserve(capacity_.max_receipts);
}

LifecycleTransactionKernel::TransactionSlot* LifecycleTransactionKernel::find_transaction(
    LifecycleTransactionId transaction) noexcept {
  const auto result = std::find_if(
      transactions_.begin(), transactions_.end(), [transaction](const TransactionSlot& candidate) {
        return candidate.occupied && candidate.id == transaction;
      });
  return result == transactions_.end() ? nullptr : &*result;
}

const LifecycleTransactionKernel::TransactionSlot* LifecycleTransactionKernel::find_transaction(
    LifecycleTransactionId transaction) const noexcept {
  const auto result = std::find_if(
      transactions_.begin(), transactions_.end(), [transaction](const TransactionSlot& candidate) {
        return candidate.occupied && candidate.id == transaction;
      });
  return result == transactions_.end() ? nullptr : &*result;
}

LifecycleTransactionKernel::ReceiptRecord* LifecycleTransactionKernel::find_receipt(
    std::uint64_t receipt_id) noexcept {
  const auto result = std::find_if(
      receipts_.begin(), receipts_.end(), [receipt_id](const ReceiptRecord& candidate) {
        return candidate.view.receipt.receipt_id == receipt_id;
      });
  return result == receipts_.end() ? nullptr : &*result;
}

const LifecycleTransactionKernel::ReceiptRecord* LifecycleTransactionKernel::find_receipt(
    std::uint64_t receipt_id) const noexcept {
  const auto result = std::find_if(
      receipts_.begin(), receipts_.end(), [receipt_id](const ReceiptRecord& candidate) {
        return candidate.view.receipt.receipt_id == receipt_id;
      });
  return result == receipts_.end() ? nullptr : &*result;
}

LifecycleTransactionKernel::FenceRecord* LifecycleTransactionKernel::find_fence(
    OperatorInstanceId owner) noexcept {
  const auto result =
      std::find_if(fences_.begin(), fences_.end(), [owner](const FenceRecord& candidate) {
        return candidate.owner == owner;
      });
  return result == fences_.end() ? nullptr : &*result;
}

const LifecycleTransactionKernel::FenceRecord* LifecycleTransactionKernel::find_fence(
    OperatorInstanceId owner) const noexcept {
  const auto result =
      std::find_if(fences_.begin(), fences_.end(), [owner](const FenceRecord& candidate) {
        return candidate.owner == owner;
      });
  return result == fences_.end() ? nullptr : &*result;
}

LifecycleTransactionKernel::TransactionSlot*
LifecycleTransactionKernel::find_coalescing_transaction(
    LifecycleTransactionCompatibilityKey compatibility) noexcept {
  TransactionSlot* selected = nullptr;
  for (auto& transaction : transactions_) {
    if (!transaction.occupied || transaction.state != LifecycleTransactionState::kQueued ||
        transaction.compatibility != compatibility) {
      continue;
    }
    if (selected == nullptr || transaction.id < selected->id) {
      selected = &transaction;
    }
  }
  return selected;
}

LifecycleTransactionKernel::TransactionSlot*
LifecycleTransactionKernel::find_free_transaction_slot() noexcept {
  const auto result = std::find_if(
      transactions_.begin(), transactions_.end(), [](const TransactionSlot& candidate) {
        return !candidate.occupied;
      });
  return result == transactions_.end() ? nullptr : &*result;
}

bool LifecycleTransactionKernel::request_valid(
    const LifecycleTransactionRequest& request) const noexcept {
  if (!request.target.run_bound() || !is_lifecycle_forward_stage(request.requested_stage) ||
      !is_lifecycle_forward_stage(request.effective_stage) ||
      stage_index(request.effective_stage) > stage_index(request.requested_stage) ||
      request.closure.empty() || request.structural_event_delta == 0U) {
    return false;
  }
  bool contains_target = false;
  OperatorInstanceId previous = 0U;
  for (const OperatorInstanceId owner : request.closure) {
    if (owner == 0U || owner <= previous) {
      return false;
    }
    previous = owner;
    contains_target = contains_target || owner == request.target.id;
  }
  if (!contains_target ||
      (!is_parameter_origin(request.origin) &&
       (request.parameter_candidate_equivalent || request.candidate_generation != 0U)) ||
      (is_parameter_origin(request.origin) && !request.parameter_candidate_equivalent &&
       request.candidate_generation == 0U)) {
    return false;
  }
  return true;
}

bool LifecycleTransactionKernel::is_head_transaction(
    LifecycleTransactionId transaction) const noexcept {
  LifecycleTransactionId head = 0U;
  for (const auto& candidate : transactions_) {
    if (candidate.occupied && (head == 0U || candidate.id < head)) {
      head = candidate.id;
    }
  }
  return head == transaction;
}

void LifecycleTransactionKernel::terminalize_receipt(ReceiptRecord& receipt,
                                                     LifecycleTransactionState state,
                                                     LifecycleTransactionId related) noexcept {
  if (!is_terminal(receipt.view.state) && pending_requests_ != 0U) {
    --pending_requests_;
  }
  receipt.view.state = state;
  receipt.view.related_transaction_id = related;
}

void LifecycleTransactionKernel::set_transaction_receipts_state(
    LifecycleTransactionId transaction, LifecycleTransactionState state) noexcept {
  for (auto& receipt : receipts_) {
    if (receipt.view.transaction_id != transaction || is_terminal(receipt.view.state)) {
      continue;
    }
    if (is_terminal(state)) {
      terminalize_receipt(receipt, state);
    } else {
      receipt.view.state = state;
    }
  }
}

void LifecycleTransactionKernel::set_transaction_receipts_effective_stage(
    LifecycleTransactionId transaction, LifecycleStage stage) noexcept {
  for (auto& receipt : receipts_) {
    if (receipt.view.transaction_id == transaction) {
      receipt.view.effective_stage = stage;
    }
  }
}

void LifecycleTransactionKernel::release_transaction(TransactionSlot& transaction,
                                                     bool retain_failed_fence) noexcept {
  for (const OperatorInstanceId owner : transaction.members) {
    FenceRecord* const fence = find_fence(owner);
    if (fence == nullptr) {
      continue;
    }
    if (retain_failed_fence) {
      fence->failed_stopped = true;
    }
    if (fence->holders != 0U) {
      --fence->holders;
    }
  }
  fences_.erase(std::remove_if(fences_.begin(),
                               fences_.end(),
                               [](const FenceRecord& fence) {
                                 return fence.holders == 0U && !fence.failed_stopped;
                               }),
                fences_.end());

  candidate_generations_ -= transaction.candidates.size();
  quiesce_records_ -= transaction.members.size();
  rollback_records_ -= transaction.members.size();
  transaction.members.clear();
  transaction.candidates.clear();
  transaction.gates.clear();
  transaction.occupied = false;
  transaction.id = 0U;
  transaction.state = LifecycleTransactionState::kRejected;
  transaction.fence_id = 0U;
  transaction.has_explicit_intent = false;
  if (pending_transactions_ != 0U) {
    --pending_transactions_;
  }
}

// Admission is a single bounded transaction: identity reservation, coalescing, closure union, and
// rollback checkpoints must remain visibly ordered so every failure restores the same prefix.
expected<LifecycleRestartReceipt, LifecycleTransactionKernelError>
LifecycleTransactionKernel::submit(  // NOLINT(readability-function-cognitive-complexity)
    const LifecycleTransactionRequest& request) noexcept {
  if (!ingress_open_) {
    return make_unexpected(error(LifecycleTransactionKernelErrorCode::kIngressClosed));
  }
  if (!is_lifecycle_forward_stage(request.requested_stage) ||
      !is_lifecycle_forward_stage(request.effective_stage)) {
    return make_unexpected(error(LifecycleTransactionKernelErrorCode::kInvalidRestartStage));
  }
  if (!request_valid(request)) {
    return make_unexpected(error(LifecycleTransactionKernelErrorCode::kInvalidRequest));
  }
  if (target_state_rejected(request.target_state) ||
      (request.target_state == LifecycleTransactionTargetState::kFailedStopped &&
       is_parameter_origin(request.origin))) {
    return make_unexpected(error(LifecycleTransactionKernelErrorCode::kTargetStateRejected));
  }
  if (next_receipt_id_ == 0U || next_acceptance_sequence_ == 0U) {
    return make_unexpected(error(LifecycleTransactionKernelErrorCode::kIdentityExhausted));
  }

  TransactionSlot* transaction = find_coalescing_transaction(request.compatibility);
  CandidateRecord* queued_candidate = nullptr;
  if (transaction != nullptr && is_parameter_origin(request.origin)) {
    const auto found = std::find_if(transaction->candidates.begin(),
                                    transaction->candidates.end(),
                                    [&request](const CandidateRecord& candidate) {
                                      return candidate.owner == request.target.id;
                                    });
    if (found != transaction->candidates.end()) {
      queued_candidate = &*found;
    }
  }
  const bool equivalent_supersedes_queued =
      request.parameter_candidate_equivalent && transaction != nullptr &&
      !transaction->has_explicit_intent && queued_candidate != nullptr;
  const bool no_change = is_parameter_origin(request.origin) &&
                         request.parameter_candidate_equivalent &&
                         (transaction == nullptr || !transaction->has_explicit_intent);
  const bool create_transaction = transaction == nullptr && !no_change;
  if (create_transaction) {
    if (pending_transactions_ != 0U && pending_transactions_ >= 2U) {
      return make_unexpected(error(LifecycleTransactionKernelErrorCode::kSuccessorUnavailable));
    }
    if (pending_transactions_ == capacity_.max_pending_transactions) {
      return make_unexpected(error(LifecycleTransactionKernelErrorCode::kCapacityExceeded,
                                   LifecycleTransactionCapacityKind::kPendingTransactions));
    }
    transaction = find_free_transaction_slot();
    if (transaction == nullptr) {
      return make_unexpected(error(LifecycleTransactionKernelErrorCode::kCapacityExceeded,
                                   LifecycleTransactionCapacityKind::kPendingTransactions));
    }
    if (next_transaction_id_ == 0U || next_fence_id_ == 0U) {
      return make_unexpected(error(LifecycleTransactionKernelErrorCode::kIdentityExhausted));
    }
  }

  CandidateRecord* replaced_candidate = nullptr;
  if (!no_change && request.candidate_generation != 0U && !create_transaction) {
    replaced_candidate = queued_candidate;
  }

  std::size_t membership_delta = 0U;
  std::size_t fence_delta = 0U;
  if (!no_change) {
    for (const OperatorInstanceId owner : request.closure) {
      const bool transaction_has_member =
          !create_transaction &&
          std::find(transaction->members.begin(), transaction->members.end(), owner) !=
              transaction->members.end();
      if (!transaction_has_member) {
        ++membership_delta;
      }
      if (find_fence(owner) == nullptr) {
        ++fence_delta;
      }
    }
  }
  const std::size_t candidate_delta =
      !no_change && request.candidate_generation != 0U && replaced_candidate == nullptr ? 1U : 0U;
  const bool supersedes_pending =
      replaced_candidate != nullptr && find_receipt(replaced_candidate->receipt_id) != nullptr &&
      !is_terminal(find_receipt(replaced_candidate->receipt_id)->view.state);
  const std::size_t pending_request_delta = no_change || supersedes_pending ? 0U : 1U;

  const auto capacity_failure = [&](std::size_t used,
                                    std::size_t delta,
                                    std::size_t limit,
                                    LifecycleTransactionCapacityKind kind)
      -> expected<void, LifecycleTransactionKernelError> {
    if (delta > limit || used > limit - delta) {
      return make_unexpected(error(LifecycleTransactionKernelErrorCode::kCapacityExceeded, kind));
    }
    return {};
  };

  if (auto checked = capacity_failure(pending_requests_,
                                      pending_request_delta,
                                      capacity_.max_pending_requests,
                                      LifecycleTransactionCapacityKind::kPendingRequests);
      !checked) {
    return make_unexpected(checked.error());
  }
  if (auto checked = capacity_failure(receipts_.size(),
                                      1U,
                                      capacity_.max_receipts,
                                      LifecycleTransactionCapacityKind::kReceipts);
      !checked) {
    return make_unexpected(checked.error());
  }
  if (auto checked = capacity_failure(fences_.size(),
                                      fence_delta,
                                      capacity_.max_fenced_operators,
                                      LifecycleTransactionCapacityKind::kFencedOperators);
      !checked) {
    return make_unexpected(checked.error());
  }
  if (auto checked = capacity_failure(candidate_generations_,
                                      candidate_delta,
                                      capacity_.max_candidate_generations,
                                      LifecycleTransactionCapacityKind::kCandidateGenerations);
      !checked) {
    return make_unexpected(checked.error());
  }
  if (auto checked = capacity_failure(reserved_events_,
                                      request.structural_event_delta,
                                      capacity_.max_events,
                                      LifecycleTransactionCapacityKind::kEvents);
      !checked) {
    return make_unexpected(checked.error());
  }
  if (auto checked = capacity_failure(receipts_.size(),
                                      1U,
                                      capacity_.max_audit_records,
                                      LifecycleTransactionCapacityKind::kAuditRecords);
      !checked) {
    return make_unexpected(checked.error());
  }
  if (auto checked = capacity_failure(quiesce_records_,
                                      membership_delta,
                                      capacity_.max_quiesce_records,
                                      LifecycleTransactionCapacityKind::kQuiesceRecords);
      !checked) {
    return make_unexpected(checked.error());
  }
  if (auto checked = capacity_failure(rollback_records_,
                                      membership_delta,
                                      capacity_.max_rollback_records,
                                      LifecycleTransactionCapacityKind::kRollbackRecords);
      !checked) {
    return make_unexpected(checked.error());
  }

  const LifecycleTransactionId transaction_id = no_change            ? 0U
                                                : create_transaction ? next_transaction_id_
                                                                     : transaction->id;
  const LifecycleFenceId fence_id = no_change            ? 0U
                                    : create_transaction ? next_fence_id_
                                                         : transaction->fence_id;
  LifecycleRestartReceipt receipt{
      .receipt_id = next_receipt_id_,
      .acceptance_sequence = next_acceptance_sequence_,
      .target = request.target,
      .requested_stage = request.requested_stage,
  };
  const LifecycleTransactionReceiptView receipt_view{
      .receipt = receipt,
      .state = no_change ? LifecycleTransactionState::kAcceptedNoChange
                         : LifecycleTransactionState::kQueued,
      .transaction_id = transaction_id,
      .coalesced_into_transaction = !no_change && !create_transaction ? transaction_id : 0U,
      .related_transaction_id = 0U,
      .fence_id = fence_id,
      .effective_stage = request.effective_stage,
      .candidate_generation = request.candidate_generation,
  };
  receipts_.push_back({
      .view = receipt_view,
      .reserved_events = request.structural_event_delta,
  });
  reserved_events_ += request.structural_event_delta;

  ++next_receipt_id_;
  ++next_acceptance_sequence_;
  if (no_change) {
    if (equivalent_supersedes_queued) {
      if (ReceiptRecord* const replaced = find_receipt(queued_candidate->receipt_id);
          replaced != nullptr && !is_terminal(replaced->view.state)) {
        replaced->view.coalesced_into_transaction = transaction->id;
        terminalize_receipt(*replaced, LifecycleTransactionState::kSuperseded);
      }
      const auto candidate = std::find_if(
          transaction->candidates.begin(),
          transaction->candidates.end(),
          [&request](const CandidateRecord& value) { return value.owner == request.target.id; });
      transaction->candidates.erase(candidate);
      --candidate_generations_;
      if (transaction->candidates.empty()) {
        release_transaction(*transaction, false);
      }
    }
    return receipt;
  }

  if (create_transaction) {
    transaction->occupied = true;
    transaction->id = next_transaction_id_++;
    transaction->state = LifecycleTransactionState::kQueued;
    transaction->fence_id = next_fence_id_++;
    transaction->effective_stage = request.effective_stage;
    transaction->compatibility = request.compatibility;
    transaction->has_explicit_intent = !is_parameter_origin(request.origin);
    ++pending_transactions_;
  } else {
    transaction->effective_stage =
        stage_index(request.effective_stage) < stage_index(transaction->effective_stage)
            ? request.effective_stage
            : transaction->effective_stage;
    transaction->has_explicit_intent =
        transaction->has_explicit_intent || !is_parameter_origin(request.origin);
  }
  set_transaction_receipts_effective_stage(transaction->id, transaction->effective_stage);

  const std::uint8_t request_gate = gate_bit(required_gate(request.target_state));
  const auto gate = std::find_if(
      transaction->gates.begin(), transaction->gates.end(), [&request](const GateRecord& value) {
        return value.target == request.target.id;
      });
  if (gate == transaction->gates.end()) {
    transaction->gates.push_back(
        {.target = request.target.id, .required = request_gate, .satisfied = 0U});
  } else {
    gate->required = static_cast<std::uint8_t>(gate->required | request_gate);
  }

  for (const OperatorInstanceId owner : request.closure) {
    const auto position =
        std::lower_bound(transaction->members.begin(), transaction->members.end(), owner);
    if (position == transaction->members.end() || *position != owner) {
      transaction->members.insert(position, owner);
      ++quiesce_records_;
      ++rollback_records_;
      if (FenceRecord* const existing = find_fence(owner); existing != nullptr) {
        ++existing->holders;
      } else {
        fences_.push_back({.owner = owner, .holders = 1U, .failed_stopped = false});
      }
    }
  }

  if (request.candidate_generation != 0U) {
    if (replaced_candidate != nullptr) {
      if (ReceiptRecord* const replaced = find_receipt(replaced_candidate->receipt_id);
          replaced != nullptr && !is_terminal(replaced->view.state)) {
        replaced->view.coalesced_into_transaction = transaction->id;
        terminalize_receipt(*replaced, LifecycleTransactionState::kSuperseded);
      }
      replaced_candidate->generation = request.candidate_generation;
      replaced_candidate->receipt_id = receipt.receipt_id;
    } else {
      transaction->candidates.push_back({.owner = request.target.id,
                                         .generation = request.candidate_generation,
                                         .receipt_id = receipt.receipt_id});
      ++candidate_generations_;
    }
  }
  if (!no_change) {
    ++pending_requests_;
  }

  return receipt;
}

expected<void, LifecycleTransactionKernelError> LifecycleTransactionKernel::begin_quiescing(
    LifecycleTransactionId transaction_id) noexcept {
  if (stop_requested_) {
    return make_unexpected(error(LifecycleTransactionKernelErrorCode::kInvalidTransition));
  }
  TransactionSlot* const transaction = find_transaction(transaction_id);
  if (transaction == nullptr) {
    return make_unexpected(error(LifecycleTransactionKernelErrorCode::kUnknownTransaction));
  }
  if (transaction->state != LifecycleTransactionState::kQueued ||
      !is_head_transaction(transaction_id)) {
    return make_unexpected(error(LifecycleTransactionKernelErrorCode::kInvalidTransition));
  }
  const bool gates_satisfied =
      std::all_of(transaction->gates.begin(), transaction->gates.end(), [](const GateRecord& gate) {
        return gate.satisfied == gate.required;
      });
  if (!gates_satisfied) {
    return make_unexpected(error(LifecycleTransactionKernelErrorCode::kGateUnsatisfied));
  }
  transaction->state = LifecycleTransactionState::kQuiescing;
  set_transaction_receipts_state(transaction_id, LifecycleTransactionState::kQuiescing);
  return {};
}

expected<void, LifecycleTransactionKernelError> LifecycleTransactionKernel::acknowledge_gate(
    LifecycleTransactionId transaction_id, OperatorInstanceId target,
    LifecycleTransactionGate gate) noexcept {
  TransactionSlot* const transaction = find_transaction(transaction_id);
  if (transaction == nullptr) {
    return make_unexpected(error(LifecycleTransactionKernelErrorCode::kUnknownTransaction));
  }
  if (stop_requested_ || transaction->state != LifecycleTransactionState::kQueued) {
    return make_unexpected(error(LifecycleTransactionKernelErrorCode::kInvalidTransition));
  }
  const auto target_gate = std::find_if(
      transaction->gates.begin(), transaction->gates.end(), [target](const GateRecord& value) {
        return value.target == target;
      });
  const std::uint8_t bit = gate_bit(gate);
  if (target_gate == transaction->gates.end() || (target_gate->required & bit) == 0) {
    return make_unexpected(error(LifecycleTransactionKernelErrorCode::kInvalidTransition));
  }
  target_gate->satisfied = static_cast<std::uint8_t>(target_gate->satisfied | bit);
  return {};
}

expected<void, LifecycleTransactionKernelError> LifecycleTransactionKernel::begin_restarting(
    LifecycleTransactionId transaction_id) noexcept {
  if (stop_requested_) {
    return make_unexpected(error(LifecycleTransactionKernelErrorCode::kInvalidTransition));
  }
  TransactionSlot* const transaction = find_transaction(transaction_id);
  if (transaction == nullptr) {
    return make_unexpected(error(LifecycleTransactionKernelErrorCode::kUnknownTransaction));
  }
  if (transaction->state != LifecycleTransactionState::kQuiescing ||
      !is_head_transaction(transaction_id)) {
    return make_unexpected(error(LifecycleTransactionKernelErrorCode::kInvalidTransition));
  }
  transaction->state = LifecycleTransactionState::kRestarting;
  set_transaction_receipts_state(transaction_id, LifecycleTransactionState::kRestarting);
  return {};
}

expected<void, LifecycleTransactionKernelError> LifecycleTransactionKernel::publish(
    LifecycleTransactionId transaction_id) noexcept {
  if (stop_requested_) {
    return make_unexpected(error(LifecycleTransactionKernelErrorCode::kInvalidTransition));
  }
  TransactionSlot* const transaction = find_transaction(transaction_id);
  if (transaction == nullptr) {
    return make_unexpected(error(LifecycleTransactionKernelErrorCode::kUnknownTransaction));
  }
  if (transaction->state != LifecycleTransactionState::kRestarting ||
      !is_head_transaction(transaction_id)) {
    return make_unexpected(error(LifecycleTransactionKernelErrorCode::kInvalidTransition));
  }
  for (const OperatorInstanceId owner : transaction->members) {
    if (FenceRecord* const fence = find_fence(owner); fence != nullptr) {
      fence->failed_stopped = false;
    }
  }
  set_transaction_receipts_state(transaction_id, LifecycleTransactionState::kPublished);
  release_transaction(*transaction, false);
  return {};
}

expected<void, LifecycleTransactionKernelError> LifecycleTransactionKernel::rollback(
    LifecycleTransactionId transaction_id, bool restored_old_epoch) noexcept {
  if (stop_requested_) {
    return make_unexpected(error(LifecycleTransactionKernelErrorCode::kInvalidTransition));
  }
  TransactionSlot* const transaction = find_transaction(transaction_id);
  if (transaction == nullptr) {
    return make_unexpected(error(LifecycleTransactionKernelErrorCode::kUnknownTransaction));
  }
  if ((transaction->state != LifecycleTransactionState::kQuiescing &&
       transaction->state != LifecycleTransactionState::kRestarting) ||
      !is_head_transaction(transaction_id)) {
    return make_unexpected(error(LifecycleTransactionKernelErrorCode::kInvalidTransition));
  }
  const auto terminal = restored_old_epoch ? LifecycleTransactionState::kRolledBack
                                           : LifecycleTransactionState::kFailedStopped;
  set_transaction_receipts_state(transaction_id, terminal);
  release_transaction(*transaction, !restored_old_epoch);
  return {};
}

void LifecycleTransactionKernel::request_stop(LifecycleTransactionId cause) noexcept {
  ingress_open_ = false;
  stop_requested_ = true;
  for (auto& receipt : receipts_) {
    if (!is_terminal(receipt.view.state)) {
      terminalize_receipt(receipt, LifecycleTransactionState::kCanceled, cause);
    }
  }
}

expected<void, LifecycleTransactionKernelError>
LifecycleTransactionKernel::complete_stop_cleanup() noexcept {
  if (!stop_requested_) {
    return make_unexpected(error(LifecycleTransactionKernelErrorCode::kInvalidTransition));
  }
  for (auto& transaction : transactions_) {
    if (!transaction.occupied) {
      continue;
    }
    candidate_generations_ -= transaction.candidates.size();
    quiesce_records_ -= transaction.members.size();
    rollback_records_ -= transaction.members.size();
    transaction.members.clear();
    transaction.candidates.clear();
    transaction.gates.clear();
    transaction.occupied = false;
    transaction.id = 0U;
    transaction.state = LifecycleTransactionState::kRejected;
  }
  fences_.clear();
  pending_requests_ = 0U;
  pending_transactions_ = 0U;
  return {};
}

expected<void, LifecycleTransactionKernelError> LifecycleTransactionKernel::retire_receipt(
    LifecycleRestartReceipt receipt) noexcept {
  const auto record =
      std::find_if(receipts_.begin(), receipts_.end(), [&receipt](const auto& value) {
        return value.view.receipt.receipt_id == receipt.receipt_id;
      });
  if (record == receipts_.end() || record->view.receipt.receipt_id != receipt.receipt_id ||
      record->view.receipt.acceptance_sequence != receipt.acceptance_sequence ||
      record->view.receipt.target != receipt.target ||
      record->view.receipt.requested_stage != receipt.requested_stage) {
    return make_unexpected(error(LifecycleTransactionKernelErrorCode::kUnknownReceipt));
  }
  if (!is_terminal(record->view.state)) {
    return make_unexpected(error(LifecycleTransactionKernelErrorCode::kInvalidTransition));
  }
  reserved_events_ -= record->reserved_events;
  receipts_.erase(record);
  return {};
}

expected<LifecycleTransactionReceiptView, LifecycleTransactionKernelError>
LifecycleTransactionKernel::inspect(LifecycleRestartReceipt receipt) const noexcept {
  const ReceiptRecord* const record = find_receipt(receipt.receipt_id);
  if (record == nullptr || record->view.receipt.receipt_id != receipt.receipt_id ||
      record->view.receipt.acceptance_sequence != receipt.acceptance_sequence ||
      record->view.receipt.target != receipt.target ||
      record->view.receipt.requested_stage != receipt.requested_stage) {
    return make_unexpected(error(LifecycleTransactionKernelErrorCode::kUnknownReceipt));
  }
  return record->view;
}

expected<LifecycleTransactionState, LifecycleTransactionKernelError>
LifecycleTransactionKernel::transaction_state(LifecycleTransactionId transaction) const noexcept {
  const TransactionSlot* const record = find_transaction(transaction);
  if (record == nullptr) {
    return make_unexpected(error(LifecycleTransactionKernelErrorCode::kUnknownTransaction));
  }
  return record->state;
}

expected<ParamGeneration, LifecycleTransactionKernelError>
LifecycleTransactionKernel::candidate_generation(LifecycleTransactionId transaction,
                                                 OperatorInstanceId owner) const noexcept {
  const TransactionSlot* const record = find_transaction(transaction);
  if (record == nullptr) {
    return make_unexpected(error(LifecycleTransactionKernelErrorCode::kUnknownTransaction));
  }
  const auto candidate = std::find_if(
      record->candidates.begin(), record->candidates.end(), [owner](const CandidateRecord& value) {
        return value.owner == owner;
      });
  if (candidate == record->candidates.end()) {
    return make_unexpected(error(LifecycleTransactionKernelErrorCode::kInvalidRequest));
  }
  return candidate->generation;
}

expected<LifecycleStage, LifecycleTransactionKernelError>
LifecycleTransactionKernel::effective_stage(LifecycleTransactionId transaction) const noexcept {
  const TransactionSlot* const record = find_transaction(transaction);
  if (record == nullptr) {
    return make_unexpected(error(LifecycleTransactionKernelErrorCode::kUnknownTransaction));
  }
  return record->effective_stage;
}

expected<std::span<const OperatorInstanceId>, LifecycleTransactionKernelError>
LifecycleTransactionKernel::transaction_members(LifecycleTransactionId transaction) const noexcept {
  const TransactionSlot* const record = find_transaction(transaction);
  if (record == nullptr) {
    return make_unexpected(error(LifecycleTransactionKernelErrorCode::kUnknownTransaction));
  }
  return std::span<const OperatorInstanceId>{record->members};
}

bool LifecycleTransactionKernel::is_fenced(OperatorInstanceId owner) const noexcept {
  return find_fence(owner) != nullptr;
}

}  // namespace holoscan::detail
