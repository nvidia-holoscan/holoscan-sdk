/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "deterministic_serial_ready_policy.hpp"

#include <cassert>
#include <limits>
#include <memory>
#include <string>

#include "core/source_id_internal.hpp"

namespace holoscan::detail::scheduling {
namespace {

[[nodiscard]] unexpected<Error> plan_error() {
  return make_unexpected(
      Error{ErrorCode::kInvalidArgument, std::string{"READY_POLICY_PLAN_INVALID"}});
}

[[nodiscard]] bool candidate_less(const ReadyCandidateView& lhs,
                                  const ReadyCandidateView& rhs) noexcept {
  if (lhs.tag_time != rhs.tag_time) {
    return lhs.tag_time < rhs.tag_time;
  }
  if (lhs.plan_vertex_order != rhs.plan_vertex_order) {
    return lhs.plan_vertex_order < rhs.plan_vertex_order;
  }

  const auto lhs_source = source_id_value(lhs.stable_source_id);
  const auto rhs_source = source_id_value(rhs.stable_source_id);
  if (lhs_source != rhs_source) {
    return lhs_source < rhs_source;
  }
  if (lhs.stable_port_id.value() != rhs.stable_port_id.value()) {
    return lhs.stable_port_id.value() < rhs.stable_port_id.value();
  }
  return lhs.stable_candidate_tie < rhs.stable_candidate_tie;
}

}  // namespace

AlgorithmKey deterministic_serial_ready_policy_algorithm() {
  return AlgorithmKey{std::string{deterministic_serial_ready_policy_id()},
                      kDeterministicSerialReadyPolicyVersion};
}

ReadyPolicyDescriptor DeterministicSerialReadyPolicy::descriptor() const {
  return ReadyPolicyDescriptor{
      .algorithm = deterministic_serial_ready_policy_algorithm(),
      .interface_version = kReadyPolicyInterfaceVersion,
      .maturity = AlgorithmMaturity::kProduction,
      .required_capabilities = kAllReadyPolicyCapabilities,
  };
}

expected<void, Error> DeterministicSerialReadyPolicy::initialize(const SchedulerPlanView& plan,
                                                                 ReadyPolicyState& state) {
  // Runtime creates one policy instance per partition runner. The view retains the complete plan's
  // lane count, but normalizes concurrency and overlap to this one partition-local serial owner.
  if (state.initialized || state.activation_in_progress || plan.scheduler == nullptr ||
      plan.max_ready_candidates == 0U ||
      plan.max_ready_candidates > std::numeric_limits<std::uint32_t>::max() ||
      plan.scheduler->ready_policy.id != deterministic_serial_ready_policy_id() ||
      plan.scheduler->ready_policy.version != kDeterministicSerialReadyPolicyVersion ||
      plan.scheduler->execution_lane_count == 0U ||
      plan.scheduler->max_concurrent_activations != 1U || plan.scheduler->activation_overlap ||
      plan.scheduler->worker_count != 1U || plan.scheduler->intra_path_parallelism != 1U ||
      !plan.available_capabilities.contains(kAllReadyPolicyCapabilities)) {
    return plan_error();
  }

  state.max_ready_candidates = plan.max_ready_candidates;
  state.initialized = true;
  state.activation_in_progress = false;
  return {};
}

ReadyCandidateIndex DeterministicSerialReadyPolicy::select(ReadySetView ready,
                                                           ReadyPolicyState& state) noexcept {
  if (!state.initialized || state.activation_in_progress || ready.candidates.empty() ||
      ready.candidates.size() > state.max_ready_candidates) {
    return {};
  }

  std::size_t selected = ReadyCandidateIndex::kInvalid;
  for (std::size_t index = 0U; index < ready.candidates.size(); ++index) {
    if (ready.candidates[index].active &&
        (selected == ReadyCandidateIndex::kInvalid ||
         candidate_less(ready.candidates[index], ready.candidates[selected]))) {
      selected = index;
    }
  }
  return ReadyCandidateIndex{selected};
}

void DeterministicSerialReadyPolicy::on_activation_started(const ReadyCandidateView&,
                                                           ReadyPolicyState& state) noexcept {
  assert(state.initialized);
  assert(!state.activation_in_progress);
  state.activation_in_progress = true;
}

void DeterministicSerialReadyPolicy::on_activation_completed(const ReadyCandidateView&,
                                                             ReadyPolicyState& state) noexcept {
  assert(state.initialized);
  assert(state.activation_in_progress);
  state.activation_in_progress = false;
}

std::unique_ptr<ReadyPolicy> make_deterministic_serial_ready_policy() {
  return std::make_unique<DeterministicSerialReadyPolicy>();
}

}  // namespace holoscan::detail::scheduling
