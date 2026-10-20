/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime_support_thread_placement.hpp"

#include <algorithm>
#include <string>
#include <utility>

#include "../compiler/support_thread_affinity_contracts.hpp"
#include "../threading/linux_thread_controller.hpp"
#include "../threading/thread_placement_contracts.hpp"

namespace holoscan::detail {
namespace {

[[nodiscard]] expected<void, Error> fail_support_thread_placement(
    SupportThreadRole role, RuntimeSupportThreadPlacementPhase phase, Error error,
    const CPUAffinity& requested, std::optional<CPUAffinity> observed,
    RuntimeSupportThreadPlacementFailure* failure) {
  if (failure != nullptr) {
    *failure = RuntimeSupportThreadPlacementFailure{
        .present = true,
        .role = role,
        .phase = phase,
        .error = error,
        .requested = requested,
        .observed = std::move(observed),
    };
  }
  return make_unexpected(std::move(error));
}

}  // namespace

std::string_view runtime_support_thread_placement_phase_name(
    RuntimeSupportThreadPlacementPhase phase) noexcept {
  switch (phase) {
    case RuntimeSupportThreadPlacementPhase::kValidate:
      return "runtime_validate";
    case RuntimeSupportThreadPlacementPhase::kApply:
      return "runtime_apply";
    case RuntimeSupportThreadPlacementPhase::kReadback:
      return "runtime_readback";
    case RuntimeSupportThreadPlacementPhase::kVerify:
      return "runtime_verify";
  }
  return "runtime_unknown";
}

std::string runtime_support_thread_placement_required(
    const RuntimeSupportThreadPlacementFailure& failure) {
  return "role=" + std::string(support_thread_role_name(failure.role)) +
         ",requested=" + support_thread_affinity_summary(failure.requested);
}

std::string runtime_support_thread_placement_provided(
    const RuntimeSupportThreadPlacementFailure& failure) {
  std::string result =
      "phase=" + std::string(runtime_support_thread_placement_phase_name(failure.phase));
  if (failure.observed) {
    result += ",observed=" + support_thread_affinity_summary(*failure.observed);
  }
  return result;
}

expected<void, Error> apply_support_thread_affinity(
    std::span<const SupportThreadAffinityPlanRecord> records, SupportThreadRole role,
    RuntimeSupportThreadPlacementFailure* failure, std::optional<CPUAffinity>* previous_affinity) {
  if (failure != nullptr) {
    *failure = RuntimeSupportThreadPlacementFailure{};
  }
  if (previous_affinity != nullptr) {
    previous_affinity->reset();
  }
  const auto found = std::find_if(
      records.begin(), records.end(), [role](const auto& record) { return record.role == role; });
  if (found == records.end()) {
    return {};
  }
  const auto invalid_record = [&] {
    return fail_support_thread_placement(
        role,
        RuntimeSupportThreadPlacementPhase::kValidate,
        Error{ErrorCode::kInvalidArgument, "invalid Runtime support-thread affinity record"},
        found->requested.affinity.value_or(CPUAffinity{}),
        std::nullopt,
        failure);
  };
  const auto& accepted = found->accepted;
  if (found->disposition != PlanDisposition::kAccepted || found->reason || !accepted.has_value()) {
    return invalid_record();
  }
  const auto& affinity = accepted->affinity;
  if (!affinity.has_value()) {
    return invalid_record();
  }
  const CPUAffinity& requested = *affinity;
#if defined(__linux__)
  if (previous_affinity != nullptr) {
    auto previous = LinuxThreadController::snapshot();
    if (!previous) {
      return fail_support_thread_placement(role,
                                           RuntimeSupportThreadPlacementPhase::kReadback,
                                           std::move(previous).error(),
                                           requested,
                                           std::nullopt,
                                           failure);
    }
    *previous_affinity = std::move(previous->affinity);
  }
  auto applied = LinuxThreadController::apply(*accepted);
  if (!applied) {
    return fail_support_thread_placement(role,
                                         RuntimeSupportThreadPlacementPhase::kApply,
                                         std::move(applied).error(),
                                         requested,
                                         std::nullopt,
                                         failure);
  }
  auto observed = LinuxThreadController::snapshot();
  if (!observed) {
    return fail_support_thread_placement(role,
                                         RuntimeSupportThreadPlacementPhase::kReadback,
                                         std::move(observed).error(),
                                         requested,
                                         std::nullopt,
                                         failure);
  }
  if (!thread_affinity_readback_matches(affinity, observed->affinity)) {
    return fail_support_thread_placement(
        role,
        RuntimeSupportThreadPlacementPhase::kVerify,
        Error{ErrorCode::kIncompatible, "Runtime support-thread affinity readback mismatch"},
        requested,
        std::move(observed->affinity),
        failure);
  }
  return {};
#else
  return fail_support_thread_placement(
      role,
      RuntimeSupportThreadPlacementPhase::kApply,
      Error{ErrorCode::kNotSupported, "Runtime support-thread affinity is supported only on Linux"},
      requested,
      std::nullopt,
      failure);
#endif
}

}  // namespace holoscan::detail
