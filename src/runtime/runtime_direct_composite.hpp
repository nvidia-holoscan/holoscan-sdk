/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/source_id.hpp>

#include "runtime_callback_context.hpp"

namespace holoscan::detail {

class PortBinding;

inline constexpr std::size_t kRuntimeDirectCompositeCallbackCapacity = 2U;

/// Describe how an attempted direct composite handled continuation evidence.
enum class RuntimeDirectCompositeContinuationDisposition : std::uint8_t {
  kNotAttempted = 0,
  kOrdinaryPath,
  kAccepted,
  kRejected,
  kFailed,
};

/// Describe which owner resumes execution after one direct-composite lane task.
enum class RuntimeDirectCompositeResumeDisposition : std::uint8_t {
  kControllerRetiresSource = 0,
  kControllerResumesScan,
  kCompositeCompleted,
  kTerminalFailure,
};

/// Identify the bounded phase deltas returned by one direct-composite lane task.
enum class RuntimeDirectCompositePhaseDelta : std::uint8_t {
  kCallbacksAttempted = 0,
  kCallbacksCompleted,
  kActivationsRetired,
  kActivationCuts,
  kContinuationReceipts,
  kCount,
};

inline constexpr std::size_t kRuntimeDirectCompositePhaseDeltaCount =
    static_cast<std::size_t>(RuntimeDirectCompositePhaseDelta::kCount);

/// Cache one startup-validated direct-composite destination without a warm scan.
///
/// The borrowed binding is owned by the enclosing ReadyRunState. Construction
/// finishes after binding reconstruction and before the first dispatch while
/// every lane is idle. The synchronous lane dispatch transfers exclusive access
/// to the cache together with the rest of the controller-owned invocation.
struct RuntimeDirectCompositeBinding {
  std::size_t producer_receiver{std::numeric_limits<std::size_t>::max()};
  std::size_t successor_receiver{std::numeric_limits<std::size_t>::max()};
  VertexId producer{};
  VertexId successor{};
  PortId source_output{};
  PortId successor_input{};
  SourceId readiness_source{};
  PortBinding* successor_binding{};
  std::uint32_t execution_lane{};

  [[nodiscard]] bool valid() const noexcept {
    return producer_receiver != std::numeric_limits<std::size_t>::max() &&
           successor_receiver != std::numeric_limits<std::size_t>::max() && producer && successor &&
           source_output && successor_input && readiness_source && successor_binding != nullptr;
  }
};

/// Return one fixed-capacity direct-composite lane transaction to the controller.
///
/// The common successful path owns no dynamic storage. A terminal failure may
/// populate the optional Error message in the same way as ordinary Runtime
/// failure reporting. Every callback retains an independent identity even
/// though both execute within one physical lane task.
///
/// The controller owns GraphExec, ReadyRunState, the validated binding cache,
/// source invocation/selection/candidate, Parameter binding, and this report
/// before dispatch. Sleeping-lane dispatch publishes those borrowed fields
/// under the lane mutex and condition-variable predicate; completion publishes
/// the report and all mutations under the same mutex before the controller
/// reacquires it. Busy-lane dispatch publishes the task with a release store,
/// the lane consumes it with an acquire load, then returns all mutations with a
/// release store consumed by the controller's acquire load. The process-wide
/// dispatch token excludes any second publisher in both modes. The controller
/// neither reads nor mutates a borrowed field until that completion edge, and
/// the lane retains no pointer after it. The PortBinding pointer remains valid
/// because its enclosing ReadyRunState outlives the synchronous transaction.
struct RuntimeDirectCompositeCompletionReport {
  std::array<RuntimeCallbackIdentity, kRuntimeDirectCompositeCallbackCapacity>
      callback_identities{};
  std::array<ErrorCode, kRuntimeDirectCompositeCallbackCapacity> callback_errors{
      ErrorCode::kSuccess, ErrorCode::kSuccess};
  std::array<std::uint64_t, kRuntimeDirectCompositePhaseDeltaCount> phase_deltas{};
  Error first_error{ErrorCode::kSuccess};
  RuntimeDirectCompositeContinuationDisposition continuation{
      RuntimeDirectCompositeContinuationDisposition::kNotAttempted};
  RuntimeDirectCompositeResumeDisposition resume{
      RuntimeDirectCompositeResumeDisposition::kControllerRetiresSource};
  std::uint8_t callbacks_attempted{};
  std::uint8_t callbacks_completed{};
  bool source_retired{};
  bool successor_retired{};
  bool activation_cut_completed{};
  bool stop_observed{};
  bool restart_observed{};

  [[nodiscard]] bool complete() const noexcept {
    constexpr auto phase = [](RuntimeDirectCompositePhaseDelta delta) {
      return static_cast<std::size_t>(delta);
    };
    const RuntimeCallbackIdentity& source = callback_identities[0];
    const RuntimeCallbackIdentity& successor = callback_identities[1];
    const bool next_logical_time = successor.tag_time.logical_ns > source.tag_time.logical_ns &&
                                   successor.tag_time.microstep == 0U;
    const bool next_microstep =
        successor.tag_time.logical_ns == source.tag_time.logical_ns &&
        source.tag_time.microstep != std::numeric_limits<std::uint32_t>::max() &&
        successor.tag_time.microstep == source.tag_time.microstep + 1U;
    const bool next_coordinate = next_logical_time || next_microstep;
    return resume == RuntimeDirectCompositeResumeDisposition::kCompositeCompleted &&
           continuation == RuntimeDirectCompositeContinuationDisposition::kAccepted &&
           callbacks_attempted == kRuntimeDirectCompositeCallbackCapacity &&
           callbacks_completed == kRuntimeDirectCompositeCallbackCapacity && source_retired &&
           successor_retired && activation_cut_completed && !stop_observed && !restart_observed &&
           first_error.code == ErrorCode::kSuccess && callback_errors[0] == ErrorCode::kSuccess &&
           callback_errors[1] == ErrorCode::kSuccess &&
           source.context == RuntimeCallbackContext::kCompute &&
           successor.context == RuntimeCallbackContext::kCompute && source.operator_id != 0U &&
           successor.operator_id != 0U && source.operator_id != successor.operator_id &&
           source.tag.valid() && successor.tag.valid() && source.tag_time.valid() &&
           successor.tag_time.valid() && next_coordinate &&
           phase_deltas[phase(RuntimeDirectCompositePhaseDelta::kCallbacksAttempted)] == 2U &&
           phase_deltas[phase(RuntimeDirectCompositePhaseDelta::kCallbacksCompleted)] == 2U &&
           phase_deltas[phase(RuntimeDirectCompositePhaseDelta::kActivationsRetired)] == 2U &&
           phase_deltas[phase(RuntimeDirectCompositePhaseDelta::kActivationCuts)] == 2U &&
           phase_deltas[phase(RuntimeDirectCompositePhaseDelta::kContinuationReceipts)] == 1U;
  }
};

static_assert(
    std::tuple_size_v<decltype(RuntimeDirectCompositeCompletionReport::callback_identities)> ==
    kRuntimeDirectCompositeCallbackCapacity);

}  // namespace holoscan::detail
