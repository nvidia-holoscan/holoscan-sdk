/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/tag.hpp>

namespace holoscan::detail::scheduling {

/// Identify the normalized runtime facts required by a private READY policy.
enum class ReadyPolicyCapability : std::uint32_t {
  kDeterministicCandidateOrder = 1U << 0U,
  kPartitionLocalSerialExecution = 1U << 1U,
};

/// Carry a finite set of normalized READY-policy capabilities.
struct ReadyPolicyCapabilities {
  std::uint32_t bits{};

  /// Return whether this set contains every bit in \p required.
  [[nodiscard]] constexpr bool contains(ReadyPolicyCapabilities required) const noexcept {
    return (bits & required.bits) == required.bits;
  }

  /// Add one capability to this set.
  constexpr void add(ReadyPolicyCapability capability) noexcept {
    bits |= static_cast<std::uint32_t>(capability);
  }

  friend constexpr bool operator==(ReadyPolicyCapabilities,
                                   ReadyPolicyCapabilities) noexcept = default;
};

/// Return every READY-policy capability supplied by the EA1 runtime.
[[nodiscard]] constexpr ReadyPolicyCapabilities all_ready_policy_capabilities() noexcept {
  ReadyPolicyCapabilities capabilities;
  capabilities.add(ReadyPolicyCapability::kDeterministicCandidateOrder);
  capabilities.add(ReadyPolicyCapability::kPartitionLocalSerialExecution);
  return capabilities;
}

inline constexpr auto kAllReadyPolicyCapabilities = all_ready_policy_capabilities();
inline constexpr std::uint32_t kReadyPolicyInterfaceVersion = 1U;

/// Describe one private READY-policy implementation.
struct ReadyPolicyDescriptor {
  AlgorithmKey algorithm;
  std::uint32_t interface_version{};
  AlgorithmMaturity maturity{AlgorithmMaturity::kProduction};
  ReadyPolicyCapabilities required_capabilities;
};

/// Borrow the finite scheduler facts needed to initialize a READY policy.
///
/// The scheduler record remains owned by the immutable ExecutionPlan. Runtime constructs this
/// view once, before user callbacks, and retains neither a compiler catalog nor a Graph traversal.
struct SchedulerPlanView {
  const SchedulerPlanRecord* scheduler{};
  std::size_t max_ready_candidates{};
  ReadyPolicyCapabilities available_capabilities;
};

/// Borrow one candidate already admitted to the Dispatcher READY set.
///
/// Tag identity is intentionally outside this view. The Dispatcher reserves a Tag when it freezes
/// the source/witness, before policy selection, but never exposes that identity to the policy, so
/// it cannot accidentally become priority.
/// Partition and vertex identities remain available for evidence and future policies, but the EA1
/// deterministic-serial policy does not compare them.
struct ReadyCandidateView {
  ExecutionPartitionId partition;
  std::uint32_t execution_lane{};
  VertexId vertex;
  TagTime tag_time;
  std::uint32_t plan_vertex_order{};
  SourceId stable_source_id;
  PortId stable_port_id;
  std::uint64_t stable_candidate_tie{};
  bool active{true};
};

/// Borrow the current immutable, non-owning READY candidate set.
struct ReadySetView {
  std::span<const ReadyCandidateView> candidates;
};

/// Identify one entry in a borrowed READY set.
struct ReadyCandidateIndex {
  static constexpr std::size_t kInvalid = std::numeric_limits<std::size_t>::max();

  std::size_t value{kInvalid};

  /// Return whether this index names one candidate.
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != kInvalid; }

  friend constexpr bool operator==(ReadyCandidateIndex, ReadyCandidateIndex) noexcept = default;
};

/// Hold the complete bounded per-session state used by an EA1 READY policy.
///
/// The state is initialized once from accepted plan maxima. It owns no container, string, graph,
/// planner, catalog, or resource and therefore cannot allocate or grow on the selection path.
struct ReadyPolicyState {
  std::size_t max_ready_candidates{};
  bool initialized{};
  bool activation_in_progress{};
};

/// Select one candidate from an already-admitted READY set.
class ReadyPolicy {
 public:
  virtual ~ReadyPolicy() = default;

  /// Return stable algorithm, interface, maturity, and capability identity.
  [[nodiscard]] virtual ReadyPolicyDescriptor descriptor() const = 0;

  /// Initialize bounded policy state from one accepted immutable scheduler plan.
  [[nodiscard]] virtual expected<void, Error> initialize(const SchedulerPlanView& plan,
                                                         ReadyPolicyState& state) = 0;

  /// Return the selected index in \p ready without taking ownership or mutating readiness.
  ///
  /// @pre \p state was initialized successfully and \p ready is nonempty, valid, and no larger
  /// than the admitted maximum.
  [[nodiscard]] virtual ReadyCandidateIndex select(ReadySetView ready,
                                                   ReadyPolicyState& state) noexcept = 0;

  /// Record that the selected activation entered its partition-local serial interval.
  virtual void on_activation_started(const ReadyCandidateView& candidate,
                                     ReadyPolicyState& state) noexcept = 0;

  /// Record that output and state commit completed for the selected activation.
  virtual void on_activation_completed(const ReadyCandidateView& candidate,
                                       ReadyPolicyState& state) noexcept = 0;
};

}  // namespace holoscan::detail::scheduling
