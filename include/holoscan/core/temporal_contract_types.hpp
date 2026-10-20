/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <concepts>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/core/port_ref.hpp>
#include <holoscan/time/clock_registry.hpp>
#include <holoscan/time/time_point.hpp>

namespace holoscan {

/// @brief Select the sample at the exact frozen event-time reference.
struct Exact {};

/// @brief Select every causally available FIFO survivor as one batch.
///
/// The selector is intentionally fieldless. Compilation derives its finite maximum from retained
/// queue and cache geometry; adding an authored limit would change its consume-all contract.
struct DrainAll {};

/// @brief Select strict outer brackets and the inclusive middle between consecutive anchors.
struct BracketedWindow {
  std::uint32_t lookbehind{1};  ///< Positive count of nearest strict predecessors.
  std::uint32_t lookahead{1};   ///< Positive count of earliest strict successors.
  std::uint32_t max_samples{};  ///< Required positive all-or-nothing complete-set bound.
};

/// @brief Suppress one candidate when the corresponding selector cannot produce a value.
struct Skip {};

/// @brief Enter the operator-authored fieldless degraded path without fabricating input data.
struct SafeDefault {};

/// @brief Reuse one previously admitted scalar selection within a finite inclusive age.
struct ReuseLast {
  std::chrono::nanoseconds max_age{};  ///< Required positive inclusive retention age.
};

/// @brief Trigger when any connected input accepts a sample.
struct OnAnyInput {};

/// @brief Trigger once for every accepted sample on one local input.
struct OnEach {
  PortRef port;  ///< Borrowed authoring port copied into the owning \ref Contract.
};

/// @brief Coalesce late periodic ticks while preserving one fixed phase.
enum class ClockTickPolicy : std::uint8_t {
  kCoalesceLateTicks = 0,  ///< Freeze only the greatest due unhandled tick in O(1).
};

/// @brief Trigger from one positive-period clock schedule.
struct OnClock {
  std::chrono::nanoseconds period{};    ///< Required positive representable period.
  ClockKey clock{};                     ///< Cold key; omission selects the graph default clock.
  std::optional<TimePoint> start_at{};  ///< Optional absolute phase in a comparable domain.
  ClockTickPolicy tick_policy{ClockTickPolicy::kCoalesceLateTicks};  ///< Closed baseline policy.
};

/// @brief Own a cold declaration-order list of borrowed input port references.
///
/// This value may allocate during authoring. A \ref Contract immediately copies each referenced
/// name and direction, so neither this list nor an authoring address enters a plan or run session.
class PortList {
 public:
  /// @brief Construct an empty list.
  PortList() = default;

  /// @brief Construct a declaration-order list from borrowed references.
  /// @param ports Borrowed references whose declarations outlive this cold value.
  explicit PortList(std::vector<PortRef> ports) : ports_(std::move(ports)) {}

  /// @brief Return declaration-order borrowed references.
  /// @return Span valid until this list is modified or destroyed.
  [[nodiscard]] std::span<const PortRef> values() const noexcept { return ports_; }

 private:
  std::vector<PortRef> ports_;
};

/// @brief Create an ordered port list for a closed multi-input trigger.
/// @tparam Ports Input-port declaration types convertible to \ref PortRef.
/// @param ports Stable input declarations borrowed for the duration of contract construction.
/// @return Cold declaration-order list.
template <typename... Ports>
  requires(sizeof...(Ports) > 0 &&
           (std::derived_from<std::remove_cvref_t<Ports>, InputBase> && ...))
[[nodiscard]] PortList ports(const Ports&... ports) {
  std::vector<PortRef> refs;
  refs.reserve(sizeof...(Ports));
  (refs.emplace_back(ports), ...);
  return PortList{std::move(refs)};
}

/// @brief Trigger a deterministic consuming FIFO-head alignment over a closed port set.
struct OnAligned {
  PortList ports;                     ///< At least two declaration-order member inputs.
  std::chrono::nanoseconds window{};  ///< Nonnegative inclusive maximum head spread.
};

/// @brief Trigger one bounded pending interval for each accepted anchor after priming.
struct OnWindow {
  PortRef anchor;                        ///< Local exact anchor input.
  std::chrono::nanoseconds max_wait{};   ///< Required positive processing wait bound.
  ClockKey wait_clock{};                 ///< Omission selects the graph default clock.
  std::uint32_t max_pending_windows{1};  ///< Required positive pending-record capacity.
};

/// @brief Trigger predicate evaluation for every accepted exact driver sample.
struct OnInputUpdate {
  PortRef port;  ///< Local scalar driver pinned as the implicit first evaluator input.
};

/// @brief Identify a stable semantic version for independently keyed predicate logic.
struct SemanticVersion {
  std::uint16_t major{};  ///< Breaking semantic version component.
  std::uint16_t minor{};  ///< Backward-compatible semantic version component.
  std::uint16_t patch{};  ///< Corrective semantic version component.

  friend constexpr bool operator==(SemanticVersion, SemanticVersion) noexcept = default;
};

/// @brief Opt into a qualified predicate identity independent of one operator instance.
struct PredicateKey {
  std::string_view qualified_name;     ///< Nonempty borrowed identity copied by \ref predicate.
  SemanticVersion semantic_version{};  ///< Explicit semantic version for replay evidence.
};

/// @brief Identify when a successful predicate result creates a compute candidate.
enum class PredicateFiring : std::uint8_t {
  kOnFalseToTrue = 0,      ///< Fire only after remembered false changes to true.
  kOnEveryTrueEvaluation,  ///< Fire after every successful true evaluation.
};

/// @brief Classify the owned callable form for later profile admission.
enum class PredicateCallableKind : std::uint8_t {
  kStaticFunction = 0,      ///< Free or static function pointer.
  kNonCapturingLambda = 1,  ///< Stateless fixed-signature callable value.
  kInlineFunctor = 2,       ///< Fixed-size owned functor with inspectable inline state.
  kOpaqueClosure = 3,       ///< Owned closure whose captured semantics are opaque to Core.
  kBoundMember = 4,         ///< Pointer-free self-member descriptor rebound per session.
  kTypeErasedFunction = 5,  ///< Explicit typed \c std::function value.
  kPythonFunction = 6,      ///< Reserved later Standard/Research Python function adapter.
  kPythonLambda = 7,        ///< Reserved later Standard/Research Python lambda adapter.
  kPythonBoundMethod = 8,   ///< Reserved later Standard/Research Python method adapter.
};

}  // namespace holoscan
