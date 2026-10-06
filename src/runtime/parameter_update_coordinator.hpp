/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/parameter_control.hpp>

#include "../core/execution_context_internal.hpp"

namespace holoscan::detail {

/// @brief Bind one run-owned Operator to its admitted Parameter plan and local descriptors.
///
/// Every pointer borrows immutable plan, static descriptor, or Operator storage that must outlive
/// the coordinator. Construction verifies the pointer-free plan facts before copying generation
/// zero into bounded store-owned storage.
struct ParameterRuntimeOperatorBinding {
  OperatorRef run_bound_owner{};                     ///< Exact active run/plan/Operator identity.
  std::vector<std::string> operator_path;            ///< Canonical decoded Operator path.
  PlanDigest plan_digest{};                          ///< Exact semantic source-plan identity.
  const OperatorParameterPlanRecord* plan{};         ///< Pointer-free admission record.
  std::span<const ParameterFieldPlanRecord> fields;  ///< This Operator's plan field records.
  std::span<const ParameterDerivedTermPlanRecord>
      derived_terms;  ///< This Operator's pointer-free temporal Parameter dependencies.
  const ParamTypeDescriptor* type{};      ///< Defining-module whole-aggregate operations.
  const ParamSchemaDescriptor* schema{};  ///< Defining-module schema, or null for opaque mode.
  const void* initial_value{};            ///< Live run-owned generation-zero aggregate.
};

/// @brief Retain one generation-coherent canonical duration term for runtime selection.
struct ParameterRuntimeDerivedTermValue {
  ParameterDerivedTermKind kind{ParameterDerivedTermKind::kLatestWithinMaxAge};
  PortId input{};
  ParamFieldId field{};
  std::int64_t value_ns{};
};

/// @brief Describe one fully prepared Parameter-owned lifecycle candidate to the shared executor.
///
/// The candidate is already copied, patched, aggregate-validated, and backed by a finite immutable
/// generation slot. The callee must either reject without retaining any identity or atomically
/// reserve the complete lifecycle transaction before returning success. It never owns the value
/// or retains this callback-lifetime request. The callback runs under the Parameter coordinator's
/// mutation lock and therefore must not call back into the coordinator.
struct ParameterLifecycleSubmitRequest {
  OperatorRef owner{};           ///< Exact current-run candidate owner.
  ParamGeneration generation{};  ///< Complete candidate generation retained by this coordinator.
  LifecycleStage earliest_stage{LifecycleStage::kConfigure};  ///< Earliest invalidated stage.
  bool equivalent_to_published{};  ///< Complete semantic equality with the published generation.
};

/// @brief Return the shared lifecycle kernel's accepted lineage for one Parameter candidate.
struct ParameterLifecycleSubmitOutcome {
  LifecycleRestartReceipt receipt{};  ///< Per-submission lifecycle receipt.
  LifecycleTransactionState state{LifecycleTransactionState::kRejected};  ///< Immediate state.
  LifecycleTransactionId transaction{};     ///< Owning transaction, or zero for no-change.
  LifecycleTransactionId coalesced_into{};  ///< Existing transaction selected by coalescing.
  LifecycleTransactionId
      superseded_transaction{};  ///< Transaction removed by an equivalent return.
};

/// @brief Submit one prepared candidate to the shared bounded lifecycle kernel.
///
/// Rejection retains no lifecycle identity or resource. Success atomically owns every required
/// lifecycle resource and returns complete lineage. The function must not block on or reenter the
/// Parameter coordinator.
using ParameterLifecycleSubmitFunction = expected<ParameterLifecycleSubmitOutcome, Error> (*)(
    void*, const ParameterLifecycleSubmitRequest&) noexcept;

/// @brief Unwind one structurally malformed success returned by the lifecycle submit callback.
///
/// Task11 invokes this callback exactly once before discarding its local candidate and result/event
/// reservations. It is not called for ordinary typed rejection and must not reenter the Parameter
/// coordinator.
using ParameterLifecycleCancelFunction = void (*)(void*,
                                                  const ParameterLifecycleSubmitOutcome&) noexcept;

/// @brief Bind the Parameter candidate store to the RunSession-owned lifecycle executor.
struct ParameterLifecycleSubmitBinding {
  void* state{};                              ///< RunSession-owned executor state.
  ParameterLifecycleSubmitFunction submit{};  ///< Atomic nine-resource lifecycle admission.
  ParameterLifecycleCancelFunction cancel_accepted{};  ///< Roll back malformed accepted output.
};

/// @brief Retain one callback-lifetime pin into a fixed Parameter generation slot.
///
/// The caller must invoke \ref ParameterUpdateCoordinator::release exactly once after the
/// callback retires. Copies are forbidden so one pin cannot be released twice.
class ParameterGenerationPin {
 public:
  ParameterGenerationPin() noexcept = default;
  ParameterGenerationPin(const ParameterGenerationPin&) = delete;
  ParameterGenerationPin& operator=(const ParameterGenerationPin&) = delete;
  ParameterGenerationPin(ParameterGenerationPin&& other) noexcept;
  ParameterGenerationPin& operator=(ParameterGenerationPin&& other) noexcept;

  /// @brief Return the coherent pinned header for callback binding.
  /// @return Stable header, or null for an inert/moved-from pin.
  [[nodiscard]] const PinnedParameterHeader* header() const noexcept { return header_; }

  /// @brief Return whether this value owns one unreleased pin.
  /// @return \c true when \ref header is non-null.
  [[nodiscard]] explicit operator bool() const noexcept { return header_ != nullptr; }

  /// @brief Return one canonical duration term from the same immutable generation.
  /// @param kind Selector/deadline term kind.
  /// @param input Exact input for per-input terms, or zero for Operator-wide terms.
  /// @return Signed canonical nanoseconds, or empty when this generation has no such term.
  [[nodiscard]] std::optional<std::int64_t> duration_term(ParameterDerivedTermKind kind,
                                                          PortId input = {}) const noexcept;

  /// @brief Return one cold-prebound canonical duration term in constant time.
  /// @param index Operator-local index validated against the immutable plan at startup.
  /// @param kind Expected selector/deadline term kind.
  /// @param input Exact input for a per-input term, or zero for an Operator-wide term.
  /// @return Signed canonical nanoseconds, or empty for a stale/mismatched index.
  [[nodiscard]] std::optional<std::int64_t> duration_term_at(std::uint32_t index,
                                                             ParameterDerivedTermKind kind,
                                                             PortId input = {}) const noexcept;

 private:
  ParameterGenerationPin(const PinnedParameterHeader* header, void* store_cookie,
                         std::uint32_t slot, const ParameterRuntimeDerivedTermValue* derived_terms,
                         std::uint32_t derived_term_count) noexcept
      : header_(header),
        store_cookie_(store_cookie),
        slot_(slot),
        derived_terms_(derived_terms),
        derived_term_count_(derived_term_count) {}

  const PinnedParameterHeader* header_{};
  void* store_cookie_{};
  std::uint32_t slot_{};
  const ParameterRuntimeDerivedTermValue* derived_terms_{};
  std::uint32_t derived_term_count_{};

  friend class ParameterUpdateCoordinator;
  friend class ParameterGenerationStore;
};

class ParameterGenerationStore;

/// @brief Coordinate bounded immutable Parameter generations for one active run.
///
/// All queues, generation slots, result/audit/event ledgers, and Operator stores are allocated
/// during \ref create. After construction, submission, publication, pinning, reads, result lookup,
/// closure, and detach perform no container growth and use no fallback allocation.
class ParameterUpdateCoordinator {
 public:
  ParameterUpdateCoordinator(const ParameterUpdateCoordinator&) = delete;
  ParameterUpdateCoordinator& operator=(const ParameterUpdateCoordinator&) = delete;
  ~ParameterUpdateCoordinator();

  /// @brief Construct and populate all fixed Parameter generation stores.
  /// @param bindings Exact process-local run-owned Operator bindings in canonical identity order.
  /// @param capacity Compile-admitted finite capacity record.
  /// @param operator_slot_count Full plan Operator identity bound. Zero uses the bindings size.
  /// @return Fully initialized coordinator, or an error before any runtime authority is exposed.
  [[nodiscard]] static expected<std::unique_ptr<ParameterUpdateCoordinator>, Error> create(
      std::span<const ParameterRuntimeOperatorBinding> bindings,
      ParameterRuntimeCapacityPlanRecord capacity, std::uint32_t operator_slot_count = 0U);

  /// @brief Bind one shared lifecycle transaction submitter before control ingress is attached.
  /// @param binding RunSession-owned state and nonblocking atomic-admission callback.
  /// @return Success, or a typed duplicate/invalid binding error without changing the old binding.
  [[nodiscard]] expected<void, Error> bind_lifecycle_submitter(
      ParameterLifecycleSubmitBinding binding) noexcept;

  /// @brief Submit one same-Operator all-or-nothing fixed-scalar patch batch.
  /// @param patches Owning caller values consumed synchronously and never retained.
  /// @param options Bounded audit identity and `NextCompute` request.
  /// @return Immediate atomic admission or rejection result.
  [[nodiscard]] ParamBatchUpdateResult submit(std::span<const ParamPatch> patches,
                                              ParamUpdateOptions options = {});

  /// @brief Try to submit fixed compute-originated patches without blocking or allocating.
  /// @param patches Contiguous callback-lifetime records copied before return.
  /// @param patch_count Number of records; zero rejects.
  /// @return Submitted/accepted outcome or immediate typed rejection.
  [[nodiscard]] ParamSubmitResult try_submit_from_compute(const ParamInlinePatch* patches,
                                                          std::size_t patch_count) noexcept;

  /// @brief Drain preallocated compute-originated ingress on a non-callback runtime path.
  /// @return Number of submitted requests removed from the compute ingress.
  std::size_t drain_compute_ingress() noexcept;

  /// @brief Read one supported scalar from one coherent Parameter view.
  /// @param address Exact run-bound field identity.
  /// @param view Published or latest-accepted view.
  /// @return Fixed scalar value and immutable generation lineage.
  [[nodiscard]] expected<ParamReadResult, Error> read(ParamAddress address, ParamReadView view);

  /// @brief Return permission-relevant static metadata for an exact current-run field address.
  /// @param address Exact run-bound field identity.
  /// @return Static defining-module descriptor, or null for foreign/unknown input.
  /// @note The caller remains responsible for applying capability permissions and origin policy.
  [[nodiscard]] const ParamFieldDescriptor* field_descriptor(ParamAddress address) const noexcept;

  /// @brief Return the verified immutable run catalog used for capability filtering.
  [[nodiscard]] const ParameterCatalogView& catalog() const noexcept;

  /// @brief Return the compile-admitted simultaneous cursor bound.
  [[nodiscard]] std::uint32_t max_event_cursors() const noexcept;

  /// @brief Reserve one bounded policy-projected event cursor.
  /// @param policy Immutable capability policy applied for the cursor lifetime.
  /// @return Nonzero private cursor identity, or a typed capacity/closed error.
  [[nodiscard]] expected<std::uint32_t, Error> register_event_cursor(
      const ParamAccessPolicy& policy) noexcept;

  /// @brief Release one bounded event-cursor reservation.
  /// @param cursor Nonzero private identity returned by \ref register_event_cursor.
  void unregister_event_cursor(std::uint32_t cursor) noexcept;

  /// @brief Capture policy-filtered owning Parameter state.
  [[nodiscard]] expected<ParamSnapshot, Error> capture(const ParamAccessPolicy& policy,
                                                       ParamCaptureOptions options);

  /// @brief Capture Parameter state with an already-prepared source catalog identity.
  /// @param policy Exact capability policy used for value projection.
  /// @param source_catalog_digest Digest of either the immutable read-enabled capture catalog or
  /// the empty discovery catalog prepared for a capture-only capability.
  /// @param options Requested consistency and unavailable-value behavior.
  /// @return Owning snapshot whose source identity names the exact read-enabled capture projection.
  [[nodiscard]] expected<ParamSnapshot, Error> capture(const ParamAccessPolicy& policy,
                                                       ParamCatalogDigest source_catalog_digest,
                                                       ParamCaptureOptions options);

  /// @brief Drain retained policy-visible events at one independent cursor position.
  /// @param cursor Nonzero private identity returned by \ref register_event_cursor.
  /// @param output Caller-owned storage populated with projected transitions or one opaque gap.
  /// @return Number of initialized records, or a typed stale/closed error.
  [[nodiscard]] expected<std::size_t, Error> drain_events(std::uint32_t cursor,
                                                          std::span<ParamEvent> output) noexcept;

  /// @brief Return the retained state of one accepted sequence without waiting.
  /// @param sequence Nonzero run-local accepted sequence.
  /// @return Retained bounded result, or `kNotFound` after eviction/foreign input.
  [[nodiscard]] expected<ParamUpdateResult, Error> result(ParamUpdateSequence sequence);

  /// @brief Wait for one accepted sequence to become terminal, detach, or reach a timeout.
  /// @param sequence Nonzero run-local accepted sequence.
  /// @param timeout Maximum relative wait; zero performs a nonblocking query.
  /// @return Terminal retained result, `kTimeout`, `kClosed`, or `kNotFound`.
  [[nodiscard]] expected<ParamUpdateResult, Error> await(ParamUpdateSequence sequence,
                                                         std::chrono::nanoseconds timeout);

  /// @brief Wait only when one capability can observe the accepted request's complete scope.
  /// @param sequence Nonzero run-local accepted sequence.
  /// @param timeout Maximum relative wait; zero performs a nonblocking query.
  /// @param policy Already validated run-scoped capability policy.
  /// @return Terminal result, or permission denied without disclosing an out-of-scope sequence.
  [[nodiscard]] expected<ParamUpdateResult, Error> await_authorized(
      ParamUpdateSequence sequence, std::chrono::nanoseconds timeout,
      const ParamAccessPolicy& policy);

  /// @brief Publish at most one oldest accepted candidate at one callback boundary.
  /// @param owner Exact current-run Operator boundary owner.
  /// @param accepted_cutoff Frozen accepted frontier for this boundary.
  /// @return Published sequence result, or empty when no eligible candidate is pending or the
  /// current published generation has not yet been observed by an activation.
  /// @note A successful publication cannot be superseded until one activation commits its
  /// observation. This keeps every published generation observable and releases its bounded
  /// observation evidence before the next generation advances.
  [[nodiscard]] std::optional<ParamUpdateResult> publish_next_at_callback_boundary(
      OperatorRef owner, ParamUpdateSequence accepted_cutoff);

  /// @brief Snapshot the accepted request frontier for one finite publication boundary.
  /// @return Highest sequence accepted before this load.
  [[nodiscard]] ParamUpdateSequence publication_frontier() const noexcept;

  /// @brief Pin the currently published immutable generation for one callback.
  /// @param owner Exact current-run Operator identity.
  /// @return One release-required stable pin, or a code-only identity/state error.
  [[nodiscard]] expected<ParameterGenerationPin, Error> pin_published(OperatorRef owner) noexcept;

  /// @brief Commit the first activation observation of one exact published pin.
  /// @param owner Exact current-run Operator identity used to obtain \p pin.
  /// @param pin Still-live pin returned by \ref pin_published.
  /// @return Success, including an already-observed or generation-zero pin, or a typed identity
  /// failure before user compute begins.
  ///
  /// This dispatcher-side operation consumes the accepted request's pre-reserved observation event
  /// exactly once. It must run immediately after a successful activation pin and before invoking
  /// user compute; \ref ExecutionContext::params remains lock-free and allocation-free.
  [[nodiscard]] expected<void, Error> commit_activation_observation(
      OperatorRef owner, const ParameterGenerationPin& pin) noexcept;

  /// @brief Pin one exact published or private lifecycle candidate generation.
  /// @param owner Exact current-run Operator identity.
  /// @param generation Exact generation selected by the lifecycle transaction.
  /// @return One release-required stable pin, or a code-only identity/state error.
  [[nodiscard]] expected<ParameterGenerationPin, Error> pin_lifecycle_generation(
      OperatorRef owner, ParamGeneration generation) noexcept;

  /// @brief Apply one lifecycle-kernel state transition to an exact retained candidate.
  /// @param owner Exact current-run candidate owner.
  /// @param generation Candidate generation supplied to the lifecycle kernel.
  /// @param transaction Nonzero owning transaction identity.
  /// @param state Quiescing, restarting, published, superseded, canceled, rolled-back, or
  /// failed-stopped state.
  /// @return Success, or a typed stale/invalid transition error with no publication.
  /// @note `kPublished` performs the sole effective-generation swap after successful `kStart`.
  /// Terminal non-publication states retain the old published generation and retire the candidate
  /// only after its callback pins have drained.
  [[nodiscard]] expected<void, Error> transition_lifecycle_candidate(
      OperatorRef owner, ParamGeneration generation, LifecycleTransactionId transaction,
      LifecycleTransactionState state) noexcept;

  /// @brief Release one callback pin and make its retired slot reclaimable.
  /// @param pin Pin returned by \ref pin_published; inert pins are ignored.
  void release(ParameterGenerationPin& pin) noexcept;

  /// @brief Irreversibly close update ingress and return the terminal accepted frontier.
  /// @return Idempotent close result; accepted candidates remain publishable/drainable.
  [[nodiscard]] ParamStopFenceResult close_ingress();

  /// @brief Request irreversible ingress closure without waiting for coordinator ownership.
  void request_close_ingress() noexcept;

  /// @brief Close ingress and cancel every candidate left unpublished when worker execution ends.
  /// @note Attached capabilities retain read/result/await access to the resulting terminal records.
  void terminalize_unpublished_at_run_end() noexcept;

  /// @brief Wake waiters and reject subsequent operations before session storage is destroyed.
  /// @note This operation first terminalizes unpublished work, then marks capabilities detached.
  void begin_detach() noexcept;

  /// @brief Return the last sequence accepted before the current instant.
  /// @return Zero before first acceptance, otherwise the monotonic run-local frontier.
  [[nodiscard]] ParamUpdateSequence terminal_sequence() const noexcept;

 private:
  ParameterUpdateCoordinator() = default;

  [[nodiscard]] ParameterGenerationStore* find_store(OperatorRef owner) noexcept;
  [[nodiscard]] const ParameterGenerationStore* find_store(OperatorRef owner) const noexcept;

  struct State;
  std::unique_ptr<State> state_;

  friend class ParameterUpdateCoordinatorTestAccess;
};

/// @brief Expose deterministic source-tree-only coordinator contention controls to runtime tests.
class ParameterUpdateCoordinatorTestAccess {
 public:
  /// @brief Hold the coordinator mutation lock until the caller releases it.
  static void hold_mutation_lock(ParameterUpdateCoordinator& coordinator, std::atomic_bool& entered,
                                 const std::atomic_bool& release) noexcept;

  /// @brief Exercise the exact fixed-scalar equality used by no-change admission.
  [[nodiscard]] static bool scalar_semantically_equal(ParamScalarValue left,
                                                      ParamScalarValue right) noexcept;

  /// @brief Return the exact number of source event identities owned by accepted requests.
  [[nodiscard]] static std::uint64_t reserved_event_sequences(
      ParameterUpdateCoordinator& coordinator) noexcept;

  /// @brief Set the next source event sequence for deterministic no-wrap boundary tests.
  static void set_event_sequence(ParameterUpdateCoordinator& coordinator,
                                 ParamEventSequence sequence) noexcept;

  /// @brief Set the accepted-update identity frontier for deterministic no-wrap tests.
  static void set_update_sequence(ParameterUpdateCoordinator& coordinator,
                                  ParamUpdateSequence sequence) noexcept;

  /// @brief Set one published generation identity for deterministic no-wrap tests.
  static void set_published_generation(ParameterUpdateCoordinator& coordinator, OperatorRef owner,
                                       ParamGeneration generation) noexcept;

  /// @brief Set the published slot's synthetic pin count for no-wrap boundary tests.
  static void set_published_pin_count(ParameterUpdateCoordinator& coordinator, OperatorRef owner,
                                      std::uint64_t pin_count) noexcept;

  /// @brief Exercise reservation replacement against the real private ledger.
  static void replace_event_reservation(ParameterUpdateCoordinator& coordinator,
                                        std::uint32_t released_count) noexcept;

  /// @brief Exercise reservation release against the real private ledger.
  static void release_event_reservations(ParameterUpdateCoordinator& coordinator,
                                         std::uint32_t count) noexcept;

  /// @brief Set one result's event reservation count for fatal-invariant tests.
  static void set_result_event_reservations(ParameterUpdateCoordinator& coordinator,
                                            ParamUpdateSequence sequence,
                                            std::uint32_t count) noexcept;

  /// @brief Remove one result's validity for fatal retained-evidence tests.
  static void invalidate_result(ParameterUpdateCoordinator& coordinator,
                                ParamUpdateSequence sequence) noexcept;

  /// @brief Return the published slot's complete claim word for no-wrap boundary tests.
  [[nodiscard]] static std::uint64_t published_claim_word(ParameterUpdateCoordinator& coordinator,
                                                          OperatorRef owner) noexcept;
};

}  // namespace holoscan::detail
