/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <holoscan/core/activation.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/time_point.hpp>

namespace holoscan::detail {

class RuntimeInputSelectionState;

// Every valid TimePoint accepted by this file must already be normalized into one validated
// immutable plan/clock-catalog scope. These payload-independent cache records intentionally carry
// no catalog identity and cannot detect a foreign ClockId that happens to collide numerically.
// Payload-independent candidate evidence lets HSDK5-005 prove the causal algorithms without
// acquiring Channel or Tensor ownership. Later bindings retain the actual Sample and lease under
// the exact same arrival/source identities.
struct TemporalSampleRecord {
  TimePoint capture_time{};
  std::uint64_t source_sequence{};
  std::uint64_t producer_epoch{};
  std::uint64_t arrival_sequence{};

  friend constexpr bool operator==(const TemporalSampleRecord& left,
                                   const TemporalSampleRecord& right) noexcept {
    return left.capture_time == right.capture_time &&
           left.source_sequence == right.source_sequence &&
           left.producer_epoch == right.producer_epoch &&
           left.arrival_sequence == right.arrival_sequence;
  }
};

/// @brief Order two records by producer generation, then by generation-local arrival order.
/// @param candidate Record being tested for supersession.
/// @param reference Record the candidate would supersede.
/// @return True when \p candidate is strictly newer than \p reference.
/// @note \c source_sequence is epoch-local: it restarts at the request's initial value on an
/// admitted rearm while the producer epoch is bumped. Comparing sequences alone therefore reports a
/// pre-restart record as newer than every post-restart one, permanently and silently. Comparing
/// records from two different subscribers is meaningless under any ordering, because their sequence
/// spaces are independent; callers must establish a single subscriber space first.
[[nodiscard]] constexpr bool is_strictly_newer_sample(
    const TemporalSampleRecord& candidate, const TemporalSampleRecord& reference) noexcept {
  return candidate.producer_epoch != reference.producer_epoch
             ? candidate.producer_epoch > reference.producer_epoch
             : candidate.source_sequence > reference.source_sequence;
}

struct ScalarSelectionResult {
  InputSelection selection{};
  std::optional<TemporalSampleRecord> sample;
};

enum class MissingAggregateDisposition : std::uint8_t {
  kNominal,
  kDegraded,
  kNoFire,
};

[[nodiscard]] inline MissingAggregateDisposition aggregate_missing_inputs(
    std::span<const InputSelection> selections) noexcept {
  bool has_safe_default = false;
  bool has_absent = false;
  // Evaluate the complete immutable-plan ordered set before applying precedence. This intentionally
  // avoids a short-circuit whose diagnostic set would depend on which missing input appeared first.
  for (const auto selection : selections) {
    has_safe_default = has_safe_default || selection.state == InputState::kSafeDefault;
    has_absent = has_absent || selection.state == InputState::kAbsent;
  }
  if (has_safe_default) {
    return MissingAggregateDisposition::kDegraded;
  }
  if (has_absent) {
    return MissingAggregateDisposition::kNoFire;
  }
  return MissingAggregateDisposition::kNominal;
}

class TemporalInputCache {
 public:
  class ScalarTransaction {
   public:
    ScalarTransaction() = default;
    ScalarTransaction(const ScalarTransaction&) = delete;
    ScalarTransaction& operator=(const ScalarTransaction&) = delete;
    ScalarTransaction(ScalarTransaction&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)),
          result_(other.result_),
          missing_evidence_(other.missing_evidence_) {}
    ScalarTransaction& operator=(ScalarTransaction&&) = delete;
    ~ScalarTransaction() {
      if (owner_ != nullptr) {
        owner_->scalar_active_ = false;
      }
    }

    [[nodiscard]] const ScalarSelectionResult& result() const noexcept { return result_; }

    [[nodiscard]] expected<void, Error> commit() noexcept {
      if (owner_ == nullptr || !owner_->scalar_active_) {
        return make_unexpected(Error{ErrorCode::kStaleEndpoint});
      }
      if (result_.sample) {
        const auto found = std::find_if(
            owner_->samples_.begin(), owner_->samples_.end(), [&](const auto& candidate) {
              return candidate.arrival_sequence == result_.sample->arrival_sequence;
            });
        if (found == owner_->samples_.end()) {
          return make_unexpected(Error{ErrorCode::kFailure});
        }
        owner_->samples_.erase(found);
      }
      if (missing_evidence_) {
        const auto found = std::find_if(
            owner_->samples_.begin(), owner_->samples_.end(), [&](const auto& candidate) {
              return candidate.arrival_sequence == missing_evidence_->arrival_sequence;
            });
        if (found == owner_->samples_.end()) {
          return make_unexpected(Error{ErrorCode::kFailure});
        }
        owner_->samples_.erase(found);
      }
      owner_->scalar_active_ = false;
      owner_ = nullptr;
      return {};
    }

   private:
    ScalarTransaction(TemporalInputCache* owner, ScalarSelectionResult result,
                      std::optional<TemporalSampleRecord> missing_evidence = std::nullopt) noexcept
        : owner_(owner), result_(result), missing_evidence_(missing_evidence) {}

    TemporalInputCache* owner_{};
    ScalarSelectionResult result_{};
    std::optional<TemporalSampleRecord> missing_evidence_;

    friend class TemporalInputCache;
  };

  class DrainTransaction {
   public:
    DrainTransaction() = default;
    DrainTransaction(const DrainTransaction&) = delete;
    DrainTransaction& operator=(const DrainTransaction&) = delete;
    DrainTransaction(DrainTransaction&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), committed_(other.committed_) {}
    DrainTransaction& operator=(DrainTransaction&&) = delete;
    ~DrainTransaction() {
      if (owner_ != nullptr && !committed_) {
        owner_->drain_active_ = false;
        owner_->drain_snapshot_.clear();
      }
    }

    [[nodiscard]] std::span<const TemporalSampleRecord> samples() const noexcept {
      return owner_ != nullptr ? std::span<const TemporalSampleRecord>{owner_->drain_snapshot_}
                               : std::span<const TemporalSampleRecord>{};
    }

    [[nodiscard]] expected<void, Error> commit() noexcept {
      if (owner_ == nullptr || committed_) {
        return make_unexpected(Error{ErrorCode::kStaleEndpoint});
      }
      // Strictly increasing arrival identity makes the frozen set a queue prefix. Arrivals after
      // the watermark only append, so validate that prefix before retiring it with one range erase.
      const std::size_t frozen_count = owner_->drain_snapshot_.size();
      if (owner_->samples_.size() < frozen_count) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      for (std::size_t index = 0; index < frozen_count; ++index) {
        if (owner_->samples_[index].arrival_sequence !=
            owner_->drain_snapshot_[index].arrival_sequence) {
          return make_unexpected(Error{ErrorCode::kFailure});
        }
      }
      owner_->samples_.erase(owner_->samples_.begin(),
                             owner_->samples_.begin() + static_cast<std::ptrdiff_t>(frozen_count));
      committed_ = true;
      owner_->drain_active_ = false;
      owner_->drain_snapshot_.clear();
      return {};
    }

   private:
    explicit DrainTransaction(TemporalInputCache* owner) : owner_(owner) {}

    TemporalInputCache* owner_{};
    bool committed_{};

    friend class TemporalInputCache;
  };

  explicit TemporalInputCache(std::size_t capacity, std::size_t active_capacity = 0U)
      : capacity_(capacity),
        active_capacity_(active_capacity == 0U ? capacity : std::min(active_capacity, capacity)) {
    samples_.reserve(capacity);
    drain_snapshot_.reserve(capacity);
  }

  [[nodiscard]] std::size_t size() const noexcept { return samples_.size(); }
  /// Return the immutable physical reservation used for payload-claim scratch.
  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
  /// Return the published logical admission limit for the current rate generation.
  [[nodiscard]] std::size_t active_capacity() const noexcept { return active_capacity_; }

  /// Validate one lifecycle-selected logical limit without publishing it.
  [[nodiscard]] expected<void, Error> validate_active_capacity(
      std::size_t capacity) const noexcept {
    if (scalar_active_ || drain_active_) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
    if (capacity == 0U || capacity > capacity_ || samples_.size() > capacity) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    return {};
  }

  /// Publish one lifecycle-prevalidated logical limit without changing physical storage.
  [[nodiscard]] expected<void, Error> set_active_capacity(std::size_t capacity) noexcept {
    auto valid = validate_active_capacity(capacity);
    if (!valid) {
      return valid;
    }
    commit_validated_active_capacity(capacity);
    return {};
  }

  [[nodiscard]] expected<void, Error> push(TemporalSampleRecord sample) noexcept {
    if (sample.arrival_sequence == 0U || sample.source_sequence == 0U ||
        sample.producer_epoch == 0U) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (samples_.size() >= active_capacity_) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    if (sample.arrival_sequence <= last_arrival_sequence_) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    samples_.push_back(sample);
    last_arrival_sequence_ = sample.arrival_sequence;
    return {};
  }

  /// @brief Remove one exactly identified retained record without disturbing arrival identity.
  ///
  /// This is the caller-directed counterpart to \ref push, which deliberately fails closed on a
  /// full cache rather than discarding causal evidence. Eviction never happens implicitly: the
  /// caller names the record, so this cache never has to know an edge's \c QueuePolicy or whether
  /// a payload is still borrowed. A \c kDropOldest edge whose retained candidate has been
  /// superseded uses this to release the stale record before ingesting the newer one; retiring
  /// that record's payload claim remains the caller's responsibility.
  ///
  /// \c last_arrival_sequence_ is intentionally left untouched. It is the monotonic admission
  /// watermark, not a property of the retained set, and rewinding it would let an already-admitted
  /// arrival identity be pushed a second time.
  ///
  /// @param arrival_sequence Arrival identity of the record to remove.
  /// @return Empty success, \c kNotFound when no retained record carries that identity, or
  ///         \c kBackpressured while a scalar or drain transaction is reading the retained set.
  [[nodiscard]] expected<void, Error> evict(std::uint64_t arrival_sequence) noexcept {
    if (arrival_sequence == 0U) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    // An open transaction holds the frozen selection for the running activation; erasing under it
    // would invalidate the record the caller is about to commit.
    if (scalar_active_ || drain_active_) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
    const auto found =
        std::find_if(samples_.begin(), samples_.end(), [&](const TemporalSampleRecord& candidate) {
          return candidate.arrival_sequence == arrival_sequence;
        });
    if (found == samples_.end()) {
      return make_unexpected(Error{ErrorCode::kNotFound});
    }
    samples_.erase(found);
    return {};
  }

  [[nodiscard]] ScalarSelectionResult select_exact(TimePoint reference) const noexcept {
    if (!reference.valid()) {
      return {{InputState::kAbsent, SelectionReason::kCaptureTimeAbsent}, std::nullopt};
    }
    const TemporalSampleRecord* best = nullptr;
    bool observed_invalid = false;
    bool observed_other_clock = false;
    for (const auto& sample : samples_) {
      if (!sample.capture_time.valid()) {
        observed_invalid = true;
        continue;
      }
      if (sample.capture_time.clock_id != reference.clock_id) {
        observed_other_clock = true;
        continue;
      }
      if (sample.capture_time.timestamp_ns == reference.timestamp_ns &&
          (best == nullptr || sample.source_sequence > best->source_sequence)) {
        best = &sample;
      }
    }
    if (best != nullptr) {
      return {{InputState::kSelected, SelectionReason::kNone}, *best};
    }
    if (observed_other_clock) {
      return {{InputState::kAbsent, SelectionReason::kClockIncomparable}, std::nullopt};
    }
    if (observed_invalid) {
      return {{InputState::kAbsent, SelectionReason::kCaptureTimeAbsent}, std::nullopt};
    }
    return {{InputState::kAbsent, SelectionReason::kNeverSeen}, std::nullopt};
  }

  [[nodiscard]] expected<ScalarTransaction, Error> select_exact_transaction(
      TimePoint reference,
      std::uint64_t arrival_watermark = std::numeric_limits<std::uint64_t>::max()) noexcept {
    if (scalar_active_ || drain_active_) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
    scalar_active_ = true;
    const ScalarSelectionResult result = select_exact(reference);
    return ScalarTransaction{
        this, result, capture_absent_missing_evidence(result, arrival_watermark)};
  }

  [[nodiscard]] ScalarSelectionResult select_latest_within(
      TimePoint reference, std::chrono::nanoseconds max_age) const noexcept {
    if (!reference.valid()) {
      return {{InputState::kAbsent, SelectionReason::kCaptureTimeAbsent}, std::nullopt};
    }
    if (max_age.count() <= 0) {
      return {{InputState::kAbsent, SelectionReason::kTooStale}, std::nullopt};
    }
    const TemporalSampleRecord* newest_causal = nullptr;
    bool observed_invalid = false;
    bool observed_other_clock = false;
    for (const auto& sample : samples_) {
      if (!sample.capture_time.valid()) {
        observed_invalid = true;
        continue;
      }
      if (sample.capture_time.clock_id != reference.clock_id) {
        observed_other_clock = true;
        continue;
      }
      if (sample.capture_time.timestamp_ns > reference.timestamp_ns) {
        continue;
      }
      if (newest_causal == nullptr ||
          sample.capture_time.timestamp_ns > newest_causal->capture_time.timestamp_ns ||
          (sample.capture_time.timestamp_ns == newest_causal->capture_time.timestamp_ns &&
           sample.source_sequence > newest_causal->source_sequence)) {
        newest_causal = &sample;
      }
    }
    if (newest_causal == nullptr) {
      if (observed_other_clock) {
        return {{InputState::kAbsent, SelectionReason::kClockIncomparable}, std::nullopt};
      }
      if (observed_invalid) {
        return {{InputState::kAbsent, SelectionReason::kCaptureTimeAbsent}, std::nullopt};
      }
      return {{InputState::kAbsent, SelectionReason::kNeverSeen}, std::nullopt};
    }
    const auto age = static_cast<std::uint64_t>(reference.timestamp_ns) -
                     static_cast<std::uint64_t>(newest_causal->capture_time.timestamp_ns);
    if (age > static_cast<std::uint64_t>(max_age.count())) {
      return {{InputState::kAbsent, SelectionReason::kTooStale}, std::nullopt};
    }
    return {{InputState::kSelected, SelectionReason::kNone}, *newest_causal};
  }

  [[nodiscard]] expected<ScalarTransaction, Error> select_latest_within_transaction(
      TimePoint reference, std::chrono::nanoseconds max_age,
      std::uint64_t arrival_watermark = std::numeric_limits<std::uint64_t>::max(),
      std::optional<TimePoint> future_reference_lower_bound = std::nullopt) noexcept {
    if (scalar_active_ || drain_active_) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
    scalar_active_ = true;
    const ScalarSelectionResult result = select_latest_within(reference, max_age);
    const std::optional<TemporalSampleRecord> retirement =
        result.selection.reason == SelectionReason::kTooStale
            ? stale_missing_evidence(
                  result, reference, future_reference_lower_bound, max_age, arrival_watermark)
            : capture_absent_missing_evidence(result, arrival_watermark);
    return ScalarTransaction{this, result, retirement};
  }

  [[nodiscard]] expected<DrainTransaction, Error> snapshot_drain_all(
      std::uint64_t arrival_watermark, std::size_t selected_capacity) noexcept {
    if (drain_active_ || scalar_active_) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
    if (selected_capacity > active_capacity_) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    drain_snapshot_.clear();
    for (const auto sample : samples_) {
      if (sample.arrival_sequence <= arrival_watermark) {
        if (drain_snapshot_.size() == selected_capacity) {
          drain_snapshot_.clear();
          return make_unexpected(Error{ErrorCode::kResourceExhausted});
        }
        drain_snapshot_.push_back(sample);
      }
    }
    drain_active_ = true;
    return DrainTransaction{this};
  }

  [[nodiscard]] std::span<const TemporalSampleRecord> records() const noexcept { return samples_; }

  /// Clear all retained metadata at one accepted runtime generation boundary.
  [[nodiscard]] expected<void, Error> reset() noexcept {
    if (scalar_active_ || drain_active_) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
    samples_.clear();
    drain_snapshot_.clear();
    last_arrival_sequence_ = 0U;
    return {};
  }

  /// Release payload-retaining history after the current input generation is terminal.
  ///
  /// Unlike \ref reset, this preserves the last accepted arrival sequence. A terminal drain can
  /// therefore release typed Channel claims without making a late or duplicate record admissible
  /// as the first record of a new generation.
  [[nodiscard]] expected<void, Error> release_terminal_history() noexcept {
    if (scalar_active_ || drain_active_) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
    samples_.clear();
    drain_snapshot_.clear();
    return {};
  }

 private:
  /// Publish a limit already accepted by \ref validate_active_capacity.
  void commit_validated_active_capacity(std::size_t capacity) noexcept {
    active_capacity_ = capacity;
  }

  [[nodiscard]] std::optional<TemporalSampleRecord> capture_absent_missing_evidence(
      const ScalarSelectionResult& result, std::uint64_t arrival_watermark) const noexcept {
    if (result.selection.state == InputState::kSelected ||
        result.selection.reason != SelectionReason::kCaptureTimeAbsent) {
      return std::nullopt;
    }
    const TemporalSampleRecord* newest = nullptr;
    for (const auto& sample : samples_) {
      if (!sample.capture_time.valid() && sample.arrival_sequence <= arrival_watermark &&
          (newest == nullptr || sample.arrival_sequence > newest->arrival_sequence)) {
        newest = &sample;
      }
    }
    return newest != nullptr ? std::optional<TemporalSampleRecord>{*newest} : std::nullopt;
  }

  [[nodiscard]] std::optional<TemporalSampleRecord> stale_missing_evidence(
      const ScalarSelectionResult& result, TimePoint reference,
      std::optional<TimePoint> future_reference_lower_bound, std::chrono::nanoseconds max_age,
      std::uint64_t arrival_watermark) const noexcept {
    if (result.selection.state == InputState::kSelected ||
        result.selection.reason != SelectionReason::kTooStale || !future_reference_lower_bound ||
        !reference.valid() || !future_reference_lower_bound->valid() ||
        future_reference_lower_bound->clock_id != reference.clock_id || max_age.count() <= 0) {
      return std::nullopt;
    }
    // Arrival order is the cache's queue order. Retire one oldest record only when the caller has
    // proved a same-clock lower bound for every future semantic reference. The current reference
    // alone is insufficient because independently accepted input witnesses can become visible in
    // decreasing semantic-reference order.
    for (const auto& sample : samples_) {
      if (sample.arrival_sequence > arrival_watermark || !sample.capture_time.valid() ||
          sample.capture_time.clock_id != future_reference_lower_bound->clock_id ||
          sample.capture_time.timestamp_ns > future_reference_lower_bound->timestamp_ns) {
        continue;
      }
      const auto age = static_cast<std::uint64_t>(future_reference_lower_bound->timestamp_ns) -
                       static_cast<std::uint64_t>(sample.capture_time.timestamp_ns);
      if (age > static_cast<std::uint64_t>(max_age.count())) {
        return sample;
      }
    }
    return std::nullopt;
  }

  std::size_t capacity_{};
  std::size_t active_capacity_{};
  std::vector<TemporalSampleRecord> samples_;
  std::vector<TemporalSampleRecord> drain_snapshot_;
  bool drain_active_{};
  bool scalar_active_{};
  std::uint64_t last_arrival_sequence_{};

  friend class RuntimeInputSelectionState;
};

[[nodiscard]] inline ScalarSelectionResult apply_scalar_missing_policy(
    ScalarSelectionResult selected, std::variant<Skip, SafeDefault, ReuseLast> policy,
    std::optional<TemporalSampleRecord> retained, std::uint64_t current_producer_epoch,
    TimePoint reference) noexcept {
  if (selected.selection.state == InputState::kSelected) {
    return selected;
  }
  if (std::holds_alternative<SafeDefault>(policy)) {
    selected.selection.state = InputState::kSafeDefault;
    return selected;
  }
  if (const auto* reuse = std::get_if<ReuseLast>(&policy);
      reuse != nullptr && retained && retained->capture_time.valid() && reference.valid() &&
      current_producer_epoch != 0U && retained->producer_epoch == current_producer_epoch &&
      retained->capture_time.clock_id == reference.clock_id &&
      retained->capture_time.timestamp_ns <= reference.timestamp_ns && reuse->max_age.count() > 0) {
    const auto age = static_cast<std::uint64_t>(reference.timestamp_ns) -
                     static_cast<std::uint64_t>(retained->capture_time.timestamp_ns);
    if (age <= static_cast<std::uint64_t>(reuse->max_age.count())) {
      selected.selection.state = InputState::kReused;
      selected.sample = *retained;
      return selected;
    }
  }
  selected.selection.state = InputState::kAbsent;
  return selected;
}

enum class AlignmentClosureProof : std::uint8_t {
  kOpen,
  kOrderedAdvance,
  kRetentionLoss,
  kWatermark,
  kEndOfStream,
  kProducerEpochClosed,
};

enum class AlignedMissingPolicyKind : std::uint8_t {
  kUnspecified,
  kSkip,
  kSafeDefault,
  kReuseLast,
};

struct AlignedMissingPolicy {
  AlignedMissingPolicyKind kind{AlignedMissingPolicyKind::kUnspecified};
  std::chrono::nanoseconds reuse_max_age{};
};

enum class AlignedCandidateDisposition : std::uint8_t {
  kPending,
  kNominal,
  kDegraded,
  kNoFire,
  kRetireMisaligned,
};

enum class AlignedMemberEpochTransitionReason : std::uint8_t {
  kProducerEpoch,
};

class AlignedJoinEngine {
 public:
  class Transaction {
   public:
    Transaction() = default;
    Transaction(const Transaction&) = delete;
    Transaction& operator=(const Transaction&) = delete;
    Transaction(Transaction&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)),
          disposition_(other.disposition_),
          reference_time_(other.reference_time_),
          reason_(other.reason_) {}
    Transaction& operator=(Transaction&&) = delete;
    ~Transaction() {
      if (owner_ != nullptr) {
        owner_->active_ = false;
      }
    }

    [[nodiscard]] AlignedCandidateDisposition disposition() const noexcept { return disposition_; }
    [[nodiscard]] TimePoint reference_time() const noexcept { return reference_time_; }
    [[nodiscard]] SelectionReason reason() const noexcept { return reason_; }
    [[nodiscard]] std::span<const TemporalSampleRecord> members() const noexcept {
      return owner_ != nullptr ? std::span<const TemporalSampleRecord>{owner_->selected_}
                               : std::span<const TemporalSampleRecord>{};
    }
    [[nodiscard]] std::span<const InputSelection> selections() const noexcept {
      return owner_ != nullptr ? std::span<const InputSelection>{owner_->selections_}
                               : std::span<const InputSelection>{};
    }
    [[nodiscard]] std::span<const AlignmentClosureProof> closure_evidence() const noexcept {
      return owner_ != nullptr ? std::span<const AlignmentClosureProof>{owner_->closure_evidence_}
                               : std::span<const AlignmentClosureProof>{};
    }

    [[nodiscard]] expected<void, Error> commit() noexcept {
      if (owner_ == nullptr || !owner_->active_) {
        return make_unexpected(Error{ErrorCode::kStaleEndpoint});
      }
      for (std::size_t index = 0; index < owner_->queues_.size(); ++index) {
        if (owner_->consume_[index] != 0U) {
          if (disposition_ == AlignedCandidateDisposition::kNominal ||
              disposition_ == AlignedCandidateDisposition::kDegraded) {
            owner_->retained_[index] = owner_->queues_[index].front();
          }
          owner_->queues_[index].erase(owner_->queues_[index].begin());
        }
      }
      owner_->active_ = false;
      owner_ = nullptr;
      return {};
    }

   private:
    Transaction(AlignedJoinEngine* owner, AlignedCandidateDisposition disposition,
                TimePoint reference_time, SelectionReason reason) noexcept
        : owner_(owner),
          disposition_(disposition),
          reference_time_(reference_time),
          reason_(reason) {}

    AlignedJoinEngine* owner_{};
    AlignedCandidateDisposition disposition_{AlignedCandidateDisposition::kPending};
    TimePoint reference_time_{};
    SelectionReason reason_{SelectionReason::kNone};

    friend class AlignedJoinEngine;
  };

  AlignedJoinEngine(std::size_t member_count, std::size_t per_member_capacity)
      : per_member_capacity_(per_member_capacity),
        queues_(member_count),
        consume_(member_count),
        selections_(member_count),
        retained_(member_count),
        closure_evidence_(member_count),
        last_observed_(member_count),
        member_epochs_(member_count) {
    for (auto& queue : queues_) {
      queue.reserve(per_member_capacity);
    }
    selected_.reserve(member_count);
  }

  [[nodiscard]] expected<void, Error> push(std::size_t member,
                                           TemporalSampleRecord sample) noexcept {
    if (member >= queues_.size() || !sample.capture_time.valid() || sample.source_sequence == 0U ||
        sample.producer_epoch == 0U || sample.arrival_sequence == 0U) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    auto& queue = queues_[member];
    // `reserve()` provisions storage but does not define the logical contract bound: standard
    // libraries may reserve more than requested. Admission must remain identical across them.
    if (queue.size() >= per_member_capacity_) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    if (member_epochs_[member] != 0U && member_epochs_[member] != sample.producer_epoch) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const auto& previous = last_observed_[member];
    if (previous && (previous->capture_time.clock_id != sample.capture_time.clock_id ||
                     sample.capture_time.timestamp_ns < previous->capture_time.timestamp_ns ||
                     sample.source_sequence <= previous->source_sequence ||
                     sample.arrival_sequence <= previous->arrival_sequence)) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    queue.push_back(sample);
    member_epochs_[member] = sample.producer_epoch;
    last_observed_[member] = sample;
    return {};
  }

  [[nodiscard]] expected<void, Error> transition_member_epoch(
      std::size_t member, std::uint64_t new_epoch,
      AlignedMemberEpochTransitionReason reason) noexcept {
    if (active_) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
    if (member >= queues_.size() || new_epoch == 0U ||
        reason != AlignedMemberEpochTransitionReason::kProducerEpoch ||
        member_epochs_[member] == new_epoch) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }

    // Epoch transition is an explicit ownership boundary. No queue head, retained fallback, or
    // ordering witness from the closed producer generation may cross it.
    queues_[member].clear();
    retained_[member].reset();
    last_observed_[member].reset();
    member_epochs_[member] = new_epoch;
    consume_[member] = 0U;
    selections_[member] = {InputState::kAbsent, SelectionReason::kNeverSeen};
    closure_evidence_[member] = AlignmentClosureProof::kOpen;
    return {};
  }

  [[nodiscard]] expected<Transaction, Error> select(std::chrono::nanoseconds window,
                                                    std::span<const AlignedMissingPolicy> missing,
                                                    std::span<const AlignmentClosureProof> closure,
                                                    TimePoint closure_time = {}) noexcept {
    if (active_ || queues_.empty() || window.count() < 0 || missing.size() != queues_.size() ||
        closure.size() != queues_.size()) {
      return make_unexpected(
          Error{active_ ? ErrorCode::kBackpressured : ErrorCode::kInvalidArgument});
    }
    for (std::size_t index = 0; index < queues_.size(); ++index) {
      if (!valid_closure(closure[index]) || !valid_missing_policy(missing[index])) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
    }
    selected_.clear();
    std::fill(consume_.begin(), consume_.end(), 0U);
    std::copy(closure.begin(), closure.end(), closure_evidence_.begin());
    std::fill(selections_.begin(),
              selections_.end(),
              InputSelection{InputState::kAbsent, SelectionReason::kNeverSeen});

    const bool complete = std::all_of(
        queues_.begin(), queues_.end(), [](const auto& queue) { return !queue.empty(); });
    if (complete) {
      return select_complete(window);
    }
    return select_proof_closed_partial(window, missing, closure, closure_time);
  }

  [[nodiscard]] std::size_t queue_size(std::size_t member) const noexcept {
    return member < queues_.size() ? queues_[member].size() : 0U;
  }

  /// Return the retained FIFO records for one member.
  ///
  /// Runtime uses this bounded view to keep typed Channel storage alive while
  /// payload-independent temporal metadata can still select the record.
  [[nodiscard]] std::span<const TemporalSampleRecord> queue_records(
      std::size_t member) const noexcept {
    return member < queues_.size() ? std::span<const TemporalSampleRecord>{queues_[member]}
                                   : std::span<const TemporalSampleRecord>{};
  }

  /// Return the last consumed record retained for finite ReuseLast selection.
  [[nodiscard]] std::optional<TemporalSampleRecord> retained_record(
      std::size_t member) const noexcept {
    return member < retained_.size() ? retained_[member] : std::nullopt;
  }

  /// Clear every queue, retained fallback, closure fact, and ordering witness.
  [[nodiscard]] expected<void, Error> reset() noexcept {
    if (active_) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
    for (auto& queue : queues_) {
      queue.clear();
    }
    std::fill(consume_.begin(), consume_.end(), 0U);
    std::fill(selections_.begin(),
              selections_.end(),
              InputSelection{InputState::kAbsent, SelectionReason::kNeverSeen});
    for (auto& retained : retained_) {
      retained.reset();
    }
    std::fill(closure_evidence_.begin(), closure_evidence_.end(), AlignmentClosureProof::kOpen);
    for (auto& observed : last_observed_) {
      observed.reset();
    }
    std::fill(member_epochs_.begin(), member_epochs_.end(), 0U);
    selected_.clear();
    return {};
  }

  /// Release payload-retaining queues and reuse state after every member generation is terminal.
  ///
  /// Member epochs and last-observed ordering witnesses remain intact so this terminal operation
  /// cannot masquerade as a new generation boundary.
  [[nodiscard]] expected<void, Error> release_terminal_history() noexcept {
    if (active_) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
    for (auto& queue : queues_) {
      queue.clear();
    }
    std::fill(consume_.begin(), consume_.end(), 0U);
    std::fill(selections_.begin(),
              selections_.end(),
              InputSelection{InputState::kAbsent, SelectionReason::kNeverSeen});
    for (auto& retained : retained_) {
      retained.reset();
    }
    std::fill(closure_evidence_.begin(), closure_evidence_.end(), AlignmentClosureProof::kOpen);
    selected_.clear();
    return {};
  }

 private:
  [[nodiscard]] static constexpr bool valid_closure(AlignmentClosureProof value) noexcept {
    switch (value) {
      case AlignmentClosureProof::kOpen:
      case AlignmentClosureProof::kOrderedAdvance:
      case AlignmentClosureProof::kRetentionLoss:
      case AlignmentClosureProof::kWatermark:
      case AlignmentClosureProof::kEndOfStream:
      case AlignmentClosureProof::kProducerEpochClosed:
        return true;
    }
    return false;
  }

  [[nodiscard]] static constexpr bool valid_missing_policy(
      const AlignedMissingPolicy& value) noexcept {
    switch (value.kind) {
      case AlignedMissingPolicyKind::kUnspecified:
      case AlignedMissingPolicyKind::kSkip:
      case AlignedMissingPolicyKind::kSafeDefault:
        return value.reuse_max_age.count() == 0;
      case AlignedMissingPolicyKind::kReuseLast:
        return value.reuse_max_age.count() > 0;
    }
    return false;
  }

  [[nodiscard]] static expected<std::uint64_t, Error> checked_spread(std::int64_t low,
                                                                     std::int64_t high) noexcept {
    if (high < low) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    const auto spread = static_cast<std::uint64_t>(high) - static_cast<std::uint64_t>(low);
    if (spread > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    return spread;
  }

  [[nodiscard]] expected<Transaction, Error> select_complete(
      std::chrono::nanoseconds window) noexcept {
    const ClockId clock = queues_.front().front().capture_time.clock_id;
    std::int64_t low = queues_.front().front().capture_time.timestamp_ns;
    std::int64_t high = low;
    for (const auto& queue : queues_) {
      const auto& head = queue.front();
      if (head.capture_time.clock_id != clock) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      low = std::min(low, head.capture_time.timestamp_ns);
      high = std::max(high, head.capture_time.timestamp_ns);
    }
    auto spread = checked_spread(low, high);
    if (!spread) {
      return make_unexpected(std::move(spread.error()));
    }
    const bool matched = *spread <= static_cast<std::uint64_t>(window.count());
    for (std::size_t index = 0; index < queues_.size(); ++index) {
      const auto& head = queues_[index].front();
      if (matched || head.capture_time.timestamp_ns == low) {
        selected_.push_back(head);
        consume_[index] = 1U;
      }
      selections_[index] =
          matched ? InputSelection{InputState::kSelected, SelectionReason::kNone}
                  : InputSelection{InputState::kAbsent, SelectionReason::kAlignmentWindowMiss};
    }
    active_ = true;
    return Transaction{this,
                       matched ? AlignedCandidateDisposition::kNominal
                               : AlignedCandidateDisposition::kRetireMisaligned,
                       matched ? TimePoint{high, clock} : TimePoint{},
                       matched ? SelectionReason::kNone : SelectionReason::kAlignmentWindowMiss};
  }

  // Closure selection validates and stages one all-or-nothing aligned transaction. Keeping the
  // proof and provisional state updates together makes rollback obligations visible.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<Transaction, Error> select_proof_closed_partial(
      std::chrono::nanoseconds window, std::span<const AlignedMissingPolicy> missing,
      std::span<const AlignmentClosureProof> closure, TimePoint closure_time) noexcept {
    for (std::size_t index = 0; index < queues_.size(); ++index) {
      if (queues_[index].empty() && closure[index] == AlignmentClosureProof::kOpen) {
        return Transaction{
            nullptr, AlignedCandidateDisposition::kPending, {}, SelectionReason::kNone};
      }
    }
    if (!closure_time.valid()) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    bool has_safe_default = false;
    for (std::size_t index = 0; index < queues_.size(); ++index) {
      if (!queues_[index].empty()) {
        selected_.push_back(queues_[index].front());
        selections_[index] = {InputState::kSelected, SelectionReason::kNone};
        consume_[index] = 1U;
        continue;
      }
      const SelectionReason closure_reason = [&]() noexcept {
        switch (closure[index]) {
          case AlignmentClosureProof::kOrderedAdvance:
            return SelectionReason::kAlignmentWindowMiss;
          case AlignmentClosureProof::kEndOfStream:
          case AlignmentClosureProof::kProducerEpochClosed:
            return SelectionReason::kEndOfStream;
          case AlignmentClosureProof::kRetentionLoss:
          case AlignmentClosureProof::kWatermark:
          case AlignmentClosureProof::kOpen:
            return SelectionReason::kInsufficientHistory;
        }
        return SelectionReason::kInsufficientHistory;
      }();
      if (missing[index].kind == AlignedMissingPolicyKind::kSafeDefault) {
        selections_[index] = {InputState::kSafeDefault, closure_reason};
        has_safe_default = true;
      } else if (missing[index].kind == AlignedMissingPolicyKind::kReuseLast &&
                 closure[index] != AlignmentClosureProof::kProducerEpochClosed) {
        const auto& retained = retained_[index];
        if (missing[index].reuse_max_age.count() <= 0 || !retained ||
            retained->capture_time.clock_id != closure_time.clock_id ||
            retained->capture_time.timestamp_ns > closure_time.timestamp_ns) {
          selections_[index] = {InputState::kAbsent, closure_reason};
          continue;
        }
        const auto age = static_cast<std::uint64_t>(closure_time.timestamp_ns) -
                         static_cast<std::uint64_t>(retained->capture_time.timestamp_ns);
        if (age > static_cast<std::uint64_t>(missing[index].reuse_max_age.count())) {
          selections_[index] = {InputState::kAbsent, SelectionReason::kTooStale};
          continue;
        }
        selected_.push_back(*retained);
        selections_[index] = {InputState::kReused, closure_reason};
      } else {
        selections_[index] = {InputState::kAbsent, closure_reason};
      }
    }
    const bool has_absent = std::any_of(selections_.begin(), selections_.end(), [](auto value) {
      return value.state == InputState::kAbsent;
    });
    if (!has_safe_default && has_absent) {
      std::fill(consume_.begin(), consume_.end(), 0U);
      selected_.clear();
    }
    if (!selected_.empty()) {
      std::int64_t low = closure_time.timestamp_ns;
      std::int64_t high = closure_time.timestamp_ns;
      for (const auto& member : selected_) {
        if (member.capture_time.clock_id != closure_time.clock_id) {
          std::fill(consume_.begin(), consume_.end(), 0U);
          selected_.clear();
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        low = std::min(low, member.capture_time.timestamp_ns);
        high = std::max(high, member.capture_time.timestamp_ns);
      }
      auto spread = checked_spread(low, high);
      if (!spread) {
        std::fill(consume_.begin(), consume_.end(), 0U);
        selected_.clear();
        return make_unexpected(std::move(spread.error()));
      }
      if (*spread > static_cast<std::uint64_t>(window.count())) {
        // Preserve FIFO-head progress only when the oldest incompatible evidence is an owned
        // queue head. A retained fallback is never consumed or silently promoted into a tuple.
        bool can_retire = false;
        std::fill(consume_.begin(), consume_.end(), 0U);
        for (std::size_t index = 0; index < queues_.size(); ++index) {
          if (!queues_[index].empty() && queues_[index].front().capture_time.timestamp_ns == low) {
            consume_[index] = 1U;
            can_retire = true;
          }
        }
        selected_.clear();
        active_ = true;
        return Transaction{this,
                           can_retire ? AlignedCandidateDisposition::kRetireMisaligned
                                      : AlignedCandidateDisposition::kNoFire,
                           {},
                           SelectionReason::kAlignmentWindowMiss};
      }
    }
    active_ = true;
    return Transaction{this,
                       has_safe_default ? AlignedCandidateDisposition::kDegraded
                       : has_absent     ? AlignedCandidateDisposition::kNoFire
                                        : AlignedCandidateDisposition::kNominal,
                       closure_time,
                       SelectionReason::kInsufficientHistory};
  }

  std::size_t per_member_capacity_{};
  std::vector<std::vector<TemporalSampleRecord>> queues_;
  std::vector<std::uint8_t> consume_;
  std::vector<InputSelection> selections_;
  std::vector<std::optional<TemporalSampleRecord>> retained_;
  std::vector<AlignmentClosureProof> closure_evidence_;
  std::vector<std::optional<TemporalSampleRecord>> last_observed_;
  std::vector<std::uint64_t> member_epochs_;
  std::vector<TemporalSampleRecord> selected_;
  bool active_{};
};

enum class WindowAnchorDisposition : std::uint8_t {
  kPrimed,
  kPending,
  kClosed,
  kResetAndReprimed,
};

enum class WindowInputDisposition : std::uint8_t {
  kAccepted,
  kResetAndAccepted,
};

enum class WindowResetReason : std::uint8_t {
  kNone,
  kProducerEpoch,
  kClockDomain,
  kTimestampRegression,
};

struct WindowAnchorResult {
  WindowAnchorDisposition disposition{WindowAnchorDisposition::kPending};
  WindowResetReason reset_reason{WindowResetReason::kNone};
};

struct WindowInputResult {
  WindowInputDisposition disposition{WindowInputDisposition::kAccepted};
  WindowResetReason reset_reason{WindowResetReason::kNone};
};

struct BracketedSelectionResult {
  InputSelection selection{};
  std::vector<TemporalSampleRecord> samples;
  TimePoint reference_time{};
};

class BracketedWindowEngine {
 public:
  BracketedWindowEngine(BracketedWindow selector, std::size_t cache_capacity,
                        std::size_t pending_capacity, std::size_t active_cache_capacity = 0U)
      : selector_(selector),
        cache_capacity_(cache_capacity),
        active_cache_capacity_(active_cache_capacity == 0U ? cache_capacity
                                                           : active_cache_capacity),
        pending_capacity_(pending_capacity),
        valid_(selector.lookbehind > 0U && selector.lookahead > 0U && selector.max_samples > 0U &&
               cache_capacity > 0U && active_cache_capacity_ <= cache_capacity &&
               pending_capacity > 0U) {
    cache_.reserve(cache_capacity);
    pending_.reserve(pending_capacity);
    before_.reserve(cache_capacity);
    middle_.reserve(cache_capacity);
    after_.reserve(cache_capacity);
    // One extra result slot records the terminal attempt that encounters a full pending set.
    // The steady-state footprint remains constructor-bounded and no hot-path growth is permitted.
    if (pending_capacity != std::numeric_limits<std::size_t>::max()) {
      closed_slots_.resize(pending_capacity + 1U);
    } else {
      valid_ = false;
    }
    for (auto& slot : closed_slots_) {
      slot.samples.reserve(selector_.max_samples);
    }
  }

  [[nodiscard]] expected<WindowAnchorResult, Error> push_anchor(
      TemporalSampleRecord anchor) noexcept {
    if (!valid_ || !anchor.capture_time.valid() || anchor.source_sequence == 0U ||
        anchor.producer_epoch == 0U || anchor.arrival_sequence == 0U) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    const WindowResetReason reset_reason = anchor_reset_reason(anchor);
    if (reset_reason != WindowResetReason::kNone) {
      reset_state();
      normalized_clock_ = anchor.capture_time.clock_id;
      previous_anchor_ = anchor;
      return WindowAnchorResult{WindowAnchorDisposition::kResetAndReprimed, reset_reason};
    }
    if (previous_anchor_ && (anchor.source_sequence <= previous_anchor_->source_sequence ||
                             anchor.arrival_sequence <= previous_anchor_->arrival_sequence)) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (!previous_anchor_) {
      normalized_clock_ = anchor.capture_time.clock_id;
      previous_anchor_ = anchor;
      return WindowAnchorResult{WindowAnchorDisposition::kPrimed, WindowResetReason::kNone};
    }
    if (pending_.size() == pending_capacity_) {
      if (!enqueue({InputState::kAbsent, SelectionReason::kWindowPendingLimit},
                   anchor.capture_time,
                   {})) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      previous_anchor_ = anchor;
      prune_cache();
      return WindowAnchorResult{WindowAnchorDisposition::kClosed, WindowResetReason::kNone};
    }
    const std::size_t closed_before = closed_count_;
    pending_.push_back(Pending{*previous_anchor_, anchor});
    previous_anchor_ = anchor;
    close_ready();
    return WindowAnchorResult{closed_count_ == closed_before ? WindowAnchorDisposition::kPending
                                                             : WindowAnchorDisposition::kClosed,
                              WindowResetReason::kNone};
  }

  [[nodiscard]] expected<WindowInputResult, Error> push_sample(
      TemporalSampleRecord sample) noexcept {
    if (!valid_ || !sample.capture_time.valid() || sample.source_sequence == 0U ||
        sample.producer_epoch == 0U || sample.arrival_sequence == 0U) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    const WindowResetReason reset_reason = sample_reset_reason(sample);
    if (reset_reason != WindowResetReason::kNone) {
      reset_state();
      normalized_clock_ = sample.capture_time.clock_id;
    } else if (!normalized_clock_) {
      normalized_clock_ = sample.capture_time.clock_id;
    }
    if (last_sample_ && sample.source_sequence <= last_sample_->source_sequence) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (last_sample_ && sample.arrival_sequence <= last_sample_->arrival_sequence) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (std::any_of(cache_.begin(), cache_.end(), [&](const auto& existing) {
          return existing.arrival_sequence == sample.arrival_sequence;
        })) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (cache_.size() >= active_cache_capacity_) {
      prune_cache();
      if (cache_.size() >= active_cache_capacity_) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
    }
    cache_.push_back(sample);
    last_sample_ = sample;
    close_ready();
    return WindowInputResult{reset_reason == WindowResetReason::kNone
                                 ? WindowInputDisposition::kAccepted
                                 : WindowInputDisposition::kResetAndAccepted,
                             reset_reason};
  }

  [[nodiscard]] expected<void, Error> close_oldest(SelectionReason reason) noexcept {
    if (pending_.empty() ||
        (reason != SelectionReason::kWindowTimeout && reason != SelectionReason::kEndOfStream)) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (!enqueue({InputState::kAbsent, reason}, pending_.front().current.capture_time, {})) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    pending_.erase(pending_.begin());
    prune_cache();
    return {};
  }

  /// Latch one producer EOS and close every window already pending for that generation.
  ///
  /// Closed decisions use the constructor-bounded result ring. If earlier decisions occupy that
  /// ring, the latch closes the remaining pending windows incrementally as callers retire results;
  /// any ordinarily ready windows publish first, and no second transport EOS or timeout is
  /// required.
  void close_end_of_stream() noexcept {
    end_of_stream_latched_ = true;
    close_ready();
    close_end_of_stream_pending();
  }

  [[nodiscard]] const BracketedSelectionResult* front_closed() const noexcept {
    return closed_count_ == 0U ? nullptr : &closed_slots_[closed_head_];
  }

  void pop_closed() noexcept {
    if (closed_count_ == 0U) {
      return;
    }
    closed_slots_[closed_head_].samples.clear();
    closed_head_ = (closed_head_ + 1U) % closed_slots_.size();
    --closed_count_;
    close_ready();
    if (end_of_stream_latched_) {
      close_end_of_stream_pending();
    }
    prune_cache();
  }

  [[nodiscard]] std::size_t pending_count() const noexcept { return pending_.size(); }

  /// Return the immutable physical retained-entry reservation.
  [[nodiscard]] std::size_t cache_capacity() const noexcept { return cache_capacity_; }

  /// Return the published logical retained-entry limit.
  [[nodiscard]] std::size_t active_cache_capacity() const noexcept {
    return active_cache_capacity_;
  }

  /// Validate one lifecycle-selected retained-entry limit without publishing it.
  [[nodiscard]] expected<void, Error> validate_active_cache_capacity(
      std::size_t capacity) const noexcept {
    if (capacity == 0U || capacity > cache_capacity_ || cache_.size() > capacity ||
        closed_count_ != 0U || !pending_.empty()) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    return {};
  }

  /// Publish one lifecycle-prevalidated logical limit without changing backing vectors.
  [[nodiscard]] expected<void, Error> set_active_cache_capacity(std::size_t capacity) noexcept {
    auto valid = validate_active_cache_capacity(capacity);
    if (!valid) {
      return valid;
    }
    commit_validated_active_cache_capacity(capacity);
    return {};
  }

  [[nodiscard]] std::optional<TimePoint> oldest_pending_reference() const noexcept {
    return pending_.empty() ? std::nullopt
                            : std::optional<TimePoint>{pending_.front().current.capture_time};
  }

  /// Return the maximum number of payload claims copied by
  /// \ref copy_payload_claims.
  [[nodiscard]] std::size_t payload_claim_capacity() const noexcept {
    if (selector_.max_samples != 0U &&
        closed_slots_.size() >
            (std::numeric_limits<std::size_t>::max() - cache_capacity_) / selector_.max_samples) {
      return 0U;
    }
    const std::size_t closed_capacity = closed_slots_.size() * selector_.max_samples;
    return cache_capacity_ + closed_capacity;
  }

  /// Copy every record whose typed payload can still appear in a future window.
  ///
  /// The output can contain duplicates because overlapping closed windows and
  /// the live cache may reference the same record. The Runtime caller sorts and
  /// deduplicates the preallocated buffer before reconciling Channel claims.
  [[nodiscard]] expected<std::size_t, Error> copy_payload_claims(
      std::span<TemporalSampleRecord> output) const noexcept {
    std::size_t count = 0U;
    const auto append = [&](const TemporalSampleRecord& record) {
      if (count == output.size()) {
        return false;
      }
      output[count++] = record;
      return true;
    };
    for (const TemporalSampleRecord& record : cache_) {
      if (!append(record)) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
    }
    for (std::size_t offset = 0U; offset < closed_count_; ++offset) {
      const auto& closed = closed_slots_[(closed_head_ + offset) % closed_slots_.size()];
      for (const TemporalSampleRecord& record : closed.samples) {
        if (!append(record)) {
          return make_unexpected(Error{ErrorCode::kResourceExhausted});
        }
      }
    }
    return count;
  }

  /// Discard every pending, closed, and retained record at a generation boundary.
  void reset() noexcept {
    reset_state();
    reset_closed();
  }

 private:
  /// Publish a limit already accepted by \ref validate_active_cache_capacity.
  void commit_validated_active_cache_capacity(std::size_t capacity) noexcept {
    active_cache_capacity_ = capacity;
  }

  struct Pending {
    TemporalSampleRecord previous;
    TemporalSampleRecord current;
  };

  [[nodiscard]] WindowResetReason anchor_reset_reason(
      const TemporalSampleRecord& anchor) const noexcept {
    if (normalized_clock_ && *normalized_clock_ != anchor.capture_time.clock_id) {
      return WindowResetReason::kClockDomain;
    }
    if (!previous_anchor_) {
      return WindowResetReason::kNone;
    }
    if (previous_anchor_->producer_epoch != anchor.producer_epoch) {
      return WindowResetReason::kProducerEpoch;
    }
    if (anchor.capture_time.timestamp_ns < previous_anchor_->capture_time.timestamp_ns) {
      return WindowResetReason::kTimestampRegression;
    }
    return WindowResetReason::kNone;
  }

  [[nodiscard]] WindowResetReason sample_reset_reason(
      const TemporalSampleRecord& sample) const noexcept {
    if (normalized_clock_ && *normalized_clock_ != sample.capture_time.clock_id) {
      return WindowResetReason::kClockDomain;
    }
    if (!last_sample_) {
      return WindowResetReason::kNone;
    }
    if (last_sample_->producer_epoch != sample.producer_epoch) {
      return WindowResetReason::kProducerEpoch;
    }
    if (sample.capture_time.timestamp_ns < last_sample_->capture_time.timestamp_ns) {
      return WindowResetReason::kTimestampRegression;
    }
    return WindowResetReason::kNone;
  }

  void close_end_of_stream_pending() noexcept {
    std::size_t closed_pending_count = 0U;
    while (closed_pending_count < pending_.size() && closed_count_ < closed_slots_.size()) {
      const Pending& window = pending_[closed_pending_count];
      if (!enqueue({InputState::kAbsent, SelectionReason::kEndOfStream},
                   window.current.capture_time,
                   {})) {
        break;
      }
      ++closed_pending_count;
    }
    if (closed_pending_count != 0U) {
      pending_.erase(pending_.begin(),
                     pending_.begin() + static_cast<std::ptrdiff_t>(closed_pending_count));
      prune_cache();
    }
    release_terminal_history_if_closed();
  }

  void release_terminal_history_if_closed() noexcept {
    if (!end_of_stream_latched_ || !pending_.empty()) {
      return;
    }
    // Closed result slots own copies of every sample they can still expose. Once no pending
    // window can observe live history, discard the selector cache and anchor scratch so Channel
    // payload claims can retire even if bounded closed decisions await ordinary consumption.
    cache_.clear();
    before_.clear();
    middle_.clear();
    after_.clear();
    previous_anchor_.reset();
    last_sample_.reset();
    normalized_clock_.reset();
  }

  void reset_state() noexcept {
    pending_.clear();
    cache_.clear();
    previous_anchor_.reset();
    last_sample_.reset();
    normalized_clock_.reset();
    end_of_stream_latched_ = false;
  }

  void close_ready() {
    std::size_t index = 0;
    while (index < pending_.size() && closed_count_ < closed_slots_.size()) {
      const Pending& window = pending_[index];
      before_.clear();
      middle_.clear();
      after_.clear();
      for (const auto sample : cache_) {
        if (sample.capture_time.timestamp_ns < window.previous.capture_time.timestamp_ns) {
          before_.push_back(sample);
        } else if (sample.capture_time.timestamp_ns <= window.current.capture_time.timestamp_ns) {
          middle_.push_back(sample);
        } else if (after_.size() < selector_.lookahead) {
          after_.push_back(sample);
        }
      }
      if (after_.size() < selector_.lookahead) {
        ++index;
        continue;
      }
      if (before_.size() < selector_.lookbehind) {
        // The loop's free-slot predicate makes this failure unreachable. Keep the pending window
        // intact if that invariant is ever broken so a result-ring fault cannot silently consume
        // causal state.
        if (!enqueue({InputState::kAbsent, SelectionReason::kInsufficientHistory},
                     window.current.capture_time,
                     {})) {
          return;
        }
        pending_.erase(pending_.begin() + static_cast<std::ptrdiff_t>(index));
        continue;
      }
      const auto selected_size = selector_.lookbehind + middle_.size() + selector_.lookahead;
      if (selected_size > selector_.max_samples) {
        if (!enqueue({InputState::kAbsent, SelectionReason::kWindowSampleLimit},
                     window.current.capture_time,
                     {})) {
          return;
        }
      } else {
        const auto first_before = before_.end() - static_cast<std::ptrdiff_t>(selector_.lookbehind);
        auto& slot = closed_slots_[(closed_head_ + closed_count_) % closed_slots_.size()];
        slot.samples.clear();
        slot.samples.insert(slot.samples.end(), first_before, before_.end());
        slot.samples.insert(slot.samples.end(), middle_.begin(), middle_.end());
        slot.samples.insert(slot.samples.end(), after_.begin(), after_.end());
        slot.selection = {InputState::kSelected, SelectionReason::kNone};
        slot.reference_time = window.current.capture_time;
        ++closed_count_;
      }
      pending_.erase(pending_.begin() + static_cast<std::ptrdiff_t>(index));
    }
  }

  [[nodiscard]] bool enqueue(InputSelection selection, TimePoint reference_time,
                             std::span<const TemporalSampleRecord> samples) {
    if (closed_count_ == closed_slots_.size()) {
      return false;
    }
    auto& slot = closed_slots_[(closed_head_ + closed_count_) % closed_slots_.size()];
    slot.samples.clear();
    slot.samples.insert(slot.samples.end(), samples.begin(), samples.end());
    slot.selection = selection;
    slot.reference_time = reference_time;
    ++closed_count_;
    return true;
  }

  void reset_closed() noexcept {
    for (auto& slot : closed_slots_) {
      slot.samples.clear();
    }
    closed_head_ = 0U;
    closed_count_ = 0U;
  }

  void prune_cache() {
    if (!previous_anchor_ || cache_.empty()) {
      return;
    }
    const std::int64_t earliest_boundary =
        pending_.empty() ? previous_anchor_->capture_time.timestamp_ns
                         : pending_.front().previous.capture_time.timestamp_ns;
    const auto boundary =
        std::lower_bound(cache_.begin(),
                         cache_.end(),
                         earliest_boundary,
                         [](const TemporalSampleRecord& sample, std::int64_t time) {
                           return sample.capture_time.timestamp_ns < time;
                         });
    const auto predecessor_count = static_cast<std::size_t>(boundary - cache_.begin());
    const auto retain_before = std::min<std::size_t>(selector_.lookbehind, predecessor_count);
    cache_.erase(cache_.begin(), boundary - static_cast<std::ptrdiff_t>(retain_before));
  }

  BracketedWindow selector_;
  std::size_t cache_capacity_{};
  std::size_t active_cache_capacity_{};
  std::size_t pending_capacity_{};
  std::vector<TemporalSampleRecord> cache_;
  std::vector<Pending> pending_;
  std::vector<TemporalSampleRecord> before_;
  std::vector<TemporalSampleRecord> middle_;
  std::vector<TemporalSampleRecord> after_;
  std::vector<BracketedSelectionResult> closed_slots_;
  std::size_t closed_head_{};
  std::size_t closed_count_{};
  std::optional<TemporalSampleRecord> previous_anchor_;
  std::optional<TemporalSampleRecord> last_sample_;
  std::optional<ClockId> normalized_clock_;
  bool end_of_stream_latched_{};
  bool valid_{};

  friend class RuntimeInputSelectionState;
};

struct ClockTickCandidate {
  std::uint64_t index{};
  TimePoint scheduled_time{};
  std::uint64_t missed_first{};
  std::uint64_t missed_last{};
  std::uint64_t missed_count{};
};

enum class DeadlineGateDisposition : std::uint8_t {
  kInvokeNominal,
  kInvokeViolation,
  kFailRun,
};

struct DeadlineGateResult {
  DeadlineGateDisposition disposition{DeadlineGateDisposition::kInvokeNominal};
  ActivationStatus status{};
  TimePoint latest_start{};
  TimePoint dispatch_start{};
  std::int64_t checked_lateness_ns{};
};

[[nodiscard]] inline expected<DeadlineGateResult, Error> evaluate_deadline_gate(
    TimePoint eligible_time, std::chrono::nanoseconds budget, TimePoint dispatch_time,
    OnViolation policy) noexcept {
  if (!eligible_time.valid() || !dispatch_time.valid() ||
      eligible_time.clock_id != dispatch_time.clock_id || budget.count() <= 0 ||
      (policy != OnViolation::kSafeDefault && policy != OnViolation::kEmitWitness &&
       policy != OnViolation::kFailRun) ||
      eligible_time.timestamp_ns > std::numeric_limits<std::int64_t>::max() - budget.count()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const TimePoint latest{eligible_time.timestamp_ns + budget.count(), eligible_time.clock_id};
  if (dispatch_time.timestamp_ns <= latest.timestamp_ns) {
    return DeadlineGateResult{
        .disposition = DeadlineGateDisposition::kInvokeNominal,
        .status = {ActivationReason::kNominal, policy},
        .latest_start = latest,
        .dispatch_start = dispatch_time,
    };
  }
  if (latest.timestamp_ns < 0 &&
      dispatch_time.timestamp_ns > std::numeric_limits<std::int64_t>::max() + latest.timestamp_ns) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  return DeadlineGateResult{
      .disposition = policy == OnViolation::kFailRun ? DeadlineGateDisposition::kFailRun
                                                     : DeadlineGateDisposition::kInvokeViolation,
      .status = {ActivationReason::kDeadlineExceeded, policy},
      .latest_start = latest,
      .dispatch_start = dispatch_time,
      .checked_lateness_ns = dispatch_time.timestamp_ns - latest.timestamp_ns,
  };
}

class CoalescingClockSchedule {
 public:
  CoalescingClockSchedule(TimePoint phase, std::chrono::nanoseconds period) noexcept
      : phase_(phase), period_(period) {}

  [[nodiscard]] expected<std::optional<ClockTickCandidate>, Error> due(TimePoint now) noexcept {
    if (!accepts(now)) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (last_observed_now_ && now.timestamp_ns < last_observed_now_->timestamp_ns) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    // Recorded before the candidate is computed, so a schedule that reports nothing due, or that
    // overflows its coordinate, still advances the monotonic observation bound exactly as before.
    last_observed_now_ = now;
    auto candidate = candidate_at(now);
    if (!candidate) {
      return make_unexpected(std::move(candidate).error());
    }
    const auto& due_candidate = *candidate;
    if (!due_candidate.has_value()) {
      return std::optional<ClockTickCandidate>{};
    }
    handled_ = due_candidate->index;
    return candidate;
  }

  /// @brief Report the coordinate of the tick that would fire at \p now without consuming it.
  ///
  /// This is the reference time the witness produced by the next drain will carry. Callers that
  /// must reason about a witness before it exists need that coordinate, and must not take the tick
  /// while doing so: \ref due marks the tick handled and advances the monotonic observation bound,
  /// so calling it speculatively would drop a real firing.
  ///
  /// @param now Current reading of this schedule's clock.
  /// @return The scheduled coordinate, \c std::nullopt when no tick is due, or
  ///         \ref ErrorCode::kInvalidArgument for a foreign or invalid clock reading.
  /// @note This function does not throw and leaves the schedule unchanged. It deliberately omits
  /// \ref due's monotonicity check, because observing a coordinate is not an observation of time.
  [[nodiscard]] expected<std::optional<TimePoint>, Error> peek_scheduled(
      TimePoint now) const noexcept {
    if (!accepts(now)) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    auto candidate = candidate_at(now);
    if (!candidate) {
      return make_unexpected(std::move(candidate).error());
    }
    const auto& due_candidate = *candidate;
    if (!due_candidate.has_value()) {
      return std::optional<TimePoint>{};
    }
    return std::optional<TimePoint>{due_candidate->scheduled_time};
  }

 private:
  /// @brief Return whether \p now is a usable reading of this schedule's clock.
  [[nodiscard]] bool accepts(const TimePoint now) const noexcept {
    return phase_.valid() && now.valid() && phase_.clock_id == now.clock_id && period_.count() > 0;
  }

  /// @brief Compute the tick due at \p now without recording that it was observed or taken.
  /// @pre \ref accepts returned true for \p now.
  [[nodiscard]] expected<std::optional<ClockTickCandidate>, Error> candidate_at(
      const TimePoint now) const noexcept {
    if (now.timestamp_ns < phase_.timestamp_ns) {
      return std::optional<ClockTickCandidate>{};
    }
    const auto elapsed = static_cast<std::uint64_t>(now.timestamp_ns) -
                         static_cast<std::uint64_t>(phase_.timestamp_ns);
    const auto index = elapsed / static_cast<std::uint64_t>(period_.count());
    if (handled_ && index <= *handled_) {
      return std::optional<ClockTickCandidate>{};
    }
    // Quotient multiplication cannot overflow: index * period is at most elapsed. Add that
    // unsigned offset to a signed phase by magnitude so the full uint64 tick-index range remains
    // usable when a negative phase spans the tagged-time domain.
    const std::uint64_t offset = index * static_cast<std::uint64_t>(period_.count());
    std::int64_t scheduled_timestamp{};
    if (phase_.timestamp_ns >= 0) {
      const auto phase = static_cast<std::uint64_t>(phase_.timestamp_ns);
      if (offset > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) - phase) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      scheduled_timestamp = static_cast<std::int64_t>(phase + offset);
    } else {
      const std::uint64_t phase_magnitude =
          static_cast<std::uint64_t>(-(phase_.timestamp_ns + 1)) + 1U;
      if (offset < phase_magnitude) {
        const std::uint64_t remaining = phase_magnitude - offset;
        scheduled_timestamp = remaining == (std::uint64_t{1} << 63U)
                                  ? std::numeric_limits<std::int64_t>::min()
                                  : -static_cast<std::int64_t>(remaining);
      } else {
        const std::uint64_t positive = offset - phase_magnitude;
        if (positive > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
          return make_unexpected(Error{ErrorCode::kResourceExhausted});
        }
        scheduled_timestamp = static_cast<std::int64_t>(positive);
      }
    }
    const std::uint64_t first = handled_ ? *handled_ + 1U : 0U;
    ClockTickCandidate candidate{
        .index = index,
        .scheduled_time = TimePoint{scheduled_timestamp, phase_.clock_id},
    };
    if (index > first) {
      candidate.missed_first = first;
      candidate.missed_last = index - 1U;
      candidate.missed_count = index - first;
    }
    return std::optional<ClockTickCandidate>{candidate};
  }

  TimePoint phase_{};
  std::chrono::nanoseconds period_{};
  std::optional<std::uint64_t> handled_;
  std::optional<TimePoint> last_observed_now_;
};

}  // namespace holoscan::detail
