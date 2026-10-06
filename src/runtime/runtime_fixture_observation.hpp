/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <holoscan/core/run_session.hpp>

#include "port_binding.hpp"
#include "runtime_channel_binding.hpp"
#include "temporal_input_cache.hpp"

namespace holoscan::detail {

enum class FixtureObservationKind : std::uint8_t {
  kInputIngested,
  kCandidateFinished,
  kPublicationRejectedOutsideActivation,
};
enum class FixtureCandidateOutcome : std::uint8_t { kRetired, kSuppressed, kFailed };

struct FixtureObservedInput {
  PortId input{};
  InputSelection selection{};
  TemporalSampleRecord sample{};
  bool has_sample{};
};

struct FixtureObservedPublication {
  PortId output{};
  std::uint64_t sequence{};
  ErrorCode error{ErrorCode::kSuccess};
  bool sequence_consumed{};
};

struct FixtureObservationRecord {
  FixtureObservationKind kind{FixtureObservationKind::kCandidateFinished};
  VertexId owner{};
  RuntimeEvent witness{};
  Tag tag{};
  std::uint64_t activation_generation{};
  FixtureCandidateOutcome outcome{FixtureCandidateOutcome::kFailed};
  bool admitted{};
  std::span<const FixtureObservedInput> inputs;
  std::span<const FixtureObservedPublication> publications;
};

/// Bounded, attempt-owned evidence for the single-partition testing profile.
/// One logical Runtime writer owns all mutable state. The observed profile has one ordinary
/// partition: inline callbacks execute on that owner; ExecutionLaneSet::dispatch transfers the
/// callback to one lane and synchronously joins it before the owner touches scratch again. Compute
/// callbacks only fill activation scratch. A rejected emit during a synchronous lifecycle callback
/// may append an outside-activation record while the Runtime owner is waiting for that callback.
/// No callback may concurrently emit from an application-created thread. The release/acquire
/// prefix publishes immutable records to readers; it does not make scratch or append multi-writer.
/// Capacity exhaustion invalidates the attempt without changing an already committed prefix.
class RuntimeFixtureObservation final {
 public:
  RuntimeFixtureObservation(std::size_t capacity, std::size_t inputs_per_record,
                            std::size_t publication_capacity)
      : records_(capacity),
        inputs_(checked_size(capacity, inputs_per_record)),
        publications_(publication_capacity),
        input_stride_(inputs_per_record),
        activation_inputs_(inputs_per_record),
        activation_publications_(publication_capacity) {}

  // Called once, cold, before a Runtime thread can publish. Reusing observation storage across
  // attempts would let numerically identical port/sequence values certify a different generation.
  [[nodiscard]] bool bind(PlanGenerationId plan, RunGenerationId run) noexcept {
    if (bound_ || plan == PlanGenerationId{} || run == RunGenerationId{}) {
      invalidate();
      return false;
    }
    plan_ = plan;
    run_ = run;
    bound_ = true;
    return true;
  }
  [[nodiscard]] PlanGenerationId plan_generation() const noexcept { return plan_; }
  [[nodiscard]] RunGenerationId run_generation() const noexcept { return run_; }

  void begin_activation(FixtureObservationRecord record) noexcept {
    if (activation_active_) {
      invalidate();
      return;
    }
    activation_record_ = record;
    activation_input_count_ = 0U;
    activation_publication_count_ = 0U;
    activation_active_ = true;
  }
  void select_inputs(std::span<const RuntimeChannelActivationSelection> selections) noexcept {
    if (selections.size() > activation_inputs_.size()) {
      invalidate();
      return;
    }
    activation_input_count_ = selections.size();
    for (std::size_t i = 0; i < selections.size(); ++i) {
      const auto& selection = selections[i];
      if (selection.samples.size() > 1U || selection.batch) {
        invalidate();
        return;
      }
      activation_inputs_[i] = FixtureObservedInput{
          .input = selection.port,
          .selection = selection.selection,
          .sample = selection.samples.empty() ? TemporalSampleRecord{} : selection.samples.front(),
          .has_sample = !selection.samples.empty(),
      };
    }
  }
  // The currently dispatched lane exclusively owns scratch until synchronous dispatch returns.
  // Outside-activation rejection is a separate fact: it proves neither admission nor publication
  // nor retirement. Successful/authority-consuming activity outside an activation is unsupported.
  void publication(VertexId owner, std::uint64_t generation,
                   FixtureObservedPublication publication) noexcept {
    if (!activation_active_) {
      if (publication.error == ErrorCode::kSuccess || publication.sequence_consumed) {
        invalidate();
        return;
      }
      static_cast<void>(append(
          FixtureObservationRecord{
              .kind = FixtureObservationKind::kPublicationRejectedOutsideActivation,
              .owner = owner,
              .activation_generation = generation},
          {},
          std::span{&publication, std::size_t{1U}}));
      return;
    }
    if (owner != activation_record_.owner ||
        generation != activation_record_.activation_generation ||
        activation_publication_count_ == activation_publications_.size()) {
      invalidate();
      return;
    }
    activation_publications_[activation_publication_count_++] = publication;
  }
  void activation_generation(std::uint64_t generation) noexcept {
    activation_record_.activation_generation = generation;
  }
  void finish_activation(bool admitted, FixtureCandidateOutcome outcome) noexcept {
    activation_record_.admitted = admitted;
    activation_record_.outcome = outcome;
    static_cast<void>(
        append(activation_record_,
               std::span{activation_inputs_}.first(activation_input_count_),
               std::span{activation_publications_}.first(activation_publication_count_)));
    activation_active_ = false;
  }

  [[nodiscard]] bool append(FixtureObservationRecord record,
                            std::span<const FixtureObservedInput> inputs,
                            std::span<const FixtureObservedPublication> publications) noexcept {
    if (invalid() || written_ == records_.size() || inputs.size() > input_stride_ ||
        publications.size() > publications_.size() - publication_count_) {
      invalidate();
      return false;
    }
    auto input_slot = std::span{inputs_}.subspan(written_ * input_stride_, inputs.size());
    auto publication_slot =
        std::span{publications_}.subspan(publication_count_, publications.size());
    std::copy(inputs.begin(), inputs.end(), input_slot.begin());
    std::copy(publications.begin(), publications.end(), publication_slot.begin());
    record.inputs = input_slot;
    record.publications = publication_slot;
    records_[written_] = record;
    publication_count_ += publications.size();
    // There is one logical writer, including a synchronously dispatched lifecycle callback.
    // No reservation or mutable detail escapes before this commit.
    published_.store(++written_, std::memory_order_release);
    return true;
  }

  [[nodiscard]] std::span<const FixtureObservationRecord> records() const noexcept {
    return std::span{records_}.first(published_.load(std::memory_order_acquire));
  }
  [[nodiscard]] bool invalid() const noexcept { return invalid_.load(std::memory_order_acquire); }
  void invalidate() noexcept { invalid_.store(true, std::memory_order_release); }

 private:
  [[nodiscard]] static std::size_t checked_size(std::size_t count, std::size_t stride) {
    if (stride != 0U && count > std::numeric_limits<std::size_t>::max() / stride) {
      throw std::length_error{"operator test observation capacity overflow"};
    }
    return count * stride;
  }
  std::vector<FixtureObservationRecord> records_;
  std::vector<FixtureObservedInput> inputs_;
  std::vector<FixtureObservedPublication> publications_;
  std::size_t input_stride_{};
  std::size_t publication_count_{};
  std::size_t written_{};
  std::atomic<std::size_t> published_{};
  std::atomic<bool> invalid_{};
  PlanGenerationId plan_{};
  RunGenerationId run_{};
  bool bound_{};
  // Non-atomic scratch is safe only under the synchronous single-partition dispatch contract
  // above. More partitions or overlapping callbacks need a different ownership design, not just
  // atomic counters around these records and their span-backed payloads.
  FixtureObservationRecord activation_record_{};
  std::vector<FixtureObservedInput> activation_inputs_;
  std::vector<FixtureObservedPublication> activation_publications_;
  std::size_t activation_input_count_{};
  std::size_t activation_publication_count_{};
  bool activation_active_{};
};

template <bool Observed>
class FixtureActivationObservation;

/// Compile the entire observation scope out of ordinary dispatch, including record construction.
template <>
class FixtureActivationObservation<false> final {
 public:
  template <class RecordFactory>
  FixtureActivationObservation(RuntimeFixtureObservation*, const RecordFactory&) noexcept {}
  void select_inputs(std::span<const RuntimeChannelActivationSelection>) noexcept {}
  void generation(std::uint64_t) noexcept {}
  void admitted(bool) noexcept {}
  void finish(FixtureCandidateOutcome) noexcept {}
};

/// Keep every observed early activation exit fail-closed; only the retirement cut marks success.
template <>
class FixtureActivationObservation<true> final {
 public:
  template <class RecordFactory>
  FixtureActivationObservation(RuntimeFixtureObservation* observation,
                               RecordFactory&& make_record) noexcept
      : observation_(observation) {
    // Cold dispatch selection guarantees this precondition before any Runtime thread starts.
    assert(observation_ != nullptr);
    observation_->begin_activation(std::forward<RecordFactory>(make_record)());
  }
  FixtureActivationObservation(const FixtureActivationObservation&) = delete;
  FixtureActivationObservation& operator=(const FixtureActivationObservation&) = delete;
  ~FixtureActivationObservation() {
    if (observation_ != nullptr)
      observation_->finish_activation(admitted_, FixtureCandidateOutcome::kFailed);
  }
  void select_inputs(std::span<const RuntimeChannelActivationSelection> inputs) noexcept {
    if (observation_ != nullptr)
      observation_->select_inputs(inputs);
  }
  void generation(std::uint64_t generation) noexcept {
    if (observation_ != nullptr)
      observation_->activation_generation(generation);
  }
  void admitted(bool admitted) noexcept { admitted_ = admitted; }
  void finish(FixtureCandidateOutcome outcome) noexcept {
    if (observation_ != nullptr) {
      observation_->finish_activation(admitted_, outcome);
      observation_ = nullptr;
    }
  }

 private:
  RuntimeFixtureObservation* observation_{};
  bool admitted_{};
};

/// Start the real Runtime with one private observation owner; never install this bridge as API.
[[nodiscard]] HOLOSCAN_CORE_EXPORT RunSession start_operator_test_run(
    const ExecutionPlan& plan, const std::shared_ptr<RuntimeFixtureObservation>& observation);

}  // namespace holoscan::detail
