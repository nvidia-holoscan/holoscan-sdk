/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/run.hpp>
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "core/domain/tensor_access.hpp"
#include "runtime/run_session_internal.hpp"
#include "runtime/runtime_cuda_execution_resources.hpp"
#include "runtime/runtime_diagnostic_sink.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_support_thread_control.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime_inline_execution_test_support.hpp"

namespace holoscan {

// REQ: RUNTIME-040
// REQ: RUNTIME-041

namespace {

using namespace std::chrono_literals;

using namespace test::inline_execution;

[[nodiscard]] std::int32_t current_cpu() noexcept {
#if defined(__linux__)
  return sched_getcpu();
#else
  return -1;
#endif
}

inline constexpr std::size_t kPartitionRunnerProbeCount = 4U;

struct PartitionRunnerCallbackRecord {
  std::size_t partition_index{};
  std::uint32_t operator_id{};
  std::thread::id thread;
  std::int32_t cpu{-1};
};

struct PartitionRunnerProbeSnapshot {
  std::array<std::thread::id, kPartitionRunnerProbeCount> owner_threads;
  std::array<bool, kPartitionRunnerProbeCount> owner_seen{};
  std::array<std::uint32_t, kPartitionRunnerProbeCount> max_partition_concurrency{};
  std::uint32_t max_cross_partition_concurrency{};
  bool stable_owner_threads{true};
  bool balanced_callbacks{true};
  bool barrier_timed_out{};
  std::vector<PartitionRunnerCallbackRecord> callback_records;
  std::vector<std::uint32_t> sink_values;
};

class PartitionRunnerProbeState {
 public:
  static void reset() {
    const std::lock_guard lock(mutex_);
    owner_threads_ = {};
    owner_seen_ = {};
    active_callbacks_ = {};
    max_partition_concurrency_ = {};
    active_cross_partition_callbacks_ = 0U;
    max_cross_partition_concurrency_ = 0U;
    waiting_sources_ = 0U;
    release_sources_ = false;
    stable_owner_threads_ = true;
    balanced_callbacks_ = true;
    barrier_timed_out_ = false;
    callback_records_.clear();
    sink_values_.clear();
  }

  static void enter_source(std::size_t partition_index, std::uint32_t operator_id) {
    enter_barrier(partition_index, operator_id, kPartitionRunnerProbeCount);
  }

  static void enter_barrier(std::size_t partition_index, std::uint32_t operator_id,
                            std::size_t participant_count) {
    std::unique_lock lock(mutex_);
    enter_callback_locked(partition_index, operator_id);
    ++waiting_sources_;
    if (waiting_sources_ == participant_count) {
      release_sources_ = true;
      cv_.notify_all();
      return;
    }
    if (!cv_.wait_for(lock, 1s, [] { return release_sources_; })) {
      barrier_timed_out_ = true;
    }
  }

  static void enter_callback(std::size_t partition_index, std::uint32_t operator_id) {
    const std::lock_guard lock(mutex_);
    enter_callback_locked(partition_index, operator_id);
    cv_.notify_all();
  }

  static void enter_sink(std::size_t partition_index, std::uint32_t operator_id) {
    enter_callback(partition_index, operator_id);
  }

  static void leave_callback(std::size_t partition_index) {
    const std::lock_guard lock(mutex_);
    if (partition_index >= active_callbacks_.size() || active_callbacks_[partition_index] == 0U ||
        active_cross_partition_callbacks_ == 0U) {
      balanced_callbacks_ = false;
      return;
    }
    --active_callbacks_[partition_index];
    --active_cross_partition_callbacks_;
  }

  static void record_sink(std::uint32_t value) {
    {
      const std::lock_guard lock(mutex_);
      sink_values_.push_back(value);
    }
    cv_.notify_all();
  }

  [[nodiscard]] static bool wait_for_sinks(std::size_t count) {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, 5s, [count] { return sink_values_.size() >= count; });
  }

  [[nodiscard]] static bool wait_for_callbacks(std::size_t count) {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, 5s, [count] { return callback_records_.size() >= count; });
  }

  [[nodiscard]] static bool wait_for_each_operator(std::span<const std::uint32_t> ids,
                                                   std::size_t minimum_count) {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, 5s, [&] {
      return std::ranges::all_of(ids, [&](std::uint32_t id) {
        return static_cast<std::size_t>(std::ranges::count(
                   callback_records_, id, &PartitionRunnerCallbackRecord::operator_id)) >=
               minimum_count;
      });
    });
  }

  [[nodiscard]] static bool wait_for_owners(std::size_t count) {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, 5s, [count] {
      return static_cast<std::size_t>(std::ranges::count(owner_seen_, true)) >= count;
    });
  }

  [[nodiscard]] static PartitionRunnerProbeSnapshot snapshot() {
    const std::lock_guard lock(mutex_);
    return PartitionRunnerProbeSnapshot{
        .owner_threads = owner_threads_,
        .owner_seen = owner_seen_,
        .max_partition_concurrency = max_partition_concurrency_,
        .max_cross_partition_concurrency = max_cross_partition_concurrency_,
        .stable_owner_threads = stable_owner_threads_,
        .balanced_callbacks = balanced_callbacks_,
        .barrier_timed_out = barrier_timed_out_,
        .callback_records = callback_records_,
        .sink_values = sink_values_,
    };
  }

 private:
  static void enter_callback_locked(std::size_t partition_index, std::uint32_t operator_id) {
    if (partition_index >= active_callbacks_.size()) {
      balanced_callbacks_ = false;
      return;
    }
    const std::thread::id thread = std::this_thread::get_id();
    if (!owner_seen_[partition_index]) {
      owner_threads_[partition_index] = thread;
      owner_seen_[partition_index] = true;
    } else if (owner_threads_[partition_index] != thread) {
      stable_owner_threads_ = false;
    }
    ++active_callbacks_[partition_index];
    max_partition_concurrency_[partition_index] =
        std::max(max_partition_concurrency_[partition_index], active_callbacks_[partition_index]);
    ++active_cross_partition_callbacks_;
    max_cross_partition_concurrency_ =
        std::max(max_cross_partition_concurrency_, active_cross_partition_callbacks_);
    callback_records_.push_back(PartitionRunnerCallbackRecord{
        .partition_index = partition_index,
        .operator_id = operator_id,
        .thread = thread,
        .cpu = current_cpu(),
    });
  }

  static inline std::mutex mutex_;
  static inline std::condition_variable cv_;
  static inline std::array<std::thread::id, kPartitionRunnerProbeCount> owner_threads_;
  static inline std::array<bool, kPartitionRunnerProbeCount> owner_seen_;
  static inline std::array<std::uint32_t, kPartitionRunnerProbeCount> active_callbacks_;
  static inline std::array<std::uint32_t, kPartitionRunnerProbeCount> max_partition_concurrency_;
  static inline std::uint32_t active_cross_partition_callbacks_{};
  static inline std::uint32_t max_cross_partition_concurrency_{};
  static inline std::size_t waiting_sources_{};
  static inline bool release_sources_{};
  static inline bool stable_owner_threads_{true};
  static inline bool balanced_callbacks_{true};
  static inline bool barrier_timed_out_{};
  static inline std::vector<PartitionRunnerCallbackRecord> callback_records_;
  static inline std::vector<std::uint32_t> sink_values_;
};

struct CrossPartitionFlowRecord {
  std::uint32_t value{};
  SampleMetadata metadata{};
  TagTime tag_time{};
  std::thread::id thread;
};

struct CrossPartitionFlowSnapshot {
  std::vector<CrossPartitionFlowRecord> source_records;
  std::vector<CrossPartitionFlowRecord> sink_records;
  bool first_sink_held{};
  bool hold_timed_out{};
  bool failure_triggered{};
};

class CrossPartitionFlowProbeState {
 public:
  static void reset(bool hold_first_sink) {
    const std::lock_guard lock(mutex_);
    source_records_.clear();
    sink_records_.clear();
    hold_first_sink_ = hold_first_sink;
    first_sink_held_ = false;
    release_first_sink_ = false;
    hold_timed_out_ = false;
    failure_triggered_ = false;
  }

  static void record_source(std::uint32_t value, const SampleMetadata& metadata, TagTime tag_time) {
    {
      const std::lock_guard lock(mutex_);
      source_records_.push_back(CrossPartitionFlowRecord{
          .value = value,
          .metadata = metadata,
          .tag_time = tag_time,
          .thread = std::this_thread::get_id(),
      });
    }
    cv_.notify_all();
  }

  static void record_sink_and_maybe_hold(std::uint32_t value, const SampleMetadata& metadata,
                                         TagTime tag_time) {
    std::unique_lock lock(mutex_);
    sink_records_.push_back(CrossPartitionFlowRecord{
        .value = value,
        .metadata = metadata,
        .tag_time = tag_time,
        .thread = std::this_thread::get_id(),
    });
    if (hold_first_sink_ && sink_records_.size() == 1U) {
      first_sink_held_ = true;
      cv_.notify_all();
      if (!cv_.wait_for(lock, 5s, [] { return release_first_sink_; })) {
        hold_timed_out_ = true;
      }
    }
    cv_.notify_all();
  }

  [[nodiscard]] static bool wait_for_first_sink_hold() {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, 5s, [] { return first_sink_held_; });
  }

  [[nodiscard]] static std::optional<std::size_t> wait_for_source_stall(
      std::size_t minimum_count, std::chrono::milliseconds quiet_period = 30ms) {
    std::unique_lock lock(mutex_);
    if (!cv_.wait_for(lock, 5s, [minimum_count] {
          return first_sink_held_ && source_records_.size() >= minimum_count;
        })) {
      return std::nullopt;
    }

    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (std::chrono::steady_clock::now() < deadline) {
      const std::size_t observed = source_records_.size();
      if (!cv_.wait_for(lock, quiet_period, [observed] {
            return release_first_sink_ || source_records_.size() != observed;
          })) {
        return observed;
      }
      if (release_first_sink_) {
        return std::nullopt;
      }
    }
    return std::nullopt;
  }

  [[nodiscard]] static bool wait_for_sink_count(std::size_t count) {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, 5s, [count] { return sink_records_.size() >= count; });
  }

  static void release_first_sink() {
    {
      const std::lock_guard lock(mutex_);
      release_first_sink_ = true;
    }
    cv_.notify_all();
  }

  static void record_failure_triggered() {
    {
      const std::lock_guard lock(mutex_);
      failure_triggered_ = true;
    }
    cv_.notify_all();
  }

  [[nodiscard]] static bool wait_for_failure_triggered() {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, 5s, [] { return failure_triggered_; });
  }

  [[nodiscard]] static CrossPartitionFlowSnapshot snapshot() {
    const std::lock_guard lock(mutex_);
    return CrossPartitionFlowSnapshot{
        .source_records = source_records_,
        .sink_records = sink_records_,
        .first_sink_held = first_sink_held_,
        .hold_timed_out = hold_timed_out_,
        .failure_triggered = failure_triggered_,
    };
  }

 private:
  static inline std::mutex mutex_;
  static inline std::condition_variable cv_;
  static inline std::vector<CrossPartitionFlowRecord> source_records_;
  static inline std::vector<CrossPartitionFlowRecord> sink_records_;
  static inline bool hold_first_sink_{};
  static inline bool first_sink_held_{};
  static inline bool release_first_sink_{};
  static inline bool hold_timed_out_{};
  static inline bool failure_triggered_{};
};

struct PartitionControlLifecycleRecord {
  std::size_t partition_index{};
  std::uint32_t operator_id{};
  LifecycleStage stage{LifecycleStage::kConfigure};
  std::thread::id thread;
};

struct PartitionControlProbeSnapshot {
  std::array<std::thread::id, 2U> owner_threads;
  std::array<std::uint32_t, 2U> compute_counts{};
  std::vector<PartitionControlLifecycleRecord> lifecycle_records;
  bool stable_owner_threads{true};
  bool barrier_timed_out{};
  bool restart_requested{};
  bool restart_completed{};
  bool lifecycle_before_peer_quiescence{};
  bool callback_crossed_control_epoch{};
};

class PartitionControlProbeState {
 public:
  static void reset() {
    const std::lock_guard lock(mutex_);
    owner_threads_ = {};
    owner_seen_ = {};
    compute_counts_ = {};
    active_callbacks_ = {};
    first_compute_waiters_ = 0U;
    release_peer_ = false;
    stable_owner_threads_ = true;
    barrier_timed_out_ = false;
    restart_requested_ = false;
    restart_completed_ = false;
    lifecycle_before_peer_quiescence_ = false;
    callback_crossed_control_epoch_ = false;
    target_start_count_ = 0U;
    lifecycle_records_.clear();
  }

  [[nodiscard]] static bool enter_compute(std::size_t partition_index) {
    std::unique_lock lock(mutex_);
    if (partition_index >= owner_threads_.size()) {
      return false;
    }
    const std::thread::id thread = std::this_thread::get_id();
    if (!owner_seen_[partition_index]) {
      owner_threads_[partition_index] = thread;
      owner_seen_[partition_index] = true;
    } else if (owner_threads_[partition_index] != thread) {
      stable_owner_threads_ = false;
    }
    ++compute_counts_[partition_index];
    ++active_callbacks_[partition_index];
    if (restart_requested_ && !restart_completed_) {
      callback_crossed_control_epoch_ = true;
    }
    if (compute_counts_[partition_index] != 1U) {
      cv_.notify_all();
      return true;
    }

    ++first_compute_waiters_;
    cv_.notify_all();
    const auto both_owners_entered = [] { return first_compute_waiters_ == 2U; };
    if (!cv_.wait_for(lock, 5s, both_owners_entered)) {
      barrier_timed_out_ = true;
      return false;
    }
    cv_.notify_all();
    if (partition_index == 1U && !cv_.wait_for(lock, 5s, [] { return release_peer_; })) {
      barrier_timed_out_ = true;
      return false;
    }
    return true;
  }

  static void leave_compute(std::size_t partition_index) {
    const std::lock_guard lock(mutex_);
    if (partition_index >= active_callbacks_.size() || active_callbacks_[partition_index] == 0U) {
      stable_owner_threads_ = false;
      return;
    }
    --active_callbacks_[partition_index];
    cv_.notify_all();
  }

  static void record_restart_requested() {
    {
      const std::lock_guard lock(mutex_);
      restart_requested_ = true;
    }
    cv_.notify_all();
  }

  static void record_lifecycle(std::size_t partition_index, std::uint32_t operator_id,
                               LifecycleStage stage) {
    {
      const std::lock_guard lock(mutex_);
      lifecycle_records_.push_back(PartitionControlLifecycleRecord{
          .partition_index = partition_index,
          .operator_id = operator_id,
          .stage = stage,
          .thread = std::this_thread::get_id(),
      });
      if (partition_index == 0U && stage == LifecycleStage::kStop && restart_requested_ &&
          !restart_completed_ && active_callbacks_[1] != 0U) {
        lifecycle_before_peer_quiescence_ = true;
      }
      if (partition_index == 0U && stage == LifecycleStage::kStart && ++target_start_count_ == 2U) {
        restart_completed_ = true;
      }
    }
    cv_.notify_all();
  }

  [[nodiscard]] static bool wait_until_restart_requested() {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, 5s, [] { return restart_requested_; });
  }

  static void release_peer() {
    {
      const std::lock_guard lock(mutex_);
      release_peer_ = true;
    }
    cv_.notify_all();
  }

  [[nodiscard]] static bool wait_until_resumed() {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, 5s, [] {
      return restart_completed_ && compute_counts_[0] >= 2U && compute_counts_[1] >= 2U;
    });
  }

  [[nodiscard]] static PartitionControlProbeSnapshot snapshot() {
    const std::lock_guard lock(mutex_);
    return PartitionControlProbeSnapshot{
        .owner_threads = owner_threads_,
        .compute_counts = compute_counts_,
        .lifecycle_records = lifecycle_records_,
        .stable_owner_threads = stable_owner_threads_,
        .barrier_timed_out = barrier_timed_out_,
        .restart_requested = restart_requested_,
        .restart_completed = restart_completed_,
        .lifecycle_before_peer_quiescence = lifecycle_before_peer_quiescence_,
        .callback_crossed_control_epoch = callback_crossed_control_epoch_,
    };
  }

 private:
  static inline std::mutex mutex_;
  static inline std::condition_variable cv_;
  static inline std::array<std::thread::id, 2U> owner_threads_;
  static inline std::array<bool, 2U> owner_seen_;
  static inline std::array<std::uint32_t, 2U> compute_counts_;
  static inline std::array<std::uint32_t, 2U> active_callbacks_;
  static inline std::size_t first_compute_waiters_{};
  static inline bool release_peer_{};
  static inline bool stable_owner_threads_{true};
  static inline bool barrier_timed_out_{};
  static inline bool restart_requested_{};
  static inline bool restart_completed_{};
  static inline bool lifecycle_before_peer_quiescence_{};
  static inline bool callback_crossed_control_epoch_{};
  static inline std::uint32_t target_start_count_{};
  static inline std::vector<PartitionControlLifecycleRecord> lifecycle_records_;
};

class PartitionStartupGate final : public detail::RuntimeSupportThreadControl {
 public:
  [[nodiscard]] expected<void, Error> enter(SupportThreadRole role) override {
    if (role != SupportThreadRole::kController) {
      return {};
    }
    std::unique_lock lock(mutex_);
    controller_entered_ = true;
    cv_.notify_all();
    cv_.wait(lock, [this] { return controller_released_; });
    return {};
  }

  [[nodiscard]] bool wait_until_controller_entered() {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, 5s, [this] { return controller_entered_; });
  }

  void release_controller() {
    {
      const std::lock_guard lock(mutex_);
      controller_released_ = true;
    }
    cv_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool controller_entered_{};
  bool controller_released_{};
};

// Synchronizes independent partition roots so their stable owners must overlap in time.
// Representative test: AuthoredPartitionsUseStableSerialOwnersAndOverlapIndependently.
class PartitionRunnerBarrierSource final : public Operator<> {
 public:
  PartitionRunnerBarrierSource(std::size_t partition_index, std::uint32_t operator_id,
                               std::uint32_t value)
      : partition_index_(partition_index), operator_id_(operator_id), value_(value) {}

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 30s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    PartitionRunnerProbeState::enter_source(partition_index_, operator_id_);
    expected<void, Error> result;
    if (!emitted_) {
      emitted_ = true;
      result = output.emit(InlinePartitionPod{.value = value_});
    }
    PartitionRunnerProbeState::leave_callback(partition_index_);
    return result;
  }

  Output<InlinePartitionPod> output;

 private:
  std::size_t partition_index_{};
  std::uint32_t operator_id_{};
  std::uint32_t value_{};
  bool emitted_{};
};

// Records sink execution on an expected partition owner and consumes one POD sample.
// Representative test: AuthoredPartitionsUseStableSerialOwnersAndOverlapIndependently.
class PartitionRunnerProbeSink final : public Operator<> {
 public:
  PartitionRunnerProbeSink(std::size_t partition_index, std::uint32_t operator_id)
      : partition_index_(partition_index), operator_id_(operator_id) {}

  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(2U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    PartitionRunnerProbeState::enter_sink(partition_index_, operator_id_);
    auto sample = input.receive();
    if (!sample) {
      PartitionRunnerProbeState::leave_callback(partition_index_);
      return make_unexpected(std::move(sample.error()));
    }
    PartitionRunnerProbeState::record_sink(sample->data.value);
    PartitionRunnerProbeState::leave_callback(partition_index_);
    return {};
  }

  Input<InlinePartitionPod> input;

 private:
  std::size_t partition_index_{};
  std::uint32_t operator_id_{};
};

// Emits one POD sample while recording its exact partition owner and callback boundaries.
// Representative test: CrossPartitionChannelConfinesFusedCallbacksToExactStableOwners.
class PartitionRunnerProbeSource final : public Operator<> {
 public:
  PartitionRunnerProbeSource(std::size_t partition_index, std::uint32_t operator_id,
                             std::uint32_t value)
      : partition_index_(partition_index), operator_id_(operator_id), value_(value) {}

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 30s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    PartitionRunnerProbeState::enter_callback(partition_index_, operator_id_);
    expected<void, Error> result;
    if (!emitted_) {
      emitted_ = true;
      result = output.emit(InlinePartitionPod{.value = value_});
    }
    PartitionRunnerProbeState::leave_callback(partition_index_);
    return result;
  }

  Output<InlinePartitionPod> output;

 private:
  std::size_t partition_index_{};
  std::uint32_t operator_id_{};
  std::uint32_t value_{};
  bool emitted_{};
};

// Relays a POD sample while recording fused execution on its partition owner.
// Representative test: CrossPartitionChannelConfinesFusedCallbacksToExactStableOwners.
class PartitionRunnerProbeRelay final : public Operator<> {
 public:
  PartitionRunnerProbeRelay(std::size_t partition_index, std::uint32_t operator_id,
                            std::uint32_t increment)
      : partition_index_(partition_index), operator_id_(operator_id), increment_(increment) {}

  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(2U);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    PartitionRunnerProbeState::enter_callback(partition_index_, operator_id_);
    auto sample = input.receive();
    if (!sample) {
      PartitionRunnerProbeState::leave_callback(partition_index_);
      return make_unexpected(std::move(sample.error()));
    }
    auto result = output.emit(InlinePartitionPod{.value = sample->data.value + increment_});
    PartitionRunnerProbeState::leave_callback(partition_index_);
    return result;
  }

  Input<InlinePartitionPod> input;
  Output<InlinePartitionPod> output;

 private:
  std::size_t partition_index_{};
  std::uint32_t operator_id_{};
  std::uint32_t increment_{};
};

// Emits a metadata-rich POD sequence until cross-partition blocking credit stalls publication.
// Representative test: CrossPartitionBlockingCreditRecoversInOrderWithExactMetadataAndTagTime.
class CrossPartitionFlowSource final : public Operator<> {
 public:
  explicit CrossPartitionFlowSource(std::size_t sample_count) : sample_count_(sample_count) {}

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 100us});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (next_sample_ >= sample_count_) {
      return {};
    }
    const std::uint32_t value = static_cast<std::uint32_t>(101U + next_sample_);
    const SampleMetadata metadata{
        .sequence = next_sample_ + 1U,
        .capture_timestamp_ns = static_cast<std::int64_t>(10'000U + next_sample_),
        .source_clock_id = ClockId{1U},
        .trace_id = 20'000U + next_sample_,
        .frame_id = 30'000U + next_sample_,
        .byte_size = sizeof(InlinePartitionPod),
        .payload_kind = PayloadKind::kInlineTyped,
    };
    auto emitted =
        output.emit(Sample<InlinePartitionPod>{InlinePartitionPod{.value = value}, metadata});
    if (!emitted) {
      if (emitted.error().code == ErrorCode::kBackpressured) {
        return {};
      }
      return make_unexpected(std::move(emitted.error()));
    }
    CrossPartitionFlowProbeState::record_source(value, metadata, context.tag_time());
    ++next_sample_;
    return {};
  }

  Output<InlinePartitionPod> output;

 private:
  std::size_t sample_count_{};
  std::uint64_t next_sample_{};
};

// Consumes cross-partition POD samples and can hold the first sample to exhaust channel credit.
// Representative test: CrossPartitionBlockingCreditRecoversInOrderWithExactMetadataAndTagTime.
class CrossPartitionFlowSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    CrossPartitionFlowProbeState::record_sink_and_maybe_hold(
        sample->data.value, sample->metadata, context.tag_time());
    return {};
  }

  Input<InlinePartitionPod> input;
};

// Fails a peer partition after observing that a cross-partition source is backpressured.
// Representative test: StopAndFailureRetireOwnersWhileCrossPartitionBlockingCreditIsHeld.
class CrossPartitionFlowFailure final : public Operator<> {
 public:
  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 100us});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    const auto stalled = CrossPartitionFlowProbeState::wait_for_source_stall(1U);
    CrossPartitionFlowProbeState::record_failure_triggered();
    if (!stalled) {
      return make_unexpected(
          Error{ErrorCode::kFailure, "CROSS_PARTITION_BACKPRESSURE_NOT_OBSERVED"});
    }
    return make_unexpected(Error{ErrorCode::kFailure, "CROSS_PARTITION_PEER_FAILURE"});
  }
};

// Repeatedly records partition ownership and optionally synchronizes, completes, or fails.
// Representative test: StopBroadcastRetiresEveryPartitionOwner.
class PartitionRunnerPeriodicProbe final : public Operator<> {
 public:
  PartitionRunnerPeriodicProbe(std::size_t partition_index, std::uint32_t operator_id,
                               bool fail_compute = false, bool synchronize_first_compute = false,
                               bool request_completion = false)
      : partition_index_(partition_index),
        operator_id_(operator_id),
        fail_compute_(fail_compute),
        synchronize_first_compute_(synchronize_first_compute),
        request_completion_(request_completion) {}

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (synchronize_first_compute_ && !first_compute_completed_) {
      PartitionRunnerProbeState::enter_barrier(partition_index_, operator_id_, 2U);
      first_compute_completed_ = true;
    } else {
      PartitionRunnerProbeState::enter_callback(partition_index_, operator_id_);
    }
    if (request_completion_ && !completion_requested_) {
      completion_requested_ = true;
      context.request_completion();
    }
    PartitionRunnerProbeState::leave_callback(partition_index_);
    if (fail_compute_) {
      return make_unexpected(Error{ErrorCode::kFailure, "PARTITION_OWNER_FAILURE"});
    }
    return {};
  }

 private:
  std::size_t partition_index_{};
  std::uint32_t operator_id_{};
  bool fail_compute_{};
  bool synchronize_first_compute_{};
  bool request_completion_{};
  bool first_compute_completed_{};
  bool completion_requested_{};
};

// Publishes once and requests completion while a peer partition waits at its admission gate.
// Representative test: CallbackCompletionClosesBeforeClaimAndDoesNotDrainQueuedDelivery.
class PartitionRunnerCompletionSource final : public Operator<> {
 public:
  static void set_admission_control(
      detail::RuntimeTerminationAdmissionTestControl* admission_control) noexcept {
    admission_control_.store(admission_control, std::memory_order_release);
  }

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto* const admission_control = admission_control_.load(std::memory_order_acquire);
    if (admission_control == nullptr) {
      return make_unexpected(
          Error{ErrorCode::kFailure, "PARTITION_COMPLETION_ADMISSION_CONTROL_MISSING"});
    }
    PartitionRunnerProbeState::enter_callback(0U, 1U);
    auto emitted = output.emit(InlinePartitionPod{.value = 101U});
    if (!emitted) {
      PartitionRunnerProbeState::leave_callback(0U);
      return emitted;
    }
    const bool peer_waiting = admission_control->wait_until_arrived();
    context.request_completion();
    admission_control->release();
    PartitionRunnerProbeState::leave_callback(0U);
    if (!peer_waiting) {
      return make_unexpected(
          Error{ErrorCode::kFailure, "PARTITION_COMPLETION_PEER_DID_NOT_REACH_ADMISSION"});
    }
    return {};
  }

  Output<InlinePartitionPod> output;

 private:
  static inline std::atomic<detail::RuntimeTerminationAdmissionTestControl*> admission_control_{};
};

// Requests a self-restart while a peer partition remains inside its coordinated callback.
// Representative test: RestartControlEpochQuiescesEveryPartitionOwnerBeforeLifecycleCallbacks.
class PartitionControlRestartTarget final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.lifecycle()
        .stage(LifecycleStage::kStart, &PartitionControlRestartTarget::start_stage)
        .stage(LifecycleStage::kStop, &PartitionControlRestartTarget::stop_stage);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (!PartitionControlProbeState::enter_compute(0U)) {
      PartitionControlProbeState::leave_compute(0U);
      return make_unexpected(
          Error{ErrorCode::kFailure, "PARTITION_CONTROL_TARGET_BARRIER_TIMEOUT"});
    }
    if (!restart_requested_) {
      auto receipt =
          context.request_restart(LifecycleRestartOptions{.from_stage = LifecycleStage::kStart});
      if (!receipt) {
        PartitionControlProbeState::leave_compute(0U);
        return make_unexpected(std::move(receipt.error()));
      }
      restart_requested_ = true;
      PartitionControlProbeState::record_restart_requested();
    }
    PartitionControlProbeState::leave_compute(0U);
    return {};
  }

 private:
  LifecycleStatus start_stage(LifecycleContext&) noexcept {
    PartitionControlProbeState::record_lifecycle(0U, 1U, LifecycleStage::kStart);
    return LifecycleStatus::kOk;
  }

  LifecycleStatus stop_stage(LifecycleContext&) noexcept {
    PartitionControlProbeState::record_lifecycle(0U, 1U, LifecycleStage::kStop);
    return LifecycleStatus::kOk;
  }

  bool restart_requested_{};
};

// Holds a peer callback across the restart control epoch and records its lifecycle ordering.
// Representative test: RestartControlEpochQuiescesEveryPartitionOwnerBeforeLifecycleCallbacks.
class PartitionControlPeer final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.lifecycle()
        .stage(LifecycleStage::kStart, &PartitionControlPeer::start_stage)
        .stage(LifecycleStage::kStop, &PartitionControlPeer::stop_stage);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    if (!PartitionControlProbeState::enter_compute(1U)) {
      PartitionControlProbeState::leave_compute(1U);
      return make_unexpected(Error{ErrorCode::kFailure, "PARTITION_CONTROL_PEER_BARRIER_TIMEOUT"});
    }
    PartitionControlProbeState::leave_compute(1U);
    return {};
  }

 private:
  LifecycleStatus start_stage(LifecycleContext&) noexcept {
    PartitionControlProbeState::record_lifecycle(1U, 2U, LifecycleStage::kStart);
    return LifecycleStatus::kOk;
  }

  LifecycleStatus stop_stage(LifecycleContext&) noexcept {
    PartitionControlProbeState::record_lifecycle(1U, 2U, LifecycleStage::kStop);
    return LifecycleStatus::kOk;
  }
};

[[nodiscard]] ExecutionPlan make_four_partition_overlap_plan() {
  // [source-N] -> [sink-N]  (four independent partitions, N = 0...3)
  Graph graph{"partition-runner-overlap"};
  for (std::size_t index = 0U; index < kPartitionRunnerProbeCount; ++index) {
    const std::string suffix = std::to_string(index);
    const std::uint32_t source_operator_id = static_cast<std::uint32_t>(index * 2U + 1U);
    const std::uint32_t sink_operator_id = source_operator_id + 1U;
    const auto source = graph.op<PartitionRunnerBarrierSource>(
        "source-" + suffix, index, source_operator_id, 100U + static_cast<std::uint32_t>(index));
    const auto sink = graph.op<PartitionRunnerProbeSink>("sink-" + suffix, index, sink_operator_id);
    graph.add_flow(source->output, sink->input, queue_options(QueuePolicy::kBlocking));
    graph.partition("partition-" + suffix).add(source).add(sink);
  }
  graph.set_default_clock(graph.add_clock<RealtimeClock>("partition-runner-clock"));
  return compile(graph);
}

[[nodiscard]] ExecutionPlan make_cross_partition_fused_chain_plan() {
  // [source -> relay] -- partition boundary --> [sink]
  Graph graph{"partition-runner-fused-channel"};
  const auto source = graph.op<PartitionRunnerProbeSource>("source", 0U, 1U, 100U);
  const auto relay = graph.op<PartitionRunnerProbeRelay>("relay", 0U, 2U, 1U);
  const auto sink = graph.op<PartitionRunnerProbeSink>("sink", 1U, 3U);
  graph.add_flow(source->output, relay->input, queue_options(QueuePolicy::kBlocking));
  graph.add_flow(relay->output, sink->input, queue_options(QueuePolicy::kBlocking));
  graph.partition("source-fusion").add(source).add(relay);
  graph.partition("sink").add(sink);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("partition-runner-clock"));
  return compile(graph);
}

[[nodiscard]] ExecutionPlan make_cross_partition_fanout_plan() {
  // [source] -- partition boundary --> [first sink]
  //     \---- partition boundary --> [second sink]
  Graph graph{"partition-runner-channel-fanout"};
  const auto source = graph.op<PartitionRunnerProbeSource>("source", 0U, 1U, 101U);
  const auto first = graph.op<PartitionRunnerProbeSink>("first", 1U, 2U);
  const auto second = graph.op<PartitionRunnerProbeSink>("second", 2U, 3U);
  graph.add_flow(source->output, first->input, queue_options(QueuePolicy::kBlocking));
  graph.add_flow(source->output, second->input, queue_options(QueuePolicy::kBlocking));
  graph.partition("source").add(source);
  graph.partition("first-sink").add(first);
  graph.partition("second-sink").add(second);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("partition-runner-clock"));
  // The fanout control keeps this blocking fanout on HoloMQ, whose behavior these tests cover;
  // compile() would deliver it directly (CHAN-167).
  return detail::compile_with_process_local_holomq_fanout(graph, CompileOptions{});
}

[[nodiscard]] ExecutionPlan make_cross_partition_flow_plan(std::size_t sample_count,
                                                           bool include_failure = false) {
  // [source] -- partition boundary --> [sink]    [optional failing peer]
  Graph graph{include_failure ? "partition-runner-channel-failure"
                              : "partition-runner-channel-credit"};
  const auto source = graph.op<CrossPartitionFlowSource>("source", sample_count);
  const auto sink = graph.op<CrossPartitionFlowSink>("sink");
  graph.add_flow(source->output, sink->input, queue_options(QueuePolicy::kBlocking));
  graph.partition("source").add(source);
  graph.partition("sink").add(sink);
  if (include_failure) {
    const auto failure = graph.op<CrossPartitionFlowFailure>("failure");
    graph.partition("failure").add(failure);
  }
  graph.set_default_clock(graph.add_clock<RealtimeClock>("partition-runner-clock"));
  return compile(graph);
}

// Two independent clock sources cross an import-forwarding boundary inside one selected tree.
class AtomicProbeLeaf final : public Graph {
 public:
  AtomicProbeLeaf(std::size_t partition, std::uint32_t id) : partition_(partition), id_(id) {}
  [[nodiscard]] ClockKey clock_import() const noexcept { return clock_; }

 protected:
  void compose() override {
    clock_ = import_clock("timeline");
    set_default_clock(clock_);
    (void)op<PartitionRunnerPeriodicProbe>("second", partition_, id_);
  }

 private:
  ClockKey clock_;
  std::size_t partition_{};
  std::uint32_t id_{};
};

class AtomicProbeGraph final : public Graph {
 public:
  AtomicProbeGraph(std::size_t partition = 0U, std::uint32_t first_id = 1U)
      : partition_(partition), first_id_(first_id) {}
  [[nodiscard]] ClockKey clock_import() const noexcept { return clock_; }

 protected:
  void compose() override {
    clock_ = import_clock("timeline");
    set_default_clock(clock_);
    (void)op<PartitionRunnerPeriodicProbe>("first", partition_, first_id_, false, true);
    const auto nested = embed<AtomicProbeLeaf>("nested", partition_, first_id_ + 1U);
    bind_clock(nested, nested->clock_import(), clock_);
  }

 private:
  ClockKey clock_;
  std::size_t partition_{};
  std::uint32_t first_id_{};
};

[[nodiscard]] ExecutionPlan make_atomic_instance_plan(bool automatic, CompileOptions options = {}) {
  Graph graph{"atomic-embedded-runtime"};
  const auto bundle = graph.embed<AtomicProbeGraph>("bundle");
  const auto peer = graph.op<PartitionRunnerPeriodicProbe>("peer", 1U, 3U, false, true);
  const ClockKey clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(clock);
  graph.bind_clock(bundle, bundle->clock_import(), clock);
  graph.partition("bundle-owner").add(bundle);
  if (automatic) {
    graph.auto_partition_remaining();
  } else {
    graph.partition("peer-owner").add(peer);
  }
  return compile(graph, std::move(options));
}

[[nodiscard]] ExecutionPlan make_partition_cpu_placement_plan(std::uint32_t first_cpu,
                                                              std::uint32_t second_cpu,
                                                              CompileOptions options = {}) {
  // [first periodic root]    [second periodic root]
  Graph graph{"partition-runner-cpu-placement"};
  const auto first = graph.op<PartitionRunnerPeriodicProbe>("first", 0U, 1U);
  const auto second = graph.op<PartitionRunnerPeriodicProbe>("second", 1U, 2U);
  graph.partition("first").add(first);
  graph.partition("second").add(second);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("partition-runner-clock"));
  options.deployment
      .bind_thread_placement("first", ThreadPlacement{.affinity = CPUAffinity{first_cpu}})
      .bind_thread_placement("second", ThreadPlacement{.affinity = CPUAffinity{second_cpu}});
  return compile(graph, std::move(options));
}

[[nodiscard]] ExecutionPlan make_partition_stop_plan() {
  // [first periodic root]    [second periodic root]
  Graph graph{"partition-runner-stop"};
  const auto first = graph.op<PartitionRunnerPeriodicProbe>("first", 0U, 1U);
  const auto second = graph.op<PartitionRunnerPeriodicProbe>("second", 1U, 2U);
  graph.partition("first").add(first);
  graph.partition("second").add(second);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("partition-runner-clock"));
  return compile(graph);
}

[[nodiscard]] ExecutionPlan make_partition_completion_plan() {
  // [first completing root]    [second completing root]
  Graph graph{"partition-runner-completion"};
  const auto first = graph.op<PartitionRunnerPeriodicProbe>("first", 0U, 1U, false, false, true);
  const auto second = graph.op<PartitionRunnerPeriodicProbe>("second", 1U, 2U, false, false, true);
  graph.partition("first").add(first);
  graph.partition("second").add(second);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("partition-runner-clock"));
  return compile(graph);
}

[[nodiscard]] ExecutionPlan make_partition_completion_delivery_plan() {
  // [completing source] -- partition boundary --> [sink]
  Graph graph{"partition-runner-completion-delivery"};
  const auto source = graph.op<PartitionRunnerCompletionSource>("source");
  const auto sink = graph.op<PartitionRunnerProbeSink>("sink", 1U, 2U);
  graph.add_flow(source->output, sink->input, queue_options(QueuePolicy::kBlocking));
  graph.partition("source").add(source);
  graph.partition("sink").add(sink);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("partition-runner-clock"));
  return compile(graph);
}

[[nodiscard]] ExecutionPlan make_partition_failure_plan() {
  // [failing periodic root]    [periodic peer]
  Graph graph{"partition-runner-failure"};
  const auto failing = graph.op<PartitionRunnerPeriodicProbe>("failing", 0U, 1U, true, true);
  const auto peer = graph.op<PartitionRunnerPeriodicProbe>("peer", 1U, 2U, false, true);
  graph.partition("failing").add(failing);
  graph.partition("peer").add(peer);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("partition-runner-clock"));
  return compile(graph);
}

[[nodiscard]] ExecutionPlan make_partition_restart_control_plan() {
  // [restart target]    [coordinated peer]
  Graph graph{"partition-runner-restart-control"};
  const auto target = graph.op<PartitionControlRestartTarget>("target");
  const auto peer = graph.op<PartitionControlPeer>("peer");
  graph.partition("target").add(target);
  graph.partition("peer").add(peer);
  graph.lifecycle().allow_self_restart(target);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("partition-runner-clock"));
  CompileOptions options;
  options.lifecycle_transactions = inline_lifecycle_capacity();
  return compile(graph, std::move(options));
}

// REQ: RUNTIME-053
TEST(RuntimeInlinePartitionExecution, AtomicEmbeddedIndependentSourcesShareOwnerAndProgress) {
  for (const bool automatic : {false, true}) {
    PartitionRunnerProbeState::reset();
    ExecutionPlan compiled = make_atomic_instance_plan(automatic);
    ASSERT_TRUE(compiled.ok()) << compiled.json();
    const auto selected = compiled.partitions().lookup_named("bundle-owner");
    ASSERT_TRUE(selected);
    ASSERT_EQ(selected->graph_members().size(), 1U);
    EXPECT_EQ(std::vector<std::string>(selected->operator_paths().begin(),
                                       selected->operator_paths().end()),
              (std::vector<std::string>{"bundle/first", "bundle/nested/second"}));
    const auto state = detail::ExecutionPlanAccess::retain(compiled);
    ASSERT_TRUE(state);
    ASSERT_EQ(state->partition_runners.size(), 2U);
    const auto runner = std::ranges::find(
        state->partition_runners, selected->id(), &detail::PartitionRunnerPlanRecord::partition);
    ASSERT_NE(runner, state->partition_runners.end());
    ASSERT_EQ(runner->operators.size(), 2U);
    EXPECT_TRUE(runner->operators[0].topology_root);
    EXPECT_TRUE(runner->operators[1].topology_root);
    EXPECT_NE(runner->operators[0].readiness_source, runner->operators[1].readiness_source);
    const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
    compiled = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
    RunSession session = run_async(compiled);
    const std::array ids{1U, 2U, 3U};
    const bool observed = PartitionRunnerProbeState::wait_for_each_operator(ids, 3U);
    session.request_stop();
    session.wait();
    ASSERT_TRUE(observed);
    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_TRUE(session.diagnostics().empty());
    const auto probe = PartitionRunnerProbeState::snapshot();
    EXPECT_TRUE(probe.stable_owner_threads);
    EXPECT_TRUE(probe.balanced_callbacks);
    EXPECT_FALSE(probe.barrier_timed_out);
    EXPECT_EQ(probe.max_partition_concurrency[0], 1U);
    EXPECT_EQ(probe.max_partition_concurrency[1], 1U);
    EXPECT_GE(probe.max_cross_partition_concurrency, 2U);
    EXPECT_NE(probe.owner_threads[0], probe.owner_threads[1]);
    for (const auto& callback : probe.callback_records) {
      EXPECT_EQ(callback.thread, probe.owner_threads[callback.operator_id == 3U ? 1U : 0U]);
    }
    expect_no_legacy_lane_handoffs(counters->snapshot());
  }
}

// REQ: RUNTIME-053
TEST(RuntimeInlinePartitionExecution, AtomicEmbeddedRepeatedInstancesKeepDistinctOwners) {
  Graph graph{"repeated-atomic-instances"};
  const auto left = graph.embed<AtomicProbeGraph>("left", 0U, 1U);
  const auto right = graph.embed<AtomicProbeGraph>("right", 1U, 3U);
  const auto clock = graph.add_clock<RealtimeClock>("timeline");
  graph.set_default_clock(clock);
  graph.bind_clock(left, left->clock_import(), clock);
  graph.bind_clock(right, right->clock_import(), clock);
  graph.partition("left-owner").add(left);
  graph.partition("right-owner").add(right);
  const auto plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 2U);
  PartitionRunnerProbeState::reset();
  RunSession session = run_async(plan);
  const std::array ids{1U, 2U, 3U, 4U};
  const bool observed = PartitionRunnerProbeState::wait_for_each_operator(ids, 3U);
  session.request_stop();
  session.wait();
  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  const auto probe = PartitionRunnerProbeState::snapshot();
  EXPECT_TRUE(probe.stable_owner_threads);
  EXPECT_FALSE(probe.barrier_timed_out);
  EXPECT_NE(probe.owner_threads[0], probe.owner_threads[1]);
  EXPECT_EQ(probe.max_partition_concurrency[0], 1U);
  EXPECT_EQ(probe.max_partition_concurrency[1], 1U);
  for (const auto& callback : probe.callback_records) {
    EXPECT_EQ(callback.thread, probe.owner_threads[callback.operator_id <= 2U ? 0U : 1U]);
  }
}

// REQ: RUNTIME-053
TEST(RuntimeInlinePartitionExecution, AtomicEmbeddedInvalidProvenanceRejectsBeforeCallbacks) {
  const auto compiled = make_atomic_instance_plan(false);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto baseline = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(baseline);
  for (const bool omit : {false, true}) {
    auto state = std::make_shared<detail::ExecutionPlanState>(*baseline);
    auto selected = std::ranges::find_if(state->execution_partitions, [](const auto& partition) {
      return partition.authored_name == "bundle-owner";
    });
    ASSERT_NE(selected, state->execution_partitions.end());
    if (omit) {
      selected->graph_members.clear();
    } else {
      ASSERT_EQ(selected->graph_members.size(), 1U);
      selected->graph_members[0] = GraphId{9999U};
    }
    PartitionRunnerProbeState::reset();
    const auto malformed = detail::ExecutionPlanAccess::make(std::move(state));
    RunSession session = run_async(malformed);
    session.wait();
    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    EXPECT_TRUE(std::ranges::any_of(session.diagnostics(), [](const auto& diagnostic) {
      return diagnostic.token == "RUNTIME_PLACEMENT_BINDING_UNAVAILABLE";
    }));
    EXPECT_TRUE(PartitionRunnerProbeState::snapshot().callback_records.empty());
  }
}

// REQ: RUNTIME-053
TEST(RuntimeInlinePartitionExecution, AtomicEmbeddedIndependentSourcesApplyPartitionCpuPlacement) {
#if defined(__linux__)
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
    GTEST_SKIP() << "sched_getaffinity could not read the process cpuset";
  }
  std::vector<std::uint32_t> cpus;
  for (std::uint32_t cpu = 0U; cpu < CPU_SETSIZE && cpus.size() < 2U; ++cpu) {
    if (CPU_ISSET(static_cast<int>(cpu), &allowed))
      cpus.push_back(cpu);
  }
  if (cpus.size() < 2U) {
    GTEST_SKIP() << "At least two allowed CPUs are required for distinct partition placement";
  }
  CompileOptions options;
  options.deployment
      .bind_thread_placement("bundle-owner", ThreadPlacement{.affinity = CPUAffinity{cpus[0]}})
      .bind_thread_placement("peer-owner", ThreadPlacement{.affinity = CPUAffinity{cpus[1]}});
  PartitionRunnerProbeState::reset();
  const auto compiled = make_atomic_instance_plan(false, std::move(options));
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  RunSession session = run_async(compiled);
  const std::array ids{1U, 2U, 3U};
  const bool observed = PartitionRunnerProbeState::wait_for_each_operator(ids, 3U);
  session.request_stop();
  session.wait();
  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  const auto probe = PartitionRunnerProbeState::snapshot();
  EXPECT_TRUE(probe.stable_owner_threads);
  EXPECT_NE(probe.owner_threads[0], probe.owner_threads[1]);
  for (const auto& callback : probe.callback_records) {
    EXPECT_EQ(callback.cpu, static_cast<std::int32_t>(cpus[callback.partition_index]));
    EXPECT_EQ(callback.thread, probe.owner_threads[callback.partition_index]);
  }
#else
  GTEST_SKIP() << "CPU-affinity readback is supported only on Linux";
#endif
}

// REQ: RUNTIME-041
TEST(RuntimeInlinePartitionExecution,
     AuthoredPartitionsUseStableSerialOwnersAndOverlapIndependently) {
  // Key setup: synchronize four independent source-sink partitions at their root callbacks.
  PartitionRunnerProbeState::reset();
  ExecutionPlan compiled = make_four_partition_overlap_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_EQ(state->partition_runners.size(), kPartitionRunnerProbeCount) << compiled.json();
  ASSERT_EQ(compiled.execution_partitions().size(), kPartitionRunnerProbeCount);
  for (std::size_t index = 0U; index < state->partition_runners.size(); ++index) {
    const auto& runner = state->partition_runners[index];
    EXPECT_EQ(runner.ordinal, index);
    EXPECT_EQ(runner.partition, compiled.execution_partitions()[index].id);
    EXPECT_EQ(runner.operators.size(), 2U);
    EXPECT_EQ(runner.internal_edges.size(), 1U);
    EXPECT_TRUE(runner.incoming_boundary_edges.empty());
    EXPECT_TRUE(runner.outgoing_boundary_edges.empty());
  }

  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const auto control = std::make_shared<InlinePartitionControllerProbe>();
  compiled = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  compiled = detail::RuntimeTestPlanAccess::with_support_thread_control(compiled, control);
  RunSession session = run_async(compiled);
  const bool observed = PartitionRunnerProbeState::wait_for_sinks(kPartitionRunnerProbeCount);
  session.request_stop();
  session.wait();

  // Key assertion: each partition is serial on one stable owner while partitions overlap.
  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  const PartitionRunnerProbeSnapshot probe = PartitionRunnerProbeState::snapshot();
  EXPECT_TRUE(probe.stable_owner_threads);
  EXPECT_TRUE(probe.balanced_callbacks);
  EXPECT_FALSE(probe.barrier_timed_out);
  EXPECT_GE(probe.max_cross_partition_concurrency, 2U);
  EXPECT_TRUE(std::ranges::all_of(probe.owner_seen, [](bool seen) { return seen; }));
  EXPECT_TRUE(std::ranges::all_of(probe.max_partition_concurrency,
                                  [](std::uint32_t value) { return value == 1U; }));
  for (std::size_t lhs = 0U; lhs < probe.owner_threads.size(); ++lhs) {
    for (std::size_t rhs = lhs + 1U; rhs < probe.owner_threads.size(); ++rhs) {
      EXPECT_NE(probe.owner_threads[lhs], probe.owner_threads[rhs]);
    }
  }
  for (std::size_t partition = 0U; partition < kPartitionRunnerProbeCount; ++partition) {
    std::vector<std::uint32_t> callback_order;
    for (const auto& callback : probe.callback_records) {
      if (callback.partition_index == partition) {
        callback_order.push_back(callback.operator_id);
        EXPECT_EQ(callback.thread, probe.owner_threads[partition]);
      }
    }
    const std::uint32_t source_operator_id = static_cast<std::uint32_t>(partition * 2U + 1U);
    EXPECT_EQ(callback_order,
              (std::vector<std::uint32_t>{source_operator_id, source_operator_id + 1U}));
  }
  auto sink_values = probe.sink_values;
  std::ranges::sort(sink_values);
  EXPECT_EQ(sink_values, (std::vector<std::uint32_t>{100U, 101U, 102U, 103U}));

  const detail::RuntimePhaseCounters::Snapshot phase_counters = counters->snapshot();
  const auto count = [&](detail::RuntimePhaseCounter counter) {
    return phase_counters[detail::RuntimePhaseCounters::index(counter)];
  };
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kLaneDispatches), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kLaneComputeDispatches), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kLaneCompletionsObserved), 0U);
  EXPECT_GE(count(detail::RuntimePhaseCounter::kPartitionRunnerSweeps), kPartitionRunnerProbeCount);
  EXPECT_GE(count(detail::RuntimePhaseCounter::kPartitionRunnerWakeSignals),
            kPartitionRunnerProbeCount);
  EXPECT_GE(count(detail::RuntimePhaseCounter::kPartitionTailDispatchAttempts),
            kPartitionRunnerProbeCount);
  EXPECT_GE(count(detail::RuntimePhaseCounter::kPartitionTailDispatchAccepted),
            kPartitionRunnerProbeCount);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kPartitionTailDispatchFallbacks), 0U);
  EXPECT_GE(count(detail::RuntimePhaseCounter::kPartitionControlBarriers), 1U);
}

// REQ: RUNTIME-041
TEST(RuntimeInlinePartitionExecution,
     CrossPartitionChannelConfinesFusedCallbacksToExactStableOwners) {
  // Key setup: fuse source and relay in one partition and place the sink across a direct boundary.
  PartitionRunnerProbeState::reset();
  ExecutionPlan compiled = make_cross_partition_fused_chain_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_EQ(state->partition_runners.size(), 2U) << compiled.json();

  const auto source_runner = std::ranges::find_if(
      state->partition_runners, [](const auto& runner) { return runner.operators.size() == 2U; });
  const auto sink_runner = std::ranges::find_if(
      state->partition_runners, [](const auto& runner) { return runner.operators.size() == 1U; });
  ASSERT_NE(source_runner, state->partition_runners.end());
  ASSERT_NE(sink_runner, state->partition_runners.end());
  ASSERT_EQ(state->process_local_channel_services.size(), 2U);
  EXPECT_TRUE(std::ranges::all_of(state->process_local_channel_services, [](const auto& service) {
    return service.delivery_kind == detail::ProcessLocalChannelDeliveryKind::kDirectRetained;
  }));
  EXPECT_EQ(source_runner->internal_edges.size(), 1U);
  EXPECT_TRUE(source_runner->incoming_boundary_edges.empty());
  EXPECT_EQ(source_runner->outgoing_boundary_edges.size(), 1U);
  EXPECT_TRUE(sink_runner->internal_edges.empty());
  EXPECT_EQ(sink_runner->incoming_boundary_edges.size(), 1U);
  EXPECT_TRUE(sink_runner->outgoing_boundary_edges.empty());

  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const auto control = std::make_shared<InlinePartitionControllerProbe>();
  compiled = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  compiled = detail::RuntimeTestPlanAccess::with_support_thread_control(compiled, control);
  RunSession session = run_async(compiled);
  const bool observed = PartitionRunnerProbeState::wait_for_sinks(1U);
  session.request_stop();
  session.wait();

  // Key assertion: fused callbacks share their owner and the sink runs on its distinct owner.
  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  const PartitionRunnerProbeSnapshot probe = PartitionRunnerProbeState::snapshot();
  EXPECT_TRUE(probe.stable_owner_threads);
  EXPECT_TRUE(probe.balanced_callbacks);
  ASSERT_TRUE(probe.owner_seen[0]);
  ASSERT_TRUE(probe.owner_seen[1]);
  EXPECT_NE(probe.owner_threads[0], probe.owner_threads[1]);
  ASSERT_TRUE(control->controller_thread().has_value());
  // The fatal assertion above proves the optional is engaged.
  // NOLINTBEGIN(bugprone-unchecked-optional-access)
  EXPECT_NE(probe.owner_threads[0], *control->controller_thread());
  EXPECT_NE(probe.owner_threads[1], *control->controller_thread());
  // NOLINTEND(bugprone-unchecked-optional-access)
  ASSERT_EQ(probe.callback_records.size(), 3U);
  EXPECT_EQ(probe.callback_records[0].operator_id, 1U);
  EXPECT_EQ(probe.callback_records[1].operator_id, 2U);
  EXPECT_EQ(probe.callback_records[2].operator_id, 3U);
  EXPECT_EQ(probe.callback_records[0].partition_index, 0U);
  EXPECT_EQ(probe.callback_records[1].partition_index, 0U);
  EXPECT_EQ(probe.callback_records[2].partition_index, 1U);
  EXPECT_EQ(probe.callback_records[0].thread, probe.owner_threads[0]);
  EXPECT_EQ(probe.callback_records[1].thread, probe.owner_threads[0]);
  EXPECT_EQ(probe.callback_records[2].thread, probe.owner_threads[1]);
  EXPECT_EQ(probe.sink_values, (std::vector<std::uint32_t>{101U}));
  expect_no_legacy_lane_handoffs(counters->snapshot());
}

// REQ: RUNTIME-041
TEST(RuntimeInlinePartitionExecution,
     CrossPartitionOneRootFanoutDeliversOncePerConsumerWithoutLaneHandoffs) {
  // Key setup: fan one source publication across boundaries to two independently owned sinks.
  const ExecutionPlan baseline = make_cross_partition_fanout_plan();
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const auto state = detail::ExecutionPlanAccess::retain(baseline);
  ASSERT_TRUE(state);
  ASSERT_EQ(state->partition_runners.size(), 3U) << baseline.json();
  ASSERT_EQ(state->process_local_channel_services.size(), 1U);
  EXPECT_EQ(state->process_local_channel_services.front().delivery_kind,
            detail::ProcessLocalChannelDeliveryKind::kHoloMQ);
  EXPECT_EQ(std::ranges::count_if(
                state->partition_runners,
                [](const auto& runner) { return runner.outgoing_boundary_edges.size() == 2U; }),
            1);
  EXPECT_EQ(std::ranges::count_if(
                state->partition_runners,
                [](const auto& runner) { return runner.incoming_boundary_edges.size() == 1U; }),
            2);

  for (std::size_t iteration = 0U; iteration < 2U; ++iteration) {
    SCOPED_TRACE(iteration);
    PartitionRunnerProbeState::reset();
    const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
    const auto control = std::make_shared<InlinePartitionControllerProbe>();
    ExecutionPlan compiled = detail::RuntimeTestPlanAccess::with_phase_counters(baseline, counters);
    compiled = detail::RuntimeTestPlanAccess::with_support_thread_control(compiled, control);
    RunSession session = run_async(compiled);
    const bool observed = PartitionRunnerProbeState::wait_for_sinks(2U);
    session.request_stop();
    session.wait();

    // Key assertion: both stable sink owners receive exactly one copy without lane handoffs.
    ASSERT_TRUE(observed);
    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_TRUE(session.diagnostics().empty());
    const PartitionRunnerProbeSnapshot probe = PartitionRunnerProbeState::snapshot();
    EXPECT_TRUE(probe.stable_owner_threads);
    EXPECT_TRUE(probe.balanced_callbacks);
    EXPECT_TRUE(
        std::ranges::all_of(std::span{probe.owner_seen}.first(3U), [](bool seen) { return seen; }));
    for (std::size_t lhs = 0U; lhs < 3U; ++lhs) {
      for (std::size_t rhs = lhs + 1U; rhs < 3U; ++rhs) {
        EXPECT_NE(probe.owner_threads[lhs], probe.owner_threads[rhs]);
      }
    }
    ASSERT_EQ(probe.callback_records.size(), 3U);
    EXPECT_EQ(probe.callback_records.front().operator_id, 1U);
    std::array<std::uint32_t, 2U> sink_operators{
        probe.callback_records[1].operator_id,
        probe.callback_records[2].operator_id,
    };
    std::ranges::sort(sink_operators);
    EXPECT_EQ(sink_operators, (std::array<std::uint32_t, 2U>{2U, 3U}));
    auto sink_values = probe.sink_values;
    std::ranges::sort(sink_values);
    EXPECT_EQ(sink_values, (std::vector<std::uint32_t>{101U, 101U}));
    const auto snapshot = counters->snapshot();
    expect_no_legacy_lane_handoffs(snapshot);
    const auto count = [&](detail::RuntimePhaseCounter counter) {
      return snapshot[detail::RuntimePhaseCounters::index(counter)];
    };
    EXPECT_EQ(count(detail::RuntimePhaseCounter::kPartitionCausalFrontierAccepted), 0U);
  }
}

// REQ: RUNTIME-041
TEST(RuntimeInlinePartitionExecution,
     CrossPartitionBlockingCreditRecoversInOrderWithExactMetadataAndTagTime) {
  // Key setup: hold the first sink callback until capacity-one cross-partition credit stalls
  // source.
  constexpr std::size_t kSampleCount = 5U;
  CrossPartitionFlowProbeState::reset(true);
  ExecutionPlan compiled = make_cross_partition_flow_plan(kSampleCount);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_EQ(state->partition_runners.size(), 2U) << compiled.json();
  ASSERT_EQ(state->process_local_channel_services.size(), 1U);
  EXPECT_EQ(state->process_local_channel_services.front().delivery_kind,
            detail::ProcessLocalChannelDeliveryKind::kDirectRetained);

  std::array<detail::RuntimePhaseTraceRecord, 2048U> phase_trace_storage{};
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>(phase_trace_storage);
  compiled = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  RunSession session = run_async(compiled);
  const bool first_sink_held = CrossPartitionFlowProbeState::wait_for_first_sink_hold();
  const auto stalled_source_count =
      first_sink_held ? CrossPartitionFlowProbeState::wait_for_source_stall(1U) : std::nullopt;
  const CrossPartitionFlowSnapshot held = CrossPartitionFlowProbeState::snapshot();
  CrossPartitionFlowProbeState::release_first_sink();
  const bool all_received = CrossPartitionFlowProbeState::wait_for_sink_count(kSampleCount);
  session.request_stop();
  session.wait();

  // Key assertion: credit recovers and all samples arrive in order with exact transported metadata.
  ASSERT_TRUE(first_sink_held);
  ASSERT_TRUE(stalled_source_count.has_value());
  // The fatal assertion above proves the optional is engaged.
  // NOLINTBEGIN(bugprone-unchecked-optional-access)
  EXPECT_GE(*stalled_source_count, 1U);
  EXPECT_LT(*stalled_source_count, kSampleCount);
  // NOLINTEND(bugprone-unchecked-optional-access)
  ASSERT_EQ(held.sink_records.size(), 1U);
  EXPECT_FALSE(held.hold_timed_out);
  ASSERT_TRUE(all_received);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());

  const CrossPartitionFlowSnapshot probe = CrossPartitionFlowProbeState::snapshot();
  ASSERT_EQ(probe.source_records.size(), kSampleCount);
  ASSERT_EQ(probe.sink_records.size(), kSampleCount);
  EXPECT_FALSE(probe.hold_timed_out);
  for (std::size_t index = 0U; index < kSampleCount; ++index) {
    SCOPED_TRACE(index);
    const auto& source = probe.source_records[index];
    const auto& sink = probe.sink_records[index];
    const std::uint32_t expected_value = static_cast<std::uint32_t>(101U + index);
    EXPECT_EQ(source.value, expected_value);
    EXPECT_EQ(sink.value, expected_value);
    EXPECT_EQ(source.metadata.sequence, index + 1U);
    EXPECT_EQ(sink.metadata.sequence, source.metadata.sequence);
    EXPECT_EQ(source.metadata.capture_timestamp_ns, static_cast<std::int64_t>(10'000U + index));
    EXPECT_EQ(sink.metadata.capture_timestamp_ns, source.metadata.capture_timestamp_ns);
    EXPECT_EQ(source.metadata.source_clock_id, ClockId{1U});
    EXPECT_EQ(sink.metadata.source_clock_id, source.metadata.source_clock_id);
    EXPECT_EQ(sink.metadata.trace_id, source.metadata.trace_id);
    EXPECT_EQ(sink.metadata.frame_id, source.metadata.frame_id);
    EXPECT_EQ(sink.metadata.byte_size, sizeof(InlinePartitionPod));
    EXPECT_EQ(sink.metadata.payload_kind, PayloadKind::kInlineTyped);
    ASSERT_TRUE(source.tag_time.valid());
    ASSERT_TRUE(sink.tag_time.valid());
    // A cross-partition receive is a new activation whose exact logical coordinate comes from
    // the transported capture coordinate, rather than inheriting the producer's clock activation.
    EXPECT_EQ(sink.tag_time.logical_ns, source.metadata.capture_timestamp_ns);
    EXPECT_EQ(sink.tag_time.microstep, 0U);
    EXPECT_NE(sink.thread, source.thread);
  }
  const auto phase_snapshot = counters->snapshot();
  expect_no_legacy_lane_handoffs(phase_snapshot);
  EXPECT_GE(phase_snapshot[detail::RuntimePhaseCounters::index(
                detail::RuntimePhaseCounter::kPartitionRunnerWakeSignals)],
            1U);
  EXPECT_TRUE(std::ranges::any_of(counters->trace_records(), [](const auto& record) {
    return record.point == detail::RuntimePhaseTracePoint::kChannelReadinessWakeSignaled;
  }));
}

// REQ: RUNTIME-041
TEST(RuntimeInlinePartitionExecution,
     StopAndFailureRetireOwnersWhileCrossPartitionBlockingCreditIsHeld) {
  // Key setup: hold capacity-one channel credit during separate external-stop and peer-failure
  // runs.
  constexpr std::size_t kSampleCount = 1000U;
  {
    SCOPED_TRACE("external stop");
    CrossPartitionFlowProbeState::reset(true);
    ExecutionPlan compiled = make_cross_partition_flow_plan(kSampleCount);
    ASSERT_TRUE(compiled.ok()) << compiled.json();
    const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
    compiled = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
    RunSession session = run_async(compiled);
    const bool first_sink_held = CrossPartitionFlowProbeState::wait_for_first_sink_hold();
    const auto stalled_source_count =
        first_sink_held ? CrossPartitionFlowProbeState::wait_for_source_stall(1U) : std::nullopt;
    session.request_stop();
    CrossPartitionFlowProbeState::release_first_sink();
    session.wait();

    // Key assertion: external stop retires blocked owners without draining all source samples.
    ASSERT_TRUE(first_sink_held);
    ASSERT_TRUE(stalled_source_count.has_value());
    // The fatal assertion above proves the optional is engaged.
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    EXPECT_LT(*stalled_source_count, kSampleCount);
    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_TRUE(session.diagnostics().empty());
    EXPECT_FALSE(CrossPartitionFlowProbeState::snapshot().hold_timed_out);
    expect_no_legacy_lane_handoffs(counters->snapshot());
  }
  {
    SCOPED_TRACE("peer failure");
    CrossPartitionFlowProbeState::reset(true);
    ExecutionPlan compiled = make_cross_partition_flow_plan(kSampleCount, true);
    ASSERT_TRUE(compiled.ok()) << compiled.json();
    const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
    compiled = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
    RunSession session = run_async(compiled);
    const bool first_sink_held = CrossPartitionFlowProbeState::wait_for_first_sink_hold();
    const bool failure_triggered =
        first_sink_held && CrossPartitionFlowProbeState::wait_for_failure_triggered();
    CrossPartitionFlowProbeState::release_first_sink();
    const bool terminal = wait_for_terminal(session);
    if (!terminal) {
      session.request_stop();
    }
    session.wait();

    // Key assertion: peer failure also wakes and retires every owner while credit remains held.
    ASSERT_TRUE(first_sink_held);
    ASSERT_TRUE(failure_triggered);
    ASSERT_TRUE(terminal);
    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    EXPECT_TRUE(std::ranges::any_of(session.diagnostics(), [](const auto& diagnostic) {
      return diagnostic.token == "RUNTIME_OPERATOR_CALLBACK_FAILED";
    }));
    const CrossPartitionFlowSnapshot probe = CrossPartitionFlowProbeState::snapshot();
    EXPECT_TRUE(probe.failure_triggered);
    EXPECT_FALSE(probe.hold_timed_out);
    EXPECT_LT(probe.source_records.size(), kSampleCount);
    expect_no_legacy_lane_handoffs(counters->snapshot());
  }
}

// REQ: RUNTIME-041
TEST(RuntimeInlinePartitionExecution,
     AcceptedCpuPlacementPinsDistinctStablePartitionOwnersToAllowedCpus) {
#if defined(__linux__)
  // Key setup: bind two independent partition owners to distinct CPUs in the allowed process set.
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
    GTEST_SKIP() << "sched_getaffinity could not read the process cpuset";
  }
  std::vector<std::uint32_t> allowed_cpus;
  for (std::uint32_t cpu = 0U; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(static_cast<int>(cpu), &allowed)) {
      allowed_cpus.push_back(cpu);
    }
  }
  if (allowed_cpus.size() < 2U) {
    GTEST_SKIP() << "At least two allowed CPUs are required for distinct partition placement";
  }
  const std::array selected{allowed_cpus[0], allowed_cpus[1]};

  PartitionRunnerProbeState::reset();
  ExecutionPlan compiled = make_partition_cpu_placement_plan(selected[0], selected[1]);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  ASSERT_EQ(compiled.partition_placements().size(), 2U) << compiled.json();
  for (std::size_t index = 0U; index < selected.size(); ++index) {
    const auto& placement = compiled.partition_placements()[index];
    ASSERT_TRUE(placement.accepted.has_value());
    ASSERT_TRUE(placement.accepted->affinity.has_value());
    EXPECT_EQ(placement.accepted->affinity->cores, (std::vector<std::uint32_t>{selected[index]}));
  }

  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  compiled = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  RunSession session = run_async(compiled);
  const bool owners_started = PartitionRunnerProbeState::wait_for_owners(2U);
  session.request_stop();
  session.wait();

  // Key assertion: every callback remains on its stable owner and requested CPU.
  ASSERT_TRUE(owners_started);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  const PartitionRunnerProbeSnapshot probe = PartitionRunnerProbeState::snapshot();
  EXPECT_TRUE(probe.stable_owner_threads);
  EXPECT_TRUE(probe.balanced_callbacks);
  EXPECT_NE(probe.owner_threads[0], probe.owner_threads[1]);
  for (std::size_t partition = 0U; partition < selected.size(); ++partition) {
    std::size_t observations = 0U;
    for (const auto& callback : probe.callback_records) {
      if (callback.partition_index != partition) {
        continue;
      }
      ++observations;
      EXPECT_EQ(callback.cpu, static_cast<std::int32_t>(selected[partition]));
      EXPECT_EQ(callback.thread, probe.owner_threads[partition]);
    }
    EXPECT_GT(observations, 0U);
  }
  expect_no_legacy_lane_handoffs(counters->snapshot());
#else
  GTEST_SKIP() << "CPU-affinity readback is supported only on Linux";
#endif
}

// REQ: COMPILE-101
// REQ: RUNTIME-041
TEST(RuntimeInlinePartitionExecution, StartsAndRestartsWorkersOutsideCallerAffinity) {
#if defined(__linux__)
  std::thread([] {
    cpu_set_t original;
    ASSERT_EQ(sched_getaffinity(0, sizeof(original), &original), 0);
    struct RestoreAffinity {
      cpu_set_t mask;
      ~RestoreAffinity() { EXPECT_EQ(sched_setaffinity(0, sizeof(mask), &mask), 0); }
    };
    const RestoreAffinity restore{original};
    std::vector<std::uint32_t> cpus;
    for (unsigned cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
      if (CPU_ISSET(cpu, &original)) {
        cpus.push_back(cpu);
      }
    }
    if (cpus.size() < 2U) {
      GTEST_SKIP() << "Requires two permitted CPUs";
    }
    cpu_set_t caller;
    CPU_ZERO(&caller);
    CPU_SET(cpus[0], &caller);
    // Observations use production placement and are read only after session.wait().
    class PlacementEvidence final : public detail::RuntimeSupportThreadControl,
                                    public detail::RuntimeStructuralDiagnosticSink {
     public:
      expected<void, Error> enter(SupportThreadRole role) override {
        if (role == SupportThreadRole::kController &&
            sched_getaffinity(0, sizeof(controller_mask), &controller_mask) != 0) {
          return make_unexpected(Error{ErrorCode::kFailure, "controller mask read failed"});
        }
        return {};
      }
      void publish(std::span<const Diagnostic>) override {}
      void publish_lane_startup_evidence(
          std::span<const detail::ExecutionLaneStartupEvidence> evidence) override {
        lanes.assign(evidence.begin(), evidence.end());
      }
      cpu_set_t controller_mask{};
      std::vector<detail::ExecutionLaneStartupEvidence> lanes;
    };
    RecordProperty("controller_cpu", cpus[0]);
    RecordProperty("second_worker_cpu", cpus[1]);
    for (const std::string_view mode : {"already-narrow", "broad-compile-then-narrow"}) {
      SCOPED_TRACE(mode);
      const cpu_set_t& compile_mask = mode == "already-narrow" ? caller : original;
      ASSERT_EQ(sched_setaffinity(0, sizeof(compile_mask), &compile_mask), 0);
      CompileOptions options;
      options.deployment.bind_support_thread_affinity(SupportThreadRole::kController,
                                                      CPUAffinity{cpus[0]});
      const auto plan = make_partition_cpu_placement_plan(cpus[0], cpus[1], std::move(options));
      ASSERT_TRUE(plan.ok()) << plan.json();
      cpu_set_t after_compile;
      ASSERT_EQ(sched_getaffinity(0, sizeof(after_compile), &after_compile), 0);
      EXPECT_TRUE(CPU_EQUAL(&compile_mask, &after_compile));
      ASSERT_EQ(sched_setaffinity(0, sizeof(caller), &caller), 0);
      for (unsigned run = 0; run < 2; ++run) {
        PartitionRunnerProbeState::reset();
        const auto evidence = std::make_shared<PlacementEvidence>();
        auto observed_plan =
            detail::RuntimeTestPlanAccess::with_support_thread_control(plan, evidence);
        observed_plan =
            detail::RuntimeTestPlanAccess::with_diagnostic_sink(observed_plan, evidence);
        auto session = run_async(observed_plan);
        const bool started = PartitionRunnerProbeState::wait_for_owners(2U);
        session.request_stop();
        session.wait();
        ASSERT_TRUE(started);
        EXPECT_TRUE(session.diagnostics().empty());
        EXPECT_TRUE(CPU_EQUAL(&caller, &evidence->controller_mask));
        ASSERT_EQ(evidence->lanes.size(), plan.execution_lanes().size());
        for (const auto& lane : evidence->lanes) {
          const auto accepted = std::ranges::find(plan.execution_lanes(),
                                                  lane.execution_lane,
                                                  &ExecutionLanePlanRecord::execution_lane);
          ASSERT_NE(accepted, plan.execution_lanes().end());
          if (!lane.applied || !accepted->accepted_placement.affinity) {
            FAIL() << "Missing exact placement readback";
          }
          EXPECT_EQ(lane.applied->affinity.cores, accepted->accepted_placement.affinity->cores);
        }
        const auto probe = PartitionRunnerProbeState::snapshot();
        EXPECT_TRUE(probe.stable_owner_threads);
        std::array<unsigned, 2> observations{};
        for (const auto& callback : probe.callback_records) {
          ASSERT_LT(callback.partition_index, observations.size());
          ++observations[callback.partition_index];
          EXPECT_EQ(callback.cpu, static_cast<std::int32_t>(cpus[callback.partition_index]));
        }
        EXPECT_GT(observations[0], 0U);
        EXPECT_GT(observations[1], 0U);
      }
      cpu_set_t after;
      ASSERT_EQ(sched_getaffinity(0, sizeof(after), &after), 0);
      EXPECT_TRUE(CPU_EQUAL(&caller, &after));
    }
  }).join();
#else
  GTEST_SKIP() << "Linux CPU affinity is required";
#endif
}

// REQ: RUNTIME-041
TEST(RuntimeInlinePartitionExecution, StopBroadcastRetiresEveryPartitionOwner) {
  // Key setup: start two independent periodic partition owners and then request external stop.
  PartitionRunnerProbeState::reset();
  ExecutionPlan compiled = make_partition_stop_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_EQ(state->partition_runners.size(), 2U) << compiled.json();

  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const auto control = std::make_shared<InlinePartitionControllerProbe>();
  compiled = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  compiled = detail::RuntimeTestPlanAccess::with_support_thread_control(compiled, control);
  RunSession session = run_async(compiled);
  const bool owners_started = PartitionRunnerProbeState::wait_for_owners(2U);
  session.request_stop();
  session.wait();

  // Key assertion: the stop broadcast wakes and cleanly retires both stable owners.
  ASSERT_TRUE(owners_started);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  const PartitionRunnerProbeSnapshot probe = PartitionRunnerProbeState::snapshot();
  EXPECT_TRUE(probe.stable_owner_threads);
  EXPECT_TRUE(probe.balanced_callbacks);
  ASSERT_TRUE(probe.owner_seen[0]);
  ASSERT_TRUE(probe.owner_seen[1]);
  EXPECT_NE(probe.owner_threads[0], probe.owner_threads[1]);
  expect_no_legacy_lane_handoffs(counters->snapshot());
}

// REQ: RUNTIME-046
TEST(RuntimeInlinePartitionExecution,
     CallbackCompletionClosesBeforeClaimAndDoesNotDrainQueuedDelivery) {
  // Key setup: request completion after publication while the sink waits before callback claim.
  PartitionRunnerProbeState::reset();
  auto control = std::make_shared<detail::RuntimeTerminationAdmissionTestControl>(
      detail::RuntimeTerminationAdmissionTestControl::Point::kBeforeClaim, 1U, 1U);
  ExecutionPlan compiled = make_partition_completion_delivery_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_EQ(state->process_local_channel_services.size(), 1U);
  EXPECT_EQ(state->process_local_channel_services.front().delivery_kind,
            detail::ProcessLocalChannelDeliveryKind::kDirectRetained);
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  compiled = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  compiled =
      detail::RuntimeTestPlanAccess::with_termination_admission_test_control(compiled, control);
  PartitionRunnerCompletionSource::set_admission_control(control.get());

  RunSession session = run_async(compiled);
  session.wait();
  PartitionRunnerCompletionSource::set_admission_control(nullptr);

  // Key assertion: completion closes admission before the sink claims its queued delivery.
  EXPECT_EQ(control->arrival_count(), 1U);
  EXPECT_EQ(session.termination(), RunTermination::kRequestedCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  const PartitionRunnerProbeSnapshot probe = PartitionRunnerProbeState::snapshot();
  ASSERT_EQ(probe.callback_records.size(), 1U);
  EXPECT_EQ(probe.callback_records.front().operator_id, 1U);
  EXPECT_EQ(probe.callback_records.front().partition_index, 0U);
  EXPECT_TRUE(probe.owner_seen[0]);
  EXPECT_FALSE(probe.owner_seen[1]);
  EXPECT_TRUE(probe.sink_values.empty());
  EXPECT_TRUE(probe.balanced_callbacks);
  const auto snapshot = counters->snapshot();
  const auto count = [&](detail::RuntimePhaseCounter counter) {
    return snapshot[detail::RuntimePhaseCounters::index(counter)];
  };
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kComputeDispatches), 2U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kActivationsCompleted), 1U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kComputeAdmissionsRunClosing), 1U);
}

// REQ: RUNTIME-046
TEST(RuntimeInlinePartitionExecution,
     CallbackCompletionAllowsAlreadyClaimedPartitionCallbacksToRetire) {
  // Key setup: hold two independent completion callbacks immediately after both claims succeed.
  PartitionRunnerProbeState::reset();
  ExecutionPlan compiled = make_partition_completion_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  auto control = std::make_shared<detail::RuntimeTerminationAdmissionTestControl>(
      detail::RuntimeTerminationAdmissionTestControl::Point::kAfterClaim, 2U);
  compiled =
      detail::RuntimeTestPlanAccess::with_termination_admission_test_control(compiled, control);

  RunSession session = run_async(compiled);
  const bool arrived = control->wait_until_arrived();
  if (!arrived) {
    session.request_stop();
  }
  control->release();
  session.wait();

  // Key assertion: both already-claimed callbacks retire before requested completion terminates.
  ASSERT_TRUE(arrived);
  EXPECT_EQ(control->arrival_count(), 2U);
  EXPECT_EQ(session.termination(), RunTermination::kRequestedCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  const PartitionRunnerProbeSnapshot probe = PartitionRunnerProbeState::snapshot();
  ASSERT_EQ(probe.callback_records.size(), 2U);
  EXPECT_TRUE(probe.stable_owner_threads);
  EXPECT_TRUE(probe.balanced_callbacks);
  ASSERT_TRUE(probe.owner_seen[0]);
  ASSERT_TRUE(probe.owner_seen[1]);
  EXPECT_NE(probe.owner_threads[0], probe.owner_threads[1]);
  std::array<std::uint32_t, 2U> operator_ids{
      probe.callback_records[0].operator_id,
      probe.callback_records[1].operator_id,
  };
  std::ranges::sort(operator_ids);
  EXPECT_EQ(operator_ids, (std::array<std::uint32_t, 2U>{1U, 2U}));
  std::array<std::size_t, 2U> partition_indices{
      probe.callback_records[0].partition_index,
      probe.callback_records[1].partition_index,
  };
  std::ranges::sort(partition_indices);
  EXPECT_EQ(partition_indices, (std::array<std::size_t, 2U>{0U, 1U}));
  for (const auto& callback : probe.callback_records) {
    ASSERT_LT(callback.partition_index, probe.owner_threads.size());
    EXPECT_EQ(callback.thread, probe.owner_threads[callback.partition_index]);
  }
}

// REQ: RUNTIME-041
TEST(RuntimeInlinePartitionExecution,
     ImmediateStopBroadcastUsesPrepublishedPartitionWakeInventory) {
  // Key setup: block the controller before graph reconstruction, then request immediate stop.
  ExecutionPlan compiled = make_partition_stop_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_EQ(state->partition_runners.size(), 2U) << compiled.json();

  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const auto startup_gate = std::make_shared<PartitionStartupGate>();
  compiled = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  compiled = detail::RuntimeTestPlanAccess::with_support_thread_control(compiled, startup_gate);
  RunSession session = run_async(compiled);
  const bool controller_entered = startup_gate->wait_until_controller_entered();
  if (!controller_entered) {
    startup_gate->release_controller();
    session.request_stop();
    session.wait();
  }
  ASSERT_TRUE(controller_entered);

  // The controller is blocked before Graph reconstruction and runner-state materialization.
  // An external stop must nevertheless observe and signal the complete immutable wake inventory.
  session.request_stop();
  const detail::RuntimePhaseCounters::Snapshot stopped = counters->snapshot();
  // Key assertion: the prepublished inventory wakes both owners before materialization finishes.
  EXPECT_EQ(stopped[detail::RuntimePhaseCounters::index(
                detail::RuntimePhaseCounter::kPartitionRunnerWakeSignals)],
            2U);

  startup_gate->release_controller();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
}

// REQ: RUNTIME-041
TEST(RuntimeInlinePartitionExecution, CallbackFailureWakesAndRetiresEveryPartitionOwner) {
  // Key setup: synchronize two periodic partition callbacks and fail one of their stable owners.
  PartitionRunnerProbeState::reset();
  ExecutionPlan compiled = make_partition_failure_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_EQ(state->partition_runners.size(), 2U) << compiled.json();

  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const auto control = std::make_shared<InlinePartitionControllerProbe>();
  compiled = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  compiled = detail::RuntimeTestPlanAccess::with_support_thread_control(compiled, control);
  RunSession session = run_async(compiled);
  const bool terminal = wait_for_terminal(session);
  if (!terminal) {
    session.request_stop();
  }
  session.wait();

  // Key assertion: the failure wakes both owners, preserves ownership, and terminates the run.
  ASSERT_TRUE(terminal);
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_FALSE(session.diagnostics().empty());
  EXPECT_TRUE(std::ranges::any_of(session.diagnostics(), [](const auto& diagnostic) {
    return diagnostic.token == "RUNTIME_OPERATOR_CALLBACK_FAILED";
  }));
  const PartitionRunnerProbeSnapshot probe = PartitionRunnerProbeState::snapshot();
  EXPECT_TRUE(probe.stable_owner_threads);
  EXPECT_TRUE(probe.balanced_callbacks);
  ASSERT_TRUE(probe.owner_seen[0]);
  ASSERT_TRUE(probe.owner_seen[1]);
  EXPECT_NE(probe.owner_threads[0], probe.owner_threads[1]);
  // The peer's 1 ms clock may fire again before the failure broadcast reaches its owner, most
  // often under load, so later peer callbacks are legitimate. The failing owner runs exactly once.
  ASSERT_GE(probe.callback_records.size(), 2U);
  std::array<std::uint32_t, 2U> operators{
      probe.callback_records[0].operator_id,
      probe.callback_records[1].operator_id,
  };
  std::ranges::sort(operators);
  EXPECT_EQ(operators, (std::array<std::uint32_t, 2U>{1U, 2U}));
  EXPECT_EQ(
      std::ranges::count(probe.callback_records, 1U, &PartitionRunnerCallbackRecord::operator_id),
      1);
  expect_no_legacy_lane_handoffs(counters->snapshot());
}

// REQ: RUNTIME-041
TEST(RuntimeInlinePartitionExecution,
     RestartControlEpochQuiescesEveryPartitionOwnerBeforeLifecycleCallbacks) {
  // Key setup: request target self-restart while a second partition is held inside its callback.
  PartitionControlProbeState::reset();
  ExecutionPlan compiled = make_partition_restart_control_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_EQ(state->partition_runners.size(), 2U) << compiled.json();
  ASSERT_EQ(std::ranges::count(compiled.lifecycle_restart_targets(),
                               true,
                               &LifecycleRestartTargetPlanRecord::self_allowed),
            1)
      << compiled.json();

  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  compiled = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  RunSession session = run_async(compiled);
  const bool restart_requested = PartitionControlProbeState::wait_until_restart_requested();
  if (!restart_requested) {
    PartitionControlProbeState::release_peer();
    session.request_stop();
    session.wait();
  }
  ASSERT_TRUE(restart_requested);

  const PartitionControlProbeSnapshot held = PartitionControlProbeState::snapshot();
  ASSERT_TRUE(held.restart_requested);
  ASSERT_FALSE(held.restart_completed);
  ASSERT_EQ(held.lifecycle_records.size(), 2U);
  EXPECT_TRUE(std::ranges::all_of(held.lifecycle_records, [](const auto& record) {
    return record.stage == LifecycleStage::kStart;
  }));
  EXPECT_FALSE(held.lifecycle_before_peer_quiescence);
  EXPECT_FALSE(held.callback_crossed_control_epoch);

  PartitionControlProbeState::release_peer();
  const bool resumed = PartitionControlProbeState::wait_until_resumed();
  session.request_stop();
  session.wait();

  // Key assertion: restart lifecycle callbacks occur only after every owner crosses quiescence.
  ASSERT_TRUE(resumed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  const PartitionControlProbeSnapshot probe = PartitionControlProbeState::snapshot();
  EXPECT_TRUE(probe.stable_owner_threads);
  EXPECT_FALSE(probe.barrier_timed_out);
  EXPECT_TRUE(probe.restart_requested);
  EXPECT_TRUE(probe.restart_completed);
  EXPECT_FALSE(probe.lifecycle_before_peer_quiescence);
  EXPECT_FALSE(probe.callback_crossed_control_epoch);
  EXPECT_NE(probe.owner_threads[0], probe.owner_threads[1]);
  EXPECT_GE(probe.compute_counts[0], 2U);
  EXPECT_GE(probe.compute_counts[1], 2U);

  std::vector<std::pair<std::uint32_t, LifecycleStage>> lifecycle_order;
  lifecycle_order.reserve(probe.lifecycle_records.size());
  for (const auto& record : probe.lifecycle_records) {
    ASSERT_LT(record.partition_index, probe.owner_threads.size());
    EXPECT_EQ(record.thread, probe.owner_threads[record.partition_index]);
    lifecycle_order.emplace_back(record.operator_id, record.stage);
  }
  EXPECT_EQ(lifecycle_order,
            (std::vector<std::pair<std::uint32_t, LifecycleStage>>{
                {2U, LifecycleStage::kStart},
                {1U, LifecycleStage::kStart},
                {1U, LifecycleStage::kStop},
                {1U, LifecycleStage::kStart},
                {1U, LifecycleStage::kStop},
                {2U, LifecycleStage::kStop},
            }));

  const detail::RuntimePhaseCounters::Snapshot phase_counters = counters->snapshot();
  const auto count = [&](detail::RuntimePhaseCounter counter) {
    return phase_counters[detail::RuntimePhaseCounters::index(counter)];
  };
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kLaneDispatches),
            count(detail::RuntimePhaseCounter::kLaneLifecycleDispatches));
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kLaneCompletionsObserved),
            count(detail::RuntimePhaseCounter::kLaneLifecycleDispatches));
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kLaneComputeDispatches), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kLaneDispatcherDispatches), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kLaneUnknownDispatches), 0U);
  EXPECT_GE(count(detail::RuntimePhaseCounter::kPartitionControlBarriers), 2U);
}

}  // namespace
}  // namespace holoscan
