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
#include <limits>
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
#include <holoscan/core/predicate.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
// Provides generated owning types and schema-trait specializations.
#include <holoscan/schema/sdk_schemas.hpp>  // NOLINT(misc-include-cleaner)
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/run.hpp>
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "core/domain/tensor_access.hpp"
#include "runtime/runtime_cuda_execution_resources.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_support_thread_control.hpp"
#include "runtime/runtime_test_access.hpp"

namespace holoscan {

// REQ: RUNTIME-040
// REQ: RUNTIME-041

struct InlinePartitionPod {
  std::uint32_t value{};
};

template <>
struct ChannelTypeIdentity<InlinePartitionPod> {
  static constexpr std::string_view name = "nvidia.holoscan.test.InlinePartitionPod";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<InlinePartitionPod> {
  static constexpr auto fields = pod_fields(
      pod_field("value", &InlinePartitionPod::value, offsetof(InlinePartitionPod, value)));
};

struct InlinePartitionParams {
  std::uint32_t value{1U};
};

HOLOSCAN_PARAM_SCHEMA(InlinePartitionParams,
                      param_schema("nvidia.holoscan.test.InlinePartitionParams", 1),
                      field<&InlinePartitionParams::value>("value"));

struct InlinePartitionRuntimeParams {
  std::uint32_t value{1U};
};

HOLOSCAN_PARAM_SCHEMA(InlinePartitionRuntimeParams,
                      param_schema("nvidia.holoscan.test.InlinePartitionRuntimeParams", 1),
                      field<&InlinePartitionRuntimeParams::value>("value").runtime_mutable());

struct InlineTemporalParams {
  std::uint32_t value{200U};
  std::chrono::nanoseconds max_age{std::chrono::seconds{1}};
  std::chrono::nanoseconds deadline{std::chrono::milliseconds{1}};
};

HOLOSCAN_PARAM_SCHEMA(InlineTemporalParams,
                      param_schema("nvidia.holoscan.test.InlineTemporalParams", 1),
                      field<&InlineTemporalParams::value>("value").runtime_mutable(),
                      field<&InlineTemporalParams::max_age>("max_age").runtime_mutable().range(
                          std::chrono::milliseconds{1}, std::chrono::seconds{10}),
                      field<&InlineTemporalParams::deadline>("deadline")
                          .runtime_mutable()
                          .range(std::chrono::milliseconds{1}, std::chrono::seconds{10}));

namespace {

using namespace std::chrono_literals;

class ScopedCUDADeviceRestore final {
 public:
  explicit ScopedCUDADeviceRestore(std::int32_t device) noexcept : device_(device) {}

  ScopedCUDADeviceRestore(const ScopedCUDADeviceRestore&) = delete;
  ScopedCUDADeviceRestore& operator=(const ScopedCUDADeviceRestore&) = delete;

  ~ScopedCUDADeviceRestore() {
    if (device_ >= 0) {
      static_cast<void>(cudaSetDevice(device_));
    }
  }

 private:
  std::int32_t device_{-1};
};

[[nodiscard]] std::int32_t current_cpu() noexcept {
#if defined(__linux__)
  return sched_getcpu();
#else
  return -1;
#endif
}

struct InlinePartitionComputeRecord {
  std::uint32_t operator_id{};
  std::uint32_t value{};
  TagTime tag_time{};
  std::thread::id thread;
};

struct InlinePartitionLifecycleRecord {
  std::uint32_t operator_id{};
  LifecycleStage stage{LifecycleStage::kConfigure};
  std::thread::id thread;
};

struct InlinePartitionParamComputeRecord {
  std::uint32_t operator_id{};
  std::uint32_t value{};
  ParamGeneration generation{};
  const void* address{};
  bool repeated_address_stable{};
  bool repeated_generation_stable{};
  ActivationReason activation_reason{ActivationReason::kNominal};
  OnViolation activation_policy{OnViolation::kEmitWitness};
  std::thread::id thread;
};

struct InlinePartitionParamLifecycleRecord {
  std::uint32_t operator_id{};
  LifecycleStage stage{LifecycleStage::kConfigure};
  std::uint32_t value{};
  ParamGeneration generation{};
  const void* address{};
  bool repeated_address_stable{};
  bool repeated_generation_stable{};
  std::thread::id thread;
};

struct InlineTemporalPredicateRecord {
  std::uint32_t threshold{};
  std::uint32_t driver{};
  std::uint32_t side{};
  std::thread::id thread;
};

struct InlineTensorObservation {
  MemoryKind memory_kind{MemoryKind::kHost};
  const void* source_pointer{};
  const void* relay_pointer{};
  cudaStream_t source_stream{};
  cudaStream_t relay_stream{};
  cudaStream_t readiness_stream{};
  std::int32_t source_device{-1};
  std::int32_t relay_device{-1};
  std::int32_t stop_device{-1};
  std::uint8_t value{};
  bool source_recorded{};
  bool relay_recorded{};
  bool stop_device_recorded{};
};

class InlineTensorProbeState {
 public:
  static void reset(MemoryKind memory_kind) {
    const std::lock_guard lock(mutex_);
    observation_ = InlineTensorObservation{.memory_kind = memory_kind};
    retained_tensor_.reset();
  }

  static void record_source(const void* pointer, cudaStream_t stream, std::int32_t device) {
    const std::lock_guard lock(mutex_);
    observation_.source_pointer = pointer;
    observation_.source_stream = stream;
    observation_.source_device = device;
    observation_.source_recorded = true;
  }

  static void record_relay(const void* pointer, cudaStream_t stream, cudaStream_t readiness_stream,
                           std::int32_t device, std::uint8_t value) {
    const std::lock_guard lock(mutex_);
    observation_.relay_pointer = pointer;
    observation_.relay_stream = stream;
    observation_.readiness_stream = readiness_stream;
    observation_.relay_device = device;
    observation_.value = value;
    observation_.relay_recorded = true;
  }

  static void retain_tensor(const Tensor& tensor) {
    const std::lock_guard lock(mutex_);
    retained_tensor_ = tensor;
  }

  [[nodiscard]] static bool has_retained_tensor() {
    const std::lock_guard lock(mutex_);
    return retained_tensor_.has_value();
  }

  static void release_retained_tensor() {
    const std::lock_guard lock(mutex_);
    retained_tensor_.reset();
  }

  static void record_stop_device(std::int32_t device) {
    const std::lock_guard lock(mutex_);
    observation_.stop_device = device;
    observation_.stop_device_recorded = true;
  }

  [[nodiscard]] static InlineTensorObservation observation() {
    const std::lock_guard lock(mutex_);
    return observation_;
  }

 private:
  static inline std::mutex mutex_;
  static inline InlineTensorObservation observation_;
  static inline std::optional<Tensor> retained_tensor_;
};

class InlinePartitionProbeState {
 public:
  static void reset() {
    const std::lock_guard lock(mutex_);
    compute_records_.clear();
    lifecycle_records_.clear();
    param_compute_records_.clear();
    param_lifecycle_records_.clear();
    temporal_predicate_records_.clear();
    sink_values_.clear();
    schema_value_.reset();
    first_runtime_compute_entered_ = false;
    release_first_runtime_compute_ = false;
  }

  static void record_compute(std::uint32_t operator_id, std::uint32_t value,
                             TagTime tag_time = {}) {
    const std::lock_guard lock(mutex_);
    compute_records_.push_back(InlinePartitionComputeRecord{
        .operator_id = operator_id,
        .value = value,
        .tag_time = tag_time,
        .thread = std::this_thread::get_id(),
    });
  }

  static void record_lifecycle(std::uint32_t operator_id, LifecycleStage stage) {
    const std::lock_guard lock(mutex_);
    lifecycle_records_.push_back(InlinePartitionLifecycleRecord{
        .operator_id = operator_id,
        .stage = stage,
        .thread = std::this_thread::get_id(),
    });
  }

  template <typename Params>
  static void record_param_compute(std::uint32_t operator_id, ExecutionContext& context) {
    const auto* const first = &context.params<Params>();
    const ParamGeneration first_generation = context.params_generation();
    const auto* const second = &context.params<Params>();
    const std::lock_guard lock(mutex_);
    param_compute_records_.push_back(InlinePartitionParamComputeRecord{
        .operator_id = operator_id,
        .value = first->value,
        .generation = first_generation,
        .address = first,
        .repeated_address_stable = first == second,
        .repeated_generation_stable = first_generation == context.params_generation(),
        .activation_reason = context.activation().reason,
        .activation_policy = context.activation().policy,
        .thread = std::this_thread::get_id(),
    });
    cv_.notify_all();
  }

  template <typename Params>
  [[nodiscard]] static bool record_runtime_param_compute(std::uint32_t operator_id,
                                                         ExecutionContext& context, bool hold) {
    const auto* const first = &context.params<Params>();
    const ParamGeneration first_generation = context.params_generation();
    if (hold) {
      std::unique_lock lock(mutex_);
      first_runtime_compute_entered_ = true;
      cv_.notify_all();
      if (!cv_.wait_for(lock, 5s, [] { return release_first_runtime_compute_; })) {
        return false;
      }
    }
    const auto* const second = &context.params<Params>();
    const std::lock_guard lock(mutex_);
    param_compute_records_.push_back(InlinePartitionParamComputeRecord{
        .operator_id = operator_id,
        .value = first->value,
        .generation = first_generation,
        .address = first,
        .repeated_address_stable = first == second,
        .repeated_generation_stable = first_generation == context.params_generation(),
        .activation_reason = context.activation().reason,
        .activation_policy = context.activation().policy,
        .thread = std::this_thread::get_id(),
    });
    cv_.notify_all();
    return true;
  }

  template <typename Params>
  static void record_param_lifecycle(std::uint32_t operator_id, LifecycleStage stage,
                                     LifecycleContext& context) {
    const auto* const first = &context.params<Params>();
    const ParamGeneration first_generation = context.params_generation();
    const auto* const second = &context.params<Params>();
    const std::lock_guard lock(mutex_);
    param_lifecycle_records_.push_back(InlinePartitionParamLifecycleRecord{
        .operator_id = operator_id,
        .stage = stage,
        .value = first->value,
        .generation = first_generation,
        .address = first,
        .repeated_address_stable = first == second,
        .repeated_generation_stable = first_generation == context.params_generation(),
        .thread = std::this_thread::get_id(),
    });
  }

  static void record_sink(std::uint32_t value) {
    {
      const std::lock_guard lock(mutex_);
      sink_values_.push_back(value);
    }
    cv_.notify_all();
  }

  static void record_schema(std::string value) {
    {
      const std::lock_guard lock(mutex_);
      schema_value_ = std::move(value);
    }
    cv_.notify_all();
  }

  static void record_temporal_predicate(std::uint32_t threshold, std::uint32_t driver,
                                        std::uint32_t side) {
    const std::lock_guard lock(mutex_);
    temporal_predicate_records_.push_back(InlineTemporalPredicateRecord{
        .threshold = threshold,
        .driver = driver,
        .side = side,
        .thread = std::this_thread::get_id(),
    });
  }

  [[nodiscard]] static bool wait_for_first_runtime_compute() {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, 5s, [] { return first_runtime_compute_entered_; });
  }

  static void release_first_runtime_compute() {
    {
      const std::lock_guard lock(mutex_);
      release_first_runtime_compute_ = true;
    }
    cv_.notify_all();
  }

  [[nodiscard]] static bool block_first_runtime_compute() {
    std::unique_lock lock(mutex_);
    first_runtime_compute_entered_ = true;
    cv_.notify_all();
    return cv_.wait_for(lock, 5s, [] { return release_first_runtime_compute_; });
  }

  [[nodiscard]] static bool wait_for_sinks(std::size_t count) {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, 5s, [count] { return sink_values_.size() >= count; });
  }

  [[nodiscard]] static bool wait_for_schema() {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, 5s, [] { return schema_value_.has_value(); });
  }

  [[nodiscard]] static std::vector<InlinePartitionComputeRecord> compute_records() {
    const std::lock_guard lock(mutex_);
    return compute_records_;
  }

  [[nodiscard]] static std::vector<InlinePartitionLifecycleRecord> lifecycle_records() {
    const std::lock_guard lock(mutex_);
    return lifecycle_records_;
  }

  [[nodiscard]] static std::vector<InlinePartitionParamComputeRecord> param_compute_records() {
    const std::lock_guard lock(mutex_);
    return param_compute_records_;
  }

  [[nodiscard]] static std::vector<InlinePartitionParamLifecycleRecord> param_lifecycle_records() {
    const std::lock_guard lock(mutex_);
    return param_lifecycle_records_;
  }

  [[nodiscard]] static std::vector<InlineTemporalPredicateRecord> temporal_predicate_records() {
    const std::lock_guard lock(mutex_);
    return temporal_predicate_records_;
  }

  [[nodiscard]] static std::vector<std::uint32_t> sink_values() {
    const std::lock_guard lock(mutex_);
    return sink_values_;
  }

  [[nodiscard]] static std::optional<std::string> schema_value() {
    const std::lock_guard lock(mutex_);
    return schema_value_;
  }

 private:
  static inline std::mutex mutex_;
  static inline std::condition_variable cv_;
  static inline std::vector<InlinePartitionComputeRecord> compute_records_;
  static inline std::vector<InlinePartitionLifecycleRecord> lifecycle_records_;
  static inline std::vector<InlinePartitionParamComputeRecord> param_compute_records_;
  static inline std::vector<InlinePartitionParamLifecycleRecord> param_lifecycle_records_;
  static inline std::vector<InlineTemporalPredicateRecord> temporal_predicate_records_;
  static inline std::vector<std::uint32_t> sink_values_;
  static inline std::optional<std::string> schema_value_;
  static inline bool first_runtime_compute_entered_{};
  static inline bool release_first_runtime_compute_{};
};

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

class InlinePartitionControllerProbe final : public detail::RuntimeSupportThreadControl {
 public:
  [[nodiscard]] expected<void, Error> enter(SupportThreadRole role) override {
    if (role == SupportThreadRole::kController) {
      const std::lock_guard lock(mutex_);
      controller_thread_ = std::this_thread::get_id();
    }
    return {};
  }

  [[nodiscard]] std::optional<std::thread::id> controller_thread() const {
    const std::lock_guard lock(mutex_);
    return controller_thread_;
  }

 private:
  mutable std::mutex mutex_;
  std::optional<std::thread::id> controller_thread_;
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

template <typename Derived>
class InlinePartitionLifecycle {
 protected:
  explicit InlinePartitionLifecycle(std::uint32_t operator_id, bool record_stop_device = false)
      : operator_id_(operator_id), record_stop_device_(record_stop_device) {}

  void setup_lifecycle(OperatorSpec& spec) {
    using Callback = LifecycleStatus (Derived::*)(LifecycleContext&) noexcept;
    spec.lifecycle()
        .stage(LifecycleStage::kStart,
               static_cast<Callback>(&InlinePartitionLifecycle::start_stage))
        .stage(LifecycleStage::kStop, static_cast<Callback>(&InlinePartitionLifecycle::stop_stage));
  }

  LifecycleStatus start_stage(LifecycleContext&) noexcept {
    InlinePartitionProbeState::record_lifecycle(operator_id_, LifecycleStage::kStart);
    return LifecycleStatus::kOk;
  }

  LifecycleStatus stop_stage(LifecycleContext&) noexcept {
    if (record_stop_device_) {
      std::int32_t device = -1;
      if (cudaGetDevice(&device) == cudaSuccess) {
        InlineTensorProbeState::record_stop_device(device);
      }
    }
    InlinePartitionProbeState::record_lifecycle(operator_id_, LifecycleStage::kStop);
    return LifecycleStatus::kOk;
  }

  std::uint32_t operator_id_{};
  bool record_stop_device_{};
};

// Emits bounded POD samples and optionally requests a self-restart before its first emission.
// Representative test: RunsComputeAndLifecycleCallbacksOnTheControllerWithoutHandoffs.
class InlinePodSource final : public Operator<>, public InlinePartitionLifecycle<InlinePodSource> {
 public:
  InlinePodSource(std::uint32_t operator_id, std::uint32_t value, bool restart_before_emit = false,
                  std::size_t emission_limit = 1U, std::chrono::nanoseconds period = 30s)
      : InlinePartitionLifecycle<InlinePodSource>(operator_id),
        value_(value),
        emission_limit_(emission_limit),
        period_(period),
        restart_before_emit_(restart_before_emit) {}

  void setup(OperatorSpec& spec) override {
    setup_lifecycle(spec);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = period_});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (emitted_count_ >= emission_limit_) {
      return {};
    }
    InlinePartitionProbeState::record_compute(operator_id_, value_, context.tag_time());
    if (restart_before_emit_ && !restart_requested_) {
      restart_requested_ = true;
      auto receipt =
          context.request_restart(LifecycleRestartOptions{.from_stage = LifecycleStage::kStart});
      if (!receipt) {
        return make_unexpected(std::move(receipt.error()));
      }
      return {};
    }
    ++emitted_count_;
    return output.emit(InlinePartitionPod{.value = value_});
  }

  Output<InlinePartitionPod> output;

 private:
  std::uint32_t value_{};
  std::size_t emission_limit_{};
  std::chrono::nanoseconds period_{};
  bool restart_before_emit_{};
  bool restart_requested_{};
  std::size_t emitted_count_{};
};

// Transforms POD samples while recording lifecycle and optional callback-failure behavior.
// Representative test: RunsComputeAndLifecycleCallbacksOnTheControllerWithoutHandoffs.
class InlinePodRelay final : public Operator<>, public InlinePartitionLifecycle<InlinePodRelay> {
 public:
  InlinePodRelay(std::uint32_t operator_id, std::uint32_t increment, bool fail_compute = false)
      : InlinePartitionLifecycle<InlinePodRelay>(operator_id),
        increment_(increment),
        fail_compute_(fail_compute) {}

  void setup(OperatorSpec& spec) override {
    setup_lifecycle(spec);
    spec.input(input, "input").queue_depth(2U);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

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
    const std::uint32_t value = sample->data.value + increment_;
    InlinePartitionProbeState::record_compute(operator_id_, value, context.tag_time());
    if (fail_compute_) {
      return make_unexpected(Error{ErrorCode::kFailure, "INLINE_CALLBACK_FAILURE"});
    }
    return output.emit(InlinePartitionPod{.value = value});
  }

  Input<InlinePartitionPod> input;
  Output<InlinePartitionPod> output;

 private:
  std::uint32_t increment_{};
  bool fail_compute_{};
};

// Publishes one runtime-pooled raw Tensor in the requested memory kind.
// Representative test: InlinePayloadParity.HostPreservesBackingMetadataAndControllerOwnership.
template <MemoryKind Kind>
class InlineTensorSource final : public Operator<>,
                                 public InlinePartitionLifecycle<InlineTensorSource<Kind>> {
 public:
  explicit InlineTensorSource(std::int32_t ambient_device_after_emit = -1)
      : InlinePartitionLifecycle<InlineTensorSource<Kind>>(1U),
        ambient_device_after_emit_(ambient_device_after_emit) {}

  void setup(OperatorSpec& spec) override {
    this->setup_lifecycle(spec);
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = Kind,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 1U},
            .bounds = tensor_bounds(1U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 30s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    constexpr std::array<std::int64_t, 1U> kShape{1};
    constexpr std::uint8_t kValue = 0x5AU;
    InlinePartitionProbeState::record_compute(this->operator_id_, kValue);
    if (emitted_) {
      return {};
    }

    auto* const callback_stream = context.cuda_stream().get();
    // Device-backed template specializations populate this through cudaGetDevice().
    std::int32_t callback_device = -1;  // NOLINT(misc-const-correctness)
    void* pointer = nullptr;
    {
      auto loan = output.allocate_tensor(TensorLoanRequest{
          .shape = kShape,
          .dtype = DLDataType{kDLUInt, 8U, 1U},
      });
      if (!loan) {
        return make_unexpected(std::move(loan.error()));
      }

      if constexpr (Kind == MemoryKind::kCudaDevice || Kind == MemoryKind::kCudaManaged) {
        if (callback_stream == nullptr || cudaGetDevice(&callback_device) != cudaSuccess) {
          return make_unexpected(Error{ErrorCode::kFailure});
        }
        auto writer = loan->write(context.cuda_stream());
        if (!writer) {
          return make_unexpected(std::move(writer.error()));
        }
        pointer = writer->data();
        if (cudaMemsetAsync(pointer, kValue, 1U, callback_stream) != cudaSuccess) {
          return make_unexpected(Error{ErrorCode::kFailure});
        }
        auto committed = std::move(*writer).commit();
        if (!committed) {
          return make_unexpected(std::move(committed.error()));
        }
      } else {
        auto writer = loan->write_host();
        if (!writer || writer->data_as<std::uint8_t>() == nullptr) {
          return make_unexpected(writer ? Error{ErrorCode::kFailure} : std::move(writer.error()));
        }
        pointer = writer->data();
        *writer->data_as<std::uint8_t>() = kValue;
        auto committed = std::move(*writer).commit();
        if (!committed) {
          return make_unexpected(std::move(committed.error()));
        }
      }

      InlineTensorProbeState::record_source(pointer, callback_stream, callback_device);
      auto published = output.emit(std::move(*loan));
      if (!published) {
        return make_unexpected(std::move(published.error()));
      }
    }
    emitted_ = true;

    if constexpr (Kind == MemoryKind::kCudaDevice || Kind == MemoryKind::kCudaManaged) {
      if (ambient_device_after_emit_ >= 0 &&
          cudaSetDevice(ambient_device_after_emit_) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
    }
    return {};
  }

  Output<Tensor> output;

 private:
  std::int32_t ambient_device_after_emit_{-1};
  bool emitted_{};
};

// Reads a raw Tensor, captures backing/device evidence, and emits a POD result.
// Representative test: InlinePayloadParity.HostPreservesBackingMetadataAndControllerOwnership.
template <MemoryKind Kind>
class InlineTensorRelay final : public Operator<>,
                                public InlinePartitionLifecycle<InlineTensorRelay<Kind>> {
 public:
  explicit InlineTensorRelay(std::uint32_t operator_id = 2U, bool fail_compute = false,
                             bool retain_tensor = false,
                             std::int32_t ambient_device_before_failure = -1)
      : InlinePartitionLifecycle<InlineTensorRelay<Kind>>(operator_id,
                                                          ambient_device_before_failure >= 0),
        fail_compute_(fail_compute),
        retain_tensor_(retain_tensor),
        ambient_device_before_failure_(ambient_device_before_failure) {}

  void setup(OperatorSpec& spec) override {
    this->setup_lifecycle(spec);
    spec.input(input, "input")
        .queue_depth(2U)
        .expects_tensor(TensorInputSpec{
            .representation = {.memory_kind = Kind,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 1U},
        });
    spec.output(output, "output").max_emits_per_compute(1U);
  }

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

    // Device-backed template specializations populate this through cudaGetDevice().
    std::int32_t callback_device = -1;  // NOLINT(misc-const-correctness)
    if constexpr (Kind == MemoryKind::kCudaDevice || Kind == MemoryKind::kCudaManaged) {
      if (cudaGetDevice(&callback_device) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
    }
    auto* const callback_stream = context.cuda_stream().get();
    auto* const readiness_stream = detail::TensorAccess::readiness_stream(sample->data);
    std::uint8_t value{};
    auto copied = sample->data.copy_to_host(&value, sizeof(value), context.cuda_stream());
    if (!copied) {
      return make_unexpected(std::move(copied.error()));
    }

    InlineTensorProbeState::record_relay(
        sample->data.data(), callback_stream, readiness_stream, callback_device, value);
    if (retain_tensor_) {
      InlineTensorProbeState::retain_tensor(sample->data);
    }
    InlinePartitionProbeState::record_compute(this->operator_id_, value);
    if (fail_compute_) {
      if constexpr (Kind == MemoryKind::kCudaDevice || Kind == MemoryKind::kCudaManaged) {
        if (ambient_device_before_failure_ >= 0 &&
            cudaSetDevice(ambient_device_before_failure_) != cudaSuccess) {
          return make_unexpected(Error{ErrorCode::kFailure});
        }
      }
      return make_unexpected(Error{ErrorCode::kFailure, "INLINE_TENSOR_CALLBACK_FAILURE"});
    }
    return output.emit(InlinePartitionPod{.value = value});
  }

  Input<Tensor> input;
  Output<InlinePartitionPod> output;

 private:
  bool fail_compute_{};
  bool retain_tensor_{};
  std::int32_t ambient_device_before_failure_{-1};
};

constexpr std::array<std::int64_t, 1U> kCausalTensorShape{1};
constexpr TensorRepresentation kCausalTensorRepresentation{
    .memory_kind = MemoryKind::kHost,
    .dtype = DLDataType{kDLUInt, 32U, 1U},
    .rank = 1U,
    .strides = TensorStridePolicy::kContiguousOnly,
};

// Publishes numbered raw-Tensor frames from an effectively continuous clock.
// Representative test: RawTensorPassThroughContinuesCausalFrameBeforeNextClockTick.
class CausalTensorSource final : public Operator<> {
 public:
  explicit CausalTensorSource(std::size_t frame_count) : frame_count_(frame_count) {}

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = kCausalTensorRepresentation,
            .bounds = tensor_bounds(kCausalTensorShape),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ns});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (next_frame_ == frame_count_) {
      return {};
    }
    auto loan = output.allocate_tensor(kCausalTensorShape);
    if (!loan) {
      return make_unexpected(std::move(loan.error()));
    }
    auto committed = loan->write_host_and_commit(
        [frame = static_cast<std::uint32_t>(next_frame_)](const TensorOutputWriteGuard& writer) {
          *writer.data_as<std::uint32_t>() = frame;
        });
    if (!committed) {
      return make_unexpected(std::move(committed.error()));
    }
    auto emitted = output.emit(std::move(*loan));
    if (!emitted) {
      return emitted;
    }
    InlinePartitionProbeState::record_compute(
        1U, static_cast<std::uint32_t>(next_frame_), context.tag_time());
    ++next_frame_;
    return {};
  }

  Output<Tensor> output;

 private:
  std::size_t frame_count_{};
  std::size_t next_frame_{};
};

// Forwards each raw Tensor without replacing its backing and records the causal middle stage.
// Representative test: RawTensorPassThroughContinuesCausalFrameBeforeNextClockTick.
class CausalTensorPassThrough final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{.representation = kCausalTensorRepresentation});
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = kCausalTensorRepresentation,
            .storage = TensorOutputStorage::kPassThrough,
        });
  }

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
    auto synchronized = sample->data.synchronize_for_host();
    if (!synchronized) {
      return make_unexpected(std::move(synchronized.error()));
    }
    const std::uint32_t frame = *sample->data.data_as<std::uint32_t>();
    auto emitted = output.emit(std::move(sample->data));
    if (!emitted) {
      return emitted;
    }
    InlinePartitionProbeState::record_compute(2U, frame, context.tag_time());
    return {};
  }

  Input<Tensor> input;
  Output<Tensor> output;
};

// Consumes numbered raw Tensors and signals completion after recording the terminal stage.
// Representative test: RawTensorPassThroughContinuesCausalFrameBeforeNextClockTick.
class CausalTensorSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{.representation = kCausalTensorRepresentation});
  }

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
    auto synchronized = sample->data.synchronize_for_host();
    if (!synchronized) {
      return make_unexpected(std::move(synchronized.error()));
    }
    const std::uint32_t frame = *sample->data.data_as<std::uint32_t>();
    InlinePartitionProbeState::record_compute(3U, frame, context.tag_time());
    InlinePartitionProbeState::record_sink(frame);
    return {};
  }

  Input<Tensor> input;
};

// Publishes numbered one-Tensor schema samples from an effectively continuous clock.
// Representative test: TensorPacketPassThroughContinuesCausalFrameBeforeNextClockTick.
class CausalTensorPacketSource final : public Operator<> {
 public:
  explicit CausalTensorPacketSource(std::size_t frame_count) : frame_count_(frame_count) {}

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = kCausalTensorRepresentation,
            .bounds = tensor_bounds(kCausalTensorShape),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ns});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (next_frame_ == frame_count_) {
      return {};
    }
    auto loan = output.allocate_tensor(kCausalTensorShape);
    if (!loan) {
      return make_unexpected(std::move(loan.error()));
    }
    auto committed = loan->write_host_and_commit(
        [frame = static_cast<std::uint32_t>(next_frame_)](const TensorOutputWriteGuard& writer) {
          *writer.data_as<std::uint32_t>() = frame;
        });
    if (!committed) {
      return make_unexpected(std::move(committed.error()));
    }
    schema::TensorPacketT descriptor{};
    descriptor.sequence = next_frame_;
    descriptor.source = "causal-tensor-packet";
    auto emitted =
        output.emit_tensor(std::move(*loan), descriptor, EmitOptions{.frame_id = next_frame_});
    if (!emitted) {
      return emitted;
    }
    InlinePartitionProbeState::record_compute(
        1U, static_cast<std::uint32_t>(next_frame_), context.tag_time());
    ++next_frame_;
    return {};
  }

  Output<schema::TensorPacketT> output;

 private:
  std::size_t frame_count_{};
  std::size_t next_frame_{};
};

// Forwards an owning TensorPacket sample while preserving its Tensor and semantic metadata.
// Representative test: TensorPacketPassThroughContinuesCausalFrameBeforeNextClockTick.
class CausalTensorPacketPassThrough final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{.representation = kCausalTensorRepresentation});
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = kCausalTensorRepresentation,
            .storage = TensorOutputStorage::kPassThrough,
        });
  }

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
    if (!sample->data.data) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto synchronized = sample->data.data->synchronize_for_host();
    if (!synchronized) {
      return make_unexpected(std::move(synchronized.error()));
    }
    const std::uint32_t frame = *sample->data.data->data_as<std::uint32_t>();
    if (sample->data.sequence != frame || sample->metadata.frame_id != frame) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto emitted = output.emit(std::move(*sample));
    if (!emitted) {
      return emitted;
    }
    InlinePartitionProbeState::record_compute(2U, frame, context.tag_time());
    return {};
  }

  Input<schema::TensorPacketT> input;
  Output<schema::TensorPacketT> output;
};

// Validates forwarded TensorPacket fields and records the terminal causal stage.
// Representative test: TensorPacketPassThroughContinuesCausalFrameBeforeNextClockTick.
class CausalTensorPacketSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{.representation = kCausalTensorRepresentation});
  }

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
    if (!sample->data.data) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto synchronized = sample->data.data->synchronize_for_host();
    if (!synchronized) {
      return make_unexpected(std::move(synchronized.error()));
    }
    const std::uint32_t frame = *sample->data.data->data_as<std::uint32_t>();
    if (sample->data.sequence != frame || sample->metadata.frame_id != frame) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    InlinePartitionProbeState::record_compute(3U, frame, context.tag_time());
    InlinePartitionProbeState::record_sink(frame);
    return {};
  }

  Input<schema::TensorPacketT> input;
};

// Publishes a descriptor-backed Image with a runtime-pooled Tensor payload.
// Representative test: InlinePayloadParity.HostPreservesBackingMetadataAndControllerOwnership.
template <MemoryKind Kind>
class InlineImageSource final : public Operator<>,
                                public InlinePartitionLifecycle<InlineImageSource<Kind>> {
 public:
  explicit InlineImageSource(std::int32_t ambient_device_after_emit = -1,
                             bool declare_allocation = true)
      : InlinePartitionLifecycle<InlineImageSource<Kind>>(1U),
        ambient_device_after_emit_(ambient_device_after_emit),
        declare_allocation_(declare_allocation) {}

  void setup(OperatorSpec& spec) override {
    this->setup_lifecycle(spec);
    TensorOutputSpec contract{
        .representation = {.memory_kind = Kind, .dtype = DLDataType{kDLUInt, 8U, 1U}, .rank = 1U},
        .bounds = tensor_bounds(8U),
        .transfer =
            TensorTransferPolicy{
                .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
                .max_staging_byte_span = 8U,
            },
    };
    if (declare_allocation_) {
      contract.storage = TensorOutputStorage::kRuntimePool;
    }
    spec.output(output, "output").max_emits_per_compute(1U).produces_tensor(contract);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 30s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    constexpr std::array<std::int64_t, 1U> kShape{8};
    constexpr std::uint8_t kValue = 0x5AU;
    InlinePartitionProbeState::record_compute(this->operator_id_, kValue);
    if (emitted_) {
      return {};
    }

    auto* const callback_stream = context.cuda_stream().get();
    // Device-backed template specializations populate this through cudaGetDevice().
    std::int32_t callback_device = -1;  // NOLINT(misc-const-correctness)
    void* pointer = nullptr;
    auto loan = output.allocate_tensor(TensorLoanRequest{
        .shape = kShape,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
    });
    if (!loan) {
      return make_unexpected(std::move(loan.error()));
    }

    if constexpr (Kind == MemoryKind::kCudaDevice || Kind == MemoryKind::kCudaManaged) {
      if (callback_stream == nullptr || cudaGetDevice(&callback_device) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      auto writer = loan->write(context.cuda_stream());
      if (!writer) {
        return make_unexpected(std::move(writer.error()));
      }
      pointer = writer->data();
      if (cudaMemsetAsync(pointer, kValue, 8U, callback_stream) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      auto committed = std::move(*writer).commit();
      if (!committed) {
        return make_unexpected(std::move(committed.error()));
      }
    } else {
      auto writer = loan->write_host();
      if (!writer || writer->data_as<std::uint8_t>() == nullptr) {
        return make_unexpected(writer ? Error{ErrorCode::kFailure} : std::move(writer.error()));
      }
      pointer = writer->data();
      std::ranges::fill(std::span<std::uint8_t>{writer->data_as<std::uint8_t>(), 8U}, kValue);
      auto committed = std::move(*writer).commit();
      if (!committed) {
        return make_unexpected(std::move(committed.error()));
      }
    }

    InlineTensorProbeState::record_source(pointer, callback_stream, callback_device);
    schema::ImageT descriptor{};
    descriptor.header = std::make_shared<schema::HeaderT>();
    descriptor.header->stamp = std::make_shared<schema::TimeT>();
    descriptor.header->stamp->time_ns = 777;
    descriptor.header->stamp->acqtime_ns = 888;
    descriptor.header->frame_id = "inline-image";
    descriptor.height = 1;
    descriptor.width = 8;
    descriptor.encoding = schema::ImageEncoding_CUSTOM;
    auto published = output.emit_tensor(
        std::move(*loan),
        descriptor,
        EmitOptions{
            .capture_time = TimePoint{.timestamp_ns = 1234, .clock_id = ClockId{1U}},
            .frame_id = 9U,
            .trace = TraceContext::root(0xCAFEU),
        });
    if (!published) {
      return make_unexpected(std::move(published.error()));
    }
    emitted_ = true;

    if constexpr (Kind == MemoryKind::kCudaDevice || Kind == MemoryKind::kCudaManaged) {
      if (ambient_device_after_emit_ >= 0 &&
          cudaSetDevice(ambient_device_after_emit_) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
    }
    return {};
  }

  Output<schema::ImageT> output;

 private:
  std::int32_t ambient_device_after_emit_{-1};
  bool declare_allocation_{true};
  bool emitted_{};
};

// Validates an Image descriptor and Tensor payload before emitting a POD result.
// Representative test: InlinePayloadParity.HostPreservesBackingMetadataAndControllerOwnership.
template <MemoryKind Kind>
class InlineImageRelay final : public Operator<>,
                               public InlinePartitionLifecycle<InlineImageRelay<Kind>> {
 public:
  explicit InlineImageRelay(std::uint32_t operator_id = 2U, bool fail_compute = false,
                            bool retain_tensor = false)
      : InlinePartitionLifecycle<InlineImageRelay<Kind>>(operator_id),
        fail_compute_(fail_compute),
        retain_tensor_(retain_tensor) {}

  void setup(OperatorSpec& spec) override {
    this->setup_lifecycle(spec);
    spec.input(input, "input")
        .queue_depth(2U)
        .expects_tensor(TensorInputSpec{
            .representation = {.memory_kind = Kind,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 1U},
        });
    spec.output(output, "output").max_emits_per_compute(1U);
  }

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
    if (!sample->data.data) {
      return make_unexpected(Error{ErrorCode::kFailure, "INLINE_IMAGE_TENSOR_MISSING"});
    }

    const Tensor& tensor = *sample->data.data;
    // Device-backed template specializations populate this through cudaGetDevice().
    std::int32_t callback_device = -1;  // NOLINT(misc-const-correctness)
    if constexpr (Kind == MemoryKind::kCudaDevice || Kind == MemoryKind::kCudaManaged) {
      if (cudaGetDevice(&callback_device) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
    }
    std::array<std::uint8_t, 8U> bytes{};
    auto copied = tensor.copy_to_host(bytes.data(), bytes.size(), context.cuda_stream());
    if (!copied) {
      return make_unexpected(std::move(copied.error()));
    }

    constexpr ClockId kExpectedClockId{1U};
    const bool descriptor_valid =
        tensor.nbytes() == bytes.size() &&
        std::ranges::all_of(bytes, [](std::uint8_t value) { return value == 0x5AU; }) &&
        sample->data.height == 1 && sample->data.width == 8 &&
        sample->data.encoding == schema::ImageEncoding_CUSTOM && sample->data.header &&
        sample->data.header->stamp && sample->data.header->stamp->time_ns == 777 &&
        sample->data.header->stamp->acqtime_ns == 888 &&
        sample->data.header->frame_id == "inline-image" &&
        sample->metadata.capture_timestamp_ns == 1234 &&
        sample->metadata.source_clock_id == kExpectedClockId &&
        sample->metadata.trace_id == 0xCAFEU && sample->metadata.frame_id == 9U &&
        sample->metadata.sequence == 1U && sample->metadata.byte_size == bytes.size() &&
        sample->metadata.payload_kind == PayloadKind::kDescriptorBacked;
    if (!descriptor_valid) {
      return make_unexpected(Error{ErrorCode::kFailure, "INLINE_IMAGE_METADATA_MISMATCH"});
    }

    auto* const callback_stream = context.cuda_stream().get();
    InlineTensorProbeState::record_relay(tensor.data(),
                                         callback_stream,
                                         detail::TensorAccess::readiness_stream(tensor),
                                         callback_device,
                                         bytes.front());
    InlinePartitionProbeState::record_compute(this->operator_id_, bytes.front());
    InlinePartitionProbeState::record_schema("image-ok");
    if (retain_tensor_) {
      InlineTensorProbeState::retain_tensor(tensor);
    }
    if (fail_compute_) {
      return make_unexpected(Error{ErrorCode::kFailure, "INLINE_IMAGE_CALLBACK_FAILURE"});
    }
    return output.emit(InlinePartitionPod{.value = bytes.front()});
  }

  Input<schema::ImageT> input;
  Output<InlinePartitionPod> output;

 private:
  bool fail_compute_{};
  bool retain_tensor_{};
};

// Converts one POD fork branch into a Tensor with an independently selected placement.
// Representative test: MixedTensorPlacementKeepsTheOneRootForkOrdinary.
template <MemoryKind Kind>
class InlineTensorBranch final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(2U);
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = Kind,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 1U},
            .bounds = tensor_bounds(1U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

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
    constexpr std::array<std::int64_t, 1U> kShape{1};
    auto loan = output.allocate_tensor(TensorLoanRequest{
        .shape = kShape,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
    });
    if (!loan) {
      return make_unexpected(std::move(loan.error()));
    }
    if constexpr (Kind == MemoryKind::kCudaDevice) {
      auto writer = loan->write(context.cuda_stream());
      if (!writer) {
        return make_unexpected(std::move(writer.error()));
      }
      if (cudaMemsetAsync(writer->data(),
                          static_cast<int>(sample->data.value),
                          1U,
                          context.cuda_stream().get()) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      auto committed = std::move(*writer).commit();
      if (!committed) {
        return make_unexpected(std::move(committed.error()));
      }
    } else {
      auto writer = loan->write_host();
      if (!writer || writer->data_as<std::uint8_t>() == nullptr) {
        return make_unexpected(writer ? Error{ErrorCode::kFailure} : std::move(writer.error()));
      }
      *writer->data_as<std::uint8_t>() = static_cast<std::uint8_t>(sample->data.value);
      auto committed = std::move(*writer).commit();
      if (!committed) {
        return make_unexpected(std::move(committed.error()));
      }
    }
    return output.emit(std::move(*loan));
  }

  Input<InlinePartitionPod> input;
  Output<Tensor> output;
};

// Emits a POD value read from immutable typed parameters and records parameter pinning.
// Representative test: FrozenTypedParamsRemainCoherentAcrossRepeatedInlineRunsWithoutHandoffs.
class InlineParamSource final : public Operator<InlinePartitionParams> {
 public:
  explicit InlineParamSource(std::uint32_t operator_id) : operator_id_(operator_id) {}

  void setup(OperatorSpec& spec) override {
    spec.lifecycle()
        .stage(LifecycleStage::kStart, &InlineParamSource::start_stage)
        .stage(LifecycleStage::kStop, &InlineParamSource::stop_stage);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 30s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    InlinePartitionProbeState::record_param_compute<InlinePartitionParams>(operator_id_, context);
    const auto& params = context.params<InlinePartitionParams>();
    InlinePartitionProbeState::record_compute(operator_id_, params.value);
    if (emitted_) {
      return {};
    }
    emitted_ = true;
    return output.emit(InlinePartitionPod{.value = params.value});
  }

  Output<InlinePartitionPod> output;

 private:
  LifecycleStatus start_stage(LifecycleContext& context) noexcept {
    InlinePartitionProbeState::record_param_lifecycle<InlinePartitionParams>(
        operator_id_, LifecycleStage::kStart, context);
    InlinePartitionProbeState::record_lifecycle(operator_id_, LifecycleStage::kStart);
    return LifecycleStatus::kOk;
  }

  LifecycleStatus stop_stage(LifecycleContext& context) noexcept {
    InlinePartitionProbeState::record_param_lifecycle<InlinePartitionParams>(
        operator_id_, LifecycleStage::kStop, context);
    InlinePartitionProbeState::record_lifecycle(operator_id_, LifecycleStage::kStop);
    return LifecycleStatus::kOk;
  }

  std::uint32_t operator_id_{};
  bool emitted_{};
};

// Adds an immutable typed parameter to each POD sample and records parameter pinning.
// Representative test: FrozenTypedParamsRemainCoherentAcrossRepeatedInlineRunsWithoutHandoffs.
class InlineParamRelay final : public Operator<InlinePartitionParams> {
 public:
  explicit InlineParamRelay(std::uint32_t operator_id) : operator_id_(operator_id) {}

  void setup(OperatorSpec& spec) override {
    spec.lifecycle()
        .stage(LifecycleStage::kStart, &InlineParamRelay::start_stage)
        .stage(LifecycleStage::kStop, &InlineParamRelay::stop_stage);
    spec.input(input, "input").queue_depth(2U);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

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
    InlinePartitionProbeState::record_param_compute<InlinePartitionParams>(operator_id_, context);
    const std::uint32_t value = sample->data.value + context.params<InlinePartitionParams>().value;
    InlinePartitionProbeState::record_compute(operator_id_, value);
    return output.emit(InlinePartitionPod{.value = value});
  }

  Input<InlinePartitionPod> input;
  Output<InlinePartitionPod> output;

 private:
  LifecycleStatus start_stage(LifecycleContext& context) noexcept {
    InlinePartitionProbeState::record_param_lifecycle<InlinePartitionParams>(
        operator_id_, LifecycleStage::kStart, context);
    InlinePartitionProbeState::record_lifecycle(operator_id_, LifecycleStage::kStart);
    return LifecycleStatus::kOk;
  }

  LifecycleStatus stop_stage(LifecycleContext& context) noexcept {
    InlinePartitionProbeState::record_param_lifecycle<InlinePartitionParams>(
        operator_id_, LifecycleStage::kStop, context);
    InlinePartitionProbeState::record_lifecycle(operator_id_, LifecycleStage::kStop);
    return LifecycleStatus::kOk;
  }

  std::uint32_t operator_id_{};
};

// Emits values from two parameter generations while allowing a NextCompute update between them.
// Representative test: RuntimeMutableNextComputePublishesPinsAndObservesOnTheController.
class RuntimeMutableInlineParamSource final : public Operator<InlinePartitionRuntimeParams> {
 public:
  explicit RuntimeMutableInlineParamSource(std::uint32_t operator_id) : operator_id_(operator_id) {}

  void setup(OperatorSpec& spec) override {
    spec.lifecycle()
        .stage(LifecycleStage::kStart, &RuntimeMutableInlineParamSource::start_stage)
        .stage(LifecycleStage::kStop, &RuntimeMutableInlineParamSource::stop_stage);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 50ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (compute_count_ >= 2U) {
      return {};
    }
    const bool first = compute_count_ == 0U;
    if (!InlinePartitionProbeState::record_runtime_param_compute<InlinePartitionRuntimeParams>(
            operator_id_, context, first)) {
      return make_unexpected(Error{ErrorCode::kFailure, "INLINE_RUNTIME_PARAM_RELEASE_TIMEOUT"});
    }
    const std::uint32_t value = context.params<InlinePartitionRuntimeParams>().value;
    InlinePartitionProbeState::record_compute(operator_id_, value);
    ++compute_count_;
    return output.emit(InlinePartitionPod{
        .value = value,
    });
  }

  Output<InlinePartitionPod> output;

 private:
  LifecycleStatus start_stage(LifecycleContext&) noexcept {
    InlinePartitionProbeState::record_lifecycle(operator_id_, LifecycleStage::kStart);
    return LifecycleStatus::kOk;
  }

  LifecycleStatus stop_stage(LifecycleContext&) noexcept {
    InlinePartitionProbeState::record_lifecycle(operator_id_, LifecycleStage::kStop);
    return LifecycleStatus::kOk;
  }

  std::uint32_t operator_id_{};
  std::uint32_t compute_count_{};
};

// Emits driver and deliberately older side samples for temporal-selection parameter tests.
// Representative test: UpdatedSelectorDeadlineAndCallbackShareOnePinnedGeneration.
class InlineTemporalForkSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.lifecycle()
        .stage(LifecycleStage::kStart, &InlineTemporalForkSource::start_stage)
        .stage(LifecycleStage::kStop, &InlineTemporalForkSource::stop_stage);
    spec.output(driver, "driver").max_emits_per_compute(1U);
    spec.output(side, "side").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 30s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    InlinePartitionProbeState::record_compute(1U, 100U);
    if (emitted_) {
      return {};
    }
    if (!InlinePartitionProbeState::block_first_runtime_compute()) {
      return make_unexpected(Error{ErrorCode::kFailure, "INLINE_TEMPORAL_RELEASE_TIMEOUT"});
    }
    const TimePoint activation = context.activation_time();
    if (!activation.valid()) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const SampleMetadata driver_metadata{
        .capture_timestamp_ns = activation.timestamp_ns,
        .source_clock_id = activation.clock_id,
    };
    const SampleMetadata side_metadata{
        .capture_timestamp_ns = activation.timestamp_ns - 2'000'000'000LL,
        .source_clock_id = activation.clock_id,
    };
    emitted_ = true;
    auto side_result =
        side.emit(Sample<InlinePartitionPod>{InlinePartitionPod{.value = 102U}, side_metadata});
    if (!side_result) {
      return side_result;
    }
    return driver.emit(
        Sample<InlinePartitionPod>{InlinePartitionPod{.value = 101U}, driver_metadata});
  }

  Output<InlinePartitionPod> driver;
  Output<InlinePartitionPod> side;

 private:
  LifecycleStatus start_stage(LifecycleContext&) noexcept {
    InlinePartitionProbeState::record_lifecycle(1U, LifecycleStage::kStart);
    return LifecycleStatus::kOk;
  }

  LifecycleStatus stop_stage(LifecycleContext&) noexcept {
    InlinePartitionProbeState::record_lifecycle(1U, LifecycleStage::kStop);
    return LifecycleStatus::kOk;
  }

  bool emitted_{};
};

// Evaluates a parameterized temporal predicate and joins its driver and side inputs.
// Representative test: UpdatedSelectorDeadlineAndCallbackShareOnePinnedGeneration.
class InlineTemporalRelay final : public Operator<InlineTemporalParams> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.lifecycle()
        .stage(LifecycleStage::kStart, &InlineTemporalRelay::start_stage)
        .stage(LifecycleStage::kStop, &InlineTemporalRelay::stop_stage);
    spec.input(driver, "driver").queue_depth(2U);
    spec.input(side, "side").queue_depth(2U);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    auto evaluator = predicate("inline-temporal-predicate",
                               predicate_inputs(side),
                               bind_predicate(*this, &InlineTemporalRelay::evaluate));
    Contract result;
    result
        .trigger(OnPredicate{
            .evaluate_on = OnInputUpdate{driver},
            .predicate = std::move(evaluator),
            .firing = PredicateFiring::kOnEveryTrueEvaluation,
        })
        .input(driver, Exact{})
        .input(side, LatestWithin{.max_age = param_ref<&InlineTemporalParams::max_age>()})
        .on_missing(side, Skip{})
        .deadline(param_ref<&InlineTemporalParams::deadline>())
        .on_violation(OnViolation::kEmitWitness);
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto driver_sample = driver.receive();
    if (!driver_sample) {
      return make_unexpected(std::move(driver_sample.error()));
    }
    auto side_sample = side.receive();
    if (!side_sample) {
      return make_unexpected(std::move(side_sample.error()));
    }
    InlinePartitionProbeState::record_param_compute<InlineTemporalParams>(2U, context);
    const std::uint32_t value = driver_sample->data.value + side_sample->data.value;
    InlinePartitionProbeState::record_compute(2U, value);
    return output.emit(InlinePartitionPod{.value = value});
  }

  [[nodiscard]] bool evaluate(
      const PredicateEvaluationContext&, const PredicateInputView<InlinePartitionPod>& driver_view,
      const PredicateInputView<InlinePartitionPod>& side_view) const noexcept {
    constexpr std::uint32_t threshold = 200U;
    const std::uint32_t driver_value = driver_view.sample().data.value;
    const std::uint32_t side_value = side_view.sample().data.value;
    InlinePartitionProbeState::record_temporal_predicate(threshold, driver_value, side_value);
    return driver_value + side_value >= threshold;
  }

  Input<InlinePartitionPod> driver;
  Input<InlinePartitionPod> side;
  Output<InlinePartitionPod> output;

 private:
  LifecycleStatus start_stage(LifecycleContext&) noexcept {
    InlinePartitionProbeState::record_lifecycle(2U, LifecycleStage::kStart);
    return LifecycleStatus::kOk;
  }

  LifecycleStatus stop_stage(LifecycleContext&) noexcept {
    InlinePartitionProbeState::record_lifecycle(2U, LifecycleStage::kStop);
    return LifecycleStatus::kOk;
  }
};

// Selects an anchor plus a recent side sample to exercise LatestWithin partition planning.
// Representative test: LatestWithinSelectionUsesTheInlinePartitionPlan.
class LatestWithinInlineRelay final : public Operator<>,
                                      public InlinePartitionLifecycle<LatestWithinInlineRelay> {
 public:
  LatestWithinInlineRelay() : InlinePartitionLifecycle<LatestWithinInlineRelay>(2U) {}

  void setup(OperatorSpec& spec) override {
    setup_lifecycle(spec);
    spec.input(anchor, "anchor").queue_depth(2U);
    spec.input(side, "side").queue_depth(2U);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{anchor})
        .input(side, LatestWithin{.max_age = 30s})
        .on_missing(side, Skip{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = anchor.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    return output.emit(std::move(*sample));
  }

  Input<InlinePartitionPod> anchor;
  Input<InlinePartitionPod> side;
  Output<InlinePartitionPod> output;
};

// Records terminal POD values, tag times, lifecycle calls, and callback ownership.
// Representative test: RunsComputeAndLifecycleCallbacksOnTheControllerWithoutHandoffs.
class InlinePodSink final : public Operator<>, public InlinePartitionLifecycle<InlinePodSink> {
 public:
  explicit InlinePodSink(std::uint32_t operator_id)
      : InlinePartitionLifecycle<InlinePodSink>(operator_id) {}

  void setup(OperatorSpec& spec) override {
    setup_lifecycle(spec);
    spec.input(input, "input").queue_depth(2U);
  }

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
    InlinePartitionProbeState::record_compute(operator_id_, sample->data.value, context.tag_time());
    InlinePartitionProbeState::record_sink(sample->data.value);
    return {};
  }

  Input<InlinePartitionPod> input;
};

// Emits distinct POD values on two output ports for multi-input selection tests.
// Representative test: LatestWithinSelectionUsesTheInlinePartitionPlan.
class InlineForkSource final : public Operator<>,
                               public InlinePartitionLifecycle<InlineForkSource> {
 public:
  InlineForkSource() : InlinePartitionLifecycle<InlineForkSource>(1U) {}

  void setup(OperatorSpec& spec) override {
    setup_lifecycle(spec);
    spec.output(first, "first").max_emits_per_compute(1U);
    spec.output(second, "second").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 30s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    InlinePartitionProbeState::record_compute(operator_id_, 100U);
    if (emitted_) {
      return {};
    }
    emitted_ = true;
    auto first_result = first.emit(InlinePartitionPod{.value = 101U});
    if (!first_result) {
      return first_result;
    }
    return second.emit(InlinePartitionPod{.value = 102U});
  }

  Output<InlinePartitionPod> first;
  Output<InlinePartitionPod> second;

 private:
  bool emitted_{};
};

// Publishes one POD sample through a single output connected to two consumers.
// Representative test: AdmitsOneRootWithTwoConsumersInStableSiblingOrder.
class InlineFanoutSource final : public Operator<>,
                                 public InlinePartitionLifecycle<InlineFanoutSource> {
 public:
  InlineFanoutSource() : InlinePartitionLifecycle<InlineFanoutSource>(1U) {}

  void setup(OperatorSpec& spec) override {
    setup_lifecycle(spec);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 30s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    InlinePartitionProbeState::record_compute(operator_id_, 100U);
    if (emitted_) {
      return {};
    }
    emitted_ = true;
    return output.emit(InlinePartitionPod{.value = 101U});
  }

  Output<InlinePartitionPod> output;

 private:
  bool emitted_{};
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

// Publishes numbered owning String schema samples from a configurable clock.
// Representative test: RunsAnOwningSchemaChainOnTheControllerWithoutHandoffs.
class InlineSchemaSource final : public Operator<>,
                                 public InlinePartitionLifecycle<InlineSchemaSource> {
 public:
  explicit InlineSchemaSource(std::size_t emission_limit = 1U,
                              std::chrono::nanoseconds period = 30s)
      : InlinePartitionLifecycle<InlineSchemaSource>(1U),
        emission_limit_(emission_limit),
        period_(period) {}

  void setup(OperatorSpec& spec) override {
    setup_lifecycle(spec);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = period_});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (next_frame_ >= emission_limit_) {
      return {};
    }
    const std::uint32_t frame = static_cast<std::uint32_t>(next_frame_);
    schema::StringT sample;
    sample.data = "root";
    auto emitted = output.emit(Sample<schema::StringT>{
        std::move(sample),
        SampleMetadata{.frame_id = frame},
    });
    if (!emitted) {
      return emitted;
    }
    InlinePartitionProbeState::record_compute(operator_id_, frame, context.tag_time());
    ++next_frame_;
    return {};
  }

  Output<schema::StringT> output;

 private:
  std::size_t emission_limit_{};
  std::chrono::nanoseconds period_{};
  std::size_t next_frame_{};
};

// Appends its operator ID to an owning String schema while preserving frame metadata.
// Representative test: RunsAnOwningSchemaChainOnTheControllerWithoutHandoffs.
class InlineSchemaRelay final : public Operator<>,
                                public InlinePartitionLifecycle<InlineSchemaRelay> {
 public:
  explicit InlineSchemaRelay(std::uint32_t operator_id)
      : InlinePartitionLifecycle<InlineSchemaRelay>(operator_id) {}

  void setup(OperatorSpec& spec) override {
    setup_lifecycle(spec);
    spec.input(input, "input").queue_depth(2U);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

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
    sample->data.data += ":" + std::to_string(operator_id_);
    const std::uint32_t frame = static_cast<std::uint32_t>(sample->metadata.frame_id);
    auto emitted = output.emit(std::move(*sample));
    if (!emitted) {
      return emitted;
    }
    InlinePartitionProbeState::record_compute(operator_id_, frame, context.tag_time());
    return {};
  }

  Input<schema::StringT> input;
  Output<schema::StringT> output;
};

// Records the final owning String schema, frame order, tag time, and lifecycle activity.
// Representative test: RunsAnOwningSchemaChainOnTheControllerWithoutHandoffs.
class InlineSchemaSink final : public Operator<>,
                               public InlinePartitionLifecycle<InlineSchemaSink> {
 public:
  InlineSchemaSink() : InlinePartitionLifecycle<InlineSchemaSink>(4U) {}

  void setup(OperatorSpec& spec) override {
    setup_lifecycle(spec);
    spec.input(input, "input").queue_depth(2U);
  }

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
    const std::uint32_t frame = static_cast<std::uint32_t>(sample->metadata.frame_id);
    InlinePartitionProbeState::record_compute(operator_id_, frame, context.tag_time());
    InlinePartitionProbeState::record_sink(frame);
    InlinePartitionProbeState::record_schema(sample->data.data);
    return {};
  }

  Input<schema::StringT> input;
};

struct InlinePartitionRunEvidence {
  bool observed{};
  RunTermination termination{RunTermination::kFailed};
  std::size_t diagnostic_count{};
  std::optional<std::thread::id> controller_thread;
  std::vector<InlinePartitionComputeRecord> compute_records;
  std::vector<InlinePartitionLifecycleRecord> lifecycle_records;
  std::vector<InlinePartitionParamComputeRecord> param_compute_records;
  std::vector<InlinePartitionParamLifecycleRecord> param_lifecycle_records;
  std::vector<InlineTemporalPredicateRecord> temporal_predicate_records;
  std::vector<std::uint32_t> sink_values;
  std::optional<std::string> schema_value;
  detail::RuntimePhaseCounters::Snapshot counters{};
};

[[nodiscard]] std::uint64_t phase_count(const InlinePartitionRunEvidence& evidence,
                                        detail::RuntimePhaseCounter counter) {
  return evidence.counters[detail::RuntimePhaseCounters::index(counter)];
}

void expect_no_legacy_lane_handoffs(const detail::RuntimePhaseCounters::Snapshot& counters) {
  const auto count = [&](detail::RuntimePhaseCounter counter) {
    return counters[detail::RuntimePhaseCounters::index(counter)];
  };
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kLaneDispatches), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kLaneComputeDispatches), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kLaneDispatcherDispatches), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kLaneCompletionsObserved), 0U);
}

[[nodiscard]] InlinePartitionRunEvidence capture_run_evidence(
    const RunSession& session, bool observed,
    const std::shared_ptr<detail::RuntimePhaseCounters>& counters,
    const std::shared_ptr<InlinePartitionControllerProbe>& control) {
  return InlinePartitionRunEvidence{
      .observed = observed,
      .termination = session.termination(),
      .diagnostic_count = session.diagnostics().size(),
      .controller_thread = control->controller_thread(),
      .compute_records = InlinePartitionProbeState::compute_records(),
      .lifecycle_records = InlinePartitionProbeState::lifecycle_records(),
      .param_compute_records = InlinePartitionProbeState::param_compute_records(),
      .param_lifecycle_records = InlinePartitionProbeState::param_lifecycle_records(),
      .temporal_predicate_records = InlinePartitionProbeState::temporal_predicate_records(),
      .sink_values = InlinePartitionProbeState::sink_values(),
      .schema_value = InlinePartitionProbeState::schema_value(),
      .counters = counters->snapshot(),
  };
}

[[nodiscard]] InlinePartitionRunEvidence run_and_observe(ExecutionPlan plan, std::size_t sink_count,
                                                         bool schema_sink = false) {
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const auto control = std::make_shared<InlinePartitionControllerProbe>();
  plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
  plan = detail::RuntimeTestPlanAccess::with_support_thread_control(plan, control);

  RunSession session = run_async(plan);
  const bool observed = schema_sink ? InlinePartitionProbeState::wait_for_schema()
                                    : InlinePartitionProbeState::wait_for_sinks(sink_count);
  session.request_stop();
  session.wait();
  return capture_run_evidence(session, observed, counters, control);
}

[[nodiscard]] bool wait_for_terminal(const RunSession& session) {
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (session.termination() == RunTermination::kRunning &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  return session.termination() != RunTermination::kRunning;
}

[[nodiscard]] ConnectionOptions queue_options(QueuePolicy policy) {
  return ConnectionOptions{
      .queue_policy = policy,
      .queue_depth = 2U,
  };
}

[[nodiscard]] ExecutionPlan make_pod_chain_plan(QueuePolicy policy, bool explicit_placement = false,
                                                std::size_t emission_limit = 1U,
                                                std::chrono::nanoseconds period = 30s) {
  // [source] -> [first relay] -> [second relay] -> [sink]
  Graph graph{explicit_placement ? "inline-pod-chain-fallback" : "inline-pod-chain"};
  auto source = graph.op<InlinePodSource>("source", 1U, 41U, false, emission_limit, period);
  auto first = graph.op<InlinePodRelay>("first-relay", 2U, 1U);
  auto second = graph.op<InlinePodRelay>("second-relay", 3U, 2U);
  auto sink = graph.op<InlinePodSink>("sink", 4U);
  graph.add_flow(source->output, first->input, queue_options(policy));
  graph.add_flow(first->output, second->input, queue_options(policy));
  graph.add_flow(second->output, sink->input, queue_options(policy));
  graph.set_default_clock(graph.add_clock<RealtimeClock>("inline-clock"));
  CompileOptions options;
  if (explicit_placement) {
    auto partition = graph.partition("inline-partition-fallback-worker");
    partition.add(source).add(first).add(second).add(sink);
    options.deployment.bind_thread_placement("inline-partition-fallback-worker", ThreadPlacement{});
  }
  return compile(graph, std::move(options));
}

enum class TensorPlanRoute : std::uint8_t {
  kInline,
  kHoloMQ,
  kFanout,
};

template <MemoryKind Kind>
[[nodiscard]] ExecutionPlan make_tensor_chain_plan(
    QueuePolicy policy, TensorPlanRoute route = TensorPlanRoute::kInline,
    DeviceId device = DeviceId{}, std::int32_t ambient_device_after_emit = -1,
    bool fail_relay = false, bool retain_tensor = false,
    std::int32_t ambient_device_before_failure = -1) {
  // [source] -- Tensor --> [relay] -- POD --> [sink]
  //     \---- optional fanout --> [second relay] --> [second sink]
  Graph graph{"inline-tensor-chain"};
  auto source = graph.op<InlineTensorSource<Kind>>("source", ambient_device_after_emit);
  auto relay = graph.op<InlineTensorRelay<Kind>>(
      "relay", 2U, fail_relay, retain_tensor, ambient_device_before_failure);
  auto sink = graph.op<InlinePodSink>("sink", 3U);
  graph.add_flow(source->output, relay->input, queue_options(policy));
  graph.add_flow(relay->output, sink->input, queue_options(policy));
  if (route == TensorPlanRoute::kFanout) {
    auto second_relay = graph.op<InlineTensorRelay<Kind>>("second-relay", 4U);
    auto second_sink = graph.op<InlinePodSink>("second-sink", 5U);
    graph.add_flow(source->output, second_relay->input, queue_options(policy));
    graph.add_flow(second_relay->output, second_sink->input, queue_options(policy));
  }
  if (route == TensorPlanRoute::kInline) {
    graph.partition("inline-tensor-complete-cover").include_all();
  }
  graph.set_default_clock(graph.add_clock<RealtimeClock>("inline-clock"));

  CompileOptions options;
  options.profile = Profile::kStandard;
  if constexpr (Kind == MemoryKind::kCudaDevice || Kind == MemoryKind::kCudaManaged) {
    options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
        .operator_path = "source",
        .output_port = "output",
        .device = device,
    });
  }
  return route == TensorPlanRoute::kHoloMQ
             ? detail::compile_with_process_local_holomq(graph, std::move(options))
             : compile(graph, std::move(options));
}

template <MemoryKind Kind>
[[nodiscard]] ExecutionPlan make_image_chain_plan(
    QueuePolicy policy, TensorPlanRoute route = TensorPlanRoute::kInline,
    DeviceId device = DeviceId{}, std::int32_t ambient_device_after_emit = -1,
    bool fail_relay = false, bool retain_tensor = false, bool declare_allocation = true) {
  // [source] -- Image --> [relay] -- POD --> [sink]
  //     \---- optional fanout --> [second relay] --> [second sink]
  Graph graph{"inline-image-chain"};
  auto source =
      graph.op<InlineImageSource<Kind>>("source", ambient_device_after_emit, declare_allocation);
  auto relay = graph.op<InlineImageRelay<Kind>>("relay", 2U, fail_relay, retain_tensor);
  auto sink = graph.op<InlinePodSink>("sink", 3U);
  graph.add_flow(source->output, relay->input, queue_options(policy));
  graph.add_flow(relay->output, sink->input, queue_options(policy));
  if (route == TensorPlanRoute::kFanout) {
    auto second_relay = graph.op<InlineImageRelay<Kind>>("second-relay", 4U);
    auto second_sink = graph.op<InlinePodSink>("second-sink", 5U);
    graph.add_flow(source->output, second_relay->input, queue_options(policy));
    graph.add_flow(second_relay->output, second_sink->input, queue_options(policy));
  }
  if (route == TensorPlanRoute::kInline) {
    graph.partition("inline-image-complete-cover").include_all();
  }
  graph.set_default_clock(graph.add_clock<RealtimeClock>("inline-clock"));

  CompileOptions options;
  options.profile = Profile::kStandard;
  if constexpr (Kind == MemoryKind::kCudaDevice || Kind == MemoryKind::kCudaManaged) {
    options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
        .operator_path = "source",
        .output_port = "output",
        .device = device,
    });
  }
  return route == TensorPlanRoute::kHoloMQ
             ? detail::compile_with_process_local_holomq(graph, std::move(options))
             : compile(graph, std::move(options));
}

struct InlinePayloadPlanOptions {
  QueuePolicy policy{QueuePolicy::kBlocking};
  TensorPlanRoute route{TensorPlanRoute::kInline};
  DeviceId device{};
  std::int32_t ambient_device_after_emit{-1};
  bool fail_relay{false};
  bool retain_tensor{false};
};

// Exercises direct raw-Tensor inline execution through the common parity scenarios.
struct RawTensorCase {
  static constexpr std::string_view name{"RawTensor"};

  template <MemoryKind Kind>
  [[nodiscard]] static ExecutionPlan make_plan(const InlinePayloadPlanOptions& options) {
    return make_tensor_chain_plan<Kind>(options.policy,
                                        options.route,
                                        options.device,
                                        options.ambient_device_after_emit,
                                        options.fail_relay,
                                        options.retain_tensor);
  }

  static void expect_payload_evidence(const InlinePartitionRunEvidence& evidence) {
    EXPECT_FALSE(evidence.schema_value.has_value());
  }
};

// Exercises descriptor-backed Image inline execution through the common parity scenarios.
struct DescriptorImageCase {
  static constexpr std::string_view name{"DescriptorImage"};

  template <MemoryKind Kind>
  [[nodiscard]] static ExecutionPlan make_plan(const InlinePayloadPlanOptions& options) {
    return make_image_chain_plan<Kind>(options.policy,
                                       options.route,
                                       options.device,
                                       options.ambient_device_after_emit,
                                       options.fail_relay,
                                       options.retain_tensor);
  }

  static void expect_payload_evidence(const InlinePartitionRunEvidence& evidence) {
    EXPECT_EQ(evidence.schema_value, std::optional<std::string>{"image-ok"});
  }
};

template <MemoryKind LeftKind, MemoryKind RightKind>
[[nodiscard]] ExecutionPlan make_mixed_tensor_fork_plan(DeviceId left_device = DeviceId{},
                                                        DeviceId right_device = DeviceId{}) {
  // [source] -> [left branch]  -- Tensor --> [left relay]  -> [left sink]
  //     \----> [right branch] -- Tensor --> [right relay] -> [right sink]
  Graph graph{"mixed-inline-tensor-fork"};
  auto source = graph.op<InlineForkSource>("source");
  auto left_branch = graph.op<InlineTensorBranch<LeftKind>>("left-branch");
  auto left_relay = graph.op<InlineTensorRelay<LeftKind>>("left-relay", 4U);
  auto left_sink = graph.op<InlinePodSink>("left-sink", 6U);
  auto right_branch = graph.op<InlineTensorBranch<RightKind>>("right-branch");
  auto right_relay = graph.op<InlineTensorRelay<RightKind>>("right-relay", 5U);
  auto right_sink = graph.op<InlinePodSink>("right-sink", 7U);
  graph.add_flow(source->first, left_branch->input, queue_options(QueuePolicy::kBlocking));
  graph.add_flow(left_branch->output, left_relay->input, queue_options(QueuePolicy::kBlocking));
  graph.add_flow(left_relay->output, left_sink->input, queue_options(QueuePolicy::kBlocking));
  graph.add_flow(source->second, right_branch->input, queue_options(QueuePolicy::kBlocking));
  graph.add_flow(right_branch->output, right_relay->input, queue_options(QueuePolicy::kBlocking));
  graph.add_flow(right_relay->output, right_sink->input, queue_options(QueuePolicy::kBlocking));
  graph.set_default_clock(graph.add_clock<RealtimeClock>("inline-clock"));

  CompileOptions options;
  options.profile = Profile::kStandard;
  if constexpr (LeftKind == MemoryKind::kCudaDevice) {
    options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
        .operator_path = "left-branch",
        .output_port = "output",
        .device = left_device,
    });
  }
  if constexpr (RightKind == MemoryKind::kCudaDevice) {
    options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
        .operator_path = "right-branch",
        .output_port = "output",
        .device = right_device,
    });
  }
  return compile(graph, std::move(options));
}

[[nodiscard]] LifecycleTransactionOptions inline_lifecycle_capacity() {
  return LifecycleTransactionOptions{
      .max_pending_requests = 4U,
      .max_pending_transactions = 4U,
      .max_receipts = 8U,
      .max_fenced_operators = 4U,
      .max_candidate_generations = 4U,
      .max_events = 64U,
      .max_audit_records = 8U,
      .max_quiesce_records = 8U,
      .max_rollback_records = 8U,
  };
}

[[nodiscard]] ExecutionPlan make_restart_chain_plan() {
  // [restarting source] -> [relay] -> [sink]
  Graph graph{"inline-restart-chain"};
  auto source = graph.op<InlinePodSource>("source", 1U, 41U, true);
  auto relay = graph.op<InlinePodRelay>("relay", 2U, 1U);
  auto sink = graph.op<InlinePodSink>("sink", 3U);
  graph.add_flow(source->output, relay->input, queue_options(QueuePolicy::kBlocking));
  graph.add_flow(relay->output, sink->input, queue_options(QueuePolicy::kBlocking));
  graph.lifecycle().allow_self_restart(source);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("inline-clock"));
  CompileOptions options;
  options.lifecycle_transactions = inline_lifecycle_capacity();
  return compile(graph, std::move(options));
}

[[nodiscard]] ExecutionPlan make_failing_chain_plan() {
  // [source] -> [failing relay] -> [sink]
  Graph graph{"inline-failing-chain"};
  auto source = graph.op<InlinePodSource>("source", 1U, 41U);
  auto relay = graph.op<InlinePodRelay>("failing-relay", 2U, 1U, true);
  auto sink = graph.op<InlinePodSink>("sink", 3U);
  graph.add_flow(source->output, relay->input, queue_options(QueuePolicy::kBlocking));
  graph.add_flow(relay->output, sink->input, queue_options(QueuePolicy::kBlocking));
  graph.set_default_clock(graph.add_clock<RealtimeClock>("inline-clock"));
  return compile(graph);
}

[[nodiscard]] ExecutionPlan make_param_chain_plan(bool explicit_placement = false) {
  // [parameter source] -> [parameter relay] -> [sink]
  Graph graph{explicit_placement ? "inline-param-chain-fallback" : "inline-param-chain"};
  auto source = graph.op<InlineParamSource>("source", 1U);
  auto relay = graph.op<InlineParamRelay>("relay", 2U);
  auto sink = graph.op<InlinePodSink>("sink", 3U);
  source.set_params(InlinePartitionParams{.value = 41U});
  relay.set_params(InlinePartitionParams{.value = 3U});
  graph.add_flow(source->output, relay->input, queue_options(QueuePolicy::kBlocking));
  graph.add_flow(relay->output, sink->input, queue_options(QueuePolicy::kBlocking));
  graph.set_default_clock(graph.add_clock<RealtimeClock>("inline-clock"));
  CompileOptions options;
  if (explicit_placement) {
    auto partition = graph.partition("inline-partition-fallback-worker");
    partition.add(source).add(relay).add(sink);
    options.deployment.bind_thread_placement("inline-partition-fallback-worker", ThreadPlacement{});
  }
  return compile(graph, std::move(options));
}

enum class RuntimeMutablePlanRoute : std::uint8_t {
  kInline,
  kExplicitPlacement,
  kHoloMQ,
};

[[nodiscard]] ExecutionPlan make_runtime_mutable_param_chain_plan(
    RuntimeMutablePlanRoute route = RuntimeMutablePlanRoute::kInline) {
  // [runtime-parameter source] -> [relay] -> [sink]
  Graph graph{"runtime-mutable-param-chain"};
  const bool explicit_placement = route == RuntimeMutablePlanRoute::kExplicitPlacement;
  auto source = graph.op<RuntimeMutableInlineParamSource>("source", 1U);
  auto relay = graph.op<InlinePodRelay>("relay", 2U, 3U);
  auto sink = graph.op<InlinePodSink>("sink", 3U);
  source.set_params(InlinePartitionRuntimeParams{.value = 41U});
  graph.add_flow(source->output, relay->input, queue_options(QueuePolicy::kBlocking));
  graph.add_flow(relay->output, sink->input, queue_options(QueuePolicy::kBlocking));
  graph.set_default_clock(graph.add_clock<RealtimeClock>("inline-clock"));
  CompileOptions options;
  options.profile = Profile::kStandard;
  options.parameters.runtime_updates = ParamRuntimeUpdatePolicy::kEnabledIfAdmitted;
  if (explicit_placement) {
    auto partition = graph.partition("inline-partition-fallback-worker");
    partition.add(source).add(relay).add(sink);
    options.deployment.bind_thread_placement("inline-partition-fallback-worker", ThreadPlacement{});
  }
  return route == RuntimeMutablePlanRoute::kHoloMQ
             ? detail::compile_with_process_local_holomq(graph, std::move(options))
             : compile(graph, std::move(options));
}

[[nodiscard]] ExecutionPlan make_temporal_param_chain_plan() {
  // [source] -- driver + side --> [temporal relay] -> [sink]
  Graph graph{"inline-temporal-param-chain"};
  auto source = graph.op<InlineTemporalForkSource>("source");
  auto relay = graph.op<InlineTemporalRelay>("relay");
  auto sink = graph.op<InlinePodSink>("sink", 3U);
  relay.set_params(InlineTemporalParams{});
  graph.add_flow(source->driver, relay->driver, queue_options(QueuePolicy::kBlocking));
  graph.add_flow(source->side, relay->side, queue_options(QueuePolicy::kBlocking));
  graph.add_flow(relay->output, sink->input, queue_options(QueuePolicy::kBlocking));
  graph.set_default_clock(graph.add_clock<RealtimeClock>("inline-clock"));
  CompileOptions options;
  options.profile = Profile::kStandard;
  options.parameters.runtime_updates = ParamRuntimeUpdatePolicy::kEnabledIfAdmitted;
  return compile(graph, std::move(options));
}

[[nodiscard]] ExecutionPlan make_latest_within_chain_plan() {
  // [source] -- anchor + side --> [LatestWithin relay] -> [sink]
  Graph graph{"latest-within-inline-fallback"};
  auto source = graph.op<InlineForkSource>("source");
  auto relay = graph.op<LatestWithinInlineRelay>("relay");
  auto sink = graph.op<InlinePodSink>("sink", 3U);
  graph.add_flow(source->first, relay->anchor, queue_options(QueuePolicy::kBlocking));
  graph.add_flow(source->second, relay->side, queue_options(QueuePolicy::kBlocking));
  graph.add_flow(relay->output, sink->input, queue_options(QueuePolicy::kBlocking));
  graph.set_default_clock(graph.add_clock<RealtimeClock>("inline-clock"));
  return compile(graph);
}

[[nodiscard]] ExecutionPlan make_fork_plan() {
  // [source] -> [first sink]
  //     \----> [second sink]
  Graph graph{"inline-pod-fork"};
  auto source = graph.op<InlineFanoutSource>("source");
  auto first = graph.op<InlinePodSink>("first-sink", 2U);
  auto second = graph.op<InlinePodSink>("second-sink", 3U);
  graph.add_flow(source->output, first->input, queue_options(QueuePolicy::kBlocking));
  graph.add_flow(source->output, second->input, queue_options(QueuePolicy::kBlocking));
  graph.partition("inline-pod-fork-complete-cover").include_all();
  graph.set_default_clock(graph.add_clock<RealtimeClock>("inline-clock"));
  return compile(graph);
}

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
  return compile(graph);
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
                                                              std::uint32_t second_cpu) {
  // [first periodic root]    [second periodic root]
  Graph graph{"partition-runner-cpu-placement"};
  const auto first = graph.op<PartitionRunnerPeriodicProbe>("first", 0U, 1U);
  const auto second = graph.op<PartitionRunnerPeriodicProbe>("second", 1U, 2U);
  graph.partition("first").add(first);
  graph.partition("second").add(second);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("partition-runner-clock"));
  CompileOptions options;
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

[[nodiscard]] ExecutionPlan make_schema_chain_plan(std::size_t emission_limit = 1U,
                                                   std::chrono::nanoseconds period = 30s) {
  // [source] -> [first relay] -> [second relay] -> [sink]
  Graph graph{"inline-schema-chain"};
  auto source = graph.op<InlineSchemaSource>("source", emission_limit, period);
  auto first = graph.op<InlineSchemaRelay>("first-relay", 2U);
  auto second = graph.op<InlineSchemaRelay>("second-relay", 3U);
  auto sink = graph.op<InlineSchemaSink>("sink");
  graph.add_flow(source->output, first->input, queue_options(QueuePolicy::kDropOldest));
  graph.add_flow(first->output, second->input, queue_options(QueuePolicy::kDropOldest));
  graph.add_flow(second->output, sink->input, queue_options(QueuePolicy::kDropOldest));
  graph.set_default_clock(graph.add_clock<RealtimeClock>("inline-clock"));
  return compile(graph);
}

void expect_inline_thread_evidence(
    const InlinePartitionRunEvidence& evidence, std::size_t expected_compute_count,
    std::size_t expected_operator_count,
    std::optional<std::size_t> expected_lifecycle_count = std::nullopt) {
  const std::size_t lifecycle_count =
      expected_lifecycle_count.value_or(expected_operator_count * 2U);
  ASSERT_TRUE(evidence.controller_thread.has_value());
  ASSERT_EQ(evidence.compute_records.size(), expected_compute_count);
  ASSERT_EQ(evidence.lifecycle_records.size(), lifecycle_count);
  EXPECT_TRUE(std::ranges::all_of(evidence.compute_records, [&](const auto& record) {
    return record.thread == *evidence.controller_thread;
  }));
  EXPECT_TRUE(std::ranges::all_of(evidence.lifecycle_records, [&](const auto& record) {
    return record.thread == *evidence.controller_thread;
  }));
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kLaneDispatches), 0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kLaneComputeDispatches), 0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kLaneDispatcherDispatches), 0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kLaneLifecycleDispatches), 0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kLaneUnknownDispatches), 0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kLaneSleepingWakeSignals), 0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kLaneBusyTaskPublications), 0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kLaneCompletionsObserved), 0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kLaneInlineOwnerInvocations),
            expected_compute_count + lifecycle_count +
                phase_count(evidence, detail::RuntimePhaseCounter::kReadinessPollDispatches));
}

void expect_fast_chain_controller_evidence(const InlinePartitionRunEvidence& evidence) {
  ASSERT_TRUE(evidence.controller_thread.has_value());
  EXPECT_TRUE(std::ranges::all_of(evidence.compute_records, [&](const auto& record) {
    return record.thread == *evidence.controller_thread;
  }));
  EXPECT_TRUE(std::ranges::all_of(evidence.lifecycle_records, [&](const auto& record) {
    return record.thread == *evidence.controller_thread;
  }));
  expect_no_legacy_lane_handoffs(evidence.counters);
  EXPECT_GE(phase_count(evidence, detail::RuntimePhaseCounter::kLaneInlineOwnerInvocations),
            evidence.compute_records.size() + evidence.lifecycle_records.size());
}

void expect_param_coherence(const InlinePartitionRunEvidence& evidence, std::uint32_t operator_id,
                            std::uint32_t expected_value, bool inline_owner = true) {
  ASSERT_TRUE(evidence.controller_thread.has_value());
  const auto compute = std::ranges::find(
      evidence.param_compute_records, operator_id, &InlinePartitionParamComputeRecord::operator_id);
  ASSERT_NE(compute, evidence.param_compute_records.end());
  EXPECT_EQ(compute->value, expected_value);
  EXPECT_EQ(compute->generation, 0U);
  EXPECT_TRUE(compute->repeated_address_stable);
  EXPECT_TRUE(compute->repeated_generation_stable);
  // The fatal assertion above proves the optional is engaged for these test-only dereferences.
  // NOLINTBEGIN(bugprone-unchecked-optional-access)
  if (inline_owner) {
    EXPECT_EQ(compute->thread, *evidence.controller_thread);
  } else {
    EXPECT_NE(compute->thread, *evidence.controller_thread);
  }

  std::size_t lifecycle_count = 0U;
  for (const auto& lifecycle : evidence.param_lifecycle_records) {
    if (lifecycle.operator_id != operator_id) {
      continue;
    }
    ++lifecycle_count;
    EXPECT_EQ(lifecycle.value, expected_value);
    EXPECT_EQ(lifecycle.generation, compute->generation);
    EXPECT_EQ(lifecycle.address, compute->address);
    EXPECT_TRUE(lifecycle.repeated_address_stable);
    EXPECT_TRUE(lifecycle.repeated_generation_stable);
    if (inline_owner) {
      EXPECT_EQ(lifecycle.thread, *evidence.controller_thread);
    } else {
      EXPECT_NE(lifecycle.thread, *evidence.controller_thread);
    }
  }
  // NOLINTEND(bugprone-unchecked-optional-access)
  EXPECT_EQ(lifecycle_count, 2U);
}

class RuntimeMutableComputeRelease {
 public:
  RuntimeMutableComputeRelease() = default;
  RuntimeMutableComputeRelease(const RuntimeMutableComputeRelease&) = delete;
  RuntimeMutableComputeRelease& operator=(const RuntimeMutableComputeRelease&) = delete;

  ~RuntimeMutableComputeRelease() { release(); }

  void release() noexcept {
    if (!released_) {
      InlinePartitionProbeState::release_first_runtime_compute();
      released_ = true;
    }
  }

 private:
  bool released_{};
};

void expect_runtime_mutable_update_route(RuntimeMutablePlanRoute route, bool expect_inline_owner) {
  InlinePartitionProbeState::reset();
  ExecutionPlan compiled = make_runtime_mutable_param_chain_plan(route);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  ASSERT_EQ(compiled.parameter_fields().size(), 1U);
  const ParameterFieldPlanRecord field = compiled.parameter_fields().front();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  if (route != RuntimeMutablePlanRoute::kHoloMQ) {
    EXPECT_EQ(state->inline_partition_execution.has_value(),
              route == RuntimeMutablePlanRoute::kInline)
        << compiled.json();
  }
  ASSERT_EQ(state->partition_runners.size(), 1U);
  ASSERT_TRUE(std::ranges::any_of(state->operator_parameters, [](const auto& parameters) {
    return parameters.runtime_updates_enabled;
  }));

  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const auto control = std::make_shared<InlinePartitionControllerProbe>();
  compiled = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  compiled = detail::RuntimeTestPlanAccess::with_support_thread_control(compiled, control);

  RunSession session = run_async(compiled);
  RuntimeMutableComputeRelease release;
  ASSERT_TRUE(InlinePartitionProbeState::wait_for_first_runtime_compute());
  const auto run_owner = session.operator_ref(field.owner);
  ASSERT_TRUE(run_owner);
  const ParamAccessPolicy policy{
      .origin = ParamControlOrigin::kLocalApplication,
      .permissions = ParamPermission::kRead | ParamPermission::kUpdate | ParamPermission::kAwait,
  };
  auto handle_result = session.parameter_control(policy);
  ASSERT_TRUE(handle_result);
  ParamControlHandle handle = std::move(*handle_result);
  const std::array patch{
      ParamPatch{
          .target = ParamAddress{.operator_instance = *run_owner, .field = field.field},
          .value = ParamValue::from(std::uint32_t{43U}),
      },
  };
  const ParamBatchUpdateResult accepted =
      handle.submit(patch, ParamUpdateOptions{.effective = NextCompute{}});
  ASSERT_TRUE(accepted.committed);
  EXPECT_EQ(accepted.status, ParamUpdateStatus::kAccepted);

  release.release();
  const bool observed = InlinePartitionProbeState::wait_for_sinks(2U);
  session.request_stop();
  session.wait();

  const InlinePartitionRunEvidence evidence{
      .observed = observed,
      .termination = session.termination(),
      .diagnostic_count = session.diagnostics().size(),
      .controller_thread = control->controller_thread(),
      .compute_records = InlinePartitionProbeState::compute_records(),
      .lifecycle_records = InlinePartitionProbeState::lifecycle_records(),
      .param_compute_records = InlinePartitionProbeState::param_compute_records(),
      .param_lifecycle_records = InlinePartitionProbeState::param_lifecycle_records(),
      .temporal_predicate_records = InlinePartitionProbeState::temporal_predicate_records(),
      .sink_values = InlinePartitionProbeState::sink_values(),
      .schema_value = InlinePartitionProbeState::schema_value(),
      .counters = counters->snapshot(),
  };
  ASSERT_TRUE(evidence.observed);
  EXPECT_EQ(evidence.termination, RunTermination::kStopped);
  EXPECT_EQ(evidence.diagnostic_count, 0U);
  EXPECT_EQ(evidence.sink_values, (std::vector<std::uint32_t>{44U, 46U}));
  ASSERT_EQ(evidence.param_compute_records.size(), 2U);
  const auto& initial = evidence.param_compute_records[0];
  EXPECT_EQ(initial.operator_id, 1U);
  EXPECT_EQ(initial.value, 41U);
  EXPECT_EQ(initial.generation, 0U);
  EXPECT_TRUE(initial.repeated_address_stable);
  EXPECT_TRUE(initial.repeated_generation_stable);
  const auto& updated = evidence.param_compute_records[1];
  EXPECT_EQ(updated.operator_id, 1U);
  EXPECT_EQ(updated.value, 43U);
  EXPECT_EQ(updated.generation, accepted.target_generation);
  EXPECT_TRUE(updated.repeated_address_stable);
  EXPECT_TRUE(updated.repeated_generation_stable);
  const auto observed_update = handle.await(accepted.update_sequence, 0ns);
  ASSERT_TRUE(observed_update);
  EXPECT_EQ(observed_update->state, ParamUpdateState::kObserved);
  EXPECT_EQ(observed_update->target_generation, accepted.target_generation);

  if (expect_inline_owner) {
    ASSERT_TRUE(evidence.controller_thread.has_value());
    // The fatal assertion above proves the optional is engaged for these test-only dereferences.
    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    expect_inline_thread_evidence(evidence, 6U, 3U);
    EXPECT_EQ(initial.thread, *evidence.controller_thread);
    EXPECT_EQ(updated.thread, *evidence.controller_thread);
    // NOLINTEND(bugprone-unchecked-optional-access)
  }
}

class InlinePodChainTest : public testing::TestWithParam<QueuePolicy> {};

// REQ: RUNTIME-041
TEST_P(InlinePodChainTest, RunsComputeAndLifecycleCallbacksOnTheControllerWithoutHandoffs) {
  constexpr std::size_t kFrameCount = 3U;
  InlinePartitionProbeState::reset();
  // Key setup: run three frames through a four-Operator direct POD chain whose 1 ns source remains
  // ready while every edge is eligible for causal continuation.
  const ExecutionPlan compiled = make_pod_chain_plan(GetParam(), false, kFrameCount, 1ns);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_TRUE(state->inline_partition_execution.has_value()) << compiled.json();

  const InlinePartitionRunEvidence evidence = run_and_observe(compiled, kFrameCount);
  // Key assertion: each frame drains in topological order at one logical time and successive
  // microsteps, with one accepted causal frontier per edge and no fallback.
  ASSERT_TRUE(evidence.observed);
  EXPECT_EQ(evidence.termination, RunTermination::kStopped);
  EXPECT_EQ(evidence.diagnostic_count, 0U);
  ASSERT_EQ(evidence.sink_values, (std::vector<std::uint32_t>{44U, 44U, 44U}));
  ASSERT_EQ(evidence.compute_records.size(), kFrameCount * 4U);
  constexpr std::array<std::uint32_t, 4U> kStageValues{41U, 42U, 44U, 44U};
  for (std::size_t frame = 0U; frame < kFrameCount; ++frame) {
    const std::size_t source_index = frame * kStageValues.size();
    const TagTime source_tag = evidence.compute_records[source_index].tag_time;
    ASSERT_TRUE(source_tag.valid());
    ASSERT_LE(source_tag.microstep, std::numeric_limits<std::uint32_t>::max() - 3U);
    for (std::size_t stage = 0U; stage < kStageValues.size(); ++stage) {
      const auto& record = evidence.compute_records[source_index + stage];
      SCOPED_TRACE(frame);
      SCOPED_TRACE(stage);
      EXPECT_EQ(record.operator_id, stage + 1U);
      EXPECT_EQ(record.value, kStageValues[stage]);
      EXPECT_EQ(record.tag_time.logical_ns, source_tag.logical_ns);
      EXPECT_EQ(record.tag_time.microstep,
                source_tag.microstep + static_cast<std::uint32_t>(stage));
    }
  }
  expect_fast_chain_controller_evidence(evidence);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierAttempts),
            kFrameCount * 3U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierAccepted),
            kFrameCount * 3U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierFallbacks),
            0U);
}

class InlineTensorContinuationTest : public testing::TestWithParam<QueuePolicy> {};

// REQ: CHAN-128
// REQ: RUNTIME-041
TEST_P(InlineTensorContinuationTest, RawTensorPassThroughContinuesCausalFrameBeforeNextClockTick) {
  constexpr std::size_t kFrameCount = 3U;
  InlinePartitionProbeState::reset();

  // Key setup: place a 1 ns source, pass-through relay, and sink in one direct raw-Tensor partition
  // with capacity-one edges so later root frames continuously compete with channel readiness.
  // [source] -- Tensor --> [relay] -- Tensor --> [sink]
  Graph graph{"inline-causal-tensor-chain"};
  auto source = graph.op<CausalTensorSource>("source", kFrameCount);
  auto relay = graph.op<CausalTensorPassThrough>("relay");
  auto sink = graph.op<CausalTensorSink>("sink");
  graph.add_flow(source->output,
                 relay->input,
                 ConnectionOptions{.queue_policy = GetParam(), .queue_depth = 1U});
  graph.add_flow(
      relay->output, sink->input, ConnectionOptions{.queue_policy = GetParam(), .queue_depth = 1U});
  graph.partition("inline-causal-tensor-complete-cover").include_all();
  graph.set_default_clock(graph.add_clock<RealtimeClock>("inline-causal-tensor-clock"));

  const ExecutionPlan compiled = compile(graph);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_EQ(state->partition_runners.size(), 1U) << compiled.json();

  const InlinePartitionRunEvidence evidence = run_and_observe(compiled, kFrameCount);
  // Key assertion: two receipts per frame preserve source->relay->sink order at successive
  // microsteps before the next source frame, without a causal-frontier fallback.
  ASSERT_TRUE(evidence.observed);
  EXPECT_EQ(evidence.termination, RunTermination::kStopped);
  EXPECT_EQ(evidence.diagnostic_count, 0U);
  EXPECT_EQ(evidence.sink_values, (std::vector<std::uint32_t>{0U, 1U, 2U}));
  ASSERT_EQ(evidence.compute_records.size(), kFrameCount * 3U);
  for (std::size_t frame = 0U; frame < kFrameCount; ++frame) {
    const std::size_t source_index = frame * 3U;
    const TagTime source_tag = evidence.compute_records[source_index].tag_time;
    ASSERT_TRUE(source_tag.valid());
    ASSERT_LE(source_tag.microstep, std::numeric_limits<std::uint32_t>::max() - 2U);
    for (std::size_t stage = 0U; stage < 3U; ++stage) {
      const auto& record = evidence.compute_records[source_index + stage];
      SCOPED_TRACE(frame);
      SCOPED_TRACE(stage);
      EXPECT_EQ(record.operator_id, stage + 1U);
      EXPECT_EQ(record.value, frame);
      EXPECT_EQ(record.tag_time.logical_ns, source_tag.logical_ns);
      EXPECT_EQ(record.tag_time.microstep,
                source_tag.microstep + static_cast<std::uint32_t>(stage));
    }
  }
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kDirectContinuationReceiptsAccepted),
            2U * kFrameCount);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierAttempts),
            2U * kFrameCount);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierAccepted),
            2U * kFrameCount);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierFallbacks),
            0U);
}

// REQ: RUNTIME-041
TEST_P(InlineTensorContinuationTest,
       TensorPacketPassThroughContinuesCausalFrameBeforeNextClockTick) {
  constexpr std::size_t kFrameCount = 3U;
  InlinePartitionProbeState::reset();

  // Key setup: forward owning one-Tensor schema samples through a direct three-Operator partition
  // while a 1 ns source keeps later root frames ready.
  // [source] -- TensorPacket --> [relay] -- TensorPacket --> [sink]
  Graph graph{"inline-causal-tensor-packet-chain"};
  auto source = graph.op<CausalTensorPacketSource>("source", kFrameCount);
  auto relay = graph.op<CausalTensorPacketPassThrough>("relay");
  auto sink = graph.op<CausalTensorPacketSink>("sink");
  graph.add_flow(source->output,
                 relay->input,
                 ConnectionOptions{.queue_policy = GetParam(), .queue_depth = 1U});
  graph.add_flow(
      relay->output, sink->input, ConnectionOptions{.queue_policy = GetParam(), .queue_depth = 1U});
  graph.partition("inline-causal-tensor-packet-complete-cover").include_all();
  graph.set_default_clock(graph.add_clock<RealtimeClock>("inline-causal-tensor-packet-clock"));

  const ExecutionPlan compiled = compile(graph);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_EQ(state->partition_runners.size(), 1U) << compiled.json();

  const InlinePartitionRunEvidence evidence = run_and_observe(compiled, kFrameCount);
  // Key assertion: schema publication produces one receipt per edge, so each TensorPacket frame
  // drains at successive microsteps before the next root frame starts.
  ASSERT_TRUE(evidence.observed);
  EXPECT_EQ(evidence.termination, RunTermination::kStopped);
  EXPECT_EQ(evidence.diagnostic_count, 0U);
  EXPECT_EQ(evidence.sink_values, (std::vector<std::uint32_t>{0U, 1U, 2U}));
  ASSERT_EQ(evidence.compute_records.size(), kFrameCount * 3U);
  for (std::size_t frame = 0U; frame < kFrameCount; ++frame) {
    const std::size_t source_index = frame * 3U;
    const TagTime source_tag = evidence.compute_records[source_index].tag_time;
    ASSERT_TRUE(source_tag.valid());
    ASSERT_LE(source_tag.microstep, std::numeric_limits<std::uint32_t>::max() - 2U);
    for (std::size_t stage = 0U; stage < 3U; ++stage) {
      const auto& record = evidence.compute_records[source_index + stage];
      SCOPED_TRACE(frame);
      SCOPED_TRACE(stage);
      EXPECT_EQ(record.operator_id, stage + 1U);
      EXPECT_EQ(record.value, frame);
      EXPECT_EQ(record.tag_time.logical_ns, source_tag.logical_ns);
      EXPECT_EQ(record.tag_time.microstep,
                source_tag.microstep + static_cast<std::uint32_t>(stage));
    }
  }
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kDirectContinuationReceiptsAccepted),
            2U * kFrameCount);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierAttempts),
            2U * kFrameCount);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierAccepted),
            2U * kFrameCount);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierFallbacks),
            0U);
}

// Once the causal path destructively accepts a staged receipt, freeze failure is terminal. The
// controller must retire the sole pending coordinator credit instead of attempting an ordinary
// readiness poll that could neither reconstruct nor safely replay the consumed receipt.
// REQ: RUNTIME-041
TEST(RuntimeInlinePartitionExecution,
     CausalFrontierFreezeFailureIsTerminalAndRetiresCoordinatorCredit) {
  // Key setup: force causal-frontier freezing to fail after the source receipt is consumed.
  InlinePartitionProbeState::reset();
  ExecutionPlan plan = make_pod_chain_plan(QueuePolicy::kBlocking);
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(state);
  ASSERT_TRUE(state->inline_partition_execution.has_value()) << plan.json();

  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const auto control = std::make_shared<detail::RuntimeDirectContinuationTestControl>();
  control->force_causal_freeze_failure();
  plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
  plan = detail::RuntimeTestPlanAccess::with_direct_continuation_test_control(plan, control);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool cleanup_recorded = control->wait_until_causal_freeze_failure_cleanup();
  if (!cleanup_recorded) {
    session.request_stop();
  }
  session.wait();

  // Key assertion: the run fails without polling fallback and retires its sole coordinator credit.
  ASSERT_TRUE(cleanup_recorded);
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_READINESS_BINDING_UNAVAILABLE");
  EXPECT_EQ(session.diagnostics().front().service, "partition_causal_frontier_freeze");
  const Diagnostic& diagnostic = session.diagnostics().front();
  const auto vertices = plan.graph().vertices();
  const auto successor = std::ranges::find(vertices, "first-relay", &VertexRecord::name);
  ASSERT_NE(successor, vertices.end());
  EXPECT_EQ(diagnostic.source.vertex, successor->id);
  if (!diagnostic.detail.has_value()) {
    FAIL() << "Expected diagnostic detail";
  }
  const auto& detail = *diagnostic.detail;
  EXPECT_EQ(detail.error.code, ErrorCode::kProtocolError);
  EXPECT_TRUE(detail.error.message.empty());
  EXPECT_TRUE(detail.causes.empty());
  EXPECT_FALSE(detail.truncated);
  EXPECT_TRUE(InlinePartitionProbeState::sink_values().empty());
  const auto records = InlinePartitionProbeState::compute_records();
  ASSERT_EQ(records.size(), 1U);
  EXPECT_EQ(records.front().operator_id, 1U);

  const auto cleanup = control->causal_freeze_failure_observation();
  ASSERT_TRUE(cleanup.recorded);
  EXPECT_EQ(cleanup.pending_before_cancel, 1U);
  EXPECT_EQ(cleanup.outstanding_before_cancel, 1U);
  EXPECT_EQ(cleanup.pending_after_cancel, 0U);
  EXPECT_EQ(cleanup.outstanding_after_cancel, 0U);
  EXPECT_FALSE(cleanup.receipt_present_after_cancel);
  EXPECT_TRUE(cleanup.retired_all);

  const auto snapshot = counters->snapshot();
  const auto count = [&](const detail::RuntimePhaseCounter counter) {
    return snapshot[detail::RuntimePhaseCounters::index(counter)];
  };
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kChannelEmitAccepted), 1U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kChannelSamplesReceived), 1U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectContinuationReceiptsAccepted), 1U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectContinuationPollElisions), 1U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kPartitionCausalFrontierAttempts), 1U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kPartitionCausalFrontierAccepted), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kPartitionCausalFrontierFallbacks), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kActivationsCompleted), 1U);
}

INSTANTIATE_TEST_SUITE_P(QueuePolicies, InlinePodChainTest,
                         testing::Values(QueuePolicy::kBlocking, QueuePolicy::kDropOldest),
                         [](const testing::TestParamInfo<QueuePolicy>& info) {
                           return info.param == QueuePolicy::kBlocking ? "Blocking" : "DropOldest";
                         });

INSTANTIATE_TEST_SUITE_P(QueuePolicies, InlineTensorContinuationTest,
                         testing::Values(QueuePolicy::kBlocking, QueuePolicy::kDropOldest),
                         [](const testing::TestParamInfo<QueuePolicy>& info) {
                           return info.param == QueuePolicy::kBlocking ? "Blocking" : "DropOldest";
                         });

TEST(RuntimeInlinePartitionExecution,
     CachedClockRootCoordinateRefreshesAcrossConsecutiveCausalChains) {
  // Key setup: run three clock activations through the same cached four-Operator POD chain.
  InlinePartitionProbeState::reset();
  constexpr std::size_t kActivationCount = 3U;
  constexpr std::size_t kOperatorsPerActivation = 4U;
  const ExecutionPlan compiled =
      make_pod_chain_plan(QueuePolicy::kBlocking, false, kActivationCount, 1ms);
  ASSERT_TRUE(compiled.ok()) << compiled.json();

  const InlinePartitionRunEvidence evidence = run_and_observe(compiled, kActivationCount);
  // Key assertion: each chain has a newer root coordinate and internally consecutive microsteps.
  ASSERT_TRUE(evidence.observed);
  EXPECT_EQ(evidence.termination, RunTermination::kStopped);
  EXPECT_EQ(evidence.diagnostic_count, 0U);
  EXPECT_EQ(evidence.sink_values,
            (std::vector<std::uint32_t>(kActivationCount, std::uint32_t{44U})));
  ASSERT_EQ(evidence.compute_records.size(), kActivationCount * kOperatorsPerActivation);

  std::optional<std::int64_t> previous_root_logical_ns;
  for (std::size_t activation = 0U; activation < kActivationCount; ++activation) {
    const std::size_t begin = activation * kOperatorsPerActivation;
    const TagTime root_tag_time = evidence.compute_records[begin].tag_time;
    ASSERT_TRUE(root_tag_time.valid());
    if (previous_root_logical_ns) {
      EXPECT_GT(root_tag_time.logical_ns, *previous_root_logical_ns);
    }
    previous_root_logical_ns = root_tag_time.logical_ns;
    for (std::size_t member = 0U; member < kOperatorsPerActivation; ++member) {
      SCOPED_TRACE(activation);
      SCOPED_TRACE(member);
      const auto& record = evidence.compute_records[begin + member];
      EXPECT_EQ(record.operator_id, static_cast<std::uint32_t>(member + 1U));
      EXPECT_EQ(record.tag_time.logical_ns, root_tag_time.logical_ns);
      EXPECT_EQ(record.tag_time.microstep,
                root_tag_time.microstep + static_cast<std::uint32_t>(member));
    }
  }
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierAttempts),
            9U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierAccepted),
            9U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierFallbacks),
            0U);
}

void expect_tensor_chain_observation(const InlineTensorObservation& observation,
                                     MemoryKind memory_kind,
                                     std::optional<std::int32_t> device = std::nullopt) {
  ASSERT_TRUE(observation.source_recorded);
  ASSERT_TRUE(observation.relay_recorded);
  EXPECT_EQ(observation.memory_kind, memory_kind);
  EXPECT_NE(observation.source_pointer, nullptr);
  EXPECT_EQ(observation.relay_pointer, observation.source_pointer);
  EXPECT_EQ(observation.value, 0x5AU);
  EXPECT_EQ(observation.relay_stream, observation.source_stream);
  EXPECT_EQ(observation.readiness_stream, observation.source_stream);
  if (device) {
    EXPECT_NE(observation.source_stream, nullptr);
    EXPECT_EQ(observation.source_device, *device);
    EXPECT_EQ(observation.relay_device, *device);
  }
}

template <typename PayloadCase>
class InlinePayloadParity : public testing::Test {};

using InlinePayloadCases = testing::Types<RawTensorCase, DescriptorImageCase>;

struct InlinePayloadCaseNames {
  template <typename PayloadCase>
  static std::string GetName(int) {
    return std::string{PayloadCase::name};
  }
};

TYPED_TEST_SUITE(InlinePayloadParity, InlinePayloadCases, InlinePayloadCaseNames);

TYPED_TEST(InlinePayloadParity, HostPreservesBackingMetadataAndControllerOwnership) {
  // Key setup: run the host raw-Tensor or descriptor-Image chain under both queue policies.
  for (const QueuePolicy policy : {QueuePolicy::kBlocking, QueuePolicy::kDropOldest}) {
    SCOPED_TRACE(static_cast<std::uint32_t>(policy));
    InlinePartitionProbeState::reset();
    InlineTensorProbeState::reset(MemoryKind::kHost);
    const ExecutionPlan compiled =
        TypeParam::template make_plan<MemoryKind::kHost>({.policy = policy});
    ASSERT_TRUE(compiled.ok()) << compiled.json();
    const auto state = detail::ExecutionPlanAccess::retain(compiled);
    ASSERT_TRUE(state);
    ASSERT_TRUE(state->inline_partition_execution.has_value()) << compiled.json();

    const InlinePartitionRunEvidence evidence = run_and_observe(compiled, 1U);
    // Key assertion: payload identity and metadata survive while all callbacks stay inline.
    ASSERT_TRUE(evidence.observed);
    EXPECT_EQ(evidence.termination, RunTermination::kStopped);
    EXPECT_EQ(evidence.diagnostic_count, 0U);
    EXPECT_EQ(evidence.sink_values, (std::vector<std::uint32_t>{0x5AU}));
    TypeParam::expect_payload_evidence(evidence);
    expect_tensor_chain_observation(InlineTensorProbeState::observation(), MemoryKind::kHost);
    expect_inline_thread_evidence(evidence, 3U, 3U);
  }
}

TYPED_TEST(InlinePayloadParity, CudaReentersPlacedDeviceAndStreamForEveryCallback) {
  // Key setup: run the CUDA payload chain on each tested device after changing caller affinity.
  std::int32_t device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }
  std::int32_t original_device = -1;
  ASSERT_EQ(cudaGetDevice(&original_device), cudaSuccess);
  const ScopedCUDADeviceRestore restore_device{original_device};

  const std::int32_t tested_device_count = std::min(device_count, 2);
  for (std::int32_t device = 0; device < tested_device_count; ++device) {
    for (const QueuePolicy policy : {QueuePolicy::kBlocking, QueuePolicy::kDropOldest}) {
      SCOPED_TRACE(device);
      SCOPED_TRACE(static_cast<std::uint32_t>(policy));
      const std::int32_t ambient_device = device_count > 1 ? (device + 1) % device_count : device;
      ASSERT_EQ(cudaSetDevice(ambient_device), cudaSuccess);
      InlinePartitionProbeState::reset();
      InlineTensorProbeState::reset(MemoryKind::kCudaDevice);
      const ExecutionPlan compiled = TypeParam::template make_plan<MemoryKind::kCudaDevice>({
          .policy = policy,
          .device = DeviceId{device},
          .ambient_device_after_emit = ambient_device,
      });
      ASSERT_TRUE(compiled.ok()) << compiled.json();
      const auto state = detail::ExecutionPlanAccess::retain(compiled);
      ASSERT_TRUE(state);
      ASSERT_TRUE(state->inline_partition_execution.has_value()) << compiled.json();
      EXPECT_EQ(compiled.cuda_compute_stream_policy(),
                CudaComputeStreamPolicy::kPerExecutionPartition);
      ASSERT_EQ(compiled.cuda_compute_streams().size(), 1U);
      ASSERT_TRUE(compiled.cuda_compute_streams().front().partition.has_value());
      ASSERT_FALSE(compiled.operator_cuda_stream_bindings().empty());
      EXPECT_TRUE(
          std::ranges::all_of(compiled.operator_cuda_stream_bindings(), [&](const auto& binding) {
            return binding.stream_id == compiled.cuda_compute_streams().front().stream_id &&
                   binding.partition == *compiled.cuda_compute_streams().front().partition;
          }));

      const InlinePartitionRunEvidence evidence = run_and_observe(compiled, 1U);
      // Key assertion: each callback re-enters the placed device and shares the source stream.
      ASSERT_TRUE(evidence.observed);
      EXPECT_EQ(evidence.termination, RunTermination::kStopped);
      EXPECT_EQ(evidence.diagnostic_count, 0U);
      EXPECT_EQ(evidence.sink_values, (std::vector<std::uint32_t>{0x5AU}));
      TypeParam::expect_payload_evidence(evidence);
      expect_tensor_chain_observation(
          InlineTensorProbeState::observation(), MemoryKind::kCudaDevice, device);
      expect_inline_thread_evidence(evidence, 3U, 3U);
      std::int32_t caller_device = -1;
      ASSERT_EQ(cudaGetDevice(&caller_device), cudaSuccess);
      EXPECT_EQ(caller_device, ambient_device);
    }
  }
}

TYPED_TEST(InlinePayloadParity, RetainedHandleKeepsPoolSlotUntilRelease) {
  // Key setup: retain the host Tensor at the relay while observing its runtime output pool.
  InlinePartitionProbeState::reset();
  InlineTensorProbeState::reset(MemoryKind::kHost);
  const ExecutionPlan compiled = TypeParam::template make_plan<MemoryKind::kHost>({
      .retain_tensor = true,
  });
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_TRUE(state->inline_partition_execution.has_value()) << compiled.json();

  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const auto support_control = std::make_shared<InlinePartitionControllerProbe>();
  const auto tensor_control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  ExecutionPlan run_plan =
      detail::RuntimeTestPlanAccess::with_tensor_output_test_control(compiled, tensor_control);
  run_plan = detail::RuntimeTestPlanAccess::with_phase_counters(run_plan, counters);
  run_plan = detail::RuntimeTestPlanAccess::with_support_thread_control(run_plan, support_control);

  std::optional<RunSession> session{run_async(run_plan)};
  ASSERT_TRUE(InlinePartitionProbeState::wait_for_sinks(1U));
  // Key assertion: the slot remains claimed until handle release, then retires without leaks.
  ASSERT_TRUE(InlineTensorProbeState::has_retained_tensor());
  const auto retained = tensor_control->snapshot(0U);
  EXPECT_EQ(retained.active_slot_count, 1U);
  EXPECT_NE(retained.active_claim_mask, 0U);

  InlineTensorProbeState::release_retained_tensor();
  session->request_stop();
  session->wait();
  const InlinePartitionRunEvidence evidence =
      capture_run_evidence(*session, true, counters, support_control);
  EXPECT_EQ(evidence.termination, RunTermination::kStopped);
  EXPECT_EQ(evidence.diagnostic_count, 0U);
  EXPECT_EQ(evidence.sink_values, (std::vector<std::uint32_t>{0x5AU}));
  TypeParam::expect_payload_evidence(evidence);
  expect_inline_thread_evidence(evidence, 3U, 3U);

  session.reset();
  const auto released = tensor_control->snapshot(0U);
  EXPECT_TRUE(released.terminal);
  EXPECT_EQ(released.active_slot_count, 0U);
  EXPECT_EQ(released.active_claim_mask, 0U);
  EXPECT_EQ(released.invariant_failure_count, 0U);
}

enum class RuntimeTensorMismatchMode {
  kPropagate,
  kHandle,
  kFirstRejectedPort,
  kUnrelatedErrorNextActivation,
  kUnrelatedErrorSameActivation,
};

class RuntimeTensorMismatchSource final : public Operator<> {
 public:
  explicit RuntimeTensorMismatchSource(RuntimeTensorMismatchMode mode) : mode_(mode) {}
  void setup(OperatorSpec& spec) override {
    const auto declare_output = [&](Output<Tensor>& port, const char* name) {
      spec.output(port, name)
          .max_emits_per_compute(1U)
          .produces_tensor(TensorOutputSpec{
              .representation = {.memory_kind = MemoryKind::kCudaDevice,
                                 .dtype = DLDataType{kDLUInt, 8U, 1U},
                                 .rank = 1U},
              .bounds = tensor_bounds(8U),
              .storage = TensorOutputStorage::kRuntimePool,
          });
    };
    declare_output(output, "output");
    if (mode_ == RuntimeTensorMismatchMode::kFirstRejectedPort) {
      declare_output(first_rejected, "z_first_rejected");
    }
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    if (emitted_) {
      if (mode_ == RuntimeTensorMismatchMode::kUnrelatedErrorNextActivation) {
        return make_unexpected(
            Error{ErrorCode::kNotSupported, "unrelated failure in next activation"});
      }
      context.request_completion();
      return {};
    }
    emitted_ = true;
    constexpr std::array<std::int64_t, 1U> shape{8};
    auto storage = std::make_unique<std::array<std::uint8_t, 8U>>();
    auto* const storage_owner = storage.get();
    auto tensor = make_tensor(shape,
                              DLDataType{kDLUInt, 8U, 1U},
                              std::nullopt,
                              MemoryStorageType::kSystem,
                              storage->data(),
                              [storage_owner](void*) -> expected<void, Error> {
                                delete storage_owner;
                                deleted.fetch_add(1U, std::memory_order_relaxed);
                                return {};
                              });
    if (!tensor) {
      return make_unexpected(std::move(tensor.error()));
    }
    EXPECT_EQ(storage.release(), storage_owner);
    Sample<Tensor> sample{std::move(*tensor), {}};
    expected<void, Error> first;
    if (mode_ == RuntimeTensorMismatchMode::kFirstRejectedPort) {
      first = first_rejected.emit(sample);
      if (!first) {
        rejected.fetch_add(1U, std::memory_order_relaxed);
      }
    }
    auto emitted = output.emit(std::move(sample));
    if (!emitted) {
      rejected.fetch_add(1U, std::memory_order_relaxed);
    }
    if (mode_ == RuntimeTensorMismatchMode::kUnrelatedErrorSameActivation) {
      return make_unexpected(
          Error{ErrorCode::kNotSupported, "unrelated failure in same activation"});
    }
    if (mode_ == RuntimeTensorMismatchMode::kHandle ||
        mode_ == RuntimeTensorMismatchMode::kUnrelatedErrorNextActivation) {
      return {};
    }
    return mode_ == RuntimeTensorMismatchMode::kFirstRejectedPort ? std::move(first)
                                                                  : std::move(emitted);
  }
  Output<Tensor> output;
  Output<Tensor> first_rejected;
  static inline std::atomic_uint32_t deleted{};
  static inline std::atomic_uint32_t rejected{};

 private:
  RuntimeTensorMismatchMode mode_;
  bool emitted_{};
};

class RuntimeTensorMismatchSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{
            .representation = {.memory_kind = MemoryKind::kCudaDevice,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 1U},
        });
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override {
    received.fetch_add(1U, std::memory_order_relaxed);
    return {};
  }
  Input<Tensor> input;
  static inline std::atomic_uint32_t received{};
};

// REQ: RUNTIME-054
// REQ: RUNTIME-060
TEST(RuntimeInlinePartitionExecution, RuntimeTensorMismatchRetainsPortMemoryAndCleanup) {
  std::int32_t device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }
  for (const auto mode : {RuntimeTensorMismatchMode::kPropagate,
                          RuntimeTensorMismatchMode::kHandle,
                          RuntimeTensorMismatchMode::kFirstRejectedPort,
                          RuntimeTensorMismatchMode::kUnrelatedErrorNextActivation,
                          RuntimeTensorMismatchMode::kUnrelatedErrorSameActivation}) {
    SCOPED_TRACE(static_cast<int>(mode));
    const bool two_ports = mode == RuntimeTensorMismatchMode::kFirstRejectedPort;
    RuntimeTensorMismatchSource::deleted.store(0U, std::memory_order_relaxed);
    RuntimeTensorMismatchSource::rejected.store(0U, std::memory_order_relaxed);
    RuntimeTensorMismatchSink::received.store(0U, std::memory_order_relaxed);
    Graph graph{"runtime-tensor-mismatch"};
    const auto source = graph.op<RuntimeTensorMismatchSource>("source", mode);
    const auto sink = graph.op<RuntimeTensorMismatchSink>("sink");
    graph.add_flow(source->output, sink->input);
    if (two_ports) {
      const auto second_sink = graph.op<RuntimeTensorMismatchSink>("second_sink");
      graph.add_flow(source->first_rejected, second_sink->input);
    }
    graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
    CompileOptions options;
    options.deployment.bind_tensor_output_device({"source", "output", DeviceId{0}});
    if (two_ports) {
      options.deployment.bind_tensor_output_device({"source", "z_first_rejected", DeviceId{0}});
    }
    const auto compiled = compile(graph, options);
    ASSERT_TRUE(compiled.ok()) << compiled.json();
    const auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
    const auto plan =
        detail::RuntimeTestPlanAccess::with_tensor_output_test_control(compiled, control);
    std::optional<RunSession> session{run_async(plan)};
    const bool terminal = wait_for_terminal(*session);
    if (!terminal) {
      session->request_stop();
    }
    session->wait();
    ASSERT_TRUE(terminal) << "Run did not terminate within 5 seconds";
    EXPECT_EQ(RuntimeTensorMismatchSource::rejected.load(std::memory_order_relaxed),
              two_ports ? 2U : 1U);
    EXPECT_EQ(RuntimeTensorMismatchSource::deleted.load(std::memory_order_relaxed), 1U);
    EXPECT_EQ(RuntimeTensorMismatchSink::received.load(std::memory_order_relaxed), 0U);
    if (mode == RuntimeTensorMismatchMode::kHandle) {
      EXPECT_EQ(session->termination(), RunTermination::kRequestedCompleted);
      EXPECT_TRUE(session->diagnostics().empty());
    } else {
      EXPECT_EQ(session->termination(), RunTermination::kFailed);
      const auto diagnostics = session->diagnostics();
      const auto found = std::find_if(diagnostics.begin(), diagnostics.end(), [](const auto& row) {
        return row.token == "RUNTIME_OPERATOR_CALLBACK_FAILED";
      });
      ASSERT_NE(found, diagnostics.end());
      const auto& failure_detail = found->detail;
      if (!failure_detail.has_value()) {
        FAIL() << "Expected diagnostic detail";
      }
      EXPECT_EQ(failure_detail->operation, DiagnosticOperation::kCompute);
      EXPECT_EQ(failure_detail->operator_path, "source");
      EXPECT_TRUE(failure_detail->causes.empty());
      if (mode == RuntimeTensorMismatchMode::kUnrelatedErrorNextActivation ||
          mode == RuntimeTensorMismatchMode::kUnrelatedErrorSameActivation) {
        EXPECT_EQ(failure_detail->error.code, ErrorCode::kNotSupported);
        EXPECT_EQ(failure_detail->error.message,
                  mode == RuntimeTensorMismatchMode::kUnrelatedErrorNextActivation
                      ? "unrelated failure in next activation"
                      : "unrelated failure in same activation");
      } else {
        EXPECT_EQ(failure_detail->error.code, ErrorCode::kInvalidArgument);
      }
      if (mode == RuntimeTensorMismatchMode::kUnrelatedErrorNextActivation) {
        EXPECT_FALSE(found->source.port);
        EXPECT_TRUE(failure_detail->port_name.empty());
        EXPECT_TRUE(failure_detail->mismatches.empty());
      } else {
        const std::string expected_port = two_ports ? "z_first_rejected" : "output";
        EXPECT_EQ(failure_detail->port_name, expected_port);
        EXPECT_TRUE(found->source.port);
        const auto& mismatches = failure_detail->mismatches;
        ASSERT_EQ(mismatches.size(), 1U);
        EXPECT_EQ(mismatches.front().field, "memory_kind");
        EXPECT_EQ(mismatches.front().provided, "host");
        EXPECT_EQ(mismatches.front().required, "cuda_device");
        const auto formatted = format_diagnostics(diagnostics, ExecutionGraphView{});
        EXPECT_NE(formatted.find(expected_port), std::string::npos);
        EXPECT_NE(formatted.find("host"), std::string::npos);
        EXPECT_NE(formatted.find("cuda_device"), std::string::npos);
      }
    }
    session.reset();
    for (std::size_t pool_index = 0U; pool_index < (two_ports ? 2U : 1U); ++pool_index) {
      const auto pool = control->snapshot(pool_index);
      EXPECT_TRUE(pool.terminal);
      EXPECT_EQ(pool.active_slot_count, 0U);
      EXPECT_EQ(pool.active_claim_mask, 0U);
      EXPECT_EQ(pool.cuda_streams_created, pool.cuda_streams_destroyed);
      EXPECT_EQ(pool.cuda_events_created, pool.cuda_events_destroyed);
      EXPECT_EQ(pool.invariant_failure_count, 0U);
    }
  }
}

TEST(RuntimeInlinePartitionExecution,
     RawCudaTensorFailureCleansEveryDeviceWithoutLeakingCallerDevice) {
  // Key setup: fail a placed CUDA Tensor relay after deliberately changing the ambient device.
  std::int32_t device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }
  std::int32_t original_device = -1;
  ASSERT_EQ(cudaGetDevice(&original_device), cudaSuccess);
  const ScopedCUDADeviceRestore restore_device{original_device};

  const std::int32_t tested_device_count = std::min(device_count, 2);
  for (std::int32_t device = 0; device < tested_device_count; ++device) {
    SCOPED_TRACE(device);
    const std::int32_t ambient_device = device_count > 1 ? (device + 1) % device_count : device;
    ASSERT_EQ(cudaSetDevice(ambient_device), cudaSuccess);
    InlinePartitionProbeState::reset();
    InlineTensorProbeState::reset(MemoryKind::kCudaDevice);
    const ExecutionPlan compiled =
        make_tensor_chain_plan<MemoryKind::kCudaDevice>(QueuePolicy::kBlocking,
                                                        TensorPlanRoute::kInline,
                                                        DeviceId{device},
                                                        ambient_device,
                                                        true,
                                                        false,
                                                        ambient_device);
    ASSERT_TRUE(compiled.ok()) << compiled.json();
    const auto state = detail::ExecutionPlanAccess::retain(compiled);
    ASSERT_TRUE(state);
    ASSERT_TRUE(state->inline_partition_execution.has_value()) << compiled.json();

    const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
    const auto support_control = std::make_shared<InlinePartitionControllerProbe>();
    const auto tensor_control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
    ExecutionPlan run_plan =
        detail::RuntimeTestPlanAccess::with_tensor_output_test_control(compiled, tensor_control);
    run_plan = detail::RuntimeTestPlanAccess::with_phase_counters(run_plan, counters);
    run_plan =
        detail::RuntimeTestPlanAccess::with_support_thread_control(run_plan, support_control);

    std::optional<RunSession> session{run_async(run_plan)};
    session->wait();
    const InlinePartitionRunEvidence evidence =
        capture_run_evidence(*session, true, counters, support_control);
    // Key assertion: failure releases all CUDA resources and restores the caller's device.
    EXPECT_EQ(evidence.termination, RunTermination::kFailed);
    ASSERT_EQ(session->diagnostics().size(), 1U);
    EXPECT_EQ(session->diagnostics().front().token, "RUNTIME_OPERATOR_CALLBACK_FAILED");
    expect_tensor_chain_observation(
        InlineTensorProbeState::observation(), MemoryKind::kCudaDevice, device);
    EXPECT_TRUE(InlineTensorProbeState::observation().stop_device_recorded);
    EXPECT_GE(InlineTensorProbeState::observation().stop_device, 0);
    EXPECT_LT(InlineTensorProbeState::observation().stop_device, device_count);
    expect_inline_thread_evidence(evidence, 2U, 3U);

    session.reset();
    const auto pool = tensor_control->snapshot(0U);
    EXPECT_TRUE(pool.terminal);
    EXPECT_EQ(pool.active_slot_count, 0U);
    EXPECT_EQ(pool.active_claim_mask, 0U);
    EXPECT_EQ(pool.cuda_streams_created, pool.cuda_streams_destroyed);
    EXPECT_EQ(pool.cuda_events_created, pool.cuda_events_destroyed);
    EXPECT_EQ(pool.invariant_failure_count, 0U);
    std::int32_t caller_device = -1;
    ASSERT_EQ(cudaGetDevice(&caller_device), cudaSuccess);
    EXPECT_EQ(caller_device, ambient_device);
  }
}

// REQ: RUNTIME-054
// REQ: RUNTIME-060
TEST(RuntimeInlinePartitionExecution, CudaRestoreFailureIsSeparateFromCallbackCause) {
  std::int32_t device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }
  std::int32_t original_device = -1;
  ASSERT_EQ(cudaGetDevice(&original_device), cudaSuccess);
  const ScopedCUDADeviceRestore restore_device{original_device};
  const std::int32_t ambient_device = device_count > 1 ? 1 : 0;
  for (const bool fail_callback : {true, false}) {
    SCOPED_TRACE(fail_callback ? "callback and restore" : "restore only");
    ASSERT_EQ(cudaSetDevice(ambient_device), cudaSuccess);
    InlinePartitionProbeState::reset();
    InlineTensorProbeState::reset(MemoryKind::kCudaDevice);
    const auto compiled = make_tensor_chain_plan<MemoryKind::kCudaDevice>(QueuePolicy::kBlocking,
                                                                          TensorPlanRoute::kInline,
                                                                          DeviceId{0},
                                                                          ambient_device,
                                                                          fail_callback,
                                                                          false,
                                                                          ambient_device);
    ASSERT_TRUE(compiled.ok()) << compiled.json();
    const auto fault = std::make_shared<detail::RuntimeCudaExecutionTestControl>(
        detail::RuntimeCudaExecutionFailurePoint::kComputeDeviceRestore, 1U);
    const auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
    auto plan = detail::RuntimeTestPlanAccess::with_cuda_execution_test_control(compiled, fault);
    plan = detail::RuntimeTestPlanAccess::with_tensor_output_test_control(plan, control);
    std::optional<RunSession> session{run_async(plan)};
    const bool terminal = wait_for_terminal(*session);
    if (!terminal) {
      session->request_stop();
    }
    session->wait();
    ASSERT_TRUE(terminal) << "Run did not terminate within 5 seconds";
    EXPECT_GE(fault->visits(), 2U);
    EXPECT_EQ(session->termination(), RunTermination::kFailed);
    const auto diagnostics = session->diagnostics();
    ASSERT_EQ(diagnostics.size(), fail_callback ? 2U : 1U);
    const auto& restoration = diagnostics.back();
    EXPECT_EQ(restoration.token, "RUNTIME_COMPUTE_CUDA_DEVICE_RESTORE_FAILED");
    const auto& restoration_detail = restoration.detail;
    if (!restoration_detail.has_value()) {
      FAIL() << "Expected diagnostic detail";
    }
    EXPECT_EQ(restoration_detail->operation, DiagnosticOperation::kRuntime);
    EXPECT_EQ(restoration_detail->operator_path, "relay");
    EXPECT_EQ(restoration_detail->error.code, ErrorCode::kNotSupported);
    EXPECT_TRUE(restoration_detail->causes.empty());
    if (fail_callback) {
      const auto& callback = diagnostics.front();
      EXPECT_EQ(callback.token, "RUNTIME_OPERATOR_CALLBACK_FAILED");
      const auto& callback_detail = callback.detail;
      if (!callback_detail.has_value()) {
        FAIL() << "Expected diagnostic detail";
      }
      EXPECT_EQ(callback_detail->operation, DiagnosticOperation::kCompute);
      EXPECT_EQ(callback_detail->error.message, "INLINE_TENSOR_CALLBACK_FAILURE");
      EXPECT_TRUE(callback_detail->causes.empty());
      EXPECT_EQ(restoration_detail->role, DiagnosticFailureRole::kTeardown);
      EXPECT_EQ(restoration_detail->primary_diagnostic_index, 0U);
    } else {
      EXPECT_EQ(restoration_detail->role, DiagnosticFailureRole::kPrimary);
      EXPECT_FALSE(restoration_detail->primary_diagnostic_index);
    }
    session.reset();
    const auto pool = control->snapshot(0U);
    EXPECT_TRUE(pool.terminal);
    EXPECT_EQ(pool.active_slot_count, 0U);
    EXPECT_EQ(pool.active_claim_mask, 0U);
    EXPECT_EQ(pool.cuda_streams_created, pool.cuda_streams_destroyed);
    EXPECT_EQ(pool.cuda_events_created, pool.cuda_events_destroyed);
    EXPECT_EQ(pool.invariant_failure_count, 0U);
    std::int32_t caller_device = -1;
    ASSERT_EQ(cudaGetDevice(&caller_device), cudaSuccess);
    EXPECT_EQ(caller_device, ambient_device);
  }
}

TEST(RuntimeInlinePartitionExecution, MixedTensorPlacementKeepsTheOneRootForkOrdinary) {
  // Key setup: compile one-root forks whose branches require mixed memory or CUDA devices.
  const ExecutionPlan host_cuda =
      make_mixed_tensor_fork_plan<MemoryKind::kHost, MemoryKind::kCudaDevice>(DeviceId{},
                                                                              DeviceId{0});
  ASSERT_TRUE(host_cuda.ok()) << host_cuda.json();
  const auto host_cuda_state = detail::ExecutionPlanAccess::retain(host_cuda);
  ASSERT_TRUE(host_cuda_state);
  // Key assertion: neither placement-incompatible fork is admitted as inline execution.
  EXPECT_FALSE(host_cuda_state->inline_partition_execution.has_value());

  const ExecutionPlan multi_device =
      make_mixed_tensor_fork_plan<MemoryKind::kCudaDevice, MemoryKind::kCudaDevice>(DeviceId{0},
                                                                                    DeviceId{1});
  ASSERT_TRUE(multi_device.ok()) << multi_device.json();
  const auto multi_device_state = detail::ExecutionPlanAccess::retain(multi_device);
  ASSERT_TRUE(multi_device_state);
  EXPECT_FALSE(multi_device_state->inline_partition_execution.has_value());
}

TYPED_TEST(InlinePayloadParity, IneligibleTransportAndMemoryRemainOrdinary) {
  // Key setup: compile equivalent payload chains over HoloMQ, fanout, and ineligible memory kinds.
  const ExecutionPlan holomq = TypeParam::template make_plan<MemoryKind::kHost>({
      .route = TensorPlanRoute::kHoloMQ,
  });
  ASSERT_TRUE(holomq.ok()) << holomq.json();
  const auto holomq_state = detail::ExecutionPlanAccess::retain(holomq);
  ASSERT_TRUE(holomq_state);
  // Key assertion: every ineligible route remains outside the inline execution fast path.
  EXPECT_FALSE(holomq_state->inline_partition_execution.has_value());

  const ExecutionPlan fanout = TypeParam::template make_plan<MemoryKind::kHost>({
      .route = TensorPlanRoute::kFanout,
  });
  ASSERT_TRUE(fanout.ok()) << fanout.json();
  const auto fanout_state = detail::ExecutionPlanAccess::retain(fanout);
  ASSERT_TRUE(fanout_state);
  EXPECT_FALSE(fanout_state->inline_partition_execution.has_value());

  for (const MemoryKind memory_kind : {MemoryKind::kPinnedHost, MemoryKind::kCudaManaged}) {
    SCOPED_TRACE(static_cast<std::uint32_t>(memory_kind));
    const ExecutionPlan plan = memory_kind == MemoryKind::kPinnedHost
                                   ? TypeParam::template make_plan<MemoryKind::kPinnedHost>({})
                                   : TypeParam::template make_plan<MemoryKind::kCudaManaged>({
                                         .device = DeviceId{0},
                                     });
    ASSERT_TRUE(plan.ok()) << plan.json();
    const auto state = detail::ExecutionPlanAccess::retain(plan);
    ASSERT_TRUE(state);
    EXPECT_FALSE(state->inline_partition_execution.has_value());
  }
}

TYPED_TEST(InlinePayloadParity, FailureReleasesPoolAndReplaysCleanly) {
  // Key setup: reuse a host payload plan whose relay fails, running it twice against fresh pools.
  const ExecutionPlan compiled = TypeParam::template make_plan<MemoryKind::kHost>({
      .fail_relay = true,
  });
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_TRUE(state->inline_partition_execution.has_value()) << compiled.json();

  for (std::size_t iteration = 0U; iteration < 2U; ++iteration) {
    SCOPED_TRACE(iteration);
    InlinePartitionProbeState::reset();
    InlineTensorProbeState::reset(MemoryKind::kHost);
    const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
    const auto support_control = std::make_shared<InlinePartitionControllerProbe>();
    const auto tensor_control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
    ExecutionPlan run_plan =
        detail::RuntimeTestPlanAccess::with_tensor_output_test_control(compiled, tensor_control);
    run_plan = detail::RuntimeTestPlanAccess::with_phase_counters(run_plan, counters);
    run_plan =
        detail::RuntimeTestPlanAccess::with_support_thread_control(run_plan, support_control);

    std::optional<RunSession> session{run_async(run_plan)};
    session->wait();
    const InlinePartitionRunEvidence evidence =
        capture_run_evidence(*session, true, counters, support_control);
    // Key assertion: each run replays the typed failure and retires every active pool claim.
    EXPECT_EQ(evidence.termination, RunTermination::kFailed);
    ASSERT_EQ(session->diagnostics().size(), 1U);
    EXPECT_EQ(session->diagnostics().front().token, "RUNTIME_OPERATOR_CALLBACK_FAILED");
    EXPECT_TRUE(evidence.sink_values.empty());
    TypeParam::expect_payload_evidence(evidence);
    expect_tensor_chain_observation(InlineTensorProbeState::observation(), MemoryKind::kHost);
    expect_inline_thread_evidence(evidence, 2U, 3U);

    session.reset();
    const auto pool = tensor_control->snapshot(0U);
    EXPECT_TRUE(pool.terminal);
    EXPECT_EQ(pool.active_slot_count, 0U);
    EXPECT_EQ(pool.active_claim_mask, 0U);
    EXPECT_EQ(pool.invariant_failure_count, 0U);
  }
}

TEST(RuntimeInlinePartitionExecution, DescriptorImageMissingAllocationRemainsOrdinary) {
  // Key setup: compile a descriptor-Image chain without runtime Tensor allocation evidence.
  const ExecutionPlan no_allocation = make_image_chain_plan<MemoryKind::kHost>(
      QueuePolicy::kBlocking, TensorPlanRoute::kInline, DeviceId{}, -1, false, false, false);
  ASSERT_TRUE(no_allocation.ok()) << no_allocation.json();
  const auto no_allocation_state = detail::ExecutionPlanAccess::retain(no_allocation);
  ASSERT_TRUE(no_allocation_state);
  // Key assertion: missing allocation proof prevents admission to inline execution.
  EXPECT_FALSE(no_allocation_state->inline_partition_execution.has_value());
}

// REQ: RUNTIME-041
TEST(RuntimeInlinePartitionExecution, RunsAnOwningSchemaChainOnTheControllerWithoutHandoffs) {
  constexpr std::size_t kFrameCount = 3U;
  InlinePartitionProbeState::reset();
  // Key setup: drive three frames through a four-Operator generated-schema chain from a 1 ns
  // source, providing the established non-Tensor continuation behavior for comparison.
  const ExecutionPlan compiled = make_schema_chain_plan(kFrameCount, 1ns);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_TRUE(state->inline_partition_execution.has_value()) << compiled.json();

  const InlinePartitionRunEvidence evidence = run_and_observe(compiled, kFrameCount);
  // Key assertion: every generated-schema frame drains at successive microsteps and all three
  // direct edges accept their causal continuations without fallback.
  ASSERT_TRUE(evidence.observed);
  EXPECT_EQ(evidence.termination, RunTermination::kStopped);
  EXPECT_EQ(evidence.diagnostic_count, 0U);
  EXPECT_EQ(evidence.sink_values, (std::vector<std::uint32_t>{0U, 1U, 2U}));
  ASSERT_TRUE(evidence.schema_value.has_value());
  // The fatal assertion above proves the optional is engaged.
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  EXPECT_EQ(*evidence.schema_value, "root:2:3");
  ASSERT_EQ(evidence.compute_records.size(), kFrameCount * 4U);
  for (std::size_t frame = 0U; frame < kFrameCount; ++frame) {
    const std::size_t source_index = frame * 4U;
    const TagTime source_tag = evidence.compute_records[source_index].tag_time;
    ASSERT_TRUE(source_tag.valid());
    ASSERT_LE(source_tag.microstep, std::numeric_limits<std::uint32_t>::max() - 3U);
    for (std::size_t stage = 0U; stage < 4U; ++stage) {
      const auto& record = evidence.compute_records[source_index + stage];
      SCOPED_TRACE(frame);
      SCOPED_TRACE(stage);
      EXPECT_EQ(record.operator_id, stage + 1U);
      EXPECT_EQ(record.value, frame);
      EXPECT_EQ(record.tag_time.logical_ns, source_tag.logical_ns);
      EXPECT_EQ(record.tag_time.microstep,
                source_tag.microstep + static_cast<std::uint32_t>(stage));
    }
  }
  expect_fast_chain_controller_evidence(evidence);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kDirectContinuationReceiptsAccepted),
            kFrameCount * 3U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierAttempts),
            kFrameCount * 3U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierAccepted),
            kFrameCount * 3U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierFallbacks),
            0U);
}

TEST(RuntimeInlinePartitionExecution, AdmitsOneRootWithTwoConsumersInStableSiblingOrder) {
  // Key setup: connect one direct POD output to two sinks in a single authored partition.
  InlinePartitionProbeState::reset();
  const ExecutionPlan compiled = make_fork_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_EQ(state->partition_runners.size(), 1U) << compiled.json();
  EXPECT_FALSE(state->inline_partition_execution.has_value());
  ASSERT_EQ(state->partition_runners.front().operators.size(), 3U);
  EXPECT_EQ(state->partition_runners.front().internal_edges.size(), 2U);

  const InlinePartitionRunEvidence evidence = run_and_observe(compiled, 2U);
  // Key assertion: the ordinary partition runner executes both siblings once in stable order.
  ASSERT_TRUE(evidence.observed);
  EXPECT_EQ(evidence.termination, RunTermination::kStopped);
  EXPECT_EQ(evidence.diagnostic_count, 0U);
  EXPECT_EQ(evidence.sink_values, (std::vector<std::uint32_t>{101U, 101U}));
  ASSERT_EQ(evidence.compute_records.size(), 3U);
  EXPECT_EQ(evidence.compute_records[0].operator_id, 1U);
  EXPECT_EQ(evidence.compute_records[1].operator_id, 2U);
  EXPECT_EQ(evidence.compute_records[2].operator_id, 3U);
  EXPECT_TRUE(std::ranges::all_of(evidence.compute_records, [&](const auto& record) {
    return record.thread == evidence.compute_records.front().thread;
  }));
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kLaneDispatches), 0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kLaneComputeDispatches), 0U);
  EXPECT_GE(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionRunnerSweeps), 1U);
  EXPECT_GE(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionRunnerWakeSignals), 1U);
  const std::uint64_t tail_attempts =
      phase_count(evidence, detail::RuntimePhaseCounter::kPartitionTailDispatchAttempts);
  EXPECT_GT(tail_attempts, 0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionTailDispatchAccepted), 0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionTailDispatchFallbacks),
            tail_attempts);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierAttempts),
            0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierAccepted),
            0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierFallbacks),
            0U);
  EXPECT_GE(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionControlBarriers), 1U);
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
  ASSERT_EQ(probe.callback_records.size(), 2U);
  std::array<std::uint32_t, 2U> operators{
      probe.callback_records[0].operator_id,
      probe.callback_records[1].operator_id,
  };
  std::ranges::sort(operators);
  EXPECT_EQ(operators, (std::array<std::uint32_t, 2U>{1U, 2U}));
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

TEST(RuntimeInlinePartitionExecution,
     FrozenTypedParamsRemainCoherentAcrossRepeatedInlineRunsWithoutHandoffs) {
  // Key setup: reuse immutable typed-parameter source and relay chains across two inline runs.
  for (std::size_t iteration = 0U; iteration < 2U; ++iteration) {
    SCOPED_TRACE(iteration);
    InlinePartitionProbeState::reset();
    const ExecutionPlan compiled = make_param_chain_plan();
    ASSERT_TRUE(compiled.ok()) << compiled.json();
    ASSERT_TRUE(compiled.parameter_runtime_capacity().has_value());
    // The fatal assertion above proves the optional is engaged.
    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    EXPECT_FALSE(compiled.parameter_runtime_capacity()->runtime_updates_enabled);
    EXPECT_FALSE(compiled.parameter_runtime_capacity()->lifecycle_updates_enabled);
    // NOLINTEND(bugprone-unchecked-optional-access)
    EXPECT_EQ(compiled.parameter_fields().size(), 2U);
    EXPECT_EQ(std::ranges::count(
                  compiled.operator_parameters(), false, &OperatorParameterPlanRecord::no_params),
              2);
    const auto state = detail::ExecutionPlanAccess::retain(compiled);
    ASSERT_TRUE(state);
    ASSERT_TRUE(state->inline_partition_execution.has_value()) << compiled.json();

    const InlinePartitionRunEvidence evidence = run_and_observe(compiled, 1U);
    // Key assertion: callback and lifecycle views share frozen values on the inline owner.
    ASSERT_TRUE(evidence.observed);
    EXPECT_EQ(evidence.termination, RunTermination::kStopped);
    EXPECT_EQ(evidence.diagnostic_count, 0U);
    EXPECT_EQ(evidence.sink_values, (std::vector<std::uint32_t>{44U}));
    ASSERT_EQ(evidence.param_compute_records.size(), 2U);
    ASSERT_EQ(evidence.param_lifecycle_records.size(), 4U);
    expect_param_coherence(evidence, 1U, 41U);
    expect_param_coherence(evidence, 2U, 3U);
    expect_inline_thread_evidence(evidence, 3U, 3U);
  }
}

TEST(RuntimeInlinePartitionExecution,
     ExplicitSinglePartitionKeepsFrozenParamsCoherentOnItsInlineOwner) {
  // Key setup: explicitly place the immutable parameter chain in one authored partition.
  InlinePartitionProbeState::reset();
  const ExecutionPlan compiled = make_param_chain_plan(true);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_FALSE(state->inline_partition_execution.has_value());
  ASSERT_EQ(state->partition_runners.size(), 1U);

  const InlinePartitionRunEvidence evidence = run_and_observe(compiled, 1U);
  // Key assertion: frozen parameter views remain coherent on the authored partition owner.
  ASSERT_TRUE(evidence.observed);
  EXPECT_EQ(evidence.termination, RunTermination::kStopped);
  EXPECT_EQ(evidence.diagnostic_count, 0U);
  EXPECT_EQ(evidence.sink_values, (std::vector<std::uint32_t>{44U}));
  ASSERT_TRUE(evidence.controller_thread.has_value());
  ASSERT_EQ(evidence.param_compute_records.size(), 2U);
  ASSERT_EQ(evidence.param_lifecycle_records.size(), 4U);
  expect_param_coherence(evidence, 1U, 41U);
  expect_param_coherence(evidence, 2U, 3U);
  expect_inline_thread_evidence(evidence, 3U, 3U);
}

TEST(RuntimeInlinePartitionExecution,
     RuntimeMutableNextComputePublishesPinsAndObservesOnTheController) {
  // Key setup: submit a NextCompute parameter update while the implicit inline source is blocked.
  // Key assertion: the shared route helper verifies pin publication and observation on controller.
  expect_runtime_mutable_update_route(RuntimeMutablePlanRoute::kInline, true);
}

TEST(RuntimeInlinePartitionExecution,
     RuntimeMutableNextComputeMatchesExplicitPartitionAndHoloMQControls) {
  // Key setup: submit equivalent NextCompute updates through explicit-partition and HoloMQ routes.
  // Key assertion: both control routes publish and observe the same pinned parameter generation.
  expect_runtime_mutable_update_route(RuntimeMutablePlanRoute::kExplicitPlacement, true);
  expect_runtime_mutable_update_route(RuntimeMutablePlanRoute::kHoloMQ, false);
}

TEST(RuntimeInlinePartitionExecution, UpdatedSelectorDeadlineAndCallbackShareOnePinnedGeneration) {
  // Key setup: update value, LatestWithin age, and deadline while temporal source compute is held.
  InlinePartitionProbeState::reset();
  ExecutionPlan compiled = make_temporal_param_chain_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_EQ(state->partition_runners.size(), 1U) << compiled.json();
  ASSERT_EQ(compiled.parameter_fields().size(), 3U);
  const auto value_field = std::ranges::find(
      compiled.parameter_fields(), "value", &ParameterFieldPlanRecord::canonical_name);
  const auto max_age_field = std::ranges::find(
      compiled.parameter_fields(), "max_age", &ParameterFieldPlanRecord::canonical_name);
  const auto deadline_field = std::ranges::find(
      compiled.parameter_fields(), "deadline", &ParameterFieldPlanRecord::canonical_name);
  ASSERT_NE(value_field, compiled.parameter_fields().end());
  ASSERT_NE(max_age_field, compiled.parameter_fields().end());
  ASSERT_NE(deadline_field, compiled.parameter_fields().end());
  ASSERT_EQ(compiled.parameter_derived_terms().size(), 2U);
  EXPECT_TRUE(std::ranges::any_of(compiled.parameter_derived_terms(), [](const auto& term) {
    return term.kind == ParameterDerivedTermKind::kLatestWithinMaxAge;
  }));
  EXPECT_TRUE(std::ranges::any_of(compiled.parameter_derived_terms(), [](const auto& term) {
    return term.kind == ParameterDerivedTermKind::kContractDeadline;
  }));

  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const auto control = std::make_shared<InlinePartitionControllerProbe>();
  compiled = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  compiled = detail::RuntimeTestPlanAccess::with_support_thread_control(compiled, control);
  RunSession session = run_async(compiled);
  RuntimeMutableComputeRelease release;
  ASSERT_TRUE(InlinePartitionProbeState::wait_for_first_runtime_compute());
  const auto run_owner = session.operator_ref(value_field->owner);
  ASSERT_TRUE(run_owner);
  const ParamAccessPolicy policy{
      .origin = ParamControlOrigin::kLocalApplication,
      .permissions = ParamPermission::kRead | ParamPermission::kUpdate | ParamPermission::kAwait,
  };
  auto handle_result = session.parameter_control(policy);
  ASSERT_TRUE(handle_result);
  ParamControlHandle handle = std::move(*handle_result);
  const ParamAddress value_address{
      .operator_instance = *run_owner,
      .field = value_field->field,
  };
  const std::array patch{
      ParamPatch{
          .target = value_address,
          .value = ParamValue::from(std::uint32_t{100U}),
      },
      ParamPatch{
          .target = ParamAddress{.operator_instance = *run_owner, .field = max_age_field->field},
          .value = ParamValue::from(3s),
      },
      ParamPatch{
          .target = ParamAddress{.operator_instance = *run_owner, .field = deadline_field->field},
          .value = ParamValue::from(2s),
      },
  };
  const ParamBatchUpdateResult accepted =
      handle.submit(patch, ParamUpdateOptions{.effective = NextCompute{}});
  ASSERT_TRUE(accepted.committed);
  const auto pending = handle.read(value_address, ParamReadView::kLatestAccepted);
  ASSERT_TRUE(pending);
  EXPECT_EQ(pending->generation, accepted.target_generation);
  EXPECT_EQ(pending->update_sequence, accepted.update_sequence);
  const auto pending_value = pending->value.as<std::uint32_t>();
  ASSERT_TRUE(pending_value);
  EXPECT_EQ(*pending_value, 100U);
  release.release();
  const bool observed = InlinePartitionProbeState::wait_for_sinks(1U);
  session.request_stop();
  session.wait();

  const InlinePartitionRunEvidence evidence{
      .observed = observed,
      .termination = session.termination(),
      .diagnostic_count = session.diagnostics().size(),
      .controller_thread = control->controller_thread(),
      .compute_records = InlinePartitionProbeState::compute_records(),
      .lifecycle_records = InlinePartitionProbeState::lifecycle_records(),
      .param_compute_records = InlinePartitionProbeState::param_compute_records(),
      .param_lifecycle_records = InlinePartitionProbeState::param_lifecycle_records(),
      .temporal_predicate_records = InlinePartitionProbeState::temporal_predicate_records(),
      .sink_values = InlinePartitionProbeState::sink_values(),
      .schema_value = InlinePartitionProbeState::schema_value(),
      .counters = counters->snapshot(),
  };
  // Key assertion: predicate, deadline, and callback observe one newly pinned generation.
  ASSERT_TRUE(evidence.observed);
  EXPECT_EQ(evidence.termination, RunTermination::kStopped);
  EXPECT_EQ(evidence.diagnostic_count, 0U);
  EXPECT_EQ(evidence.sink_values, (std::vector<std::uint32_t>{203U}));
  ASSERT_EQ(evidence.temporal_predicate_records.size(), 1U);
  EXPECT_EQ(evidence.temporal_predicate_records[0].threshold, 200U);
  EXPECT_EQ(evidence.temporal_predicate_records[0].driver, 101U);
  EXPECT_EQ(evidence.temporal_predicate_records[0].side, 102U);
  ASSERT_EQ(evidence.param_compute_records.size(), 1U);
  EXPECT_EQ(evidence.param_compute_records[0].value, 100U);
  EXPECT_EQ(evidence.param_compute_records[0].generation, accepted.target_generation);
  EXPECT_EQ(evidence.param_compute_records[0].activation_reason, ActivationReason::kNominal);
  EXPECT_EQ(evidence.param_compute_records[0].activation_policy, OnViolation::kEmitWitness);
  ASSERT_TRUE(evidence.controller_thread.has_value());
  // The fatal assertion above proves the optional is engaged.
  // NOLINTBEGIN(bugprone-unchecked-optional-access)
  EXPECT_EQ(evidence.temporal_predicate_records[0].thread, *evidence.controller_thread);
  EXPECT_EQ(evidence.param_compute_records[0].thread, *evidence.controller_thread);
  // NOLINTEND(bugprone-unchecked-optional-access)
  expect_inline_thread_evidence(evidence, 3U, 3U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierAttempts),
            0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierAccepted),
            0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierFallbacks),
            0U);
  const auto observed_update = handle.await(accepted.update_sequence, 0ns);
  ASSERT_TRUE(observed_update);
  EXPECT_EQ(observed_update->state, ParamUpdateState::kObserved);
}

TEST(RuntimeInlinePartitionExecution, SelfRestartRunsAtTheCallbackBoundaryWithoutWorkerHandoffs) {
  // Key setup: request a source self-restart before its first POD emission in an inline chain.
  InlinePartitionProbeState::reset();
  const ExecutionPlan compiled = make_restart_chain_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_TRUE(state->inline_partition_execution.has_value()) << compiled.json();
  ASSERT_FALSE(compiled.lifecycle_restart_targets().empty());

  const InlinePartitionRunEvidence evidence = run_and_observe(compiled, 1U);
  // Key assertion: restart lifecycle stages complete at the boundary before the chain drains.
  ASSERT_TRUE(evidence.observed);
  EXPECT_EQ(evidence.termination, RunTermination::kStopped);
  EXPECT_EQ(evidence.diagnostic_count, 0U);
  EXPECT_EQ(evidence.sink_values, (std::vector<std::uint32_t>{42U}));
  ASSERT_EQ(evidence.compute_records.size(), 4U);
  EXPECT_EQ(evidence.compute_records[0].operator_id, 1U);
  EXPECT_EQ(evidence.compute_records[1].operator_id, 1U);
  EXPECT_EQ(evidence.compute_records[2].operator_id, 2U);
  EXPECT_EQ(evidence.compute_records[3].operator_id, 3U);

  std::vector<LifecycleStage> source_stages;
  for (const auto& record : evidence.lifecycle_records) {
    if (record.operator_id == 1U) {
      source_stages.push_back(record.stage);
    }
  }
  EXPECT_EQ(source_stages,
            (std::vector{LifecycleStage::kStart,
                         LifecycleStage::kStop,
                         LifecycleStage::kStart,
                         LifecycleStage::kStop}));
  expect_inline_thread_evidence(evidence, 4U, 3U, 8U);
}

TEST(RuntimeInlinePartitionExecution,
     StopDuringCallbackCancelsPendingGenerationAndFencesSuccessors) {
  // Key setup: queue a NextCompute parameter generation while the current source callback is held.
  InlinePartitionProbeState::reset();
  ExecutionPlan compiled = make_runtime_mutable_param_chain_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_TRUE(state->inline_partition_execution.has_value()) << compiled.json();
  ASSERT_EQ(compiled.parameter_fields().size(), 1U);
  const ParameterFieldPlanRecord field = compiled.parameter_fields().front();

  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const auto control = std::make_shared<InlinePartitionControllerProbe>();
  compiled = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  compiled = detail::RuntimeTestPlanAccess::with_support_thread_control(compiled, control);

  RunSession session = run_async(compiled);
  RuntimeMutableComputeRelease release;
  ASSERT_TRUE(InlinePartitionProbeState::wait_for_first_runtime_compute());
  const auto run_owner = session.operator_ref(field.owner);
  ASSERT_TRUE(run_owner);
  const ParamAccessPolicy policy{
      .origin = ParamControlOrigin::kLocalApplication,
      .permissions = ParamPermission::kRead | ParamPermission::kUpdate | ParamPermission::kAwait,
  };
  auto handle_result = session.parameter_control(policy);
  ASSERT_TRUE(handle_result);
  ParamControlHandle handle = std::move(*handle_result);
  const std::array patch{
      ParamPatch{
          .target = ParamAddress{.operator_instance = *run_owner, .field = field.field},
          .value = ParamValue::from(std::uint32_t{43U}),
      },
  };
  const ParamBatchUpdateResult accepted =
      handle.submit(patch, ParamUpdateOptions{.effective = NextCompute{}});
  ASSERT_TRUE(accepted.committed);

  session.request_stop();
  release.release();
  session.wait();
  const InlinePartitionRunEvidence evidence =
      capture_run_evidence(session, true, counters, control);

  // Key assertion: stop cancels the pending generation and prevents relay and sink activation.
  EXPECT_EQ(evidence.termination, RunTermination::kStopped);
  EXPECT_EQ(evidence.diagnostic_count, 0U);
  EXPECT_TRUE(evidence.sink_values.empty());
  ASSERT_EQ(evidence.compute_records.size(), 1U);
  EXPECT_EQ(evidence.compute_records.front().operator_id, 1U);
  ASSERT_EQ(evidence.param_compute_records.size(), 1U);
  EXPECT_EQ(evidence.param_compute_records.front().value, 41U);
  EXPECT_EQ(evidence.param_compute_records.front().generation, 0U);
  const auto terminal = handle.await(accepted.update_sequence, 0ns);
  ASSERT_TRUE(terminal);
  EXPECT_EQ(terminal->status, ParamUpdateStatus::kClosed);
  EXPECT_EQ(terminal->state, ParamUpdateState::kCanceled);
  EXPECT_EQ(terminal->target_generation, accepted.target_generation);
  expect_inline_thread_evidence(evidence, 1U, 3U);
}

TEST(RuntimeInlinePartitionExecution, CallbackFailureReleasesInlineStateAndReplaysTheTypedFailure) {
  // Key setup: reuse an inline POD plan whose relay callback fails for two consecutive runs.
  const ExecutionPlan compiled = make_failing_chain_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_TRUE(state->inline_partition_execution.has_value()) << compiled.json();

  for (std::size_t iteration = 0U; iteration < 2U; ++iteration) {
    SCOPED_TRACE(iteration);
    InlinePartitionProbeState::reset();
    const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
    const auto control = std::make_shared<InlinePartitionControllerProbe>();
    ExecutionPlan run_plan = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
    run_plan = detail::RuntimeTestPlanAccess::with_support_thread_control(run_plan, control);
    RunSession session = run_async(run_plan);
    session.wait();
    const InlinePartitionRunEvidence evidence =
        capture_run_evidence(session, true, counters, control);

    // Key assertion: each run reports the same typed failure and releases inline callback state.
    EXPECT_EQ(evidence.termination, RunTermination::kFailed);
    ASSERT_EQ(session.diagnostics().size(), 1U);
    EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_OPERATOR_CALLBACK_FAILED");
    EXPECT_TRUE(evidence.sink_values.empty());
    ASSERT_EQ(evidence.compute_records.size(), 2U);
    EXPECT_EQ(evidence.compute_records[0].operator_id, 1U);
    EXPECT_EQ(evidence.compute_records[1].operator_id, 2U);
    expect_inline_thread_evidence(evidence, 2U, 3U);
  }
}

TEST(RuntimeInlinePartitionExecution, LatestWithinSelectionUsesTheInlinePartitionPlan) {
  // Key setup: compile a two-input POD graph with a LatestWithin side selection.
  const ExecutionPlan compiled = make_latest_within_chain_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  // Key assertion: the selection remains represented within a single partition-runner plan.
  ASSERT_TRUE(std::ranges::any_of(state->input_selections, [](const auto& selection) {
    return selection.selector == InputSelectorKind::kLatestWithin;
  }));
  EXPECT_EQ(state->partition_runners.size(), 1U);
}

TEST(RuntimeInlinePartitionExecution, ExplicitSinglePartitionPlacementUsesTheControllerOwner) {
  // Key setup: explicitly place the four-Operator POD chain into one authored partition.
  InlinePartitionProbeState::reset();
  const ExecutionPlan compiled = make_pod_chain_plan(QueuePolicy::kBlocking, true);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_FALSE(state->inline_partition_execution.has_value());
  ASSERT_EQ(state->partition_runners.size(), 1U);

  const InlinePartitionRunEvidence evidence = run_and_observe(compiled, 1U);
  // Key assertion: all callbacks execute on the controller-owned partition without lane dispatch.
  ASSERT_TRUE(evidence.observed);
  EXPECT_EQ(evidence.termination, RunTermination::kStopped);
  EXPECT_EQ(evidence.diagnostic_count, 0U);
  ASSERT_TRUE(evidence.controller_thread.has_value());
  ASSERT_EQ(evidence.compute_records.size(), 4U);
  EXPECT_TRUE(std::ranges::all_of(evidence.compute_records, [&](const auto& record) {
    return record.thread == *evidence.controller_thread;
  }));
  EXPECT_GT(phase_count(evidence, detail::RuntimePhaseCounter::kLaneInlineOwnerInvocations), 0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kLaneDispatches), 0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kLaneCompletionsObserved), 0U);
}

}  // namespace
}  // namespace holoscan
