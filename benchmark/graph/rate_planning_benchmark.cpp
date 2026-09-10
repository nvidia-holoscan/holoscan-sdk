/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <sys/resource.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/rate_generation.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "compiler/compile_internal.hpp"
#include "partition_planning_allocation_probe.hpp"

namespace holoscan {
namespace {

using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

constexpr std::uint64_t kSmallPayloadBytes = 64U;
constexpr std::uint64_t kLargePayloadBytes = 4U * 1024U * 1024U;
constexpr auto kHeadroomSinkPeriod = std::chrono::nanoseconds{16666667};
constexpr auto kHeadroomSourcePeriod = std::chrono::milliseconds{1};
constexpr std::uint32_t kDefaultTrafficWarmupSamples = 34U;
constexpr std::uint32_t kDefaultTrafficSamples = 64U;
constexpr auto kDefaultTrafficTimeout = std::chrono::seconds{10};
constexpr std::array<std::string_view, 7U> kCompileTopologies{
    "linear_fixed",
    "fanout_fixed",
    "dual_rate_fusion",
    "multi_follows",
    "embedded_follows",
    "lifecycle_bridge",
    "lifecycle_cross_process_rejected",
};
constexpr std::array<std::string_view, 2U> kPayloadProfiles{"small", "large"};

template <std::size_t Size>
[[nodiscard]] constexpr bool contains(const std::array<std::string_view, Size>& values,
                                      std::string_view candidate) noexcept {
  return std::ranges::find(values, candidate) != values.end();
}

[[nodiscard]] std::int64_t now_ns() noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
      .count();
}

struct HeadroomTrafficSnapshot {
  std::uint64_t accepted_publications{};
  std::uint64_t delivered_samples{};
  std::uint64_t backpressured_emit_attempts{};
  std::uint64_t resource_exhausted_source_attempts{};
  std::uint64_t not_ready_source_attempts{};
  std::uint64_t consumer_visible_replacements{};
  std::uint32_t sampled_occupancy_high_water{};
  std::int64_t measurement_duration_ns{};
  std::vector<std::int64_t> latency_ns;
};

class HeadroomTrafficState final {
 public:
  HeadroomTrafficState(std::uint32_t warmup_samples, std::uint32_t measured_samples)
      : warmup_samples_(warmup_samples),
        measured_samples_(measured_samples),
        total_samples_(warmup_samples + measured_samples),
        attempt_timestamps_(std::make_unique<std::atomic<std::int64_t>[]>(
            static_cast<std::size_t>(total_samples_) + 1U)) {
    for (std::uint32_t sequence = 0U; sequence <= total_samples_; ++sequence) {
      attempt_timestamps_[sequence].store(0, std::memory_order_relaxed);
    }
    latency_ns_.reserve(measured_samples_);
  }

  [[nodiscard]] std::uint32_t total_samples() const noexcept { return total_samples_; }

  [[nodiscard]] bool measured(std::uint32_t sequence) const noexcept {
    return sequence > warmup_samples_ && sequence <= total_samples_;
  }

  void begin_attempt(std::uint32_t sequence) noexcept {
    if (measured(sequence)) {
      attempt_timestamps_[sequence].store(now_ns(), std::memory_order_release);
    }
  }

  void record_accepted(std::uint32_t sequence) noexcept {
    if (!measured(sequence)) {
      return;
    }
    {
      const std::lock_guard lock{mutex_};
      ++accepted_publications_;
      if (sequence == warmup_samples_ + 1U) {
        measurement_started_at_ = attempt_timestamps_[sequence].load(std::memory_order_acquire);
      }
    }
    completion_.notify_all();
  }

  void record_retry(ErrorCode code, std::uint32_t sequence) noexcept {
    if (!measured(sequence)) {
      return;
    }
    const std::lock_guard lock{mutex_};
    if (code == ErrorCode::kBackpressured) {
      ++backpressured_emit_attempts_;
    } else if (code == ErrorCode::kResourceExhausted) {
      ++resource_exhausted_source_attempts_;
    } else if (code == ErrorCode::kNotReady) {
      ++not_ready_source_attempts_;
    }
  }

  void record_failure(ErrorCode code) noexcept {
    {
      const std::lock_guard lock{mutex_};
      failure_ = code;
    }
    completion_.notify_all();
  }

  template <typename Batch>
  void record_batch(const Batch& batch) noexcept {
    const std::int64_t received_at = now_ns();
    const std::lock_guard lock{mutex_};
    sampled_occupancy_high_water_ =
        std::max(sampled_occupancy_high_water_, static_cast<std::uint32_t>(batch.size()));
    for (const auto& sample : batch) {
      if (sample.metadata.frame_id == 0U || sample.metadata.frame_id > total_samples_ ||
          sample.metadata.frame_id <= last_delivered_sequence_) {
        failure_ = ErrorCode::kProtocolError;
        continue;
      }
      const auto sequence = static_cast<std::uint32_t>(sample.metadata.frame_id);
      last_delivered_sequence_ = sequence;
      if (!measured(sequence)) {
        continue;
      }
      const std::int64_t started_at = attempt_timestamps_[sequence].load(std::memory_order_acquire);
      if (started_at <= 0 || received_at < started_at) {
        failure_ = ErrorCode::kProtocolError;
        continue;
      }
      latency_ns_.push_back(received_at - started_at);
      ++delivered_samples_;
      final_sample_delivered_ = final_sample_delivered_ || sequence == total_samples_;
      if (sequence == total_samples_) {
        measurement_completed_at_ = received_at;
      }
    }
    completion_.notify_all();
  }

  [[nodiscard]] bool wait_for_completion(std::chrono::milliseconds timeout) noexcept {
    std::unique_lock lock{mutex_};
    return completion_.wait_for(
               lock,
               timeout,
               [&] {
                 return failure_.has_value() ||
                        (final_sample_delivered_ && accepted_publications_ == measured_samples_);
               }) &&
           !failure_.has_value() && final_sample_delivered_ &&
           accepted_publications_ == measured_samples_;
  }

  [[nodiscard]] HeadroomTrafficSnapshot snapshot() const {
    const std::lock_guard lock{mutex_};
    const std::uint64_t replacements = accepted_publications_ >= delivered_samples_
                                           ? accepted_publications_ - delivered_samples_
                                           : 0U;
    return HeadroomTrafficSnapshot{
        .accepted_publications = accepted_publications_,
        .delivered_samples = delivered_samples_,
        .backpressured_emit_attempts = backpressured_emit_attempts_,
        .resource_exhausted_source_attempts = resource_exhausted_source_attempts_,
        .not_ready_source_attempts = not_ready_source_attempts_,
        .consumer_visible_replacements = replacements,
        .sampled_occupancy_high_water = sampled_occupancy_high_water_,
        .measurement_duration_ns =
            measurement_started_at_ > 0 && measurement_completed_at_ >= measurement_started_at_
                ? measurement_completed_at_ - measurement_started_at_
                : 0,
        .latency_ns = latency_ns_,
    };
  }

 private:
  std::uint32_t warmup_samples_{};
  std::uint32_t measured_samples_{};
  std::uint32_t total_samples_{};
  std::unique_ptr<std::atomic<std::int64_t>[]> attempt_timestamps_;
  mutable std::mutex mutex_;
  std::condition_variable completion_;
  std::optional<ErrorCode> failure_;
  std::uint64_t accepted_publications_{};
  std::uint64_t delivered_samples_{};
  std::uint64_t backpressured_emit_attempts_{};
  std::uint64_t resource_exhausted_source_attempts_{};
  std::uint64_t not_ready_source_attempts_{};
  std::uint32_t sampled_occupancy_high_water_{};
  std::int64_t measurement_started_at_{};
  std::int64_t measurement_completed_at_{};
  std::uint32_t last_delivered_sequence_{};
  bool final_sample_delivered_{};
  std::vector<std::int64_t> latency_ns_;
};

HeadroomTrafficState* active_headroom_traffic_state{};

[[nodiscard]] HeadroomTrafficState& headroom_traffic_state() noexcept {
  return *active_headroom_traffic_state;
}

class FixedRateScalarSource final : public Operator<> {
 public:
  explicit FixedRateScalarSource(Rate rate) : rate_(rate) {}

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").nominal_rate(rate_).max_emits_per_compute(1U);
  }
  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint64_t> output;

 private:
  Rate rate_{Rate::hz(60U)};
};

class LifecycleScalarSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .lifecycle_nominal_rate(LifecycleStage::kDiscover)
        .max_nominal_rate(Rate::hz(120U))
        .max_emits_per_compute(1U);
  }
  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint64_t> output;
};

class ScalarRelay final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input");
    spec.output(output, "output").follows(input).max_emits_per_compute(1U);
  }
  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint64_t> input;
  Output<std::uint64_t> output;
};

class ScalarEachSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input"); }
  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint64_t> input;
};

class ScalarFusionSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(camera, "camera");
    spec.input(imu, "imu");
  }
  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{camera}).input(imu, DrainAll{});
    return result;
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint64_t> camera;
  Input<std::uint64_t> imu;
};

class MultiFollowsRelay final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input");
    spec.output(slow, "slow").follows(input, RateRatio{1U, 2U}).max_emits_per_compute(1U);
    spec.output(fast, "fast").follows(input, RateRatio{3U, 2U}).max_emits_per_compute(1U);
  }
  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint64_t> input;
  Output<std::uint64_t> slow;
  Output<std::uint64_t> fast;
};

class EmbeddedFollowsGraph final : public Graph {
 public:
  Input<std::uint64_t> input{"input"};
  Output<std::uint64_t> output{"output"};

 protected:
  void compose() override {
    auto relay = op<ScalarRelay>("relay");
    export_input(input, relay->input);
    export_output(output, relay->output);
  }
};

class CycleFollowsOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").expected_rate(Rate::hz(60U));
    spec.output(output, "output").follows(input, RateRatio{1U, 1U}).max_emits_per_compute(1U);
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint64_t> input;
  Output<std::uint64_t> output;
};

class HeadroomTensorSource final : public Operator<> {
 public:
  HeadroomTensorSource(MemoryKind memory_kind, std::uint64_t byte_span)
      : memory_kind_(memory_kind), byte_span_(byte_span) {}

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .nominal_rate(Rate::hz(1000U))
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = memory_kind_,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 1U,
                               .strides = TensorStridePolicy::kContiguousOnly},
            .bounds = tensor_bounds(byte_span_),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }
  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = kHeadroomSourcePeriod});
    return result;
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto& traffic = headroom_traffic_state();
    if (next_sequence_ > traffic.total_samples()) {
      return {};
    }

    traffic.begin_attempt(next_sequence_);
    const std::array<std::int64_t, 1U> shape{static_cast<std::int64_t>(byte_span_)};
    auto loan = output.allocate_tensor(
        TensorLoanRequest{.shape = shape, .dtype = DLDataType{kDLUInt, 8U, 1U}});
    if (!loan) {
      if (loan.error().code == ErrorCode::kResourceExhausted ||
          loan.error().code == ErrorCode::kNotReady) {
        traffic.record_retry(loan.error().code, next_sequence_);
        return {};
      }
      traffic.record_failure(loan.error().code);
      return make_unexpected(std::move(loan.error()));
    }

    expected<void, Error> committed;
    if (memory_kind_ == MemoryKind::kCudaDevice) {
      committed = loan->write_and_commit(
          context.cuda_stream(),
          [&](const TensorOutputWriteGuard& writer) -> expected<void, Error> {
            if (cudaMemsetAsync(writer.data(),
                                static_cast<int>(next_sequence_ & 0xFFU),
                                writer.byte_size(),
                                context.cuda_stream().get()) != cudaSuccess) {
              return make_unexpected(Error{ErrorCode::kFailure, "cudaMemsetAsync failed"});
            }
            return {};
          });
    } else {
      committed = loan->write_host_and_commit([&](const TensorOutputWriteGuard& writer) {
        std::memset(writer.data(), static_cast<int>(next_sequence_ & 0xFFU), writer.byte_size());
      });
    }
    if (!committed) {
      traffic.record_failure(committed.error().code);
      return make_unexpected(std::move(committed.error()));
    }

    const std::uint32_t published_sequence = next_sequence_;
    auto emitted = output.emit(std::move(*loan),
                               EmitOptions{.frame_id = published_sequence,
                                           .trace = TraceContext::root(published_sequence)});
    if (!emitted) {
      if (emitted.error().code == ErrorCode::kBackpressured ||
          emitted.error().code == ErrorCode::kResourceExhausted ||
          emitted.error().code == ErrorCode::kNotReady) {
        traffic.record_retry(emitted.error().code, published_sequence);
        return {};
      }
      traffic.record_failure(emitted.error().code);
      return make_unexpected(std::move(emitted.error()));
    }
    traffic.record_accepted(published_sequence);
    ++next_sequence_;
    return {};
  }

  Output<Tensor> output;

 private:
  MemoryKind memory_kind_{MemoryKind::kHost};
  std::uint64_t byte_span_{kSmallPayloadBytes};
  std::uint32_t next_sequence_{1U};
};

class HeadroomDrainSink final : public Operator<> {
 public:
  HeadroomDrainSink(MemoryKind memory_kind, std::uint64_t byte_span)
      : memory_kind_(memory_kind), byte_span_(byte_span) {}

  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .expects_tensor(TensorInputSpec{
            .representation = {.memory_kind = memory_kind_,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 1U,
                               .strides = TensorStridePolicy::kContiguousOnly},
            .bounds = tensor_bounds(byte_span_),
        });
  }
  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = kHeadroomSinkPeriod}).input(input, DrainAll{});
    return result;
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto batch = input.selected_batch();
    if (!batch) {
      headroom_traffic_state().record_failure(batch.error().code);
      return make_unexpected(std::move(batch.error()));
    }
    headroom_traffic_state().record_batch(*batch);
    return {};
  }

  Input<Tensor> input;

 private:
  MemoryKind memory_kind_{MemoryKind::kHost};
  std::uint64_t byte_span_{kSmallPayloadBytes};
};

void add_benchmark_clock(Graph& graph) {
  const ClockKey clock = graph.add_clock<ManualClock>("benchmark-clock", std::int64_t{0});
  graph.set_default_clock(clock);
}

[[nodiscard]] std::string name(std::string_view prefix, std::size_t index) {
  return std::string(prefix) + "-" + std::to_string(index);
}

[[nodiscard]] std::string json_escape(std::string_view value) {
  std::ostringstream output;
  for (const char ch : value) {
    switch (ch) {
      case '\\':
        output << "\\\\";
        break;
      case '"':
        output << "\\\"";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20U) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<unsigned>(static_cast<unsigned char>(ch)) << std::dec
                 << std::setfill(' ');
        } else {
          output << ch;
        }
        break;
    }
  }
  return output.str();
}

[[nodiscard]] std::string hex_digest(PlanDigest digest) {
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const std::byte value : digest.value) {
    output << std::setw(2) << std::to_integer<unsigned>(value);
  }
  return output.str();
}

[[nodiscard]] std::string_view memory_kind_name(MemoryKind value) noexcept {
  switch (value) {
    case MemoryKind::kHost:
      return "host";
    case MemoryKind::kPinnedHost:
      return "pinned_host";
    case MemoryKind::kCudaDevice:
      return "cuda_device";
    case MemoryKind::kCudaManaged:
      return "cuda_managed";
    case MemoryKind::kUnknown:
      break;
  }
  return "unknown";
}

[[nodiscard]] std::string_view geometry_mode_name(AllocationGeometryMode value) noexcept {
  switch (value) {
    case AllocationGeometryMode::kCompileConcrete:
      return "compile_concrete";
    case AllocationGeometryMode::kLifecycleCandidate:
      return "lifecycle_candidate";
  }
  return "unknown";
}

[[nodiscard]] std::string_view action_name(AllocationAction value) noexcept {
  switch (value) {
    case AllocationAction::kFresh:
      return "fresh";
    case AllocationAction::kReusedExact:
      return "reused_exact";
    case AllocationAction::kReusedLarger:
      return "reused_larger";
    case AllocationAction::kReplaced:
      return "replaced";
    case AllocationAction::kRejected:
      break;
  }
  return "rejected";
}

[[nodiscard]] std::string_view queue_policy_name(QueuePolicy value) noexcept {
  switch (value) {
    case QueuePolicy::kBlocking:
      return "blocking";
    case QueuePolicy::kDropOldest:
      return "drop_oldest";
  }
  return "unknown";
}

[[nodiscard]] std::int64_t percentile(std::vector<std::int64_t> values, std::uint32_t percent) {
  std::ranges::sort(values);
  const std::size_t rank = (values.size() * static_cast<std::size_t>(percent) + 99U) / 100U;
  return values[std::max<std::size_t>(1U, rank) - 1U];
}

[[nodiscard]] MemoryKind parse_memory_kind(std::string_view value) {
  if (value == "host") {
    return MemoryKind::kHost;
  }
  if (value == "pinned_host") {
    return MemoryKind::kPinnedHost;
  }
  if (value == "cuda_device") {
    return MemoryKind::kCudaDevice;
  }
  return MemoryKind::kUnknown;
}

struct RatioValue {
  std::uint32_t numerator{};
  std::uint32_t denominator{1U};
};

[[nodiscard]] std::optional<RatioValue> parse_ratio(std::string_view value) {
  const auto slash = value.find('/');
  if (slash == std::string_view::npos) {
    return std::nullopt;
  }
  try {
    const auto numerator =
        static_cast<std::uint32_t>(std::stoul(std::string(value.substr(0, slash))));
    const auto denominator =
        static_cast<std::uint32_t>(std::stoul(std::string(value.substr(slash + 1U))));
    if (numerator == 0U || denominator == 0U) {
      return std::nullopt;
    }
    return RatioValue{numerator, denominator};
  } catch (...) {
    return std::nullopt;
  }
}

[[nodiscard]] bool cuda_device_zero_available() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

[[nodiscard]] expected<RateGenerationSnapshot, Error> wait_for_published_rate_generation(
    RunSession& session) {
  const auto deadline = Clock::now() + 3s;
  while (Clock::now() < deadline) {
    auto snapshot = session.capture_rate_generations();
    if (snapshot && snapshot->published() != nullptr) {
      return snapshot;
    }
    if (!snapshot && snapshot.error().code != ErrorCode::kNotReady) {
      return make_unexpected(std::move(snapshot.error()));
    }
    std::this_thread::sleep_for(1ms);
  }
  return make_unexpected(Error{ErrorCode::kNotReady, "RATE_PUBLICATION_TIMEOUT"});
}

[[nodiscard]] detail::ProcessPlacementBinding split_lifecycle_source_sink_placement() {
  constexpr std::array source_selector{
      detail::GraphObjectSelector{.graph_path = "source"},
  };
  constexpr std::array sink_selector{
      detail::GraphObjectSelector{.graph_path = "sink"},
  };
  detail::ProcessPlacementBinding placement;
  placement
      .bind_process_role("source-role",
                         detail::ProcessPlacement{
                             .members = source_selector,
                             .launch_group = "source-process",
                         })
      .bind_process_role("sink-role",
                         detail::ProcessPlacement{
                             .members = sink_selector,
                             .launch_group = "sink-process",
                         });
  return placement;
}

struct CompileProbeStats {
  benchmarking::PlanningAllocationStats compile_allocations;
  benchmarking::PlanningAllocationStats probe_end_allocations;
  bool plan_retained_at_probe_end{true};
  std::int64_t compile_ns{};
  rusage usage{};
};

template <typename BuildFn, typename CompileFn>
[[nodiscard]] expected<std::pair<ExecutionPlan, CompileProbeStats>, std::string> run_compile_probe(
    BuildFn&& build_graph, CompileOptions options, CompileFn&& compile_graph) {
  {
    Graph warmup_graph{"rate-planning-warmup"};
    add_benchmark_clock(warmup_graph);
    const ExecutionPlan warmup_plan = compile(warmup_graph);
    if (!warmup_plan.ok() || !warmup_plan.partitions().empty()) {
      return make_unexpected("rate-planning warm-up compile failed");
    }
  }

  Graph graph{"rate-planning-benchmark"};
  build_graph(graph);
  benchmarking::PlanningAllocationProbe allocations;
  const auto started = Clock::now();
  ExecutionPlan plan = compile_graph(graph, std::move(options));
  const auto compile_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started).count();
  const auto live_allocations = allocations.snapshot();
  const auto probe_end_allocations = allocations.finish();
  rusage usage{};
  if (probe_end_allocations.cross_thread_activity ||
      probe_end_allocations.cross_generation_activity) {
    return make_unexpected("allocation probe observed cross-thread or cross-generation activity");
  }
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    return make_unexpected("getrusage failed");
  }
  return std::pair{std::move(plan),
                   CompileProbeStats{
                       .compile_allocations = live_allocations,
                       .probe_end_allocations = probe_end_allocations,
                       .plan_retained_at_probe_end = true,
                       .compile_ns = compile_ns,
                       .usage = usage,
                   }};
}

template <typename BuildFn>
[[nodiscard]] expected<std::pair<ExecutionPlan, CompileProbeStats>, std::string> run_compile_probe(
    BuildFn&& build_graph, CompileOptions options = {}) {
  return run_compile_probe(std::forward<BuildFn>(build_graph),
                           std::move(options),
                           [](Graph& graph, CompileOptions compile_options) {
                             return compile(graph, std::move(compile_options));
                           });
}

void build_linear_fixed(Graph& graph, std::size_t count) {
  add_benchmark_clock(graph);
  if (count == 0U) {
    return;
  }
  auto source = graph.op<FixedRateScalarSource>("source", Rate::hz(1000U));
  if (count == 1U) {
    return;
  }
  auto* prior = &source->output;
  for (std::size_t index = 0U; index + 2U < count; ++index) {
    auto relay = graph.op<ScalarRelay>(name("relay", index));
    graph.add_flow(*prior, relay->input);
    prior = &relay->output;
  }
  auto sink = graph.op<ScalarEachSink>("sink");
  graph.add_flow(*prior, sink->input);
}

void build_fanout_fixed(Graph& graph, std::size_t count) {
  add_benchmark_clock(graph);
  if (count == 0U) {
    return;
  }
  auto source = graph.op<FixedRateScalarSource>("source", Rate::hz(1000U));
  for (std::size_t index = 1U; index < count; ++index) {
    auto sink = graph.op<ScalarEachSink>(name("sink", index));
    graph.add_flow(source->output, sink->input);
  }
}

void build_dual_rate_fusion(Graph& graph, std::size_t count) {
  add_benchmark_clock(graph);
  std::size_t index{};
  while (index + 3U <= count) {
    auto camera = graph.op<FixedRateScalarSource>(name("camera", index), Rate::hz(60U));
    auto imu = graph.op<FixedRateScalarSource>(name("imu", index), Rate::hz(1000U));
    auto fusion = graph.op<ScalarFusionSink>(name("fusion", index));
    graph.add_flow(camera->output, fusion->camera);
    graph.add_flow(imu->output, fusion->imu);
    index += 3U;
  }
  while (index < count) {
    graph.op<FixedRateScalarSource>(name("remainder", index++), Rate::hz(1000U));
  }
}

void build_lifecycle_bridge(Graph& graph, std::size_t count) {
  add_benchmark_clock(graph);
  std::size_t index{};
  while (index + 2U <= count) {
    auto source = graph.op<LifecycleScalarSource>(name("source", index));
    auto sink = graph.op<ScalarEachSink>(name("sink", index));
    graph.add_flow(source->output, sink->input);
    index += 2U;
  }
  while (index < count) {
    graph.op<FixedRateScalarSource>(name("remainder", index++), Rate::hz(60U));
  }
}

void build_multi_follows(Graph& graph, std::size_t count) {
  add_benchmark_clock(graph);
  if (count == 0U) {
    return;
  }
  std::size_t index{};
  while (index + 4U <= count) {
    auto source = graph.op<FixedRateScalarSource>(name("source", index), Rate::hz(120U));
    auto relay = graph.op<MultiFollowsRelay>(name("relay", index));
    auto slow_sink = graph.op<ScalarEachSink>(name("slow-sink", index));
    auto fast_sink = graph.op<ScalarEachSink>(name("fast-sink", index));
    graph.add_flow(source->output, relay->input);
    graph.add_flow(relay->slow, slow_sink->input);
    graph.add_flow(relay->fast, fast_sink->input);
    index += 4U;
  }
  while (index < count) {
    graph.op<FixedRateScalarSource>(name("remainder", index++), Rate::hz(120U));
  }
}

void build_embedded_follows(Graph& graph, std::size_t count) {
  add_benchmark_clock(graph);
  if (count == 0U) {
    return;
  }
  auto source = graph.op<FixedRateScalarSource>("source", Rate::hz(240U));
  if (count == 1U) {
    return;
  }
  auto* prior = &source->output;
  for (std::size_t index = 0U; index + 2U < count; ++index) {
    auto embedded = graph.embed<EmbeddedFollowsGraph>(name("embedded", index));
    graph.add_flow(*prior, embedded->input);
    prior = &embedded->output;
  }
  auto sink = graph.op<ScalarEachSink>("sink");
  graph.add_flow(*prior, sink->input);
}

void build_lifecycle_cross_process_rejected(Graph& graph, std::size_t count) {
  add_benchmark_clock(graph);
  if (count == 0U) {
    return;
  }
  auto source = graph.op<LifecycleScalarSource>("source");
  if (count == 1U) {
    return;
  }
  auto sink = graph.op<ScalarEachSink>("sink");
  graph.add_flow(source->output, sink->input);
  for (std::size_t index = 2U; index < count; ++index) {
    graph.op<FixedRateScalarSource>(name("remainder", index), Rate::hz(60U));
  }
}

[[nodiscard]] std::string_view compile_disposition_name(const ExecutionPlan& plan) noexcept {
  return plan.ok() ? "accepted" : "rejected";
}

struct RateDependencyCounts {
  std::size_t follows_input_count{};
  std::size_t inverse_rate_horizon_count{};
  [[nodiscard]] std::size_t total() const noexcept {
    return follows_input_count + inverse_rate_horizon_count;
  }
};

[[nodiscard]] RateDependencyCounts rate_dependency_counts(const ExecutionPlan& plan) noexcept {
  return RateDependencyCounts{
      .follows_input_count = static_cast<std::size_t>(std::ranges::count_if(
          plan.port_rates(), [](const auto& record) { return record.follows_input.has_value(); })),
      .inverse_rate_horizon_count = static_cast<std::size_t>(std::ranges::count_if(
          plan.accumulation_horizons(),
          [](const auto& record) { return record.inverse_rate_port.has_value(); })),
  };
}

struct Options {
  std::string mode;
  std::string topology{"linear_fixed"};
  std::size_t operators{16U};
  std::string payload{"small"};
  std::string memory{"host"};
  QueuePolicy queue_policy{QueuePolicy::kBlocking};
  RatioValue headroom{2U, 1U};
  std::uint32_t traffic_warmup_samples{kDefaultTrafficWarmupSamples};
  std::uint32_t traffic_samples{kDefaultTrafficSamples};
  std::chrono::milliseconds traffic_timeout{
      std::chrono::duration_cast<std::chrono::milliseconds>(kDefaultTrafficTimeout)};
  std::string output;
};

[[nodiscard]] std::optional<Options> parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) {
      return std::nullopt;
    }
    const std::string_view argument{argv[index]};
    const std::string value{argv[index + 1]};
    if (argument == "--mode") {
      options.mode = value;
    } else if (argument == "--topology") {
      if (!contains(kCompileTopologies, value)) {
        return std::nullopt;
      }
      options.topology = value;
    } else if (argument == "--operators") {
      try {
        const auto parsed = std::stoull(value);
        if (parsed > std::numeric_limits<std::uint32_t>::max()) {
          return std::nullopt;
        }
        options.operators = static_cast<std::size_t>(parsed);
      } catch (...) {
        return std::nullopt;
      }
    } else if (argument == "--payload") {
      if (!contains(kPayloadProfiles, value)) {
        return std::nullopt;
      }
      options.payload = value;
    } else if (argument == "--memory") {
      options.memory = value;
    } else if (argument == "--queue-policy") {
      if (value == "blocking") {
        options.queue_policy = QueuePolicy::kBlocking;
      } else if (value == "drop_oldest") {
        options.queue_policy = QueuePolicy::kDropOldest;
      } else {
        return std::nullopt;
      }
    } else if (argument == "--headroom") {
      const auto parsed = parse_ratio(value);
      if (!parsed) {
        return std::nullopt;
      }
      options.headroom = *parsed;
    } else if (argument == "--traffic-warmup-samples" || argument == "--traffic-samples" ||
               argument == "--traffic-timeout-ms") {
      try {
        const auto parsed = std::stoull(value);
        if (parsed > std::numeric_limits<std::uint32_t>::max() ||
            (argument != "--traffic-warmup-samples" && parsed == 0U)) {
          return std::nullopt;
        }
        if (argument == "--traffic-warmup-samples") {
          options.traffic_warmup_samples = static_cast<std::uint32_t>(parsed);
        } else if (argument == "--traffic-samples") {
          options.traffic_samples = static_cast<std::uint32_t>(parsed);
        } else {
          options.traffic_timeout = std::chrono::milliseconds{parsed};
        }
      } catch (...) {
        return std::nullopt;
      }
    } else if (argument == "--output") {
      options.output = value;
    } else {
      return std::nullopt;
    }
  }
  if (options.output.empty()) {
    return std::nullopt;
  }
  if (options.traffic_warmup_samples >
      std::numeric_limits<std::uint32_t>::max() - options.traffic_samples) {
    return std::nullopt;
  }
  if (options.mode == "compile_scaling" && options.topology == "lifecycle_cross_process_rejected" &&
      options.operators < 2U) {
    return std::nullopt;
  }
  return options;
}

[[nodiscard]] std::optional<std::uint64_t> payload_bytes(std::string_view value) {
  if (value == "small") {
    return kSmallPayloadBytes;
  }
  if (value == "large") {
    return kLargePayloadBytes;
  }
  return std::nullopt;
}

[[nodiscard]] std::string unavailable_reason_from_session(const RunSession& session,
                                                          const Error& snapshot_error) {
  if (!session.diagnostics().empty()) {
    return session.diagnostics().front().token;
  }
  return snapshot_error.message.empty() ? "RATE_PUBLICATION_UNAVAILABLE" : snapshot_error.message;
}

int write_compile_scaling_result(const ExecutionPlan& plan, const CompileProbeStats& stats,
                                 const Options& options) {
  std::ofstream output(options.output, std::ios::out | std::ios::trunc);
  if (!output) {
    return 2;
  }
  const auto port_rates = plan.port_rates();
  const auto queue_recipes = plan.queue_sizing_recipes();
  const auto providers = plan.provider_allocation_plans();
  const auto edges = plan.payload_edges();
  const auto diagnostics = plan.diagnostics();
  const auto dependencies = rate_dependency_counts(plan);
  const auto automatic_depth_edge_count =
      static_cast<std::size_t>(std::ranges::count_if(queue_recipes, [](const auto& recipe) {
        return recipe.disposition == PlanDisposition::kAccepted &&
               recipe.origin == QueueDepthOrigin::kRateAndContractDerived;
      }));
  const auto lifecycle_candidate_recipe_count =
      static_cast<std::size_t>(std::ranges::count_if(queue_recipes, [](const auto& recipe) {
        return recipe.disposition == PlanDisposition::kAccepted &&
               recipe.geometry_mode == AllocationGeometryMode::kLifecycleCandidate;
      }));
  const auto compile_concrete_provider_count =
      static_cast<std::size_t>(std::ranges::count_if(providers, [](const auto& provider) {
        return provider.disposition == PlanDisposition::kAccepted &&
               provider.geometry_mode == AllocationGeometryMode::kCompileConcrete;
      }));

  output << "{\n"
         << "  \"schema\":\"holoscan.ea2.rate-planning-benchmark/v1\",\n"
         << "  \"mode\":\"compile_scaling\",\n"
         << "  \"topology\":\"" << json_escape(options.topology) << "\",\n"
         << "  \"operator_count\":" << options.operators << ",\n"
         << "  \"warmup_compile\":\"accepted_zero_work\",\n"
         << "  \"compile_ns\":" << stats.compile_ns << ",\n"
         << "  \"allocation_count\":" << stats.compile_allocations.allocation_count << ",\n"
         << "  \"cumulative_requested_bytes\":"
         << stats.compile_allocations.cumulative_requested_bytes << ",\n"
         << "  \"peak_live_requested_bytes\":"
         << stats.compile_allocations.peak_live_requested_bytes << ",\n"
         << "  \"live_requested_bytes_after_compile\":"
         << stats.compile_allocations.live_requested_bytes << ",\n"
         << "  \"live_requested_bytes_at_probe_end\":"
         << stats.probe_end_allocations.live_requested_bytes << ",\n"
         << "  \"plan_retained_at_probe_end\":"
         << (stats.plan_retained_at_probe_end ? "true" : "false") << ",\n"
         << "  \"process_max_rss_kib\":" << stats.usage.ru_maxrss << ",\n"
         << "  \"compile_disposition\":\"" << compile_disposition_name(plan) << "\",\n"
         << "  \"plan_digest\":" << (plan.ok() ? "\"" + hex_digest(plan.digest()) + "\"" : "null")
         << ",\n"
         << "  \"edge_count\":" << edges.size() << ",\n"
         << "  \"rate_port_count\":" << port_rates.size() << ",\n"
         << "  \"rate_dependency_count\":" << dependencies.total() << ",\n"
         << "  \"rate_dependency_relations\":{\"follows_input_count\":"
         << dependencies.follows_input_count
         << ",\"inverse_rate_horizon_count\":" << dependencies.inverse_rate_horizon_count << "},\n"
         << "  \"queue_recipe_count\":" << queue_recipes.size() << ",\n"
         << "  \"automatic_depth_edge_count\":" << automatic_depth_edge_count << ",\n"
         << "  \"lifecycle_candidate_recipe_count\":" << lifecycle_candidate_recipe_count << ",\n"
         << "  \"compile_concrete_provider_count\":" << compile_concrete_provider_count << ",\n"
         << "  \"diagnostic_tokens\":[";
  for (std::size_t index = 0U; index < diagnostics.size(); ++index) {
    output << (index == 0U ? "" : ",") << "\"" << json_escape(diagnostics[index].token) << "\"";
  }
  output << "]\n"
         << "}\n";
  return output ? 0 : 2;
}

void write_traffic_outcomes(std::ostream& output, const Options& options,
                            const std::optional<HeadroomTrafficSnapshot>& traffic,
                            std::string_view unavailable_reason) {
  output << "  \"traffic_outcomes\":{" << "\"availability\":\""
         << (traffic ? "available" : "unavailable") << "\"" << ",\"unavailable_reason\":"
         << (traffic ? "null" : "\"" + json_escape(std::string(unavailable_reason)) + "\"")
         << ",\"observation_scope\":\"finite_realtime_1khz_to_60hz_drain_boundaries\""
         << ",\"latency_scope\":\"last_source_attempt_to_drain_boundary\""
         << ",\"throughput_scope\":\"last_attempt_of_first_measured_sample_to_final_sample_drain\""
         << ",\"warmup_samples\":" << options.traffic_warmup_samples
         << ",\"requested_samples\":" << options.traffic_samples;
  if (!traffic) {
    output << ",\"accepted_publications\":null" << ",\"delivered_samples\":null"
           << ",\"sampled_occupancy_high_water\":null" << ",\"backpressured_emit_attempts\":null"
           << ",\"resource_exhausted_source_attempts\":null"
           << ",\"not_ready_source_attempts\":null" << ",\"consumer_visible_replacements\":null"
           << ",\"measurement_duration_ns\":null"
           << ",\"finite_pipeline_accepted_publications_per_second\":null"
           << ",\"finite_pipeline_deliveries_per_second\":null" << ",\"latency_ns\":null},\n";
    return;
  }

  const double duration_seconds = static_cast<double>(traffic->measurement_duration_ns) / 1.0e9;
  const double publication_rate =
      duration_seconds > 0.0
          ? static_cast<double>(traffic->accepted_publications) / duration_seconds
          : 0.0;
  const double delivery_rate =
      duration_seconds > 0.0 ? static_cast<double>(traffic->delivered_samples) / duration_seconds
                             : 0.0;
  const auto [minimum, maximum] = std::ranges::minmax(traffic->latency_ns);
  output << ",\"accepted_publications\":" << traffic->accepted_publications
         << ",\"delivered_samples\":" << traffic->delivered_samples
         << ",\"sampled_occupancy_high_water\":" << traffic->sampled_occupancy_high_water
         << ",\"backpressured_emit_attempts\":" << traffic->backpressured_emit_attempts
         << ",\"resource_exhausted_source_attempts\":"
         << traffic->resource_exhausted_source_attempts
         << ",\"not_ready_source_attempts\":" << traffic->not_ready_source_attempts
         << ",\"consumer_visible_replacements\":" << traffic->consumer_visible_replacements
         << ",\"measurement_duration_ns\":" << traffic->measurement_duration_ns
         << ",\"finite_pipeline_accepted_publications_per_second\":" << std::setprecision(17)
         << publication_rate << ",\"finite_pipeline_deliveries_per_second\":" << delivery_rate
         << ",\"latency_ns\":{\"samples\":" << traffic->latency_ns.size()
         << ",\"minimum\":" << minimum << ",\"p50\":" << percentile(traffic->latency_ns, 50U)
         << ",\"p95\":" << percentile(traffic->latency_ns, 95U)
         << ",\"p99\":" << percentile(traffic->latency_ns, 99U) << ",\"maximum\":" << maximum
         << "}},\n";
}

int write_headroom_result(
    const ExecutionPlan& plan, const CompileProbeStats& stats, const Options& options,
    std::string_view availability, std::string_view unavailable_reason,
    std::optional<std::int64_t> startup_to_published_generation_ns = std::nullopt,
    std::optional<QueueSizingGenerationRecord> queue_record = std::nullopt,
    std::optional<PoolCapacityGenerationRecord> pool_capacity_record = std::nullopt,
    std::optional<ProviderAllocationGenerationRecord> provider_record = std::nullopt,
    std::optional<std::string> inspection_json = std::nullopt,
    std::optional<HeadroomTrafficSnapshot> traffic = std::nullopt,
    std::string_view traffic_unavailable_reason = {}) {
  const auto memory_kind = parse_memory_kind(options.memory);
  std::ofstream output(options.output, std::ios::out | std::ios::trunc);
  if (!output) {
    return 2;
  }

  std::vector<const TensorAllocationPlanRecord*> unique_allocations;
  for (const auto& allocation : plan.tensor_allocations()) {
    if (allocation.disposition != PlanDisposition::kAccepted) {
      continue;
    }
    const auto duplicate = std::ranges::find_if(unique_allocations, [&](const auto* candidate) {
      return candidate->pool_id == allocation.pool_id;
    });
    if (duplicate == unique_allocations.end()) {
      unique_allocations.push_back(&allocation);
    }
  }
  std::vector<std::string_view> memory_kinds;
  std::vector<std::int32_t> devices;
  std::uint32_t total_slots = 0U;
  std::uint64_t total_bytes = 0U;
  for (const auto* allocation : unique_allocations) {
    total_slots += allocation->slot_count;
    total_bytes += allocation->aggregate_byte_capacity;
    if (std::ranges::find(memory_kinds, memory_kind_name(allocation->memory_kind)) ==
        memory_kinds.end()) {
      memory_kinds.push_back(memory_kind_name(allocation->memory_kind));
    }
    if (allocation->device.valid() &&
        std::ranges::find(devices, allocation->device.value) == devices.end()) {
      devices.push_back(allocation->device.value);
    }
  }
  std::ranges::sort(memory_kinds);
  std::ranges::sort(devices);

  const auto queue_recipes = plan.queue_sizing_recipes();
  const auto queue_recipe = std::ranges::find_if(queue_recipes, [](const auto& recipe) {
    return recipe.disposition == PlanDisposition::kAccepted;
  });
  if (queue_recipe == queue_recipes.end()) {
    return 2;
  }
  const std::string_view effective_traffic_unavailable_reason =
      !traffic_unavailable_reason.empty()
          ? traffic_unavailable_reason
          : (!unavailable_reason.empty() ? unavailable_reason : "traffic_not_completed");

  output
      << "{\n"
      << "  \"schema\":\"holoscan.ea2.rate-planning-benchmark/v1\",\n"
      << "  \"mode\":\"headroom\",\n"
      << "  \"payload_profile\":\"" << json_escape(options.payload) << "\",\n"
      << "  \"memory_kind\":\"" << memory_kind_name(memory_kind) << "\",\n"
      << "  \"queue_policy\":\"" << queue_policy_name(options.queue_policy) << "\",\n"
      << "  \"headroom_ratio\":{\"numerator\":" << options.headroom.numerator
      << ",\"denominator\":" << options.headroom.denominator << "},\n"
      << "  \"availability\":\"" << availability << "\",\n"
      << "  \"unavailable_reason\":"
      << (unavailable_reason.empty() ? "null"
                                     : "\"" + json_escape(std::string(unavailable_reason)) + "\"")
      << ",\n"
      << "  \"source_nominal_hz\":1000,\n"
      << "  \"source_period_ns\":"
      << std::chrono::duration_cast<std::chrono::nanoseconds>(kHeadroomSourcePeriod).count()
      << ",\n"
      << "  \"sink_period_ns\":" << kHeadroomSinkPeriod.count() << ",\n"
      << "  \"warmup_compile\":\"accepted_zero_work\",\n"
      << "  \"compile_ns\":" << stats.compile_ns << ",\n"
      << "  \"allocation_count\":" << stats.compile_allocations.allocation_count << ",\n"
      << "  \"cumulative_requested_bytes\":" << stats.compile_allocations.cumulative_requested_bytes
      << ",\n"
      << "  \"peak_live_requested_bytes\":" << stats.compile_allocations.peak_live_requested_bytes
      << ",\n"
      << "  \"live_requested_bytes_after_compile\":"
      << stats.compile_allocations.live_requested_bytes << ",\n"
      << "  \"live_requested_bytes_at_probe_end\":"
      << stats.probe_end_allocations.live_requested_bytes << ",\n"
      << "  \"plan_retained_at_probe_end\":"
      << (stats.plan_retained_at_probe_end ? "true" : "false") << ",\n"
      << "  \"process_max_rss_kib\":" << stats.usage.ru_maxrss << ",\n"
      << "  \"compile_disposition\":\"accepted\",\n"
      << "  \"plan_digest\":\"" << hex_digest(plan.digest()) << "\",\n"
      << "  \"queue_recipe_count\":" << queue_recipes.size() << ",\n"
      << "  \"queue_recipe_headroom_ratio\":{\"numerator\":"
      << queue_recipe->headroom_ratio.numerator
      << ",\"denominator\":" << queue_recipe->headroom_ratio.denominator << "},\n"
      << "  \"startup_to_published_generation_ns\":"
      << (startup_to_published_generation_ns ? std::to_string(*startup_to_published_generation_ns)
                                             : "null")
      << ",\n"
      << "  \"restart_latency_ns\":null,\n"
      << "  \"restart_latency_unavailable_reason\":\"not_measured_in_task041_headroom_probe\",\n";
  if (queue_record) {
    output << "  \"queue_effective_logical_depth\":" << queue_record->effective_logical_depth
           << ",\n"
           << "  \"queue_rate_recommendation\":"
           << (queue_record->rate_recommendation
                   ? std::to_string(*queue_record->rate_recommendation)
                   : "null")
           << ",\n"
           << "  \"queue_nominal_arrivals\":"
           << (queue_record->nominal_arrivals ? std::to_string(*queue_record->nominal_arrivals)
                                              : "null")
           << ",\n"
           << "  \"pool_capacity_generation_id\":"
           << (pool_capacity_record ? std::to_string(pool_capacity_record->id.value) : "null")
           << ",\n"
           << "  \"pool_required_entries\":"
           << (pool_capacity_record ? std::to_string(pool_capacity_record->required_pool_entries)
                                    : "null")
           << ",\n"
           << "  \"pool_logical_payload_capacity_bytes\":"
           << (pool_capacity_record && pool_capacity_record->logical_payload_capacity_bytes
                   ? std::to_string(*pool_capacity_record->logical_payload_capacity_bytes)
                   : "null")
           << ",\n"
           << "  \"pool_max_payload_byte_span_per_slot\":"
           << (pool_capacity_record && pool_capacity_record->max_payload_byte_span_per_slot
                   ? std::to_string(*pool_capacity_record->max_payload_byte_span_per_slot)
                   : "null")
           << ",\n"
           << "  \"provider_action\":\"" << action_name(provider_record->action) << "\",\n"
           << "  \"provider_decision_id\":" << provider_record->id.value << ",\n"
           << "  \"provider_plan_id\":" << provider_record->recipe.value << ",\n"
           << "  \"allocation_generation_id\":"
           << (provider_record->allocation_generation
                   ? std::to_string(provider_record->allocation_generation->value)
                   : "null")
           << ",\n"
           << "  \"storage_binding_generation_id\":"
           << (provider_record->storage_binding_generation
                   ? std::to_string(provider_record->storage_binding_generation->value)
                   : "null")
           << ",\n"
           << "  \"provider_materialized_entries\":"
           << (provider_record->provider_materialized_entries
                   ? std::to_string(*provider_record->provider_materialized_entries)
                   : "null")
           << ",\n"
           << "  \"provider_stride_bytes_per_entry\":"
           << (provider_record->provider_stride_bytes_per_entry
                   ? std::to_string(*provider_record->provider_stride_bytes_per_entry)
                   : "null")
           << ",\n"
           << "  \"provider_reserved_bytes_total\":"
           << (provider_record->provider_reserved_bytes_total
                   ? std::to_string(*provider_record->provider_reserved_bytes_total)
                   : "null")
           << ",\n"
           << "  \"provider_reserved_bytes_unavailable_reason\":"
           << (provider_record->provider_reserved_bytes_total
                   ? "null"
                   : "\"provider_did_not_publish_reserved_bytes\"")
           << ",\n"
           << "  \"physical_queue_entries\":null,\n"
           << "  \"physical_ring_entries\":null,\n"
           << "  \"physical_queue_geometry_unavailable_reason\":"
              "\"runtime_did_not_publish_physical_queue_geometry\",\n"
           << "  \"unique_pool_summary\":{\"pool_count\":" << unique_allocations.size()
           << ",\"slot_count\":" << total_slots << ",\"aggregate_byte_capacity\":" << total_bytes
           << ",\"memory_kinds\":[";
    for (std::size_t index = 0U; index < memory_kinds.size(); ++index) {
      output << (index == 0U ? "" : ",") << "\"" << memory_kinds[index] << "\"";
    }
    output << "],\"devices\":[";
    for (std::size_t index = 0U; index < devices.size(); ++index) {
      output << (index == 0U ? "" : ",") << devices[index];
    }
    output << "]},\n";
    write_traffic_outcomes(output, options, traffic, effective_traffic_unavailable_reason);
    output << "  \"inspection_json\":\"" << json_escape(*inspection_json) << "\"\n";
  } else {
    output << "  \"queue_effective_logical_depth\":null,\n"
           << "  \"queue_rate_recommendation\":null,\n"
           << "  \"queue_nominal_arrivals\":null,\n"
           << "  \"pool_capacity_generation_id\":null,\n"
           << "  \"pool_required_entries\":null,\n"
           << "  \"pool_logical_payload_capacity_bytes\":null,\n"
           << "  \"pool_max_payload_byte_span_per_slot\":null,\n"
           << "  \"provider_action\":null,\n"
           << "  \"provider_decision_id\":null,\n"
           << "  \"provider_plan_id\":null,\n"
           << "  \"allocation_generation_id\":null,\n"
           << "  \"storage_binding_generation_id\":null,\n"
           << "  \"provider_materialized_entries\":null,\n"
           << "  \"provider_stride_bytes_per_entry\":null,\n"
           << "  \"provider_reserved_bytes_total\":null,\n"
           << "  \"provider_reserved_bytes_unavailable_reason\":"
              "\"runtime_generation_unavailable\",\n"
           << "  \"physical_queue_entries\":null,\n"
           << "  \"physical_ring_entries\":null,\n"
           << "  \"physical_queue_geometry_unavailable_reason\":"
              "\"runtime_generation_unavailable\",\n"
           << "  \"unique_pool_summary\":null,\n";
    write_traffic_outcomes(output, options, std::nullopt, effective_traffic_unavailable_reason);
    output << "  \"inspection_json\":null\n";
  }
  output << "}\n";
  return output ? 0 : 2;
}

int run_compile_scaling(const Options& options) {
  auto compile_result = run_compile_probe(
      [&](Graph& graph) {
        if (options.topology == "linear_fixed") {
          build_linear_fixed(graph, options.operators);
        } else if (options.topology == "fanout_fixed") {
          build_fanout_fixed(graph, options.operators);
        } else if (options.topology == "dual_rate_fusion") {
          build_dual_rate_fusion(graph, options.operators);
        } else if (options.topology == "multi_follows") {
          build_multi_follows(graph, options.operators);
        } else if (options.topology == "embedded_follows") {
          build_embedded_follows(graph, options.operators);
        } else if (options.topology == "lifecycle_cross_process_rejected") {
          build_lifecycle_cross_process_rejected(graph, options.operators);
        } else if (options.topology == "lifecycle_bridge") {
          build_lifecycle_bridge(graph, options.operators);
        }
      },
      CompileOptions{},
      [&](Graph& graph, CompileOptions compile_options) {
        if (options.topology == "lifecycle_cross_process_rejected") {
          return detail::compile_with_process_placement(
              graph, std::move(compile_options), split_lifecycle_source_sink_placement());
        }
        return compile(graph, std::move(compile_options));
      });
  if (!compile_result) {
    std::cerr << compile_result.error() << '\n';
    return 2;
  }
  auto& [plan, stats] = *compile_result;
  if (options.topology == "lifecycle_cross_process_rejected" && plan.ok()) {
    std::cerr << "lifecycle_cross_process_rejected topology compiled successfully\n";
    return 2;
  }
  return write_compile_scaling_result(plan, stats, options);
}

int run_headroom(const Options& options) {
  const auto memory_kind = parse_memory_kind(options.memory);
  if (memory_kind == MemoryKind::kUnknown) {
    std::cerr << "unknown memory kind\n";
    return 2;
  }
  const auto requested_payload_bytes = payload_bytes(options.payload);
  if (!requested_payload_bytes) {
    std::cerr << "unknown payload profile\n";
    return 2;
  }
  CompileOptions compile_options;
  compile_options.automatic_queue_sizing.headroom_ratio =
      RateRatio{options.headroom.numerator, options.headroom.denominator};
  if (memory_kind == MemoryKind::kCudaDevice) {
    compile_options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
        .operator_path = "source", .output_port = "output", .device = DeviceId{0}});
  }
  auto traffic = std::make_unique<HeadroomTrafficState>(options.traffic_warmup_samples,
                                                        options.traffic_samples);
  active_headroom_traffic_state = traffic.get();
  auto compile_result = run_compile_probe(
      [&](Graph& graph) {
        graph.set_default_clock(graph.add_clock<RealtimeClock>("headroom-traffic-clock"));
        auto source =
            graph.op<HeadroomTensorSource>("source", memory_kind, *requested_payload_bytes);
        auto sink = graph.op<HeadroomDrainSink>("sink", memory_kind, *requested_payload_bytes);
        graph.add_flow(
            source->output, sink->input, ConnectionOptions{.queue_policy = options.queue_policy});
      },
      std::move(compile_options));
  if (!compile_result) {
    std::cerr << compile_result.error() << '\n';
    return 2;
  }
  auto& [plan, stats] = *compile_result;
  if (!plan.ok()) {
    std::cerr << "headroom topology compile rejected\n";
    return 2;
  }

  const auto startup_started = Clock::now();
  RunSession session = run_async(plan);
  auto snapshot = wait_for_published_rate_generation(session);
  const auto startup_to_published_generation_ns =
      snapshot ? std::optional<std::int64_t>{std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                 Clock::now() - startup_started)
                                                 .count()}
               : std::nullopt;
  const bool traffic_completed = snapshot && traffic->wait_for_completion(options.traffic_timeout);
  if (!snapshot) {
    session.request_stop();
    session.wait();
    const std::string reason =
        memory_kind == MemoryKind::kCudaDevice && !cuda_device_zero_available()
            ? "CUDA device 0 unavailable"
            : unavailable_reason_from_session(session, snapshot.error());
    return write_headroom_result(plan, stats, options, "unavailable", reason);
  }
  session.request_stop();
  session.wait();
  if (snapshot->published() == nullptr) {
    return write_headroom_result(plan, stats, options, "unavailable", "RATE_PUBLICATION_MISSING");
  }

  const auto queues = snapshot->queue_sizings(*snapshot->published());
  const auto providers = snapshot->provider_allocations(*snapshot->published());
  if (queues.size() != 1U || providers.size() != 1U) {
    return write_headroom_result(
        plan, stats, options, "unavailable", "RATE_PUBLICATION_STRUCTURE_UNEXPECTED");
  }

  const auto pool_capacities = snapshot->pool_capacities(*snapshot->published());
  const auto pool_capacity = std::ranges::find(pool_capacities,
                                               providers.front().pool_capacity_generation,
                                               &PoolCapacityGenerationRecord::id);
  if (pool_capacity == pool_capacities.end()) {
    return write_headroom_result(
        plan, stats, options, "unavailable", "RATE_PUBLICATION_STRUCTURE_UNEXPECTED");
  }

  const auto inspection = render_rate_queue_inspection(plan,
                                                       *snapshot,
                                                       snapshot->published()->rate_generation,
                                                       plan.payload_edges().front().edge,
                                                       RateQueueInspectionFormat::kCanonicalJson);
  if (!inspection) {
    return write_headroom_result(plan, stats, options, "unavailable", inspection.error().message);
  }
  std::optional<HeadroomTrafficSnapshot> traffic_snapshot;
  std::string traffic_unavailable_reason;
  if (traffic_completed) {
    HeadroomTrafficSnapshot candidate = traffic->snapshot();
    const bool accounting_valid =
        candidate.accepted_publications == options.traffic_samples &&
        candidate.delivered_samples + candidate.consumer_visible_replacements ==
            candidate.accepted_publications &&
        candidate.sampled_occupancy_high_water <= queues.front().effective_logical_depth &&
        candidate.measurement_duration_ns > 0 &&
        candidate.latency_ns.size() == candidate.delivered_samples &&
        (options.queue_policy != QueuePolicy::kBlocking ||
         candidate.consumer_visible_replacements == 0U);
    if (accounting_valid) {
      traffic_snapshot = std::move(candidate);
    } else {
      traffic_unavailable_reason = "TRAFFIC_ACCOUNTING_INVALID";
    }
  } else {
    traffic_unavailable_reason = session.diagnostics().empty()
                                     ? "TRAFFIC_COMPLETION_TIMEOUT"
                                     : session.diagnostics().front().token;
  }
  return write_headroom_result(plan,
                               stats,
                               options,
                               "available",
                               {},
                               startup_to_published_generation_ns,
                               queues.front(),
                               *pool_capacity,
                               providers.front(),
                               *inspection,
                               std::move(traffic_snapshot),
                               traffic_unavailable_reason);
}

}  // namespace
}  // namespace holoscan

int main(int argc, char** argv) {
  const auto options = holoscan::parse_options(argc, argv);
  if (!options) {
    std::cerr << "usage: rate_planning_benchmark --mode <compile_scaling|headroom> "
                 "[--topology <linear_fixed|fanout_fixed|dual_rate_fusion|multi_follows|"
                 "embedded_follows|lifecycle_bridge|lifecycle_cross_process_rejected> "
                 "--operators <nonnegative>] [--payload <small|large> --memory "
                 "<host|pinned_host|cuda_device> --headroom <numerator/denominator> "
                 "--queue-policy <blocking|drop_oldest> "
                 "--traffic-warmup-samples <nonnegative> --traffic-samples <positive> "
                 "--traffic-timeout-ms <positive>] "
                 "--output <path>\n";
    return 2;
  }
  if (options->mode == "compile_scaling") {
    return holoscan::run_compile_scaling(*options);
  }
  if (options->mode == "headroom") {
    return holoscan::run_headroom(*options);
  }
  std::cerr << "unknown mode\n";
  return 2;
}
