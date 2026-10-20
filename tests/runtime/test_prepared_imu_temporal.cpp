/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "common/allocation_counter.hpp"
#include "compiler/compile_internal.hpp"

namespace holoscan::test {
namespace {
using namespace std::chrono_literals;

constexpr std::size_t kBatchSamples = 32U;
constexpr std::size_t kWindowSamples = 48U;
constexpr std::size_t kWindowCount = 4U;

schema::ImuT temporal_imu(std::uint64_t sequence) {
  schema::ImuT value;
  value.header = std::make_shared<schema::HeaderT>();
  value.header->frame_id.assign(128U, 'i');
  value.header->device_sequence = sequence;
  value.orientation = schema::Quaternionf{0.F, 0.F, 0.F, 1.F};
  value.angular_velocity = schema::Vector3f{1.F, 2.F, 3.F};
  value.linear_acceleration = schema::Vector3f{4.F, 5.F, static_cast<float>(sequence)};
  value.flags = schema::ImuFlags_ORIENTATION_VALID;
  return value;
}

bool valid_temporal_imu(const Sample<schema::ImuT>& sample) {
  const auto sequence = sample.metadata.frame_id;
  return sequence != 0U && sample.data.header && sample.data.header->device_sequence == sequence &&
         sample.data.header->frame_id.size() == 128U &&
         sample.data.linear_acceleration.z() == static_cast<float>(sequence) &&
         sample.metadata.capture_timestamp_ns != 0 && sample.metadata.source_clock_id.valid();
}

class TemporalImuSource final : public Operator<> {
 public:
  TemporalImuSource(std::size_t limit, std::chrono::nanoseconds period)
      : limit_(limit), period_(period) {
    for (std::size_t index = 0; index < values_.size(); ++index)
      values_[index] = temporal_imu(index + 1U);
  }
  void setup(OperatorSpec& spec) override { spec.output(output).max_emits_per_compute(1U); }
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = period_});
    return result;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    if (sent_ >= limit_)
      return {};
    const auto sequence = sent_ + 1U;
    auto emitted = output.emit_ref(
        values_[sent_], {.capture_time = context.activation_time(), .frame_id = sequence});
    if (!emitted && emitted.error().code == ErrorCode::kBackpressured)
      return {};
    if (emitted)
      ++sent_;
    return emitted;
  }
  Output<schema::ImuT> output{"output"};

 private:
  std::size_t limit_;
  std::chrono::nanoseconds period_;
  std::array<schema::ImuT, kWindowSamples> values_;
  std::size_t sent_{};
};

struct TemporalProbe {
  std::mutex mutex;
  std::condition_variable ready;
  std::atomic<std::size_t> samples{};
  std::atomic<std::size_t> activations{};
  std::atomic<std::size_t> allocations{};
  std::atomic<bool> valid{true};

  bool wait_for(std::size_t expected_samples, std::size_t expected_activations) {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 15s, [&] {
      return samples.load(std::memory_order_acquire) >= expected_samples &&
             activations.load(std::memory_order_acquire) >= expected_activations;
    });
  }
  void notify() { ready.notify_all(); }
};

class DrainAllImuSink final : public Operator<> {
 public:
  explicit DrainAllImuSink(std::shared_ptr<TemporalProbe> probe) : probe_(std::move(probe)) {}
  void setup(OperatorSpec& spec) override { spec.input(input).queue_depth(16U); }
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 5ms}).input(input, DrainAll{});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override {
    const ScopedAllocationCounter counter;
    HOLOSCAN_TRY(auto batch, input.selected_batch());
    HOLOSCAN_TRY(auto repeated, input.selected_batch());
    if (batch.size() != repeated.size())
      probe_->valid.store(false);
    for (std::size_t index = 0; index < batch.size(); ++index) {
      const auto& sample = batch[index];
      if (&sample != &repeated[index] || !valid_temporal_imu(sample) ||
          sample.metadata.frame_id <= last_sequence_)
        probe_->valid.store(false);
      last_sequence_ = sample.metadata.frame_id;
      probe_->samples.fetch_add(1U, std::memory_order_release);
    }
    probe_->allocations.fetch_add(counter.count(), std::memory_order_relaxed);
    probe_->activations.fetch_add(1U, std::memory_order_release);
    probe_->notify();
    return {};
  }
  Input<schema::ImuT> input{"input"};

 private:
  std::shared_ptr<TemporalProbe> probe_;
  std::uint64_t last_sequence_{};
};

class WindowAnchorSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output).max_emits_per_compute(1U); }
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 40ms});
    return result;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    // The first accepted anchor primes the bracketed selector. Emit one extra anchor so the
    // qualification observes kWindowCount closed windows after that deterministic priming step.
    if (ticks_++ < 1U || sent_ >= kWindowCount + 1U)
      return {};
    const auto value = sent_ + 1U;
    auto emitted = output.emit(Sample<std::uint64_t>{
        value,
        SampleMetadata{.capture_timestamp_ns = context.activation_time().timestamp_ns,
                       .source_clock_id = context.activation_time().clock_id,
                       .frame_id = value}});
    if (!emitted && emitted.error().code == ErrorCode::kBackpressured)
      return {};
    if (emitted)
      ++sent_;
    return emitted;
  }
  Output<std::uint64_t> output{"output"};

 private:
  std::size_t ticks_{};
  std::uint64_t sent_{};
};

class WindowImuSink final : public Operator<> {
 public:
  explicit WindowImuSink(std::shared_ptr<TemporalProbe> probe) : probe_(std::move(probe)) {}
  void setup(OperatorSpec& spec) override {
    spec.input(anchor).queue_depth(4U);
    spec.input(imu).queue_depth(32U);
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnWindow{.anchor = anchor, .max_wait = 20ms, .max_pending_windows = 2U})
        .input(anchor, Exact{})
        .input(imu, BracketedWindow{.lookbehind = 1U, .lookahead = 1U, .max_samples = 8U})
        .on_missing(imu, Skip{});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override {
    HOLOSCAN_TRY(auto anchor_sample, anchor.receive());
    const ScopedAllocationCounter counter;
    HOLOSCAN_TRY(auto batch, imu.selected_batch());
    HOLOSCAN_TRY(auto repeated, imu.selected_batch());
    if (batch.empty() || batch.size() != repeated.size() || anchor_sample.data <= last_anchor_)
      probe_->valid.store(false);
    std::int64_t last_capture{};
    for (std::size_t index = 0; index < batch.size(); ++index) {
      const auto& sample = batch[index];
      if (&sample != &repeated[index] || !valid_temporal_imu(sample) ||
          (last_capture != 0 && sample.metadata.capture_timestamp_ns < last_capture))
        probe_->valid.store(false);
      last_capture = sample.metadata.capture_timestamp_ns;
      probe_->samples.fetch_add(1U, std::memory_order_release);
    }
    last_anchor_ = anchor_sample.data;
    probe_->allocations.fetch_add(counter.count(), std::memory_order_relaxed);
    probe_->activations.fetch_add(1U, std::memory_order_release);
    probe_->notify();
    return {};
  }
  Input<std::uint64_t> anchor{"anchor"};
  Input<schema::ImuT> imu{"imu"};

 private:
  std::shared_ptr<TemporalProbe> probe_;
  std::uint64_t last_anchor_{};
};

class PreparedImuTemporal : public ::testing::TestWithParam<bool> {};

// REQ: SCHEMA-009
TEST_P(PreparedImuTemporal, DrainAllReadsPreparedBatchesWithoutNativeCopies) {
  auto probe = std::make_shared<TemporalProbe>();
  Graph graph{"prepared-imu-drain-all"};
  auto source = graph.op<TemporalImuSource>("source", kBatchSamples, 1ms);
  auto sink = graph.op<DrainAllImuSink>("sink", probe);
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 16U});
  graph.partition("source").add(source);
  graph.partition("sink").add(sink);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  const auto plan =
      GetParam() ? detail::compile_with_process_local_holomq(graph, {}) : compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto session = run_async(plan);
  const bool complete = probe->wait_for(kBatchSamples, 1U);
  session.request_stop();
  session.wait();
  ASSERT_TRUE(complete) << (session.diagnostics().empty() ? "timeout"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(probe->valid.load());
  EXPECT_EQ(probe->samples.load(), kBatchSamples);
  EXPECT_EQ(probe->allocations.load(), 0U);
}

// REQ: SCHEMA-009
TEST_P(PreparedImuTemporal, BracketedWindowsReadPreparedOverlappingSamplesWithoutCopies) {
  auto probe = std::make_shared<TemporalProbe>();
  Graph graph{"prepared-imu-window"};
  auto imu = graph.op<TemporalImuSource>("imu-source", kWindowSamples, 10ms);
  auto anchor = graph.op<WindowAnchorSource>("anchor-source");
  auto sink = graph.op<WindowImuSink>("sink", probe);
  graph.add_flow(imu->output, sink->imu, ConnectionOptions{.queue_depth = 32U});
  graph.add_flow(anchor->output, sink->anchor, ConnectionOptions{.queue_depth = 4U});
  graph.partition("imu-source").add(imu);
  graph.partition("anchor-source").add(anchor);
  graph.partition("sink").add(sink);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  const auto plan =
      GetParam() ? detail::compile_with_process_local_holomq(graph, {}) : compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto session = run_async(plan);
  const bool complete = probe->wait_for(1U, kWindowCount);
  session.request_stop();
  session.wait();
  ASSERT_TRUE(complete) << (session.diagnostics().empty() ? "timeout"
                                                          : session.diagnostics().front().token)
                        << " activations=" << probe->activations.load()
                        << " samples=" << probe->samples.load();
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(probe->valid.load());
  EXPECT_EQ(probe->activations.load(), kWindowCount);
  EXPECT_GT(probe->samples.load(), kWindowCount);
  EXPECT_EQ(probe->allocations.load(), 0U);
}

INSTANTIATE_TEST_SUITE_P(PreparedPaths, PreparedImuTemporal, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "HoloMQ" : "Direct";
                         });

struct RestartProbe {
  std::mutex mutex;
  std::condition_variable ready;
  std::atomic<unsigned> starts{};
  std::atomic<unsigned> computes{};
  std::array<std::size_t, 2> start_allocations{};
  std::array<std::size_t, 2> callback_allocations{};
  std::array<const void*, 2> sample_addresses{};
  std::array<const void*, 2> input_header_addresses{};
  std::array<const void*, 2> scratch_header_addresses{};
  std::atomic<bool> valid{true};

  bool wait() {
    std::unique_lock lock(mutex);
    return ready.wait_for(
        lock, 15s, [&] { return computes.load(std::memory_order_acquire) == 2U; });
  }
};

class RestartImuSource final : public Operator<> {
 public:
  explicit RestartImuSource(std::shared_ptr<RestartProbe> probe) : probe_(std::move(probe)) {
    values_[0] = temporal_imu(1U);
    values_[1] = temporal_imu(2U);
  }
  void setup(OperatorSpec& spec) override { spec.output(output).max_emits_per_compute(1U); }
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 100us});
    return result;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    if (sent_ >= values_.size() || probe_->starts.load(std::memory_order_acquire) <= sent_)
      return {};
    const auto sequence = sent_ + 1U;
    auto emitted = output.emit_ref(
        values_[sent_], {.capture_time = context.activation_time(), .frame_id = sequence});
    if (!emitted && emitted.error().code == ErrorCode::kBackpressured)
      return {};
    if (emitted)
      ++sent_;
    return emitted;
  }
  Output<schema::ImuT> output{"output"};

 private:
  std::shared_ptr<RestartProbe> probe_;
  std::array<schema::ImuT, 2> values_;
  std::size_t sent_{};
};

class RestartImuTransform final : public Operator<> {
 public:
  explicit RestartImuTransform(std::shared_ptr<RestartProbe> probe) : probe_(std::move(probe)) {}
  void setup(OperatorSpec& spec) override { spec.input(input).queue_depth(1U); }
  void start() override {
    const ScopedAllocationCounter counter;
    if (!scratch_) {
      auto prepared = schema::SchemaScratch<schema::ImuT>::create();
      if (!prepared)
        throw RuntimeError(prepared.error());
      scratch_.emplace(std::move(*prepared));
    }
    const auto index = probe_->starts.fetch_add(1U, std::memory_order_release);
    if (index < probe_->start_allocations.size())
      probe_->start_allocations[index] = counter.count();
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    const auto index = probe_->computes.load(std::memory_order_relaxed);
    if (index >= 2U)
      return make_unexpected(Error{ErrorCode::kFailure});
    const ScopedAllocationCounter counter;
    HOLOSCAN_TRY(auto view, input.selected_sample());
    if (!valid_temporal_imu(view.sample()))
      probe_->valid.store(false);
    probe_->sample_addresses[index] = &view.data();
    probe_->input_header_addresses[index] = view.data().header.get();
    HOLOSCAN_TRY(scratch_->copy_from(view.data()));
    probe_->scratch_header_addresses[index] = scratch_->value().header.get();
    scratch_->value().angular_velocity = schema::Vector3f{7.F, 8.F, 9.F};
    HOLOSCAN_TRY(scratch_->reset());
    if (index == 0U) {
      HOLOSCAN_TRY(auto receipt, context.request_restart({.from_stage = LifecycleStage::kStart}));
      static_cast<void>(receipt);
    }
    probe_->callback_allocations[index] = counter.count();
    probe_->computes.fetch_add(1U, std::memory_order_release);
    probe_->ready.notify_all();
    return {};
  }
  Input<schema::ImuT> input{"input"};

 private:
  std::shared_ptr<RestartProbe> probe_;
  std::optional<schema::SchemaScratch<schema::ImuT>> scratch_;
};

class PreparedImuRestart : public ::testing::TestWithParam<bool> {};

// REQ: SCHEMA-009
TEST_P(PreparedImuRestart, InPlaceLifecycleRestartReusesPreparedSlotAndOperatorScratch) {
  auto probe = std::make_shared<RestartProbe>();
  Graph graph{"prepared-imu-restart"};
  auto source = graph.op<RestartImuSource>("source", probe);
  auto transform = graph.op<RestartImuTransform>("transform", probe);
  graph.add_flow(source->output, transform->input, ConnectionOptions{.queue_depth = 1U});
  graph.partition("source").add(source);
  graph.partition("transform").add(transform);
  graph.lifecycle().allow_self_restart(transform);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  CompileOptions options;
  options.lifecycle_transactions = LifecycleTransactionOptions{.max_pending_requests = 4U,
                                                               .max_pending_transactions = 4U,
                                                               .max_receipts = 8U,
                                                               .max_fenced_operators = 8U,
                                                               .max_candidate_generations = 8U,
                                                               .max_events = 64U,
                                                               .max_audit_records = 16U,
                                                               .max_quiesce_records = 16U,
                                                               .max_rollback_records = 32U};
  const auto plan = GetParam()
                        ? detail::compile_with_process_local_holomq(graph, std::move(options))
                        : compile(graph, std::move(options));
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto session = run_async(plan);
  const bool complete = probe->wait();
  session.request_stop();
  session.wait();
  ASSERT_TRUE(complete) << (session.diagnostics().empty() ? "timeout"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(probe->valid.load());
  EXPECT_EQ(probe->starts.load(), 2U);
  EXPECT_EQ(probe->computes.load(), 2U);
  EXPECT_GT(probe->start_allocations[0], 0U);
  EXPECT_EQ(probe->start_allocations[1], 0U);
  EXPECT_EQ(probe->callback_allocations[0], 0U);
  EXPECT_EQ(probe->callback_allocations[1], 0U);
  EXPECT_EQ(probe->sample_addresses[0], probe->sample_addresses[1]);
  EXPECT_EQ(probe->input_header_addresses[0], probe->input_header_addresses[1]);
  EXPECT_EQ(probe->scratch_header_addresses[0], probe->scratch_header_addresses[1]);
}

INSTANTIATE_TEST_SUITE_P(PreparedPaths, PreparedImuRestart, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "HoloMQ" : "Direct";
                         });

}  // namespace
}  // namespace holoscan::test
