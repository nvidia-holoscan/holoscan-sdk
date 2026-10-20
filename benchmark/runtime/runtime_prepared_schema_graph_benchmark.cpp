/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmark/benchmark.h>
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
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/schema/prepared_schema_storage.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "common/allocation_counter.hpp"
#include "compiler/compile_internal.hpp"

namespace holoscan {
namespace {
using namespace std::chrono_literals;
constexpr std::size_t kMessages = 512U;
enum class Mode { kOwning, kBorrow, kPrepared };

std::int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct Counts {
  std::size_t first{};
  std::size_t total{};
  void add(std::size_t value, std::size_t call) {
    if (call == 0U)
      first = value;
    total += value;
  }
};

class OptionalAllocationCounter {
 public:
  explicit OptionalAllocationCounter(bool enabled) {
    if (enabled)
      counter_.emplace();
  }
  std::size_t count() const noexcept { return counter_ ? counter_->count() : 0U; }

 private:
  std::optional<test::ScopedAllocationCounter> counter_;
};

struct Probe {
  std::mutex mutex;
  std::condition_variable ready;
  std::array<std::atomic<std::int64_t>, kMessages> sent_at{};
  std::vector<std::int64_t> latency_ns;
  std::atomic<std::size_t> received{};
  std::atomic<bool> valid{true};
  bool count_allocations{};
  bool tracing{};
  std::size_t calls{};
  std::int64_t callback_ns{};
  Counts receive;
  Counts scratch;
  Counts publication;
  Counts callback;
  bool wait() {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 15s, [&] { return received.load() == kMessages; });
  }
};

schema::ImuT make_imu(std::uint64_t sequence) {
  schema::ImuT value;
  value.header = std::make_shared<schema::HeaderT>();
  value.header->frame_id.assign(128U, 'i');
  value.header->device_sequence = sequence;
  value.orientation = schema::Quaternionf{0.F, 0.F, 0.F, 1.F};
  value.angular_velocity = schema::Vector3f{1.F, 2.F, 3.F};
  value.linear_acceleration = schema::Vector3f{4.F, 5.F, 6.F};
  value.flags = schema::ImuFlags_ORIENTATION_VALID;
  if (sequence % 2U == 0U)
    value.orientation_covariance = std::make_shared<schema::Matrix3f>();
  if (sequence % 3U == 0U)
    value.angular_velocity_covariance = std::make_shared<schema::Matrix3f>();
  if (sequence % 5U == 0U)
    value.linear_acceleration_covariance = std::make_shared<schema::Matrix3f>();
  return value;
}

class Source final : public Operator<> {
 public:
  explicit Source(std::shared_ptr<Probe> probe) : probe_(std::move(probe)) {
    for (std::size_t index = 0U; index < values_.size(); ++index)
      values_[index] = make_imu(index + 1U);
  }
  void setup(OperatorSpec& spec) override { spec.output(output).max_emits_per_compute(1U); }
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 10us});
    return result;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    if (sent_ == kMessages)
      return {};
    const auto sequence = sent_ + 1U;
    EmitOptions options{.capture_time = context.activation_time(), .frame_id = sequence};
    if (probe_->tracing)
      options.trace = TraceContext::root(77U);
    probe_->sent_at[sent_].store(now_ns(), std::memory_order_release);
    auto emitted = output.emit_ref(values_[sent_], options);
    if (!emitted && emitted.error().code == ErrorCode::kBackpressured)
      return {};
    if (emitted)
      ++sent_;
    return emitted;
  }
  Output<schema::ImuT> output{"output"};

 private:
  std::shared_ptr<Probe> probe_;
  std::array<schema::ImuT, kMessages> values_;
  std::size_t sent_{};
};

class Transform final : public Operator<> {
 public:
  Transform(std::shared_ptr<Probe> probe, Mode mode) : probe_(std::move(probe)), mode_(mode) {}
  void setup(OperatorSpec& spec) override {
    spec.input(input).queue_depth(kMessages);
    spec.output(output).max_emits_per_compute(1U);
  }
  void start() override {
    if (mode_ == Mode::kPrepared && !scratch_) {
      auto scratch = schema::SchemaScratch<schema::ImuT>::create();
      if (!scratch)
        throw RuntimeError(scratch.error());
      scratch_.emplace(std::move(*scratch));
    }
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override {
    const auto started = now_ns();
    const OptionalAllocationCounter whole{probe_->count_allocations};
    auto result = transform();
    probe_->callback_ns += now_ns() - started;
    probe_->callback.add(whole.count(), probe_->calls++);
    return result;
  }
  Input<schema::ImuT> input{"input"};
  Output<schema::ImuT> output{"output"};

 private:
  expected<void, Error> transform() {
    std::optional<Sample<schema::ImuT>> owning;
    std::optional<SelectedSampleView<schema::ImuT>> borrowed;
    const Sample<schema::ImuT>* sample{};
    {
      const OptionalAllocationCounter counter{probe_->count_allocations};
      if (mode_ == Mode::kOwning) {
        HOLOSCAN_TRY(auto received, input.receive());
        owning.emplace(std::move(received));
        sample = &*owning;
      } else {
        HOLOSCAN_TRY(auto selected, input.selected_sample());
        borrowed.emplace(selected);
        sample = &borrowed->sample();
      }
      probe_->receive.add(counter.count(), probe_->calls);
    }
    const schema::ImuT* published = &sample->data;
    if (scratch_) {
      const OptionalAllocationCounter counter{probe_->count_allocations};
      HOLOSCAN_TRY(scratch_->copy_from(sample->data));
      scratch_->value().linear_acceleration = schema::Vector3f{4.F, 5.F, 7.F};
      published = &scratch_->value();
      probe_->scratch.add(counter.count(), probe_->calls);
    }
    expected<void, Error> result;
    {
      const OptionalAllocationCounter counter{probe_->count_allocations};
      result = output.emit_ref(*published, sensor_io::derive_from(sample->metadata));
      probe_->publication.add(counter.count(), probe_->calls);
    }
    if (scratch_) {
      const OptionalAllocationCounter counter{probe_->count_allocations};
      auto reset = scratch_->reset();
      probe_->scratch.total += counter.count();
      if (!reset)
        return reset;
    }
    return result;
  }

  std::shared_ptr<Probe> probe_;
  Mode mode_;
  std::optional<schema::SchemaScratch<schema::ImuT>> scratch_;
};

class Sink final : public Operator<> {
 public:
  Sink(std::shared_ptr<Probe> probe, Mode mode) : probe_(std::move(probe)), mode_(mode) {}
  void setup(OperatorSpec& spec) override { spec.input(input).queue_depth(kMessages); }
  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override {
    HOLOSCAN_TRY(auto view, input.selected_sample());
    const auto sequence = probe_->received.fetch_add(1U) + 1U;
    if (!view.data().header || view.data().header->device_sequence != sequence ||
        view.data().linear_acceleration.z() != (mode_ == Mode::kPrepared ? 7.F : 6.F) ||
        view.metadata().frame_id != sequence ||
        (probe_->tracing && view.metadata().trace_id != 77U))
      probe_->valid.store(false);
    probe_->latency_ns.push_back(now_ns() -
                                 probe_->sent_at[sequence - 1U].load(std::memory_order_acquire));
    probe_->ready.notify_all();
    return {};
  }
  Input<schema::ImuT> input{"input"};

 private:
  std::shared_ptr<Probe> probe_;
  Mode mode_;
};

std::shared_ptr<Probe> run_graph(Mode mode, bool transport, bool tracing, bool count) {
  auto probe = std::make_shared<Probe>();
  probe->count_allocations = count;
  probe->tracing = tracing;
  probe->latency_ns.reserve(kMessages);
  Graph graph{"prepared-imu-benchmark"};
  auto source = graph.op<Source>("source", probe);
  auto transform = graph.op<Transform>("transform", probe, mode);
  auto sink = graph.op<Sink>("sink", probe, mode);
  graph.add_flow(source->output, transform->input, ConnectionOptions{.queue_depth = kMessages});
  graph.add_flow(transform->output, sink->input, ConnectionOptions{.queue_depth = kMessages});
  if (transport) {
    graph.partition("source").add(source);
    graph.partition("transform").add(transform);
    graph.partition("sink").add(sink);
  } else {
    graph.partition("shared").add(source).add(transform).add(sink);
  }
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  const auto plan =
      transport ? detail::compile_with_process_local_holomq(graph, {}) : compile(graph);
  if (!plan.ok())
    return {};
  auto session = run_async(plan);
  const bool complete = probe->wait();
  session.request_stop();
  session.wait();
  if (!complete || !probe->valid.load() || probe->received.load() != kMessages ||
      session.termination() != RunTermination::kStopped)
    return {};
  return probe;
}

// REQ: SCHEMA-008
void BM_PreparedImuGraph(benchmark::State& state) {
  const auto mode = static_cast<Mode>(state.range(0));
  const bool transport = state.range(1) != 0;
  const bool tracing = state.range(2) != 0;
  auto counted = run_graph(mode, transport, tracing, true);
  if (!counted) {
    state.SkipWithError("prepared IMU graph correctness or progress failed");
    return;
  }
  const double calls = static_cast<double>(counted->calls);
  state.counters["receive_allocations_per_hop"] = counted->receive.total / calls;
  state.counters["scratch_allocations_per_hop"] = counted->scratch.total / calls;
  state.counters["publication_allocations_per_hop"] = counted->publication.total / calls;
  state.counters["callback_allocations_per_hop"] = counted->callback.total / calls;
  state.counters["first_callback_allocations"] = counted->callback.first;
  for (auto _ : state) {
    auto timed = run_graph(mode, transport, tracing, false);
    if (!timed) {
      state.SkipWithError("timed prepared IMU graph failed");
      break;
    }
    state.SetIterationTime(static_cast<double>(timed->callback_ns) / calls / 1e9);
    std::sort(timed->latency_ns.begin(), timed->latency_ns.end());
    state.counters["source_to_sink_p50_ns"] = timed->latency_ns[kMessages / 2U];
    state.counters["source_to_sink_p95_ns"] = timed->latency_ns[kMessages * 95U / 100U];
    state.counters["source_to_sink_p99_ns"] = timed->latency_ns[kMessages * 99U / 100U];
    state.counters["messages_per_run"] = kMessages;
  }
}

BENCHMARK(BM_PreparedImuGraph)
    ->ArgsProduct({{0, 1, 2}, {0, 1}, {0, 1}})
    ->ArgNames({"mode", "transport", "tracing"})
    ->UseManualTime()
    ->Iterations(1)
    ->Repetitions(3);

}  // namespace
}  // namespace holoscan
