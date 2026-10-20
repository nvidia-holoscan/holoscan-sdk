/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

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
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "common/allocation_counter.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "support/prepared_descriptor_fixture.hpp"
#include "support/retirement_ordered_acknowledgment.hpp"

namespace holoscan::test::prepared_descriptor {
namespace {
using namespace std::chrono_literals;
constexpr std::size_t kSamples = kPatterns * 2;
enum class Mode { kOwning, kBorrow, kEdit };

std::size_t pattern(std::uint64_t sequence) {
  return (sequence + kPatterns - 2) % kPatterns;  // maximal metadata on the first message
}

struct Counts {
  std::size_t first{};
  std::size_t maximum{};
  std::size_t total{};
  void add(std::size_t count, std::size_t call) {
    if (call == 0)
      first = count;
    maximum = std::max(maximum, count);
    total += count;
  }
};

struct Probe {
  explicit Probe(std::size_t consumers) : sinks(consumers) {
    for (std::size_t i = 0; i < kPatterns; ++i) {
      original[i] = make_metadata(i);
      edited[i] = make_metadata(i);
      edit_metadata(edited[i]);
    }
  }
  const std::size_t sinks;
  std::array<Packet, kPatterns> original;
  std::array<Packet, kPatterns> edited;
  std::array<std::atomic<std::uint64_t>, 2> acknowledged{};
  std::array<std::atomic<std::int64_t>, kSamples> capture_times{};
  std::mutex mutex;
  std::condition_variable ready;
  std::atomic<bool> valid{true};
  // Only the transform writes these; read after session.wait().
  std::size_t calls{};
  Counts receive;
  Counts scratch;
  Counts publication;
  Counts callback;
  bool through(std::uint64_t sequence) const {
    for (std::size_t i = 0; i < sinks; ++i)
      if (acknowledged[i].load(std::memory_order_acquire) < sequence)
        return false;
    return true;
  }
  bool wait() {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 15s, [&] { return through(kSamples); });
  }
};

std::string_view output_owner(const ExecutionPlan& plan, PortId output) {
  for (const auto& port : plan.ports()) {
    if (port.id != output)
      continue;
    for (const auto& vertex : plan.graph().vertices())
      if (vertex.id == port.owner)
        return vertex.name;
  }
  return {};
}

class Source final : public Operator<> {
 public:
  explicit Source(std::shared_ptr<Probe> probe)
      : probe_(std::move(probe)), values_(probe_->original) {}
  void setup(OperatorSpec& spec) override { declare_output(spec, output, MemoryKind::kHost); }
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1us});
    return result;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    if (sent_ == kSamples || !probe_->through(sent_))
      return {};
    auto bundle = output.allocate_tensors(shape_requests());
    if (!bundle && bundle.error().code == ErrorCode::kResourceExhausted)
      return {};
    if (!bundle)
      return make_unexpected(std::move(bundle).error());
    const auto sequence = sent_ + 1;
    HOLOSCAN_TRY(bundle->write_host_and_commit(Fields::first, [&](const TensorOutputWriteGuard& w) {
      *w.data_as<std::uint64_t>() = sequence;
    }));
    HOLOSCAN_TRY(bundle->write_host_and_commit(
        Fields::second,
        [&](const TensorOutputWriteGuard& w) { *w.data_as<std::uint64_t>() = sequence + 1000; }));
    auto& value = values_[pattern(sequence)];
    value.sequence = sequence;
    probe_->capture_times[sent_].store(context.activation_time().timestamp_ns,
                                       std::memory_order_release);
    auto emitted = output.emit_tensors(std::move(*bundle),
                                       value,
                                       {.capture_time = context.activation_time(),
                                        .frame_id = sequence,
                                        .trace = TraceContext::root(77),
                                        .flags = SampleFlags::kDegraded});
    value.sequence = 0;  // publication owns encoded metadata before returning
    if (!emitted && emitted.error().code == ErrorCode::kBackpressured)
      return {};
    if (emitted)
      ++sent_;
    return emitted;
  }
  Output<Packet> output{"output"};

 private:
  std::shared_ptr<Probe> probe_;
  std::array<Packet, kPatterns> values_;
  std::uint64_t sent_{};
};

class Transform final : public Operator<> {
 public:
  Transform(std::shared_ptr<Probe> probe, Mode mode) : probe_(std::move(probe)), mode_(mode) {}
  void setup(OperatorSpec& spec) override {
    declare_input(spec, input, MemoryKind::kHost);
    declare_output(spec, output, MemoryKind::kHost);
  }
  void start() override {
    if (mode_ == Mode::kEdit) {
      auto scratch = Scratch::create();
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
    const ScopedAllocationCounter count;
    auto result = transform();
    probe_->callback.add(count.count(), probe_->calls++);
    return result;
  }
  Input<Packet> input{"input"};
  Output<Packet> output{"output"};

 private:
  expected<void, Error> transform() {
    std::optional<Sample<Packet>> owning;
    std::optional<SelectedSampleView<Packet>> borrowed;
    const Sample<Packet>* sample{};
    {
      const ScopedAllocationCounter count;
      if (mode_ == Mode::kOwning) {
        HOLOSCAN_TRY(auto value, input.receive());
        owning.emplace(std::move(value));
        sample = &*owning;
      } else {
        HOLOSCAN_TRY(auto value, input.selected_sample());
        borrowed.emplace(value);
        sample = &borrowed->sample();
      }
      probe_->receive.add(count.count(), probe_->calls);
    }
    const auto& value = sample->data;
    const auto sequence = probe_->calls + 1;
    if (value.sequence != sequence || !value.first || !value.second ||
        !same_metadata(value, probe_->original[pattern(sequence)]))
      return make_unexpected(Error{ErrorCode::kFailure});
    const auto first_owners = value.first.use_count();
    const auto second_owners = value.second.use_count();
    const Packet* published = &value;
    if (scratch_) {
      const ScopedAllocationCounter count;
      HOLOSCAN_TRY(scratch_->copy_metadata_from(value));
      edit_metadata(scratch_->descriptor());
      published = &scratch_->descriptor();
      if (published->first || published->second ||
          (value.metadata && value.metadata == published->metadata) ||
          value.first.use_count() != first_owners || value.second.use_count() != second_owners)
        return make_unexpected(Error{ErrorCode::kFailure});
      probe_->scratch.add(count.count(), probe_->calls);
    }
    {
      const ScopedAllocationCounter count;
      HOLOSCAN_TRY(auto bundle, output.allocate_tensors(shape_requests()));
      const auto write = [&](const TensorOutputWriteGuard& writer, const Tensor& tensor) {
        if (writer.data() == value.first->data() || writer.data() == value.second->data())
          probe_->valid.store(false);
        *writer.data_as<std::uint64_t>() = *tensor.data_as<std::uint64_t>() + 100;
      };
      HOLOSCAN_TRY(bundle.write_host_and_commit(
          Fields::first, [&](const TensorOutputWriteGuard& w) { write(w, *value.first); }));
      HOLOSCAN_TRY(bundle.write_host_and_commit(
          Fields::second, [&](const TensorOutputWriteGuard& w) { write(w, *value.second); }));
      HOLOSCAN_TRY(output.emit_tensors(
          std::move(bundle), *published, sensor_io::derive_from(sample->metadata)));
      probe_->publication.add(count.count(), probe_->calls);
    }
    if (scratch_) {
      const ScopedAllocationCounter count;
      HOLOSCAN_TRY(scratch_->reset());
      probe_->scratch.total += count.count();
    }
    if (!same_metadata(value, probe_->original[pattern(sequence)]) ||
        *value.first->data_as<std::uint64_t>() != sequence ||
        *value.second->data_as<std::uint64_t>() != sequence + 1000)
      return make_unexpected(Error{ErrorCode::kFailure});
    return {};
  }
  std::shared_ptr<Probe> probe_;
  Mode mode_;
  std::optional<Scratch> scratch_;
};

class Sink final : public Operator<> {
 public:
  Sink(std::shared_ptr<Probe> probe, Mode mode, std::size_t index)
      : probe_(std::move(probe)), mode_(mode), index_(index) {}
  void setup(OperatorSpec& spec) override {
    declare_input(spec, input, MemoryKind::kHost);
    spec.output(acknowledgment).max_emits_per_compute(1);
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override {
    HOLOSCAN_TRY(auto view, input.selected_sample());
    const auto sequence = probe_->acknowledged[index_].load(std::memory_order_acquire) + 1;
    const auto& value = view.data();
    const auto& expected = mode_ == Mode::kEdit ? probe_->edited : probe_->original;
    if (sequence > kSamples || value.sequence != sequence || !value.first || !value.second ||
        *value.first->data_as<std::uint64_t>() != sequence + 100 ||
        *value.second->data_as<std::uint64_t>() != sequence + 1100 ||
        !same_metadata(value, expected[pattern(sequence)]) ||
        view.metadata().frame_id != sequence || view.metadata().sequence != sequence ||
        view.metadata().trace_id != 77 || view.metadata().flags != SampleFlags::kDegraded ||
        !view.metadata().source_clock_id.valid() ||
        view.metadata().capture_timestamp_ns != probe_->capture_times[sequence - 1].load())
      probe_->valid.store(false);
    return acknowledgment.emit(sequence);
  }
  Input<Packet> input{"input"};
  Output<std::uint64_t> acknowledgment{"acknowledgment"};

 private:
  std::shared_ptr<Probe> probe_;
  Mode mode_;
  std::size_t index_;
};

class PreparedDescriptorGraph
    : public ::testing::TestWithParam<std::tuple<Mode, bool, std::size_t>> {};

// REQ: SCHEMA-008
// REQ: SCHEMA-009
TEST_P(PreparedDescriptorGraph, TwoTensorTransformsPreserveMetadataBackingAndProvenance) {
  const auto [mode, separate, sinks] = GetParam();
  auto probe = std::make_shared<Probe>(sinks);
  Graph graph{"prepared-descriptor-transform"};
  auto source = graph.op<Source>("source", probe);
  auto transform = graph.op<Transform>("transform", probe, mode);
  graph.add_flow(source->output, transform->input, ConnectionOptions{.queue_depth = 1});
  auto source_partition = graph.partition(separate ? "source" : "shared");
  source_partition.add(source);
  if (separate)
    graph.partition("transform").add(transform);
  else
    source_partition.add(transform);
  std::vector<
      std::pair<OperatorHandle<Sink>, OperatorHandle<RetirementOrderedAcknowledgment<Probe>>>>
      pairs;
  for (std::size_t i = 0; i < sinks; ++i) {
    const auto name = std::string("sink") + std::to_string(i);
    auto sink = graph.op<Sink>(name, probe, mode, i);
    auto ack = graph.op<RetirementOrderedAcknowledgment<Probe>>(name + "_ack", probe, i);
    graph.add_flow(transform->output, sink->input, ConnectionOptions{.queue_depth = 1});
    graph.add_flow(sink->acknowledgment, ack->input, ConnectionOptions{.queue_depth = 1});
    if (separate)
      graph.partition(name).add(sink).add(ack);
    else
      source_partition.add(sink).add(ack);
    pairs.emplace_back(sink, ack);
  }
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  // The generic multi-Tensor payload uses HoloMQ in both owner layouts, never direct retention.
  const auto plan = detail::compile_with_process_local_holomq(graph, {});
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto partitions = plan.partitions();
  for (const auto& [sink, ack] : pairs)
    ASSERT_NO_THROW(
        require_retirement_ordered_acknowledgment(partitions.of(sink), partitions.of(ack)));
  EXPECT_EQ(partitions.size(), separate ? 2 + sinks : 1);
  ASSERT_EQ(plan.tensor_field_allocations().size(), 4U);
  for (const auto& allocation : plan.tensor_field_allocations()) {
    EXPECT_EQ(allocation.max_emits_per_compute, 1U);
    EXPECT_EQ(allocation.max_outstanding_loans, 1U);
    const auto owner = output_owner(plan, allocation.output);
    ASSERT_FALSE(owner.empty());
    // Producer and retirement slots plus queue, typed holder, and endpoint head per sink.
    EXPECT_EQ(allocation.slot_count, owner == "source" ? 5U : 2U + 3U * sinks);
    EXPECT_EQ(allocation.control_capacity, allocation.slot_count);
    EXPECT_EQ(allocation.readiness_capacity, allocation.slot_count);
    EXPECT_EQ(allocation.release_capacity, allocation.slot_count);
    EXPECT_EQ(allocation.retirement_capacity, allocation.slot_count);
    EXPECT_EQ(allocation.quarantine_capacity, allocation.slot_count);
  }
  const auto* compiled = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(compiled, nullptr);
  for (const auto& subscriber : compiled->process_local_channel_subscribers)
    EXPECT_EQ(subscriber.requested_queue_depth, 1U);
  auto session = run_async(plan);
  const bool completed = probe->wait();
  session.request_stop();
  session.wait();
  ASSERT_TRUE(completed) << (session.diagnostics().empty() ? "timeout"
                                                           : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(probe->valid.load());
  EXPECT_EQ(probe->calls, kSamples);
  EXPECT_EQ(probe->scratch.total, 0U);
  // HoloMQ prepares its release-callback entries lazily on the first publication. Allow that
  // bounded transport cost (three/five entries here), but no recurring allocation after it.
  EXPECT_LE(probe->publication.first, 1U + 2U * sinks);
  EXPECT_EQ(probe->publication.total, probe->publication.first);
  if (mode == Mode::kOwning) {
    EXPECT_GT(probe->receive.total, 0U);
  } else {
    EXPECT_EQ(probe->receive.total, 0U);
    // Attribute transport/publication residuals separately from the prepared metadata path.
    EXPECT_EQ(probe->callback.total, probe->publication.total);
  }
  RecordProperty("receive_allocations", std::to_string(probe->receive.total));
  RecordProperty("receive_first_allocations", std::to_string(probe->receive.first));
  RecordProperty("receive_max_allocations", std::to_string(probe->receive.maximum));
  RecordProperty("scratch_allocations", std::to_string(probe->scratch.total));
  RecordProperty("publication_first_allocations", std::to_string(probe->publication.first));
  RecordProperty("publication_max_allocations", std::to_string(probe->publication.maximum));
  RecordProperty("publication_total_allocations", std::to_string(probe->publication.total));
  RecordProperty("callback_first_allocations", std::to_string(probe->callback.first));
  RecordProperty("callback_max_allocations", std::to_string(probe->callback.maximum));
  RecordProperty("callback_total_allocations", std::to_string(probe->callback.total));
}

std::string case_name(const ::testing::TestParamInfo<PreparedDescriptorGraph::ParamType>& info) {
  const auto [mode, separate, sinks] = info.param;
  return std::string(mode == Mode::kOwning   ? "Owning"
                     : mode == Mode::kBorrow ? "Borrow"
                                             : "Edit") +
         (separate ? "SplitHoloMQ" : "SharedHoloMQ") + std::to_string(sinks);
}

INSTANTIATE_TEST_SUITE_P(PreparedPaths, PreparedDescriptorGraph,
                         ::testing::Combine(::testing::Values(Mode::kOwning, Mode::kBorrow,
                                                              Mode::kEdit),
                                            ::testing::Bool(), ::testing::Values(1U, 2U)),
                         case_name);

struct OversizeProbe {
  std::mutex mutex;
  std::condition_variable ready;
  std::atomic<bool> rejected{};
  std::atomic<bool> received{};
  std::atomic<bool> valid{true};
  std::atomic<int> rejection_code{static_cast<int>(ErrorCode::kSuccess)};
  bool wait() {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 15s, [&] { return received.load(std::memory_order_acquire); });
  }
};

class OversizeSource final : public Operator<> {
 public:
  explicit OversizeSource(std::shared_ptr<OversizeProbe> probe) : probe_(std::move(probe)) {}
  void setup(OperatorSpec& spec) override { declare_output(spec, output, MemoryKind::kHost); }
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1us});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override {
    if (published_)
      return {};
    HOLOSCAN_TRY(auto bundle, output.allocate_tensors(shape_requests()));
    HOLOSCAN_TRY(bundle.write_host_and_commit(
        Fields::first,
        [&](const TensorOutputWriteGuard& writer) { *writer.data_as<std::uint64_t>() = 7; }));
    HOLOSCAN_TRY(bundle.write_host_and_commit(
        Fields::second,
        [&](const TensorOutputWriteGuard& writer) { *writer.data_as<std::uint64_t>() = 11; }));
    auto value = make_metadata(0);
    value.sequence = 1;
    if (!rejected_) {
      value.label.assign(129, 'x');  // The generated bound is 128 bytes.
      auto emitted = output.emit_tensors(std::move(bundle), value);
      if (emitted)
        return make_unexpected(Error{ErrorCode::kFailure});
      if (emitted.error().code == ErrorCode::kBackpressured)
        return {};
      probe_->rejection_code.store(static_cast<int>(emitted.error().code),
                                   std::memory_order_release);
      rejected_ = true;
      probe_->rejected.store(true, std::memory_order_release);
      return {};
    }
    auto emitted = output.emit_tensors(std::move(bundle), value);
    if (!emitted && emitted.error().code == ErrorCode::kBackpressured)
      return {};
    HOLOSCAN_TRY(emitted);
    published_ = true;
    return {};
  }
  Output<Packet> output{"output"};

 private:
  std::shared_ptr<OversizeProbe> probe_;
  bool rejected_{};
  bool published_{};
};

class OversizeSink final : public Operator<> {
 public:
  explicit OversizeSink(std::shared_ptr<OversizeProbe> probe) : probe_(std::move(probe)) {}
  void setup(OperatorSpec& spec) override { declare_input(spec, input, MemoryKind::kHost); }
  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override {
    HOLOSCAN_TRY(auto view, input.selected_sample());
    const auto& value = view.data();
    probe_->valid.store(probe_->rejected.load(std::memory_order_acquire) && value.sequence == 1 &&
                            value.label.empty() && value.first && value.second &&
                            *value.first->data_as<std::uint64_t>() == 7 &&
                            *value.second->data_as<std::uint64_t>() == 11,
                        std::memory_order_release);
    probe_->received.store(true, std::memory_order_release);
    probe_->ready.notify_all();
    return {};
  }
  Input<Packet> input{"input"};

 private:
  std::shared_ptr<OversizeProbe> probe_;
};

// REQ: SCHEMA-008
// REQ: SCHEMA-009
TEST(PreparedDescriptorGraph, OversizeMetadataReturnsLoansAndExactPoolsContinue) {
  auto probe = std::make_shared<OversizeProbe>();
  Graph graph{"prepared-descriptor-oversize-recovery"};
  auto source = graph.op<OversizeSource>("source", probe);
  auto sink = graph.op<OversizeSink>("sink", probe);
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 1});
  graph.partition("source").add(source);
  graph.partition("sink").add(sink);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  const auto plan = detail::compile_with_process_local_holomq(graph, {});
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_field_allocations().size(), 2U);
  for (const auto& allocation : plan.tensor_field_allocations()) {
    EXPECT_EQ(output_owner(plan, allocation.output), "source");
    EXPECT_EQ(allocation.slot_count, 5U);
  }
  auto session = run_async(plan);
  const bool completed = probe->wait();
  session.request_stop();
  session.wait();
  ASSERT_TRUE(completed) << (session.diagnostics().empty() ? "timeout"
                                                           : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(probe->rejected.load());
  EXPECT_EQ(probe->rejection_code.load(), static_cast<int>(ErrorCode::kResourceExhausted));
  EXPECT_TRUE(probe->valid.load());
}

}  // namespace
}  // namespace holoscan::test::prepared_descriptor
