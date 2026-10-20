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
#include <tuple>
#include <utility>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "common/allocation_counter.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "support/retirement_ordered_acknowledgment.hpp"

namespace holoscan::test {
namespace {
using namespace std::chrono_literals;
enum class ImuMode { kOwning, kBorrow, kEdit };
constexpr std::uint64_t kSamples = 64;

struct ImuGraphProbe {
  explicit ImuGraphProbe(std::size_t consumers) : sinks(consumers) {}
  const std::size_t sinks;
  std::array<std::atomic<std::uint64_t>, 2> acknowledged{};
  std::array<std::atomic<std::int64_t>, kSamples> capture_times{};
  std::atomic<bool> valid{true};
  std::mutex mutex;
  std::condition_variable ready;
  std::shared_ptr<AcknowledgmentRetirementGate> acknowledgment_gate;
  // Written only by the transform lane and read after session.wait().
  std::size_t calls{};
  std::size_t receive_allocations{};
  std::size_t scratch_allocations{};
  std::size_t emit_allocations{};
  std::size_t callback_allocations{};
  std::size_t first_callback_allocations{};
  std::size_t maximum_callback_allocations{};

  bool acknowledged_through(std::uint64_t sequence) const {
    for (std::size_t i = 0; i < sinks; ++i)
      if (acknowledged[i].load(std::memory_order_acquire) < sequence)
        return false;
    return true;
  }
  bool wait() {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [&] { return acknowledged_through(kSamples); });
  }
};

unsigned presence(std::uint64_t sequence) {
  // Full metadata on the first message, followed by all absent/present combinations twice.
  return static_cast<unsigned>((sequence + 30) % 32);
}

schema::ImuT imu_source(unsigned mask) {
  schema::ImuT value;
  value.orientation = schema::Quaternionf{1.F, 2.F, 3.F, 4.F};
  value.angular_velocity = schema::Vector3f{5.F, 6.F, 7.F};
  if (mask & 1U) {
    value.header = std::make_shared<schema::HeaderT>();
    value.header->frame_id.assign(128, 'i');
    if (mask & 2U) {
      value.header->stamp = std::make_shared<schema::TimeT>();
      value.header->stamp->time_ns = 123;
      value.header->stamp->acqtime_ns = 124;
    }
  }
  if (mask & 4U)
    value.orientation_covariance = std::make_shared<schema::Matrix3f>();
  if (mask & 8U)
    value.angular_velocity_covariance = std::make_shared<schema::Matrix3f>();
  if (mask & 16U)
    value.linear_acceleration_covariance = std::make_shared<schema::Matrix3f>();
  return value;
}

bool valid_imu(const schema::ImuT& value, std::uint64_t sequence, bool edited) {
  const auto mask = presence(sequence);
  const auto covariance = [edited](const auto& owner, bool present) {
    return static_cast<bool>(owner) == present &&
           (!owner || std::all_of(owner->m()->begin(), owner->m()->end(), [edited](float item) {
             return item == (edited ? 2.F : 0.F);
           }));
  };
  if (static_cast<bool>(value.header) != ((mask & 1U) != 0) || value.orientation.w() != 4.F ||
      value.angular_velocity.z() != 7.F ||
      value.linear_acceleration.x() != static_cast<float>(sequence) ||
      value.linear_acceleration.z() != (edited ? 30.F : 3.F) ||
      value.flags != (edited ? schema::ImuFlags_UNCALIBRATED : static_cast<schema::ImuFlags>(0)) ||
      !covariance(value.orientation_covariance, mask & 4U) ||
      !covariance(value.angular_velocity_covariance, mask & 8U) ||
      !covariance(value.linear_acceleration_covariance, mask & 16U))
    return false;
  if (!value.header)
    return true;
  const auto& header = *value.header;
  return header.device_sequence == sequence && header.frame_id.size() == 128 &&
         header.frame_id.front() == (edited ? 'e' : 'i') &&
         std::all_of(
             header.frame_id.begin() + 1, header.frame_id.end(), [](char c) { return c == 'i'; }) &&
         static_cast<bool>(header.stamp) == ((mask & 2U) != 0) &&
         (!header.stamp || (header.stamp->time_ns == 123 && header.stamp->acqtime_ns == 124));
}

class PreparedImuSource final : public Operator<> {
 public:
  explicit PreparedImuSource(std::shared_ptr<ImuGraphProbe> probe) : probe_(std::move(probe)) {
    for (unsigned mask = 0; mask < 32; ++mask)
      values_[mask] = imu_source(mask);
  }
  void setup(OperatorSpec& spec) override { spec.output(output).max_emits_per_compute(1); }
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1us});
    return result;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    if (sent_ == kSamples || !probe_->acknowledged_through(sent_))
      return {};
    auto& value = values_[presence(sent_ + 1)];
    value.linear_acceleration = schema::Vector3f{static_cast<float>(sent_ + 1), 2.F, 3.F};
    if (value.header)
      value.header->device_sequence = sent_ + 1;
    probe_->capture_times[sent_].store(context.activation_time().timestamp_ns,
                                       std::memory_order_release);
    auto published = output.emit_ref(value,
                                     {.capture_time = context.activation_time(),
                                      .frame_id = sent_ + 1,
                                      .trace = TraceContext::root(77),
                                      .flags = SampleFlags::kDegraded});
    // Check synchronous encoding even if transport or a downstream partition is delayed.
    value.linear_acceleration = {};
    if (!published && published.error().code == ErrorCode::kBackpressured)
      return {};
    if (!published)
      return published;
    ++sent_;
    return {};
  }
  Output<schema::ImuT> output{"output"};

 private:
  std::shared_ptr<ImuGraphProbe> probe_;
  std::array<schema::ImuT, 32> values_;
  std::uint64_t sent_{};
};

class PreparedImuTransform final : public Operator<> {
 public:
  PreparedImuTransform(std::shared_ptr<ImuGraphProbe> probe, ImuMode mode)
      : probe_(std::move(probe)), mode_(mode) {}
  void setup(OperatorSpec& spec) override {
    spec.input(input).queue_depth(1);
    spec.output(output).max_emits_per_compute(1);
  }
  void start() override {
    if (mode_ == ImuMode::kEdit) {
      auto prepared = schema::SchemaScratch<schema::ImuT>::create();
      if (!prepared)
        throw RuntimeError(prepared.error());
      scratch_.emplace(std::move(*prepared));
    }
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override {
    const ScopedAllocationCounter whole;
    auto result = transform();
    const auto allocations = whole.count();
    if (probe_->calls == 0)
      probe_->first_callback_allocations = allocations;
    probe_->maximum_callback_allocations =
        std::max(probe_->maximum_callback_allocations, allocations);
    probe_->callback_allocations += allocations;
    ++probe_->calls;
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
      const ScopedAllocationCounter counter;
      if (mode_ == ImuMode::kOwning) {
        auto received = input.receive();
        if (!received)
          return make_unexpected(std::move(received).error());
        owning.emplace(std::move(*received));
        sample = &*owning;
      } else {
        auto received = input.selected_sample();
        if (!received)
          return make_unexpected(std::move(received).error());
        borrowed.emplace(*received);
        sample = &borrowed->sample();
      }
      probe_->receive_allocations += counter.count();
    }
    const auto sequence = probe_->calls + 1;
    if (!valid_imu(sample->data, sequence, false))
      return make_unexpected(Error{ErrorCode::kFailure});
    const schema::ImuT* published = &sample->data;
    if (mode_ == ImuMode::kEdit) {
      const ScopedAllocationCounter counter;
      auto copied = scratch_->copy_from(sample->data);
      if (!copied)
        return copied;
      auto& value = scratch_->value();
      value.linear_acceleration = schema::Vector3f{static_cast<float>(sequence), 2.F, 30.F};
      value.flags = schema::ImuFlags_UNCALIBRATED;
      if (value.header)
        value.header->frame_id.front() = 'e';
      const float matrix_values[9]{2.F, 2.F, 2.F, 2.F, 2.F, 2.F, 2.F, 2.F, 2.F};
      const schema::Matrix3f matrix{flatbuffers::span<const float, 9>{matrix_values}};
      for (auto* owner : {&value.orientation_covariance,
                          &value.angular_velocity_covariance,
                          &value.linear_acceleration_covariance})
        if (*owner)
          **owner = matrix;
      published = &value;
      probe_->scratch_allocations += counter.count();
      if (!valid_imu(sample->data, sequence, false))
        return make_unexpected(Error{ErrorCode::kFailure});
    }
    expected<void, Error> result;
    {
      const ScopedAllocationCounter counter;
      result = mode_ == ImuMode::kOwning
                   ? output.emit(std::move(*owning))
                   : output.emit_ref(*published, sensor_io::derive_from(sample->metadata));
      probe_->emit_allocations += counter.count();
    }
    if (scratch_) {
      const ScopedAllocationCounter counter;
      auto reset = scratch_->reset();
      if (!reset)
        return reset;
      probe_->scratch_allocations += counter.count();
    }
    return result;
  }
  std::shared_ptr<ImuGraphProbe> probe_;
  ImuMode mode_;
  std::optional<schema::SchemaScratch<schema::ImuT>> scratch_;
};

class PreparedImuSink final : public Operator<> {
 public:
  PreparedImuSink(std::shared_ptr<ImuGraphProbe> probe, ImuMode mode, std::size_t index)
      : probe_(std::move(probe)), mode_(mode), index_(index) {}
  void setup(OperatorSpec& spec) override {
    spec.input(input).queue_depth(1);
    spec.output(acknowledgment).max_emits_per_compute(1);
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override {
    auto view = input.selected_sample();
    if (!view)
      return make_unexpected(std::move(view).error());
    const auto sequence = probe_->acknowledged[index_].load(std::memory_order_relaxed) + 1;
    const auto& metadata = view->metadata();
    if (!valid_imu(view->data(), sequence, mode_ == ImuMode::kEdit) ||
        metadata.sequence != sequence || metadata.frame_id != sequence || metadata.trace_id != 77 ||
        metadata.flags != SampleFlags::kDegraded || !metadata.source_clock_id.valid() ||
        sequence > kSamples ||
        metadata.capture_timestamp_ns !=
            probe_->capture_times[sequence - 1].load(std::memory_order_acquire))
      probe_->valid.store(false);
    // A separate callback on this same partition grants source credit after input retirement.
    auto emitted = acknowledgment.emit(sequence);
    if (!emitted)
      return emitted;
    if (index_ == 0 && sequence == 1 && probe_->acknowledgment_gate)
      return probe_->acknowledgment_gate->hold();
    return {};
  }
  Input<schema::ImuT> input{"input"};
  Output<std::uint64_t> acknowledgment{"acknowledgment"};

 private:
  std::shared_ptr<ImuGraphProbe> probe_;
  ImuMode mode_;
  std::size_t index_;
};

class PreparedImuGraph : public ::testing::TestWithParam<std::tuple<ImuMode, bool, std::size_t>> {};

// REQ: SCHEMA-008
// REQ: SCHEMA-009
// REQ: SCHEMA-010
TEST_P(PreparedImuGraph, CompiledTransformsPreserveOptionalsProvenanceAndIndependentStorage) {
  const auto [mode, transport, sinks] = GetParam();
  for (unsigned run = 0; run < 2; ++run) {
    auto probe = std::make_shared<ImuGraphProbe>(sinks);
    probe->acknowledgment_gate = std::make_shared<AcknowledgmentRetirementGate>();
    Graph graph{"prepared-imu-transform"};
    auto source = graph.op<PreparedImuSource>("source", probe);
    auto transform = graph.op<PreparedImuTransform>("transform", probe, mode);
    auto sink = graph.op<PreparedImuSink>("sink", probe, mode, std::size_t{0});
    auto acknowledgment =
        graph.op<RetirementOrderedAcknowledgment<ImuGraphProbe>>("sink_ack", probe, std::size_t{0});
    std::vector<std::pair<OperatorHandle<PreparedImuSink>,
                          OperatorHandle<RetirementOrderedAcknowledgment<ImuGraphProbe>>>>
        acknowledgment_pairs{{sink, acknowledgment}};
    graph.add_flow(source->output, transform->input, ConnectionOptions{.queue_depth = 1});
    graph.add_flow(transform->output, sink->input, ConnectionOptions{.queue_depth = 1});
    graph.add_flow(
        sink->acknowledgment, acknowledgment->input, ConnectionOptions{.queue_depth = 1});
    graph.partition("source").add(source);
    graph.partition("transform").add(transform);
    graph.partition("sink").add(sink).add(acknowledgment);
    if (sinks == 2) {
      auto sibling = graph.op<PreparedImuSink>("sibling", probe, mode, std::size_t{1});
      auto sibling_ack = graph.op<RetirementOrderedAcknowledgment<ImuGraphProbe>>(
          "sibling_ack", probe, std::size_t{1});
      graph.add_flow(transform->output, sibling->input, ConnectionOptions{.queue_depth = 1});
      graph.add_flow(
          sibling->acknowledgment, sibling_ack->input, ConnectionOptions{.queue_depth = 1});
      graph.partition("sibling").add(sibling).add(sibling_ack);
      acknowledgment_pairs.emplace_back(sibling, sibling_ack);
    }
    graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
    const auto plan =
        transport ? detail::compile_with_process_local_holomq(graph, {}) : compile(graph);
    ASSERT_TRUE(plan.ok()) << plan.json();
    const auto partitions = plan.partitions();
    for (const auto& [data_sink, ack] : acknowledgment_pairs)
      ASSERT_NO_THROW(
          require_retirement_ordered_acknowledgment(partitions.of(data_sink), partitions.of(ack)));
    const auto* compiled = detail::ExecutionPlanAccess::state(plan);
    ASSERT_NE(compiled, nullptr);
    for (const auto& subscriber : compiled->process_local_channel_subscribers)
      EXPECT_EQ(subscriber.requested_queue_depth, 1U);
    std::size_t direct{};
    for (const auto& service : compiled->process_local_channel_services)
      direct += service.delivery_kind == detail::ProcessLocalChannelDeliveryKind::kDirectRetained;
    // Source and transform delivery families, including fanout, plus each acknowledgment edge.
    EXPECT_EQ(direct, transport ? 0U : 2U + sinks);
    EXPECT_EQ(plan.execution_partitions().size(), 2U + sinks);
    auto session = run_async(plan);
    EXPECT_TRUE(probe->acknowledgment_gate->wait_until_held())
        << "Timed out waiting for the sink to hold the retirement acknowledgment gate";
    EXPECT_EQ(probe->acknowledged[0].load(std::memory_order_acquire), 0U);
    probe->acknowledgment_gate->release();
    const bool completed = probe->wait();
    session.request_stop();
    session.wait();
    ASSERT_TRUE(completed) << (session.diagnostics().empty() ? "timeout"
                                                             : session.diagnostics().front().token);
    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_TRUE(probe->valid.load());
    EXPECT_EQ(probe->calls, kSamples);
    EXPECT_EQ(probe->scratch_allocations, 0U);
    EXPECT_EQ(probe->emit_allocations, 0U);
    if (mode == ImuMode::kOwning) {
      EXPECT_GT(probe->receive_allocations, 0U);
    } else {
      EXPECT_EQ(probe->receive_allocations, 0U);
      EXPECT_EQ(probe->first_callback_allocations, 0U);
      EXPECT_EQ(probe->maximum_callback_allocations, 0U);
      EXPECT_EQ(probe->callback_allocations, 0U);
    }
  }
}

std::string imu_graph_case_name(const ::testing::TestParamInfo<PreparedImuGraph::ParamType>& info) {
  const auto [mode, transport, sinks] = info.param;
  return std::string(mode == ImuMode::kOwning   ? "Owning"
                     : mode == ImuMode::kBorrow ? "Borrow"
                                                : "Edit") +
         (transport ? "HoloMQ" : "Direct") + std::to_string(sinks);
}

INSTANTIATE_TEST_SUITE_P(PreparedPaths, PreparedImuGraph,
                         ::testing::Combine(::testing::Values(ImuMode::kOwning, ImuMode::kBorrow,
                                                              ImuMode::kEdit),
                                            ::testing::Bool(), ::testing::Values(1U, 2U)),
                         imu_graph_case_name);

}  // namespace
}  // namespace holoscan::test
