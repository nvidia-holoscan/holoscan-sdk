/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

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
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/sensor_io/image_descriptor_scratch.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "common/allocation_counter.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "support/retirement_ordered_acknowledgment.hpp"

namespace holoscan::test::borrowed_descriptor {

using namespace std::chrono_literals;
enum class Mode { kOwning, kBorrow, kEdit };
inline constexpr std::array<std::int64_t, 1> kShape{1};

inline std::int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct Probe {
  explicit Probe(std::size_t count, std::size_t consumers = 1)
      : limit(count), sinks(consumers), latency_ns(count) {}
  const std::size_t limit;
  const std::size_t sinks;
  bool count_allocations{true};
  std::array<std::atomic<std::uint64_t>, 2> acknowledged{};
  std::vector<std::int64_t> latency_ns;
  std::atomic<std::int64_t> sent_at{};
  std::atomic<bool> valid{true};
  std::mutex mutex;
  std::condition_variable ready;
  std::shared_ptr<AcknowledgmentRetirementGate> acknowledgment_gate;
  // Written only on the transform lane; inspected after session.wait().
  std::size_t receive_allocations{};
  std::size_t metadata_allocations{};
  std::size_t emit_allocations{};
  std::size_t callback_allocations{};
  std::size_t calls{};
  std::int64_t callback_ns{};

  bool complete() const {
    for (std::size_t sink = 0; sink < sinks; ++sink) {
      if (acknowledged[sink].load(std::memory_order_acquire) < limit)
        return false;
    }
    return true;
  }
  bool wait() {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [&] { return complete(); });
  }
};

class Counter {
 public:
  explicit Counter(bool enabled) {
    if (enabled)
      counter_.emplace();
  }
  std::size_t count() const { return counter_ ? counter_->count() : 0; }

 private:
  std::optional<ScopedAllocationCounter> counter_;
};

inline void declare_input(OperatorSpec& spec, Input<schema::ImageT>& input) {
  spec.input(input).queue_depth(1).expects_tensor(TensorInputSpec{
      .representation = {
          .memory_kind = MemoryKind::kHost, .dtype = DLDataType{kDLUInt, 64, 1}, .rank = 1}});
}

inline void declare_output(OperatorSpec& spec, Output<schema::ImageT>& output) {
  spec.output(output).max_emits_per_compute(1).produces_tensor(
      TensorOutputSpec{.representation = {.memory_kind = MemoryKind::kHost,
                                          .dtype = DLDataType{kDLUInt, 64, 1},
                                          .rank = 1},
                       .bounds = tensor_bounds(8),
                       .storage = TensorOutputStorage::kRuntimePool});
}

inline schema::ImageT descriptor() {
  schema::ImageT image;
  image.width = 1;
  image.height = 1;
  image.header = std::make_shared<schema::HeaderT>();
  image.header->stamp = std::make_shared<schema::TimeT>();
  image.header->frame_id.assign(128, 'c');
  image.header->stamp->time_ns = 123;
  image.gain = 2.5F;
  return image;
}

class Source final : public Operator<> {
 public:
  explicit Source(std::shared_ptr<Probe> probe) : probe_(std::move(probe)) {}
  void setup(OperatorSpec& spec) override { declare_output(spec, output); }
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1us});
    return result;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    if (sent_ == probe_->limit)
      return {};
    for (std::size_t sink = 0; sink < probe_->sinks; ++sink) {
      if (probe_->acknowledged[sink].load(std::memory_order_acquire) < sent_)
        return {};
    }
    auto loan = output.allocate_tensor({.shape = kShape, .dtype = {kDLUInt, 64, 1}});
    if (!loan) {
      if (loan.error().code == ErrorCode::kResourceExhausted ||
          loan.error().code == ErrorCode::kBackpressured)
        return {};
      return make_unexpected(std::move(loan).error());
    }
    HOLOSCAN_TRY(loan->write_host_and_commit([&](const TensorOutputWriteGuard& writer) {
      *writer.data_as<std::uint64_t>() = sent_ + 1;
    }));
    image_.header->device_sequence = sent_ + 1;
    probe_->sent_at.store(now_ns(), std::memory_order_release);
    auto emitted = output.emit_tensor(std::move(*loan),
                                      image_,
                                      EmitOptions{.capture_time = context.activation_time(),
                                                  .frame_id = sent_ + 1,
                                                  .trace = TraceContext::root(77)});
    // Source-input retirement can still overlap the later sink acknowledgment.
    if (!emitted && emitted.error().code == ErrorCode::kBackpressured)
      return {};
    HOLOSCAN_TRY(std::move(emitted));
    ++sent_;
    return {};
  }
  Output<schema::ImageT> output{"output"};

 private:
  std::shared_ptr<Probe> probe_;
  schema::ImageT image_{descriptor()};
  std::uint64_t sent_{};
};

class Transform final : public Operator<> {
 public:
  Transform(std::shared_ptr<Probe> probe, Mode mode, bool failures)
      : probe_(std::move(probe)), mode_(mode), failures_(failures) {
    oversized_.header->frame_id.resize(
        schema::flatbuffer_descriptor_metadata_capacity_v<schema::Image> + 1);
  }
  void setup(OperatorSpec& spec) override {
    declare_input(spec, input);
    declare_output(spec, output);
  }
  void start() override {
    auto scratch = sensor_io::ImageDescriptorScratch::create();
    if (!scratch)
      throw RuntimeError(scratch.error());
    scratch_.emplace(std::move(*scratch));
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override {
    const auto started = now_ns();
    const Counter whole{probe_->count_allocations};
    const auto result = transform();
    probe_->callback_allocations += whole.count();
    probe_->callback_ns += now_ns() - started;
    ++probe_->calls;
    return result;
  }
  Input<schema::ImageT> input{"input"};
  Output<schema::ImageT> output{"output"};

 private:
  expected<void, Error> transform() {
    std::optional<Sample<schema::ImageT>> owning;
    std::optional<SelectedSampleView<schema::ImageT>> view;
    const Sample<schema::ImageT>* sample{};
    {
      const Counter counter{probe_->count_allocations};
      if (mode_ == Mode::kOwning) {
        HOLOSCAN_TRY(auto received, input.receive());
        owning.emplace(std::move(received));
        sample = &*owning;
      } else {
        HOLOSCAN_TRY(auto received, input.selected_sample());
        view.emplace(received);
        sample = &view->sample();
      }
      probe_->receive_allocations += counter.count();
    }
    const auto& image = sample->data;
    if (!image.data || !image.header || image.header->frame_id[0] != 'c') {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    const schema::ImageT* published = &image;
    if (mode_ == Mode::kEdit) {
      const Counter counter{probe_->count_allocations};
      HOLOSCAN_TRY(scratch_->copy_metadata_from(image));
      if (failures_) {
        const auto rejected = scratch_->copy_metadata_from(oversized_);
        if (rejected || rejected.error().code != ErrorCode::kResourceExhausted ||
            scratch_->image().width != image.width) {
          return make_unexpected(Error{ErrorCode::kFailure});
        }
      }
      scratch_->image().roi_offset_y = 5;
      scratch_->image().header->frame_id[0] = 'e';
      published = &scratch_->image();
      probe_->metadata_allocations += counter.count();
    }
    // The input Tensor is read directly; persistent scratch must not contain it.
    if (image.roi_offset_y != 0 || image.header->frame_id[0] != 'c' || scratch_->image().data)
      return make_unexpected(Error{ErrorCode::kFailure});
    const auto publish = [&](bool invalid) -> expected<void, Error> {
      HOLOSCAN_TRY(auto loan, output.allocate_tensor({.shape = kShape, .dtype = {kDLUInt, 64, 1}}));
      HOLOSCAN_TRY(loan.write_host_and_commit([&](const TensorOutputWriteGuard& writer) {
        if (writer.data() == image.data->data())
          probe_->valid.store(false);
        *writer.data_as<std::uint64_t>() = *image.data->data_as<std::uint64_t>() + 100;
      }));
      if (invalid) {
        schema::ImageT bad;
        bad.width = -1;
        return output.emit_tensor(std::move(loan), bad, sensor_io::derive_from(sample->metadata));
      }
      const Counter counter{probe_->count_allocations};
      auto emitted =
          output.emit_tensor(std::move(loan), *published, sensor_io::derive_from(sample->metadata));
      probe_->emit_allocations += counter.count();
      return emitted;
    };
    if (failures_) {
      const auto rejected = publish(true);
      if (rejected || rejected.error().code != ErrorCode::kInvalidArgument) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
    }
    return publish(false);
  }
  std::shared_ptr<Probe> probe_;
  Mode mode_;
  bool failures_;
  std::optional<sensor_io::ImageDescriptorScratch> scratch_;
  schema::ImageT oversized_{descriptor()};
};

class Sink final : public Operator<> {
 public:
  Sink(std::shared_ptr<Probe> probe, Mode mode, std::size_t index)
      : probe_(std::move(probe)), mode_(mode), index_(index) {}
  void setup(OperatorSpec& spec) override {
    declare_input(spec, input);
    spec.output(acknowledgment).max_emits_per_compute(1);
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override {
    HOLOSCAN_TRY(auto view, input.selected_sample());
    const auto& image = view.data();
    const auto sequence = probe_->acknowledged[index_].load(std::memory_order_relaxed) + 1;
    const bool edited = mode_ == Mode::kEdit;
    if (!image.data || !image.header || !image.header->stamp ||
        *image.data->data_as<std::uint64_t>() != sequence + 100 ||
        image.header->device_sequence != sequence || image.header->stamp->time_ns != 123 ||
        image.header->frame_id.size() != 128 || image.header->frame_id[0] != (edited ? 'e' : 'c') ||
        image.roi_offset_y != (edited ? 5 : 0) || image.gain != 2.5F ||
        view.metadata().frame_id != sequence || view.metadata().trace_id != 77 ||
        view.metadata().sequence != sequence || !view.metadata().source_clock_id.valid()) {
      probe_->valid.store(false);
    }
    if (index_ == 0 && sequence <= probe_->limit) {
      probe_->latency_ns[sequence - 1] = now_ns() - probe_->sent_at.load(std::memory_order_acquire);
    }
    // Publication alone is not source credit. The same-partition acknowledgment operator runs
    // only after this callback and its selected Image have retired.
    HOLOSCAN_TRY(acknowledgment.emit(sequence));
    if (index_ == 0 && sequence == 1 && probe_->acknowledgment_gate)
      HOLOSCAN_TRY(probe_->acknowledgment_gate->hold());
    return {};
  }
  Input<schema::ImageT> input{"input"};
  Output<std::uint64_t> acknowledgment{"acknowledgment"};

 private:
  std::shared_ptr<Probe> probe_;
  Mode mode_;
  std::size_t index_;
};

inline ExecutionPlan make_plan(const std::shared_ptr<Probe>& probe, Mode mode, bool force_transport,
                               bool failures = false) {
  Graph graph{"borrowed-descriptor-transform"};
  auto source = graph.op<Source>("source", probe);
  auto transform = graph.op<Transform>("transform", probe, mode, failures);
  auto sink = graph.op<Sink>("sink", probe, mode, std::size_t{0});
  auto acknowledgment =
      graph.op<RetirementOrderedAcknowledgment<Probe>>("sink_ack", probe, std::size_t{0});
  std::vector<
      std::pair<OperatorHandle<Sink>, OperatorHandle<RetirementOrderedAcknowledgment<Probe>>>>
      acknowledgment_pairs{{sink, acknowledgment}};
  graph.add_flow(source->output, transform->input, ConnectionOptions{.queue_depth = 1});
  graph.add_flow(transform->output, sink->input, ConnectionOptions{.queue_depth = 1});
  graph.add_flow(sink->acknowledgment, acknowledgment->input, ConnectionOptions{.queue_depth = 1});
  graph.partition("source").add(source);
  graph.partition("transform").add(transform);
  graph.partition("sink").add(sink).add(acknowledgment);
  if (probe->sinks == 2) {
    auto sibling = graph.op<Sink>("sibling", probe, mode, std::size_t{1});
    auto sibling_ack =
        graph.op<RetirementOrderedAcknowledgment<Probe>>("sibling_ack", probe, std::size_t{1});
    graph.add_flow(transform->output, sibling->input, ConnectionOptions{.queue_depth = 1});
    graph.add_flow(
        sibling->acknowledgment, sibling_ack->input, ConnectionOptions{.queue_depth = 1});
    graph.partition("sibling").add(sibling).add(sibling_ack);
    acknowledgment_pairs.emplace_back(sibling, sibling_ack);
  }
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  const auto plan =
      force_transport ? detail::compile_with_process_local_holomq(graph, {}) : compile(graph);
  if (plan.ok()) {
    const auto partitions = plan.partitions();
    for (const auto& [data_sink, ack] : acknowledgment_pairs)
      require_retirement_ordered_acknowledgment(partitions.of(data_sink), partitions.of(ack));
  }
  return plan;
}

}  // namespace holoscan::test::borrowed_descriptor
