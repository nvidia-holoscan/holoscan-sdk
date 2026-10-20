/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/schema/sdk_schemas.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "common/allocation_counter.hpp"
#include "compiler/execution_plan_state.hpp"
#include "core/source_id_internal.hpp"
#include "runtime/runtime_channel_binding.hpp"
#include "runtime_multi_tensor_benchmark_traits.hpp"

namespace holoscan {
namespace {

using Multi = runtime_benchmark::MultiTensorMessageT;
using Fields = runtime_benchmark::MultiTensorMessageTensorFields;

template <typename SchemaT>
class ProjectionBenchmarkSource final : public Operator<> {
 public:
  explicit ProjectionBenchmarkSource(std::uint32_t rank) : rank_(rank) {}
  void setup(OperatorSpec& spec) override {
    auto& builder = spec.output(output, "output").max_emits_per_compute(1U);
    const TensorOutputSpec tensor{
        .representation = {.memory_kind = MemoryKind::kHost,
                           .dtype = DLDataType{kDLUInt, 8U, 1U},
                           .rank = rank_},
        .bounds = TensorBounds{.max_rank = rank_, .max_byte_span = 16U},
        .storage = std::is_same_v<SchemaT, Multi> ? TensorOutputStorage::kRuntimePool
                                                  : TensorOutputStorage::kPassThrough,
    };
    if constexpr (std::is_same_v<SchemaT, Multi>) {
      builder.tensor_fields(std::array{Fields::first, Fields::second}, tensor);
    } else {
      builder.produces_tensor(tensor);
    }
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::seconds{1}});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  Output<SchemaT> output;

 private:
  std::uint32_t rank_;
};

class ProjectionBenchmarkSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(1U).accepts_projected_tensor();
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  Input<Tensor> input;
};

// Drive the compiled Channel directly so timings include publication, readiness, receive and
// retirement, without Graph compilation, session construction, producer writes or scheduling.
template <typename SchemaT>
class ProjectionBenchmarkFixture {
 public:
  explicit ProjectionBenchmarkFixture(std::uint32_t rank) : source_(rank), rank_(rank) {}
  ~ProjectionBenchmarkFixture() {
    if (session_) {
      session_->cancel();
      (void)session_->join();
    }
  }

  expected<void, Error> initialize() {
    Graph graph;
    auto source = graph.op<ProjectionBenchmarkSource<SchemaT>>("source", rank_);
    auto sink = graph.op<ProjectionBenchmarkSink>("sink");
    if constexpr (std::is_same_v<SchemaT, Multi>) {
      graph.add_flow(source->output.project(Fields::second), sink->input);
    } else {
      graph.add_flow(source->output, sink->input);
    }
    graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
    plan_ = compile(graph);
    if (!plan_.ok()) {
      return make_unexpected(Error{ErrorCode::kFailure,
                                   plan_.diagnostics().empty()
                                       ? "projection benchmark graph rejected"
                                       : std::string{plan_.diagnostics().front().token}});
    }
    const auto retained = detail::ExecutionPlanAccess::retain(plan_);
    const auto output =
        std::ranges::find(retained->ports, PortDirection::kOutput, &PortPlanRecord::direction);
    const auto input =
        std::ranges::find(retained->ports, PortDirection::kInput, &PortPlanRecord::direction);
    if (output == retained->ports.end() || input == retained->ports.end()) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    source_owner_ = output->owner;
    sink_owner_ = input->owner;
    const std::array replay{
        detail::RuntimeChannelPortReplayRecord{
            .port = output->id,
            .owner = source_owner_,
            .direction = PortDirection::kOutput,
            .declaration = &source_.output,
            .type_ops = detail::runtime_port_type_ops<SchemaT>()},
        detail::RuntimeChannelPortReplayRecord{.port = input->id,
                                               .owner = sink_owner_,
                                               .direction = PortDirection::kInput,
                                               .declaration = &sink_.input,
                                               .type_ops = detail::runtime_port_type_ops<Tensor>()},
    };
    auto session = detail::RuntimeChannelSession::create(*retained, replay);
    if (!session) {
      return make_unexpected(std::move(session).error());
    }
    session_ = std::move(*session);
    auto binding = session_->create_readiness_binding(
        sink_owner_,
        detail::RuntimePortBindingRequest{.plan_generation = retained->plan_generation,
                                          .run_generation = 1U,
                                          .kind = detail::RuntimeEventKind::kInputArrival,
                                          .source = detail::make_source_id(1U),
                                          .source_clock = ClockId{1U},
                                          .reference_clock = ClockId{1U},
                                          .capacity = 1U});
    if (!binding) {
      return make_unexpected(std::move(binding).error());
    }
    binding_ = std::move(*binding);
    std::vector<std::int64_t> shape(rank_, 1);
    for (std::size_t index = 0U; index < backing_.size(); ++index) {
      shape.front() = index == 0U ? 8 : 16;
      auto tensor = make_tensor(shape,
                                DLDataType{kDLUInt, 8U, 1U},
                                std::nullopt,
                                MemoryStorageType::kSystem,
                                backing_[index].data(),
                                [](void*) -> expected<void, Error> { return {}; });
      if (!tensor) {
        return make_unexpected(std::move(tensor).error());
      }
      if constexpr (std::is_same_v<SchemaT, Multi>) {
        (index == 0U ? value_.first : value_.second) = std::make_shared<Tensor>(std::move(*tensor));
      } else if (index == 1U) {
        value_.data = std::make_shared<Tensor>(std::move(*tensor));
      }
    }
    return session_->arm();
  }

  expected<void, Error> round_trip() {
    auto result = session_->prepare_activation(source_owner_, generation_, {});
    if (!result)
      return result;
    result = source_.output.emit(value_);
    if (!result)
      return result;
    result = session_->retire_activation(source_owner_, generation_, true);
    if (!result)
      return result;
    result = session_->flush_process_local_activation_cut();
    if (!result)
      return result;
    result = binding_->poll(TimePoint{static_cast<std::int64_t>(generation_), ClockId{1U}});
    if (!result)
      return result;
    detail::RuntimeEvent event;
    auto drained = binding_->drain(std::span{&event, 1U});
    if (!drained || *drained != 1U) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
    auto record = session_->temporal_sample(event.port, event.source_sequence);
    if (!record)
      return make_unexpected(std::move(record).error());
    result = session_->retire_samples(event.port, std::span{&*record, 1U});
    if (!result)
      return result;
    const std::array selection{detail::RuntimeChannelActivationSelection{
        .port = event.port,
        .selection = {InputState::kSelected, SelectionReason::kNone},
        .samples = std::span{&*record, 1U},
        .destructive = true}};
    result = session_->prepare_activation(sink_owner_, generation_, selection);
    if (!result)
      return result;
    {
      auto sample = sink_.input.receive();
      if (!sample || sample->data.data() != backing_[1].data() ||
          sample->metadata.byte_size != 16U) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      benchmark::DoNotOptimize(sample->data);
    }
    result = binding_->retire(event.source_sequence);
    if (!result)
      return result;
    result = session_->retire_activation(sink_owner_, generation_++, true);
    if (!result)
      return result;
    return session_->flush_process_local_activation_cut();
  }

 private:
  ProjectionBenchmarkSource<SchemaT> source_;
  ProjectionBenchmarkSink sink_;
  std::uint32_t rank_;
  std::array<std::array<std::uint8_t, 16U>, 2U> backing_{};
  SchemaT value_;
  ExecutionPlan plan_;
  std::unique_ptr<detail::RuntimeChannelSession> session_;
  std::unique_ptr<detail::PortBinding> binding_;
  VertexId source_owner_{};
  VertexId sink_owner_{};
  std::uint64_t generation_{1U};
};

template <typename SchemaT>
void BM_ProjectedTensor(benchmark::State& state) {
  ProjectionBenchmarkFixture<SchemaT> fixture{static_cast<std::uint32_t>(state.range(0))};
  auto initialized = fixture.initialize();
  if (!initialized) {
    state.SkipWithError(initialized.error().message.c_str());
    return;
  }
  // Warm the complete publication/retirement path before counting steady-state allocations.
  auto warmup = fixture.round_trip();
  if (!warmup) {
    state.SkipWithError("projection warmup failed");
    return;
  }
  std::uint64_t allocations = 0U;
  for (auto _ : state) {
    const test::ScopedAllocationCounter counter;
    auto result = fixture.round_trip();
    allocations += counter.count();
    if (!result) {
      state.SkipWithError("projection round trip failed");
      break;
    }
  }
  state.counters["allocations_per_sample"] =
      static_cast<double>(allocations) / static_cast<double>(state.iterations());
  state.counters["tensor_fields"] = std::is_same_v<SchemaT, Multi> ? 2.0 : 1.0;
  state.counters["rank"] = static_cast<double>(state.range(0));
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_TEMPLATE(BM_ProjectedTensor, schema::ImageT)->Arg(1)->Arg(4)->Arg(8);
BENCHMARK_TEMPLATE(BM_ProjectedTensor, Multi)->Arg(1)->Arg(4)->Arg(8);

}  // namespace
}  // namespace holoscan
