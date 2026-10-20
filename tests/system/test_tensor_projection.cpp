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
#include <cstdlib>
#include <cstring>
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
#include <poll.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/schema/sdk_schemas.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "../support/optional_assertions.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "descriptor_frame_multi_order_traits.hpp"
#include "large_frame_traits.hpp"
#include "runtime/runtime_tensor_resources.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime/runtime_worker.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;
using Multi = app3p::MultiDescriptorFrameT;
using Fields = app3p::MultiDescriptorFrameTensorFields;
constexpr std::array kFields{Fields::first, Fields::second, Fields::third};
constexpr std::array<std::int64_t, 3U> kSizes{8, 16, 4};

struct ProjectionProbe {
  std::mutex mutex;
  std::condition_variable cv;
  std::optional<Sample<Tensor>> received;
  std::array<const void*, 3U> pointers{};
  std::size_t released{};
  bool started{};
};

[[nodiscard]] TensorRepresentation representation(MemoryKind memory) {
  return {.memory_kind = memory, .dtype = DLDataType{kDLUInt, 8U, 1U}, .rank = 1U};
}

[[nodiscard]] EmitOptions identity() {
  return {.capture_time = TimePoint{.timestamp_ns = 1234, .clock_id = ClockId{1U}},
          .frame_id = 9U,
          .trace = TraceContext::root(0xCAFEU),
          .flags = EmitFlags::kDegraded};
}

// A generated descriptor with unequal field sizes exposes both aggregate-byte and storage
// narrowing bugs. Select the second field while independently changing another field's memory.
class MultiProjectionSource final : public Operator<> {
 public:
  MultiProjectionSource(std::shared_ptr<ProjectionProbe> probe, int cuda_field,
                        bool transport = false, std::shared_ptr<std::atomic<bool>> gate = {},
                        std::shared_ptr<std::atomic<std::uint64_t>> publication_limit = {})
      : probe_(std::move(probe)),
        cuda_field_(cuda_field),
        transport_(transport),
        gate_(std::move(gate)),
        publication_limit_(std::move(publication_limit)) {}

  void setup(OperatorSpec& spec) override {
    auto& builder = spec.output(output, "output").max_emits_per_compute(1U);
    for (std::size_t i = 0; i < kFields.size(); ++i) {
      builder.tensor_field(
          kFields[i],
          TensorOutputSpec{
              .representation = representation(
                  static_cast<int>(i) == cuda_field_ ? MemoryKind::kCudaDevice : MemoryKind::kHost),
              .bounds = tensor_bounds(static_cast<std::uint64_t>(kSizes[i])),
              .storage = TensorOutputStorage::kRuntimePool,
              .transfer =
                  transport_ ? std::optional{TensorTransferPolicy{
                                   .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded}}
                             : std::nullopt,
          });
    }
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  expected<void, Error> compute(ExecutionContext& context) override {
    const auto limit =
        publication_limit_ ? publication_limit_->load(std::memory_order_acquire) : 1U;
    if (emitted_ >= limit || (gate_ && !gate_->load(std::memory_order_acquire))) {
      return {};
    }
    std::array<TensorFieldLoanRequest<Multi>, 3U> requests{};
    for (std::size_t i = 0; i < requests.size(); ++i) {
      requests[i] = {
          .field = kFields[i],
          .tensor = {.shape = std::span{&kSizes[i], 1U}, .dtype = DLDataType{kDLUInt, 8U, 1U}}};
    }
    auto loan = output.allocate_tensors(requests);
    if (!loan) {
      return make_unexpected(std::move(loan).error());
    }
    for (std::size_t i = 0; i < requests.size(); ++i) {
      const auto fill = [&](const TensorOutputWriteGuard& writer) -> expected<void, Error> {
        {
          const std::lock_guard lock(probe_->mutex);
          probe_->pointers[i] = writer.data();
        }
        if (static_cast<int>(i) == cuda_field_) {
          if (cudaMemsetAsync(writer.data(),
                              static_cast<int>(i + emitted_ + 1U),
                              kSizes[i],
                              context.cuda_stream().get()) != cudaSuccess) {
            return make_unexpected(Error{ErrorCode::kFailure});
          }
        } else {
          std::memset(writer.data(), static_cast<int>(i + emitted_ + 1U), kSizes[i]);
        }
        return {};
      };
      auto written = static_cast<int>(i) == cuda_field_
                         ? loan->write_and_commit(kFields[i], context.cuda_stream(), fill)
                         : loan->write_host_and_commit(kFields[i], fill);
      if (!written) {
        return written;
      }
    }
    auto published = output.emit_tensors(std::move(*loan), Multi{}, identity());
    if (published) {
      ++emitted_;
    }
    return published;
  }
  Output<Multi> output;

 private:
  std::shared_ptr<ProjectionProbe> probe_;
  int cuda_field_;
  bool transport_;
  std::shared_ptr<std::atomic<bool>> gate_;
  std::shared_ptr<std::atomic<std::uint64_t>> publication_limit_;
  std::uint64_t emitted_{};
};

enum class Provider { kPassThrough, kLifecycle };

class SingleProjectionSource final : public Operator<> {
 public:
  SingleProjectionSource(std::shared_ptr<ProjectionProbe> probe, Provider provider)
      : probe_(std::move(probe)), provider_(provider) {}
  void setup(OperatorSpec& spec) override {
    auto& builder = spec.output(output, "output").max_emits_per_compute(1U);
    builder.produces_tensor(TensorOutputSpec{
        .representation = representation(MemoryKind::kHost),
        .bounds = tensor_bounds(16U),
        .storage = provider_ == Provider::kLifecycle ? TensorOutputStorage::kRuntimePool
                                                     : TensorOutputStorage::kPassThrough,
    });
    if (provider_ == Provider::kLifecycle) {
      builder.lifecycle_nominal_rate(LifecycleStage::kDiscover);
      spec.lifecycle().stage(LifecycleStage::kDiscover, &SingleProjectionSource::discover);
    }
  }
  LifecycleStatus discover(LifecycleContext& context) noexcept {
    return context.bind_nominal_rate(output, Rate::hz(60U)) ? LifecycleStatus::kOk
                                                            : LifecycleStatus::kFatalFailure;
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override {
    if (emitted_) {
      return {};
    }
    emitted_ = true;
    constexpr std::array<std::int64_t, 1U> shape{16};
    if (provider_ == Provider::kLifecycle) {
      auto loan = output.allocate_tensor(shape);
      if (!loan) {
        return make_unexpected(std::move(loan).error());
      }
      auto written = loan->write_host_and_commit([&](const TensorOutputWriteGuard& writer) {
        const std::lock_guard lock(probe_->mutex);
        probe_->pointers[1] = writer.data();
        std::memset(writer.data(), 2, 16U);
      });
      if (!written) {
        return written;
      }
      return output.emit_tensor(std::move(*loan), schema::ImageT{}, identity());
    }
    // Transfer flat byte storage to the Tensor callback, which reclaims it with delete[].
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
    auto storage = std::make_unique<std::uint8_t[]>(16U);
    std::fill_n(storage.get(), 16U, 2U);
    auto tensor = make_tensor(shape,
                              DLDataType{kDLUInt, 8U, 1U},
                              std::nullopt,
                              MemoryStorageType::kSystem,
                              storage.get(),
                              [probe = probe_](void* pointer) -> expected<void, Error> {
                                delete[] static_cast<std::uint8_t*>(pointer);
                                const std::lock_guard lock(probe->mutex);
                                ++probe->released;
                                return {};
                              });
    if (!tensor) {
      return make_unexpected(std::move(tensor).error());
    }
    {
      const std::lock_guard lock(probe_->mutex);
      probe_->pointers[1] = storage.release();
    }
    schema::ImageT value;
    value.width = 16U;
    value.height = 1U;
    value.encoding = schema::ImageEncoding_CUSTOM;
    value.data = std::make_shared<Tensor>(std::move(*tensor));
    return output.emit(std::move(value), identity());
  }
  Output<schema::ImageT> output;

 private:
  std::shared_ptr<ProjectionProbe> probe_;
  Provider provider_;
  bool emitted_{};
};

class ProjectionSink final : public Operator<> {
 public:
  ProjectionSink(std::shared_ptr<ProjectionProbe> probe, MemoryKind memory, std::uint32_t rank = 1U,
                 bool rate_sized = false, std::shared_ptr<std::atomic<bool>> release = {})
      : probe_(std::move(probe)),
        memory_(memory),
        rank_(rank),
        rate_sized_(rate_sized),
        release_(std::move(release)) {}
  void setup(OperatorSpec& spec) override {
    auto layout = representation(memory_);
    layout.rank = rank_;
    auto& builder = spec.input(input, "input")
                        .accepts_projected_tensor()
                        .expects_tensor(TensorInputSpec{.representation = layout});
    if (!rate_sized_) {
      builder.queue_depth(1U);
    }
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  void start() override {
    {
      const std::lock_guard lock(probe_->mutex);
      probe_->started = true;
    }
    probe_->cv.notify_all();
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    if (memory_ == MemoryKind::kCudaDevice) {
      // The host assertion must wait for the producer's work on the received Tensor's stream.
      auto ready = sample->data.synchronize_for_consumer_stream(context.cuda_stream().get());
      if (!ready) {
        return ready;
      }
      if (cudaStreamSynchronize(context.cuda_stream().get()) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
    }
    {
      const std::lock_guard lock(probe_->mutex);
      probe_->received.emplace(std::move(*sample));
    }
    probe_->cv.notify_all();
    while (release_ && !release_->load(std::memory_order_acquire) &&
           !context.cancellation().stop_requested()) {
      std::this_thread::sleep_for(1ms);
    }
    return {};
  }
  Input<Tensor> input;

 private:
  std::shared_ptr<ProjectionProbe> probe_;
  MemoryKind memory_;
  std::uint32_t rank_;
  bool rate_sized_;
  std::shared_ptr<std::atomic<bool>> release_;
};

// Ordinary schema delivery is the positive control for encoding-scratch allocation evidence.
class SchemaDeliverySink final : public Operator<> {
 public:
  explicit SchemaDeliverySink(std::shared_ptr<ProjectionProbe> probe) : probe_(std::move(probe)) {}
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{.representation = representation(MemoryKind::kHost)});
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    {
      const std::lock_guard lock(probe_->mutex);
      probe_->received.emplace(*sample->data.data, sample->metadata);
    }
    probe_->cv.notify_all();
    return {};
  }
  Input<schema::ImageT> input;

 private:
  std::shared_ptr<ProjectionProbe> probe_;
};

// Both outputs borrow their backing. The large schema's generated metadata capacity exceeds
// HoloMQ's inline limit, so its otherwise valid direct route exposes sibling transport changes.
class ProjectionSiblingSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    const TensorOutputSpec output_spec{
        .representation = representation(MemoryKind::kHost),
        .bounds = tensor_bounds(16U),
        .storage = TensorOutputStorage::kPassThrough,
    };
    spec.output(small, "small").max_emits_per_compute(1U).produces_tensor(output_spec);
    spec.output(large, "large").max_emits_per_compute(1U).produces_tensor(output_spec);
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  Output<schema::ImageT> small;
  Output<projection_fixture::LargeFrameT> large;
};

class ProjectionSiblingSink final : public Operator<> {
 public:
  explicit ProjectionSiblingSink(bool with_projection, bool with_small = false)
      : with_projection_(with_projection), with_small_(with_small) {}
  void setup(OperatorSpec& spec) override {
    const TensorInputSpec input_spec{.representation = representation(MemoryKind::kHost)};
    spec.input(large, "large").queue_depth(1U).expects_tensor(input_spec);
    if (with_small_) {
      spec.input(small, "small").queue_depth(1U).expects_tensor(input_spec);
    }
    if (with_projection_) {
      spec.input(projected, "projected")
          .queue_depth(1U)
          .accepts_projected_tensor()
          .expects_tensor(input_spec);
    }
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnAnyInput{});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  Input<projection_fixture::LargeFrameT> large;
  Input<schema::ImageT> small;
  Input<Tensor> projected;

 private:
  bool with_projection_;
  bool with_small_;
};

void set_clock(Graph& graph) {
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
}

void verify_delivery(const ExecutionPlan& plan, const std::shared_ptr<ProjectionProbe>& probe,
                     bool cuda, std::size_t pool_count, std::uint64_t encoding_allocations = 0U) {
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  RunSession session = run_async(detail::RuntimeTestPlanAccess::with_phase_counters(
      detail::RuntimeTestPlanAccess::with_tensor_output_test_control(plan, control), counters));
  std::optional<Sample<Tensor>> lease;
  {
    std::unique_lock lock(probe->mutex);
    const bool received = probe->cv.wait_for(lock, 5s, [&] { return probe->received.has_value(); });
    if (!received) {
      lock.unlock();
      session.request_stop();
      session.wait();
      FAIL() << (session.diagnostics().empty() ? "no received sample"
                                               : session.diagnostics().front().token);
    }
    auto& received_sample = probe->received;
    HOLOSCAN_ASSERT_HAS_VALUE(received_sample);
    const auto& sample = *received_sample;
    EXPECT_EQ(sample.data.data(), probe->pointers[1]);
    EXPECT_EQ(sample.metadata.byte_size, 16U);
    EXPECT_EQ(sample.metadata.sequence, 1U);
    EXPECT_EQ(sample.metadata.capture_timestamp_ns, 1234);
    EXPECT_EQ(sample.metadata.source_clock_id, ClockId{1U});
    EXPECT_EQ(sample.metadata.publish_clock_id, ClockId{1U});
    EXPECT_NE(sample.metadata.publish_timestamp_ns, 0);
    EXPECT_EQ(sample.metadata.frame_id, 9U);
    EXPECT_EQ(sample.metadata.trace_id, 0xCAFEU);
    EXPECT_EQ(sample.metadata.flags, SampleFlags::kDegraded);
    EXPECT_EQ(sample.metadata.payload_kind, PayloadKind::kDescriptorBacked);
    std::array<std::uint8_t, 16U> bytes{};
    if (cuda) {
      EXPECT_EQ(cudaMemcpy(bytes.data(), sample.data.data(), bytes.size(), cudaMemcpyDeviceToHost),
                cudaSuccess);
    } else {
      std::memcpy(bytes.data(), sample.data.data(), bytes.size());
    }
    EXPECT_TRUE(std::ranges::all_of(bytes, [](auto value) { return value == 2U; }));
    if (pool_count != 0U) {
      EXPECT_GT(control->snapshot(pool_count == 3U ? 1U : 0U).active_slot_count, 0U);
    } else {
      EXPECT_EQ(probe->released, 0U);
    }
    lease.emplace(std::move(*received_sample));
    probe->received.reset();
  }
  lease.reset();
  session.request_stop();
  session.wait();
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(counters->snapshot()[detail::RuntimePhaseCounters::index(
                detail::RuntimePhaseCounter::kChannelOutputEncodingScratchAllocations)],
            encoding_allocations);
  for (std::size_t i = 0; i < pool_count; ++i) {
    EXPECT_EQ(control->snapshot(i).active_slot_count, 0U);
    EXPECT_EQ(control->snapshot(i).invariant_failure_count, 0U);
  }
  if (pool_count == 0U) {
    const std::lock_guard lock(probe->mutex);
    EXPECT_EQ(probe->released, 1U);
  }
}

// REQ: GRAPH-045
TEST(TensorProjection, EncodingScratchAllocatedOnlyForSchemaDelivery) {
  for (const bool projected : {false, true}) {
    SCOPED_TRACE(projected);
    auto probe = std::make_shared<ProjectionProbe>();
    Graph graph;
    auto source = graph.op<SingleProjectionSource>("source", probe, Provider::kPassThrough);
    if (projected) {
      auto sink = graph.op<ProjectionSink>("sink", probe, MemoryKind::kHost);
      graph.add_flow(source->output, sink->input);
    } else {
      auto sink = graph.op<SchemaDeliverySink>("sink", probe);
      graph.add_flow(source->output, sink->input);
    }
    set_clock(graph);
    verify_delivery(compile(graph), probe, false, 0U, projected ? 0U : 1U);
  }
}

// REQ: GRAPH-045
TEST(TensorProjection, MultiFieldPreservesSelectedBackingMetadataAndRetirement) {
  // -1: host only; 0: unselected CUDA field; 1: selected CUDA field at nonzero index.
  for (const int cuda_field : {-1, 0, 1}) {
    SCOPED_TRACE(cuda_field);
    int devices{};
    if (cuda_field >= 0 && (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)) {
      continue;
    }
    auto probe = std::make_shared<ProjectionProbe>();
    Graph graph;
    auto source = graph.op<MultiProjectionSource>("source", probe, cuda_field);
    auto sink = graph.op<ProjectionSink>(
        "sink", probe, cuda_field == 1 ? MemoryKind::kCudaDevice : MemoryKind::kHost);
    graph.add_flow(source->output.project(Fields::second), sink->input);
    set_clock(graph);
    CompileOptions options;
    if (cuda_field >= 0) {
      options.deployment.bind_tensor_output_device(
          {.operator_path = "source", .output_port = "output", .device = DeviceId{0}});
    }
    verify_delivery(compile(graph, std::move(options)), probe, cuda_field == 1, 3U);
  }
}

// REQ: GRAPH-045
TEST(TensorProjection, PassThroughAndLifecycleProvidersPreserveBackingAndRetirement) {
  for (const auto provider : {Provider::kPassThrough, Provider::kLifecycle}) {
    auto probe = std::make_shared<ProjectionProbe>();
    Graph graph;
    auto source = graph.op<SingleProjectionSource>("source", probe, provider);
    auto sink = graph.op<ProjectionSink>(
        "sink", probe, MemoryKind::kHost, 1U, provider == Provider::kLifecycle);
    graph.add_flow(source->output, sink->input);
    set_clock(graph);
    const auto plan = compile(graph);
    ASSERT_TRUE(plan.ok()) << plan.json();
    EXPECT_TRUE(plan.tensor_allocations().empty());
    if (provider == Provider::kLifecycle) {
      ASSERT_EQ(plan.provider_allocation_plans().size(), 1U);
      EXPECT_EQ(plan.provider_allocation_plans().front().geometry_mode,
                AllocationGeometryMode::kLifecycleCandidate);
    }
    verify_delivery(plan, probe, false, provider == Provider::kLifecycle ? 1U : 0U);
  }
}

TEST(TensorProjection, FanoutSharesBackingUntilEveryConsumerReleasesIt) {
  for (const bool mixed : {false, true}) {
    SCOPED_TRACE(mixed);
    auto producer = std::make_shared<ProjectionProbe>();
    std::array consumers{std::make_shared<ProjectionProbe>(),
                         std::make_shared<ProjectionProbe>(),
                         std::make_shared<ProjectionProbe>()};
    Graph graph;
    auto source = graph.op<SingleProjectionSource>("source", producer, Provider::kPassThrough);
    auto first = graph.op<ProjectionSink>("first", consumers[0], MemoryKind::kHost);
    auto second = graph.op<ProjectionSink>("second", consumers[1], MemoryKind::kHost);
    graph.add_flow(source->output, first->input);
    graph.add_flow(source->output, second->input);
    if (mixed) {
      auto schema = graph.op<SchemaDeliverySink>("schema", consumers[2]);
      graph.add_flow(source->output, schema->input);
    }
    set_clock(graph);
    const auto plan = compile(graph);
    ASSERT_TRUE(plan.ok()) << plan.json();
    auto session = run_async(plan);
    for (std::size_t index = 0; index < (mixed ? 3U : 2U); ++index) {
      auto& consumer = *consumers[index];
      std::unique_lock lock(consumer.mutex);
      ASSERT_TRUE(consumer.cv.wait_for(lock, 5s, [&] { return consumer.received.has_value(); }));
      EXPECT_EQ(consumer.received->data.data(), producer->pointers[1]);
      EXPECT_EQ(consumer.received->metadata.byte_size, 16U);
      EXPECT_EQ(consumer.received->metadata.sequence, 1U);
      EXPECT_EQ(consumer.received->metadata.frame_id, 9U);
      EXPECT_EQ(consumer.received->metadata.trace_id, 0xCAFEU);
      EXPECT_EQ(consumer.received->metadata.capture_timestamp_ns, 1234);
    }
    {
      const std::lock_guard lock(consumers[0]->mutex);
      consumers[0]->received.reset();
    }
    {
      const std::lock_guard lock(producer->mutex);
      EXPECT_EQ(producer->released, 0U);
    }
    for (auto& consumer : consumers) {
      const std::lock_guard lock(consumer->mutex);
      consumer->received.reset();
    }
    session.request_stop();
    session.wait();
    EXPECT_TRUE(session.diagnostics().empty());
    const std::lock_guard lock(producer->mutex);
    EXPECT_EQ(producer->released, 1U);
  }
}

TEST(TensorProjection, FanoutSelectsIndependentFieldsAndRetiresBacking) {
  auto producer = std::make_shared<ProjectionProbe>();
  std::array consumers{std::make_shared<ProjectionProbe>(), std::make_shared<ProjectionProbe>()};
  Graph graph;
  auto source = graph.op<MultiProjectionSource>("source", producer, -1);
  auto first = graph.op<ProjectionSink>("first", consumers[0], MemoryKind::kHost);
  auto second = graph.op<ProjectionSink>("second", consumers[1], MemoryKind::kHost);
  graph.add_flow(source->output.project(Fields::first), first->input);
  graph.add_flow(source->output.project(Fields::second), second->input);
  set_clock(graph);
  const auto plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  auto session =
      run_async(detail::RuntimeTestPlanAccess::with_tensor_output_test_control(plan, control));
  for (std::size_t index = 0; index < consumers.size(); ++index) {
    auto& consumer = *consumers[index];
    std::unique_lock lock(consumer.mutex);
    ASSERT_TRUE(consumer.cv.wait_for(lock, 5s, [&] { return consumer.received.has_value(); }));
    EXPECT_EQ(consumer.received->data.data(), producer->pointers[index]);
    EXPECT_EQ(consumer.received->metadata.byte_size, kSizes[index]);
    EXPECT_EQ(*static_cast<const std::uint8_t*>(consumer.received->data.data()), index + 1U);
  }
  for (auto& consumer : consumers) {
    const std::lock_guard lock(consumer->mutex);
    consumer->received.reset();
  }
  session.request_stop();
  session.wait();
  EXPECT_TRUE(session.diagnostics().empty());
  for (std::size_t index = 0; index < kFields.size(); ++index) {
    EXPECT_EQ(control->snapshot(index).active_slot_count, 0U);
    EXPECT_EQ(control->snapshot(index).invariant_failure_count, 0U);
  }
}

// REQ: GRAPH-045
TEST(TensorProjection, RepeatedFanoutRetiresUnusedClaimsAcrossReplacementAndCancellation) {
  for (const bool cancel_with_pending : {false, true}) {
    SCOPED_TRACE(cancel_with_pending);
    constexpr std::uint64_t kPublications = 128U;
    auto producer = std::make_shared<ProjectionProbe>();
    auto fast_probe = std::make_shared<ProjectionProbe>();
    auto slow_probe = std::make_shared<ProjectionProbe>();
    auto publication_limit = std::make_shared<std::atomic<std::uint64_t>>(0U);
    auto release_slow = std::make_shared<std::atomic<bool>>(false);
    Graph graph;
    auto source = graph.op<MultiProjectionSource>(
        "source", producer, -1, false, std::shared_ptr<std::atomic<bool>>{}, publication_limit);
    auto fast = graph.op<ProjectionSink>("fast", fast_probe, MemoryKind::kHost);
    auto slow =
        graph.op<ProjectionSink>("slow", slow_probe, MemoryKind::kHost, 1U, false, release_slow);
    const ConnectionOptions connection{.queue_policy = QueuePolicy::kDropOldest};
    graph.add_flow(source->output.project(Fields::first), fast->input, connection);
    graph.add_flow(source->output.project(Fields::second), slow->input, connection);
    graph.partition("source-partition").add(source);
    graph.partition("fast-partition").add(fast);
    graph.partition("slow-partition").add(slow);
    set_clock(graph);
    const auto plan = compile(graph);
    ASSERT_TRUE(plan.ok()) << plan.json();
    const auto* state = detail::ExecutionPlanAccess::state(plan);
    ASSERT_NE(state, nullptr);
    ASSERT_EQ(state->process_local_channel_services.size(), 1U);
    EXPECT_EQ(state->process_local_channel_services.front().delivery_kind,
              detail::ProcessLocalChannelDeliveryKind::kHoloMQ);
    ASSERT_EQ(plan.tensor_field_allocations().size(), kFields.size());
    for (const auto& allocation : plan.tensor_field_allocations()) {
      ASSERT_LT(allocation.slot_count, kPublications);
    }
    auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
    auto session =
        run_async(detail::RuntimeTestPlanAccess::with_tensor_output_test_control(plan, control));
    const auto wait_for_sequence = [](ProjectionProbe& probe, std::uint64_t sequence) {
      std::unique_lock lock(probe.mutex);
      return probe.cv.wait_for(lock, 5s, [&] {
        return probe.received && probe.received->metadata.sequence == sequence;
      });
    };
    publication_limit->store(1U, std::memory_order_release);
    ASSERT_TRUE(wait_for_sequence(*slow_probe, 1U));
    ASSERT_TRUE(wait_for_sequence(*fast_probe, 1U));
    {
      const std::lock_guard lock(fast_probe->mutex);
      fast_probe->received.reset();
    }
    // The slow callback pins its first sample while its depth-one queue repeatedly replaces
    // newer samples. The third field has no consumer, so every publication must retire its claims.
    for (std::uint64_t sequence = 2U; sequence <= kPublications; ++sequence) {
      publication_limit->store(sequence, std::memory_order_release);
      ASSERT_TRUE(wait_for_sequence(*fast_probe, sequence));
      {
        const std::lock_guard lock(fast_probe->mutex);
        EXPECT_EQ(*static_cast<const std::uint8_t*>(fast_probe->received->data.data()), sequence);
        // This input admits one borrowed sample; release it before receiving its successor.
        fast_probe->received.reset();
      }
      {
        const std::lock_guard lock(slow_probe->mutex);
        ASSERT_TRUE(slow_probe->received);
        EXPECT_EQ(slow_probe->received->metadata.sequence, 1U);
        EXPECT_EQ(*static_cast<const std::uint8_t*>(slow_probe->received->data.data()), 2U);
      }
    }
    if (!cancel_with_pending) {
      {
        const std::lock_guard lock(slow_probe->mutex);
        slow_probe->received.reset();
      }
      release_slow->store(true, std::memory_order_release);
      ASSERT_TRUE(wait_for_sequence(*slow_probe, kPublications));
      const std::lock_guard lock(slow_probe->mutex);
      EXPECT_EQ(*static_cast<const std::uint8_t*>(slow_probe->received->data.data()),
                kPublications + 1U);
    }
    session.request_stop();
    session.wait();
    for (const auto& probe : {fast_probe, slow_probe}) {
      const std::lock_guard lock(probe->mutex);
      probe->received.reset();
    }
    EXPECT_TRUE(session.diagnostics().empty());
    for (std::size_t index = 0U; index < kFields.size(); ++index) {
      const auto snapshot = control->snapshot(index);
      EXPECT_EQ(snapshot.active_slot_count, 0U);
      EXPECT_EQ(snapshot.invariant_failure_count, 0U);
      EXPECT_EQ(snapshot.metrics.failed_loan_count, 0U);
    }
  }
}

#if defined(__linux__)
// Exec a fresh test process: a consumer must reconstruct its codec inventory without creating
// the producer at Runtime, and CUDA state must never be used across an un-execed fork.
class ProjectionConsumerProcess {
 public:
  ProjectionConsumerProcess(std::size_t index, const std::string& instance) {
    std::array<int, 2U> descriptors{-1, -1};
    if (pipe(descriptors.data()) != 0) {
      return;
    }
    read_descriptor_ = descriptors[0];
    std::vector<std::string> environment;
    for (char** entry = environ; *entry != nullptr; ++entry) {
      environment.emplace_back(*entry);
    }
    environment.push_back("HOLOSCAN_PROJECTION_TEST_CHILD=" + std::to_string(index));
    environment.push_back("HOLOSCAN_PROJECTION_TEST_PIPE=" + std::to_string(descriptors[1]));
    environment.push_back("HOLOSCAN_PROJECTION_TEST_INSTANCE=" + instance);
    std::vector<char*> environment_pointers;
    for (auto& entry : environment) {
      environment_pointers.push_back(entry.data());
    }
    environment_pointers.push_back(nullptr);
    const std::string filter = std::string("--gtest_filter=TensorProjection.") +
                               ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::array<std::string, 2U> arguments{"tensor_projection_test", filter};
    std::array<char*, 3U> argv{arguments[0].data(), arguments[1].data(), nullptr};
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addclose(&actions, read_descriptor_);
    const int result = posix_spawn(
        &child_, "/proc/self/exe", &actions, nullptr, argv.data(), environment_pointers.data());
    posix_spawn_file_actions_destroy(&actions);
    close(descriptors[1]);
    if (result != 0) {
      child_ = -1;
    }
  }
  ProjectionConsumerProcess(const ProjectionConsumerProcess&) = delete;
  ProjectionConsumerProcess& operator=(const ProjectionConsumerProcess&) = delete;
  ~ProjectionConsumerProcess() {
    if (child_ > 0) {
      kill(child_, SIGKILL);
      int status{};
      static_cast<void>(waitpid(child_, &status, 0));
    }
    if (read_descriptor_ >= 0) {
      close(read_descriptor_);
    }
  }
  bool read_success() const {
    if (child_ <= 0) {
      return false;
    }
    pollfd descriptor{.fd = read_descriptor_, .events = POLLIN};
    char result{};
    return poll(&descriptor, 1U, 10000) > 0 &&
           read(read_descriptor_, &result, sizeof(result)) == sizeof(result) && result == 1;
  }
  bool wait_success() {
    const auto deadline = std::chrono::steady_clock::now() + 10s;
    while (child_ > 0 && std::chrono::steady_clock::now() < deadline) {
      int status{};
      if (waitpid(child_, &status, WNOHANG) == child_) {
        child_ = -1;
        return WIFEXITED(status) && WEXITSTATUS(status) == 0;
      }
      std::this_thread::sleep_for(1ms);
    }
    return false;
  }

 private:
  pid_t child_{-1};
  int read_descriptor_{-1};
};

void verify_cross_process_projection(int cuda_field) {
  const char* child = std::getenv("HOLOSCAN_PROJECTION_TEST_CHILD");
  const char* inherited_instance = std::getenv("HOLOSCAN_PROJECTION_TEST_INSTANCE");
  const std::string instance =
      inherited_instance != nullptr ? inherited_instance : std::to_string(getpid());
  auto producer = std::make_shared<ProjectionProbe>();
  auto local_consumer = std::make_shared<ProjectionProbe>();
  std::array consumers{std::make_shared<ProjectionProbe>(), std::make_shared<ProjectionProbe>()};
  auto gate = std::make_shared<std::atomic<bool>>(false);
  Graph graph{"projection-ipc-" + instance};
  auto source = graph.op<MultiProjectionSource>("source", producer, cuda_field, true, gate);
  auto first = graph.op<ProjectionSink>(
      "first", consumers[0], cuda_field == 0 ? MemoryKind::kCudaDevice : MemoryKind::kHost);
  auto second = graph.op<ProjectionSink>("second", consumers[1], MemoryKind::kHost);
  auto local = graph.op<ProjectionSink>("local", local_consumer, MemoryKind::kHost);
  graph.add_flow(source->output.project(Fields::first), first->input);
  graph.add_flow(source->output.project(Fields::second), second->input);
  graph.add_flow(source->output.project(Fields::second), local->input);
  set_clock(graph);
  detail::ProcessPlacementBinding placement;
  const std::array source_selector{detail::GraphObjectSelector{.graph_path = "source"},
                                   detail::GraphObjectSelector{.graph_path = "local"}};
  const std::array first_selector{detail::GraphObjectSelector{.graph_path = "first"}};
  const std::array second_selector{detail::GraphObjectSelector{.graph_path = "second"}};
  placement.bind_process_role(
      "producer", detail::ProcessPlacement{.members = source_selector, .launch_group = "producer"});
  placement.bind_process_role(
      "first", detail::ProcessPlacement{.members = first_selector, .launch_group = "first"});
  placement.bind_process_role(
      "second", detail::ProcessPlacement{.members = second_selector, .launch_group = "second"});
  CompileOptions options;
  if (cuda_field >= 0) {
    options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
        .operator_path = "source", .output_port = "output", .device = DeviceId{0}});
  }
  const auto plan = detail::compile_with_process_placement(graph, options, std::move(placement));
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->cross_process_channel_services.size(), 1U);
  const auto& service = state->cross_process_channel_services.front();
  ASSERT_EQ(service.target_process_groups.size(), 2U);
  ASSERT_EQ(state->cross_process_channel_subscribers.size(), 2U);
  for (const auto& subscriber : state->cross_process_channel_subscribers) {
    ASSERT_TRUE(subscriber.tensor_projection);
  }

  if (child != nullptr) {
    const auto index = static_cast<std::size_t>(std::strtoul(child, nullptr, 10));
    ASSERT_LT(index, consumers.size());
    const char* pipe = std::getenv("HOLOSCAN_PROJECTION_TEST_PIPE");
    ASSERT_NE(pipe, nullptr);
    const int descriptor = std::atoi(pipe);
    auto session = detail::run_async_worker(plan, service.target_process_groups[index]);
    auto& consumer = *consumers[index];
    {
      std::unique_lock lock(consumer.mutex);
      ASSERT_TRUE(consumer.cv.wait_for(lock, 5s, [&] { return consumer.started; }));
      const char ready = 1;
      ASSERT_EQ(write(descriptor, &ready, sizeof(ready)), sizeof(ready));
      ASSERT_TRUE(consumer.cv.wait_for(lock, 5s, [&] { return consumer.received.has_value(); }));
      EXPECT_EQ(consumer.received->metadata.byte_size, kSizes[index]);
      EXPECT_EQ(consumer.received->metadata.sequence, 1U);
      EXPECT_EQ(consumer.received->metadata.frame_id, 9U);
      EXPECT_EQ(consumer.received->metadata.trace_id, 0xCAFEU);
      EXPECT_EQ(consumer.received->metadata.capture_timestamp_ns, 1234);
      std::array<std::uint8_t, 16U> contents{};
      if (static_cast<int>(index) == cuda_field) {
        ASSERT_EQ(cudaMemcpy(contents.data(),
                             consumer.received->data.data(),
                             kSizes[index],
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
      } else {
        std::memcpy(contents.data(), consumer.received->data.data(), kSizes[index]);
      }
      EXPECT_TRUE(std::all_of(contents.begin(), contents.begin() + kSizes[index], [&](auto value) {
        return value == index + 1U;
      }));
      consumer.received.reset();
    }
    session.request_stop();
    session.wait();
    EXPECT_TRUE(session.diagnostics().empty());
    EXPECT_EQ(producer->pointers, (std::array<const void*, 3U>{}));
    EXPECT_FALSE(local_consumer->started);
    const char result = ::testing::Test::HasFailure() ? 0 : 1;
    ASSERT_EQ(write(descriptor, &result, sizeof(result)), sizeof(result));
    close(descriptor);
    return;
  }

  ProjectionConsumerProcess first_worker{0U, instance};
  ProjectionConsumerProcess second_worker{1U, instance};
  ASSERT_TRUE(first_worker.read_success());
  ASSERT_TRUE(second_worker.read_success());
  auto session = detail::run_async_worker(plan, service.source_process_group);
  // Let the newly created publisher discover both already-running consumers before emitting.
  std::this_thread::sleep_for(300ms);
  gate->store(true, std::memory_order_release);
  EXPECT_TRUE(first_worker.read_success());
  EXPECT_TRUE(second_worker.read_success());
  EXPECT_TRUE(first_worker.wait_success());
  EXPECT_TRUE(second_worker.wait_success());
  {
    std::unique_lock lock(local_consumer->mutex);
    ASSERT_TRUE(local_consumer->cv.wait_for(
        lock, 5s, [&] { return local_consumer->received.has_value(); }));
    EXPECT_EQ(local_consumer->received->data.data(), producer->pointers[1]);
    EXPECT_EQ(local_consumer->received->metadata.byte_size, 16U);
    local_consumer->received.reset();
  }
  session.request_stop();
  session.wait();
  EXPECT_TRUE(session.diagnostics().empty());
  for (const auto& consumer : consumers) {
    EXPECT_FALSE(consumer->started);
    EXPECT_FALSE(consumer->received);
  }
}

TEST(TensorProjection, CrossProcessFanoutSelectsFieldsInIsolatedWorkers) {
  verify_cross_process_projection(-1);
}

TEST(TensorProjection, CrossProcessFanoutImportsCUDAOnlyForSelectedField) {
  int devices{};
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }
  verify_cross_process_projection(0);
}
#endif

// REQ: GRAPH-045
// REQ: COMPILE-001
TEST(TensorProjection, QueuedDeliveryAdmitsFanoutAndForcedTransport) {
  for (const std::string_view route :
       {"fanout", "multi-fanout", "lifecycle-fanout", "mixed", "forced", "forced-drop-oldest"}) {
    SCOPED_TRACE(route);
    auto producer = std::make_shared<ProjectionProbe>();
    auto probe = std::make_shared<ProjectionProbe>();
    auto sibling_probe = std::make_shared<ProjectionProbe>();
    Graph graph;
    auto sink =
        graph.op<ProjectionSink>("sink", probe, MemoryKind::kHost, 1U, route == "lifecycle-fanout");
    ConnectionOptions connection;
    if (route == "forced-drop-oldest") {
      connection.queue_policy = QueuePolicy::kDropOldest;
    }
    if (route == "multi-fanout") {
      auto source = graph.op<MultiProjectionSource>("source", producer, -1);
      auto sibling = graph.op<ProjectionSink>("sibling", sibling_probe, MemoryKind::kHost);
      graph.add_flow(source->output.project(Fields::second), sink->input);
      graph.add_flow(source->output.project(Fields::first), sibling->input);
    } else {
      auto source = graph.op<SingleProjectionSource>(
          "source",
          producer,
          route == "lifecycle-fanout" ? Provider::kLifecycle : Provider::kPassThrough);
      graph.add_flow(source->output, sink->input, connection);
      if (route.ends_with("fanout")) {
        auto sibling = graph.op<ProjectionSink>(
            "sibling", sibling_probe, MemoryKind::kHost, 1U, route == "lifecycle-fanout");
        graph.add_flow(source->output, sibling->input);
      } else if (route == "mixed") {
        auto sibling = graph.op<SchemaDeliverySink>("sibling", sibling_probe);
        graph.add_flow(source->output, sibling->input);
      }
    }
    set_clock(graph);
    const auto compile_route = [&] {
      if (route.starts_with("forced")) {
        return detail::compile_with_process_local_holomq(graph, {});
      }
      return compile(graph);
    };
    ExecutionPlan plan;
    ASSERT_NO_THROW(plan = compile_route());
    if (route == "lifecycle-fanout") {
      EXPECT_FALSE(plan.ok());
      EXPECT_TRUE(std::ranges::any_of(plan.diagnostics(), [](const auto& diagnostic) {
        return diagnostic.token == "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE";
      }));
      EXPECT_THROW((void)run_async(plan), CompileError);
      continue;
    }
    ASSERT_TRUE(plan.ok()) << plan.json();
    EXPECT_EQ(plan.graph().edges().size(), route.ends_with("fanout") || route == "mixed" ? 2U : 1U);
    EXPECT_NE(plan.digest(), PlanDigest{});
    EXPECT_TRUE(std::ranges::any_of(
        plan.payload_edges(), [](const auto& edge) { return edge.tensor_projection.has_value(); }));
    const auto* state = detail::ExecutionPlanAccess::state(plan);
    ASSERT_NE(state, nullptr);
    EXPECT_TRUE(std::ranges::any_of(state->process_local_channel_subscribers, [](const auto& sub) {
      return sub.tensor_projection.has_value();
    }));
    EXPECT_TRUE(state->cross_process_channel_subscribers.empty());
    auto session = run_async(plan);
    {
      std::unique_lock lock(probe->mutex);
      EXPECT_TRUE(probe->cv.wait_for(lock, 5s, [&] { return probe->received.has_value(); }));
      probe->received.reset();
    }
    if (route.ends_with("fanout") || route == "mixed") {
      std::unique_lock lock(sibling_probe->mutex);
      EXPECT_TRUE(sibling_probe->cv.wait_for(
          lock, 5s, [&] { return sibling_probe->received.has_value(); }));
      sibling_probe->received.reset();
    }
    session.request_stop();
    session.wait();
    EXPECT_TRUE(session.diagnostics().empty());
  }
}

// REQ: COMPILE-001
TEST(TensorProjection, CrossProcessWithoutStagingReturnsInspectableRejectedPlan) {
  for (const bool multi : {false, true}) {
    SCOPED_TRACE(multi);
    Graph graph;
    auto probe = std::make_shared<ProjectionProbe>();
    auto sink = graph.op<ProjectionSink>("sink", probe, MemoryKind::kHost);
    if (multi) {
      auto source = graph.op<MultiProjectionSource>("source", probe, -1);
      graph.add_flow(source->output.project(Fields::second), sink->input);
    } else {
      auto source = graph.op<SingleProjectionSource>("source", probe, Provider::kPassThrough);
      graph.add_flow(source->output, sink->input);
    }
    set_clock(graph);
    constexpr std::array producers{detail::GraphObjectSelector{.graph_path = "source"}};
    constexpr std::array consumers{detail::GraphObjectSelector{.graph_path = "sink"}};
    detail::ProcessPlacementBinding placement;
    placement.bind_process_role(
        "producer", detail::ProcessPlacement{.members = producers, .launch_group = "producer"});
    placement.bind_process_role(
        "consumer", detail::ProcessPlacement{.members = consumers, .launch_group = "consumer"});
    ExecutionPlan plan;
    ASSERT_NO_THROW(plan = detail::compile_with_process_placement(graph, {}, std::move(placement)));
    EXPECT_FALSE(plan.ok());
    EXPECT_FALSE(plan.json().empty());
    EXPECT_NE(plan.digest(), PlanDigest{});
    EXPECT_EQ(plan.graph().edges().size(), 1U);
    EXPECT_TRUE(plan.payload_edges().empty());
    const auto* state = detail::ExecutionPlanAccess::state(plan);
    ASSERT_NE(state, nullptr);
    EXPECT_TRUE(state->cross_process_channel_services.empty());
    EXPECT_TRUE(state->cross_process_channel_subscribers.empty());
    EXPECT_THROW((void)run_async(plan), CompileError);
  }
}

// REQ: GRAPH-045
// REQ: COMPILE-001
TEST(TensorProjection, QueuedFanoutRejectsOversizedSiblingWithoutThrowing) {
  for (const std::string_view route : {"control", "projection-first", "projection-last"}) {
    SCOPED_TRACE(route);
    const bool with_projection = route != "control";
    Graph graph;
    auto source = graph.op<ProjectionSiblingSource>("source");
    auto sink = graph.op<ProjectionSiblingSink>("sink", with_projection);
    const auto add_projections = [&] {
      auto other =
          graph.op<ProjectionSink>("other", std::make_shared<ProjectionProbe>(), MemoryKind::kHost);
      graph.add_flow(source->small, sink->projected);
      graph.add_flow(source->small, other->input);
    };
    if (route == "projection-first") {
      add_projections();
    }
    graph.add_flow(source->large, sink->large);
    if (route == "projection-last") {
      add_projections();
    }
    set_clock(graph);

    ExecutionPlan plan;
    ASSERT_NO_THROW(plan = compile(graph));
    EXPECT_EQ(plan.ok(), !with_projection) << plan.json();
    EXPECT_EQ(plan.graph().edges().size(), with_projection ? 3U : 1U);
    EXPECT_EQ(
        std::ranges::count_if(plan.graph().edges(),
                              [](const auto& edge) { return edge.tensor_projection.has_value(); }),
        with_projection ? 2 : 0);
    EXPECT_NE(plan.digest(), PlanDigest{});
    EXPECT_TRUE(plan.tensor_allocations().empty());
    EXPECT_TRUE(plan.tensor_field_allocations().empty());

    // Publication ordering puts the sibling Operator pair on HoloMQ when the projections fan out.
    // Its oversized envelope is then rejected, while the admitted projections retain bindings.
    ASSERT_EQ(plan.payload_edges().size(), with_projection ? 2U : 1U);
    const auto& edge = plan.payload_edges().front();
    EXPECT_EQ(edge.tensor_projection.has_value(), with_projection);
    EXPECT_EQ(edge.fanout, with_projection ? 2U : 1U);
    const auto* state = detail::ExecutionPlanAccess::state(plan);
    ASSERT_NE(state, nullptr);
    ASSERT_EQ(state->process_local_channel_services.size(), 1U);
    const auto& service = state->process_local_channel_services.front();
    EXPECT_EQ(service.source, edge.source);
    EXPECT_EQ(service.payload_size_bytes > 32768U, !with_projection);
    EXPECT_EQ(service.delivery_kind,
              with_projection ? detail::ProcessLocalChannelDeliveryKind::kHoloMQ
                              : detail::ProcessLocalChannelDeliveryKind::kDirectRetained);
    EXPECT_EQ(service.queue_policy, QueuePolicy::kBlocking);
    EXPECT_EQ(service.max_subscribers, with_projection ? 2U : 1U);
    ASSERT_EQ(state->process_local_channel_publishers.size(), 1U);
    EXPECT_EQ(state->process_local_channel_publishers.front().source, edge.source);
    ASSERT_EQ(state->process_local_channel_subscribers.size(), with_projection ? 2U : 1U);
    EXPECT_EQ(state->process_local_channel_subscribers.front().edge, edge.edge);
    EXPECT_EQ(state->process_local_channel_subscribers.front().tensor_projection.has_value(),
              with_projection);
    EXPECT_TRUE(state->cross_process_channel_subscribers.empty());

    if (with_projection) {
      EXPECT_EQ(std::ranges::count_if(plan.diagnostics(),
                                      [](const auto& diagnostic) {
                                        return diagnostic.severity == DiagnosticSeverity::kError;
                                      }),
                1);
      EXPECT_EQ(
          std::ranges::count_if(plan.diagnostics(),
                                [](const auto& diagnostic) {
                                  return diagnostic.token ==
                                         "CHANNEL_SCHEMA_DESCRIPTOR_ENVELOPE_CAPACITY_EXCEEDED";
                                }),
          1);
      EXPECT_NE(plan.json().find("\"admitted\":false"), std::string_view::npos);
      EXPECT_NE(plan.json().find("CHANNEL_SCHEMA_DESCRIPTOR_ENVELOPE_CAPACITY_EXCEEDED"),
                std::string_view::npos);
      EXPECT_THROW((void)run_async(plan), CompileError);
    }
  }
}

// REQ: GRAPH-045
// REQ: COMPILE-001
TEST(TensorProjection, MixedLargeSchemaFanoutReturnsInspectableRejectedPlan) {
  for (const auto conversion : {MemoryConversionPolicy::kDefault,
                                MemoryConversionPolicy::kRequire,
                                MemoryConversionPolicy::kAllowIfSafe,
                                MemoryConversionPolicy::kDisable}) {
    SCOPED_TRACE(static_cast<unsigned>(conversion));
    for (const bool with_sibling : {false, true}) {
      SCOPED_TRACE(with_sibling);
      for (const std::string_view route : {"control", "projection-first", "projection-last"}) {
        SCOPED_TRACE(route);
        const bool with_projection = route != "control";
        Graph graph;
        auto source = graph.op<ProjectionSiblingSource>("source");
        auto schema_sink = graph.op<ProjectionSiblingSink>("schema_sink", false, with_sibling);
        const auto add_projection = [&] {
          auto tensor_sink = graph.op<ProjectionSink>(
              "tensor_sink", std::make_shared<ProjectionProbe>(), MemoryKind::kHost);
          ConnectionOptions options;
          options.memory_conversion = conversion;
          graph.add_flow(source->large, tensor_sink->input, options);
        };
        if (route == "projection-first") {
          add_projection();
        }
        graph.add_flow(source->large, schema_sink->large);
        if (with_sibling) {
          // This independent output shares the same Operator pair as the rejected ordinary arm.
          graph.add_flow(source->small, schema_sink->small);
        }
        if (route == "projection-last") {
          add_projection();
        }
        set_clock(graph);

        ExecutionPlan plan;
        ASSERT_NO_THROW(plan = compile(graph));
        EXPECT_EQ(plan.ok(), !with_projection) << plan.json();
        EXPECT_EQ(plan.graph().edges().size(), (with_projection ? 2U : 1U) + with_sibling);
        EXPECT_EQ(std::ranges::count_if(
                      plan.graph().edges(),
                      [](const auto& edge) { return edge.tensor_projection.has_value(); }),
                  with_projection ? 1 : 0);
        EXPECT_NE(plan.digest(), PlanDigest{});
        EXPECT_TRUE(plan.tensor_allocations().empty());
        EXPECT_TRUE(plan.tensor_field_allocations().empty());
        const auto* state = detail::ExecutionPlanAccess::state(plan);
        ASSERT_NE(state, nullptr);
        EXPECT_TRUE(state->cross_process_channel_services.empty());
        EXPECT_TRUE(state->cross_process_channel_publishers.empty());
        EXPECT_TRUE(state->cross_process_channel_subscribers.empty());

        // The rejected source family publishes no bindings; every surviving independent edge
        // keeps a complete direct service, publisher, subscriber and locality record.
        const std::size_t retained_edges = (with_projection ? 0U : 1U) + (with_sibling ? 1U : 0U);
        ASSERT_EQ(plan.payload_edges().size(), retained_edges);
        EXPECT_EQ(state->channel_locality.size(), retained_edges);
        EXPECT_EQ(state->process_local_channel_services.size(), retained_edges);
        EXPECT_EQ(state->process_local_channel_publishers.size(), retained_edges);
        EXPECT_EQ(state->process_local_channel_subscribers.size(), retained_edges);
        for (const auto& edge : plan.payload_edges()) {
          EXPECT_FALSE(edge.tensor_projection);
          EXPECT_EQ(edge.fanout, 1U);
          EXPECT_TRUE(std::ranges::any_of(state->channel_locality, [&](const auto& locality) {
            return locality.edge == edge.edge;
          }));
          const auto service =
              std::ranges::find_if(state->process_local_channel_services,
                                   [&](const auto& item) { return item.source == edge.source; });
          ASSERT_NE(service, state->process_local_channel_services.end());
          EXPECT_EQ(service->delivery_kind,
                    detail::ProcessLocalChannelDeliveryKind::kDirectRetained);
          EXPECT_EQ(service->queue_policy, QueuePolicy::kBlocking);
          EXPECT_EQ(service->max_subscribers, 1U);
          EXPECT_TRUE(std::ranges::any_of(state->process_local_channel_publishers,
                                          [&](const auto& publisher) {
                                            return publisher.source == edge.source &&
                                                   publisher.service_id == service->service_id;
                                          }));
          EXPECT_TRUE(std::ranges::any_of(
              state->process_local_channel_subscribers, [&](const auto& subscriber) {
                return subscriber.edge == edge.edge && subscriber.source == edge.source &&
                       subscriber.target == edge.target &&
                       subscriber.service_id == service->service_id &&
                       !subscriber.tensor_projection;
              }));
        }
        EXPECT_EQ(std::ranges::count_if(
                      state->process_local_channel_services,
                      [](const auto& service) { return service.payload_size_bytes > 32768U; }),
                  with_projection ? 0 : 1);

        if (with_projection) {
          const std::string_view expected_token =
              conversion == MemoryConversionPolicy::kDefault
                  ? "CHANNEL_SCHEMA_DESCRIPTOR_ENVELOPE_CAPACITY_EXCEEDED"
                  : "PHYSICAL_TENSOR_CONVERSION_UNSUPPORTED";
          EXPECT_EQ(std::ranges::count_if(plan.diagnostics(),
                                          [](const auto& diagnostic) {
                                            return diagnostic.severity ==
                                                   DiagnosticSeverity::kError;
                                          }),
                    1);
          EXPECT_EQ(std::ranges::count_if(
                        plan.diagnostics(),
                        [&](const auto& diagnostic) { return diagnostic.token == expected_token; }),
                    1);
          EXPECT_NE(plan.json().find("\"admitted\":false"), std::string_view::npos);
          EXPECT_NE(plan.json().find(expected_token), std::string_view::npos);
          EXPECT_THROW((void)run_async(plan), CompileError);
        }
      }
    }
  }
}

// REQ: GRAPH-045
TEST(TensorProjection, RejectedProjectionRemainsInspectableAndCannotRun) {
  auto probe = std::make_shared<ProjectionProbe>();
  Graph graph;
  auto source = graph.op<MultiProjectionSource>("source", probe, -1);
  auto sink = graph.op<ProjectionSink>("sink", probe, MemoryKind::kHost, 2U);
  // The selected field is rank 1. Its compatibility is checked during physical lowering,
  // after the schema channel records have been created.
  ASSERT_NO_THROW(graph.add_flow(source->output.project(Fields::second), sink->input));
  set_clock(graph);
  const auto plan = compile(graph);

  ASSERT_FALSE(plan.ok()) << plan.json();
  const auto mismatch = std::ranges::find_if(plan.diagnostics(), [](const auto& diagnostic) {
    return diagnostic.severity == DiagnosticSeverity::kError &&
           diagnostic.token == "PHYSICAL_TENSOR_LAYOUT_MISMATCH";
  });
  ASSERT_NE(mismatch, plan.diagnostics().end()) << plan.json();
  ASSERT_EQ(plan.payload_edges().size(), 1U);
  EXPECT_TRUE(plan.payload_edges().front().tensor_projection.has_value());
  EXPECT_EQ(mismatch->source.port, plan.payload_edges().front().target);
  EXPECT_NE(plan.json().find("\"admitted\":false"), std::string_view::npos);
  EXPECT_NE(plan.json().find("PHYSICAL_TENSOR_LAYOUT_MISMATCH"), std::string_view::npos);

  // Inspectable rejection artifacts never authorize execution, even with channel records.
  EXPECT_THROW(plan.require_no_errors(), CompileError);
  EXPECT_THROW((void)run_async(plan), CompileError);
  const std::lock_guard lock(probe->mutex);
  EXPECT_FALSE(probe->received.has_value());
  EXPECT_TRUE(
      std::ranges::all_of(probe->pointers, [](const void* pointer) { return pointer == nullptr; }));
}

// REQ: GRAPH-045
TEST(TensorProjection, ProvidersRejectMismatchedConsumersAndExplicitConversion) {
  for (const auto provider : {Provider::kPassThrough, Provider::kLifecycle}) {
    for (const bool conversion : {false, true}) {
      auto probe = std::make_shared<ProjectionProbe>();
      Graph graph;
      auto source = graph.op<SingleProjectionSource>("source", probe, provider);
      auto sink = graph.op<ProjectionSink>(
          "sink", probe, MemoryKind::kHost, conversion ? 1U : 2U, provider == Provider::kLifecycle);
      ConnectionOptions options;
      if (conversion) {
        options.memory_conversion = MemoryConversionPolicy::kRequire;
      }
      if (!conversion) {
        EXPECT_THROW(graph.add_flow(source->output, sink->input, options), RuntimeError);
        EXPECT_TRUE(graph.view().edges().empty());
        continue;
      }
      graph.add_flow(source->output, sink->input, options);
      set_clock(graph);
      const auto plan = compile(graph);
      EXPECT_FALSE(plan.ok());
      EXPECT_TRUE(std::ranges::any_of(plan.diagnostics(), [&](const auto& diagnostic) {
        return diagnostic.token == "PHYSICAL_TENSOR_CONVERSION_UNSUPPORTED";
      })) << plan.json();
    }
  }
}

}  // namespace
}  // namespace holoscan
