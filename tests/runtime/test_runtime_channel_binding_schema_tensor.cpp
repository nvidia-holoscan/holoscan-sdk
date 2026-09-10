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
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <optional>
#include <semaphore>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/memory/block_memory_pool.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/schema/flatbuffer_schema_package.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "../../src/channel/channel_common.hpp"
#include "../../src/channel/channel_tensor_test_access.hpp"
#include "../../src/channel/holomq_tensor.hpp"
#include "../common/cuda_test_utils.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/runtime_package_capabilities.hpp"
#include "core/graph_compiler_access.hpp"
#include "core/source_id_internal.hpp"
#include "runtime/channel_runtime_binding_access.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/runtime_channel_binding.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_test_access.hpp"

#include "runtime_channel_binding_schema_tensor_test_support.hpp"
#include "runtime_channel_binding_test_support.hpp"

namespace holoscan::detail {
namespace {

using namespace std::chrono_literals;
using namespace runtime_channel_binding_test;
using namespace runtime_channel_binding_schema_tensor_test;

class ScriptedTerminalTensorSchemaPublisher final
    : public channel_internal::TensorSchemaPayloadPublisher {
 public:
  [[nodiscard]] expected<channel_internal::TensorSchemaPayloadReservation, Error> reserve_tensors(
      std::span<const Tensor>, bool) override {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }

  [[nodiscard]] expected<std::uint32_t, Error> publish_payload(
      std::span<const std::byte>, channel_internal::TensorSchemaPayloadReservation,
      SampleMetadata) override {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }

  [[nodiscard]] channel_internal::InlinePublicationResult close_stream(
      const SampleMetadata metadata) override {
    close_sequences.push_back(metadata.sequence);
    if (close_sequences.size() == 1U) {
      return {
          .phase = channel_internal::InlinePublicationPhase::kPreSubmitRejected,
          .recipient_count = 0U,
          .error = Error{ErrorCode::kResourceExhausted},
      };
    }
    return {
        .phase = channel_internal::InlinePublicationPhase::kAccepted,
        .recipient_count = 1U,
        .error = Error{ErrorCode::kSuccess},
    };
  }

  [[nodiscard]] expected<std::uint64_t, Error> next_operation_sequence() const noexcept override {
    return 1U;
  }

  void release_tensor(channel_internal::TensorSchemaPayloadReservation) noexcept override {}
  [[nodiscard]] expected<void, Error> update_connections() noexcept override { return {}; }
  [[nodiscard]] bool has_subscribers() const noexcept override { return true; }
  [[nodiscard]] expected<MemoryBuffer::Options, Error> tensor_memory_options() const override {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }

  std::vector<std::uint64_t> close_sequences;
};

[[nodiscard]] expected<void, Error> reject_second_image_assignment(
    void* storage, const std::span<const std::byte> encoded, const std::span<Tensor> tensors,
    const DescriptorTensorWireMetadata wire_metadata,
    const DescriptorTensorSampleMetadata sample_metadata) noexcept {
  if (wire_metadata.metadata.sequence == 2U) {
    auto& sample = *static_cast<Sample<schema::ImageT>*>(storage);
    sample.data.header->frame_id = "partially-mutated-image";
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  return runtime_port_type_ops<schema::ImageT>()->assign_descriptor_tensors_sample(
      storage, encoded, tensors, wire_metadata, sample_metadata);
}

template <typename Traits>
struct TransportedTerminalReuseProbe {
  static void reset() noexcept {
    callbacks.store(0U, std::memory_order_relaxed);
    source_calls.store(0U, std::memory_order_relaxed);
    emissions.store(0U, std::memory_order_relaxed);
    value.store(0U, std::memory_order_relaxed);
    sequence.store(0U, std::memory_order_relaxed);
    state.store(InputState::kAbsent, std::memory_order_relaxed);
  }

  static inline std::atomic<std::uint64_t> callbacks{};
  static inline std::atomic<std::uint64_t> source_calls{};
  static inline std::atomic<std::uint64_t> emissions{};
  static inline std::atomic<std::uint64_t> value{};
  static inline std::atomic<std::uint64_t> sequence{};
  static inline std::atomic<InputState> state{InputState::kAbsent};
};

struct RawTensorTerminalReuseTraits {
  using Payload = Tensor;

  static void configure_output(TensorOutputSpecBuilder& declaration) {
    declaration.produces_tensor(TensorOutputSpec{
        .representation = {.memory_kind = MemoryKind::kHost,
                           .dtype = DLDataType{kDLUInt, 64U, 1U},
                           .rank = 0U},
        .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 8U},
        .storage = TensorOutputStorage::kRuntimePool,
    });
  }

  static void configure_input(InputSpecBuilder& declaration) {
    declaration.expects_tensor(TensorInputSpec{.representation = {
                                                   .memory_kind = MemoryKind::kHost,
                                                   .dtype = DLDataType{kDLUInt, 64U, 1U},
                                                   .rank = 0U,
                                               }});
  }

  static expected<void, Error> emit(Output<Payload>& output, const std::uint64_t value,
                                    const TimePoint capture_time) {
    auto loan = output.allocate_tensor(
        TensorLoanRequest{.shape = TensorShapeView{}, .dtype = DLDataType{kDLUInt, 64U, 1U}});
    if (!loan) {
      return make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write_host();
    if (!writer || writer->data_as<std::uint64_t>() == nullptr) {
      return make_unexpected(writer ? Error{ErrorCode::kFailure} : std::move(writer).error());
    }
    *writer->data_as<std::uint64_t>() = value;
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return make_unexpected(std::move(committed).error());
    }
    return output.emit(std::move(*loan), EmitOptions{.capture_time = capture_time});
  }

  static const std::uint64_t* value(const Sample<Payload>& sample) noexcept {
    return sample.data.data_as<std::uint64_t>();
  }
};

struct DescriptorSchemaTerminalReuseTraits {
  using Payload = schema::ImageT;

  static void configure_output(TensorOutputSpecBuilder& declaration) {
    declaration.produces_tensor(TensorOutputSpec{
        .representation = {.memory_kind = MemoryKind::kHost,
                           .dtype = DLDataType{kDLUInt, 64U, 1U},
                           .rank = 1U},
        .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 8U},
        .storage = TensorOutputStorage::kRuntimePool,
    });
  }

  static void configure_input(InputSpecBuilder& declaration) {
    declaration.expects_tensor(TensorInputSpec{.representation = {
                                                   .memory_kind = MemoryKind::kHost,
                                                   .dtype = DLDataType{kDLUInt, 64U, 1U},
                                                   .rank = 1U,
                                               }});
  }

  static expected<void, Error> emit(Output<Payload>& output, const std::uint64_t value,
                                    const TimePoint capture_time) {
    constexpr std::array<std::int64_t, 1U> kShape{1};
    auto loan = output.allocate_tensor(
        TensorLoanRequest{.shape = kShape, .dtype = DLDataType{kDLUInt, 64U, 1U}});
    if (!loan) {
      return make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write_host();
    if (!writer || writer->data_as<std::uint64_t>() == nullptr) {
      return make_unexpected(writer ? Error{ErrorCode::kFailure} : std::move(writer).error());
    }
    *writer->data_as<std::uint64_t>() = value;
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return make_unexpected(std::move(committed).error());
    }
    schema::ImageT descriptor{};
    descriptor.height = 1;
    descriptor.width = 1;
    descriptor.encoding = schema::ImageEncoding_CUSTOM;
    descriptor.header = std::make_shared<schema::HeaderT>();
    descriptor.header->frame_id = value == 0x31U ? "retained" : "final";
    return output.emit_tensor(
        std::move(*loan), descriptor, EmitOptions{.capture_time = capture_time});
  }

  static const std::uint64_t* value(const Sample<Payload>& sample) noexcept {
    return sample.data.data == nullptr ? nullptr : sample.data.data->data_as<std::uint64_t>();
  }
};

template <typename Traits>
class TransportedTerminalReuseSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    auto& declaration = spec.output(data, "data").max_emits_per_compute(1U);
    Traits::configure_output(declaration);
    spec.output(driver, "driver").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 20ms}).complete_after(ComputeCalls{.limit = 4U});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const std::uint64_t call = ++calls_;
    TransportedTerminalReuseProbe<Traits>::source_calls.store(call, std::memory_order_release);
    if (call > 2U) {
      return {};
    }
    const TimePoint capture_time = context.activation_time();
    auto emitted = Traits::emit(data, call == 1U ? 0x31U : 0x42U, capture_time);
    if (!emitted) {
      return emitted;
    }
    TransportedTerminalReuseProbe<Traits>::emissions.fetch_add(1U, std::memory_order_release);
    return call == 1U ? driver.emit(Sample<std::uint64_t>{
                            call,
                            SampleMetadata{
                                .capture_timestamp_ns = capture_time.timestamp_ns,
                                .source_clock_id = capture_time.clock_id,
                            },
                        })
                      : expected<void, Error>{};
  }

  Output<typename Traits::Payload> data;
  Output<std::uint64_t> driver;

 private:
  std::uint64_t calls_{};
};

template <typename Traits>
class TransportedTerminalReuseSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    auto& declaration = spec.input(data, "data").queue_depth(1U);
    Traits::configure_input(declaration);
    spec.input(driver, "driver").queue_depth(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{driver});
    result.input(data, Exact{}).on_missing(data, ReuseLast{.max_age = 30s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto trigger = driver.receive();
    auto sample = data.receive();
    if (!trigger || !sample) {
      return make_unexpected(!trigger ? std::move(trigger).error() : std::move(sample).error());
    }
    const std::uint64_t* const observed = Traits::value(*sample);
    if (observed == nullptr) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    TransportedTerminalReuseProbe<Traits>::value.store(*observed, std::memory_order_relaxed);
    TransportedTerminalReuseProbe<Traits>::sequence.store(sample->metadata.sequence,
                                                          std::memory_order_relaxed);
    TransportedTerminalReuseProbe<Traits>::state.store(data.selection().state,
                                                       std::memory_order_relaxed);
    TransportedTerminalReuseProbe<Traits>::callbacks.fetch_add(1U, std::memory_order_release);
    return {};
  }

  Input<typename Traits::Payload> data;
  Input<std::uint64_t> driver;
};

template <typename Traits>
void run_transported_terminal_reuse_without_following_activation(const std::string_view name) {
  TransportedTerminalReuseProbe<Traits>::reset();
  Graph graph{std::string{name}};
  auto source = graph.op<TransportedTerminalReuseSource<Traits>>("source");
  auto sink = graph.op<TransportedTerminalReuseSink<Traits>>("sink");
  graph.add_flow(source->data,
                 sink->data,
                 ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U});
  graph.add_flow(source->driver, sink->driver, ConnectionOptions{.queue_depth = 1U});
  graph.partition("source").add(source);
  graph.partition("sink").add(sink);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-clock"));
  auto counters = std::make_shared<RuntimePhaseCounters>();
  const ExecutionPlan compiled = compile_with_process_local_holomq(graph, CompileOptions{});
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const ExecutionPlan plan = RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  const bool completed = !session.is_running();
  if (!completed) {
    session.request_stop();
  }
  session.wait();

  ASSERT_TRUE(completed);
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(TransportedTerminalReuseProbe<Traits>::source_calls.load(std::memory_order_acquire),
            4U);
  EXPECT_EQ(TransportedTerminalReuseProbe<Traits>::emissions.load(std::memory_order_acquire), 2U);
  EXPECT_EQ(TransportedTerminalReuseProbe<Traits>::callbacks.load(std::memory_order_acquire), 1U);
  EXPECT_EQ(TransportedTerminalReuseProbe<Traits>::value.load(std::memory_order_relaxed), 0x31U);
  EXPECT_EQ(TransportedTerminalReuseProbe<Traits>::sequence.load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(TransportedTerminalReuseProbe<Traits>::state.load(std::memory_order_relaxed),
            InputState::kSelected);
  EXPECT_EQ(counters->snapshot()[RuntimePhaseCounters::index(
                RuntimePhaseCounter::kTemporalRetentionEvictions)],
            1U);
}

// REQ: CHAN-136
// REQ: RUNTIME-050
TEST(ChannelBindingParityTest, ForcedRawTensorTerminalHandoffDrainsWithoutAFollowingActivation) {
  run_transported_terminal_reuse_without_following_activation<RawTensorTerminalReuseTraits>(
      "forced-raw-tensor-terminal-reuse");
}

// REQ: CHAN-136
// REQ: RUNTIME-050
TEST(ChannelBindingParityTest,
     ForcedDescriptorSchemaTerminalHandoffDrainsWithoutAFollowingActivation) {
  run_transported_terminal_reuse_without_following_activation<DescriptorSchemaTerminalReuseTraits>(
      "forced-descriptor-schema-terminal-reuse");
}

// Descriptor-schema terminal loan exhaustion is pre-submit even for a non-local endpoint. The
// retry must preserve the endpoint-local sequence and publish exactly one terminal operation.
// REQ: CHAN-136
TEST(ChannelBindingParityTest,
     NonLocalDescriptorSchemaPreSubmitResourceExhaustionRemainsRetryable) {
  ImageSessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto publisher = std::make_unique<ScriptedTerminalTensorSchemaPublisher>();
  auto* const observed = publisher.get();
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_with_nonlocal_tensor_schema_publisher(
      **session, PortId{1U}, std::move(publisher)));

  auto pressured = (*session)->close_owner_outputs(VertexId{1U});
  ASSERT_FALSE(pressured);
  EXPECT_EQ(pressured.error().code, ErrorCode::kBackpressured);
  EXPECT_EQ(observed->close_sequences, (std::vector<std::uint64_t>{1U}));

  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  EXPECT_EQ(observed->close_sequences, (std::vector<std::uint64_t>{1U, 1U}));
  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  EXPECT_EQ(observed->close_sequences, (std::vector<std::uint64_t>{1U, 1U}));
}

void run_catalog_tensor_terminal_exhaustion_scenario(const bool permanent_sequence_exhaustion) {
  const std::string endpoint_name = unique_service();
  Graph graph{"catalog-tensor-terminal-pre-submit-" + endpoint_name};
  const ClockKey clock = graph.bind_shared_clock("runtime-clock", std::make_shared<ManualClock>(0));
  graph.set_default_clock(clock);
  const auto source = graph.op<ContractTensorSourceOperator>("source");

  static constexpr std::array axes{TensorAxisBounds{.max_extent = 16U, .max_abs_stride_bytes = 1U}};
  TopicOptions topic_options;
  topic_options.endpoint_name = endpoint_name;
  topic_options.attach = TopicAttachPolicy{.peer = PeerRequirement::kRequired, .max_wait = 2s};
  topic_options.bounds.subscriber =
      SubscriberEndpointBounds{.queue_depth = 1U, .max_borrowed_samples = 1U};
  topic_options.tensor = TensorBoundaryContract{
      .representation = {.memory_kind = MemoryKind::kHost,
                         .dtype = DLDataType{kDLUInt, 8U, 1U},
                         .rank = 1U,
                         .strides = TensorStridePolicy::kContiguousOnly},
      .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 16U, .axes = axes},
      .transfer =
          TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
  };
  auto exported = graph.export_topic<Tensor>("holomq://" + endpoint_name, std::move(topic_options));
  graph.add_flow(source->output,
                 exported.input(),
                 ConnectionOptions{.queue_policy = QueuePolicy::kBlocking, .queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  const auto* planned_output = planned_operator_port(*state, PortDirection::kOutput);
  ASSERT_NE(planned_output, nullptr);
  const std::array replay_ports{RuntimeChannelPortReplayRecord{
      .port = planned_output->id,
      .owner = planned_output->owner,
      .direction = PortDirection::kOutput,
      .declaration = const_cast<void*>(static_cast<const void*>(&source->output)),
      .type_ops = runtime_port_type_ops<Tensor>(),
  }};
  auto session = RuntimeChannelSession::create(*state, replay_ports);
  ASSERT_TRUE(session) << runtime_error_message(session.error());

  auto client_runtime = ChannelRuntime::create();
  ASSERT_TRUE(client_runtime) << runtime_error_message(client_runtime.error());
  auto subscriber =
      client_runtime->create_subscriber<Tensor>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());

  if (permanent_sequence_exhaustion) {
    expected<void, Error> exhausted = make_unexpected(Error{ErrorCode::kNotReady});
    const auto attach_deadline = std::chrono::steady_clock::now() + 2s;
    while (!exhausted && exhausted.error().code == ErrorCode::kNotReady &&
           std::chrono::steady_clock::now() < attach_deadline) {
      ASSERT_TRUE((*session)->refresh_connections());
      exhausted = RuntimeChannelSessionTestAccess::exhaust_catalog_tensor_operation_sequence(
          **session, planned_output->id);
      if (!exhausted) {
        std::this_thread::sleep_for(1ms);
      }
    }
    ASSERT_TRUE(exhausted) << runtime_error_message(exhausted.error());

    for (std::size_t attempt = 0U; attempt < 2U; ++attempt) {
      auto closed = (*session)->close_owner_outputs(planned_output->owner);
      ASSERT_FALSE(closed);
      EXPECT_EQ(closed.error().code, ErrorCode::kResourceExhausted);
      EXPECT_EQ(RuntimeChannelBindingAccess::accepted_tensor_end_of_stream_sequence(*subscriber),
                0U);
    }
    auto absent = subscriber->receive_sample();
    ASSERT_FALSE(absent);
    EXPECT_EQ(absent.error().code, ErrorCode::kNotReady);
    return;
  }

  expected<void, Error> injected = make_unexpected(Error{ErrorCode::kNotReady});
  const auto attach_deadline = std::chrono::steady_clock::now() + 2s;
  while (!injected && injected.error().code == ErrorCode::kNotReady &&
         std::chrono::steady_clock::now() < attach_deadline) {
    ASSERT_TRUE((*session)->refresh_connections());
    injected = RuntimeChannelSessionTestAccess::fail_next_catalog_tensor_terminal_pre_submit(
        **session, planned_output->id);
    if (!injected) {
      std::this_thread::sleep_for(1ms);
    }
  }
  ASSERT_TRUE(injected) << runtime_error_message(injected.error());

  auto pressured = (*session)->close_owner_outputs(planned_output->owner);
  ASSERT_FALSE(pressured);
  EXPECT_EQ(pressured.error().code, ErrorCode::kBackpressured);

  ASSERT_TRUE((*session)->close_owner_outputs(planned_output->owner));
  ASSERT_TRUE((*session)->close_owner_outputs(planned_output->owner));

  auto terminal = receive_catalog_tensor(*subscriber);
  ASSERT_FALSE(terminal);
  EXPECT_EQ(terminal.error().code, ErrorCode::kClosed);
  EXPECT_EQ(RuntimeChannelBindingAccess::accepted_tensor_end_of_stream_sequence(*subscriber), 1U);
  auto repeated = subscriber->receive_sample();
  ASSERT_FALSE(repeated);
  EXPECT_EQ(repeated.error().code, ErrorCode::kClosed);
  EXPECT_EQ(RuntimeChannelBindingAccess::accepted_tensor_end_of_stream_sequence(*subscriber), 1U);
}

// Catalog Tensor terminal-loan exhaustion is proven pre-submit after the permanent sequence
// preflight. Runtime must retry the exact unresolved output and publish one sequence-one EOS.
// REQ: CHAN-136
TEST(ChannelBindingParityTest, CatalogTensorPreSubmitResourceExhaustionRemainsRetryable) {
  ASSERT_NO_FATAL_FAILURE(run_catalog_tensor_terminal_exhaustion_scenario(false));
}

// Permanent endpoint sequence exhaustion is terminal and occurs before retry normalization.
// Repeated close attempts remain exhausted and publish no EOS.
// REQ: CHAN-136
TEST(ChannelBindingParityTest, CatalogTensorPermanentSequenceExhaustionRemainsTerminal) {
  ASSERT_NO_FATAL_FAILURE(run_catalog_tensor_terminal_exhaustion_scenario(true));
}

// Direct-retained descriptor Tensor delivery keeps the ordinary readiness path. Receipt
// bookkeeping does not improve the production-sized path enough to justify touching publication.
// REQ: CHAN-099
// REQ: RUNTIME-019
TEST(ChannelBindingParityTest, DirectDescriptorTensorUsesOrdinaryReadinessInsteadOfContinuation) {
  // Key setup: create an otherwise eligible exact shared-lane direct descriptor Tensor edge.
  ImageSessionFixture fixture;
  fixture.use_direct_retained_delivery();
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.base.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 1U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  EXPECT_FALSE((*session)->arm_direct_continuation(VertexId{1U}, 0U));

  auto published = publish_image_sample(**session, **binding, fixture.source, 91U);
  ASSERT_TRUE(published) << runtime_error_message(published.error());
  const std::array selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = InputSelection{InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&published->record, 1U},
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selection));
  ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 1U, nullptr));

  // Key assertion: ordinary polling publishes one readiness record while schema metadata and the
  // embedded Tensor backing remain valid through typed receive and retirement.
  {
    auto received = fixture.sink.input.receive();
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    ASSERT_NE(received->data.data, nullptr);
    ASSERT_NE(received->data.data->data_as<std::uint64_t>(), nullptr);
    EXPECT_EQ(*received->data.data->data_as<std::uint64_t>(), 91U);
    ASSERT_NE(received->data.header, nullptr);
    EXPECT_EQ(received->data.header->frame_id, "runtime_image");
    EXPECT_EQ(published->releases->load(std::memory_order_relaxed), 0U);
  }
  ASSERT_TRUE((*binding)->retire(published->event.source_sequence));
  ASSERT_TRUE((*session)->retire_samples(PortId{2U}, selection.front().samples));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  EXPECT_EQ(published->releases->load(std::memory_order_relaxed), 1U);
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// Runtime must not retain or surface a terminal witness until the descriptor-Tensor endpoint has
// accepted the complete canonical EOS envelope. A rejected raw terminal leaves the same endpoint
// open so the later valid terminal can close it on both ordinary and protected lanes.
// REQ: CHAN-136
TEST(ChannelBindingParityTest, MalformedDescriptorTensorTerminalDoesNotExposeRuntimeEndOfStream) {
  using ImageEnvelope = schema::detail::DescriptorTensorRuntimeEnvelope<
      schema::Image,
      schema::flatbuffer_default_descriptor_metadata_capacity_v<schema::Image>,
      1U>;
  constexpr auto bucket_size = channel_internal::inline_payload_bucket_size(sizeof(ImageEnvelope));
  static_assert(bucket_size.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by static_assert above.
  constexpr std::size_t kBucketSize = *bucket_size;
  using Bucket = channel_internal::InlinePayloadBucket<kBucketSize>;

  for (const QueuePolicy policy : {QueuePolicy::kBlocking, QueuePolicy::kDropOldest}) {
    SCOPED_TRACE(static_cast<std::uint32_t>(policy));
    ImageSessionFixture fixture;
    if (policy == QueuePolicy::kDropOldest) {
      fixture.base.use_one_slot_drop_oldest_queue();
    } else {
      fixture.base.use_one_slot_blocking_transport();
    }
    // The plan retains both the typed output and input declarations; replay must cover that entire
    // channel contract even though this test injects the terminal record through a raw publisher.
    auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    auto binding = (*session)->create_readiness_binding(
        VertexId{2U},
        RuntimePortBindingRequest{
            .plan_generation = fixture.base.plan.plan_generation,
            .run_generation = 1U,
            .kind = RuntimeEventKind::kInputArrival,
            .source = make_source_id(1U),
            .source_clock = ClockId{1U},
            .reference_clock = ClockId{1U},
            .capacity = 2U,
        });
    ASSERT_TRUE(binding) << runtime_error_message(binding.error());

    auto publisher_runtime = ChannelRuntime::create();
    ASSERT_TRUE(publisher_runtime) << runtime_error_message(publisher_runtime.error());
    holomq::Context* const context =
        channel_internal::TensorPublisherTestAccess::transport_context(*publisher_runtime);
    ASSERT_NE(context, nullptr);
    const auto& service = fixture.base.plan.process_local_channel_services.front();
    const bool protected_terminal = policy == QueuePolicy::kDropOldest;
    const std::string terminal_service =
        protected_terminal ? service.service_id + ".hsdk-terminal-v1" : service.service_id;
    const holomq::PubSubServiceOptions service_options{
        .queue_policy = holomq::QueuePolicy::kBlocking,
        .max_subscribers = service.max_subscribers,
        .max_subscriber_queue_depth = protected_terminal ? 1U : service.max_subscriber_queue_depth,
        .max_subscriber_borrowed_samples =
            protected_terminal ? 1U : service.max_subscriber_borrowed_samples,
        .publisher_history_depth = 0U,
        .enable_event_wake = service.event_wake,
    };
    const holomq::PublisherPubSubOptions publisher_options{
        .max_loaned_samples = protected_terminal ? 1U : 0U,
        .expected_recipient_count = service.max_subscribers,
        .enable_sample_release_callbacks = !protected_terminal,
        .replacement_attribution_required = false,
        .pressure_policy = holomq::PublisherPressurePolicy::kFollowService,
        .bounded_retry_budget_ns = 0U,
    };
    auto publisher = context->publisher_builder(terminal_service)
                         .service_options(service_options)
                         .publisher_options(publisher_options)
                         .payload_type<Bucket>()
                         .create();
    ASSERT_TRUE(publisher);
    ASSERT_TRUE((*session)->refresh_connections());
    const auto attach_deadline = std::chrono::steady_clock::now() + 2s;
    while (!publisher->has_subscribers() && std::chrono::steady_clock::now() < attach_deadline) {
      ASSERT_TRUE(publisher->update_connections());
      std::this_thread::sleep_for(1ms);
    }
    ASSERT_TRUE(publisher->has_subscribers());

    const auto send_terminal = [&](const bool malformed) {
      auto message = publisher->loan();
      EXPECT_TRUE(message);
      if (!message) {
        return false;
      }
      Bucket& bucket = message->payload();
      bucket.operation = channel_internal::InlinePayloadOperation::kEndOfStream;
      bucket.metadata = SampleMetadata{
          .sequence = 1U,
          .byte_size = 0U,
          .payload_kind = PayloadKind::kDescriptorBacked,
      };
      bucket.payload_size = malformed ? 1U : 0U;
      const auto sent = publisher->send(std::move(*message));
      EXPECT_TRUE(sent);
      return sent.has_value();
    };

    ASSERT_TRUE(send_terminal(true));
    const auto rejected = RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U});
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, ErrorCode::kProtocolError);
    const auto rejected_state =
        RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    ASSERT_TRUE(rejected_state) << runtime_error_message(rejected_state.error());
    EXPECT_FALSE(rejected_state->end_of_stream_received);
    RuntimeEvent unexpected;
    const auto no_witness = (*binding)->drain(std::span<RuntimeEvent>{&unexpected, 1U});
    ASSERT_TRUE(no_witness) << runtime_error_message(no_witness.error());
    EXPECT_EQ(*no_witness, 0U);

    ASSERT_TRUE(send_terminal(false));
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
    const auto accepted_state =
        RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    ASSERT_TRUE(accepted_state) << runtime_error_message(accepted_state.error());
    EXPECT_TRUE(accepted_state->end_of_stream_received);
    const auto terminal = wait_for_event(**session, **binding);
    ASSERT_TRUE(terminal) << runtime_error_message(terminal.error());
    EXPECT_EQ(terminal->kind, RuntimeEventKind::kInputEndOfStream);
    ASSERT_TRUE((*binding)->retire(terminal->source_sequence));
    (*session)->cancel();
    ASSERT_TRUE((*session)->join());
  }
}

// Ordinary raw-Tensor and descriptor-schema transports must classify EOS before requiring typed
// data capacity. The final ReuseLast sample may occupy the only slot until terminal history is
// released, so reservation-first receive would otherwise make the terminal record unreachable.
// REQ: CHAN-136
TEST(ChannelBindingParityTest, OrdinaryTensorTerminalsBypassFullReuseLastTypedSlot) {
  {
    SCOPED_TRACE("raw Tensor");
    TensorSessionFixture fixture;
    fixture.use_one_slot_blocking_transport();
    auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    ASSERT_TRUE((*session)->refresh_connections());
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
    auto owned = make_owned_tensor_sample(0x39U);
    ASSERT_TRUE(owned) << runtime_error_message(owned.error());
    ASSERT_TRUE(fixture.source.output.emit(std::move(owned->tensor)));
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_single_occupied_claim_state(
        **session,
        PortId{2U},
        /*active_selected=*/false,
        /*reuse_retained=*/true,
        /*temporal_retained=*/false,
        /*retirement_selected=*/false));

    ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
    const auto state = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    ASSERT_TRUE(state) << runtime_error_message(state.error());
    EXPECT_EQ(state->occupied_samples, 1U);
    EXPECT_TRUE(state->end_of_stream_received);
    (*session)->cancel();
    ASSERT_TRUE((*session)->join());
  }

  {
    SCOPED_TRACE("descriptor Tensor schema");
    ImageSessionFixture fixture;
    fixture.use_one_slot_blocking_transport();
    auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    ASSERT_TRUE((*session)->refresh_connections());
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
    auto owned = make_owned_tensor_sample(0x48U);
    ASSERT_TRUE(owned) << runtime_error_message(owned.error());
    schema::ImageT image{};
    image.height = 1;
    image.width = 1;
    image.encoding = schema::ImageEncoding_CUSTOM;
    image.header = std::make_shared<schema::HeaderT>();
    image.header->frame_id = "full-reuse-last-terminal";
    image.data = std::make_shared<Tensor>(std::move(owned->tensor));
    ASSERT_TRUE(fixture.source.output.emit(std::move(image)));
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_single_occupied_claim_state(
        **session,
        PortId{2U},
        /*active_selected=*/false,
        /*reuse_retained=*/true,
        /*temporal_retained=*/false,
        /*retirement_selected=*/false));

    ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
    const auto state = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    ASSERT_TRUE(state) << runtime_error_message(state.error());
    EXPECT_EQ(state->occupied_samples, 1U);
    EXPECT_TRUE(state->end_of_stream_received);
    (*session)->cancel();
    ASSERT_TRUE((*session)->join());
  }
}

// A final transported record cannot remain staged ahead of EOS merely because the previous
// record owns the capacity-one ReuseLast slot. Replacing that exact obsolete reuse record must
// preserve data-before-terminal order for every source-private payload family.
// REQ: CHAN-136
// REQ: RUNTIME-050
[[nodiscard]] expected<void, Error> receive_until_progress(RuntimeChannelSession& session,
                                                           const PortId input) {
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (std::chrono::steady_clock::now() < deadline) {
    auto received = RuntimeChannelSessionTestAccess::receive_one(session, input);
    if (received || (received.error().code != ErrorCode::kNotReady &&
                     received.error().code != ErrorCode::kBackpressured)) {
      return received;
    }
    std::this_thread::sleep_for(1ms);
  }
  return make_unexpected(Error{ErrorCode::kTimeout});
}

TEST(ChannelBindingParityTest, InlineFinalDataReplacesDepthOneReuseLastBeforeEndOfStream) {
  SessionFixture fixture;
  fixture.use_one_slot_blocking_transport();
  fixture.enable_reuse_last();
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding = (*session)->create_readiness_binding(
      VertexId{2U},
      RuntimePortBindingRequest{.plan_generation = fixture.plan.plan_generation,
                                .run_generation = 1U,
                                .kind = RuntimeEventKind::kInputArrival,
                                .source = make_source_id(1U),
                                .source_clock = ClockId{1U},
                                .reference_clock = ClockId{1U},
                                .capacity = 2U});
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());

  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(17));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
  auto first_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(first_event) << runtime_error_message(first_event.error());
  auto first_record = (*session)->temporal_sample(first_event->port, first_event->source_sequence);
  ASSERT_TRUE(first_record) << runtime_error_message(first_record.error());
  const std::array first_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&*first_record, 1U},
      .destructive = true}};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, first_selection));
  {
    auto first_sample = fixture.sink.input.receive();
    ASSERT_TRUE(first_sample) << runtime_error_message(first_sample.error());
    EXPECT_EQ(first_sample->data, 17);
    EXPECT_EQ(first_sample->metadata.sequence, 1U);
  }
  ASSERT_TRUE((*binding)->retire(first_event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  const auto selector_retirement =
      (*session)->retire_samples(PortId{2U}, first_selection.front().samples);
  ASSERT_FALSE(selector_retirement);
  EXPECT_EQ(selector_retirement.error().code, ErrorCode::kBackpressured);

  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 2U, {}));
  ASSERT_TRUE(fixture.source.output.emit(23));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 2U, true));
  const auto staged = RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U});
  ASSERT_FALSE(staged);
  EXPECT_EQ(staged.error().code, ErrorCode::kBackpressured);
  ASSERT_TRUE(
      RuntimeChannelSessionTestAccess::fail_next_reuse_replacement_receive(**session, PortId{2U}));
  const auto forced_failure = (*session)->replace_staged_reuse(PortId{2U}, *first_record);
  ASSERT_FALSE(forced_failure);
  EXPECT_EQ(forced_failure.error().code, ErrorCode::kFailure);
  const auto rolled_back =
      RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(rolled_back) << runtime_error_message(rolled_back.error());
  EXPECT_TRUE(rolled_back->reuse_retained);
  EXPECT_TRUE(rolled_back->temporal_retained);
  EXPECT_TRUE(rolled_back->reuse_slot_matches_single_occupied);
  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  const auto selector_owned = RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U});
  ASSERT_FALSE(selector_owned);
  EXPECT_EQ(selector_owned.error().code, ErrorCode::kBackpressured);
  auto final_data = (*session)->replace_staged_reuse(PortId{2U}, *first_record);
  ASSERT_TRUE(final_data) << runtime_error_message(final_data.error());
  auto final_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(final_event) << runtime_error_message(final_event.error());
  auto final_record = (*session)->temporal_sample(final_event->port, final_event->source_sequence);
  ASSERT_TRUE(final_record) << runtime_error_message(final_record.error());
  EXPECT_EQ(final_record->source_sequence, 2U);
  const std::array final_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&*final_record, 1U},
      .destructive = true}};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 2U, final_selection));
  {
    auto final_sample = fixture.sink.input.receive();
    ASSERT_TRUE(final_sample) << runtime_error_message(final_sample.error());
    EXPECT_EQ(final_sample->data, 23);
    EXPECT_EQ(final_sample->metadata.sequence, 2U);
  }
  ASSERT_TRUE((*binding)->retire(final_event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 2U, true));
  auto terminal = receive_until_progress(**session, PortId{2U});
  ASSERT_TRUE(terminal) << runtime_error_message(terminal.error());
  auto terminal_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(terminal_event) << runtime_error_message(terminal_event.error());
  EXPECT_EQ(terminal_event->kind, RuntimeEventKind::kInputEndOfStream);
  ASSERT_TRUE((*binding)->retire(terminal_event->source_sequence));
  const auto state = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(state) << runtime_error_message(state.error());
  EXPECT_EQ(state->occupied_samples, 1U);
  EXPECT_TRUE(state->end_of_stream_received);
  EXPECT_EQ(state->last_producer_sequence, 3U);
  EXPECT_EQ(state->end_of_stream_transport_sequence, 3U);
  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  const auto repeated = RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U});
  if (!repeated) {
    EXPECT_TRUE(repeated.error().code == ErrorCode::kClosed ||
                repeated.error().code == ErrorCode::kNotReady);
  }
  const auto repeated_state =
      RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(repeated_state) << runtime_error_message(repeated_state.error());
  EXPECT_EQ(repeated_state->last_producer_sequence, 3U);
  EXPECT_EQ(repeated_state->end_of_stream_transport_sequence, 3U);
  RuntimeEvent duplicate{};
  auto duplicate_count = (*binding)->drain(std::span<RuntimeEvent>{&duplicate, 1U});
  ASSERT_TRUE(duplicate_count) << runtime_error_message(duplicate_count.error());
  EXPECT_EQ(*duplicate_count, 0U);
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// REQ: CHAN-136
// REQ: RUNTIME-050
TEST(ChannelBindingParityTest, RawTensorFinalDataReplacesDepthOneReuseLastBeforeEndOfStream) {
  TensorSessionFixture fixture;
  fixture.use_one_slot_blocking_transport();
  fixture.enable_reuse_last();
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding = (*session)->create_readiness_binding(
      VertexId{2U},
      RuntimePortBindingRequest{.plan_generation = fixture.base.plan.plan_generation,
                                .run_generation = 1U,
                                .kind = RuntimeEventKind::kInputArrival,
                                .source = make_source_id(1U),
                                .source_clock = ClockId{1U},
                                .reference_clock = ClockId{1U},
                                .capacity = 2U});
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());

  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  auto first = make_owned_tensor_sample(0x39U);
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  ASSERT_TRUE(fixture.source.output.emit(std::move(first->tensor)));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
  auto first_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(first_event) << runtime_error_message(first_event.error());
  auto first_record = (*session)->temporal_sample(first_event->port, first_event->source_sequence);
  ASSERT_TRUE(first_record) << runtime_error_message(first_record.error());
  const std::array first_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&*first_record, 1U},
      .destructive = true}};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, first_selection));
  ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 1U, nullptr));
  {
    auto first_sample = fixture.sink.input.receive();
    ASSERT_TRUE(first_sample) << runtime_error_message(first_sample.error());
    ASSERT_NE(first_sample->data.data_as<std::uint64_t>(), nullptr);
    EXPECT_EQ(*first_sample->data.data_as<std::uint64_t>(), 0x39U);
    EXPECT_EQ(first_sample->metadata.sequence, 1U);
  }
  ASSERT_TRUE((*binding)->retire(first_event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  const auto selector_retirement =
      (*session)->retire_samples(PortId{2U}, first_selection.front().samples);
  ASSERT_FALSE(selector_retirement);
  EXPECT_EQ(selector_retirement.error().code, ErrorCode::kBackpressured);

  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 2U, {}));
  auto final = make_owned_tensor_sample(0x48U);
  ASSERT_TRUE(final) << runtime_error_message(final.error());
  ASSERT_TRUE(fixture.source.output.emit(std::move(final->tensor)));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 2U, true));
  const auto staged = RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U});
  ASSERT_FALSE(staged);
  EXPECT_EQ(staged.error().code, ErrorCode::kBackpressured);
  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  auto final_data = (*session)->replace_staged_reuse(PortId{2U}, *first_record);
  ASSERT_TRUE(final_data) << runtime_error_message(final_data.error());
  auto final_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(final_event) << runtime_error_message(final_event.error());
  auto final_record = (*session)->temporal_sample(final_event->port, final_event->source_sequence);
  ASSERT_TRUE(final_record) << runtime_error_message(final_record.error());
  EXPECT_EQ(final_record->source_sequence, 2U);
  const std::array final_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&*final_record, 1U},
      .destructive = true}};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, final_selection));
  ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 1U, nullptr));
  auto final_sample = fixture.sink.input.receive();
  ASSERT_TRUE(final_sample) << runtime_error_message(final_sample.error());
  ASSERT_NE(final_sample->data.data_as<std::uint64_t>(), nullptr);
  EXPECT_EQ(*final_sample->data.data_as<std::uint64_t>(), 0x48U);
  EXPECT_EQ(final_sample->metadata.sequence, 2U);
  ASSERT_TRUE((*binding)->retire(final_event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  auto terminal = receive_until_progress(**session, PortId{2U});
  ASSERT_TRUE(terminal) << runtime_error_message(terminal.error());
  auto terminal_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(terminal_event) << runtime_error_message(terminal_event.error());
  EXPECT_EQ(terminal_event->kind, RuntimeEventKind::kInputEndOfStream);
  ASSERT_TRUE((*binding)->retire(terminal_event->source_sequence));
  const auto state = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(state) << runtime_error_message(state.error());
  EXPECT_EQ(state->occupied_samples, 1U);
  EXPECT_TRUE(state->end_of_stream_received);
  EXPECT_EQ(state->last_producer_sequence, 3U);
  EXPECT_EQ(state->end_of_stream_transport_sequence, 3U);
  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  const auto repeated = RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U});
  if (!repeated) {
    EXPECT_TRUE(repeated.error().code == ErrorCode::kClosed ||
                repeated.error().code == ErrorCode::kNotReady);
  }
  const auto repeated_state =
      RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(repeated_state) << runtime_error_message(repeated_state.error());
  EXPECT_EQ(repeated_state->last_producer_sequence, 3U);
  EXPECT_EQ(repeated_state->end_of_stream_transport_sequence, 3U);
  RuntimeEvent duplicate{};
  auto duplicate_count = (*binding)->drain(std::span<RuntimeEvent>{&duplicate, 1U});
  ASSERT_TRUE(duplicate_count) << runtime_error_message(duplicate_count.error());
  EXPECT_EQ(*duplicate_count, 0U);
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// REQ: CHAN-136
// REQ: RUNTIME-050
TEST(ChannelBindingParityTest,
     DescriptorSchemaFinalDataReplacesDepthOneReuseLastBeforeEndOfStream) {
  ImageSessionFixture fixture;
  fixture.use_one_slot_blocking_transport();
  fixture.enable_reuse_last();
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding = (*session)->create_readiness_binding(
      VertexId{2U},
      RuntimePortBindingRequest{.plan_generation = fixture.base.plan.plan_generation,
                                .run_generation = 1U,
                                .kind = RuntimeEventKind::kInputArrival,
                                .source = make_source_id(1U),
                                .source_clock = ClockId{1U},
                                .reference_clock = ClockId{1U},
                                .capacity = 2U});
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  const auto emit_image = [&](const std::uint64_t value, const std::string& frame) {
    auto owned = make_owned_tensor_sample(value);
    if (!owned) {
      return expected<void, Error>{make_unexpected(std::move(owned).error())};
    }
    schema::ImageT image{};
    image.height = 1;
    image.width = 1;
    image.encoding = schema::ImageEncoding_CUSTOM;
    image.header = std::make_shared<schema::HeaderT>();
    image.header->frame_id = frame;
    image.data = std::make_shared<Tensor>(std::move(owned->tensor));
    return fixture.source.output.emit(std::move(image));
  };

  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(emit_image(0x39U, "retained"));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
  auto first_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(first_event) << runtime_error_message(first_event.error());
  auto first_record = (*session)->temporal_sample(first_event->port, first_event->source_sequence);
  ASSERT_TRUE(first_record) << runtime_error_message(first_record.error());
  const std::array first_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&*first_record, 1U},
      .destructive = true}};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, first_selection));
  ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 1U, nullptr));
  {
    auto first_sample = fixture.sink.input.receive();
    ASSERT_TRUE(first_sample) << runtime_error_message(first_sample.error());
    ASSERT_NE(first_sample->data.data, nullptr);
    ASSERT_NE(first_sample->data.data->data_as<std::uint64_t>(), nullptr);
    EXPECT_EQ(*first_sample->data.data->data_as<std::uint64_t>(), 0x39U);
    ASSERT_NE(first_sample->data.header, nullptr);
    EXPECT_EQ(first_sample->data.header->frame_id, "retained");
    EXPECT_EQ(first_sample->metadata.sequence, 1U);
  }
  ASSERT_TRUE((*binding)->retire(first_event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  const auto selector_retirement =
      (*session)->retire_samples(PortId{2U}, first_selection.front().samples);
  ASSERT_FALSE(selector_retirement);
  EXPECT_EQ(selector_retirement.error().code, ErrorCode::kBackpressured);

  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 2U, {}));
  ASSERT_TRUE(emit_image(0x48U, "final"));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 2U, true));
  const auto staged = RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U});
  ASSERT_FALSE(staged);
  EXPECT_EQ(staged.error().code, ErrorCode::kBackpressured);
  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  auto final_data = (*session)->replace_staged_reuse(PortId{2U}, *first_record);
  ASSERT_TRUE(final_data) << runtime_error_message(final_data.error());
  auto final_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(final_event) << runtime_error_message(final_event.error());
  auto final_record = (*session)->temporal_sample(final_event->port, final_event->source_sequence);
  ASSERT_TRUE(final_record) << runtime_error_message(final_record.error());
  EXPECT_EQ(final_record->source_sequence, 2U);
  const std::array final_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&*final_record, 1U},
      .destructive = true}};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, final_selection));
  ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 1U, nullptr));
  auto final_sample = fixture.sink.input.receive();
  ASSERT_TRUE(final_sample) << runtime_error_message(final_sample.error());
  ASSERT_NE(final_sample->data.data, nullptr);
  ASSERT_NE(final_sample->data.data->data_as<std::uint64_t>(), nullptr);
  EXPECT_EQ(*final_sample->data.data->data_as<std::uint64_t>(), 0x48U);
  ASSERT_NE(final_sample->data.header, nullptr);
  EXPECT_EQ(final_sample->data.header->frame_id, "final");
  EXPECT_EQ(final_sample->metadata.sequence, 2U);
  ASSERT_TRUE((*binding)->retire(final_event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  auto terminal = receive_until_progress(**session, PortId{2U});
  ASSERT_TRUE(terminal) << runtime_error_message(terminal.error());
  auto terminal_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(terminal_event) << runtime_error_message(terminal_event.error());
  EXPECT_EQ(terminal_event->kind, RuntimeEventKind::kInputEndOfStream);
  ASSERT_TRUE((*binding)->retire(terminal_event->source_sequence));
  const auto state = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(state) << runtime_error_message(state.error());
  EXPECT_EQ(state->occupied_samples, 1U);
  EXPECT_TRUE(state->end_of_stream_received);
  EXPECT_EQ(state->last_producer_sequence, 3U);
  EXPECT_EQ(state->end_of_stream_transport_sequence, 3U);
  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  const auto repeated = RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U});
  if (!repeated) {
    EXPECT_TRUE(repeated.error().code == ErrorCode::kClosed ||
                repeated.error().code == ErrorCode::kNotReady);
  }
  const auto repeated_state =
      RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(repeated_state) << runtime_error_message(repeated_state.error());
  EXPECT_EQ(repeated_state->last_producer_sequence, 3U);
  EXPECT_EQ(repeated_state->end_of_stream_transport_sequence, 3U);
  RuntimeEvent duplicate{};
  auto duplicate_count = (*binding)->drain(std::span<RuntimeEvent>{&duplicate, 1U});
  ASSERT_TRUE(duplicate_count) << runtime_error_message(duplicate_count.error());
  EXPECT_EQ(*duplicate_count, 0U);
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// Raw Tensor publication uses the same graph-owned finite-pressure policy as scalar payloads, but
// additionally transfers one backing claim into the transport release callback. A rejected
// publication must retire only its own claim and must not consume a logical sequence.
// REQ: CHAN-083
// REQ: CHAN-101
// REQ: CHAN-102
// REQ: RUNTIME-019
// REQ: TENSOR-STORAGE-010
TEST(ChannelBindingParityTest, TensorBlockingPressureRetiresRejectedBackingAndPreservesRetry) {
  TensorSessionFixture fixture;
  fixture.use_one_slot_blocking_transport();
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.base.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 2U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

  auto first = make_owned_tensor_sample(11U);
  auto rejected = make_owned_tensor_sample(22U);
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  ASSERT_TRUE(rejected) << runtime_error_message(rejected.error());
  auto first_releases = first->releases;
  auto rejected_releases = rejected->releases;
  ASSERT_TRUE(fixture.source.output.emit(std::move(first->tensor)));

  std::promise<void> entered_emit;
  auto entered = entered_emit.get_future();
  auto pressure = std::async(
      std::launch::async,
      [&fixture, tensor = std::move(rejected->tensor), signal = std::move(entered_emit)]() mutable {
        signal.set_value();
        return fixture.source.output.emit(std::move(tensor));
      });
  auto pressure_cleanup = ScopeExit{[&]() noexcept {
    if (!pressure.valid()) {
      return;
    }
    if (pressure.wait_for(0s) != std::future_status::ready) {
      static_cast<void>((*session)->arm());
    }
    (*session)->cancel();
    static_cast<void>((*session)->join());
    pressure.wait();
  }};
  ASSERT_EQ(entered.wait_for(2s), std::future_status::ready);
  if (pressure.wait_for(1s) != std::future_status::ready) {
    ASSERT_TRUE((*session)->arm());
    ASSERT_EQ(pressure.wait_for(2s), std::future_status::ready);
    static_cast<void>(pressure.get());
    (*session)->cancel();
    ASSERT_TRUE((*session)->join());
    FAIL() << "raw Tensor publication waited for full recipient capacity";
  }
  const auto pressure_result = pressure.get();
  pressure_cleanup.release();
  ASSERT_FALSE(pressure_result);
  EXPECT_EQ(pressure_result.error().code, ErrorCode::kBackpressured);
  EXPECT_EQ(first_releases->load(std::memory_order_relaxed), 0U);
  EXPECT_EQ(rejected_releases->load(std::memory_order_relaxed), 1U);

  ASSERT_TRUE((*session)->arm());
  auto first_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(first_event) << runtime_error_message(first_event.error());
  auto first_record = (*session)->temporal_sample(first_event->port, first_event->source_sequence);
  ASSERT_TRUE(first_record) << runtime_error_message(first_record.error());
  const std::array first_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&*first_record, 1U},
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, first_selection));
  ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 1U, nullptr));
  {
    auto sample = fixture.sink.input.receive();
    ASSERT_TRUE(sample) << runtime_error_message(sample.error());
    ASSERT_NE(sample->data.data_as<std::uint64_t>(), nullptr);
    EXPECT_EQ(*sample->data.data_as<std::uint64_t>(), 11U);
    EXPECT_EQ(sample->metadata.sequence, 1U);
  }
  ASSERT_TRUE((*binding)->retire(first_event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  ASSERT_TRUE((*session)->retire_samples(
      PortId{2U}, std::span<const TemporalSampleRecord>{&*first_record, 1U}));
  EXPECT_EQ(first_releases->load(std::memory_order_relaxed), 1U);

  auto retry = make_owned_tensor_sample(33U);
  ASSERT_TRUE(retry) << runtime_error_message(retry.error());
  auto retry_releases = retry->releases;
  ASSERT_TRUE(fixture.source.output.emit(std::move(retry->tensor)));
  auto retry_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(retry_event) << runtime_error_message(retry_event.error());
  auto retry_record = (*session)->temporal_sample(retry_event->port, retry_event->source_sequence);
  ASSERT_TRUE(retry_record) << runtime_error_message(retry_record.error());
  EXPECT_EQ(retry_record->source_sequence, 2U);
  const std::array retry_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&*retry_record, 1U},
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 2U, retry_selection));
  ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 2U, nullptr));
  {
    auto sample = fixture.sink.input.receive();
    ASSERT_TRUE(sample) << runtime_error_message(sample.error());
    ASSERT_NE(sample->data.data_as<std::uint64_t>(), nullptr);
    EXPECT_EQ(*sample->data.data_as<std::uint64_t>(), 33U);
    EXPECT_EQ(sample->metadata.sequence, 2U);
  }
  ASSERT_TRUE((*binding)->retire(retry_event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 2U, true));
  ASSERT_TRUE((*session)->retire_samples(
      PortId{2U}, std::span<const TemporalSampleRecord>{&*retry_record, 1U}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
  EXPECT_EQ(retry_releases->load(std::memory_order_relaxed), 1U);
}

// Real HoloMQ fanout can accept one recipient before discovering that another recipient remains
// full. Runtime must classify that result as indeterminate, consume the logical sequence, and
// retain the shared Tensor backing until the accepting recipient retires its exact claim.
// REQ: CHAN-076
// REQ: CHAN-083
// REQ: CHAN-101
// REQ: RUNTIME-019
// REQ: TENSOR-STORAGE-010
TEST(ChannelBindingParityTest, TensorPartialFanoutPressureRetainsAcceptedRecipientBacking) {
  TensorSessionFixture fixture;
  fixture.use_one_slot_blocking_fanout();
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  const auto bind = [&](const VertexId owner, const SourceId source) {
    return (*session)->create_readiness_binding(
        owner,
        RuntimePortBindingRequest{
            .plan_generation = fixture.base.plan.plan_generation,
            .run_generation = 1U,
            .kind = RuntimeEventKind::kInputArrival,
            .source = source,
            .source_clock = ClockId{1U},
            .reference_clock = ClockId{1U},
            .capacity = 1U,
        });
  };
  auto first_binding = bind(VertexId{2U}, make_source_id(1U));
  auto second_binding = bind(VertexId{3U}, make_source_id(2U));
  ASSERT_TRUE(first_binding) << runtime_error_message(first_binding.error());
  ASSERT_TRUE(second_binding) << runtime_error_message(second_binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

  const auto receive_first = [&](const RuntimeEvent& event,
                                 const std::uint64_t activation_generation,
                                 const std::uint64_t expected_value,
                                 const std::uint64_t expected_sequence) -> expected<void, Error> {
    auto record = (*session)->temporal_sample(event.port, event.source_sequence);
    if (!record) {
      return make_unexpected(std::move(record).error());
    }
    const std::array selection{RuntimeChannelActivationSelection{
        .port = PortId{2U},
        .selection = {InputState::kSelected, SelectionReason::kNone},
        .samples = std::span<const TemporalSampleRecord>{&*record, 1U},
        .destructive = true,
    }};
    if (auto prepared =
            (*session)->prepare_activation(VertexId{2U}, activation_generation, selection);
        !prepared) {
      return make_unexpected(std::move(prepared).error());
    }
    if (auto access =
            (*session)->prepare_tensor_access(VertexId{2U}, activation_generation, nullptr);
        !access) {
      return make_unexpected(std::move(access).error());
    }
    auto sample = fixture.sink.input.receive();
    if (!sample || sample->data.data_as<std::uint64_t>() == nullptr ||
        *sample->data.data_as<std::uint64_t>() != expected_value ||
        sample->metadata.sequence != expected_sequence) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    if (auto retired = (*first_binding)->retire(event.source_sequence); !retired) {
      return make_unexpected(std::move(retired).error());
    }
    if (auto retired = (*session)->retire_activation(VertexId{2U}, activation_generation, true);
        !retired) {
      return make_unexpected(std::move(retired).error());
    }
    return (*session)->retire_samples(PortId{2U},
                                      std::span<const TemporalSampleRecord>{&*record, 1U});
  };

  auto first = make_owned_tensor_sample(11U);
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  auto first_releases = first->releases;
  ASSERT_TRUE(fixture.source.output.emit(std::move(first->tensor)));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{3U}));
  auto first_event = wait_for_event(**session, **first_binding);
  auto held_second_event = wait_for_event(**session, **second_binding);
  ASSERT_TRUE(first_event) << runtime_error_message(first_event.error());
  ASSERT_TRUE(held_second_event) << runtime_error_message(held_second_event.error());
  auto received_first = receive_first(*first_event, 1U, 11U, 1U);
  ASSERT_TRUE(received_first) << runtime_error_message(received_first.error());
  // The second recipient still owns the first publication in its typed retention slot.
  EXPECT_EQ(first_releases->load(std::memory_order_relaxed), 0U);

  auto queued_second = make_owned_tensor_sample(22U);
  ASSERT_TRUE(queued_second) << runtime_error_message(queued_second.error());
  auto second_releases = queued_second->releases;
  ASSERT_TRUE(fixture.source.output.emit(std::move(queued_second->tensor)));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
  auto second_recipient_staged =
      RuntimeChannelSessionTestAccess::receive_one(**session, PortId{3U});
  ASSERT_FALSE(second_recipient_staged);
  EXPECT_EQ(second_recipient_staged.error().code, ErrorCode::kBackpressured);
  auto second_event = wait_for_event(**session, **first_binding);
  ASSERT_TRUE(second_event) << runtime_error_message(second_event.error());

  // A full typed slot no longer implies a full transport queue: the bounded endpoint transaction
  // above stages sequence two for the second recipient so terminal control remains observable. Fill
  // that recipient's transport queue with sequence three, then stage the same sequence for the
  // first recipient. Sequence four now crosses a deterministic partial-fanout visibility boundary:
  // first accepts it into an empty queue and second rejects it behind typed + staged + queued data.
  auto staged = make_owned_tensor_sample(33U);
  ASSERT_TRUE(staged) << runtime_error_message(staged.error());
  auto staged_releases = staged->releases;
  ASSERT_TRUE(fixture.source.output.emit(std::move(staged->tensor)));
  auto first_recipient_staged = RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U});
  ASSERT_FALSE(first_recipient_staged);
  EXPECT_EQ(first_recipient_staged.error().code, ErrorCode::kBackpressured);

  auto partial = make_owned_tensor_sample(44U);
  ASSERT_TRUE(partial) << runtime_error_message(partial.error());
  auto partial_releases = partial->releases;
  auto partially_published = fixture.source.output.emit(std::move(partial->tensor));
  ASSERT_FALSE(partially_published);
  EXPECT_EQ(partially_published.error().code, ErrorCode::kPublicationIndeterminate);
  EXPECT_EQ(partial_releases->load(std::memory_order_relaxed), 0U);

  auto received_second = receive_first(*second_event, 2U, 22U, 2U);
  ASSERT_TRUE(received_second) << runtime_error_message(received_second.error());
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
  auto staged_event = wait_for_event(**session, **first_binding);
  ASSERT_TRUE(staged_event) << runtime_error_message(staged_event.error());
  auto received_staged = receive_first(*staged_event, 3U, 33U, 3U);
  ASSERT_TRUE(received_staged) << runtime_error_message(received_staged.error());
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
  auto partial_event = wait_for_event(**session, **first_binding);
  ASSERT_TRUE(partial_event) << runtime_error_message(partial_event.error());
  auto received_partial = receive_first(*partial_event, 4U, 44U, 4U);
  ASSERT_TRUE(received_partial) << runtime_error_message(received_partial.error());
  // The accepting recipient has retired its exact claim, but the indeterminate publication may
  // conservatively retain unresolved peer authority until teardown. It must never release sooner.
  EXPECT_EQ(partial_releases->load(std::memory_order_relaxed), 0U);

  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
  first_binding->reset();
  second_binding->reset();
  session->reset();
  EXPECT_EQ(first_releases->load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(second_releases->load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(staged_releases->load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(partial_releases->load(std::memory_order_relaxed), 1U);
}

// Generated values with an embedded Tensor use the same finite graph-owned pressure policy as raw
// Tensor payloads. A rejected schema envelope must retire only its own embedded backing, while a
// later retry remains the second logical publication and reconstructs the owning value normally.
// REQ: CHAN-084
// REQ: CHAN-119
// REQ: CHAN-101
// REQ: CHAN-102
// REQ: RUNTIME-019
// REQ: TENSOR-SCHEMA-005
// REQ: TENSOR-STORAGE-010
TEST(ChannelBindingParityTest,
     DescriptorTensorImageBlockingPressureRetiresRejectedBackingAndPreservesRetry) {
  ImageSessionFixture fixture;
  fixture.use_one_slot_blocking_transport();
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.base.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 2U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

  const auto make_image = [](std::uint64_t value, std::string frame_id)
      -> expected<std::pair<schema::ImageT, std::shared_ptr<std::atomic<std::uint32_t>>>, Error> {
    auto owned = make_owned_tensor_sample(value);
    if (!owned) {
      return make_unexpected(std::move(owned).error());
    }
    schema::ImageT image{};
    image.height = 1;
    image.width = 1;
    image.encoding = schema::ImageEncoding_CUSTOM;
    image.header = std::make_shared<schema::HeaderT>();
    image.header->frame_id = std::move(frame_id);
    image.data = std::make_shared<Tensor>(std::move(owned->tensor));
    return std::pair{std::move(image), std::move(owned->releases)};
  };
  const auto emit_image = [&fixture](schema::ImageT image, std::int64_t capture_timestamp_ns) {
    return fixture.source.output.emit(Sample<schema::ImageT>{
        std::move(image),
        SampleMetadata{.capture_timestamp_ns = capture_timestamp_ns,
                       .source_clock_id = ClockId{1U}},
    });
  };

  auto first = make_image(11U, "first_image");
  auto rejected = make_image(22U, "rejected_image");
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  ASSERT_TRUE(rejected) << runtime_error_message(rejected.error());
  auto first_releases = first->second;
  auto rejected_releases = rejected->second;
  ASSERT_TRUE(emit_image(std::move(first->first), 10));

  std::promise<void> entered_emit;
  auto entered = entered_emit.get_future();
  auto pressure = std::async(std::launch::async,
                             [image = std::move(rejected->first),
                              &emit_image,
                              signal = std::move(entered_emit)]() mutable {
                               signal.set_value();
                               return emit_image(std::move(image), 20);
                             });
  auto pressure_cleanup = ScopeExit{[&]() noexcept {
    if (!pressure.valid()) {
      return;
    }
    if (pressure.wait_for(0s) != std::future_status::ready) {
      static_cast<void>((*session)->arm());
    }
    (*session)->cancel();
    static_cast<void>((*session)->join());
    pressure.wait();
  }};
  ASSERT_EQ(entered.wait_for(2s), std::future_status::ready);
  if (pressure.wait_for(1s) != std::future_status::ready) {
    ASSERT_TRUE((*session)->arm());
    ASSERT_EQ(pressure.wait_for(2s), std::future_status::ready);
    static_cast<void>(pressure.get());
    (*session)->cancel();
    ASSERT_TRUE((*session)->join());
    FAIL() << "descriptor-Tensor publication waited for full recipient capacity";
  }
  const auto pressure_result = pressure.get();
  pressure_cleanup.release();
  ASSERT_FALSE(pressure_result);
  EXPECT_EQ(pressure_result.error().code, ErrorCode::kBackpressured);
  EXPECT_EQ(first_releases->load(std::memory_order_relaxed), 0U);
  EXPECT_EQ(rejected_releases->load(std::memory_order_relaxed), 1U);

  ASSERT_TRUE((*session)->arm());
  const auto receive_image = [&](std::uint64_t activation_generation,
                                 std::uint64_t expected_value,
                                 std::string_view expected_frame_id,
                                 std::uint64_t expected_sequence) {
    auto event = wait_for_event(**session, **binding);
    ASSERT_TRUE(event) << runtime_error_message(event.error());
    auto record = (*session)->temporal_sample(event->port, event->source_sequence);
    ASSERT_TRUE(record) << runtime_error_message(record.error());
    EXPECT_EQ(record->source_sequence, expected_sequence);
    const std::array selection{RuntimeChannelActivationSelection{
        .port = PortId{2U},
        .selection = {InputState::kSelected, SelectionReason::kNone},
        .samples = std::span<const TemporalSampleRecord>{&*record, 1U},
        .destructive = true,
    }};
    ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, activation_generation, selection));
    ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, activation_generation, nullptr));
    {
      auto sample = fixture.sink.input.receive();
      ASSERT_TRUE(sample) << runtime_error_message(sample.error());
      ASSERT_NE(sample->data.data, nullptr);
      ASSERT_NE(sample->data.data->data_as<std::uint64_t>(), nullptr);
      EXPECT_EQ(*sample->data.data->data_as<std::uint64_t>(), expected_value);
      ASSERT_NE(sample->data.header, nullptr);
      EXPECT_EQ(sample->data.header->frame_id, expected_frame_id);
      EXPECT_EQ(sample->metadata.sequence, expected_sequence);
    }
    ASSERT_TRUE((*binding)->retire(event->source_sequence));
    ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, activation_generation, true));
    ASSERT_TRUE((*session)->retire_samples(PortId{2U},
                                           std::span<const TemporalSampleRecord>{&*record, 1U}));
  };
  receive_image(1U, 11U, "first_image", 1U);
  EXPECT_EQ(first_releases->load(std::memory_order_relaxed), 1U);

  auto retry = make_image(33U, "retry_image");
  ASSERT_TRUE(retry) << runtime_error_message(retry.error());
  auto retry_releases = retry->second;
  ASSERT_TRUE(emit_image(std::move(retry->first), 30));
  receive_image(2U, 33U, "retry_image", 2U);

  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
  EXPECT_EQ(retry_releases->load(std::memory_order_relaxed), 1U);
}

// A descriptor-schema assignment validates the envelope and generated metadata before committing
// into its persistent typed slot. A failed drop-oldest replacement must therefore preserve the
// valid incumbent rather than exposing an empty cache. REQ: CHAN-084 REQ: RUNTIME-026
TEST(ChannelBindingParityTest, DescriptorTensorImageFailedReplacementPreservesRetainedSample) {
  // Key setup: retain one Image in a capacity-one drop-oldest cache, then make the assignment hook
  // mutate an ordinary field before rejecting the second producer sequence.
  ImageSessionFixture fixture;
  fixture.base.use_one_slot_drop_oldest_queue();
  RuntimePortTypeOps rejecting_ops = *fixture.replay[1].type_ops;
  rejecting_ops.assign_descriptor_tensors_sample = &reject_second_image_assignment;
  fixture.replay[1].type_ops = &rejecting_ops;
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE((*session)->refresh_connections());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.base.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 2U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());

  const auto emit_image = [&fixture](const std::uint64_t value,
                                     const std::int64_t capture_timestamp_ns) {
    auto owned = make_owned_tensor_sample(value);
    if (!owned) {
      return expected<void, Error>{make_unexpected(std::move(owned).error())};
    }
    schema::ImageT image{};
    image.height = 1;
    image.width = 1;
    image.encoding = schema::ImageEncoding_CUSTOM;
    image.header = std::make_shared<schema::HeaderT>();
    image.header->frame_id = "transactional-image";
    image.data = std::make_shared<Tensor>(std::move(owned->tensor));
    auto emitted = fixture.source.output.emit(Sample<schema::ImageT>{
        std::move(image),
        SampleMetadata{.capture_timestamp_ns = capture_timestamp_ns,
                       .source_clock_id = ClockId{1U}},
    });
    if (!emitted) {
      return expected<void, Error>{make_unexpected(std::move(emitted).error())};
    }
    return expected<void, Error>{};
  };

  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(emit_image(11U, 10));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 2U, {}));
  ASSERT_TRUE(emit_image(22U, 20));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 2U, true));
  auto rejected = RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U});
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kProtocolError);

  // Key assertion: sequence one and its exact Tensor value remain selectable after sequence two
  // fails; the typed cache never exposes the destructive pre-assignment gap.
  const auto retained = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(retained) << runtime_error_message(retained.error());
  EXPECT_EQ(retained->occupied_samples, 1U);
  EXPECT_EQ(retained->next_arrival_sequence, 2U);
  ASSERT_TRUE((*binding)->poll(TimePoint{1, ClockId{1U}}));
  std::array<RuntimeEvent, 1U> events{};
  auto drained = (*binding)->drain(events);
  ASSERT_TRUE(drained) << runtime_error_message(drained.error());
  ASSERT_EQ(*drained, 1U);
  auto record = (*session)->temporal_sample(events.front().port, events.front().source_sequence);
  ASSERT_TRUE(record) << runtime_error_message(record.error());
  const std::array selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&*record, 1U},
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selection));
  ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 1U, nullptr));
  auto sample = fixture.sink.input.receive();
  ASSERT_TRUE(sample) << runtime_error_message(sample.error());
  ASSERT_NE(sample->data.data, nullptr);
  ASSERT_NE(sample->data.data->data_as<std::uint64_t>(), nullptr);
  EXPECT_EQ(*sample->data.data->data_as<std::uint64_t>(), 11U);
  ASSERT_NE(sample->data.header, nullptr);
  EXPECT_EQ(sample->data.header->frame_id, "transactional-image");
  EXPECT_EQ(sample->metadata.sequence, 1U);
  ASSERT_TRUE((*binding)->retire(events.front().source_sequence));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// A controller activation cut may drain a raw-Tensor descriptor on a thread other than the
// background pump. Cold initialization must consume the exact frozen ordinal, restore the
// controller's ambient device, cache a same-thread repeat, and reject an unavailable ordinal.
// The exec-child matrix separately proves that the initialized import and callback expose the
// correct bytes on device zero and on a nonzero device when the host provides one.
// REQ: CHAN-103
TEST(ChannelBindingParityTest,
     ControllerDescriptorImportInitializesExactCudaDeviceAndRestoresCaller) {
  test::skip_without_cuda_device();

  int device_count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
  ASSERT_GT(device_count, 0);
  int ambient_device = -1;
  ASSERT_EQ(cudaGetDevice(&ambient_device), cudaSuccess);
  const int frozen_device = device_count > 1 ? (ambient_device == 0 ? 1 : 0) : ambient_device;

  TensorSessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::initialize_controller_descriptor_import(
      **session, PortId{2U}, DeviceId{frozen_device}));
  auto initialization_count =
      RuntimeChannelSessionTestAccess::controller_descriptor_import_initialization_count(**session);
  ASSERT_TRUE(initialization_count);
  EXPECT_EQ(*initialization_count, 1U);
  int restored_device = -1;
  ASSERT_EQ(cudaGetDevice(&restored_device), cudaSuccess);
  EXPECT_EQ(restored_device, ambient_device);

  // A repeated cut on the same sequential controller thread uses the cold-bound result and must
  // leave the caller unchanged.
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::initialize_controller_descriptor_import(
      **session, PortId{2U}, DeviceId{frozen_device}));
  initialization_count =
      RuntimeChannelSessionTestAccess::controller_descriptor_import_initialization_count(**session);
  ASSERT_TRUE(initialization_count);
  EXPECT_EQ(*initialization_count, 1U);
  ASSERT_EQ(cudaGetDevice(&restored_device), cudaSuccess);
  EXPECT_EQ(restored_device, ambient_device);

  if (device_count > 1) {
    const int replacement_device = frozen_device == 0 ? 1 : 0;
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::initialize_controller_descriptor_import(
        **session, PortId{2U}, DeviceId{replacement_device}));
    initialization_count =
        RuntimeChannelSessionTestAccess::controller_descriptor_import_initialization_count(
            **session);
    ASSERT_TRUE(initialization_count);
    EXPECT_EQ(*initialization_count, 2U);
    ASSERT_EQ(cudaGetDevice(&restored_device), cudaSuccess);
    EXPECT_EQ(restored_device, ambient_device);
  }

  TensorSessionFixture invalid_fixture;
  auto invalid_session =
      RuntimeChannelSession::create(invalid_fixture.base.plan, invalid_fixture.replay);
  ASSERT_TRUE(invalid_session) << runtime_error_message(invalid_session.error());
  auto unavailable = RuntimeChannelSessionTestAccess::initialize_controller_descriptor_import(
      **invalid_session, PortId{2U}, DeviceId{device_count});
  ASSERT_FALSE(unavailable);
  EXPECT_EQ(unavailable.error().code, ErrorCode::kNotSupported);
  ASSERT_EQ(cudaGetDevice(&restored_device), cudaSuccess);
  EXPECT_EQ(restored_device, ambient_device);
}

// A raw-Tensor scan can move from a device-pinned subscriber to a host/unspecified subscriber.
// The latter import belongs to the controller's ambient device, not the peer selected for the
// preceding endpoint, and the scan must leave that ambient selection intact for its caller.
// REQ: CHAN-103
TEST(ChannelBindingParityTest, HostDescriptorImportRestoresAmbientDeviceAfterPeerSelection) {
  test::skip_without_cuda_device();

  int device_count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
  if (device_count < 2) {
    GTEST_SKIP() << "peer-to-host descriptor selection requires two CUDA devices";
  }
  int ambient_device = -1;
  ASSERT_EQ(cudaGetDevice(&ambient_device), cudaSuccess);
  const int peer_device = ambient_device == 0 ? 1 : 0;

  auto observed = RuntimeChannelSessionTestAccess::cuda_device_after_import_selection(
      DeviceId{peer_device}, DeviceId{});
  ASSERT_TRUE(observed) << runtime_error_message(observed.error());
  EXPECT_EQ(*observed, DeviceId{ambient_device});

  int restored_device = -1;
  ASSERT_EQ(cudaGetDevice(&restored_device), cudaSuccess);
  EXPECT_EQ(restored_device, ambient_device);
}

// Event-wake-enabled raw Tensor and descriptor-Tensor services use typed Channel endpoints and
// join the same wait set as inline payloads instead of relying on the pump timeout.
// REQ: RUNTIME-032
TEST(RuntimeChannelBinding, PrebindsTypedTensorSubscribersIntoTheSharedWaitSet) {
  {
    TensorSessionFixture fixture;
    auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());

    // Key assertion: one raw-Tensor endpoint contributes one wait handle and one drain entry.
    const auto indexes = RuntimeChannelSessionTestAccess::pump_index_state(**session);
    ASSERT_TRUE(indexes) << runtime_error_message(indexes.error());
    EXPECT_TRUE(indexes->exact);
    EXPECT_EQ(indexes->wait_handle_count, 1U);
    EXPECT_EQ(indexes->drain_subscriber_count, 1U);
  }
  {
    ImageSessionFixture fixture;
    auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());

    // Key assertion: the descriptor-Tensor endpoint participates in the identical bounded index.
    const auto indexes = RuntimeChannelSessionTestAccess::pump_index_state(**session);
    ASSERT_TRUE(indexes) << runtime_error_message(indexes.error());
    EXPECT_TRUE(indexes->exact);
    EXPECT_EQ(indexes->wait_handle_count, 1U);
    EXPECT_EQ(indexes->drain_subscriber_count, 1U);
  }
}

// An exact same-partition Tensor edge needs no transport wake when the controller's serialized
// post-compute activation cut drains the typed endpoint before the next readiness freeze.
// REQ: CHAN-098
// REQ: CHAN-099
// REQ: RUNTIME-019
// REQ: RUNTIME-032
TEST(RuntimeChannelBinding, TensorActivationCutDrainsExactEventlessServices) {
  {
    // Key setup: publish a raw Tensor through an exact 1:1 HoloMQ service with no wait handle.
    TensorSessionFixture fixture;
    fixture.use_one_slot_blocking_transport();
    fixture.base.use_same_partition_eventless_transport();
    fixture.base.use_single_partition_runner();
    const auto counters = std::make_shared<RuntimePhaseCounters>();
    fixture.base.plan.executable_factories.phase_counters = counters;
    auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    auto binding = (*session)->create_readiness_binding(
        VertexId{2U},
        RuntimePortBindingRequest{
            .plan_generation = fixture.base.plan.plan_generation,
            .run_generation = 1U,
            .kind = RuntimeEventKind::kInputArrival,
            .source = make_source_id(1U),
            .source_clock = ClockId{1U},
            .reference_clock = ClockId{1U},
            .capacity = 1U,
        });
    ASSERT_TRUE(binding) << runtime_error_message(binding.error());
    ASSERT_TRUE((*session)->refresh_connections());
    ASSERT_TRUE((*session)->arm());
    auto indexes = RuntimeChannelSessionTestAccess::pump_index_state(**session);
    ASSERT_TRUE(indexes) << runtime_error_message(indexes.error());
    EXPECT_TRUE(indexes->exact);
    EXPECT_EQ(indexes->wait_handle_count, 0U);
    EXPECT_EQ(indexes->drain_subscriber_count, 1U);
    EXPECT_FALSE(indexes->background_thread_required);
    EXPECT_FALSE(indexes->background_thread_running);
    const auto controller_initializations =
        RuntimeChannelSessionTestAccess::controller_descriptor_import_initialization_count(
            **session);
    ASSERT_TRUE(controller_initializations)
        << runtime_error_message(controller_initializations.error());
    EXPECT_EQ(*controller_initializations, 1U);

    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
    auto owned = make_owned_tensor_sample(0xA11CEU);
    ASSERT_TRUE(owned) << runtime_error_message(owned.error());
    const void* const expected_data = owned->data;
    auto releases = owned->releases;
    ASSERT_TRUE(fixture.source.output.emit(std::move(owned->tensor)));
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
    ASSERT_TRUE((*session)->flush_process_local_activation_cut());

    // Key assertion: the ordinary cut publishes one typed Tensor readiness record and preserves
    // the backing identity and lifetime without a transport event attachment.
    auto event = wait_for_event(**session, **binding);
    ASSERT_TRUE(event) << runtime_error_message(event.error());
    auto record = (*session)->temporal_sample(event->port, event->source_sequence);
    ASSERT_TRUE(record) << runtime_error_message(record.error());
    const std::array selection{RuntimeChannelActivationSelection{
        .port = PortId{2U},
        .selection = {InputState::kSelected, SelectionReason::kNone},
        .samples = std::span<const TemporalSampleRecord>{&*record, 1U},
        .destructive = true,
    }};
    ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selection));
    ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 1U, nullptr));
    {
      auto received = fixture.sink.input.receive();
      ASSERT_TRUE(received) << runtime_error_message(received.error());
      EXPECT_EQ(received->data.data(), expected_data);
      ASSERT_NE(received->data.data_as<std::uint64_t>(), nullptr);
      EXPECT_EQ(*received->data.data_as<std::uint64_t>(), 0xA11CEU);
      EXPECT_EQ(releases->load(std::memory_order_relaxed), 0U);
    }
    ASSERT_TRUE((*binding)->retire(event->source_sequence));
    ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
    ASSERT_TRUE((*session)->retire_samples(PortId{2U}, selection.front().samples));
    EXPECT_EQ(releases->load(std::memory_order_relaxed), 1U);
    const auto snapshot = counters->snapshot();
    EXPECT_EQ(snapshot[RuntimePhaseCounters::index(RuntimePhaseCounter::kChannelActivationCuts)],
              1U);
    EXPECT_EQ(snapshot[RuntimePhaseCounters::index(RuntimePhaseCounter::kChannelSamplesReceived)],
              1U);
    EXPECT_EQ(
        snapshot[RuntimePhaseCounters::index(RuntimePhaseCounter::kChannelReadinessPublished)], 1U);
    EXPECT_EQ(snapshot[RuntimePhaseCounters::index(
                  RuntimePhaseCounter::kChannelActivationCutSubscriberProbes)],
              1U);
    EXPECT_EQ(snapshot[RuntimePhaseCounters::index(
                  RuntimePhaseCounter::kChannelActivationCutTransportProbes)],
              0U);
    EXPECT_EQ(snapshot[RuntimePhaseCounters::index(
                  RuntimePhaseCounter::kChannelActivationCutMaterializations)],
              1U);
    (*session)->cancel();
    ASSERT_TRUE((*session)->join());
  }

  {
    // Key setup: publish Image metadata plus its Tensor sidecar through the same eventless shape.
    ImageSessionFixture fixture;
    fixture.use_one_slot_blocking_transport();
    fixture.base.use_same_partition_eventless_transport();
    fixture.base.use_single_partition_runner();
    const auto counters = std::make_shared<RuntimePhaseCounters>();
    fixture.base.plan.executable_factories.phase_counters = counters;
    auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    auto binding = (*session)->create_readiness_binding(
        VertexId{2U},
        RuntimePortBindingRequest{
            .plan_generation = fixture.base.plan.plan_generation,
            .run_generation = 1U,
            .kind = RuntimeEventKind::kInputArrival,
            .source = make_source_id(1U),
            .source_clock = ClockId{1U},
            .reference_clock = ClockId{1U},
            .capacity = 1U,
        });
    ASSERT_TRUE(binding) << runtime_error_message(binding.error());
    ASSERT_TRUE((*session)->refresh_connections());
    ASSERT_TRUE((*session)->arm());
    auto indexes = RuntimeChannelSessionTestAccess::pump_index_state(**session);
    ASSERT_TRUE(indexes) << runtime_error_message(indexes.error());
    EXPECT_TRUE(indexes->exact);
    EXPECT_EQ(indexes->wait_handle_count, 0U);
    EXPECT_EQ(indexes->drain_subscriber_count, 1U);
    EXPECT_FALSE(indexes->background_thread_required);
    EXPECT_FALSE(indexes->background_thread_running);

    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
    auto owned = make_owned_tensor_sample(0x1A6EU);
    ASSERT_TRUE(owned) << runtime_error_message(owned.error());
    auto releases = owned->releases;
    schema::ImageT image{};
    image.height = 1;
    image.width = 1;
    image.encoding = schema::ImageEncoding_CUSTOM;
    image.header = std::make_shared<schema::HeaderT>();
    image.header->frame_id = "eventless-image";
    image.data = std::make_shared<Tensor>(std::move(owned->tensor));
    ASSERT_TRUE(fixture.source.output.emit(Sample<schema::ImageT>{
        std::move(image),
        SampleMetadata{.capture_timestamp_ns = 10, .source_clock_id = ClockId{1U}},
    }));
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
    ASSERT_TRUE((*session)->flush_process_local_activation_cut());

    // Key assertion: the activation cut exposes one verified Image and keeps its Tensor sidecar
    // alive through typed access and sample retirement.
    auto event = wait_for_event(**session, **binding);
    ASSERT_TRUE(event) << runtime_error_message(event.error());
    auto record = (*session)->temporal_sample(event->port, event->source_sequence);
    ASSERT_TRUE(record) << runtime_error_message(record.error());
    const std::array selection{RuntimeChannelActivationSelection{
        .port = PortId{2U},
        .selection = {InputState::kSelected, SelectionReason::kNone},
        .samples = std::span<const TemporalSampleRecord>{&*record, 1U},
        .destructive = true,
    }};
    ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selection));
    ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 1U, nullptr));
    {
      auto received = fixture.sink.input.receive();
      ASSERT_TRUE(received) << runtime_error_message(received.error());
      ASSERT_NE(received->data.header, nullptr);
      EXPECT_EQ(received->data.header->frame_id, "eventless-image");
      ASSERT_NE(received->data.data, nullptr);
      ASSERT_NE(received->data.data->data_as<std::uint64_t>(), nullptr);
      EXPECT_EQ(*received->data.data->data_as<std::uint64_t>(), 0x1A6EU);
      EXPECT_EQ(releases->load(std::memory_order_relaxed), 0U);
    }
    ASSERT_TRUE((*binding)->retire(event->source_sequence));
    ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
    ASSERT_TRUE((*session)->retire_samples(PortId{2U}, selection.front().samples));
    EXPECT_EQ(releases->load(std::memory_order_relaxed), 1U);
    const auto snapshot = counters->snapshot();
    EXPECT_EQ(snapshot[RuntimePhaseCounters::index(
                  RuntimePhaseCounter::kChannelActivationCutSubscriberProbes)],
              1U);
    EXPECT_EQ(snapshot[RuntimePhaseCounters::index(
                  RuntimePhaseCounter::kChannelActivationCutTransportProbes)],
              0U);
    EXPECT_EQ(snapshot[RuntimePhaseCounters::index(
                  RuntimePhaseCounter::kChannelActivationCutMaterializations)],
              1U);
    (*session)->cancel();
    ASSERT_TRUE((*session)->join());
  }
}

// A queued raw Tensor must wake the event-driven pump. The transport-wait trace is emitted only
// after a real wait-set readiness result, so it distinguishes this path from the timed fallback.
// REQ: RUNTIME-032
// REQ: RUNTIME-030
TEST(RuntimeChannelBinding, TensorPumpUsesEventWakeBeforeTheTimedFallback) {
  TensorSessionFixture fixture;
  std::array<RuntimePhaseTraceRecord, 16U> trace_storage;
  const auto counters = std::make_shared<RuntimePhaseCounters>(std::span{trace_storage});
  fixture.base.plan.executable_factories.phase_counters = counters;
  std::binary_semaphore wake{0};
  auto session = RuntimeChannelSession::create(
      fixture.base.plan,
      fixture.replay,
      {},
      RuntimeChannelReadinessWake{
          .state = &wake,
          .signal =
              [](void* state) noexcept { static_cast<std::binary_semaphore*>(state)->release(); },
      });
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.base.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 2U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());

  // Key setup: publish before arm so the retained backend event is the pump's first observation.
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  auto owned = make_owned_tensor_sample(0xCAFE'BEEFU);
  ASSERT_TRUE(owned) << runtime_error_message(owned.error());
  ASSERT_TRUE(fixture.source.output.emit(std::move(owned->tensor)));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  EXPECT_FALSE(wake.try_acquire());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE(wake.try_acquire_for(1s));

  auto event = wait_for_event(**session, **binding);
  ASSERT_TRUE(event) << runtime_error_message(event.error());
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());

  // Key assertion: typed ingress followed an event-wait return, not the no-attachment sleep path.
  EXPECT_TRUE(std::any_of(counters->trace_records().begin(),
                          counters->trace_records().end(),
                          [](const RuntimePhaseTraceRecord& record) {
                            return record.point ==
                                   RuntimePhaseTracePoint::kChannelTransportWaitReturned;
                          }));
  constexpr std::array kTensorEmitPoints{
      RuntimePhaseTracePoint::kTensorEmitStarted,
      RuntimePhaseTracePoint::kTensorEmitValidated,
      RuntimePhaseTracePoint::kTensorTransportPublicationStarted,
      RuntimePhaseTracePoint::kTensorTransportPublicationReturned,
      RuntimePhaseTracePoint::kTensorEmitCompleted,
  };
  std::vector<RuntimePhaseTraceRecord> tensor_emit_trace;
  std::copy_if(counters->trace_records().begin(),
               counters->trace_records().end(),
               std::back_inserter(tensor_emit_trace),
               [&](const RuntimePhaseTraceRecord& record) {
                 return std::find(kTensorEmitPoints.begin(),
                                  kTensorEmitPoints.end(),
                                  record.point) != kTensorEmitPoints.end();
               });
  // Key assertion: the transport-backed Tensor emit records every ordered publication phase with
  // one stable output-port and sample identity for post-run attribution.
  ASSERT_EQ(tensor_emit_trace.size(), kTensorEmitPoints.size());
  for (std::size_t index = 0U; index < kTensorEmitPoints.size(); ++index) {
    EXPECT_EQ(tensor_emit_trace[index].point, kTensorEmitPoints[index]);
    EXPECT_EQ(tensor_emit_trace[index].output_port_id, 1U);
    EXPECT_EQ(tensor_emit_trace[index].source_sequence, 1U);
  }
  EXPECT_EQ(counters->trace_dropped(), 0U);
}
// The accepted cross-process Tensor plan constructs the same typed runtime binding but selects the
// BufferHandle transfer arm. A pass-through source owns no fabricated allocation pool.
// REQ: CHAN-085
// REQ: CHAN-088
TEST(ChannelBindingParityTest, CrossProcessTensorPassThroughConstructsBufferHandleEndpoints) {
  TensorSessionFixture fixture;
  fixture.use_cross_process_edge();

  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto transfer =
      RuntimeChannelSessionTestAccess::tensor_transfer_state(**session, PortId{1U}, PortId{2U});
  ASSERT_TRUE(transfer) << runtime_error_message(transfer.error());
  EXPECT_EQ(transfer->buffer_handle_publishers, 1U);
  EXPECT_EQ(transfer->inline_publishers, 0U);
  EXPECT_TRUE(transfer->buffer_handle_subscriber);
  EXPECT_FALSE(transfer->inline_subscriber);
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// An unpooled source cannot prove its concrete Tensor layout at compile time. The target's frozen
// expectation therefore remains a receive-time contract, checked before assigning arrival identity
// or retaining the sample. Rejected and matching representations use fresh fail-closed sessions.
// REQ: OPERATOR-028
// REQ: OPERATOR-041
// REQ: OPERATOR-046
TEST(ChannelBindingParityTest, PassThroughTensorEnforcesTargetLayoutBeforeCacheMutation) {
  const auto exercise = [](Tensor tensor,
                           const bool should_match,
                           const bool target_declares_layout,
                           const std::optional<TensorContractRecord>& target_contract =
                               std::nullopt) {
    TensorSessionFixture fixture;
    if (target_contract) {
      fixture.base.plan.ports[1].tensor_layout = target_contract->representation;
      fixture.base.plan.ports[1].tensor_contract = target_contract;
    } else if (target_declares_layout) {
      fixture.base.plan.ports[1].tensor_layout = TensorRepresentation{
          .memory_kind = MemoryKind::kHost,
          .dtype = DLDataType{kDLUInt, 64U, 1U},
          .rank = 1U,
      };
    }
    auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    auto binding = (*session)->create_readiness_binding(
        VertexId{2U},
        RuntimePortBindingRequest{
            .plan_generation = fixture.base.plan.plan_generation,
            .run_generation = 1U,
            .kind = RuntimeEventKind::kInputArrival,
            .source = make_source_id(1U),
            .source_clock = ClockId{1U},
            .reference_clock = ClockId{1U},
            .capacity = 8U,
        });
    ASSERT_TRUE(binding) << runtime_error_message(binding.error());
    ASSERT_TRUE((*session)->refresh_connections());
    ASSERT_TRUE((*session)->arm());

    // Capture the mutation baseline before publication. A matching sample may be drained into the
    // cache as soon as emit returns, independently of when this test begins polling readiness.
    const auto before = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    ASSERT_TRUE(before) << runtime_error_message(before.error());
    ASSERT_EQ(before->occupied_samples, 0U);

    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
    ASSERT_TRUE(fixture.source.output.emit(Sample<Tensor>{std::move(tensor), {}}));
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

    auto event = wait_for_event(**session, **binding);
    if (should_match) {
      ASSERT_TRUE(event) << runtime_error_message(event.error());
      auto record = (*session)->temporal_sample(event->port, event->source_sequence);
      ASSERT_TRUE(record) << runtime_error_message(record.error());
      EXPECT_EQ(record->source_sequence, 1U);
      EXPECT_EQ(record->arrival_sequence, 1U);
      ASSERT_TRUE((*binding)->retire(event->source_sequence));
      ASSERT_TRUE((*session)->retire_samples(PortId{2U},
                                             std::span<const TemporalSampleRecord>{&*record, 1U}));
    } else {
      ASSERT_FALSE(event);
      EXPECT_EQ(event.error().code, ErrorCode::kProtocolError);
      const auto after = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
      ASSERT_TRUE(after) << runtime_error_message(after.error());
      EXPECT_EQ(after->occupied_samples, before->occupied_samples);
      EXPECT_EQ(after->next_arrival_sequence, before->next_arrival_sequence);
      EXPECT_EQ(after->next_global_arrival_order, before->next_global_arrival_order);
    }
    (*session)->cancel();
    EXPECT_EQ(static_cast<bool>((*session)->join()), should_match);
  };

  const std::array<std::int64_t, 1U> rank_one{1};
  const std::array<std::int64_t, 2U> rank_two{1, 1};
  auto wrong_memory = make_stream_tracking_tensor();
  auto wrong_dtype = make_contract_test_tensor(rank_one, DLDataType{kDLUInt, 32U, 1U});
  auto wrong_rank = make_contract_test_tensor(rank_two, DLDataType{kDLUInt, 64U, 1U});
  auto unconstrained = make_contract_test_tensor(rank_two, DLDataType{kDLUInt, 32U, 1U});
  auto matching = make_owned_tensor_sample(42U);
  ASSERT_TRUE(wrong_dtype) << runtime_error_message(wrong_dtype.error());
  ASSERT_TRUE(wrong_rank) << runtime_error_message(wrong_rank.error());
  ASSERT_TRUE(unconstrained) << runtime_error_message(unconstrained.error());
  ASSERT_TRUE(matching) << runtime_error_message(matching.error());

  {
    SCOPED_TRACE("wrong memory kind");
    exercise(std::move(wrong_memory.tensor), false, true);
  }
  {
    SCOPED_TRACE("wrong dtype");
    exercise(std::move(*wrong_dtype), false, true);
  }
  {
    SCOPED_TRACE("wrong rank");
    exercise(std::move(*wrong_rank), false, true);
  }
  {
    SCOPED_TRACE("matching layout");
    exercise(std::move(matching->tensor), true, true);
  }
  {
    SCOPED_TRACE("undeclared target layout");
    exercise(std::move(*unconstrained), true, false);
  }

  const auto normalized_input = [](const TensorInputSpec& contract) {
    auto normalized = normalize_tensor_input_spec(contract);
    EXPECT_TRUE(normalized) << (normalized ? "" : normalized.error().message);
    return normalized ? std::optional<TensorContractRecord>{std::move(*normalized)} : std::nullopt;
  };
  {
    SCOPED_TRACE("maximum byte span");
    constexpr std::array<std::int64_t, 1U> shape{3};
    auto tensor = make_contract_test_tensor(shape, DLDataType{kDLUInt, 64U, 1U});
    ASSERT_TRUE(tensor) << runtime_error_message(tensor.error());
    const auto contract = normalized_input(TensorInputSpec{
        .representation = {.memory_kind = MemoryKind::kHost,
                           .dtype = DLDataType{kDLUInt, 64U, 1U},
                           .rank = 1U},
        .bounds = tensor_bounds(16U),
    });
    ASSERT_TRUE(contract);
    exercise(std::move(*tensor), false, true, contract);
  }
  {
    SCOPED_TRACE("maximum axis extent with in-range span");
    constexpr std::array<std::int64_t, 2U> maximum_shape{2, 3};
    constexpr std::array<std::int64_t, 2U> delivered_shape{3, 2};
    auto tensor = make_contract_test_tensor(delivered_shape, DLDataType{kDLUInt, 8U, 1U});
    ASSERT_TRUE(tensor) << runtime_error_message(tensor.error());
    const auto contract = normalized_input(TensorInputSpec{
        .representation = {.memory_kind = MemoryKind::kHost,
                           .dtype = DLDataType{kDLUInt, 8U, 1U},
                           .rank = 2U},
        .bounds = tensor_bounds(maximum_shape),
    });
    ASSERT_TRUE(contract);
    exercise(std::move(*tensor), false, true, contract);
  }
  {
    SCOPED_TRACE("contiguous stride family");
    constexpr std::array<std::int64_t, 2U> shape{2, 2};
    constexpr std::array<std::int64_t, 2U> padded_byte_strides{3, 1};
    auto tensor =
        make_contract_test_tensor(shape, DLDataType{kDLUInt, 8U, 1U}, padded_byte_strides);
    ASSERT_TRUE(tensor) << runtime_error_message(tensor.error());
    const auto contract = normalized_input(TensorInputSpec{
        .representation = {.memory_kind = MemoryKind::kHost,
                           .dtype = DLDataType{kDLUInt, 8U, 1U},
                           .rank = 2U},
        .bounds = tensor_bounds(std::array<std::int64_t, 2U>{2, 3}),
    });
    ASSERT_TRUE(contract);
    exercise(std::move(*tensor), false, true, contract);
  }
  {
    SCOPED_TRACE("maximum per-axis stride");
    constexpr std::array<std::int64_t, 2U> shape{2, 2};
    constexpr std::array<std::int64_t, 2U> padded_byte_strides{3, 1};
    constexpr std::array axes{
        TensorAxisBounds{.max_extent = 2U, .max_abs_stride_bytes = 2U},
        TensorAxisBounds{.max_extent = 2U, .max_abs_stride_bytes = 1U},
    };
    auto tensor =
        make_contract_test_tensor(shape, DLDataType{kDLUInt, 8U, 1U}, padded_byte_strides);
    ASSERT_TRUE(tensor) << runtime_error_message(tensor.error());
    const auto contract = normalized_input(TensorInputSpec{
        .representation = {.memory_kind = MemoryKind::kHost,
                           .dtype = DLDataType{kDLUInt, 8U, 1U},
                           .rank = 2U,
                           .strides = TensorStridePolicy::kBoundedStrided},
        .bounds = TensorBounds{.max_rank = 2U, .max_byte_span = 5U, .axes = axes},
    });
    ASSERT_TRUE(contract);
    exercise(std::move(*tensor), false, true, contract);
  }
}

// A pass-through producer contract remains authoritative even though its storage is supplied by
// the application. Runtime validates the complete contract before assigning a publication
// sequence or entering fanout, so a rejected axis bound cannot consume emit authority.
// REQ: OPERATOR-042
TEST(ChannelBindingParityTest, PassThroughTensorEnforcesProducerAxisBoundsBeforeFanout) {
  TensorSessionFixture fixture;
  constexpr std::array maximum_shape{std::int64_t{1}};
  auto contract = normalize_tensor_output_spec(TensorOutputSpec{
      .representation = {.memory_kind = MemoryKind::kHost,
                         .dtype = DLDataType{kDLUInt, 64U, 1U},
                         .rank = 1U},
      .bounds = tensor_bounds(maximum_shape),
  });
  ASSERT_TRUE(contract) << (contract ? "" : contract.error().message);
  fixture.base.plan.ports[0].tensor_layout = contract->representation;
  fixture.base.plan.ports[0].tensor_contract = *contract;

  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

  // Key setup: the first Tensor fits aggregate service storage but exceeds the producer's exact
  // per-axis bound; the second Tensor satisfies the same frozen contract.
  constexpr std::array oversized_shape{std::int64_t{2}};
  auto oversized = make_contract_test_tensor(oversized_shape, DLDataType{kDLUInt, 64U, 1U});
  auto matching = make_contract_test_tensor(maximum_shape, DLDataType{kDLUInt, 64U, 1U});
  ASSERT_TRUE(oversized) << runtime_error_message(oversized.error());
  ASSERT_TRUE(matching) << runtime_error_message(matching.error());

  const auto rejected = fixture.source.output.emit(std::move(*oversized));
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kInvalidArgument);

  // Key assertion: rejecting the producer-contract violation preserves publication authority, so
  // the next conforming Tensor is accepted as the first sample.
  ASSERT_TRUE(fixture.source.output.emit(std::move(*matching)));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// A catalog Tensor import retains the complete normalized logical record, not only its physical
// byte-span and memory-kind subset. Every mismatch is rejected before reserving arrival identity
// or constructing a typed cache slot, even when the backing allocation itself fits the pool.
// REQ: SYSTEM-001
TEST(ChannelBindingParityTest, ExternalTensorImportRejectsLogicalContractDriftBeforeCacheMutation) {
  TensorSessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  const PortId input{2U};
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_external_tensor_contract(
      **session,
      input,
      TensorBoundaryContractRecord{
          .memory_kind = MemoryKind::kHost,
          .dtype = DLDataType{kDLUInt, 64U, 1U},
          .rank = 2U,
          .max_rank = 2U,
          .max_byte_span = 64U,
          .axes = {{.max_extent = 2U, .max_abs_stride_bytes = 16U},
                   {.max_extent = 2U, .max_abs_stride_bytes = 8U}},
          .strides = TensorStridePolicy::kContiguousOnly,
      }));
  const auto before = RuntimeChannelSessionTestAccess::input_reset_state(**session, input);
  ASSERT_TRUE(before) << runtime_error_message(before.error());

  const auto expect_rejected_without_mutation = [&](Tensor tensor, const std::uint64_t byte_size) {
    auto rejected = RuntimeChannelSessionTestAccess::inject_received_tensor(
        **session,
        input,
        Sample<Tensor>{std::move(tensor),
                       SampleMetadata{
                           .sequence = 1U,
                           .byte_size = byte_size,
                           .payload_kind = PayloadKind::kDescriptorBacked,
                       }});
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, ErrorCode::kProtocolError);
    const auto after = RuntimeChannelSessionTestAccess::input_reset_state(**session, input);
    ASSERT_TRUE(after) << runtime_error_message(after.error());
    EXPECT_EQ(after->next_arrival_sequence, before->next_arrival_sequence);
    EXPECT_EQ(after->next_global_arrival_order, before->next_global_arrival_order);
    EXPECT_EQ(after->occupied_samples, before->occupied_samples);
  };

  constexpr std::array<std::int64_t, 2U> shape_2x2{2, 2};
  auto wrong_dtype = make_contract_test_tensor(shape_2x2, DLDataType{kDLUInt, 32U, 1U});
  ASSERT_TRUE(wrong_dtype) << runtime_error_message(wrong_dtype.error());
  expect_rejected_without_mutation(std::move(*wrong_dtype), 16U);

  constexpr std::array<std::int64_t, 1U> shape_rank_one{4};
  auto wrong_rank = make_contract_test_tensor(shape_rank_one, DLDataType{kDLUInt, 64U, 1U});
  ASSERT_TRUE(wrong_rank) << runtime_error_message(wrong_rank.error());
  expect_rejected_without_mutation(std::move(*wrong_rank), 32U);

  constexpr std::array<std::int64_t, 2U> shape_3x2{3, 2};
  auto oversized_axis = make_contract_test_tensor(shape_3x2, DLDataType{kDLUInt, 64U, 1U});
  ASSERT_TRUE(oversized_axis) << runtime_error_message(oversized_axis.error());
  expect_rejected_without_mutation(std::move(*oversized_axis), 48U);

  constexpr std::array<std::int64_t, 2U> noncontiguous_byte_strides{24, 8};
  auto noncontiguous = make_contract_test_tensor(
      shape_2x2, DLDataType{kDLUInt, 64U, 1U}, noncontiguous_byte_strides);
  ASSERT_TRUE(noncontiguous) << runtime_error_message(noncontiguous.error());
  expect_rejected_without_mutation(std::move(*noncontiguous), 32U);

  {
    SCOPED_TRACE("max_rank");
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_external_tensor_contract(
        **session,
        input,
        TensorBoundaryContractRecord{
            .memory_kind = MemoryKind::kHost,
            .dtype = DLDataType{kDLUInt, 64U, 1U},
            .max_rank = 1U,
            .max_byte_span = 64U,
            .strides = TensorStridePolicy::kContiguousOnly,
        }));
    auto excessive_rank = make_contract_test_tensor(shape_2x2, DLDataType{kDLUInt, 64U, 1U});
    ASSERT_TRUE(excessive_rank) << runtime_error_message(excessive_rank.error());
    expect_rejected_without_mutation(std::move(*excessive_rank), 32U);
  }

  {
    SCOPED_TRACE("max_byte_span");
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_external_tensor_contract(
        **session,
        input,
        TensorBoundaryContractRecord{
            .memory_kind = MemoryKind::kHost,
            .dtype = DLDataType{kDLUInt, 64U, 1U},
            .rank = 2U,
            .max_rank = 2U,
            .max_byte_span = 16U,
            .axes = {{.max_extent = 2U, .max_abs_stride_bytes = 16U},
                     {.max_extent = 2U, .max_abs_stride_bytes = 8U}},
            .strides = TensorStridePolicy::kContiguousOnly,
        }));
    auto excessive_span = make_contract_test_tensor(shape_2x2, DLDataType{kDLUInt, 64U, 1U});
    ASSERT_TRUE(excessive_span) << runtime_error_message(excessive_span.error());
    expect_rejected_without_mutation(std::move(*excessive_span), 32U);
  }

  {
    SCOPED_TRACE("memory_kind");
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_external_tensor_contract(
        **session,
        input,
        TensorBoundaryContractRecord{
            .memory_kind = MemoryKind::kHost,
            .dtype = DLDataType{kDLUInt, 64U, 1U},
            .rank = 2U,
            .max_rank = 2U,
            .max_byte_span = 64U,
            .axes = {{.max_extent = 2U, .max_abs_stride_bytes = 16U},
                     {.max_extent = 2U, .max_abs_stride_bytes = 8U}},
            .strides = TensorStridePolicy::kContiguousOnly,
        }));
    auto wrong_memory = make_contract_test_tensor(shape_2x2,
                                                  DLDataType{kDLUInt, 64U, 1U},
                                                  std::span<const std::int64_t>{},
                                                  MemoryStorageType::kHost);
    ASSERT_TRUE(wrong_memory) << runtime_error_message(wrong_memory.error());
    expect_rejected_without_mutation(std::move(*wrong_memory), 32U);
  }

  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_external_tensor_contract(
      **session,
      input,
      TensorBoundaryContractRecord{
          .memory_kind = MemoryKind::kHost,
          .dtype = DLDataType{kDLUInt, 64U, 1U},
          .rank = 2U,
          .max_rank = 2U,
          .max_byte_span = 64U,
          .axes = {{.max_extent = 2U, .max_abs_stride_bytes = 8U},
                   {.max_extent = 2U, .max_abs_stride_bytes = 8U}},
          .strides = TensorStridePolicy::kContiguousOnly,
      }));
  auto oversized_derived_stride =
      make_contract_test_tensor(shape_2x2, DLDataType{kDLUInt, 64U, 1U});
  ASSERT_TRUE(oversized_derived_stride) << runtime_error_message(oversized_derived_stride.error());
  expect_rejected_without_mutation(std::move(*oversized_derived_stride), 32U);

  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// Tensor-only test seams are deliberately narrower than the generic input state helpers. Reject a
// scalar subscriber before taking its lock or interpreting its typed slot storage as
// Sample<Tensor>. REQ: SYSTEM-001
TEST(ChannelBindingParityTest, TensorTestSeamsRejectScalarInputWithoutMutation) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  const PortId input{2U};
  const auto before = RuntimeChannelSessionTestAccess::input_reset_state(**session, input);
  ASSERT_TRUE(before) << runtime_error_message(before.error());

  auto tensor =
      make_contract_test_tensor(std::array<std::int64_t, 1U>{1U}, DLDataType{kDLUInt, 8U, 1U});
  ASSERT_TRUE(tensor) << runtime_error_message(tensor.error());
  auto injected = RuntimeChannelSessionTestAccess::inject_received_tensor(
      **session,
      input,
      Sample<Tensor>{
          std::move(*tensor),
          SampleMetadata{.byte_size = 1U, .payload_kind = PayloadKind::kDescriptorBacked}});
  ASSERT_FALSE(injected);
  EXPECT_EQ(injected.error().code, ErrorCode::kInvalidArgument);

  auto configured = RuntimeChannelSessionTestAccess::set_external_tensor_contract(
      **session,
      input,
      TensorBoundaryContractRecord{
          .memory_kind = MemoryKind::kHost,
          .max_byte_span = 1U,
          .strides = TensorStridePolicy::kContiguousOnly,
      });
  ASSERT_FALSE(configured);
  EXPECT_EQ(configured.error().code, ErrorCode::kInvalidArgument);

  const auto after = RuntimeChannelSessionTestAccess::input_reset_state(**session, input);
  ASSERT_TRUE(after) << runtime_error_message(after.error());
  EXPECT_EQ(after->active_generation, before->active_generation);
  EXPECT_EQ(after->view_generation, before->view_generation);
  EXPECT_EQ(after->producer_epoch, before->producer_epoch);
  EXPECT_EQ(after->next_arrival_sequence, before->next_arrival_sequence);
  EXPECT_EQ(after->next_global_arrival_order, before->next_global_arrival_order);
  EXPECT_EQ(after->occupied_samples, before->occupied_samples);
}

// Catalog Tensor ingress is driven by a publisher outside this Runtime session. Its typed receive
// cannot serve as evidence that a session-owned process-local publication has completed its cut.
// REQ: RUNTIME-029
TEST(ChannelBindingParityTest, CatalogTensorImportDoesNotAdvanceProcessLocalProgress) {
  const std::string endpoint_name = unique_service();
  Graph graph{"catalog-tensor-progress-isolation"};
  const ClockKey clock = graph.bind_shared_clock("runtime-clock", std::make_shared<ManualClock>(0));
  graph.set_default_clock(clock);
  constexpr std::array axes{TensorAxisBounds{.max_extent = 8U, .max_abs_stride_bytes = 1U}};
  TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 2U};
  options.tensor = TensorBoundaryContract{
      .representation = {.memory_kind = MemoryKind::kHost,
                         .dtype = DLDataType{kDLUInt, 8U, 1U},
                         .rank = 1U,
                         .strides = TensorStridePolicy::kContiguousOnly},
      .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 8U, .axes = axes},
      .transfer =
          TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
  };
  const auto imported = graph.import_topic<Tensor>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<TensorSinkOperator>("sink");
  graph.add_flow(imported.output(), sink->input);

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  const auto* input = planned_operator_port(*state, PortDirection::kInput);
  ASSERT_NE(input, nullptr);
  const std::array replay_ports{RuntimeChannelPortReplayRecord{
      .port = input->id,
      .owner = input->owner,
      .direction = PortDirection::kInput,
      .declaration = const_cast<void*>(static_cast<const void*>(&sink->input)),
      .type_ops = runtime_port_type_ops<Tensor>(),
  }};
  auto session = RuntimeChannelSession::create(*state, replay_ports);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto client_runtime = ChannelRuntime::create();
  ASSERT_TRUE(client_runtime) << runtime_error_message(client_runtime.error());
  auto publisher = client_runtime->create_publisher<Tensor>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_for_catalog_subscriber(*publisher));
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE(wait_for_catalog_subscriber(*publisher));
  const auto pump_state = RuntimeChannelSessionTestAccess::pump_index_state(**session);
  ASSERT_TRUE(pump_state) << runtime_error_message(pump_state.error());
  EXPECT_EQ(pump_state->drain_subscriber_count, 1U);
  EXPECT_TRUE(pump_state->background_thread_required);
  EXPECT_TRUE(pump_state->background_thread_running);
  EXPECT_TRUE(pump_state->exact);

  // Key setup: let the production pump retain one externally published host Tensor.
  expected<void, Error> published = make_unexpected(Error{ErrorCode::kNotReady});
  const auto publish_deadline = std::chrono::steady_clock::now() + 2s;
  do {
    published = publish_catalog_host_tensor(*publisher, std::byte{0x31});
    if (published || (published.error().code != ErrorCode::kNotReady &&
                      published.error().code != ErrorCode::kPublicationPartial)) {
      break;
    }
    ASSERT_TRUE(publisher->update_connections());
    std::this_thread::sleep_for(1ms);
  } while (std::chrono::steady_clock::now() < publish_deadline);
  ASSERT_TRUE(published) << runtime_error_message(published.error());
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  expected<RuntimeChannelInputResetState, Error> retained =
      make_unexpected(Error{ErrorCode::kNotReady});
  do {
    retained = RuntimeChannelSessionTestAccess::input_reset_state(**session, input->id);
    ASSERT_TRUE(retained) << runtime_error_message(retained.error());
    if (retained->occupied_samples == 0U) {
      std::this_thread::sleep_for(1ms);
    }
  } while (retained->occupied_samples == 0U && std::chrono::steady_clock::now() < deadline);
  ASSERT_EQ(retained->occupied_samples, 1U);

  const auto progress = RuntimeChannelSessionTestAccess::process_local_progress_state(**session);
  ASSERT_TRUE(progress) << runtime_error_message(progress.error());
  // Key assertion: the catalog receive leaves local graph delivery accounting untouched.
  EXPECT_EQ(progress->published_deliveries, 0U);
  EXPECT_EQ(progress->received_deliveries, 0U);
  EXPECT_EQ(progress->publications_in_flight, 0U);
  EXPECT_TRUE(progress->tracking_valid);
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// A host catalog Tensor kDropOldest cache materializes the replacement before destroying its
// previous retained Tensor. Runtime therefore reserves one receive-control slot beyond the typed
// cache and one readiness credit for ordered EOS. Repeated observation of that exact EOS is
// idempotent, while a conflicting terminal sequence remains a protocol error. REQ: CHAN-078 REQ:
// RUNTIME-025 REQ: SYSTEM-006
TEST(ChannelBindingParityTest, HostCatalogTensorDropOldestReplacementRetainsOrderedTerminalRecord) {
  const std::string endpoint_name = unique_service();
  Graph graph{"host-catalog-tensor-drop-oldest-replacement"};
  const ClockKey clock = graph.bind_shared_clock("runtime-clock", std::make_shared<ManualClock>(0));
  graph.set_default_clock(clock);
  constexpr std::array axes{TensorAxisBounds{.max_extent = 8U, .max_abs_stride_bytes = 1U}};
  TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 2U};
  options.tensor = TensorBoundaryContract{
      .representation = {.memory_kind = MemoryKind::kHost,
                         .dtype = DLDataType{kDLUInt, 8U, 1U},
                         .rank = 1U,
                         .strides = TensorStridePolicy::kContiguousOnly},
      .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 8U, .axes = axes},
      .transfer =
          TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
  };
  const auto imported = graph.import_topic<Tensor>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<TensorSinkOperator>("sink");
  graph.add_flow(imported.output(),
                 sink->input,
                 ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  const auto* input = planned_operator_port(*state, PortDirection::kInput);
  ASSERT_NE(input, nullptr);
  const std::array replay_ports{RuntimeChannelPortReplayRecord{
      .port = input->id,
      .owner = input->owner,
      .direction = PortDirection::kInput,
      .declaration = const_cast<void*>(static_cast<const void*>(&sink->input)),
      .type_ops = runtime_port_type_ops<Tensor>(),
  }};
  auto session = RuntimeChannelSession::create(*state, replay_ports);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto readiness_capacity = (*session)->readiness_capacity(input->owner);
  ASSERT_TRUE(readiness_capacity) << runtime_error_message(readiness_capacity.error());
  EXPECT_EQ(*readiness_capacity, 2U);

  auto client_runtime = ChannelRuntime::create();
  ASSERT_TRUE(client_runtime) << runtime_error_message(client_runtime.error());
  auto publisher = client_runtime->create_publisher<Tensor>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_for_catalog_subscriber(*publisher));

  ASSERT_TRUE(publish_catalog_host_tensor(*publisher, std::byte{0x11}));
  auto first_received = RuntimeChannelSessionTestAccess::receive_one(**session, input->id);
  ASSERT_TRUE(first_received) << runtime_error_message(first_received.error());
  ASSERT_TRUE(publish_catalog_host_tensor(*publisher, std::byte{0x22}));
  auto replacement_received = RuntimeChannelSessionTestAccess::receive_one(**session, input->id);
  ASSERT_TRUE(replacement_received) << runtime_error_message(replacement_received.error());
  auto retained = RuntimeChannelSessionTestAccess::input_reset_state(**session, input->id);
  ASSERT_TRUE(retained) << runtime_error_message(retained.error());
  EXPECT_EQ(retained->occupied_samples, 1U);
  EXPECT_EQ(retained->next_arrival_sequence, 3U);

  ASSERT_TRUE(wait_for_catalog_subscriber(*publisher));
  const auto closed = publisher->close();
  ASSERT_TRUE(closed) << runtime_error_message(closed.error());
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, input->id));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, input->id));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::store_end_of_stream(**session, input->id, 3U));
  auto conflicting = RuntimeChannelSessionTestAccess::store_end_of_stream(**session, input->id, 4U);
  ASSERT_FALSE(conflicting);
  EXPECT_EQ(conflicting.error().code, ErrorCode::kProtocolError);

  auto binding = (*session)->create_readiness_binding(input->owner,
                                                      RuntimePortBindingRequest{
                                                          .plan_generation = state->plan_generation,
                                                          .run_generation = 1U,
                                                          .kind = RuntimeEventKind::kInputArrival,
                                                          .source = make_source_id(1U),
                                                          .source_clock = ClockId{1U},
                                                          .reference_clock = ClockId{1U},
                                                          .capacity = *readiness_capacity,
                                                      });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*binding)->poll(TimePoint{1, ClockId{1U}}));
  std::array<RuntimeEvent, 2U> events{};
  auto drained = (*binding)->drain(events);
  ASSERT_TRUE(drained) << runtime_error_message(drained.error());
  ASSERT_EQ(*drained, 2U);
  EXPECT_EQ(events[0].kind, RuntimeEventKind::kInputArrival);
  EXPECT_EQ(events[1].kind, RuntimeEventKind::kInputEndOfStream);
  auto sample = (*session)->temporal_sample(events[0].port, events[0].source_sequence);
  ASSERT_TRUE(sample) << runtime_error_message(sample.error());
  EXPECT_EQ(sample->source_sequence, 2U);
  ASSERT_TRUE((*binding)->retire(events[0].source_sequence));
  ASSERT_TRUE((*binding)->retire(events[1].source_sequence));
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// Every lossy raw-Tensor transport reserves the exact occupied replacement destination before
// consuming its staged head. Readiness and activation cannot claim that incumbent while first-use
// Tensor materialization is complete but typed commit is deliberately paused.
// REQ: RUNTIME-026
// REQ: RUNTIME-042
// REQ: RUNTIME-044
TEST(ChannelBindingParityTest, RawTensorIngressRetainsExactLossyDestinationAcrossMaterialization) {
  // Key setup: pause raw-Tensor ingress after materialization while a pre-witness incumbent
  // occupies the only typed slot.
  for (const QueuePolicy policy : {QueuePolicy::kDropOldest}) {
    SCOPED_TRACE(static_cast<std::uint32_t>(policy));
    const std::string endpoint_name = unique_service();
    Graph graph{"raw-tensor-ingress-transaction"};
    const ClockKey clock =
        graph.bind_shared_clock("runtime-clock", std::make_shared<ManualClock>(0));
    graph.set_default_clock(clock);
    constexpr std::array axes{TensorAxisBounds{.max_extent = 8U, .max_abs_stride_bytes = 1U}};
    TopicOptions options;
    options.endpoint_name = endpoint_name;
    options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 2U};
    options.tensor = TensorBoundaryContract{
        .representation = {.memory_kind = MemoryKind::kHost,
                           .dtype = DLDataType{kDLUInt, 8U, 1U},
                           .rank = 1U,
                           .strides = TensorStridePolicy::kContiguousOnly},
        .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 8U, .axes = axes},
        .transfer =
            TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
    };
    const auto imported =
        graph.import_topic<Tensor>("holomq://" + endpoint_name, std::move(options));
    const auto sink = graph.op<TensorSinkOperator>("sink");
    auto& sink_input = const_cast<Input<Tensor>&>(sink->input);
    graph.add_flow(imported.output(),
                   sink->input,
                   ConnectionOptions{.queue_policy = policy, .queue_depth = 1U});

    const ExecutionPlan plan = compile(graph);
    ASSERT_TRUE(plan.ok()) << plan.json();
    const auto state = ExecutionPlanAccess::retain(plan);
    ASSERT_NE(state, nullptr);
    const auto input =
        std::find_if(state->ports.begin(), state->ports.end(), [](const PortPlanRecord& candidate) {
          return candidate.owner_kind == PortOwnerKind::kOperator &&
                 candidate.direction == PortDirection::kInput;
        });
    ASSERT_NE(input, state->ports.end());
    const std::array replay_ports{RuntimeChannelPortReplayRecord{
        .port = input->id,
        .owner = input->owner,
        .direction = PortDirection::kInput,
        .declaration = &sink_input,
        .type_ops = runtime_port_type_ops<Tensor>(),
    }};
    BlockingTypedIngressObserver observer;
    auto session = RuntimeChannelSession::create(*state, replay_ports);
    ASSERT_TRUE(session) << runtime_error_message(session.error());

    auto client_runtime = ChannelRuntime::create();
    ASSERT_TRUE(client_runtime) << runtime_error_message(client_runtime.error());
    auto publisher =
        client_runtime->create_publisher<Tensor>(plan.endpoint_catalog(), endpoint_name);
    ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
    ASSERT_TRUE(wait_for_catalog_subscriber(*publisher));
    const auto publish_tensor = [&](const std::byte value) -> expected<void, Error> {
      const auto deadline = std::chrono::steady_clock::now() + 2s;
      while (std::chrono::steady_clock::now() < deadline) {
        auto published = publish_catalog_host_tensor(*publisher, value);
        if (published || published.error().code != ErrorCode::kNotReady) {
          return published;
        }
        auto refreshed = publisher->update_connections();
        if (!refreshed) {
          return make_unexpected(std::move(refreshed).error());
        }
        std::this_thread::sleep_for(1ms);
      }
      return make_unexpected(Error{ErrorCode::kTimeout});
    };
    ASSERT_TRUE(publish_tensor(std::byte{0x11}));
    ASSERT_TRUE(wait_for_catalog_input(**session, input->id));
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_pre_witness_replacement_allowed(
        **session, input->id, true));

    auto binding =
        (*session)->create_readiness_binding(input->owner,
                                             RuntimePortBindingRequest{
                                                 .plan_generation = state->plan_generation,
                                                 .run_generation = 1U,
                                                 .kind = RuntimeEventKind::kInputArrival,
                                                 .source = make_source_id(1U),
                                                 .source_clock = ClockId{1U},
                                                 .reference_clock = ClockId{1U},
                                                 .capacity = 2U,
                                             });
    ASSERT_TRUE(binding) << runtime_error_message(binding.error());
    observer.block.store(true, std::memory_order_release);
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_typed_ingress_observer(
        **session, input->id, &observer, &BlockingTypedIngressObserver::observe));
    ASSERT_TRUE(publish_tensor(std::byte{0x22}));
    expected<void, Error> receive_result = make_unexpected(Error{ErrorCode::kNotReady});
    std::thread receiver([&] { receive_result = wait_for_catalog_input(**session, input->id); });
    const auto observer_deadline = std::chrono::steady_clock::now() + 2s;
    while (!observer.entered.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < observer_deadline) {
      std::this_thread::yield();
    }
    std::promise<void> terminal_probe_started_promise;
    auto terminal_probe_started = terminal_probe_started_promise.get_future();
    auto terminal_probe = std::async(std::launch::async, [&] {
      terminal_probe_started_promise.set_value();
      return (*session)->input_terminal_observed(input->id);
    });

    // Key assertion: the exact occupied destination remains reserved and hidden from readiness
    // until the newer Tensor backing commits once. The owner-side terminal probe must remain
    // pending while that first-use/import seam owns the transport mutex; it may inspect the exact
    // staged state only after the receive transaction releases the mutex.
    const auto reservations =
        RuntimeChannelSessionTestAccess::typed_ingress_reservations(**session, input->id);
    const auto retained = RuntimeChannelSessionTestAccess::input_reset_state(**session, input->id);
    const auto blocked_poll = (*binding)->poll(TimePoint{1, ClockId{1U}});
    RuntimeEvent blocked_event;
    const auto blocked_drain = (*binding)->drain(std::span<RuntimeEvent>{&blocked_event, 1U});
    const bool retry_remains_visible = (*binding)->poll_target().has_value();
    const auto terminal_probe_start_status = terminal_probe_started.wait_for(2s);
    const auto terminal_probe_status_before_release =
        terminal_probe_start_status == std::future_status::ready ? terminal_probe.wait_for(10ms)
                                                                 : std::future_status::deferred;
    observer.release.store(true, std::memory_order_release);
    receiver.join();
    const auto terminal_probe_status_after_release = terminal_probe.wait_for(2s);
    ASSERT_TRUE(observer.entered.load(std::memory_order_acquire))
        << (receive_result ? std::string{} : runtime_error_message(receive_result.error()));
    EXPECT_EQ(observer.calls.load(std::memory_order_relaxed), 1U);
    EXPECT_EQ(terminal_probe_start_status, std::future_status::ready);
    EXPECT_EQ(terminal_probe_status_before_release, std::future_status::timeout);
    ASSERT_EQ(terminal_probe_status_after_release, std::future_status::ready);
    auto terminal_observed = terminal_probe.get();
    ASSERT_TRUE(terminal_observed) << runtime_error_message(terminal_observed.error());
    EXPECT_FALSE(*terminal_observed);
    ASSERT_TRUE(reservations) << runtime_error_message(reservations.error());
    EXPECT_EQ(*reservations, 1U);
    ASSERT_TRUE(retained) << runtime_error_message(retained.error());
    EXPECT_EQ(retained->occupied_samples, 1U);
    EXPECT_EQ(retained->next_arrival_sequence, 2U);
    ASSERT_TRUE(blocked_poll) << runtime_error_message(blocked_poll.error());
    ASSERT_TRUE(blocked_drain) << runtime_error_message(blocked_drain.error());
    EXPECT_EQ(*blocked_drain, 0U);
    EXPECT_TRUE(retry_remains_visible);
    ASSERT_TRUE(receive_result) << runtime_error_message(receive_result.error());
    const auto released =
        RuntimeChannelSessionTestAccess::typed_ingress_reservations(**session, input->id);
    ASSERT_TRUE(released) << runtime_error_message(released.error());
    EXPECT_EQ(*released, 0U);

    ASSERT_TRUE((*binding)->poll(TimePoint{1, ClockId{1U}}));
    RuntimeEvent event;
    auto drained = (*binding)->drain(std::span<RuntimeEvent>{&event, 1U});
    ASSERT_TRUE(drained) << runtime_error_message(drained.error());
    ASSERT_EQ(*drained, 1U);
    auto record = (*session)->temporal_sample(event.port, event.source_sequence);
    ASSERT_TRUE(record) << runtime_error_message(record.error());
    EXPECT_EQ(record->source_sequence, 2U);
    EXPECT_EQ(record->arrival_sequence, 2U);
    const std::array selection{RuntimeChannelActivationSelection{
        .port = input->id,
        .selection = {InputState::kSelected, SelectionReason::kNone},
        .samples = std::span<const TemporalSampleRecord>{&*record, 1U},
        .destructive = true,
    }};
    ASSERT_TRUE((*session)->prepare_activation(input->owner, 1U, selection));
    ASSERT_TRUE((*session)->prepare_tensor_access(input->owner, 1U, nullptr));
    auto sample = sink_input.receive();
    ASSERT_TRUE(sample) << runtime_error_message(sample.error());
    ASSERT_NE(sample->data.data(), nullptr);
    EXPECT_EQ(*static_cast<const std::byte*>(sample->data.data()), std::byte{0x22});
    EXPECT_EQ(sample->metadata.sequence, 2U);
    ASSERT_TRUE((*binding)->retire(event.source_sequence));
    ASSERT_TRUE((*session)->retire_activation(input->owner, 1U, true));
    ASSERT_TRUE(
        (*session)->retire_samples(input->id, std::span<const TemporalSampleRecord>{&*record, 1U}));
    (*session)->cancel();
    EXPECT_TRUE((*session)->join());
  }
}

// Catalog Tensor import readiness derives from the complete typed retention geometry rather than
// a capacity-one shortcut. Every finite kDropOldest cache reserves its N data candidates plus one
// independent ordered terminal witness.
// REQ: CHAN-078
// REQ: RUNTIME-025
TEST(ChannelBindingParityTest, CatalogTensorDropOldestReadinessScalesWithTypedCacheDepth) {
  constexpr std::array<std::uint32_t, 3U> kQueueDepths{1U, 2U, 3U};
  for (const std::uint32_t queue_depth : kQueueDepths) {
    SCOPED_TRACE(queue_depth);
    const std::string endpoint_name = unique_service();
    Graph graph{"catalog-tensor-drop-oldest-readiness-capacity"};
    const ClockKey clock =
        graph.bind_shared_clock("runtime-clock", std::make_shared<ManualClock>(0));
    graph.set_default_clock(clock);
    constexpr std::array axes{TensorAxisBounds{.max_extent = 8U, .max_abs_stride_bytes = 1U}};
    TopicOptions options;
    options.endpoint_name = endpoint_name;
    options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = queue_depth + 1U};
    options.tensor = TensorBoundaryContract{
        .representation = {.memory_kind = MemoryKind::kHost,
                           .dtype = DLDataType{kDLUInt, 8U, 1U},
                           .rank = 1U,
                           .strides = TensorStridePolicy::kContiguousOnly},
        .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 8U, .axes = axes},
        .transfer =
            TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
    };
    const auto imported =
        graph.import_topic<Tensor>("holomq://" + endpoint_name, std::move(options));
    const auto sink = graph.op<TensorSinkOperator>("sink");
    graph.add_flow(imported.output(),
                   sink->input,
                   ConnectionOptions{
                       .queue_policy = QueuePolicy::kDropOldest,
                       .queue_depth = queue_depth,
                   });

    const ExecutionPlan plan = compile(graph);
    ASSERT_TRUE(plan.ok()) << plan.json();
    const auto state = ExecutionPlanAccess::retain(plan);
    ASSERT_NE(state, nullptr);
    const auto* input = planned_operator_port(*state, PortDirection::kInput);
    ASSERT_NE(input, nullptr);
    const std::array replay_ports{RuntimeChannelPortReplayRecord{
        .port = input->id,
        .owner = input->owner,
        .direction = PortDirection::kInput,
        .declaration = const_cast<void*>(static_cast<const void*>(&sink->input)),
        .type_ops = runtime_port_type_ops<Tensor>(),
    }};
    auto session = RuntimeChannelSession::create(*state, replay_ports);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    auto readiness_capacity = (*session)->readiness_capacity(input->owner);
    ASSERT_TRUE(readiness_capacity) << runtime_error_message(readiness_capacity.error());
    EXPECT_EQ(*readiness_capacity, queue_depth + 1U);
  }
}

// Catalog scalar imports reserve one non-payload readiness credit in addition to their finite
// typed cache. A capacity-one cache can therefore retain data and ordered EOS in the same cold
// cut without turning the terminal witness into backpressure.
// REQ: CHAN-078
// REQ: RUNTIME-025
TEST(ChannelBindingParityTest, FullScalarCatalogCachePublishesOrderedTerminalWitness) {
  const std::string endpoint_name = unique_service();
  Graph graph{"full-scalar-catalog-cache-end-of-stream"};
  const ClockKey clock = graph.bind_shared_clock("runtime-clock", std::make_shared<ManualClock>(0));
  graph.set_default_clock(clock);
  TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1U};
  const auto imported =
      graph.import_topic<std::int32_t>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<SinkOperator>("sink");
  auto& sink_input = const_cast<Input<std::int32_t>&>(sink->input);
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  const auto* input = planned_operator_port(*state, PortDirection::kInput);
  ASSERT_NE(input, nullptr);
  const std::array replay_ports{RuntimeChannelPortReplayRecord{
      .port = input->id,
      .owner = input->owner,
      .direction = PortDirection::kInput,
      .declaration = &sink_input,
      .type_ops = runtime_port_type_ops<std::int32_t>(),
  }};
  auto session = RuntimeChannelSession::create(*state, replay_ports);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto readiness_capacity = (*session)->readiness_capacity(input->owner);
  ASSERT_TRUE(readiness_capacity) << runtime_error_message(readiness_capacity.error());
  EXPECT_EQ(*readiness_capacity, 2U);

  auto client_runtime = ChannelRuntime::create();
  ASSERT_TRUE(client_runtime) << runtime_error_message(client_runtime.error());
  auto publisher =
      client_runtime->create_publisher<std::int32_t>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_for_catalog_subscriber(*publisher));
  expected<std::uint64_t, Error> emitted = make_unexpected(Error{ErrorCode::kNotReady});
  for (std::size_t attempt = 0U; attempt < 8U && !emitted; ++attempt) {
    const auto primed = RuntimeChannelSessionTestAccess::receive_one(**session, input->id);
    if (!primed) {
      ASSERT_EQ(primed.error().code, ErrorCode::kNotReady);
    }
    emitted = publisher->emit(42);
    if (!emitted) {
      ASSERT_EQ(emitted.error().code, ErrorCode::kNotReady);
      std::this_thread::sleep_for(1ms);
    }
  }
  ASSERT_TRUE(emitted) << runtime_error_message(emitted.error());
  ASSERT_TRUE(wait_for_catalog_input(**session, input->id));

  auto binding = (*session)->create_readiness_binding(input->owner,
                                                      RuntimePortBindingRequest{
                                                          .plan_generation = state->plan_generation,
                                                          .run_generation = 1U,
                                                          .kind = RuntimeEventKind::kInputArrival,
                                                          .source = make_source_id(1U),
                                                          .source_clock = ClockId{1U},
                                                          .reference_clock = ClockId{1U},
                                                          .capacity = *readiness_capacity,
                                                      });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE(publisher->close());
  ASSERT_TRUE((*binding)->poll(TimePoint{1, ClockId{1U}}));
  std::array<RuntimeEvent, 2U> events{};
  auto drained = (*binding)->drain(events);
  ASSERT_TRUE(drained) << runtime_error_message(drained.error());
  ASSERT_EQ(*drained, 2U);
  EXPECT_EQ(events[0].kind, RuntimeEventKind::kInputArrival);
  EXPECT_EQ(events[1].kind, RuntimeEventKind::kInputEndOfStream);
  ASSERT_TRUE((*binding)->retire(events[0].source_sequence));
  ASSERT_TRUE((*binding)->retire(events[1].source_sequence));
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// An externally driven catalog data operation advances its ordered service head before Runtime
// takes the cache mutex. The exact occupied replacement destination must therefore remain reserved
// across that boundary instead of being recomputed after the irreversible receive.
// REQ: RUNTIME-026
// REQ: RUNTIME-042
// REQ: RUNTIME-044
TEST(ChannelBindingParityTest, ExternalCatalogIngressRetainsDestinationAfterOrderedReceive) {
  // Key setup: pause a one-slot drop-oldest catalog operation after ordered receive advances but
  // before its newer scalar value commits to the typed cache.
  const std::string endpoint_name = unique_service();
  Graph graph{"external-catalog-ingress-transaction"};
  const ClockKey clock = graph.bind_shared_clock("runtime-clock", std::make_shared<ManualClock>(0));
  graph.set_default_clock(clock);
  TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 2U};
  const auto imported =
      graph.import_topic<std::int32_t>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<SinkOperator>("sink");
  auto& sink_input = const_cast<Input<std::int32_t>&>(sink->input);
  graph.add_flow(imported.output(),
                 sink->input,
                 ConnectionOptions{
                     .queue_policy = QueuePolicy::kDropOldest,
                     .queue_depth = 1U,
                 });

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  const auto input =
      std::find_if(state->ports.begin(), state->ports.end(), [](const PortPlanRecord& candidate) {
        return candidate.owner_kind == PortOwnerKind::kOperator &&
               candidate.direction == PortDirection::kInput;
      });
  ASSERT_NE(input, state->ports.end());
  const std::array replay_ports{RuntimeChannelPortReplayRecord{
      .port = input->id,
      .owner = input->owner,
      .direction = PortDirection::kInput,
      .declaration = &sink_input,
      .type_ops = runtime_port_type_ops<std::int32_t>(),
  }};
  BlockingTypedIngressObserver observer;
  auto session = RuntimeChannelSession::create(*state, replay_ports);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto client_runtime = ChannelRuntime::create();
  ASSERT_TRUE(client_runtime) << runtime_error_message(client_runtime.error());
  auto publisher =
      client_runtime->create_publisher<std::int32_t>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_for_catalog_subscriber(*publisher));
  const auto publish_value = [&](const std::int32_t value) -> expected<void, Error> {
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (std::chrono::steady_clock::now() < deadline) {
      auto published = publisher->emit(value);
      if (published) {
        return {};
      }
      if (published.error().code != ErrorCode::kNotReady) {
        return make_unexpected(std::move(published).error());
      }
      auto refreshed = publisher->update_connections();
      if (!refreshed) {
        return make_unexpected(std::move(refreshed).error());
      }
      std::this_thread::sleep_for(1ms);
    }
    return make_unexpected(Error{ErrorCode::kTimeout});
  };
  ASSERT_TRUE(publish_value(11));
  ASSERT_TRUE(wait_for_catalog_input(**session, input->id));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_pre_witness_replacement_allowed(
      **session, input->id, true));

  observer.block.store(true, std::memory_order_release);
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_typed_ingress_observer(
      **session, input->id, &observer, &BlockingTypedIngressObserver::observe));
  ASSERT_TRUE(publish_value(22));
  expected<void, Error> drain_result = make_unexpected(Error{ErrorCode::kNotReady});
  std::thread receiver([&] {
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (!observer.entered.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
      drain_result =
          RuntimeChannelSessionTestAccess::drain_external_inputs(**session, input->owner);
      if (!drain_result) {
        return;
      }
      std::this_thread::yield();
    }
  });
  const auto observer_deadline = std::chrono::steady_clock::now() + 2s;
  while (!observer.entered.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < observer_deadline) {
    std::this_thread::yield();
  }

  // Key assertion: the ordered receive owns the exact incumbent slot through commit, and the
  // arriving value replaces it without recomputing capacity.
  const auto reservations =
      RuntimeChannelSessionTestAccess::typed_ingress_reservations(**session, input->id);
  const auto retained = RuntimeChannelSessionTestAccess::input_reset_state(**session, input->id);
  observer.release.store(true, std::memory_order_release);
  receiver.join();
  ASSERT_TRUE(observer.entered.load(std::memory_order_acquire));
  ASSERT_TRUE(reservations) << runtime_error_message(reservations.error());
  EXPECT_EQ(*reservations, 1U);
  ASSERT_TRUE(retained) << runtime_error_message(retained.error());
  EXPECT_EQ(retained->occupied_samples, 1U);
  EXPECT_EQ(retained->next_arrival_sequence, 2U);
  ASSERT_TRUE(drain_result) << runtime_error_message(drain_result.error());
  const auto released =
      RuntimeChannelSessionTestAccess::typed_ingress_reservations(**session, input->id);
  ASSERT_TRUE(released) << runtime_error_message(released.error());
  EXPECT_EQ(*released, 0U);

  auto binding = (*session)->create_readiness_binding(input->owner,
                                                      RuntimePortBindingRequest{
                                                          .plan_generation = state->plan_generation,
                                                          .run_generation = 1U,
                                                          .kind = RuntimeEventKind::kInputArrival,
                                                          .source = make_source_id(1U),
                                                          .source_clock = ClockId{1U},
                                                          .reference_clock = ClockId{1U},
                                                          .capacity = 2U,
                                                      });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*binding)->poll(TimePoint{1, ClockId{1U}}));
  RuntimeEvent event;
  auto drained = (*binding)->drain(std::span<RuntimeEvent>{&event, 1U});
  ASSERT_TRUE(drained) << runtime_error_message(drained.error());
  ASSERT_EQ(*drained, 1U);
  auto record = (*session)->temporal_sample(event.port, event.source_sequence);
  ASSERT_TRUE(record) << runtime_error_message(record.error());
  EXPECT_EQ(record->source_sequence, 2U);
  EXPECT_EQ(record->arrival_sequence, 2U);
  const std::array selection{RuntimeChannelActivationSelection{
      .port = input->id,
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&*record, 1U},
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(input->owner, 1U, selection));
  auto sample = sink_input.receive();
  ASSERT_TRUE(sample) << runtime_error_message(sample.error());
  EXPECT_EQ(sample->data, 22);
  EXPECT_EQ(sample->metadata.sequence, 2U);
  ASSERT_TRUE((*binding)->retire(event.source_sequence));
  ASSERT_TRUE((*session)->retire_activation(input->owner, 1U, true));
  ASSERT_TRUE(
      (*session)->retire_samples(input->id, std::span<const TemporalSampleRecord>{&*record, 1U}));

  // Key assertion: a non-data catalog head releases the speculative data reservation before
  // retaining ordered control state; with the data slot free, close exercises that release.
  ASSERT_TRUE(publisher->close());
  expected<RuntimeChannelInputResetState, Error> closed_state =
      make_unexpected(Error{ErrorCode::kNotReady});
  const auto close_deadline = std::chrono::steady_clock::now() + 2s;
  while (std::chrono::steady_clock::now() < close_deadline) {
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::drain_external_inputs(**session, input->owner));
    closed_state = RuntimeChannelSessionTestAccess::input_reset_state(**session, input->id);
    if (closed_state && closed_state->end_of_stream_received) {
      break;
    }
    std::this_thread::yield();
  }
  ASSERT_TRUE(closed_state) << runtime_error_message(closed_state.error());
  ASSERT_TRUE(closed_state->end_of_stream_received);
  const auto control_reservations =
      RuntimeChannelSessionTestAccess::typed_ingress_reservations(**session, input->id);
  ASSERT_TRUE(control_reservations) << runtime_error_message(control_reservations.error());
  EXPECT_EQ(*control_reservations, 0U);
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// Generated-schema catalog imports use the same N+1 readiness geometry as POD and Tensor
// imports; schema wire scratch remains sized independently from terminal control retention.
// REQ: CHAN-078
TEST(ChannelBindingParityTest, GeneratedSchemaCatalogImportReservesTerminalReadinessCredit) {
  const std::string endpoint_name = unique_service();
  Graph graph{"schema-catalog-terminal-readiness"};
  const ClockKey clock = graph.bind_shared_clock("runtime-clock", std::make_shared<ManualClock>(0));
  graph.set_default_clock(clock);
  TopicOptions options;
  options.endpoint_name = endpoint_name;
  const auto imported =
      graph.import_topic<schema::StringT>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<GeneratedSchemaSinkOperator>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  const auto* input = planned_operator_port(*state, PortDirection::kInput);
  ASSERT_NE(input, nullptr);
  const std::array replay_ports{RuntimeChannelPortReplayRecord{
      .port = input->id,
      .owner = input->owner,
      .direction = PortDirection::kInput,
      .declaration = const_cast<void*>(static_cast<const void*>(&sink->input)),
      .type_ops = runtime_port_type_ops<schema::StringT>(),
  }};
  auto session = RuntimeChannelSession::create(*state, replay_ports);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto readiness_capacity = (*session)->readiness_capacity(input->owner);
  ASSERT_TRUE(readiness_capacity) << runtime_error_message(readiness_capacity.error());
  EXPECT_EQ(*readiness_capacity, 2U);

  auto client_runtime = ChannelRuntime::create();
  ASSERT_TRUE(client_runtime) << runtime_error_message(client_runtime.error());
  auto publisher =
      client_runtime->create_publisher<schema::StringT>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_for_catalog_subscriber(*publisher));
  expected<std::uint64_t, Error> emitted = make_unexpected(Error{ErrorCode::kNotReady});
  for (std::size_t attempt = 0U; attempt < 8U && !emitted; ++attempt) {
    const auto primed = RuntimeChannelSessionTestAccess::receive_one(**session, input->id);
    if (!primed) {
      ASSERT_EQ(primed.error().code, ErrorCode::kNotReady);
    }
    schema::StringT value;
    value.data = "terminal readiness";
    emitted = publisher->emit(std::move(value));
    if (!emitted) {
      ASSERT_EQ(emitted.error().code, ErrorCode::kNotReady);
      std::this_thread::sleep_for(1ms);
    }
  }
  ASSERT_TRUE(emitted) << runtime_error_message(emitted.error());
  ASSERT_TRUE(wait_for_catalog_input(**session, input->id));

  auto binding = (*session)->create_readiness_binding(input->owner,
                                                      RuntimePortBindingRequest{
                                                          .plan_generation = state->plan_generation,
                                                          .run_generation = 1U,
                                                          .kind = RuntimeEventKind::kInputArrival,
                                                          .source = make_source_id(1U),
                                                          .source_clock = ClockId{1U},
                                                          .reference_clock = ClockId{1U},
                                                          .capacity = *readiness_capacity,
                                                      });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE(publisher->close());
  ASSERT_TRUE((*binding)->poll(TimePoint{1, ClockId{1U}}));
  std::array<RuntimeEvent, 2U> events{};
  auto drained = (*binding)->drain(events);
  ASSERT_TRUE(drained) << runtime_error_message(drained.error());
  ASSERT_EQ(*drained, 2U);
  EXPECT_EQ(events[0].kind, RuntimeEventKind::kInputArrival);
  EXPECT_EQ(events[1].kind, RuntimeEventKind::kInputEndOfStream);
  ASSERT_TRUE((*binding)->retire(events[0].source_sequence));
  ASSERT_TRUE((*binding)->retire(events[1].source_sequence));
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// A terminal retry after one process-local Tensor arm accepts EOS must refresh and submit only the
// still-unresolved required catalog arm. The unresolved catalog peer is attached only after the
// first attempt so the process-local terminal lane stays occupied and exposes duplicate close.
// REQ: CHAN-136
TEST(ChannelBindingParityTest, MixedTensorTerminalRetrySkipsResolvedProcessLocalArm) {
  const std::string endpoint_name = unique_service();
  Graph graph{"mixed-tensor-terminal-retry-" + endpoint_name};
  const ClockKey clock = graph.bind_shared_clock("runtime-clock", std::make_shared<ManualClock>(0));
  graph.set_default_clock(clock);
  const auto source = graph.op<ContractTensorSourceOperator>("source");
  const auto sink = graph.op<TensorSinkOperator>("sink");

  static constexpr std::array axes{TensorAxisBounds{.max_extent = 16U, .max_abs_stride_bytes = 1U}};
  TopicOptions topic_options;
  topic_options.endpoint_name = endpoint_name;
  topic_options.attach = TopicAttachPolicy{.peer = PeerRequirement::kRequired, .max_wait = 2s};
  topic_options.bounds.subscriber =
      SubscriberEndpointBounds{.queue_depth = 1U, .max_borrowed_samples = 1U};
  topic_options.tensor = TensorBoundaryContract{
      .representation = {.memory_kind = MemoryKind::kHost,
                         .dtype = DLDataType{kDLUInt, 8U, 1U},
                         .rank = 1U,
                         .strides = TensorStridePolicy::kContiguousOnly},
      .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 16U, .axes = axes},
      .transfer =
          TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
  };
  auto exported = graph.export_topic<Tensor>("holomq://" + endpoint_name, std::move(topic_options));
  graph.add_flow(source->output,
                 sink->input,
                 ConnectionOptions{.queue_policy = QueuePolicy::kBlocking, .queue_depth = 1U});
  graph.add_flow(source->output,
                 exported.input(),
                 ConnectionOptions{.queue_policy = QueuePolicy::kBlocking, .queue_depth = 1U});

  const ExecutionPlan plan = compile_with_process_local_holomq(graph, CompileOptions{});
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  const auto* planned_output = planned_operator_port(*state, PortDirection::kOutput);
  const auto* planned_input = planned_operator_port(*state, PortDirection::kInput);
  ASSERT_NE(planned_output, nullptr);
  ASSERT_NE(planned_input, nullptr);
  const std::array replay_ports{
      RuntimeChannelPortReplayRecord{
          .port = planned_output->id,
          .owner = planned_output->owner,
          .direction = PortDirection::kOutput,
          .declaration = const_cast<void*>(static_cast<const void*>(&source->output)),
          .type_ops = runtime_port_type_ops<Tensor>(),
      },
      RuntimeChannelPortReplayRecord{
          .port = planned_input->id,
          .owner = planned_input->owner,
          .direction = PortDirection::kInput,
          .declaration = const_cast<void*>(static_cast<const void*>(&sink->input)),
          .type_ops = runtime_port_type_ops<Tensor>(),
      },
  };
  auto session = RuntimeChannelSession::create(*state, replay_ports);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE((*session)->refresh_connections());

  auto first_close = (*session)->close_owner_outputs(planned_output->owner);
  ASSERT_FALSE(first_close);
  EXPECT_EQ(first_close.error().code, ErrorCode::kNotReady);
  auto progress = RuntimeChannelSessionTestAccess::process_local_progress_state(**session);
  ASSERT_TRUE(progress) << runtime_error_message(progress.error());
  EXPECT_TRUE(progress->tracking_valid);
  EXPECT_EQ(progress->published_deliveries, 1U);
  EXPECT_EQ(progress->received_deliveries, 0U);
  EXPECT_EQ(progress->publications_in_flight, 0U);

  auto client_runtime = ChannelRuntime::create();
  ASSERT_TRUE(client_runtime) << runtime_error_message(client_runtime.error());
  auto subscriber =
      client_runtime->create_subscriber<Tensor>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());

  expected<void, Error> retried = make_unexpected(Error{ErrorCode::kNotReady});
  const auto retry_deadline = std::chrono::steady_clock::now() + 2s;
  while (!retried && retried.error().code == ErrorCode::kNotReady &&
         std::chrono::steady_clock::now() < retry_deadline) {
    retried = (*session)->close_owner_outputs(planned_output->owner);
    if (!retried) {
      std::this_thread::sleep_for(1ms);
    }
  }
  ASSERT_TRUE(retried) << runtime_error_message(retried.error());
  ASSERT_TRUE((*session)->close_owner_outputs(planned_output->owner));

  progress = RuntimeChannelSessionTestAccess::process_local_progress_state(**session);
  ASSERT_TRUE(progress) << runtime_error_message(progress.error());
  EXPECT_TRUE(progress->tracking_valid);
  EXPECT_EQ(progress->published_deliveries, 1U);
  EXPECT_EQ(progress->received_deliveries, 0U);
  EXPECT_EQ(progress->publications_in_flight, 0U);

  ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, planned_input->id));
  const auto input_state =
      RuntimeChannelSessionTestAccess::input_reset_state(**session, planned_input->id);
  ASSERT_TRUE(input_state) << runtime_error_message(input_state.error());
  EXPECT_TRUE(input_state->end_of_stream_received);
  progress = RuntimeChannelSessionTestAccess::process_local_progress_state(**session);
  ASSERT_TRUE(progress) << runtime_error_message(progress.error());
  EXPECT_TRUE(progress->tracking_valid);
  EXPECT_EQ(progress->published_deliveries, 1U);
  EXPECT_EQ(progress->received_deliveries, 1U);
  EXPECT_EQ(progress->publications_in_flight, 0U);

  auto readiness_capacity = (*session)->readiness_capacity(planned_input->owner);
  ASSERT_TRUE(readiness_capacity) << runtime_error_message(readiness_capacity.error());
  auto binding = (*session)->create_readiness_binding(
      planned_input->owner,
      RuntimePortBindingRequest{
          .plan_generation = state->plan_generation,
          .run_generation = 1U,
          .kind = RuntimeEventKind::kInputArrival,
          .source = make_source_id(static_cast<std::uint32_t>(planned_output->owner.value)),
          .source_clock = ClockId{1U},
          .reference_clock = ClockId{1U},
          .capacity = *readiness_capacity,
      });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*binding)->poll(TimePoint{1, ClockId{1U}}));
  std::array<RuntimeEvent, 2U> events{};
  auto drained = (*binding)->drain(events);
  ASSERT_TRUE(drained) << runtime_error_message(drained.error());
  ASSERT_EQ(*drained, 1U);
  EXPECT_EQ(events[0].kind, RuntimeEventKind::kInputEndOfStream);
  ASSERT_TRUE((*binding)->retire(events[0].source_sequence));

  auto catalog_terminal = receive_catalog_tensor(*subscriber);
  ASSERT_FALSE(catalog_terminal);
  EXPECT_EQ(catalog_terminal.error().code, ErrorCode::kClosed);
  EXPECT_EQ(RuntimeChannelBindingAccess::accepted_tensor_end_of_stream_sequence(*subscriber), 1U);
  auto repeated_terminal = subscriber->receive_sample();
  ASSERT_FALSE(repeated_terminal);
  EXPECT_EQ(repeated_terminal.error().code, ErrorCode::kClosed);
  EXPECT_EQ(RuntimeChannelBindingAccess::accepted_tensor_end_of_stream_sequence(*subscriber), 1U);

  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// One OutputState validates every fanout arm against one immutable Tensor contract. A second
// catalog endpoint with different geometry must not replace the contract established by the first.
// REQ: SYSTEM-001
TEST(ChannelBindingParityTest, CatalogTensorFanoutRejectsConflictingOutputContracts) {
  Graph graph{"catalog-tensor-contract-fanout"};
  const ClockKey default_clock =
      graph.bind_shared_clock("runtime-clock", std::make_shared<ManualClock>(0));
  graph.set_default_clock(default_clock);
  const auto source = graph.op<ContractTensorSourceOperator>("source");
  constexpr std::array first_axes{TensorAxisBounds{.max_extent = 8U, .max_abs_stride_bytes = 1U}};
  TopicOptions first_options;
  first_options.tensor = TensorBoundaryContract{
      .representation = {.memory_kind = MemoryKind::kHost,
                         .dtype = DLDataType{kDLUInt, 8U, 1U},
                         .rank = 1U,
                         .strides = TensorStridePolicy::kContiguousOnly},
      .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 8U, .axes = first_axes},
      .transfer =
          TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
  };
  constexpr std::array second_axes{TensorAxisBounds{.max_extent = 16U, .max_abs_stride_bytes = 1U}};
  TopicOptions second_options;
  second_options.tensor = TensorBoundaryContract{
      .representation = {.memory_kind = MemoryKind::kHost,
                         .dtype = DLDataType{kDLUInt, 8U, 1U},
                         .rank = 1U,
                         .strides = TensorStridePolicy::kContiguousOnly},
      .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 16U, .axes = second_axes},
      .transfer =
          TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
  };
  auto first = graph.export_topic<Tensor>("holomq://" + unique_service(), std::move(first_options));
  auto second =
      graph.export_topic<Tensor>("holomq://" + unique_service(), std::move(second_options));
  graph.add_flow(source->output, first.input());
  graph.add_flow(source->output, second.input());

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  const auto* planned_output = planned_operator_port(*state, PortDirection::kOutput);
  ASSERT_NE(planned_output, nullptr);
  const std::array replay_ports{RuntimeChannelPortReplayRecord{
      .port = planned_output->id,
      .owner = planned_output->owner,
      .direction = PortDirection::kOutput,
      .declaration = const_cast<void*>(static_cast<const void*>(&source->output)),
      .type_ops = runtime_port_type_ops<Tensor>(),
  }};

  auto session = RuntimeChannelSession::create(*state, replay_ports);
  ASSERT_FALSE(session);
  EXPECT_EQ(session.error().code, ErrorCode::kProtocolError);
}

// Runtime rejects a catalog whose service-static Tensor policy differs from the endpoint and
// source-Graph boundary contract only by memory conversion. The same mismatch is checked at both
// executor reconstruction and Channel binding startup.
// REQ: SYSTEM-023
TEST(ChannelBindingParityTest, CatalogTensorRejectsMemoryConversionProjectionMismatch) {
  Graph graph{"catalog-tensor-conversion-projection-mismatch"};
  const ClockKey default_clock =
      graph.bind_shared_clock("runtime-clock", std::make_shared<ManualClock>(0));
  graph.set_default_clock(default_clock);
  const auto source = graph.op<ContractTensorSourceOperator>("source");
  TopicOptions options;
  options.tensor = TensorBoundaryContract{
      .representation = {.memory_kind = MemoryKind::kHost,
                         .dtype = DLDataType{kDLUInt, 8U, 1U},
                         .rank = 1U,
                         .strides = TensorStridePolicy::kContiguousOnly},
      .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 16U},
      .transfer =
          TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
      .memory_conversion = MemoryConversionPolicy::kDisable,
  };
  auto exported = graph.export_topic<Tensor>("holomq://" + unique_service(), std::move(options));
  graph.add_flow(source->output, exported.input());

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  const auto* catalog_state = ChannelEndpointCatalogAccess::state(state->endpoint_catalog);
  ASSERT_NE(catalog_state, nullptr);
  ASSERT_EQ(catalog_state->endpoints.size(), 1U);

  auto corrupted_catalog = std::make_shared<ChannelEndpointCatalogState>(*catalog_state);
  ASSERT_TRUE(corrupted_catalog->endpoints.front().tensor);
  ASSERT_TRUE(corrupted_catalog->endpoints.front().service_static.tensor);
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  corrupted_catalog->endpoints.front().service_static.tensor->memory_conversion =
      MemoryConversionPolicy::kAllowIfSafe;
  auto corrupted_state = std::make_shared<ExecutionPlanState>(*state);
  corrupted_state->endpoint_catalog = ChannelEndpointCatalogAccess::make(corrupted_catalog);

  auto reconstructed = GraphExec::reconstruct(corrupted_state, RunGenerationId{1U});
  ASSERT_FALSE(reconstructed);
  EXPECT_EQ(reconstructed.error().message, "RUNTIME_EXTERNAL_BINDING_UNAVAILABLE");

  const auto* planned_output = planned_operator_port(*corrupted_state, PortDirection::kOutput);
  ASSERT_NE(planned_output, nullptr);
  ContractTensorSourceOperator runtime_source;
  OperatorSpec runtime_source_spec;
  runtime_source.setup(runtime_source_spec);
  const std::array replay_ports{RuntimeChannelPortReplayRecord{
      .port = planned_output->id,
      .owner = planned_output->owner,
      .direction = PortDirection::kOutput,
      .declaration = static_cast<void*>(&runtime_source.output),
      .type_ops = runtime_port_type_ops<Tensor>(),
  }};
  auto session = RuntimeChannelSession::create(*corrupted_state, replay_ports);
  ASSERT_FALSE(session);
  EXPECT_EQ(session.error().code, ErrorCode::kProtocolError);
}

// Every catalog endpoint is internally consistent, but one output cannot adopt two different
// consumer memory-conversion policies across its external fanout.
// REQ: SYSTEM-023
TEST(ChannelBindingParityTest, CatalogTensorFanoutRejectsConflictingMemoryConversion) {
  Graph graph{"catalog-tensor-conversion-fanout"};
  const ClockKey default_clock =
      graph.bind_shared_clock("runtime-clock", std::make_shared<ManualClock>(0));
  graph.set_default_clock(default_clock);
  const auto source = graph.op<ContractTensorSourceOperator>("source");
  const auto make_options = [](const MemoryConversionPolicy conversion) {
    TopicOptions options;
    options.tensor = TensorBoundaryContract{
        .representation = {.memory_kind = MemoryKind::kHost,
                           .dtype = DLDataType{kDLUInt, 8U, 1U},
                           .rank = 1U,
                           .strides = TensorStridePolicy::kContiguousOnly},
        .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 16U},
        .transfer =
            TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
        .memory_conversion = conversion,
    };
    return options;
  };
  auto first = graph.export_topic<Tensor>("holomq://" + unique_service(),
                                          make_options(MemoryConversionPolicy::kDisable));
  auto second = graph.export_topic<Tensor>("holomq://" + unique_service(),
                                           make_options(MemoryConversionPolicy::kAllowIfSafe));
  graph.add_flow(source->output, first.input());
  graph.add_flow(source->output, second.input());

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  const auto* planned_output = planned_operator_port(*state, PortDirection::kOutput);
  ASSERT_NE(planned_output, nullptr);
  ContractTensorSourceOperator runtime_source;
  OperatorSpec runtime_source_spec;
  runtime_source.setup(runtime_source_spec);
  const std::array replay_ports{RuntimeChannelPortReplayRecord{
      .port = planned_output->id,
      .owner = planned_output->owner,
      .direction = PortDirection::kOutput,
      .declaration = static_cast<void*>(&runtime_source.output),
      .type_ops = runtime_port_type_ops<Tensor>(),
  }};

  auto session = RuntimeChannelSession::create(*state, replay_ports);
  ASSERT_FALSE(session);
  EXPECT_EQ(session.error().code, ErrorCode::kProtocolError);
}

// Catalog fanout stages one foreign Tensor copy and shares its exportable BufferHandle across
// every external service. Re-emitting an imported HoloMQ Tensor stays on the zero-copy path.
// REQ: SYSTEM-006
// REQ: SYSTEM-007
// REQ: SYSTEM-009
TEST(ChannelBindingParityTest, CatalogTensorFanoutCopiesOnceAndReusesExportableBacking) {
  Graph graph{"catalog-tensor-shared-export"};
  const ClockKey default_clock =
      graph.bind_shared_clock("runtime-clock", std::make_shared<ManualClock>(0));
  graph.set_default_clock(default_clock);
  auto source = graph.op<ContractTensorSourceOperator>("source");
  static constexpr std::array axes{TensorAxisBounds{.max_extent = 16U, .max_abs_stride_bytes = 1U}};
  const auto make_options = [&](const std::string& endpoint_name) {
    TopicOptions options;
    options.endpoint_name = endpoint_name;
    options.bounds.subscriber = SubscriberEndpointBounds{
        .queue_depth = 2U,
        .max_borrowed_samples = 2U,
    };
    options.tensor = TensorBoundaryContract{
        .representation = {.memory_kind = MemoryKind::kHost,
                           .dtype = DLDataType{kDLUInt, 8U, 1U},
                           .rank = 1U,
                           .strides = TensorStridePolicy::kContiguousOnly},
        .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 16U, .axes = axes},
        .transfer =
            TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
    };
    return options;
  };
  const std::string missing_name = unique_service();
  const std::string first_name = unique_service();
  const std::string second_name = unique_service();
  auto missing = graph.export_topic<Tensor>("holomq://" + missing_name, make_options(missing_name));
  auto first = graph.export_topic<Tensor>("holomq://" + first_name, make_options(first_name));
  auto second = graph.export_topic<Tensor>("holomq://" + second_name, make_options(second_name));
  graph.add_flow(source->output, missing.input());
  graph.add_flow(source->output, first.input());
  graph.add_flow(source->output, second.input());

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  const auto* planned_output = planned_operator_port(*state, PortDirection::kOutput);
  ASSERT_NE(planned_output, nullptr);
  ContractTensorSourceOperator runtime_source;
  OperatorSpec runtime_source_spec;
  runtime_source.setup(runtime_source_spec);
  const std::array replay_ports{RuntimeChannelPortReplayRecord{
      .port = planned_output->id,
      .owner = planned_output->owner,
      .direction = PortDirection::kOutput,
      .declaration = static_cast<void*>(&runtime_source.output),
      .type_ops = runtime_port_type_ops<Tensor>(),
  }};
  auto session = RuntimeChannelSession::create(*state, replay_ports);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto client_runtime = ChannelRuntime::create();
  ASSERT_TRUE(client_runtime) << runtime_error_message(client_runtime.error());
  auto first_subscriber =
      client_runtime->create_subscriber<Tensor>(plan.endpoint_catalog(), first_name);
  auto second_subscriber =
      client_runtime->create_subscriber<Tensor>(plan.endpoint_catalog(), second_name);
  ASSERT_TRUE(first_subscriber) << runtime_error_message(first_subscriber.error());
  ASSERT_TRUE(second_subscriber) << runtime_error_message(second_subscriber.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  constexpr std::size_t kByteOffset = 2U;
  auto tracked = make_host_copy_tracking_tensor(kByteOffset);
  ASSERT_NE(tracked.tensor.data(), nullptr);
  ASSERT_TRUE((*session)->prepare_activation(planned_output->owner, 1U, {}));
  ASSERT_TRUE(runtime_source.output.emit(Sample<Tensor>{tracked.tensor, {}}));
  ASSERT_TRUE((*session)->retire_activation(planned_output->owner, 1U, true));
  auto first_received = receive_catalog_tensor(*first_subscriber);
  auto second_received = receive_catalog_tensor(*second_subscriber);
  ASSERT_TRUE(first_received) << runtime_error_message(first_received.error());
  ASSERT_TRUE(second_received) << runtime_error_message(second_received.error());
  EXPECT_EQ(tracked.state->host_synchronizations.load(std::memory_order_relaxed), 1U);
  const std::size_t logical_byte_size = tracked.state->data.size() - kByteOffset;
  ASSERT_EQ(first_received->data.nbytes(), static_cast<std::int64_t>(logical_byte_size));
  ASSERT_EQ(second_received->data.nbytes(), static_cast<std::int64_t>(logical_byte_size));
  EXPECT_EQ(
      std::memcmp(
          first_received->data.data(), tracked.state->data.data() + kByteOffset, logical_byte_size),
      0);
  EXPECT_EQ(std::memcmp(second_received->data.data(),
                        tracked.state->data.data() + kByteOffset,
                        logical_byte_size),
            0);
  EXPECT_EQ(first_received->data.data(), second_received->data.data());

  ASSERT_TRUE((*session)->prepare_activation(planned_output->owner, 2U, {}));
  ASSERT_TRUE(runtime_source.output.emit(Sample<Tensor>{first_received->data, {}}));
  ASSERT_TRUE((*session)->retire_activation(planned_output->owner, 2U, true));
  auto republished = receive_catalog_tensor(*first_subscriber);
  ASSERT_TRUE(republished) << runtime_error_message(republished.error());
  EXPECT_EQ(
      std::memcmp(
          republished->data.data(), tracked.state->data.data() + kByteOffset, logical_byte_size),
      0);
  EXPECT_EQ(tracked.state->host_synchronizations.load(std::memory_order_relaxed), 1U);
}

// A graph-bound device Tensor uses its retained producer stream to stage one ordered D2H copy.
// Both catalog services publish the same host-pageable backing rather than copying per endpoint.
// REQ: CHAN-091
// REQ: SYSTEM-007
TEST(ChannelBindingParityTest, CatalogTensorDeviceFanoutStagesOnceAfterProducerStream) {
  test::skip_without_cuda_device();

  int device_ordinal = -1;
  ASSERT_EQ(cudaGetDevice(&device_ordinal), cudaSuccess);

  Graph graph{"catalog-tensor-device-export"};
  const ClockKey default_clock =
      graph.bind_shared_clock("runtime-clock", std::make_shared<ManualClock>(0));
  graph.set_default_clock(default_clock);
  auto source = graph.op<DeviceContractTensorSourceOperator>("source");
  static constexpr std::array axes{TensorAxisBounds{.max_extent = 16U, .max_abs_stride_bytes = 1U}};
  const auto make_options = [&](const std::string& endpoint_name) {
    TopicOptions options;
    options.endpoint_name = endpoint_name;
    options.bounds.subscriber = SubscriberEndpointBounds{
        .queue_depth = 2U,
        .max_borrowed_samples = 2U,
    };
    options.tensor = TensorBoundaryContract{
        .representation = {.memory_kind = MemoryKind::kHost,
                           .dtype = DLDataType{kDLUInt, 8U, 1U},
                           .rank = 1U,
                           .strides = TensorStridePolicy::kContiguousOnly},
        .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 16U, .axes = axes},
        .transfer =
            TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
    };
    return options;
  };
  const std::string first_name = unique_service();
  const std::string second_name = unique_service();
  // Publish one device output to two independent subscribers over external catalog topics.
  auto first = graph.export_topic<Tensor>("holomq://" + first_name, make_options(first_name));
  auto second = graph.export_topic<Tensor>("holomq://" + second_name, make_options(second_name));
  graph.add_flow(source->output, first.input());
  graph.add_flow(source->output, second.input());

  // Device-visible outputs require deployment-owned device evidence; ordinary compile(graph)
  // correctly rejects an implicit device-zero assumption with IMPLICIT_DEVICE_ZERO_REJECTED.
  const auto snapshot = GraphCompilerAccess::canonical_snapshot(graph);
  const auto source_vertex =
      std::find_if(snapshot->vertices.begin(), snapshot->vertices.end(), [](const auto& vertex) {
        return vertex.kind == VertexKind::kOperator && vertex.name == "source";
      });
  const auto source_port =
      source_vertex == snapshot->vertices.end()
          ? snapshot->ports.end()
          : std::find_if(snapshot->ports.begin(), snapshot->ports.end(), [&](const auto& port) {
              return port.owner == source_vertex->id && port.direction == PortDirection::kOutput &&
                     port.name == "output";
            });
  ASSERT_NE(source_port, snapshot->ports.end());
  const std::array device_bindings{
      TensorDeviceBindingRecord{.port = PortId{source_port->id.value},
                                .device = DeviceId{device_ordinal}},
  };
  const ExecutionPlan plan =
      compile_with_runtime_evidence(graph,
                                    {},
                                    RuntimeCompileEvidenceView{
                                        .package_capabilities = runtime_package_capability_view(),
                                        .tensor_device_bindings = device_bindings,
                                    });
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  const auto* planned_output = planned_operator_port(*state, PortDirection::kOutput);
  ASSERT_NE(planned_output, nullptr);
  DeviceContractTensorSourceOperator runtime_source;
  OperatorSpec runtime_source_spec;
  runtime_source.setup(runtime_source_spec);
  const std::array replay_ports{RuntimeChannelPortReplayRecord{
      .port = planned_output->id,
      .owner = planned_output->owner,
      .direction = PortDirection::kOutput,
      .declaration = static_cast<void*>(&runtime_source.output),
      .type_ops = runtime_port_type_ops<Tensor>(),
  }};
  auto session = RuntimeChannelSession::create(*state, replay_ports);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto client_runtime = ChannelRuntime::create();
  ASSERT_TRUE(client_runtime) << runtime_error_message(client_runtime.error());
  auto first_subscriber =
      client_runtime->create_subscriber<Tensor>(plan.endpoint_catalog(), first_name);
  auto second_subscriber =
      client_runtime->create_subscriber<Tensor>(plan.endpoint_catalog(), second_name);
  ASSERT_TRUE(first_subscriber) << runtime_error_message(first_subscriber.error());
  ASSERT_TRUE(second_subscriber) << runtime_error_message(second_subscriber.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  // Produce bytes asynchronously in a foreign device allocation and retain that ordering stream.
  constexpr std::size_t kByteCount = 8U;
  auto pool =
      std::make_shared<BlockMemoryPool>(MemoryStorageType::kDevice, kByteCount, 1U, device_ordinal);
  ASSERT_TRUE(pool->initialize_pool());
  test::UniqueCudaStream producer_stream;
  ASSERT_EQ(test::create_non_blocking_cuda_stream(producer_stream), cudaSuccess);
  const std::array<std::int64_t, 1U> shape{static_cast<std::int64_t>(kByteCount)};
  auto device_tensor = make_tensor(shape,
                                   DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
                                   std::nullopt,
                                   MemoryStorageType::kDevice,
                                   pool,
                                   ::cuda::stream_ref{producer_stream.get()});
  ASSERT_TRUE(device_tensor) << runtime_error_message(device_tensor.error());
  auto device_data = device_tensor->mutable_data();
  ASSERT_TRUE(device_data) << runtime_error_message(device_data.error());
  const std::array<std::byte, kByteCount> expected{std::byte{0x81},
                                                   std::byte{0x72},
                                                   std::byte{0x63},
                                                   std::byte{0x54},
                                                   std::byte{0x45},
                                                   std::byte{0x36},
                                                   std::byte{0x27},
                                                   std::byte{0x18}};
  ASSERT_EQ(cudaMemcpyAsync(*device_data,
                            expected.data(),
                            expected.size(),
                            cudaMemcpyHostToDevice,
                            producer_stream.get()),
            cudaSuccess);
  ASSERT_TRUE(device_tensor->set_readiness_stream(producer_stream.get()));

  // Runtime must stage one ordered D2H copy before publishing either external descriptor.
  ASSERT_TRUE((*session)->prepare_activation(planned_output->owner, 1U, {}));
  ASSERT_TRUE(runtime_source.output.emit(Sample<Tensor>{*device_tensor, {}}));
  ASSERT_TRUE((*session)->retire_activation(planned_output->owner, 1U, true));
  auto first_received = receive_catalog_tensor(*first_subscriber);
  auto second_received = receive_catalog_tensor(*second_subscriber);
  ASSERT_TRUE(first_received) << runtime_error_message(first_received.error());
  ASSERT_TRUE(second_received) << runtime_error_message(second_received.error());
  ASSERT_EQ(first_received->data.nbytes(), static_cast<std::int64_t>(expected.size()));
  ASSERT_EQ(second_received->data.nbytes(), static_cast<std::int64_t>(expected.size()));
  EXPECT_EQ(std::memcmp(first_received->data.data(), expected.data(), expected.size()), 0);
  EXPECT_EQ(std::memcmp(second_received->data.data(), expected.data(), expected.size()), 0);
  // Pointer identity proves both subscribers retained the one shared staged host backing.
  EXPECT_EQ(first_received->data.data(), second_received->data.data());
}

// A device catalog contract materializes its HoloMQ pools on the compiler-resolved device. One
// producer-stream-ordered D2D copy supplies both services, and both imports retain the same
// backing.
//
// REQ: CHAN-092
// REQ: SYSTEM-007
TEST(ChannelBindingParityTest, CatalogTensorDeviceFanoutStagesOnceIntoHoloMQDevicePool) {
  test::skip_without_cuda_device();

  int device_ordinal = -1;
  ASSERT_EQ(cudaGetDevice(&device_ordinal), cudaSuccess);
  test::UniqueCudaStream producer_stream;
  ASSERT_EQ(test::create_non_blocking_cuda_stream(producer_stream), cudaSuccess);

  Graph graph{"catalog-tensor-device-pool-export"};
  const ClockKey default_clock =
      graph.bind_shared_clock("runtime-clock", std::make_shared<ManualClock>(0));
  graph.set_default_clock(default_clock);
  auto source = graph.op<DeviceContractTensorSourceOperator>("source");
  static constexpr std::array axes{TensorAxisBounds{.max_extent = 16U, .max_abs_stride_bytes = 1U}};
  const auto make_options = [&](const std::string& endpoint_name) {
    TopicOptions options;
    options.endpoint_name = endpoint_name;
    options.bounds.subscriber = SubscriberEndpointBounds{
        .queue_depth = 2U,
        .max_borrowed_samples = 2U,
    };
    options.tensor = TensorBoundaryContract{
        .representation = {.memory_kind = MemoryKind::kCudaDevice,
                           .dtype = DLDataType{kDLUInt, 8U, 1U},
                           .rank = 1U,
                           .strides = TensorStridePolicy::kContiguousOnly},
        .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 16U, .axes = axes},
        .transfer =
            TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
    };
    return options;
  };
  const std::string first_name = unique_service();
  const std::string second_name = unique_service();
  auto first = graph.export_topic<Tensor>("holomq://" + first_name, make_options(first_name));
  auto second = graph.export_topic<Tensor>("holomq://" + second_name, make_options(second_name));
  graph.add_flow(source->output, first.input());
  graph.add_flow(source->output, second.input());

  const auto snapshot = GraphCompilerAccess::canonical_snapshot(graph);
  const auto source_vertex =
      std::find_if(snapshot->vertices.begin(), snapshot->vertices.end(), [](const auto& vertex) {
        return vertex.kind == VertexKind::kOperator && vertex.name == "source";
      });
  const auto source_port =
      source_vertex == snapshot->vertices.end()
          ? snapshot->ports.end()
          : std::find_if(snapshot->ports.begin(), snapshot->ports.end(), [&](const auto& port) {
              return port.owner == source_vertex->id && port.direction == PortDirection::kOutput &&
                     port.name == "output";
            });
  ASSERT_NE(source_port, snapshot->ports.end());
  std::vector<TensorDeviceBindingRecord> device_bindings;
  device_bindings.reserve(snapshot->ports.size());
  for (const auto& port : snapshot->ports) {
    ASSERT_EQ(port.payload_identity.kind, PayloadIdentityKind::kTensor);
    device_bindings.push_back(TensorDeviceBindingRecord{.port = PortId{port.id.value},
                                                        .device = DeviceId{device_ordinal}});
  }
  const ExecutionPlan plan =
      compile_with_runtime_evidence(graph,
                                    {},
                                    RuntimeCompileEvidenceView{
                                        .package_capabilities = runtime_package_capability_view(),
                                        .tensor_device_bindings = device_bindings,
                                    });
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->tensor_interface_pools.size(), 2U);
  EXPECT_TRUE(std::all_of(state->tensor_interface_pools.begin(),
                          state->tensor_interface_pools.end(),
                          [&](const auto& pool) {
                            return pool.memory_kind == MemoryKind::kCudaDevice &&
                                   pool.device == DeviceId{device_ordinal};
                          }));
  const auto* planned_output = planned_operator_port(*state, PortDirection::kOutput);
  ASSERT_NE(planned_output, nullptr);
  DeviceContractTensorSourceOperator runtime_source;
  OperatorSpec runtime_source_spec;
  runtime_source.setup(runtime_source_spec);
  const std::array replay_ports{RuntimeChannelPortReplayRecord{
      .port = planned_output->id,
      .owner = planned_output->owner,
      .direction = PortDirection::kOutput,
      .declaration = static_cast<void*>(&runtime_source.output),
      .type_ops = runtime_port_type_ops<Tensor>(),
  }};
  auto session = RuntimeChannelSession::create(*state, replay_ports);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto client_runtime = ChannelRuntime::create();
  ASSERT_TRUE(client_runtime) << runtime_error_message(client_runtime.error());
  auto first_subscriber =
      client_runtime->create_subscriber<Tensor>(plan.endpoint_catalog(), first_name);
  auto second_subscriber =
      client_runtime->create_subscriber<Tensor>(plan.endpoint_catalog(), second_name);
  ASSERT_TRUE(first_subscriber) << runtime_error_message(first_subscriber.error());
  ASSERT_TRUE(second_subscriber) << runtime_error_message(second_subscriber.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  constexpr std::size_t kByteCount = 8U;
  auto source_pool =
      std::make_shared<BlockMemoryPool>(MemoryStorageType::kDevice, kByteCount, 1U, device_ordinal);
  ASSERT_TRUE(source_pool->initialize_pool());
  const std::array<std::int64_t, 1U> shape{static_cast<std::int64_t>(kByteCount)};
  auto device_tensor = make_tensor(shape,
                                   DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
                                   std::nullopt,
                                   MemoryStorageType::kDevice,
                                   source_pool,
                                   ::cuda::stream_ref{producer_stream.get()});
  ASSERT_TRUE(device_tensor) << runtime_error_message(device_tensor.error());
  auto device_data = device_tensor->mutable_data();
  ASSERT_TRUE(device_data) << runtime_error_message(device_data.error());
  const std::array<std::byte, kByteCount> expected{std::byte{0x91},
                                                   std::byte{0x82},
                                                   std::byte{0x73},
                                                   std::byte{0x64},
                                                   std::byte{0x55},
                                                   std::byte{0x46},
                                                   std::byte{0x37},
                                                   std::byte{0x28}};
  ASSERT_EQ(cudaMemcpyAsync(*device_data,
                            expected.data(),
                            expected.size(),
                            cudaMemcpyHostToDevice,
                            producer_stream.get()),
            cudaSuccess);
  ASSERT_TRUE(device_tensor->set_readiness_stream(producer_stream.get()));

  ASSERT_TRUE((*session)->prepare_activation(planned_output->owner, 1U, {}));
  ASSERT_TRUE(runtime_source.output.emit(Sample<Tensor>{*device_tensor, {}}));
  ASSERT_TRUE((*session)->retire_activation(planned_output->owner, 1U, true));

  test::UniqueCudaStream consumer_stream;
  ASSERT_EQ(test::create_non_blocking_cuda_stream(consumer_stream), cudaSuccess);
  std::array<std::byte, kByteCount> first_actual{};
  std::array<std::byte, kByteCount> second_actual{};
  {
    // This inner scope destroys both imported Tensors on scope exit, before consumer_stream is
    // destroyed, so their deferred releases retain a valid CUDA stream through teardown.
    auto first_received = receive_catalog_tensor(*first_subscriber);
    auto second_received = receive_catalog_tensor(*second_subscriber);
    ASSERT_TRUE(first_received) << runtime_error_message(first_received.error());
    ASSERT_TRUE(second_received) << runtime_error_message(second_received.error());
    EXPECT_EQ(first_received->data.device().device_type, kDLCUDA);
    EXPECT_EQ(second_received->data.device().device_type, kDLCUDA);
    EXPECT_EQ(first_received->data.data(), second_received->data.data());
    ASSERT_TRUE(first_received->data.synchronize_for_consumer_stream(consumer_stream.get()));
    ASSERT_TRUE(second_received->data.synchronize_for_consumer_stream(consumer_stream.get()));
    ASSERT_EQ(cudaMemcpyAsync(first_actual.data(),
                              first_received->data.data(),
                              first_actual.size(),
                              cudaMemcpyDeviceToHost,
                              consumer_stream.get()),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(second_actual.data(),
                              second_received->data.data(),
                              second_actual.size(),
                              cudaMemcpyDeviceToHost,
                              consumer_stream.get()),
              cudaSuccess);
    ASSERT_TRUE(first_received->data.set_deallocation_stream(consumer_stream.get()));
    ASSERT_TRUE(second_received->data.set_deallocation_stream(consumer_stream.get()));
    ASSERT_EQ(cudaStreamSynchronize(consumer_stream.get()), cudaSuccess);
    // NB: this sync runs BEFORE the received Tensors are destroyed at scope exit below, so it does
    // NOT drain their deferred HoloMQ ref release. set_deallocation_stream arms a cuLaunchHostFunc
    // release_ref that the ConstBuffer destructor enqueues on consumer_stream only at `}`, after
    // this sync. The test is safe solely because it never publishes again after the import. If a
    // re-publication on a finite device pool is ever added here, await slot recovery first (see
    // wait_for_recycled_tensor_loan in channel_tensor_loan.cpp) rather than relying on this sync.
  }
  EXPECT_EQ(first_actual, expected);
  EXPECT_EQ(second_actual, expected);
}

// Public output placement supplies the private device for a pinned catalog staging pool. The
// producer copy remains stream ordered, while the received public Tensor keeps DLPack's fixed
// kDLCUDAHost/device-zero identity.
// REQ: CHAN-095
// REQ: EXTERNAL-007
TEST(ChannelBindingParityTest, CatalogTensorPinnedExportUsesPlacedOutputDevice) {
  test::skip_without_cuda_device();

  int device_ordinal = -1;
  ASSERT_EQ(cudaGetDevice(&device_ordinal), cudaSuccess);
  const std::string endpoint_name = unique_service();
  Graph graph{"catalog-tensor-pinned-export"};
  const ClockKey default_clock =
      graph.bind_shared_clock("runtime-clock", std::make_shared<ManualClock>(0));
  graph.set_default_clock(default_clock);
  auto source = graph.op<DeviceContractTensorSourceOperator>("source");
  static constexpr std::array axes{TensorAxisBounds{.max_extent = 16U, .max_abs_stride_bytes = 1U}};
  TopicOptions topic_options;
  topic_options.endpoint_name = endpoint_name;
  topic_options.bounds.subscriber = SubscriberEndpointBounds{
      .queue_depth = 1U,
      .max_borrowed_samples = 1U,
  };
  topic_options.tensor = TensorBoundaryContract{
      .representation = {.memory_kind = MemoryKind::kPinnedHost,
                         .dtype = DLDataType{kDLUInt, 8U, 1U},
                         .rank = 1U,
                         .strides = TensorStridePolicy::kContiguousOnly},
      .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 16U, .axes = axes},
      .transfer =
          TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
  };
  auto exported = graph.export_topic<Tensor>("holomq://" + endpoint_name, topic_options);
  graph.add_flow(source->output, exported.input());

  CompileOptions compile_options;
  compile_options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
      .operator_path = "source",
      .output_port = "output",
      .device = DeviceId{device_ordinal},
  });
  const ExecutionPlan plan = compile(graph, std::move(compile_options));
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->tensor_interface_pools.size(), 1U);
  EXPECT_EQ(state->tensor_interface_pools.front().memory_kind, MemoryKind::kPinnedHost);
  EXPECT_EQ(state->tensor_interface_pools.front().device, DeviceId{device_ordinal});

  // Service-stream owner evidence is optional when no Green Context is active. Keep this ordinary
  // pinned export independent of that unused evidence while retaining the planned pool device.
  auto runtime_state = std::make_shared<ExecutionPlanState>(*state);
  ASSERT_TRUE(runtime_state->cuda_green_context_pools.empty());
  for (auto& transfer : runtime_state->tensor_transfer_plans) {
    transfer.service_owner_id.clear();
  }

  const auto* planned_output = planned_operator_port(*runtime_state, PortDirection::kOutput);
  ASSERT_NE(planned_output, nullptr);
  DeviceContractTensorSourceOperator runtime_source;
  OperatorSpec runtime_source_spec;
  runtime_source.setup(runtime_source_spec);
  const std::array replay_ports{RuntimeChannelPortReplayRecord{
      .port = planned_output->id,
      .owner = planned_output->owner,
      .direction = PortDirection::kOutput,
      .declaration = static_cast<void*>(&runtime_source.output),
      .type_ops = runtime_port_type_ops<Tensor>(),
  }};
  auto session = RuntimeChannelSession::create(*runtime_state, replay_ports);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto client_runtime = ChannelRuntime::create();
  ASSERT_TRUE(client_runtime) << runtime_error_message(client_runtime.error());
  auto subscriber =
      client_runtime->create_subscriber<Tensor>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  constexpr std::size_t kByteCount = 8U;
  auto source_pool =
      std::make_shared<BlockMemoryPool>(MemoryStorageType::kDevice, kByteCount, 1U, device_ordinal);
  ASSERT_TRUE(source_pool->initialize_pool());
  test::UniqueCudaStream producer_stream;
  ASSERT_EQ(test::create_non_blocking_cuda_stream(producer_stream), cudaSuccess);
  const std::array<std::int64_t, 1U> shape{static_cast<std::int64_t>(kByteCount)};
  auto device_tensor = make_tensor(shape,
                                   DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
                                   std::nullopt,
                                   MemoryStorageType::kDevice,
                                   source_pool,
                                   ::cuda::stream_ref{producer_stream.get()});
  ASSERT_TRUE(device_tensor) << runtime_error_message(device_tensor.error());
  auto device_data = device_tensor->mutable_data();
  ASSERT_TRUE(device_data) << runtime_error_message(device_data.error());
  const std::array<std::byte, kByteCount> expected{std::byte{0xA1},
                                                   std::byte{0xB2},
                                                   std::byte{0xC3},
                                                   std::byte{0xD4},
                                                   std::byte{0xE5},
                                                   std::byte{0xF6},
                                                   std::byte{0x17},
                                                   std::byte{0x28}};
  ASSERT_EQ(cudaMemcpyAsync(*device_data,
                            expected.data(),
                            expected.size(),
                            cudaMemcpyHostToDevice,
                            producer_stream.get()),
            cudaSuccess);
  ASSERT_TRUE(device_tensor->set_readiness_stream(producer_stream.get()));

  ASSERT_TRUE((*session)->prepare_activation(planned_output->owner, 1U, {}));
  ASSERT_TRUE(runtime_source.output.emit(Sample<Tensor>{*device_tensor, {}}));
  ASSERT_TRUE((*session)->retire_activation(planned_output->owner, 1U, true));
  {
    auto received = subscriber->receive_sample();
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    EXPECT_EQ(received->data.device().device_type, kDLCUDAHost);
    EXPECT_EQ(received->data.device().device_id, 0);
    EXPECT_TRUE(received->data.is_read_only());
    ASSERT_TRUE(received->data.synchronize_for_host());
    EXPECT_EQ(std::memcmp(received->data.data(), expected.data(), expected.size()), 0);
  }
}

// The require-exportable policy rejects foreign storage before any catalog endpoint submits or
// consumes the foreign Tensor's host synchronization contract.
// REQ: SYSTEM-006
// REQ: SYSTEM-007
// REQ: SYSTEM-009
TEST(ChannelBindingParityTest, CatalogTensorRequireExportableRejectsBeforeFanoutSubmission) {
  Graph graph{"catalog-tensor-require-exportable"};
  const ClockKey default_clock =
      graph.bind_shared_clock("runtime-clock", std::make_shared<ManualClock>(0));
  graph.set_default_clock(default_clock);
  auto source = graph.op<RequireExportableTensorSourceOperator>("source");
  static constexpr std::array axes{TensorAxisBounds{.max_extent = 16U, .max_abs_stride_bytes = 1U}};
  const auto make_options = [&](const std::string& endpoint_name) {
    TopicOptions options;
    options.endpoint_name = endpoint_name;
    options.bounds.subscriber = SubscriberEndpointBounds{
        .queue_depth = 2U,
        .max_borrowed_samples = 2U,
    };
    options.tensor = TensorBoundaryContract{
        .representation = {.memory_kind = MemoryKind::kHost,
                           .dtype = DLDataType{kDLUInt, 8U, 1U},
                           .rank = 1U,
                           .strides = TensorStridePolicy::kContiguousOnly},
        .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 16U, .axes = axes},
        .transfer = TensorTransferPolicy{.foreign_emit_policy =
                                             InterfaceTensorEmitPolicy::kRequireExportable},
    };
    return options;
  };
  const std::string first_name = unique_service();
  const std::string second_name = unique_service();
  auto first = graph.export_topic<Tensor>("holomq://" + first_name, make_options(first_name));
  auto second = graph.export_topic<Tensor>("holomq://" + second_name, make_options(second_name));
  graph.add_flow(source->output, first.input());
  graph.add_flow(source->output, second.input());

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  const auto* planned_output = planned_operator_port(*state, PortDirection::kOutput);
  ASSERT_NE(planned_output, nullptr);
  RequireExportableTensorSourceOperator runtime_source;
  OperatorSpec runtime_source_spec;
  runtime_source.setup(runtime_source_spec);
  const std::array replay_ports{RuntimeChannelPortReplayRecord{
      .port = planned_output->id,
      .owner = planned_output->owner,
      .direction = PortDirection::kOutput,
      .declaration = static_cast<void*>(&runtime_source.output),
      .type_ops = runtime_port_type_ops<Tensor>(),
  }};
  auto session = RuntimeChannelSession::create(*state, replay_ports);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto client_runtime = ChannelRuntime::create();
  ASSERT_TRUE(client_runtime) << runtime_error_message(client_runtime.error());
  auto first_subscriber =
      client_runtime->create_subscriber<Tensor>(plan.endpoint_catalog(), first_name);
  auto second_subscriber =
      client_runtime->create_subscriber<Tensor>(plan.endpoint_catalog(), second_name);
  ASSERT_TRUE(first_subscriber) << runtime_error_message(first_subscriber.error());
  ASSERT_TRUE(second_subscriber) << runtime_error_message(second_subscriber.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  auto foreign = make_host_copy_tracking_tensor(2U);
  ASSERT_TRUE((*session)->prepare_activation(planned_output->owner, 1U, {}));
  auto rejected = runtime_source.output.emit(Sample<Tensor>{foreign.tensor, {}});
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(foreign.state->host_synchronizations.load(std::memory_order_relaxed), 0U);
  auto first_empty = first_subscriber->receive_sample();
  auto second_empty = second_subscriber->receive_sample();
  ASSERT_FALSE(first_empty);
  ASSERT_FALSE(second_empty);
  EXPECT_EQ(first_empty.error().code, ErrorCode::kNotReady);
  EXPECT_EQ(second_empty.error().code, ErrorCode::kNotReady);
  EXPECT_EQ(foreign.state->host_synchronizations.load(std::memory_order_relaxed), 0U);
  ASSERT_TRUE((*session)->retire_activation(planned_output->owner, 1U, true));
}

// A host-only raw-Tensor endpoint has no compiler-frozen CUDA device and therefore must not enter
// the device-affinity inventory or initialize a controller cursor during destructive restart.
// HoloMQ's pageable-host transport still imports a CUDA-VMM split-generation page, but no payload
// device affinity exists for the controller to initialize before a destructive restart.
// REQ: CHAN-104
// REQ: RUNTIME-020
// REQ: RUNTIME-024
TEST(ChannelBindingParityTest, HostTensorRestartSkipsCudaDescriptorImportInitialization) {
  TensorSessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto initialization_count =
      RuntimeChannelSessionTestAccess::controller_descriptor_import_initialization_count(**session);
  ASSERT_TRUE(initialization_count) << runtime_error_message(initialization_count.error());
  EXPECT_EQ(*initialization_count, 0U);

  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  auto owned = make_owned_tensor_sample(71U);
  ASSERT_TRUE(owned) << runtime_error_message(owned.error());
  auto releases = owned->releases;
  ASSERT_TRUE(fixture.source.output.emit(std::move(owned->tensor)));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  const std::array affected{VertexId{1U}, VertexId{2U}};
  ASSERT_TRUE((*session)->commit_restart(VertexId{1U}, affected));
  EXPECT_EQ(releases->load(std::memory_order_relaxed), 1U);
  initialization_count =
      RuntimeChannelSessionTestAccess::controller_descriptor_import_initialization_count(**session);
  ASSERT_TRUE(initialization_count) << runtime_error_message(initialization_count.error());
  EXPECT_EQ(*initialization_count, 0U);
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// Lifecycle restart drains device-affine raw-Tensor descriptors on the controller thread. That
// thread must initialize every frozen import device before crossing the destructive discard
// boundary, select the subscriber's device, and restore the caller afterward.
// REQ: CHAN-103
// REQ: RUNTIME-020
// REQ: RUNTIME-024
TEST(ChannelBindingParityTest, RestartInitializesRawTensorDiscardDeviceAndRestoresCaller) {
  test::skip_without_cuda_device();

  int device_count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
  ASSERT_GT(device_count, 0);
  int ambient_device = -1;
  ASSERT_EQ(cudaGetDevice(&ambient_device), cudaSuccess);
  const int import_device = device_count > 1 ? (ambient_device == 0 ? 1 : 0) : ambient_device;

  TensorSessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::configure_descriptor_import_device(
      **session, PortId{2U}, DeviceId{import_device}));
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  auto owned = make_owned_tensor_sample(73U);
  ASSERT_TRUE(owned) << runtime_error_message(owned.error());
  auto releases = owned->releases;
  ASSERT_TRUE(fixture.source.output.emit(std::move(owned->tensor)));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  const std::array affected{VertexId{1U}, VertexId{2U}};
  ASSERT_TRUE((*session)->commit_restart(VertexId{1U}, affected));
  EXPECT_EQ(releases->load(std::memory_order_relaxed), 1U);
  auto initialization_count =
      RuntimeChannelSessionTestAccess::controller_descriptor_import_initialization_count(**session);
  ASSERT_TRUE(initialization_count) << runtime_error_message(initialization_count.error());
  EXPECT_EQ(*initialization_count, 1U);

  int restored_device = -1;
  ASSERT_EQ(cudaGetDevice(&restored_device), cudaSuccess);
  EXPECT_EQ(restored_device, ambient_device);
}

}  // namespace
}  // namespace holoscan::detail
