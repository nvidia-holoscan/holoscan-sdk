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
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/schema/sdk_schemas.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "../../src/channel/channel_common.hpp"
#include "../../src/channel/channel_payload_frame.hpp"
#include "../../src/channel/channel_payload_limits.hpp"
#include "../../src/channel/channel_tensor_test_access.hpp"
#include "../../src/channel/endpoint_catalog_state.hpp"
#include "../common/cuda_test_utils.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "core/graph_compiler_access.hpp"
#include "core/source_id_internal.hpp"
#include "runtime/channel_runtime_binding_access.hpp"
#include "runtime/dispatcher.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/runtime_channel_binding.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_test_access.hpp"

#include "../support/optional_assertions.hpp"
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
    // REQ: OPERATOR-056
    const auto borrowed = data.selected_sample();
    if (observed == nullptr || !borrowed || Traits::value(borrowed->sample()) != observed ||
        borrowed->metadata().sequence != sample->metadata.sequence) {
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
// REQ: RUNTIME-083
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

// Select the peer disposition at exhausted Runtime capacity without exhausting the endpoint's
// independent operation sequence. Used by RawTensorExportCloseClassifiesPeersAtExhaustion and
// DescriptorTensorExportCloseClassifiesPeersAtExhaustion.
enum class ExhaustedExportPeer : std::uint8_t {
  kNeverAttached,
  kRequiredRetired,
  kOptionalRetired,
  kConnected
};

// Exercise real catalog endpoints and accepted payloads under a disarmed Runtime session so
// counter injection and terminal refresh cannot race a RunSession worker. Representative caller:
// RawTensorExportCloseClassifiesPeersAtExhaustion.
template <typename Payload>
void expect_export_close_at_exhaustion(const ExhaustedExportPeer peer) {
  const std::string endpoint_name = unique_service();
  Graph graph{"exhausted-terminal-" + endpoint_name};
  graph.set_default_clock(graph.bind_shared_clock("clock", std::make_shared<ManualClock>(0)));
  using Source = std::conditional_t<std::is_same_v<Payload, Tensor>,
                                    ContractTensorSourceOperator,
                                    ImageSourceOperator>;
  auto source = graph.op<Source>("source");
  static constexpr std::array axes{TensorAxisBounds{.max_extent = 16U, .max_abs_stride_bytes = 1U}};
  const TensorBoundaryContract contract{
      .representation = {.memory_kind = MemoryKind::kHost,
                         .dtype = DLDataType{kDLUInt, 8U, 1U},
                         .rank = 1U,
                         .strides = TensorStridePolicy::kContiguousOnly},
      .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 16U, .axes = axes},
      .transfer =
          TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded}};
  TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.attach = {.peer = peer == ExhaustedExportPeer::kRequiredRetired ||
                                    peer == ExhaustedExportPeer::kConnected
                                ? PeerRequirement::kRequired
                                : PeerRequirement::kOptional,
                    .max_wait = 2s};
  options.bounds.subscriber = {.queue_depth = 1U, .max_borrowed_samples = 1U};
  auto exported = [&] {
    if constexpr (std::is_same_v<Payload, Tensor>) {
      options.tensor = contract;
      return graph.export_topic<Tensor>("holomq://" + endpoint_name, std::move(options));
    } else {
      DescriptorTensorTopicContract<schema::ImageT> image_contract;
      image_contract.field(TensorFieldSelector<schema::ImageT>{4U, 0U}, contract);
      return graph.export_topic<schema::ImageT>(
          "holomq://" + endpoint_name, std::move(options), image_contract);
    }
  }();
  graph.add_flow(source->output,
                 exported.input(),
                 ConnectionOptions{.queue_policy = QueuePolicy::kBlocking, .queue_depth = 1U});
  const auto plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  const auto* output = planned_operator_port(*state, PortDirection::kOutput);
  ASSERT_NE(output, nullptr);
  const std::array replay{RuntimeChannelPortReplayRecord{
      .port = output->id,
      .owner = output->owner,
      .direction = PortDirection::kOutput,
      .declaration = const_cast<void*>(static_cast<const void*>(&source->output)),
      .type_ops = runtime_port_type_ops<Payload>()}};
  // Replay binds this exact declaration; Graph handles expose it const to ordinary callers.
  auto& source_output = *static_cast<Output<Payload>*>(replay[0].declaration);
  auto session = RuntimeChannelSession::create(*state, replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  const auto exhausted_sequence = std::numeric_limits<std::uint64_t>::max();
  const auto expect_state = [&](const std::size_t resolved_arms) {
    auto sequence = RuntimeChannelSessionTestAccess::output_next_sequence(**session, output->id);
    ASSERT_TRUE(sequence);
    EXPECT_EQ(*sequence, exhausted_sequence);
    auto resolved =
        RuntimeChannelSessionTestAccess::resolved_output_terminal_arms(**session, output->id);
    ASSERT_TRUE(resolved);
    EXPECT_EQ(*resolved, resolved_arms);
    auto progress = RuntimeChannelSessionTestAccess::process_local_progress_state(**session);
    ASSERT_TRUE(progress);
    EXPECT_EQ(progress->published_deliveries, 0U);
    EXPECT_EQ(progress->publications_in_flight, 0U);
  };
  if (peer == ExhaustedExportPeer::kNeverAttached) {
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_output_next_sequence(
        **session, output->id, exhausted_sequence));
    ASSERT_TRUE((*session)->close_owner_outputs(output->owner));
    ASSERT_TRUE((*session)->close_owner_outputs(output->owner));
    expect_state(1U);
    return;
  }
  auto client = ChannelRuntime::create();
  ASSERT_TRUE(client);
  auto subscriber = [&] {
    if constexpr (std::is_same_v<Payload, Tensor>) {
      return client->create_subscriber<Tensor>(plan.endpoint_catalog(), endpoint_name);
    } else {
      return client
          ->create_tensor_schema_subscriber<schema::DescriptorFlatBufferPayloadFor<schema::Image>>(
              plan.endpoint_catalog(), endpoint_name);
    }
  }();
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
  const auto attach_deadline = std::chrono::steady_clock::now() + 2s;
  do {
    ASSERT_TRUE(subscriber->has_samples());
    ASSERT_TRUE((*session)->refresh_connections());
    if (subscriber->has_publishers()) {
      break;
    }
    std::this_thread::sleep_for(1ms);
  } while (std::chrono::steady_clock::now() < attach_deadline);
  ASSERT_TRUE(subscriber->has_publishers());
  ASSERT_TRUE((*session)->prepare_activation(output->owner, 1U, {}));
  auto tensor =
      make_contract_test_tensor(std::array<std::int64_t, 1U>{8}, DLDataType{kDLUInt, 8U, 1U});
  ASSERT_TRUE(tensor);
  if constexpr (std::is_same_v<Payload, Tensor>) {
    ASSERT_TRUE(source_output.emit(std::move(*tensor)));
  } else {
    schema::ImageT image{};
    image.height = 1U;
    image.width = 8U;
    image.encoding = schema::ImageEncoding_CUSTOM;
    image.data = std::make_shared<Tensor>(std::move(*tensor));
    ASSERT_TRUE(source_output.emit(std::move(image)));
  }
  ASSERT_TRUE((*session)->retire_activation(output->owner, 1U, true));
  // Receiving the payload establishes actual recipient acceptance before exhaustion injection.
  const auto receive_deadline = std::chrono::steady_clock::now() + 2s;
  expected<bool, Error> ready = false;
  do {
    ready = subscriber->has_samples();
    ASSERT_TRUE(ready);
    if (*ready) {
      break;
    }
    std::this_thread::sleep_for(1ms);
  } while (std::chrono::steady_clock::now() < receive_deadline);
  ASSERT_TRUE(*ready);
  {
    auto received = [&] {
      if constexpr (std::is_same_v<Payload, Tensor>) {
        return subscriber->receive_sample();
      } else {
        return subscriber->receive();
      }
    }();
    ASSERT_TRUE(received) << runtime_error_message(received.error());
  }
  if (peer != ExhaustedExportPeer::kConnected) {
    ASSERT_TRUE(subscriber->close());
  }
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_output_next_sequence(
      **session, output->id, exhausted_sequence));
  auto closed = (*session)->close_owner_outputs(output->owner);
  ASSERT_FALSE(closed);
  EXPECT_EQ(closed.error().code,
            peer == ExhaustedExportPeer::kRequiredRetired ? ErrorCode::kClosed
                                                          : ErrorCode::kResourceExhausted);
  expect_state(peer == ExhaustedExportPeer::kRequiredRetired ? 1U : 0U);
  if (peer == ExhaustedExportPeer::kRequiredRetired) {
    auto reset = (*session)->reset();
    ASSERT_FALSE(reset);
    EXPECT_EQ(reset.error().code, ErrorCode::kClosed);
    ASSERT_TRUE((*session)->close_owner_outputs(output->owner));
    expect_state(1U);
  } else if (peer == ExhaustedExportPeer::kConnected) {
    // The still-live recipient observes no EOS or phantom data from the rejected close.
    auto ready_after_close = subscriber->has_samples();
    ASSERT_TRUE(ready_after_close);
    EXPECT_FALSE(*ready_after_close);
  }
}

// REQ: RUNTIME-052
// REQ: CHAN-136
// REQ: CHAN-144
TEST(ChannelBindingParityTest, RawTensorExportCloseClassifiesPeersAtExhaustion) {
  // Key setup: source.output -> raw Tensor export -> external subscriber (absent/live/retired).
  // Key assertion: no-obligation close succeeds locally, required abandonment returns kClosed,
  // and connected or previously obligated optional arms remain exhausted; none advances N.
  for (const auto peer : {ExhaustedExportPeer::kNeverAttached,
                          ExhaustedExportPeer::kRequiredRetired,
                          ExhaustedExportPeer::kOptionalRetired,
                          ExhaustedExportPeer::kConnected}) {
    SCOPED_TRACE(static_cast<unsigned>(peer));
    ASSERT_NO_FATAL_FAILURE(expect_export_close_at_exhaustion<Tensor>(peer));
  }
}

// REQ: RUNTIME-052
// REQ: CHAN-136
// REQ: CHAN-144
TEST(ChannelBindingParityTest, DescriptorTensorExportCloseClassifiesPeersAtExhaustion) {
  // Key setup: source.output -> descriptor-schema export -> external Image subscriber.
  // Key assertion: classify no-EOS dispositions before exhausted Runtime capacity, preserving
  // accepted-recipient obligations and the endpoint-owned operation sequence boundary.
  for (const auto peer : {ExhaustedExportPeer::kNeverAttached,
                          ExhaustedExportPeer::kRequiredRetired,
                          ExhaustedExportPeer::kOptionalRetired,
                          ExhaustedExportPeer::kConnected}) {
    SCOPED_TRACE(static_cast<unsigned>(peer));
    ASSERT_NO_FATAL_FAILURE(expect_export_close_at_exhaustion<schema::ImageT>(peer));
  }
}

// REQ: RUNTIME-052
// REQ: CHAN-136
TEST(ChannelBindingParityTest, ExhaustedProcessLocalTensorCloseDoesNotPublish) {
  // Key setup: source.output -> process-local raw Tensor or descriptor-schema input.
  // Key assertion: exhaustion rejects EOS before delivery accounting or receiver terminal state.
  const auto check = [](auto& fixture) {
    auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
    ASSERT_TRUE(session);
    ASSERT_TRUE((*session)->refresh_connections());
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_output_next_sequence(
        **session, PortId{1U}, std::numeric_limits<std::uint64_t>::max()));
    auto closed = (*session)->close_owner_outputs(VertexId{1U});
    ASSERT_FALSE(closed);
    EXPECT_EQ(closed.error().code, ErrorCode::kResourceExhausted);
    auto sequence = RuntimeChannelSessionTestAccess::output_next_sequence(**session, PortId{1U});
    ASSERT_TRUE(sequence);
    EXPECT_EQ(*sequence, std::numeric_limits<std::uint64_t>::max());
    auto progress = RuntimeChannelSessionTestAccess::process_local_progress_state(**session);
    ASSERT_TRUE(progress);
    EXPECT_EQ(progress->published_deliveries, 0U);
    EXPECT_EQ(progress->publications_in_flight, 0U);
    auto input = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    ASSERT_TRUE(input);
    EXPECT_FALSE(input->end_of_stream_received);
    auto resolved =
        RuntimeChannelSessionTestAccess::resolved_output_terminal_arms(**session, PortId{1U});
    ASSERT_TRUE(resolved);
    EXPECT_EQ(*resolved, 0U);
  };
  TensorSessionFixture tensor;
  ASSERT_NO_FATAL_FAILURE(check(tensor));
  ImageSessionFixture image;
  ASSERT_NO_FATAL_FAILURE(check(image));
}

// Runtime must not retain or surface a terminal witness until the descriptor-Tensor endpoint has
// accepted the complete canonical EOS envelope. A rejected raw terminal leaves the same endpoint
// open so the later valid terminal can close it on both ordinary and protected lanes.
// REQ: CHAN-136
TEST(ChannelBindingParityTest, MalformedDescriptorTensorTerminalDoesNotExposeRuntimeEndOfStream) {
  using ImageEnvelope = schema::detail::DescriptorTensorRuntimeEnvelope<
      schema::Image,
      schema::flatbuffer_descriptor_metadata_capacity_v<schema::Image>,
      1U>;
  constexpr auto bucket_size = channel_internal::inline_payload_capacity(sizeof(ImageEnvelope));
  static_assert(bucket_size.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by static_assert above.
  constexpr std::size_t kPayloadCapacity = *bucket_size;

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
    const auto transport_service = RuntimeChannelSessionTestAccess::process_local_transport_name(
        **session, service.service_id);
    ASSERT_TRUE(transport_service) << runtime_error_message(transport_service.error());
    const std::string terminal_service =
        protected_terminal ? *transport_service + ".hsdk-terminal-v1" : *transport_service;
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
    auto publisher =
        context->publisher_builder(terminal_service)
            .service_options(channel_internal::payload_frame_service_options(
                service_options, channel_internal::PayloadFrameProfile::kInline, kPayloadCapacity))
            .publisher_options(publisher_options)
            .payload_type<holomq::ByteSlice>()
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
      // A malformed terminal carries the one payload byte it declares, so only the envelope check
      // can reject it.
      const std::uint32_t payload_size = malformed ? 1U : 0U;
      auto message =
          publisher->loan(channel_internal::kInlinePayloadFrameDataOffset + payload_size);
      EXPECT_TRUE(message);
      if (!message) {
        return false;
      }
      channel_internal::write_payload_frame_header(
          channel_internal::PayloadFrameProfile::kInline,
          message->payload(),
          channel_internal::PayloadFrameFields{
              .operation = channel_internal::InlinePayloadOperation::kEndOfStream,
              .metadata =
                  SampleMetadata{
                      .sequence = 1U,
                      .byte_size = 0U,
                      .payload_kind = PayloadKind::kDescriptorBacked,
                  },
              .payload_size = payload_size});
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

// Exercise the same real terminal transport contract for raw Tensor and descriptor-schema roots.
// No pump races the missing-data boundary: the existing test seam performs each bounded receive.
// Representative callers: RawTensorSequenceAndTerminalSurviveLifecycleCutovers and
// DescriptorSchemaSequenceAndTerminalSurviveLifecycleCutovers.
template <typename Fixture, typename Publish>
void expect_tensor_sequence_across_cutover(Fixture& fixture, Publish publish,
                                           const bool replacement_data) {
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(publish());
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
  // Each successful empty transition preserves the same consumed source authority.
  ASSERT_TRUE((*session)->reset());
  ASSERT_TRUE((*session)->reset(VertexId{1U}));
  const std::array affected{VertexId{1U}, VertexId{2U}};
  ASSERT_TRUE((*session)->commit_restart(VertexId{1U}, affected));
  ASSERT_TRUE((*session)->refresh_connections());
  if (replacement_data) {
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 3U, {}));
    ASSERT_TRUE(publish());
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 3U, true));
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
    auto data = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    ASSERT_TRUE(data);
    EXPECT_EQ(data->last_producer_sequence, 2U);
    EXPECT_EQ(data->producer_epoch, 4U);
  }
  auto closed = (*session)->close_owner_outputs(VertexId{1U});
  ASSERT_TRUE(closed) << runtime_error_message(closed.error());
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
  auto terminal = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(terminal);
  EXPECT_TRUE(terminal->end_of_stream_received);
  EXPECT_EQ(terminal->end_of_stream_transport_sequence, replacement_data ? 3U : 2U);
  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  auto counter = RuntimeChannelSessionTestAccess::output_next_sequence(**session, PortId{1U});
  ASSERT_TRUE(counter);
  EXPECT_EQ(*counter, replacement_data ? 4U : 3U);
}

// REQ: RUNTIME-052
// REQ: CHAN-136
TEST(ChannelBindingParityTest, RawTensorSequenceAndTerminalSurviveLifecycleCutovers) {
  // Key setup: source.output -> process-local raw Tensor -> sink.input; publish data,
  // then perform full reset, source reset and committed restart, optionally publishing again.
  // Key assertion: data sequence remains monotonic and EOS is the immediate successor,
  // including an empty replacement epoch; repeated close consumes no second coordinate.
  for (const bool replacement_data : {false, true}) {
    SCOPED_TRACE(replacement_data);
    TensorSessionFixture fixture;
    expect_tensor_sequence_across_cutover(
        fixture,
        [&]() -> expected<void, Error> {
          auto owned = make_owned_tensor_sample(71U);
          if (!owned) {
            return make_unexpected(std::move(owned).error());
          }
          return fixture.source.output.emit(std::move(owned->tensor));
        },
        replacement_data);
  }
}

// REQ: RUNTIME-052
// REQ: CHAN-136
TEST(ChannelBindingParityTest, DescriptorSchemaSequenceAndTerminalSurviveLifecycleCutovers) {
  // Key setup: source.output -> process-local descriptor-schema Image -> sink.input; publish data,
  // then perform full reset, source reset and committed restart, optionally publishing again.
  // Key assertion: data sequence remains monotonic and EOS is the immediate successor,
  // including an empty replacement epoch; repeated close consumes no second coordinate.
  for (const bool replacement_data : {false, true}) {
    SCOPED_TRACE(replacement_data);
    ImageSessionFixture fixture;
    expect_tensor_sequence_across_cutover(
        fixture,
        [&]() -> expected<void, Error> {
          auto owned = make_owned_tensor_sample(72U);
          if (!owned) {
            return make_unexpected(std::move(owned).error());
          }
          schema::ImageT image{};
          image.height = 1;
          image.width = 1;
          image.encoding = schema::ImageEncoding_CUSTOM;
          image.header = std::make_shared<schema::HeaderT>();
          image.header->frame_id = "lifecycle sequence";
          image.data = std::make_shared<Tensor>(std::move(owned->tensor));
          return fixture.source.output.emit(std::move(image));
        },
        replacement_data);
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
  auto replacement_candidate =
      (*session)->staged_reuse_replacement_candidate(PortId{2U}, *first_record);
  ASSERT_TRUE(replacement_candidate) << runtime_error_message(replacement_candidate.error());
  const auto& optional_replacement_candidate = *replacement_candidate;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_replacement_candidate);
  const auto forced_failure =
      (*session)->replace_staged_reuse(PortId{2U}, *first_record, *optional_replacement_candidate);
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
  auto final_data =
      (*session)->replace_staged_reuse(PortId{2U}, *first_record, *optional_replacement_candidate);
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

// REQ: RUNTIME-017
// REQ: CHAN-131
// A staged-reuse probe is only advisory. If the serialized transport head changes before commit,
// the stale identity must not consume the successor or release the incumbent ReuseLast claim.
TEST(ChannelBindingParityTest, StagedReuseReplacementRejectsStaleTransportHeadIdentity) {
  SessionFixture fixture;
  fixture.use_one_slot_blocking_transport();
  fixture.enable_reuse_last();
  // HoloMQ's own blocking pressure is under test, so turn off the credit check that prevents it.
  fixture.disable_publication_credits();
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
  ASSERT_TRUE(
      fixture.source.output.emit(11, EmitOptions{.capture_time = TimePoint{10, ClockId{1U}}}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
  auto retained_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(retained_event) << runtime_error_message(retained_event.error());
  auto retained =
      (*session)->temporal_sample(retained_event->port, retained_event->source_sequence);
  ASSERT_TRUE(retained) << runtime_error_message(retained.error());
  const std::array retained_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&*retained, 1U},
      .destructive = true}};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, retained_selection));
  auto retained_sample = fixture.sink.input.receive();
  ASSERT_TRUE(retained_sample) << runtime_error_message(retained_sample.error());
  EXPECT_EQ(retained_sample->data, 11);
  ASSERT_TRUE((*binding)->retire(retained_event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  const auto retained_by_reuse =
      (*session)->retire_samples(PortId{2U}, retained_selection.front().samples);
  ASSERT_FALSE(retained_by_reuse);
  ASSERT_EQ(retained_by_reuse.error().code, ErrorCode::kBackpressured);

  // With the sole typed slot retained, the failed receive stages sequence 2 without consuming it.
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 2U, {}));
  ASSERT_TRUE(
      fixture.source.output.emit(22, EmitOptions{.capture_time = TimePoint{20, ClockId{1U}}}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 2U, true));
  const auto blocked_head = RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U});
  ASSERT_FALSE(blocked_head);
  ASSERT_EQ(blocked_head.error().code, ErrorCode::kBackpressured);

  auto staged_sequence =
      RuntimeChannelSessionTestAccess::staged_data_sequence(**session, PortId{2U});
  ASSERT_TRUE(staged_sequence) << runtime_error_message(staged_sequence.error());
  ASSERT_EQ(*staged_sequence, 2U);
  auto stale_candidate = (*session)->staged_reuse_replacement_candidate(PortId{2U}, *retained);
  ASSERT_TRUE(stale_candidate) << runtime_error_message(stale_candidate.error());
  const auto& optional_stale_candidate = *stale_candidate;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_stale_candidate);
  ASSERT_EQ(optional_stale_candidate->source_sequence, 2U);

  // Queue sequence 3 behind the protected staged head, then advance the serialized head without
  // giving the stale probe token an opportunity to consume either record.
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 3U, {}));
  ASSERT_TRUE(
      fixture.source.output.emit(33, EmitOptions{.capture_time = TimePoint{30, ClockId{1U}}}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 3U, true));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::discard_staged_data_head(**session, PortId{2U}));
  staged_sequence = RuntimeChannelSessionTestAccess::staged_data_sequence(**session, PortId{2U});
  ASSERT_TRUE(staged_sequence) << runtime_error_message(staged_sequence.error());
  ASSERT_EQ(*staged_sequence, 3U);
  const auto stale_replace =
      (*session)->replace_staged_reuse(PortId{2U}, *retained, *optional_stale_candidate);
  ASSERT_FALSE(stale_replace);
  EXPECT_EQ(stale_replace.error().code, ErrorCode::kNotReady);
  const auto preserved = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(preserved) << runtime_error_message(preserved.error());
  EXPECT_EQ(preserved->occupied_samples, 1U);
  EXPECT_TRUE(preserved->reuse_retained);
  EXPECT_TRUE(preserved->temporal_retained);
  EXPECT_TRUE(preserved->reuse_slot_matches_single_occupied);

  auto fresh_candidate = (*session)->staged_reuse_replacement_candidate(PortId{2U}, *retained);
  ASSERT_TRUE(fresh_candidate) << runtime_error_message(fresh_candidate.error());
  const auto& optional_fresh_candidate = *fresh_candidate;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_fresh_candidate);
  ASSERT_EQ(optional_fresh_candidate->source_sequence, 3U);
  auto replaced =
      (*session)->replace_staged_reuse(PortId{2U}, *retained, *optional_fresh_candidate);
  ASSERT_TRUE(replaced) << runtime_error_message(replaced.error());
  EXPECT_EQ(replaced->source_sequence, 3U);
  const auto committed = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(committed) << runtime_error_message(committed.error());
  EXPECT_EQ(committed->occupied_samples, 1U);
  EXPECT_FALSE(committed->reuse_retained);
  EXPECT_FALSE(committed->reuse_slot_matches_single_occupied);

  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// REQ: CHAN-146
// Rollback must restore both ledgers, not just the selector metadata, while the fallback lives.
// The newer arrival must wait at the endpoint head, as on a cross-process edge; a process-local
// subscriber would stage it (CHAN-161).
TEST(ChannelBindingParityTest, TransportedCacheReplacementPreservesFallbackAcrossReceiveFailure) {
  TensorSessionFixture fixture;
  fixture.use_one_slot_drop_oldest_queue();
  fixture.enable_reuse_last();
  auto& selection = fixture.base.plan.input_selections.front();
  selection.selector = InputSelectorKind::kLatestWithin;
  selection.cache_capacity = 1U;
  selection.max_age_ns = 100;
  auto selector = RuntimeInputSelectionState::create(selection);
  ASSERT_TRUE(selector);
  const auto incomplete = std::make_shared<RuntimeChannelIncompletePollTestControl>(1U);
  fixture.base.plan.executable_factories.test_channel_incomplete_poll_control = incomplete;
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session);
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::stop_tensor_ingress_staging(**session, PortId{2U}));
  auto binding = (*session)->create_readiness_binding(
      VertexId{2U},
      RuntimePortBindingRequest{.plan_generation = fixture.base.plan.plan_generation,
                                .run_generation = 1U,
                                .kind = RuntimeEventKind::kInputArrival,
                                .source = make_source_id(1U),
                                .source_clock = ClockId{1U},
                                .reference_clock = ClockId{1U},
                                .capacity = 2U});
  ASSERT_TRUE(binding);
  ASSERT_TRUE((*session)->refresh_connections());
  const auto publish = [&](std::uint64_t sequence) {
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, sequence, {}));
    auto sample = make_owned_tensor_sample(sequence);
    ASSERT_TRUE(sample);
    ASSERT_TRUE(fixture.source.output.emit(
        std::move(sample->tensor),
        EmitOptions{.capture_time =
                        TimePoint{static_cast<std::int64_t>(sequence * 10U), ClockId{1U}}}));
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, sequence, true));
  };
  ASSERT_NO_FATAL_FAILURE(publish(1U));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
  auto first_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(first_event);
  auto first = (*session)->temporal_sample(PortId{2U}, first_event->source_sequence);
  ASSERT_TRUE(first);
  ASSERT_TRUE(selector->push(*first));
  auto selected = selector->begin(first->capture_time, 1U, 1U);
  ASSERT_TRUE(selected);
  const std::array selected_payload{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&*first, 1U},
      .destructive = true}};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selected_payload));
  ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 1U, nullptr));
  ASSERT_TRUE(fixture.sink.input.receive());
  ASSERT_TRUE((*binding)->retire(first_event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  ASSERT_TRUE(selected->finish(true));

  ASSERT_NO_FATAL_FAILURE(publish(2U));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
  auto second_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(second_event);
  auto second = (*session)->temporal_sample(PortId{2U}, second_event->source_sequence);
  ASSERT_TRUE(second);
  ASSERT_TRUE(selector->push(*second));
  ASSERT_TRUE((*binding)->retire(second_event->source_sequence));
  ASSERT_NO_FATAL_FAILURE(publish(3U));
  auto staged = RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U});
  ASSERT_FALSE(staged);
  ASSERT_EQ(staged.error().code, ErrorCode::kBackpressured);
  auto candidate = (*session)->staged_reuse_replacement_candidate(PortId{2U}, *second);
  ASSERT_TRUE(candidate);
  const auto& optional_candidate = *candidate;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_candidate);
  ASSERT_TRUE(
      RuntimeChannelSessionTestAccess::fail_next_reuse_replacement_receive(**session, PortId{2U}));
  {
    auto transaction = selector->reserve_cache_replacement(*second);
    ASSERT_TRUE(transaction);
    auto replaced = (*session)->replace_staged_reuse(PortId{2U}, *second, *optional_candidate);
    ASSERT_FALSE(replaced);
    EXPECT_EQ(replaced.error().code, ErrorCode::kFailure);
  }
  std::array<TemporalSampleRecord, 2U> claims{};
  auto restored = selector->copy_payload_claims(claims);
  ASSERT_TRUE(restored);
  ASSERT_EQ(*restored, 2U);
  EXPECT_NE(std::find(claims.begin(), claims.end(), *first), claims.end());
  EXPECT_NE(std::find(claims.begin(), claims.end(), *second), claims.end());
  EXPECT_TRUE((*session)->temporal_sample(PortId{2U}, first_event->source_sequence));
  EXPECT_TRUE((*session)->temporal_sample(PortId{2U}, second_event->source_sequence));
  auto state = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(state);
  EXPECT_EQ(state->occupied_samples, 2U);
  auto reservations =
      RuntimeChannelSessionTestAccess::typed_ingress_reservations(**session, PortId{2U});
  ASSERT_TRUE(reservations);
  EXPECT_EQ(*reservations, 0U);
  auto staged_sequence =
      RuntimeChannelSessionTestAccess::staged_data_sequence(**session, PortId{2U});
  ASSERT_TRUE(staged_sequence);
  EXPECT_EQ(*staged_sequence, 3U);

  auto transaction = selector->reserve_cache_replacement(*second);
  ASSERT_TRUE(transaction);
  ASSERT_TRUE((*session)->replace_staged_reuse(PortId{2U}, *second, *optional_candidate));
  transaction->commit();
  ASSERT_TRUE((*binding)->poll_blocking(TimePoint{40, ClockId{1U}}));
  EXPECT_EQ(incomplete->forced(), 1U);
  EXPECT_TRUE((*binding)->last_poll_incomplete());
  RuntimeEvent event{};
  auto drained = (*binding)->drain(std::span<RuntimeEvent>{&event, 1U});
  ASSERT_TRUE(drained);
  EXPECT_EQ(*drained, 0U);
  auto third_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(third_event);
  auto third = (*session)->temporal_sample(PortId{2U}, third_event->source_sequence);
  ASSERT_TRUE(third);
  EXPECT_EQ(third->source_sequence, 3U);
  EXPECT_EQ(selector->reuse_retained_record(), *first);
  EXPECT_TRUE((*session)->temporal_sample(PortId{2U}, first_event->source_sequence));
  EXPECT_FALSE((*session)->temporal_sample(PortId{2U}, second_event->source_sequence));
  ASSERT_TRUE(selector->push(*third));
  ASSERT_TRUE((*binding)->retire(third_event->source_sequence));
  drained = (*binding)->drain(std::span<RuntimeEvent>{&event, 1U});
  ASSERT_TRUE(drained);
  EXPECT_EQ(*drained, 0U);
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// REQ: RUNTIME-017
// REQ: CHAN-131
// A transported ReuseLast replacement has no payload claim until readiness publishes. A newer
// drop-oldest arrival must not overwrite that exact successor in the pre-witness window. Arrivals
// wait at the endpoint head, as on a cross-process edge; a process-local subscriber would stage
// them (CHAN-161).
TEST(ChannelBindingParityTest, TransportedPromotedRefillIsNotReplacedBeforeItsReadinessPublishes) {
  TensorSessionFixture fixture;
  fixture.use_one_slot_drop_oldest_queue();
  fixture.enable_reuse_last();
  const auto counters = std::make_shared<RuntimePhaseCounters>();
  fixture.base.plan.executable_factories.phase_counters = counters;
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::stop_tensor_ingress_staging(**session, PortId{2U}));
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
  auto first = make_owned_tensor_sample(0x11U);
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  ASSERT_TRUE(fixture.source.output.emit(std::move(first->tensor)));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
  auto retained_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(retained_event) << runtime_error_message(retained_event.error());
  auto retained =
      (*session)->temporal_sample(retained_event->port, retained_event->source_sequence);
  ASSERT_TRUE(retained) << runtime_error_message(retained.error());
  const std::array retained_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&*retained, 1U},
      .destructive = true}};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, retained_selection));
  ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 1U, nullptr));
  auto retained_sample = fixture.sink.input.receive();
  ASSERT_TRUE(retained_sample) << runtime_error_message(retained_sample.error());
  ASSERT_NE(retained_sample->data.data_as<std::uint64_t>(), nullptr);
  EXPECT_EQ(*retained_sample->data.data_as<std::uint64_t>(), 0x11U);
  ASSERT_TRUE((*binding)->retire(retained_event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  const auto retained_by_reuse =
      (*session)->retire_samples(PortId{2U}, retained_selection.front().samples);
  ASSERT_FALSE(retained_by_reuse);
  ASSERT_EQ(retained_by_reuse.error().code, ErrorCode::kBackpressured);

  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 2U, {}));
  auto refill = make_owned_tensor_sample(0x22U);
  ASSERT_TRUE(refill) << runtime_error_message(refill.error());
  ASSERT_TRUE(fixture.source.output.emit(std::move(refill->tensor)));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 2U, true));
  const auto staged_refill = RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U});
  ASSERT_FALSE(staged_refill);
  ASSERT_EQ(staged_refill.error().code, ErrorCode::kBackpressured);
  auto staged_sequence =
      RuntimeChannelSessionTestAccess::staged_data_sequence(**session, PortId{2U});
  ASSERT_TRUE(staged_sequence) << runtime_error_message(staged_sequence.error());
  ASSERT_EQ(*staged_sequence, 2U);

  auto replacement_candidate =
      (*session)->staged_reuse_replacement_candidate(PortId{2U}, *retained);
  ASSERT_TRUE(replacement_candidate) << runtime_error_message(replacement_candidate.error());
  const auto& optional_replacement_candidate = *replacement_candidate;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_replacement_candidate);
  auto replaced =
      (*session)->replace_staged_reuse(PortId{2U}, *retained, *optional_replacement_candidate);
  ASSERT_TRUE(replaced) << runtime_error_message(replaced.error());
  ASSERT_EQ(replaced->source_sequence, 2U);

  // Do not poll readiness yet. This creates the exact pre-witness replacement opportunity.
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 3U, {}));
  auto intruder = make_owned_tensor_sample(0x33U);
  ASSERT_TRUE(intruder) << runtime_error_message(intruder.error());
  ASSERT_TRUE(fixture.source.output.emit(std::move(intruder->tensor)));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 3U, true));
  const auto staged_intruder = RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U});
  ASSERT_FALSE(staged_intruder);
  ASSERT_EQ(staged_intruder.error().code, ErrorCode::kBackpressured);
  staged_sequence = RuntimeChannelSessionTestAccess::staged_data_sequence(**session, PortId{2U});
  ASSERT_TRUE(staged_sequence) << runtime_error_message(staged_sequence.error());
  ASSERT_EQ(*staged_sequence, 3U);
  const auto guarded = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(guarded) << runtime_error_message(guarded.error());
  EXPECT_EQ(guarded->occupied_samples, 1U);
  EXPECT_EQ(guarded->pending_direct_ingress_samples, 0U);
  EXPECT_FALSE(guarded->reuse_retained);
  EXPECT_FALSE(guarded->temporal_retained);
  auto outcome = (*session)->last_refill_outcome(PortId{2U});
  ASSERT_TRUE(outcome) << runtime_error_message(outcome.error());
  ASSERT_EQ(*outcome, 2U);

  auto refilled_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(refilled_event) << runtime_error_message(refilled_event.error());
  auto refilled =
      (*session)->temporal_sample(refilled_event->port, refilled_event->source_sequence);
  ASSERT_TRUE(refilled) << runtime_error_message(refilled.error());
  EXPECT_EQ(*refilled, *replaced);
  outcome = (*session)->last_refill_outcome(PortId{2U});
  ASSERT_TRUE(outcome) << runtime_error_message(outcome.error());
  EXPECT_EQ(*outcome, 1U);
  const auto phases = counters->snapshot();
  EXPECT_GT(phases[RuntimePhaseCounters::index(
                RuntimePhaseCounter::kChannelReleasedClaimRefillReplacementDeclined)],
            0U);
  EXPECT_GT(phases[RuntimePhaseCounters::index(
                RuntimePhaseCounter::kChannelReleasedClaimRefillPublished)],
            0U);

  ASSERT_TRUE((*binding)->retire(refilled_event->source_sequence));
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
  auto replacement_candidate =
      (*session)->staged_reuse_replacement_candidate(PortId{2U}, *first_record);
  ASSERT_TRUE(replacement_candidate) << runtime_error_message(replacement_candidate.error());
  const auto& optional_replacement_candidate = *replacement_candidate;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_replacement_candidate);
  auto final_data =
      (*session)->replace_staged_reuse(PortId{2U}, *first_record, *optional_replacement_candidate);
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
  auto replacement_candidate =
      (*session)->staged_reuse_replacement_candidate(PortId{2U}, *first_record);
  ASSERT_TRUE(replacement_candidate) << runtime_error_message(replacement_candidate.error());
  const auto& optional_replacement_candidate = *replacement_candidate;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_replacement_candidate);
  auto final_data =
      (*session)->replace_staged_reuse(PortId{2U}, *first_record, *optional_replacement_candidate);
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

}  // namespace
}  // namespace holoscan::detail
