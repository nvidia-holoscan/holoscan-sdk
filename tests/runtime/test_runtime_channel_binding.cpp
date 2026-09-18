/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <future>
#include <iterator>
#include <limits>
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
#include <holoscan/profiler/nvtx3.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "common/allocation_counter.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "core/operator_spec_compiler_access.hpp"
#include "core/source_id_internal.hpp"
#include "runtime/endpoint_clock_health_runtime.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/runtime_channel_binding.hpp"
#include "runtime/runtime_channel_plan_binding.hpp"
#include "runtime/runtime_channel_storage.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_rate_state.hpp"
#include "time/clock_registry_internal.hpp"

#include "runtime_channel_binding_test_support.hpp"

namespace holoscan::test::nvtx_emit_marker_capture {

bool reset(std::string_view label);
std::size_t emitted() noexcept;

}  // namespace holoscan::test::nvtx_emit_marker_capture

namespace holoscan::detail {
namespace {

using namespace std::chrono_literals;
using namespace runtime_channel_binding_test;

class HeldAcceptedInlinePublisher final : public channel_internal::InlinePayloadPublisher {
 public:
  [[nodiscard]] channel_internal::InlinePublicationResult emit_phase(
      std::span<const std::byte>, const SampleMetadata&) override {
    entered.release();
    resume.acquire();
    return channel_internal::InlinePublicationResult{
        .phase = channel_internal::InlinePublicationPhase::kAccepted,
        .recipient_count = 1U,
        .error = Error{ErrorCode::kSuccess},
    };
  }

  [[nodiscard]] expected<void, Error> update_connections() noexcept override { return {}; }
  [[nodiscard]] bool has_subscribers() const noexcept override { return true; }

  std::binary_semaphore entered{0};
  std::binary_semaphore resume{0};
};

struct MixedDomainRelayFixture {
  SessionFixture base;
  SourceOperator relay_output;
  OperatorSpec relay_output_spec;
  std::array<RuntimeChannelPortReplayRecord, 4U> replay;

  MixedDomainRelayFixture() {
    base.use_cross_process_edge();
    relay_output.setup(relay_output_spec);
    const auto relay_output_port = OperatorSpecCompilerAccess::port(relay_output_spec, 0U);
    const auto local_sink_port = OperatorSpecCompilerAccess::port(base.second_sink_spec, 0U);
    const std::string local_service =
        fmt::format("process-local-channel.{}.source.3.policy.0", base.plan.plan_generation);

    base.plan.ports.push_back(PortPlanRecord{
        .id = PortId{3U},
        .owner = VertexId{2U},
        .direction = PortDirection::kOutput,
        .local_name = "output",
        .payload_identity = relay_output_port.payload_identity,
        .max_emits_per_compute = 8U,
        .effective_profile = Profile::kStandard,
    });
    base.plan.ports.push_back(PortPlanRecord{
        .id = PortId{4U},
        .owner = VertexId{3U},
        .direction = PortDirection::kInput,
        .local_name = "input",
        .payload_identity = local_sink_port.payload_identity,
        .queue_depth = 1U,
        .effective_profile = Profile::kStandard,
    });
    base.plan.process_local_channel_services = {
        ProcessLocalChannelServicePlanRecord{
            .source = PortId{3U},
            .service_id = local_service,
            .payload_identity = relay_output_port.payload_identity,
            .payload_size_bytes = sizeof(std::int32_t),
            .selected_inline_bucket_bytes =
                *channel_internal::inline_payload_bucket_size(sizeof(std::int32_t)),
            .queue_policy = QueuePolicy::kBlocking,
            .max_subscribers = 1U,
            .max_subscriber_queue_depth = 1U,
            .max_subscriber_borrowed_samples = 1U,
            .publisher_history_depth = 0U,
            .event_wake = false,
        },
    };
    base.plan.process_local_channel_publishers = {
        ProcessLocalChannelPublisherPlanRecord{
            .source = PortId{3U},
            .service_id = local_service,
            .max_loaned_samples = 8U,
        },
    };
    base.plan.process_local_channel_subscribers = {
        ProcessLocalChannelSubscriberPlanRecord{
            .edge = EdgeId{2U},
            .source = PortId{3U},
            .target = PortId{4U},
            .service_id = local_service,
            .requested_queue_depth = 1U,
            .max_borrowed_samples = 1U,
            .cache_capacity = 1U,
            .readiness_capacity = 1U,
            .event_wake = false,
        },
    };
    base.plan.payload_edges.push_back(PayloadEdgePlanRecord{
        .edge = EdgeId{2U},
        .source = PortId{3U},
        .target = PortId{4U},
        .payload_identity = relay_output_port.payload_identity,
        .queue_policy = QueuePolicy::kBlocking,
        .queue_depth = 1U,
        .max_inflight_samples = 1U,
        .producer_max_emits_per_compute = 8U,
        .fanout = 1U,
        .cache_capacity = 1U,
        .locality = ChannelLocalityKind::kProcessLocal,
        .disposition = PlanDisposition::kAccepted,
    });
    base.plan.channel_locality.push_back(ChannelLocalityRecord{
        .edge = EdgeId{2U},
        .requested = ChannelLocalityKind::kProcessLocal,
        .accepted = ChannelLocalityKind::kProcessLocal,
        .requires_runtime_attach = false,
    });
    base.plan.vertex_process_placements.push_back(VertexProcessPlacementPlanRecord{
        .vertex = VertexId{3U},
        .process_group = ProcessGroupId{2U},
    });
    base.plan.execution_partitions = {
        ExecutionPartitionRecord{
            .id = ExecutionPartitionId{1U},
            .members = {VertexId{1U}},
            .execution_lane = 0U,
        },
        ExecutionPartitionRecord{
            .id = ExecutionPartitionId{2U},
            .members = {VertexId{2U}, VertexId{3U}},
            .execution_lane = 0U,
        },
    };
    replay = {
        base.replay[0],
        base.replay[1],
        RuntimeChannelPortReplayRecord{
            .port = PortId{3U},
            .owner = VertexId{2U},
            .direction = PortDirection::kOutput,
            .declaration = const_cast<void*>(relay_output_port.declaration),
            .type_ops = relay_output_port.runtime_type_ops,
        },
        RuntimeChannelPortReplayRecord{
            .port = PortId{4U},
            .owner = VertexId{3U},
            .direction = PortDirection::kInput,
            .declaration = base.replay[2].declaration,
            .type_ops = local_sink_port.runtime_type_ops,
        },
    };
  }
};

class RateObservedQueueSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U).nominal_rate(Rate::hz(30U));
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::int32_t> output;
};

class RateObservedQueueSink final : public Operator<> {
 public:
  explicit RateObservedQueueSink(bool on_each = false) noexcept : on_each_(on_each) {}

  void setup(OperatorSpec& spec) override { spec.input(input, "input"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    if (on_each_) {
      result.trigger(OnEach{.port = input});
      return result;
    }
    result.trigger(OnClock{.period = 1s}).input(input, Exact{}).on_missing(input, Skip{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::int32_t> input;

 private:
  bool on_each_{};
};

// Hold one transport receive open so the lock-boundary regression can observe whether the
// independent typed-retention cache remains available to another controller thread.
class BlockingInlineSubscriber final : public channel_internal::InlinePayloadSubscriber {
 public:
  [[nodiscard]] expected<bool, Error> has_samples() const override { return true; }

  // This double has no transport, so it has no event lane and no runtime to build a wait context
  // from. Reporting that honestly makes wait_for_readiness fail with kNotSupported rather than
  // pretend to wait.
  [[nodiscard]] bool event_wake_enabled() const noexcept override { return false; }

  [[nodiscard]] void* wait_context_runtime() noexcept override { return nullptr; }

  [[nodiscard]] bool has_publishers() const noexcept override { return true; }

  [[nodiscard]] expected<channel_internal::InlinePayloadReceiveResult, Error> receive(
      const std::span<std::byte> payload) override {
    receive_entered.store(true, std::memory_order_release);
    if (block_receive.load(std::memory_order_acquire)) {
      while (!release_receive.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
    }
    const ErrorCode error = receive_error.load(std::memory_order_acquire);
    if (error != ErrorCode::kSuccess) {
      return make_unexpected(Error{error});
    }
    const std::int32_t received_value = value.load(std::memory_order_acquire);
    if (payload.size() != sizeof(received_value)) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    std::memcpy(payload.data(), &received_value, sizeof(received_value));
    return channel_internal::InlinePayloadReceiveResult{
        .metadata =
            SampleMetadata{
                .sequence = sequence.load(std::memory_order_acquire),
                .byte_size = sizeof(received_value),
                .payload_kind = PayloadKind::kInlineTyped,
            },
        .payload_size = sizeof(received_value),
    };
  }

  [[nodiscard]] expected<std::uint64_t, Error> attach_wait_set(void*) override {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }

  [[nodiscard]] expected<void, Error> detach_wait_set() noexcept override { return {}; }

  std::atomic<bool> receive_entered{};
  std::atomic<bool> release_receive{};
  std::atomic<bool> block_receive{true};
  std::atomic<ErrorCode> receive_error{ErrorCode::kSuccess};
  std::atomic<std::int32_t> value{37};
  std::atomic<std::uint64_t> sequence{1U};
};

void observe_input_lock(void* opaque) noexcept {
  static_cast<std::atomic<bool>*>(opaque)->store(true, std::memory_order_release);
}

TEST(RuntimeChannelBinding, DirectContinuationReceiptRequiresExactNonConsumingEvidence) {
  using Receipt = RuntimeChannelDirectContinuationReceipt;
  using Target = RuntimeChannelDirectContinuationTarget;
  using Mutation = void (*)(Receipt&, Target&) noexcept;

  constexpr std::string_view kService{"process-local-channel.11.source.3.policy.0"};
  const Target expected{
      .plan_generation = 11U,
      .run_generation = 12U,
      .producer = VertexId{3U},
      .destination = VertexId{4U},
      .source_output = PortId{5U},
      .destination_input = PortId{6U},
      .service = kService,
      .readiness_source = make_source_id(7U),
      .reference_clock = ClockId{8U},
      .source_generation = 9U,
      .readiness_sequence = 13U,
      .destination_revision = 17U,
      .observed_destination_revision = 17U,
      .source_activation_generation = 18U,
      .execution_lane = 10U,
      .readiness_capacity = 1U,
  };
  const Receipt valid{
      .plan_generation = expected.plan_generation,
      .run_generation = expected.run_generation,
      .producer = expected.producer,
      .destination = expected.destination,
      .source_output = expected.source_output,
      .destination_input = expected.destination_input,
      .service = expected.service,
      .readiness_source = expected.readiness_source,
      .reference_clock = expected.reference_clock,
      .source_generation = expected.source_generation,
      .readiness_sequence = 13U,
      .producer_epoch = 14U,
      .publication_sequence = 15U,
      .prior_destination_revision = 16U,
      .destination_revision = 17U,
      .source_activation_generation = 18U,
      .execution_lane = expected.execution_lane,
      .successor_publications = 1U,
      .publication_status = ErrorCode::kSuccess,
  };

  // Key setup: every negative row changes one cold identity or live publication fact while
  // retaining an otherwise valid allocation-free receipt.
  struct Case {
    std::string_view name;
    Mutation mutate;
  };
  const std::array cases{
      Case{"zero plan generation",
           +[](Receipt& receipt, Target&) noexcept { receipt.plan_generation = 0U; }},
      Case{"different plan generation",
           +[](Receipt& receipt, Target&) noexcept { ++receipt.plan_generation; }},
      Case{"zero run generation",
           +[](Receipt& receipt, Target&) noexcept { receipt.run_generation = 0U; }},
      Case{"different run generation",
           +[](Receipt& receipt, Target&) noexcept { ++receipt.run_generation; }},
      Case{"zero producer", +[](Receipt& receipt, Target&) noexcept { receipt.producer = {}; }},
      Case{"different producer",
           +[](Receipt& receipt, Target&) noexcept { receipt.producer = VertexId{30U}; }},
      Case{"zero destination",
           +[](Receipt& receipt, Target&) noexcept { receipt.destination = {}; }},
      Case{"different destination",
           +[](Receipt& receipt, Target&) noexcept { receipt.destination = VertexId{40U}; }},
      Case{"zero source output",
           +[](Receipt& receipt, Target&) noexcept { receipt.source_output = {}; }},
      Case{"different source output",
           +[](Receipt& receipt, Target&) noexcept { receipt.source_output = PortId{50U}; }},
      Case{"zero destination input",
           +[](Receipt& receipt, Target&) noexcept { receipt.destination_input = {}; }},
      Case{"different destination input",
           +[](Receipt& receipt, Target&) noexcept { receipt.destination_input = PortId{60U}; }},
      Case{"empty service",
           +[](Receipt& receipt, Target&) noexcept { receipt.service = std::string_view{}; }},
      Case{"different service",
           +[](Receipt& receipt, Target&) noexcept {
             receipt.service = "process-local-channel.other";
           }},
      Case{"zero readiness source",
           +[](Receipt& receipt, Target&) noexcept { receipt.readiness_source = {}; }},
      Case{"different readiness source",
           +[](Receipt& receipt, Target&) noexcept {
             receipt.readiness_source = make_source_id(70U);
           }},
      Case{"invalid reference clock",
           +[](Receipt& receipt, Target&) noexcept { receipt.reference_clock = {}; }},
      Case{"different reference clock",
           +[](Receipt& receipt, Target&) noexcept { receipt.reference_clock = ClockId{80U}; }},
      Case{"zero source generation",
           +[](Receipt& receipt, Target&) noexcept { receipt.source_generation = 0U; }},
      Case{"different source generation",
           +[](Receipt& receipt, Target&) noexcept { ++receipt.source_generation; }},
      Case{"zero readiness sequence",
           +[](Receipt& receipt, Target&) noexcept { receipt.readiness_sequence = 0U; }},
      Case{"different live readiness sequence",
           +[](Receipt&, Target& target) noexcept { ++target.readiness_sequence; }},
      Case{"zero producer epoch",
           +[](Receipt& receipt, Target&) noexcept { receipt.producer_epoch = 0U; }},
      Case{"zero publication sequence",
           +[](Receipt& receipt, Target&) noexcept { receipt.publication_sequence = 0U; }},
      Case{"zero activation generation",
           +[](Receipt& receipt, Target&) noexcept { receipt.source_activation_generation = 0U; }},
      Case{"different live activation generation",
           +[](Receipt&, Target& target) noexcept { ++target.source_activation_generation; }},
      Case{"different live destination revision",
           +[](Receipt&, Target& target) noexcept { ++target.destination_revision; }},
      Case{"different observed destination revision",
           +[](Receipt&, Target& target) noexcept { ++target.observed_destination_revision; }},
      Case{"different execution lane",
           +[](Receipt& receipt, Target&) noexcept { ++receipt.execution_lane; }},
      Case{"zero successors",
           +[](Receipt& receipt, Target&) noexcept { receipt.successor_publications = 0U; }},
      Case{"multiple successors",
           +[](Receipt& receipt, Target&) noexcept { receipt.successor_publications = 2U; }},
      Case{"non-success pre-submit status",
           +[](Receipt& receipt, Target&) noexcept {
             receipt.publication_status = ErrorCode::kBackpressured;
           }},
      Case{"authority-consuming partial status",
           +[](Receipt& receipt, Target&) noexcept {
             receipt.publication_status = ErrorCode::kPublicationPartial;
           }},
      Case{"authority-consuming indeterminate status",
           +[](Receipt& receipt, Target&) noexcept {
             receipt.publication_status = ErrorCode::kPublicationIndeterminate;
           }},
      Case{"revision predecessor overflow",
           +[](Receipt& receipt, Target&) noexcept {
             receipt.prior_destination_revision = std::numeric_limits<std::uint64_t>::max();
             receipt.destination_revision = 0U;
           }},
      Case{"revision did not advance",
           +[](Receipt& receipt, Target&) noexcept {
             receipt.destination_revision = receipt.prior_destination_revision;
           }},
      Case{"revision advanced by more than one",
           +[](Receipt& receipt, Target&) noexcept {
             receipt.destination_revision = receipt.prior_destination_revision + 2U;
           }},
  };

  ASSERT_TRUE(runtime_channel_direct_continuation_matches(valid, expected));
  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    Receipt receipt = valid;
    Target target = expected;
    test_case.mutate(receipt, target);

    // Key assertion: no malformed, stale, multi-successor, or non-success evidence can suppress
    // the ordinary readiness-poll dispatch.
    EXPECT_FALSE(runtime_channel_direct_continuation_matches(receipt, target));
  }
}

TEST(RuntimeChannelBinding, OrdinarySessionRetainsNoClockHealthSerializationBoundary) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());

  ClockRegistryBuilder registry_builder;
  const ClockKey clock_key = ClockKey::named("ordinary-default");
  ASSERT_TRUE(registry_builder.add_clock(clock_key,
                                         std::make_shared<ManualClock>(0),
                                         ClockCapabilityOptions{
                                             .can_read_now = true,
                                             .can_wait_until = true,
                                             .steady = true,
                                         }));
  ASSERT_TRUE(registry_builder.set_default(clock_key));
  auto registry = registry_builder.build();
  ASSERT_TRUE(registry) << runtime_error_message(registry.error());
  auto health = EndpointClockHealthRuntimeCoordinator::create(
      {}, {}, fixture.plan.plan_generation, RunGenerationId{1U}, *registry);
  ASSERT_TRUE(health) << (health ? std::string_view{} : health.error().token);
  ASSERT_EQ((*health)->receiver_count(), 0U);

  std::mutex unused_serialization;
  ASSERT_TRUE((*session)->bind_clock_health(**health, unused_serialization));
  EXPECT_FALSE(RuntimeChannelSessionTestAccess::clock_health_serialization_bound(**session));
}

// Port-level Tensor validation consumes only trusted snapshots. An absent or failed snapshot must
// not match an otherwise unconstrained layout or contract through default-initialized metadata.
TEST(RuntimeChannelBinding, InvalidTensorSnapshotsFailClosedBeforePortContractValidation) {
  const Tensor tensor;
  const TensorRepresentation layout;
  const TensorContractRecord contract;

  // Key setup: exercise both an uninitialized snapshot and an initialized failed snapshot.
  const TensorAccessSnapshot uninitialized;
  TensorAccessSnapshot invalid;
  invalid.initialized = true;

  // Key assertion: neither snapshot can satisfy even an unconstrained Tensor port declaration.
  EXPECT_FALSE(tensor_matches_port_layout(tensor, uninitialized, layout));
  EXPECT_FALSE(tensor_matches_port_contract(tensor, uninitialized, contract));
  EXPECT_FALSE(tensor_matches_port_layout(tensor, invalid, layout));
  EXPECT_FALSE(tensor_matches_port_contract(tensor, invalid, contract));
}

// The private runtime constructors consume exact frozen plan relations. Reject corrupt records
// before ChannelRuntime or either transport family can create a service, publisher, or subscriber.
// REQ: CHAN-087
// REQ: RUNTIME-015
TEST(RuntimeChannelBinding, RejectsMalformedProcessLocalAndExternalPlansBeforeBackendConstruction) {
  SessionFixture process_local;
  process_local.plan.process_local_channel_services.front().service_id += ".corrupt";
  auto malformed_process_local =
      RuntimeChannelSession::create(process_local.plan, process_local.replay);
  ASSERT_FALSE(malformed_process_local);
  EXPECT_EQ(malformed_process_local.error().code, ErrorCode::kInvalidArgument);

  SessionFixture unbucketed_process_local;
  auto& unbucketed_service = unbucketed_process_local.plan.process_local_channel_services.front();
  unbucketed_service.payload_size_bytes = std::numeric_limits<std::uint64_t>::max();
  unbucketed_service.selected_inline_bucket_bytes = 0U;
  auto malformed_bucket = process_local_service_options(unbucketed_service);
  ASSERT_FALSE(malformed_bucket);
  EXPECT_EQ(malformed_bucket.error().code, ErrorCode::kInvalidArgument);

  SessionFixture unresolved_interface;
  unresolved_interface.plan.ports.front().owner_kind = PortOwnerKind::kGraphInterface;
  auto rejected_interface =
      RuntimeChannelSession::create(unresolved_interface.plan, unresolved_interface.replay);
  ASSERT_FALSE(rejected_interface);
  EXPECT_EQ(rejected_interface.error().code, ErrorCode::kInvalidArgument);

  Graph graph{"malformed-external-runtime-plan"};
  const ClockKey default_clock =
      graph.bind_shared_clock("runtime-clock", std::make_shared<ManualClock>(0));
  graph.set_default_clock(default_clock);
  const auto imported = graph.import_topic<std::int32_t>("holomq://" + unique_service());
  const auto sink = graph.op<SinkOperator>("sink");
  graph.add_flow(imported.output(), sink->input);
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  ExecutionPlanState malformed_external = *state;
  ASSERT_FALSE(malformed_external.channel_services.empty());
  malformed_external.channel_services.front().service_id += ".corrupt";
  auto rejected = RuntimeChannelSession::create(malformed_external, {});
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kInvalidArgument);

  enum class GenerationCapacityDrift : std::uint8_t {
    kZeroEndpoint,
    kServiceBelowEndpoint,
    kWrongRuntimeAggregate,
    kShortReleaseLedger,
    kMissingProfile,
    kRestartReservationMismatch,
    kFallbackAllowed,
  };
  constexpr std::array generation_drifts{
      GenerationCapacityDrift::kZeroEndpoint,
      GenerationCapacityDrift::kServiceBelowEndpoint,
      GenerationCapacityDrift::kWrongRuntimeAggregate,
      GenerationCapacityDrift::kShortReleaseLedger,
      GenerationCapacityDrift::kMissingProfile,
      GenerationCapacityDrift::kRestartReservationMismatch,
      GenerationCapacityDrift::kFallbackAllowed,
  };
  ASSERT_EQ(state->imported_generation_capacities.size(), 1U);
  for (const GenerationCapacityDrift drift : generation_drifts) {
    SCOPED_TRACE(static_cast<std::uint32_t>(drift));
    ExecutionPlanState malformed_capacity = *state;
    auto& capacity = malformed_capacity.imported_generation_capacities.front();
    switch (drift) {
      case GenerationCapacityDrift::kZeroEndpoint:
        capacity.endpoint_max_generations = 0U;
        break;
      case GenerationCapacityDrift::kServiceBelowEndpoint:
        capacity.endpoint_max_generations = capacity.service_max_generations + 1U;
        break;
      case GenerationCapacityDrift::kWrongRuntimeAggregate:
        ++capacity.runtime_max_generations;
        break;
      case GenerationCapacityDrift::kShortReleaseLedger:
        capacity.release_record_capacity = 0U;
        break;
      case GenerationCapacityDrift::kMissingProfile:
        capacity.effective_profile.reset();
        break;
      case GenerationCapacityDrift::kRestartReservationMismatch:
        capacity.restart_capable_minimum_two = true;
        capacity.endpoint_max_generations = 2U;
        capacity.service_max_generations = 2U;
        capacity.runtime_max_generations = 2U;
        capacity.release_record_capacity = 2U;
        capacity.quarantine_record_capacity = 2U;
        capacity.candidate_reservation_capacity = 0U;
        break;
      case GenerationCapacityDrift::kFallbackAllowed:
        capacity.fallback_rejected = false;
        break;
    }
    auto capacity_rejected = RuntimeChannelSession::create(malformed_capacity, {});
    ASSERT_FALSE(capacity_rejected);
    EXPECT_EQ(capacity_rejected.error().code, ErrorCode::kProtocolError);
  }
}

// A late replay mismatch is discovered only after earlier output and subscriber endpoints are
// owned by the unpublished construction transaction. Its failure must release all backend names
// so an exact retry can materialize the same frozen plan.
// REQ: RUNTIME-015
TEST(RuntimeChannelBinding, LateSubscriberConstructionFailureRollsBackOwnedEndpoints) {
  //          /--> first sink
  // source -|
  //          \--> second sink
  // Key setup: invalidate only the second subscriber replay so construction fails after the
  // shared publisher and first subscriber have been materialized.
  SessionFixture fixture;
  fixture.add_mixed_policy_fanout();
  auto malformed_replay = fixture.replay;
  const PortId second_port{3U};
  const auto second_input =
      std::find_if(malformed_replay.begin(), malformed_replay.end(), [&](const auto& replay) {
        return replay.port == second_port && replay.direction == PortDirection::kInput;
      });
  ASSERT_NE(second_input, malformed_replay.end());
  second_input->declaration = nullptr;

  auto rejected = RuntimeChannelSession::create(fixture.plan, malformed_replay);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kInvalidArgument);

  // Key assertion: rollback releases every partially constructed endpoint and the exact valid
  // retry can immediately reclaim the same compiler-frozen service names.
  auto retried = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(retried) << runtime_error_message(retried.error());
}

// An exact-owner wake callback always borrows the same state as the ordinary readiness wake.
// Reject a partial callback tuple before any Channel backend can retain and later invoke it.
// REQ: RUNTIME-015
TEST(RuntimeChannelBinding, RejectsOwnerProgressWakeWithoutBorrowedState) {
  SessionFixture fixture;
  auto rejected =
      RuntimeChannelSession::create(fixture.plan,
                                    fixture.replay,
                                    {},
                                    RuntimeChannelReadinessWake{
                                        .signal_owner_progress = [](void*, VertexId) noexcept {},
                                    });

  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kInvalidArgument);
}

// A controller-owned activation cut can win the transport-drain race after another partition's
// Operator publishes. The retained input must wake its exact owner instead of waiting for the
// owner's bounded fallback poll or broadcasting to every partition.
// REQ: RUNTIME-015
// REQ: RUNTIME-025
TEST(RuntimeChannelBinding, HoloMQDrainSignalsExactSubscriberOwner) {
  SessionFixture fixture;
  struct WakeProbe {
    std::counting_semaphore<4> owner_ready{0};
    std::atomic<std::uint32_t> owner{};
    std::atomic<std::uint32_t> generic_signals{};
  } wake;
  auto session = RuntimeChannelSession::create(
      fixture.plan,
      fixture.replay,
      {},
      RuntimeChannelReadinessWake{
          .state = &wake,
          .signal =
              [](void* state) noexcept {
                static_cast<WakeProbe*>(state)->generic_signals.fetch_add(
                    1U, std::memory_order_relaxed);
              },
          .signal_owner_progress =
              [](void* state, VertexId owner) noexcept {
                auto* const probe = static_cast<WakeProbe*>(state);
                probe->owner.store(owner.value, std::memory_order_relaxed);
                probe->owner_ready.release();
              },
      });
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 2U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  // Key setup: publish one HoloMQ sample, allowing either the pump or post-compute cut to retain
  // it.
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{42}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_TRUE((*session)->flush_process_local_activation_cut());
  ASSERT_TRUE(wake.owner_ready.try_acquire_for(1s));

  // Key assertion: either drain owner routes readiness only to the subscriber's partition owner.
  EXPECT_EQ(wake.owner.load(std::memory_order_relaxed), 2U);
  EXPECT_EQ(wake.generic_signals.load(std::memory_order_relaxed), 0U);

  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// Readiness bindings are owned outside RuntimeChannelSession. Releasing one must clear every
// InputPortState back-pointer so the same input can bind a replacement without retaining a stale
// direct-continuation candidate.
TEST(RuntimeChannelBinding, DestroyedReadinessBindingCanBeRecreatedForTheSameInput) {
  SessionFixture fixture;
  fixture.use_direct_retained_delivery();
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  const RuntimePortBindingRequest request{
      .plan_generation = fixture.plan.plan_generation,
      .run_generation = 1U,
      .kind = RuntimeEventKind::kInputArrival,
      .source = make_source_id(1U),
      .source_clock = ClockId{1U},
      .reference_clock = ClockId{1U},
      .capacity = 2U,
  };
  auto first = (*session)->create_readiness_binding(VertexId{2U}, request);
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  first->reset();

  auto replacement = (*session)->create_readiness_binding(VertexId{2U}, request);
  ASSERT_TRUE(replacement) << runtime_error_message(replacement.error());
  EXPECT_TRUE((*session)->arm_direct_continuation(VertexId{1U}, 0U));
}

// A binding may outlive the RuntimeChannelSession that supplied it. Teardown must detach the
// borrowed owner and input state so later binding destruction and status queries stay fail-closed.
TEST(RuntimeChannelBinding, ReadinessBindingOutlivesSessionWithoutDanglingState) {
  SessionFixture fixture;
  std::unique_ptr<PortBinding> binding;
  {
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    auto created =
        (*session)->create_readiness_binding(VertexId{2U},
                                             RuntimePortBindingRequest{
                                                 .plan_generation = fixture.plan.plan_generation,
                                                 .run_generation = 1U,
                                                 .kind = RuntimeEventKind::kInputArrival,
                                                 .source = make_source_id(1U),
                                                 .source_clock = ClockId{1U},
                                                 .reference_clock = ClockId{1U},
                                                 .capacity = 2U,
                                             });
    ASSERT_TRUE(created) << runtime_error_message(created.error());
    binding = std::move(*created);
  }

  ASSERT_TRUE(binding->closed());
  auto polled = binding->poll(TimePoint{1, ClockId{1U}});
  ASSERT_FALSE(polled);
  EXPECT_EQ(polled.error().code, ErrorCode::kClosed);
}

// A readiness source accepts observed-at coordinates only in its compiler-frozen reference clock;
// malformed timestamps must fail before retained input state is inspected or published.
TEST(RuntimeChannelBinding, ReadinessBindingRejectsInvalidObservedAtCoordinates) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 2U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());

  // Key setup: poll with an absent coordinate and then with the wrong clock identity.
  const auto absent = (*binding)->poll({});
  const auto wrong_clock = (*binding)->poll(TimePoint{1, ClockId{2U}});

  // Key assertion: both malformed coordinates fail without changing readiness state.
  ASSERT_FALSE(absent);
  EXPECT_EQ(absent.error().code, ErrorCode::kInvalidArgument);
  ASSERT_FALSE(wrong_clock);
  EXPECT_EQ(wrong_clock.error().code, ErrorCode::kInvalidArgument);
}

// Terminal drain cancels an exhausted readiness source only after its retained input state retires;
// the canceled source must remain closed rather than admitting another poll.
// REQ: CHAN-136
TEST(RuntimeChannelBinding, CancelledReadinessBindingRejectsPoll) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 2U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());

  (*binding)->cancel();

  auto polled = (*binding)->poll(TimePoint{1, ClockId{1U}});
  ASSERT_FALSE(polled);
  EXPECT_EQ(polled.error().code, ErrorCode::kClosed);
}

// A direct-continuation target exposes a service view, so SubscriberState owns the string rather
// than borrowing mutable plan storage.
TEST(RuntimeChannelBinding, DirectContinuationTargetOwnsServiceIdentity) {
  SessionFixture fixture;
  fixture.use_direct_retained_delivery();
  const std::string expected_service =
      fixture.plan.process_local_channel_services.front().service_id;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 2U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());

  fixture.plan.process_local_channel_services.front().service_id.assign(4096U, 'x');
  auto target = (*session)->arm_direct_continuation(VertexId{1U}, 0U);
  ASSERT_TRUE(target);
  // The fatal assertion above proves the optional is engaged.
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  EXPECT_EQ(target->service, expected_service);
}

// A same-lane eventless HoloMQ publication relies on the serialized post-compute activation cut.
// It is deliberately ineligible for direct-continuation receipt bookkeeping.
// REQ: CHAN-101
// REQ: RUNTIME-019
TEST(RuntimeChannelBinding, HoloMQScalarUsesOrdinaryActivationCutExactlyOnce) {
  // Key setup: publish on an exact 1:1 eventless HoloMQ edge with a shared execution lane.
  SessionFixture fixture;
  fixture.use_one_slot_blocking_transport();
  fixture.use_same_partition_eventless_transport();
  const auto counters = std::make_shared<RuntimePhaseCounters>();
  fixture.plan.executable_factories.phase_counters = counters;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
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
  EXPECT_FALSE((*session)->arm_direct_continuation(VertexId{1U}, 0U));

  ASSERT_TRUE(fixture.source.output.emit(Sample<std::int32_t>{
      42, SampleMetadata{.capture_timestamp_ns = 10, .source_clock_id = ClockId{1U}}}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_TRUE((*session)->flush_process_local_activation_cut());

  // Key assertion: one ordinary cut commits typed ingress and readiness exactly once without a
  // receipt candidate, leaving no reservation or duplicate work behind.
  const auto reservations =
      RuntimeChannelSessionTestAccess::typed_ingress_reservations(**session, PortId{2U});
  ASSERT_TRUE(reservations) << runtime_error_message(reservations.error());
  EXPECT_EQ(*reservations, 0U);
  auto snapshot = counters->snapshot();
  const auto phase = [&](RuntimePhaseCounter counter) {
    return snapshot[RuntimePhaseCounters::index(counter)];
  };
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelActivationCuts), 1U);
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelActivationCutDrains) +
                phase(RuntimePhaseCounter::kChannelActivationCutSkips),
            1U);
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelSamplesReceived), 1U);
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelReadinessPublished), 1U);

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
  auto received = fixture.sink.input.receive();
  ASSERT_TRUE(received) << runtime_error_message(received.error());
  EXPECT_EQ(received->data, 42);
  EXPECT_EQ(received->metadata.sequence, 1U);
  ASSERT_TRUE((*binding)->retire(event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  ASSERT_TRUE((*session)->retire_samples(PortId{2U}, selection.front().samples));

  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// Independent partition owners may publish concurrently, so their shared Channel session keeps
// the thread-safe complete-inventory cut even when every HoloMQ service is eventless.
// REQ: CHAN-101
// REQ: RUNTIME-019
// REQ: RUNTIME-025
TEST(RuntimeChannelBinding, OwnerRoutedEventlessCutRetainsCompleteInventoryScan) {
  // Key setup: provide the exact-owner progress route used by a multi-partition RunSession, then
  // publish through an otherwise index-eligible eventless blocking service.
  SessionFixture fixture;
  fixture.use_one_slot_blocking_transport();
  fixture.use_same_partition_eventless_transport();
  const auto counters = std::make_shared<RuntimePhaseCounters>();
  fixture.plan.executable_factories.phase_counters = counters;
  struct WakeProbe {
    std::atomic<std::uint32_t> owner{};
  } wake;
  auto session = RuntimeChannelSession::create(
      fixture.plan,
      fixture.replay,
      {},
      RuntimeChannelReadinessWake{
          .state = &wake,
          .signal = [](void*) noexcept {},
          .signal_owner_progress =
              [](void* state, VertexId owner) noexcept {
                static_cast<WakeProbe*>(state)->owner.store(owner.value, std::memory_order_relaxed);
              },
      });
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
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
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{42}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_TRUE((*session)->flush_process_local_activation_cut());

  // Key assertion: the cut queries the backend through the complete-inventory path and signals
  // the exact subscriber owner instead of mutating single-owner index scratch.
  const auto snapshot = counters->snapshot();
  EXPECT_EQ(snapshot[RuntimePhaseCounters::index(
                RuntimePhaseCounter::kChannelActivationCutTransportProbes)],
            1U);
  EXPECT_EQ(wake.owner.load(std::memory_order_relaxed), 2U);

  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// An eventless process-local service has no pump thread to revisit a transport head that could not
// enter full typed storage. Releasing either the activation or temporal claim must perform that
// bounded retry without requiring another source callback.
// REQ: CHAN-101
// REQ: RUNTIME-019
TEST(RuntimeChannelBinding, EventlessBlockedTransportRetriesWhenTypedStorageIsReleased) {
  const auto run_retirement_order = [](const bool retire_samples_first) {
    // Key setup: retain the first depth-one sample in an active sink, then leave a second HoloMQ
    // sample blocked at the eventless activation cut.
    SessionFixture fixture;
    fixture.use_one_slot_blocking_transport();
    fixture.use_same_partition_eventless_transport();
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    auto binding =
        (*session)->create_readiness_binding(VertexId{2U},
                                             RuntimePortBindingRequest{
                                                 .plan_generation = fixture.plan.plan_generation,
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
    const auto indexes = RuntimeChannelSessionTestAccess::pump_index_state(**session);
    ASSERT_TRUE(indexes) << runtime_error_message(indexes.error());
    EXPECT_FALSE(indexes->background_thread_required);
    EXPECT_FALSE(indexes->background_thread_running);

    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
    ASSERT_TRUE(fixture.source.output.emit(std::int32_t{41}));
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
    ASSERT_TRUE((*session)->flush_process_local_activation_cut());
    auto first_event = wait_for_event(**session, **binding);
    ASSERT_TRUE(first_event) << runtime_error_message(first_event.error());
    auto first_sample =
        (*session)->temporal_sample(first_event->port, first_event->source_sequence);
    ASSERT_TRUE(first_sample) << runtime_error_message(first_sample.error());
    const std::array selection{RuntimeChannelActivationSelection{
        .port = PortId{2U},
        .selection = {InputState::kSelected, SelectionReason::kNone},
        .samples = std::span<const TemporalSampleRecord>{&*first_sample, 1U},
        .destructive = true,
    }};
    ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selection));
    auto first_value = fixture.sink.input.receive();
    ASSERT_TRUE(first_value) << runtime_error_message(first_value.error());
    EXPECT_EQ(first_value->data, 41);

    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 2U, {}));
    ASSERT_TRUE(fixture.source.output.emit(std::int32_t{42}));
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 2U, true));
    ASSERT_TRUE((*session)->flush_process_local_activation_cut());
    ASSERT_TRUE((*binding)->retire(first_event->source_sequence));
    if (retire_samples_first) {
      ASSERT_TRUE((*session)->retire_samples(first_event->port, selection.front().samples));
      ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
    } else {
      ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
      ASSERT_TRUE((*session)->retire_samples(first_event->port, selection.front().samples));
    }

    // Key assertion: whichever retirement releases the final typed-slot claim synchronously
    // materializes sequence two while the session remains pump-free.
    auto second_event = wait_for_event(**session, **binding);
    ASSERT_TRUE(second_event) << runtime_error_message(second_event.error());
    auto second_sample =
        (*session)->temporal_sample(second_event->port, second_event->source_sequence);
    ASSERT_TRUE(second_sample) << runtime_error_message(second_sample.error());
    EXPECT_EQ(second_sample->source_sequence, 2U);
    const auto after_retry = RuntimeChannelSessionTestAccess::pump_index_state(**session);
    ASSERT_TRUE(after_retry) << runtime_error_message(after_retry.error());
    EXPECT_FALSE(after_retry->background_thread_required);
    EXPECT_FALSE(after_retry->background_thread_running);

    ASSERT_TRUE((*binding)->retire(second_event->source_sequence));
    ASSERT_TRUE((*session)->retire_samples(
        second_event->port, std::span<const TemporalSampleRecord>{&*second_sample, 1U}));
    (*session)->cancel();
    ASSERT_TRUE((*session)->join());
  };

  run_retirement_order(true);
  run_retirement_order(false);
}

// An event-enabled HoloMQ subscriber can make progress independently of a successful compute cut
// and therefore retains the background pump.
// REQ: CHAN-101
// REQ: RUNTIME-019
TEST(RuntimeChannelBinding, EventfulHoloMQRetainsBackgroundPump) {
  // Key setup: arm one ordinary event-enabled HoloMQ subscriber with its WaitSet attachment.
  SessionFixture fixture;
  fixture.use_one_slot_blocking_transport();
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  const auto indexes = RuntimeChannelSessionTestAccess::pump_index_state(**session);
  ASSERT_TRUE(indexes) << runtime_error_message(indexes.error());
  // Key assertion: any attached transport event keeps exactly one background pump running.
  EXPECT_TRUE(indexes->exact);
  EXPECT_GT(indexes->wait_handle_count, 0U);
  EXPECT_EQ(indexes->drain_subscriber_count, 1U);
  EXPECT_TRUE(indexes->background_thread_required);
  EXPECT_TRUE(indexes->background_thread_running);

  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// Source-tree benchmarks need an exact pump-backed control without changing event attachment,
// graph topology, or HoloMQ delivery.
// REQ: CHAN-101
// REQ: RUNTIME-019
TEST(RuntimeChannelBinding, ColdControlRetainsPumpForAllEventlessInventory) {
  // Key setup: force only background-pump startup on an exact eventless HoloMQ service.
  SessionFixture fixture;
  fixture.use_one_slot_blocking_transport();
  fixture.use_same_partition_eventless_transport();
  fixture.plan.executable_factories.test_force_process_local_channel_pump = true;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  const auto indexes = RuntimeChannelSessionTestAccess::pump_index_state(**session);
  ASSERT_TRUE(indexes) << runtime_error_message(indexes.error());
  // Key assertion: the control starts the pump while retaining zero event-wake attachments.
  EXPECT_TRUE(indexes->exact);
  EXPECT_EQ(indexes->wait_handle_count, 0U);
  EXPECT_EQ(indexes->drain_subscriber_count, 1U);
  EXPECT_TRUE(indexes->background_thread_required);
  EXPECT_TRUE(indexes->background_thread_running);

  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// One shared HoloMQ publication accounts for every accepted recipient before the serial
// controller establishes its post-compute activation cut.
// REQ: CHAN-101
// REQ: RUNTIME-019
TEST(RuntimeChannelBinding, EventlessScalarFanoutMaterializesEveryRecipientAtActivationCut) {
  SessionFixture fixture;
  fixture.use_one_slot_blocking_fanout();
  fixture.use_same_partition_eventless_transport();
  fixture.use_single_partition_runner();
  const auto counters = std::make_shared<RuntimePhaseCounters>();
  fixture.plan.executable_factories.phase_counters = counters;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto left_binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 1U,
                                           });
  auto right_binding =
      (*session)->create_readiness_binding(VertexId{3U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(2U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 1U,
                                           });
  ASSERT_TRUE(left_binding) << runtime_error_message(left_binding.error());
  ASSERT_TRUE(right_binding) << runtime_error_message(right_binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  // Key setup: publish once to two eventless subscribers in the same execution partition, then
  // establish the ordinary activation cut before observing either readiness stream.
  auto indexes = RuntimeChannelSessionTestAccess::pump_index_state(**session);
  ASSERT_TRUE(indexes) << runtime_error_message(indexes.error());
  EXPECT_TRUE(indexes->exact);
  EXPECT_EQ(indexes->wait_handle_count, 0U);
  EXPECT_EQ(indexes->drain_subscriber_count, 2U);
  EXPECT_FALSE(indexes->background_thread_required);
  EXPECT_FALSE(indexes->background_thread_running);
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{42}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_TRUE((*session)->flush_process_local_activation_cut());
  auto left_event = wait_for_event(**session, **left_binding);
  auto right_event = wait_for_event(**session, **right_binding);
  ASSERT_TRUE(left_event) << runtime_error_message(left_event.error());
  ASSERT_TRUE(right_event) << runtime_error_message(right_event.error());

  // Key assertion: the single publication becomes two typed samples/readiness witnesses while
  // using no WaitSet attachment and exactly one bounded-or-skipped activation cut.
  const auto snapshot = counters->snapshot();
  const auto phase = [&](RuntimePhaseCounter counter) {
    return snapshot[RuntimePhaseCounters::index(counter)];
  };
  EXPECT_EQ(left_event->port, PortId{2U});
  EXPECT_EQ(right_event->port, PortId{3U});
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelActivationCuts), 1U);
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelActivationCutDrains) +
                phase(RuntimePhaseCounter::kChannelActivationCutSkips),
            1U);
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelSamplesReceived), 2U);
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelReadinessPublished), 2U);
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelActivationCutInventoryScans), 0U);
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelActivationCutSubscriberProbes), 2U);
  // The accepted publication supplies the exact recipients, so the cut performs no redundant
  // backend readiness query before receiving the two immediately visible HoloMQ heads.
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelActivationCutTransportProbes), 0U);
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelActivationCutMaterializations), 2U);

  auto left_sample = (*session)->temporal_sample(left_event->port, left_event->source_sequence);
  auto right_sample = (*session)->temporal_sample(right_event->port, right_event->source_sequence);
  ASSERT_TRUE(left_sample) << runtime_error_message(left_sample.error());
  ASSERT_TRUE(right_sample) << runtime_error_message(right_sample.error());
  EXPECT_EQ(left_sample->producer_epoch, right_sample->producer_epoch);
  EXPECT_EQ(left_sample->source_sequence, right_sample->source_sequence);
  ASSERT_TRUE((*left_binding)->retire(left_event->source_sequence));
  ASSERT_TRUE((*right_binding)->retire(right_event->source_sequence));
  ASSERT_TRUE((*session)->retire_samples(left_event->port,
                                         std::span<const TemporalSampleRecord>{&*left_sample, 1U}));
  ASSERT_TRUE((*session)->retire_samples(
      right_event->port, std::span<const TemporalSampleRecord>{&*right_sample, 1U}));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// A depth-one lossy service can replace both subscriber heads during one producer callback. The
// activation cut must materialize the common newest sequence without relying on event handles.
// REQ: CHAN-101
// REQ: CHAN-115
// REQ: RUNTIME-019
TEST(RuntimeChannelBinding, EventlessDropOldestFanoutMaterializesNewestSequenceForEveryRecipient) {
  SessionFixture fixture;
  fixture.use_one_slot_drop_oldest_fanout();
  fixture.use_same_partition_eventless_transport();
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto left_binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 1U,
                                           });
  auto right_binding =
      (*session)->create_readiness_binding(VertexId{3U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(2U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 1U,
                                           });
  ASSERT_TRUE(left_binding) << runtime_error_message(left_binding.error());
  ASSERT_TRUE(right_binding) << runtime_error_message(right_binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  // Key setup: publish two values in one activation to replace both depth-one transport heads
  // before the eventless service reaches its post-compute activation cut.
  auto indexes = RuntimeChannelSessionTestAccess::pump_index_state(**session);
  ASSERT_TRUE(indexes) << runtime_error_message(indexes.error());
  EXPECT_EQ(indexes->wait_handle_count, 0U);
  EXPECT_EQ(indexes->drain_subscriber_count, 2U);
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{41}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{42}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_TRUE((*session)->flush_process_local_activation_cut());
  auto left_event = wait_for_event(**session, **left_binding);
  auto right_event = wait_for_event(**session, **right_binding);
  ASSERT_TRUE(left_event) << runtime_error_message(left_event.error());
  ASSERT_TRUE(right_event) << runtime_error_message(right_event.error());
  auto left_sample = (*session)->temporal_sample(left_event->port, left_event->source_sequence);
  auto right_sample = (*session)->temporal_sample(right_event->port, right_event->source_sequence);
  ASSERT_TRUE(left_sample) << runtime_error_message(left_sample.error());
  ASSERT_TRUE(right_sample) << runtime_error_message(right_sample.error());

  // Key assertion: both legs expose only sequence two, and each typed read observes the common
  // newest payload exactly once despite the absent WaitSet attachments.
  EXPECT_EQ(left_sample->source_sequence, 2U);
  EXPECT_EQ(right_sample->source_sequence, 2U);
  const std::array left_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&*left_sample, 1U},
      .destructive = true,
  }};
  const std::array right_selection{RuntimeChannelActivationSelection{
      .port = PortId{3U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&*right_sample, 1U},
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, left_selection));
  auto left_value = fixture.sink.input.receive();
  ASSERT_TRUE(left_value) << runtime_error_message(left_value.error());
  EXPECT_EQ(left_value->data, 42);
  ASSERT_TRUE((*left_binding)->retire(left_event->source_sequence));
  ASSERT_TRUE((*session)->retire_samples(left_event->port, left_selection.front().samples));
  auto left_retired = (*session)->retire_activation(VertexId{2U}, 1U, true);
  ASSERT_TRUE(left_retired) << runtime_error_message(left_retired.error());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{3U}, 1U, right_selection));
  auto right_value = fixture.second_sink.input.receive();
  ASSERT_TRUE(right_value) << runtime_error_message(right_value.error());
  EXPECT_EQ(right_value->data, 42);
  ASSERT_TRUE((*right_binding)->retire(right_event->source_sequence));
  ASSERT_TRUE((*session)->retire_samples(right_event->port, right_selection.front().samples));
  auto right_retired = (*session)->retire_activation(VertexId{3U}, 1U, true);
  ASSERT_TRUE(right_retired) << runtime_error_message(right_retired.error());
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

TEST(RuntimeChannelBinding, HoloMQWithoutPublicationSkipsOrdinaryActivationCutDrain) {
  // Key setup: retire an eventless HoloMQ producer activation without publishing a sample.
  SessionFixture fixture;
  fixture.use_one_slot_blocking_transport();
  fixture.use_same_partition_eventless_transport();
  const auto counters = std::make_shared<RuntimePhaseCounters>();
  fixture.plan.executable_factories.phase_counters = counters;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
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
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  EXPECT_FALSE((*session)->arm_direct_continuation(VertexId{1U}, 0U));

  // Key assertion: the ordinary cut observes balanced progress and skips its serialized drain.
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_TRUE((*session)->flush_process_local_activation_cut());
  const auto snapshot = counters->snapshot();
  EXPECT_EQ(snapshot[RuntimePhaseCounters::index(RuntimePhaseCounter::kChannelActivationCuts)], 1U);
  EXPECT_EQ(snapshot[RuntimePhaseCounters::index(RuntimePhaseCounter::kChannelActivationCutSkips)],
            1U);
  EXPECT_EQ(snapshot[RuntimePhaseCounters::index(RuntimePhaseCounter::kChannelActivationCutDrains)],
            0U);

  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// A graph-owned HoloMQ publisher must return finite pressure instead of waiting while its caller
// owns the serial dispatch resource. The retry remains one logical publication after capacity is
// released.
// REQ: CHAN-067
// REQ: CHAN-076
// REQ: CHAN-101
// REQ: RUNTIME-003
// REQ: RUNTIME-019
TEST(RuntimeChannelBinding, BlockingTransportBackpressureIsBoundedAndRetryDoesNotDuplicate) {
  SessionFixture fixture;
  fixture.use_one_slot_blocking_transport();
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 1U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{42}));

  std::promise<void> entered_emit;
  auto entered = entered_emit.get_future();
  auto pressure =
      std::async(std::launch::async, [&fixture, signal = std::move(entered_emit)]() mutable {
        signal.set_value();
        const auto started = std::chrono::steady_clock::now();
        auto result = fixture.source.output.emit(std::int32_t{43});
        return std::make_pair(std::move(result), std::chrono::steady_clock::now() - started);
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
    // Never strand a red-regression worker on the legacy blocking transport loop. Starting the
    // pump releases capacity so the baseline call can return before this assertion fails.
    ASSERT_TRUE((*session)->arm());
    auto event = wait_for_event(**session, **binding);
    ASSERT_TRUE(event) << runtime_error_message(event.error());
    ASSERT_EQ(pressure.wait_for(2s), std::future_status::ready);
    static_cast<void>(pressure.get());
    (*session)->cancel();
    ASSERT_TRUE((*session)->join());
    FAIL() << "graph publication waited while owning the serial dispatch resource";
  }
  auto [rejected, pressure_duration] = pressure.get();
  pressure_cleanup.release();
  // Key assertion: graph binding selected finite retry rather than immediate rejection or
  // unbounded service blocking. The exact backend budget is 5 ms; retain scheduling margin here.
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kBackpressured);
  EXPECT_GE(pressure_duration, 4ms);
  EXPECT_LT(pressure_duration, 1s);

  ASSERT_TRUE((*session)->arm());
  auto first_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(first_event) << runtime_error_message(first_event.error());
  auto first_temporal =
      (*session)->temporal_sample(first_event->port, first_event->source_sequence);
  ASSERT_TRUE(first_temporal) << runtime_error_message(first_temporal.error());
  const std::array first_selected{*first_temporal};
  const std::array first_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = first_selected,
      .batch = false,
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, first_selection));
  {
    auto received = fixture.sink.input.receive();
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    EXPECT_EQ(received->data, 42);
    EXPECT_EQ(received->metadata.sequence, 1U);
  }
  ASSERT_TRUE((*binding)->retire(first_event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  ASSERT_TRUE((*session)->retire_samples(PortId{2U}, first_selected));

  // The cold-bound callback storage must not add a warm accepted-publication allocation. Count
  // only the calling test thread so transport/pump progress cannot contaminate this evidence.
  std::size_t accepted_allocations = std::numeric_limits<std::size_t>::max();
  expected<void, Error> accepted;
  {
    const test::ScopedAllocationCounter allocations;
    accepted = fixture.source.output.emit(std::int32_t{43});
    accepted_allocations = allocations.count();
  }
  ASSERT_TRUE(accepted) << runtime_error_message(accepted.error());
  EXPECT_EQ(accepted_allocations, 0U);
  auto second_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(second_event) << runtime_error_message(second_event.error());
  auto second_temporal =
      (*session)->temporal_sample(second_event->port, second_event->source_sequence);
  ASSERT_TRUE(second_temporal) << runtime_error_message(second_temporal.error());
  const std::array second_selected{*second_temporal};
  const std::array second_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = second_selected,
      .batch = false,
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 2U, second_selection));
  {
    auto received = fixture.sink.input.receive();
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    EXPECT_EQ(received->data, 43);
    EXPECT_EQ(received->metadata.sequence, 2U);
  }
  ASSERT_TRUE((*binding)->retire(second_event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 2U, true));
  ASSERT_TRUE((*session)->retire_samples(PortId{2U}, second_selected));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// Cancellation and endpoint destruction must not depend on draining a graph-owned blocking
// queue. This covers the terminal path separately from the explicit retry path above.
// REQ: CHAN-101
// REQ: RUNTIME-020
// REQ: RUNTIME-024
TEST(RuntimeChannelBinding, FullBlockingTransportDoesNotDelayCancelJoinOrDestruction) {
  SessionFixture fixture;
  fixture.use_one_slot_blocking_transport();
  std::vector<RuntimeChannelTeardownPhase> phases;
  phases.reserve(3U);
  {
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_teardown_observer(
        **session, &phases, [](void* state, const RuntimeChannelTeardownPhase phase) noexcept {
          static_cast<std::vector<RuntimeChannelTeardownPhase>*>(state)->push_back(phase);
        }));
    ASSERT_TRUE((*session)->refresh_connections());
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
    ASSERT_TRUE(fixture.source.output.emit(std::int32_t{42}));

    std::promise<void> entered_emit;
    auto entered = entered_emit.get_future();
    auto pressure =
        std::async(std::launch::async, [&fixture, signal = std::move(entered_emit)]() mutable {
          signal.set_value();
          return fixture.source.output.emit(std::int32_t{43});
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
      // Release the legacy blocking implementation before failing so the regression cannot strand
      // a worker. The passing path never arms or drains the session.
      ASSERT_TRUE((*session)->arm());
      ASSERT_EQ(pressure.wait_for(2s), std::future_status::ready);
      static_cast<void>(pressure.get());
      (*session)->cancel();
      ASSERT_TRUE((*session)->join());
      FAIL() << "full graph-owned queue delayed cancellation eligibility";
    }
    const auto rejected = pressure.get();
    pressure_cleanup.release();
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, ErrorCode::kBackpressured);

    // Deliberately leave the first sample queued and the activation open. Explicit join and the
    // destructor must remain finite and preserve child-before-runtime teardown ordering.
    (*session)->cancel();
    ASSERT_TRUE((*session)->join());
  }

  const std::vector expected{
      RuntimeChannelTeardownPhase::kPumpJoined,
      RuntimeChannelTeardownPhase::kEndpointChildrenDestroyed,
      RuntimeChannelTeardownPhase::kRuntimeDestroyed,
  };
  EXPECT_EQ(phases, expected);
}

// A lifecycle-selected nominal rate may lower the active logical queue depth while retaining a
// previously admitted direct-process-local backing allocation. Candidate preparation remains
// private until publication, and increasing the active depth back to the physical ceiling proves
// that the backing allocation was not replaced.
// REQ: LIFECYCLE-012
// REQ: CHAN-141
TEST(RuntimeChannelBinding, DirectRetainedEnforcesSmallerPublishedActiveDepthOnLargerBacking) {
  SessionFixture fixture;
  fixture.use_direct_retained_delivery();
  fixture.expect_queue_depth(0U, 8U);
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());

  auto active = RuntimeChannelSessionTestAccess::active_queue_depth(**session, EdgeId{1U});
  ASSERT_TRUE(active) << runtime_error_message(active.error());
  EXPECT_EQ(*active, 8U);
  ASSERT_TRUE((*session)->prepare_rate_candidate_active_depth(EdgeId{1U}, 2U));
  active = RuntimeChannelSessionTestAccess::active_queue_depth(**session, EdgeId{1U});
  ASSERT_TRUE(active) << runtime_error_message(active.error());
  EXPECT_EQ(*active, 8U);
  ASSERT_TRUE((*session)->commit_rate_candidate_active_depths());
  active = RuntimeChannelSessionTestAccess::active_queue_depth(**session, EdgeId{1U});
  ASSERT_TRUE(active) << runtime_error_message(active.error());
  EXPECT_EQ(*active, 2U);

  ASSERT_TRUE((*session)->prepare_rate_candidate_active_depth(EdgeId{1U}, 8U));
  ASSERT_TRUE((*session)->commit_rate_candidate_active_depths());
  active = RuntimeChannelSessionTestAccess::active_queue_depth(**session, EdgeId{1U});
  ASSERT_TRUE(active) << runtime_error_message(active.error());
  EXPECT_EQ(*active, 8U);
  ASSERT_TRUE((*session)->prepare_rate_candidate_active_depth(EdgeId{1U}, 2U));
  ASSERT_TRUE((*session)->commit_rate_candidate_active_depths());

  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{11}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{12}));
  auto third = fixture.source.output.emit(std::int32_t{13});
  ASSERT_FALSE(third);
  EXPECT_EQ(third.error().code, ErrorCode::kBackpressured);
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// A non-blocking direct-retained subscriber has independent active-depth authority over both its
// typed slots and direct ingress staging. A lifecycle candidate may therefore lower and later
// restore the active logical depth without replacing the admitted physical backing.
// REQ: LIFECYCLE-012
// REQ: CHAN-141
TEST(RuntimeChannelBinding, DirectRetainedDropOldestPublishesActiveDepthWithinBacking) {
  SessionFixture fixture;
  fixture.use_two_slot_drop_oldest_queue();
  fixture.use_direct_retained_delivery();
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());

  auto active = RuntimeChannelSessionTestAccess::active_queue_depth(**session, EdgeId{1U});
  ASSERT_TRUE(active) << runtime_error_message(active.error());
  EXPECT_EQ(*active, 2U);

  ASSERT_TRUE((*session)->prepare_rate_candidate_active_depth(EdgeId{1U}, 1U));
  active = RuntimeChannelSessionTestAccess::active_queue_depth(**session, EdgeId{1U});
  ASSERT_TRUE(active) << runtime_error_message(active.error());
  EXPECT_EQ(*active, 2U);
  ASSERT_TRUE((*session)->commit_rate_candidate_active_depths());
  active = RuntimeChannelSessionTestAccess::active_queue_depth(**session, EdgeId{1U});
  ASSERT_TRUE(active) << runtime_error_message(active.error());
  EXPECT_EQ(*active, 1U);

  ASSERT_TRUE((*session)->prepare_rate_candidate_active_depth(EdgeId{1U}, 2U));
  active = RuntimeChannelSessionTestAccess::active_queue_depth(**session, EdgeId{1U});
  ASSERT_TRUE(active) << runtime_error_message(active.error());
  EXPECT_EQ(*active, 1U);
  ASSERT_TRUE((*session)->commit_rate_candidate_active_depths());
  active = RuntimeChannelSessionTestAccess::active_queue_depth(**session, EdgeId{1U});
  ASSERT_TRUE(active) << runtime_error_message(active.error());
  EXPECT_EQ(*active, 2U);

  ASSERT_TRUE((*session)->prepare_rate_candidate_active_depth(EdgeId{1U}, 1U));
  ASSERT_TRUE((*session)->commit_rate_candidate_active_depths());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{11}));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_single_occupied_claim_state(
      **session, PortId{2U}, true, false, false, false));

  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{22}));
  const auto retained = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(retained) << runtime_error_message(retained.error());
  EXPECT_EQ(retained->occupied_samples, 1U);
  EXPECT_EQ(retained->pending_direct_ingress_samples, 1U);
  EXPECT_EQ(retained->next_arrival_sequence, 3U);

  // The physical two-slot allocation remains intact, but active depth one permits only one
  // pending direct-ingress survivor. A newer publication replaces that survivor instead of
  // exposing the inactive typed capacity.
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{33}));
  const auto replaced = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(replaced) << runtime_error_message(replaced.error());
  EXPECT_EQ(replaced->occupied_samples, 1U);
  EXPECT_EQ(replaced->pending_direct_ingress_samples, 1U);
  EXPECT_EQ(replaced->next_arrival_sequence, 4U);

  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// A transport-backed HoloMQ subscriber has a construction-time queue limit and no independent
// active-depth control in EA2. It must reject a smaller logical depth instead of pretending that
// retained surplus capacity is hidden.
// REQ: LIFECYCLE-012
// REQ: CHAN-141
TEST(RuntimeChannelBinding, HoloMQRejectsSmallerActiveDepthOnRetainedTransportBacking) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());

  auto rejected = (*session)->prepare_rate_candidate_active_depth(EdgeId{1U}, 2U);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kNotSupported);
  EXPECT_EQ(rejected.error().message, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE");
  const auto active = RuntimeChannelSessionTestAccess::active_queue_depth(**session, EdgeId{1U});
  ASSERT_TRUE(active) << runtime_error_message(active.error());
  EXPECT_EQ(*active, 8U);
}

// A source-tree-only sampler joins real queue transitions to the immutable plan and published
// rate-generation row. It measures typed occupancy rather than reporting selected capacity as an
// observed high-water mark, and it leaves counters unavailable until a real attempt is bracketed.
// REQ: COMPILE-070
// REQ: RUNTIME-019
TEST(RuntimeChannelBinding, QueueObservationJoinsRateGenerationToMeasuredQueueOutcomes) {
  constexpr RunGenerationId kRunGeneration{991U};
  Graph graph{"runtime-rate-queue-observation"};
  const auto source = graph.op<RateObservedQueueSource>("source");
  const auto staged_sink = graph.op<RateObservedQueueSink>("staged-sink", true);
  const auto lossy_sink = graph.op<RateObservedQueueSink>("lossy-sink");
  const auto blocking_sink = graph.op<RateObservedQueueSink>("blocking-sink");
  graph.add_flow(source->output,
                 staged_sink->input,
                 ConnectionOptions{
                     .queue_policy = QueuePolicy::kDropOldest,
                     .queue_depth = 1U,
                 });
  graph.add_flow(source->output,
                 lossy_sink->input,
                 ConnectionOptions{
                     .queue_policy = QueuePolicy::kDropOldest,
                     .queue_depth = 2U,
                 });
  graph.add_flow(source->output,
                 blocking_sink->input,
                 ConnectionOptions{
                     .queue_policy = QueuePolicy::kBlocking,
                     .queue_depth = 2U,
                 });
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-rate-observation-clock"));
  const ExecutionPlan accepted_plan = compile(graph);
  ASSERT_TRUE(accepted_plan.ok()) << accepted_plan.json();
  const auto retained_plan = ExecutionPlanAccess::retain(accepted_plan);
  ASSERT_NE(retained_plan, nullptr);

  auto exec = GraphExec::reconstruct(retained_plan, kRunGeneration);
  ASSERT_TRUE(exec) << runtime_error_message(exec.error());
  auto rates = RuntimeRateState::create((*exec)->plan(), (*exec)->operators(), kRunGeneration, 1U);
  ASSERT_TRUE(rates) << runtime_error_message(rates.error());
  ASSERT_TRUE((*rates)->begin_startup_candidate());
  ASSERT_TRUE((*rates)->preflight_candidate_publication());
  ASSERT_TRUE((*rates)->publish_candidate());
  const RuntimeRateSnapshotCapture capture_rate_snapshot =
      [](const RuntimeRateState& state) noexcept { return state.capture_snapshot(); };
  auto rate_snapshot = (*rates)->capture_snapshot();
  ASSERT_TRUE(rate_snapshot) << runtime_error_message(rate_snapshot.error());
  ASSERT_NE(rate_snapshot->published(), nullptr);

  std::vector<RuntimeChannelPortReplayRecord> replay_ports;
  for (const RuntimeOperatorInstance& instance : (*exec)->operators()) {
    ASSERT_NE(instance.replay_spec, nullptr);
    for (std::size_t index = 0U;
         index < OperatorSpecCompilerAccess::port_count(*instance.replay_spec);
         ++index) {
      const auto replay = OperatorSpecCompilerAccess::port(*instance.replay_spec, index);
      const auto planned = std::ranges::find_if((*exec)->plan().ports, [&](const auto& candidate) {
        return candidate.owner == instance.vertex && candidate.direction == replay.direction &&
               candidate.local_name == replay.name;
      });
      ASSERT_NE(planned, (*exec)->plan().ports.end());
      replay_ports.push_back(RuntimeChannelPortReplayRecord{
          .port = planned->id,
          .owner = instance.vertex,
          .direction = replay.direction,
          .declaration = const_cast<void*>(replay.declaration),
          .type_ops = replay.runtime_type_ops,
      });
    }
  }

  const auto staged_edge =
      std::ranges::find_if((*exec)->plan().payload_edges, [](const auto& edge) {
        return edge.queue_policy == QueuePolicy::kDropOldest && edge.queue_depth == 1U;
      });
  const auto lossy_edge = std::ranges::find_if((*exec)->plan().payload_edges, [](const auto& edge) {
    return edge.queue_policy == QueuePolicy::kDropOldest && edge.queue_depth == 2U;
  });
  const auto blocking_edge = std::ranges::find(
      (*exec)->plan().payload_edges, QueuePolicy::kBlocking, &PayloadEdgePlanRecord::queue_policy);
  ASSERT_NE(staged_edge, (*exec)->plan().payload_edges.end());
  ASSERT_NE(lossy_edge, (*exec)->plan().payload_edges.end());
  ASSERT_NE(blocking_edge, (*exec)->plan().payload_edges.end());
  ASSERT_NE(lossy_edge->edge, blocking_edge->edge);
  ASSERT_TRUE(std::ranges::all_of(
      (*exec)->plan().process_local_channel_services,
      [](const auto& s) { return s.delivery_kind == ProcessLocalChannelDeliveryKind::kHoloMQ; }));

  auto session = RuntimeChannelSession::create(
      (*exec)->plan(), replay_ports, {}, {}, nullptr, {}, kRunGeneration);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto foreign_rates =
      RuntimeRateState::create((*exec)->plan(), (*exec)->operators(), RunGenerationId{992U}, 1U);
  ASSERT_TRUE(foreign_rates) << runtime_error_message(foreign_rates.error());
  ASSERT_TRUE((*foreign_rates)->begin_startup_candidate());
  ASSERT_TRUE((*foreign_rates)->preflight_candidate_publication());
  ASSERT_TRUE((*foreign_rates)->publish_candidate());
  auto foreign_observation = std::make_shared<RuntimeChannelQueueObservationTestControl>();
  auto foreign_bind = RuntimeChannelSessionTestAccess::bind_queue_observation(**session,
                                                                              (*exec)->plan(),
                                                                              **foreign_rates,
                                                                              capture_rate_snapshot,
                                                                              staged_edge->edge,
                                                                              foreign_observation);
  ASSERT_FALSE(foreign_bind);
  EXPECT_EQ(foreign_bind.error().code, ErrorCode::kInvalidArgument);

  auto staged_observation = std::make_shared<RuntimeChannelQueueObservationTestControl>();
  auto lossy_observation = std::make_shared<RuntimeChannelQueueObservationTestControl>();
  auto blocking_observation = std::make_shared<RuntimeChannelQueueObservationTestControl>();
  const RateGenerationId published_generation = rate_snapshot->published()->rate_generation;
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::bind_queue_observation(**session,
                                                                      (*exec)->plan(),
                                                                      **rates,
                                                                      capture_rate_snapshot,
                                                                      staged_edge->edge,
                                                                      staged_observation));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::bind_queue_observation(**session,
                                                                      (*exec)->plan(),
                                                                      **rates,
                                                                      capture_rate_snapshot,
                                                                      lossy_edge->edge,
                                                                      lossy_observation));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::bind_queue_observation(**session,
                                                                      (*exec)->plan(),
                                                                      **rates,
                                                                      capture_rate_snapshot,
                                                                      blocking_edge->edge,
                                                                      blocking_observation));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::sample_queue_observation(
      **session, **rates, capture_rate_snapshot, staged_edge->edge, staged_observation));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::sample_queue_observation(
      **session, **rates, capture_rate_snapshot, lossy_edge->edge, lossy_observation));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::sample_queue_observation(
      **session, **rates, capture_rate_snapshot, blocking_edge->edge, blocking_observation));
  EXPECT_FALSE(staged_observation->snapshot().observed_backpressure_count.has_value());
  EXPECT_FALSE(staged_observation->snapshot().observed_replacement_or_drop_count.has_value());
  EXPECT_FALSE(lossy_observation->snapshot().observed_backpressure_count.has_value());
  EXPECT_FALSE(lossy_observation->snapshot().observed_replacement_or_drop_count.has_value());
  EXPECT_FALSE(blocking_observation->snapshot().observed_backpressure_count.has_value());
  EXPECT_FALSE(blocking_observation->snapshot().observed_replacement_or_drop_count.has_value());

  // Authored Graph fan-in is forbidden, but this private seam deliberately gives the first
  // subscriber the lossy input's target. The observed receive must still use lossy_edge rather
  // than re-searching by that now-ambiguous target.
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_observation_target_alias(
      **session, staged_edge->edge, lossy_edge->target));

  const auto runtime_source = std::ranges::find_if((*exec)->operators(), [](const auto& instance) {
    return dynamic_cast<RateObservedQueueSource*>(instance.op.get()) != nullptr;
  });
  ASSERT_NE(runtime_source, (*exec)->operators().end());
  auto* const source_operator = dynamic_cast<RateObservedQueueSource*>(runtime_source->op.get());
  ASSERT_NE(source_operator, nullptr);
  const auto source_port =
      std::ranges::find((*exec)->plan().ports, lossy_edge->source, &PortPlanRecord::id);
  ASSERT_NE(source_port, (*exec)->plan().ports.end());

  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->prepare_activation(source_port->owner, 1U, {}));
  ASSERT_TRUE(source_operator->output.emit(std::int32_t{11}));
  auto lossy_first = RuntimeChannelSessionTestAccess::receive_one_observed(**session,
                                                                           **rates,
                                                                           capture_rate_snapshot,
                                                                           lossy_edge->target,
                                                                           lossy_edge->edge,
                                                                           lossy_observation);
  ASSERT_TRUE(lossy_first) << runtime_error_message(lossy_first.error());
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_observation_target_alias(
      **session, staged_edge->edge, staged_edge->target));
  auto staged_first = RuntimeChannelSessionTestAccess::receive_one_observed(**session,
                                                                            **rates,
                                                                            capture_rate_snapshot,
                                                                            staged_edge->target,
                                                                            staged_edge->edge,
                                                                            staged_observation);
  auto blocking_first = RuntimeChannelSessionTestAccess::receive_one_observed(**session,
                                                                              **rates,
                                                                              capture_rate_snapshot,
                                                                              blocking_edge->target,
                                                                              blocking_edge->edge,
                                                                              blocking_observation);
  ASSERT_TRUE(staged_first) << runtime_error_message(staged_first.error());
  ASSERT_TRUE(blocking_first) << runtime_error_message(blocking_first.error());

  ASSERT_TRUE((*session)->retire_activation(source_port->owner, 1U, true));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_single_occupied_claim_state(
      **session, staged_edge->target, true, false, false, false));
  ASSERT_TRUE((*session)->prepare_activation(source_port->owner, 2U, {}));
  ASSERT_TRUE(source_operator->output.emit(std::int32_t{22}));
  auto staged_second = RuntimeChannelSessionTestAccess::receive_one_observed(**session,
                                                                             **rates,
                                                                             capture_rate_snapshot,
                                                                             staged_edge->target,
                                                                             staged_edge->edge,
                                                                             staged_observation);
  auto lossy_second = RuntimeChannelSessionTestAccess::receive_one_observed(**session,
                                                                            **rates,
                                                                            capture_rate_snapshot,
                                                                            lossy_edge->target,
                                                                            lossy_edge->edge,
                                                                            lossy_observation);
  auto blocking_second =
      RuntimeChannelSessionTestAccess::receive_one_observed(**session,
                                                            **rates,
                                                            capture_rate_snapshot,
                                                            blocking_edge->target,
                                                            blocking_edge->edge,
                                                            blocking_observation);
  ASSERT_TRUE(staged_second) << runtime_error_message(staged_second.error());
  ASSERT_TRUE(lossy_second) << runtime_error_message(lossy_second.error());
  ASSERT_TRUE(blocking_second) << runtime_error_message(blocking_second.error());
  const auto staged_append = staged_observation->snapshot();
  EXPECT_EQ(staged_append.observed_occupancy, 1U);
  EXPECT_EQ(staged_append.observed_occupancy_high_water, 1U);
  EXPECT_EQ(staged_append.observed_backpressure_count, 0U);
  EXPECT_EQ(staged_append.observed_replacement_or_drop_count, 0U);

  ASSERT_TRUE((*session)->retire_activation(source_port->owner, 2U, true));
  ASSERT_TRUE((*session)->prepare_activation(source_port->owner, 3U, {}));
  ASSERT_TRUE(source_operator->output.emit(std::int32_t{33}));
  auto staged_replaced =
      RuntimeChannelSessionTestAccess::receive_one_observed(**session,
                                                            **rates,
                                                            capture_rate_snapshot,
                                                            staged_edge->target,
                                                            staged_edge->edge,
                                                            staged_observation);
  auto lossy_replaced = RuntimeChannelSessionTestAccess::receive_one_observed(**session,
                                                                              **rates,
                                                                              capture_rate_snapshot,
                                                                              lossy_edge->target,
                                                                              lossy_edge->edge,
                                                                              lossy_observation);
  auto blocking_backpressured =
      RuntimeChannelSessionTestAccess::receive_one_observed(**session,
                                                            **rates,
                                                            capture_rate_snapshot,
                                                            blocking_edge->target,
                                                            blocking_edge->edge,
                                                            blocking_observation);
  ASSERT_TRUE(staged_replaced) << runtime_error_message(staged_replaced.error());
  ASSERT_TRUE(lossy_replaced) << runtime_error_message(lossy_replaced.error());
  ASSERT_FALSE(blocking_backpressured);
  EXPECT_EQ(blocking_backpressured.error().code, ErrorCode::kBackpressured);

  const auto staged = staged_observation->snapshot();
  const auto lossy = lossy_observation->snapshot();
  const auto blocking = blocking_observation->snapshot();
  for (const auto& observation : {lossy, blocking}) {
    EXPECT_EQ(observation.plan_digest, accepted_plan.digest());
    EXPECT_EQ(observation.plan_generation, rate_snapshot->plan_generation());
    EXPECT_EQ(observation.run_generation, kRunGeneration);
    EXPECT_EQ(observation.rate_generation, published_generation);
    EXPECT_EQ(observation.selected_logical_depth, 2U);
    EXPECT_EQ(observation.rate_recommendation, 60U);
    EXPECT_EQ(observation.observed_occupancy, 2U);
    EXPECT_EQ(observation.observed_occupancy_high_water, 2U);
  }
  EXPECT_EQ(staged.plan_digest, accepted_plan.digest());
  EXPECT_EQ(staged.plan_generation, rate_snapshot->plan_generation());
  EXPECT_EQ(staged.run_generation, kRunGeneration);
  EXPECT_EQ(staged.rate_generation, published_generation);
  EXPECT_EQ(staged.selected_logical_depth, 1U);
  EXPECT_EQ(staged.rate_recommendation, 2U);
  EXPECT_EQ(staged.observed_occupancy, 1U);
  EXPECT_EQ(staged.observed_occupancy_high_water, 1U);
  EXPECT_EQ(staged.edge, staged_edge->edge);
  EXPECT_EQ(staged.observed_backpressure_count, 0U);
  EXPECT_EQ(staged.observed_replacement_or_drop_count, 1U);
  EXPECT_EQ(lossy.edge, lossy_edge->edge);
  EXPECT_EQ(lossy.observed_backpressure_count, 0U);
  EXPECT_EQ(lossy.observed_replacement_or_drop_count, 1U);
  EXPECT_EQ(blocking.edge, blocking_edge->edge);
  EXPECT_EQ(blocking.observed_backpressure_count, 1U);
  EXPECT_EQ(blocking.observed_replacement_or_drop_count, 0U);

  ASSERT_TRUE((*session)->retire_activation(source_port->owner, 3U, true));

  ASSERT_TRUE((*session)->close_owner_outputs(source_port->owner));
  auto terminal = RuntimeChannelSessionTestAccess::receive_one_observed(**session,
                                                                        **rates,
                                                                        capture_rate_snapshot,
                                                                        lossy_edge->target,
                                                                        lossy_edge->edge,
                                                                        lossy_observation);
  ASSERT_TRUE(terminal) << runtime_error_message(terminal.error());
  const auto after_terminal = lossy_observation->snapshot();
  EXPECT_EQ(after_terminal.observed_backpressure_count, lossy.observed_backpressure_count);
  EXPECT_EQ(after_terminal.observed_replacement_or_drop_count,
            lossy.observed_replacement_or_drop_count);

  std::vector<OperatorInstanceId> restart_members;
  restart_members.reserve((*exec)->operators().size());
  std::ranges::transform((*exec)->operators(),
                         std::back_inserter(restart_members),
                         &RuntimeOperatorInstance::operator_id);
  ASSERT_TRUE((*rates)->begin_restart_candidate(restart_members, LifecycleStage::kDiscover));
  ASSERT_TRUE((*rates)->preflight_candidate_publication());
  ASSERT_TRUE((*rates)->publish_candidate());
  auto stale = RuntimeChannelSessionTestAccess::sample_queue_observation(
      **session, **rates, capture_rate_snapshot, lossy_edge->edge, lossy_observation);
  ASSERT_FALSE(stale);
  EXPECT_EQ(stale.error().code, ErrorCode::kInvalidArgument);

  auto rebound_observation = std::make_shared<RuntimeChannelQueueObservationTestControl>();
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::bind_queue_observation(**session,
                                                                      (*exec)->plan(),
                                                                      **rates,
                                                                      capture_rate_snapshot,
                                                                      lossy_edge->edge,
                                                                      rebound_observation));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::sample_queue_observation(
      **session, **rates, capture_rate_snapshot, lossy_edge->edge, rebound_observation));
  EXPECT_NE(rebound_observation->snapshot().rate_generation, published_generation);

  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// A compiler-authorized 1:1 blocking scalar edge retains directly into the destination's bounded
// typed storage. It creates no HoloMQ endpoint, WaitSet attachment, or pump work, while preserving
// the same readiness, metadata, backpressure, and retirement behavior.
// REQ: CHAN-096
// REQ: CHAN-116
// REQ: CHAN-101
TEST(RuntimeChannelBinding, DirectRetainedScalarBypassesTransportAndPreservesBoundedSemantics) {
  SessionFixture fixture;
  fixture.use_one_slot_direct_retained_delivery();
  struct WakeProbe {
    std::binary_semaphore ready{0};
    std::vector<VertexId> direct_owners;
  } wake;
  auto session = RuntimeChannelSession::create(
      fixture.plan,
      fixture.replay,
      {},
      RuntimeChannelReadinessWake{
          .state = &wake,
          .signal = [](void* state) noexcept { static_cast<WakeProbe*>(state)->ready.release(); },
          .signal_owner_progress =
              [](void* state, VertexId owner) noexcept {
                auto* const probe = static_cast<WakeProbe*>(state);
                probe->direct_owners.push_back(owner);
                probe->ready.release();
              },
      });
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 1U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());

  // Key setup: the direct plan creates no transport drain or WaitSet index, then emits directly
  // into its sole destination slot without a post-compute transport cut.
  const auto indexes = RuntimeChannelSessionTestAccess::pump_index_state(**session);
  ASSERT_TRUE(indexes) << runtime_error_message(indexes.error());
  EXPECT_TRUE(indexes->exact);
  EXPECT_EQ(indexes->wait_handle_count, 0U);
  EXPECT_EQ(indexes->drain_subscriber_count, 0U);
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{42}));
  ASSERT_TRUE(wake.ready.try_acquire());
  const auto backpressured = fixture.source.output.emit(std::int32_t{43});
  ASSERT_FALSE(backpressured);
  EXPECT_EQ(backpressured.error().code, ErrorCode::kBackpressured);
  EXPECT_FALSE(wake.ready.try_acquire());

  auto event = wait_for_event(**session, **binding);
  ASSERT_TRUE(event) << runtime_error_message(event.error());
  auto temporal = (*session)->temporal_sample(event->port, event->source_sequence);
  ASSERT_TRUE(temporal) << runtime_error_message(temporal.error());
  const std::array selected{*temporal};
  const std::array selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = selected,
      .batch = false,
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selection));
  {
    auto received = fixture.sink.input.receive();
    ASSERT_TRUE(received) << runtime_error_message(received.error());

    // Key assertion: typed data and semantic metadata survive the bypass. The owning sample's
    // read claim ends at this scope before the finite slot is expected to become reusable.
    EXPECT_EQ(received->data, 42);
    EXPECT_EQ(received->metadata.sequence, 1U);
  }
  ASSERT_TRUE((*binding)->retire(event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  EXPECT_FALSE(wake.ready.try_acquire());
  ASSERT_TRUE((*session)->retire_samples(PortId{2U}, selected));
  ASSERT_TRUE(wake.ready.try_acquire());
  EXPECT_EQ(wake.direct_owners, (std::vector<VertexId>{VertexId{2U}, VertexId{1U}}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{43}));
  ASSERT_TRUE(wake.ready.try_acquire());
  EXPECT_EQ(wake.direct_owners, (std::vector<VertexId>{VertexId{2U}, VertexId{1U}, VertexId{2U}}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// Direct kDropOldest retains the destination's authored typed capacity. A merely full queue
// replaces its oldest eligible sample synchronously, preserves forward sequence gaps among the
// surviving samples, and emits exactly one readiness witness per survivor.
// REQ: CHAN-116
// REQ: RUNTIME-019
TEST(RuntimeChannelBinding, DirectDropOldestScalarReplacesOldestWithoutDoubleCountingReadiness) {
  SessionFixture fixture;
  fixture.use_two_slot_drop_oldest_queue();
  fixture.use_direct_retained_delivery();
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 2U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

  constexpr std::array values{11, 22, 33};
  constexpr std::array<std::int64_t, values.size()> capture_times{10, 20, 30};
  for (std::size_t index = 0U; index < values.size(); ++index) {
    const SampleMetadata metadata{
        .capture_timestamp_ns = capture_times[index],
        .source_clock_id = ClockId{1U},
    };
    ASSERT_TRUE(fixture.source.output.emit(Sample<std::int32_t>{values[index], metadata}));
  }
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  const auto retained = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(retained) << runtime_error_message(retained.error());
  EXPECT_EQ(retained->occupied_samples, 2U);
  EXPECT_EQ(retained->next_arrival_sequence, 4U);

  std::array<RuntimeEvent, 2U> events{};
  std::array<TemporalSampleRecord, 2U> records{};
  for (std::size_t index = 0U; index < events.size(); ++index) {
    auto event = wait_for_event(**session, **binding);
    ASSERT_TRUE(event) << runtime_error_message(event.error());
    events[index] = *event;
    auto record = (*session)->temporal_sample(event->port, event->source_sequence);
    ASSERT_TRUE(record) << runtime_error_message(record.error());
    records[index] = *record;
  }
  EXPECT_EQ(records[0].source_sequence, 2U);
  EXPECT_EQ(records[1].source_sequence, 3U);

  ASSERT_TRUE((*binding)->poll(TimePoint{100, ClockId{1U}}));
  RuntimeEvent duplicate;
  auto duplicate_count = (*binding)->drain(std::span<RuntimeEvent>{&duplicate, 1U});
  ASSERT_TRUE(duplicate_count) << runtime_error_message(duplicate_count.error());
  EXPECT_EQ(*duplicate_count, 0U);

  const std::array selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = records,
      .batch = true,
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selection));
  auto batch = fixture.sink.input.selected_batch();
  ASSERT_TRUE(batch) << runtime_error_message(batch.error());
  EXPECT_EQ(batch_values(*batch), (std::vector<std::int32_t>{22, 33}));
  for (const RuntimeEvent& event : events) {
    ASSERT_TRUE((*binding)->retire(event.source_sequence));
  }
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  ASSERT_TRUE((*session)->retire_samples(PortId{2U}, records));

  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// A protected typed slot remains immutable, but it no longer turns the independent drop-oldest
// mailbox into a blocking queue. The mailbox accepts one pending survivor and replaces that
// survivor without consuming another typed-retention position.
// REQ: CHAN-116
// REQ: RUNTIME-019
TEST(RuntimeChannelBinding,
     DirectDropOldestScalarKeepsProtectedTypedSlotWhileIngressRemainsWritable) {
  struct ProtectionCase {
    std::string_view label;
    bool pre_witness_replacement_allowed;
    bool active_selected;
    bool reuse_retained;
    bool temporal_retained;
    bool retirement_selected;
  };
  constexpr std::array cases{
      ProtectionCase{"pre-witness disabled", false, false, false, false, false},
      ProtectionCase{"active selection", true, true, false, false, false},
      ProtectionCase{"reuse retention", true, false, true, false, false},
      ProtectionCase{"temporal retention", true, false, false, true, false},
      ProtectionCase{"retirement selection", true, false, false, false, true},
  };

  for (const ProtectionCase& test_case : cases) {
    SCOPED_TRACE(test_case.label);
    SessionFixture fixture;
    fixture.use_one_slot_drop_oldest_queue();
    fixture.use_direct_retained_delivery();
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_pre_witness_replacement_allowed(
        **session, PortId{2U}, test_case.pre_witness_replacement_allowed));
    ASSERT_TRUE((*session)->refresh_connections());
    ASSERT_TRUE((*session)->arm());
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
    ASSERT_TRUE(fixture.source.output.emit(std::int32_t{11}));
    if (test_case.active_selected || test_case.reuse_retained || test_case.temporal_retained ||
        test_case.retirement_selected) {
      ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_single_occupied_claim_state(
          **session,
          PortId{2U},
          test_case.active_selected,
          test_case.reuse_retained,
          test_case.temporal_retained,
          test_case.retirement_selected));
    }

    ASSERT_TRUE(fixture.source.output.emit(std::int32_t{22}));
    const auto retained = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    ASSERT_TRUE(retained) << runtime_error_message(retained.error());
    EXPECT_EQ(retained->occupied_samples, 1U);
    EXPECT_EQ(retained->pending_direct_ingress_samples, 1U);
    // Pending ingress has not entered typed Runtime retention, but it already owns the stable
    // arrival identity that must survive an out-of-order freshness promotion.
    EXPECT_EQ(retained->next_arrival_sequence, 3U);

    // Key assertion: capacity-one kDropOldest replaces the pending unread survivor while the
    // independently claimed typed sample remains in place. Replacing it consumes another stable
    // staging identity, leaving an intentional gap rather than making later out-of-order
    // promotions expose a decreasing arrival identity.
    ASSERT_TRUE(fixture.source.output.emit(std::int32_t{33}));
    const auto replaced = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    ASSERT_TRUE(replaced) << runtime_error_message(replaced.error());
    EXPECT_EQ(replaced->occupied_samples, 1U);
    EXPECT_EQ(replaced->pending_direct_ingress_samples, 1U);
    EXPECT_EQ(replaced->next_arrival_sequence, 4U);
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
    (*session)->cancel();
    EXPECT_TRUE((*session)->join());
  }
}

// A consumed ReuseLast fallback is private temporal retention, not unread delivery capacity.
// A depth-one DropOldest input must admit one typed successor beside that fallback.
// REQ: CHAN-146
// REQ: CHAN-131
TEST(RuntimeChannelBinding, DirectDropOldestAdmitsTypedSampleBesideReuseLastFallback) {
  SessionFixture fixture;
  fixture.use_one_slot_drop_oldest_queue();
  fixture.use_direct_retained_delivery();
  fixture.enable_reuse_last();
  fixture.plan.input_selections.front().selector = InputSelectorKind::kLatestWithin;
  fixture.plan.input_selections.front().max_age_ns = 100;
  fixture.plan.input_selections.front().cache_capacity = 1U;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{11}));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_single_occupied_claim_state(
      **session, PortId{2U}, false, true, true, false));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{22}));

  const auto state = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(state) << runtime_error_message(state.error());
  EXPECT_EQ(state->occupied_samples, 2U);
  EXPECT_EQ(state->pending_direct_ingress_samples, 0U);
  EXPECT_TRUE(state->reuse_retained);
  EXPECT_TRUE(state->temporal_retained);
  EXPECT_EQ(state->next_arrival_sequence, 3U);

  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// Direct retention and HoloMQ must keep the same newest survivor when a selected typed sample
// remains protected while the producer continues publishing.
// REQ: CHAN-116
// REQ: RUNTIME-019
TEST(RuntimeChannelBinding, DropOldestKeepsSameSurvivorWhileTypedSlotIsProtected) {
  for (const bool direct_retained : {false, true}) {
    SCOPED_TRACE(direct_retained ? "direct retained" : "HoloMQ");
    SessionFixture fixture;
    fixture.use_one_slot_drop_oldest_queue();
    if (direct_retained) {
      fixture.use_direct_retained_delivery();
    }
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    auto binding =
        (*session)->create_readiness_binding(VertexId{2U},
                                             RuntimePortBindingRequest{
                                                 .plan_generation = fixture.plan.plan_generation,
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
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
    ASSERT_TRUE(fixture.source.output.emit(std::int32_t{11}));

    auto first_event = wait_for_event(**session, **binding);
    ASSERT_TRUE(first_event) << runtime_error_message(first_event.error());
    auto first_record =
        (*session)->temporal_sample(first_event->port, first_event->source_sequence);
    ASSERT_TRUE(first_record) << runtime_error_message(first_record.error());
    const std::array first_selection{RuntimeChannelActivationSelection{
        .port = PortId{2U},
        .selection = {InputState::kSelected, SelectionReason::kNone},
        .samples = std::span<const TemporalSampleRecord>{&*first_record, 1U},
        .destructive = true,
    }};
    ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, first_selection));

    ASSERT_TRUE(fixture.source.output.emit(std::int32_t{22}));
    ASSERT_TRUE(fixture.source.output.emit(std::int32_t{33}));

    ASSERT_TRUE((*binding)->retire(first_event->source_sequence));
    ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
    ASSERT_TRUE((*session)->retire_samples(
        PortId{2U}, std::span<const TemporalSampleRecord>{&*first_record, 1U}));

    auto survivor_event = wait_for_event(**session, **binding);
    ASSERT_TRUE(survivor_event) << runtime_error_message(survivor_event.error());
    auto survivor_record =
        (*session)->temporal_sample(survivor_event->port, survivor_event->source_sequence);
    ASSERT_TRUE(survivor_record) << runtime_error_message(survivor_record.error());
    EXPECT_EQ(survivor_record->source_sequence, 3U);
    // Direct ingress assigned identities to both staged publications before replacing the older
    // one. HoloMQ may replace the older publication in its transport queue before it reaches
    // Runtime staging, so only the direct path necessarily exposes this internal gap.
    EXPECT_EQ(survivor_record->arrival_sequence, direct_retained ? 3U : 2U);
    const std::array survivor_selection{RuntimeChannelActivationSelection{
        .port = PortId{2U},
        .selection = {InputState::kSelected, SelectionReason::kNone},
        .samples = std::span<const TemporalSampleRecord>{&*survivor_record, 1U},
        .destructive = true,
    }};
    ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 2U, survivor_selection));
    auto survivor = fixture.sink.input.receive();
    ASSERT_TRUE(survivor) << runtime_error_message(survivor.error());
    EXPECT_EQ(survivor->data, 33);
    EXPECT_EQ(survivor->metadata.sequence, 3U);
    ASSERT_TRUE((*binding)->retire(survivor_event->source_sequence));
    ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 2U, true));
    ASSERT_TRUE((*session)->retire_samples(
        PortId{2U}, std::span<const TemporalSampleRecord>{&*survivor_record, 1U}));
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
    (*session)->cancel();
    EXPECT_TRUE((*session)->join());
  }
}

// A later receive can enter typed storage while an earlier survivor remains staged. Promotion
// merges both populations and retains the newest survivor instead of letting staged data regress
// the drop-oldest mailbox.
// REQ: CHAN-128
TEST(RuntimeChannelBinding, DropOldestPromotionKeepsNewerTypedSampleWhenStagedArrivalIsOvertaken) {
  SessionFixture fixture;
  fixture.use_one_slot_drop_oldest_queue();
  fixture.use_direct_retained_delivery();
  const auto counters = std::make_shared<RuntimePhaseCounters>();
  fixture.plan.executable_factories.phase_counters = counters;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 1U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());

  const auto inject = [&](const std::int32_t value, const std::uint64_t sequence) {
    return RuntimeChannelSessionTestAccess::inject_direct_inline_payload(
        **session,
        PortId{2U},
        std::as_bytes(std::span{&value, 1U}),
        SampleMetadata{
            .sequence = sequence,
            .byte_size = sizeof(value),
            .payload_kind = PayloadKind::kInlineTyped,
        });
  };
  ASSERT_TRUE(inject(11, 1U));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_single_occupied_claim_state(
      **session, PortId{2U}, false, false, true, false));
  ASSERT_TRUE(inject(22, 2U));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::make_staged_sample_older_than_typed_sample(
      **session, PortId{2U}));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_single_occupied_claim_state(
      **session, PortId{2U}, false, false, false, false));

  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*binding)->poll(TimePoint{1, ClockId{1U}}));
  RuntimeEvent event;
  auto drained = (*binding)->drain(std::span<RuntimeEvent>{&event, 1U});
  ASSERT_TRUE(drained) << runtime_error_message(drained.error());
  ASSERT_EQ(*drained, 1U);
  const auto merged = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(merged) << runtime_error_message(merged.error());
  EXPECT_EQ(merged->occupied_samples, 1U);
  EXPECT_EQ(merged->pending_direct_ingress_samples, 0U);
  RuntimeEvent unexpected_event;
  auto unexpected = (*binding)->drain(std::span<RuntimeEvent>{&unexpected_event, 1U});
  ASSERT_TRUE(unexpected) << runtime_error_message(unexpected.error());
  EXPECT_EQ(*unexpected, 0U);
  const auto phase = counters->snapshot();
  EXPECT_EQ(phase[RuntimePhaseCounters::index(RuntimePhaseCounter::kChannelSamplesReceived)], 1U);
  EXPECT_EQ(phase[RuntimePhaseCounters::index(RuntimePhaseCounter::kChannelPreWitnessReplacements)],
            0U);
  auto record = (*session)->temporal_sample(event.port, event.source_sequence);
  ASSERT_TRUE(record) << runtime_error_message(record.error());
  EXPECT_EQ(record->source_sequence, 2U);
  const std::array selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&*record, 1U},
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selection));
  auto received = fixture.sink.input.receive();
  ASSERT_TRUE(received) << runtime_error_message(received.error());
  EXPECT_EQ(received->data, 22);
  EXPECT_EQ(received->metadata.sequence, 2U);
  ASSERT_TRUE((*binding)->retire(event.source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  ASSERT_TRUE((*session)->retire_samples(PortId{2U}, selection.front().samples));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// A staged survivor may still carry its exact typed-ingress reservation after a later sample
// overtakes it in typed storage. Promotion must defer destruction of that live reservation, and
// the production commit path's retry must retain only the newer sample.
// REQ: CHAN-128
TEST(RuntimeChannelBinding, DropOldestStagedReservationDefersToNewerTypedOvertake) {
  SessionFixture fixture;
  fixture.use_one_slot_drop_oldest_queue();
  fixture.use_direct_retained_delivery();
  const auto counters = std::make_shared<RuntimePhaseCounters>();
  fixture.plan.executable_factories.phase_counters = counters;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());

  const auto inject = [&](const std::int32_t value, const std::uint64_t sequence) {
    return RuntimeChannelSessionTestAccess::inject_direct_inline_payload(
        **session,
        PortId{2U},
        std::as_bytes(std::span{&value, 1U}),
        SampleMetadata{
            .sequence = sequence,
            .byte_size = sizeof(value),
            .payload_kind = PayloadKind::kInlineTyped,
        });
  };
  ASSERT_TRUE(inject(11, 1U));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_single_occupied_claim_state(
      **session, PortId{2U}, false, false, true, false));
  ASSERT_TRUE(inject(22, 2U));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::make_staged_sample_older_than_typed_sample(
      **session, PortId{2U}));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_single_occupied_claim_state(
      **session, PortId{2U}, false, false, false, false));

  ASSERT_TRUE(RuntimeChannelSessionTestAccess::commit_staged_reservation_after_newer_typed_overtake(
      **session, PortId{2U}));
  const auto released =
      RuntimeChannelSessionTestAccess::typed_ingress_reservations(**session, PortId{2U});
  ASSERT_TRUE(released) << runtime_error_message(released.error());
  EXPECT_EQ(*released, 0U);
  const auto retained = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(retained) << runtime_error_message(retained.error());
  EXPECT_EQ(retained->occupied_samples, 1U);
  EXPECT_EQ(retained->pending_direct_ingress_samples, 0U);

  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 1U,
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
  RuntimeEvent unexpected_event;
  auto unexpected = (*binding)->drain(std::span<RuntimeEvent>{&unexpected_event, 1U});
  ASSERT_TRUE(unexpected) << runtime_error_message(unexpected.error());
  EXPECT_EQ(*unexpected, 0U);
  const auto phase = counters->snapshot();
  EXPECT_EQ(phase[RuntimePhaseCounters::index(RuntimePhaseCounter::kChannelSamplesReceived)], 1U);
  EXPECT_EQ(phase[RuntimePhaseCounters::index(RuntimePhaseCounter::kChannelPreWitnessReplacements)],
            0U);
  EXPECT_EQ(phase[RuntimePhaseCounters::index(
                RuntimePhaseCounter::kChannelReleasedClaimRefillReplacementDeclined)],
            0U);
  const std::array selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&*record, 1U},
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selection));
  auto received = fixture.sink.input.receive();
  ASSERT_TRUE(received) << runtime_error_message(received.error());
  EXPECT_EQ(received->data, 22);
  EXPECT_EQ(received->metadata.sequence, 2U);
  ASSERT_TRUE((*binding)->retire(event.source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  ASSERT_TRUE((*session)->retire_samples(PortId{2U}, selection.front().samples));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// A committed freshness target is an ordering fence, so a newer unclaimed typed row cannot
// discard it during the same staged-reservation commit retry.
// REQ: CHAN-128
// REQ: RUNTIME-043
TEST(RuntimeChannelBinding, CommittedRefillTargetWinsNewerTypedOvertake) {
  SessionFixture fixture;
  fixture.use_one_slot_drop_oldest_queue();
  fixture.use_direct_retained_delivery();
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());

  const auto inject = [&](const std::int32_t value, const std::uint64_t sequence) {
    return RuntimeChannelSessionTestAccess::inject_direct_inline_payload(
        **session,
        PortId{2U},
        std::as_bytes(std::span{&value, 1U}),
        SampleMetadata{
            .sequence = sequence,
            .byte_size = sizeof(value),
            .payload_kind = PayloadKind::kInlineTyped,
        });
  };
  ASSERT_TRUE(inject(11, 1U));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_single_occupied_claim_state(
      **session, PortId{2U}, false, false, true, false));
  ASSERT_TRUE(inject(22, 2U));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::make_staged_sample_older_than_typed_sample(
      **session, PortId{2U}));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_single_occupied_claim_state(
      **session, PortId{2U}, false, false, false, false));

  const auto counters = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(counters) << runtime_error_message(counters.error());
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_arrival_counters(
      **session,
      PortId{2U},
      counters->next_arrival_sequence,
      std::numeric_limits<std::uint64_t>::max()));
  auto exhausted =
      RuntimeChannelSessionTestAccess::commit_staged_reservation_after_newer_typed_overtake(
          **session, PortId{2U}, true);
  ASSERT_FALSE(exhausted);
  EXPECT_EQ(exhausted.error().code, ErrorCode::kResourceExhausted);
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_arrival_counters(
      **session, PortId{2U}, counters->next_arrival_sequence, counters->next_global_arrival_order));

  ASSERT_TRUE(RuntimeChannelSessionTestAccess::commit_staged_reservation_after_newer_typed_overtake(
      **session, PortId{2U}, true));
  const auto retained = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(retained) << runtime_error_message(retained.error());
  EXPECT_EQ(retained->occupied_samples, 1U);
  EXPECT_EQ(retained->pending_direct_ingress_samples, 0U);

  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 1U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*binding)->poll(TimePoint{1, ClockId{1U}}));
  RuntimeEvent event;
  auto drained = (*binding)->drain(std::span<RuntimeEvent>{&event, 1U});
  ASSERT_TRUE(drained) << runtime_error_message(drained.error());
  ASSERT_EQ(*drained, 1U);
  auto record = (*session)->temporal_sample(event.port, event.source_sequence);
  ASSERT_TRUE(record) << runtime_error_message(record.error());
  EXPECT_EQ(record->source_sequence, 1U);
  const std::array selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&*record, 1U},
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selection));
  auto received = fixture.sink.input.receive();
  ASSERT_TRUE(received) << runtime_error_message(received.error());
  EXPECT_EQ(received->data, 11);
  EXPECT_EQ(received->metadata.sequence, 1U);
  ASSERT_TRUE((*binding)->retire(event.source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  ASSERT_TRUE((*session)->retire_samples(PortId{2U}, selection.front().samples));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// Direct retention and HoloMQ must preserve the same newest-N unread backlog while every typed
// slot is protected by one active batch.
// REQ: CHAN-115
// REQ: CHAN-116
// REQ: RUNTIME-019
// REQ: CHAN-128
TEST(RuntimeChannelBinding, DropOldestKeepsSameBacklogWhileTypedSlotsAreProtected) {
  for (const bool direct_retained : {false, true}) {
    SCOPED_TRACE(direct_retained ? "direct retained" : "HoloMQ");
    SessionFixture fixture;
    fixture.use_two_slot_drop_oldest_queue();
    if (direct_retained) {
      fixture.use_direct_retained_delivery();
    }
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    auto binding =
        (*session)->create_readiness_binding(VertexId{2U},
                                             RuntimePortBindingRequest{
                                                 .plan_generation = fixture.plan.plan_generation,
                                                 .run_generation = 1U,
                                                 .kind = RuntimeEventKind::kInputArrival,
                                                 .source = make_source_id(1U),
                                                 .source_clock = ClockId{1U},
                                                 .reference_clock = ClockId{1U},
                                                 .capacity = 2U,
                                             });
    ASSERT_TRUE(binding) << runtime_error_message(binding.error());
    ASSERT_TRUE((*session)->refresh_connections());
    ASSERT_TRUE((*session)->arm());
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
    ASSERT_TRUE(fixture.source.output.emit(std::int32_t{11}));
    ASSERT_TRUE(fixture.source.output.emit(std::int32_t{12}));

    std::array<RuntimeEvent, 2U> retained_events{};
    std::array<TemporalSampleRecord, 2U> retained_records{};
    for (std::size_t index = 0U; index < retained_events.size(); ++index) {
      auto event = wait_for_event(**session, **binding);
      ASSERT_TRUE(event) << runtime_error_message(event.error());
      retained_events[index] = *event;
      auto record = (*session)->temporal_sample(event->port, event->source_sequence);
      ASSERT_TRUE(record) << runtime_error_message(record.error());
      retained_records[index] = *record;
    }
    const std::array retained_selection{RuntimeChannelActivationSelection{
        .port = PortId{2U},
        .selection = {InputState::kSelected, SelectionReason::kNone},
        .samples = retained_records,
        .batch = true,
        .destructive = true,
    }};
    ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, retained_selection));

    ASSERT_TRUE(fixture.source.output.emit(std::int32_t{21}));
    ASSERT_TRUE(fixture.source.output.emit(std::int32_t{22}));
    ASSERT_TRUE(fixture.source.output.emit(std::int32_t{23}));

    for (const RuntimeEvent& event : retained_events) {
      ASSERT_TRUE((*binding)->retire(event.source_sequence));
    }
    ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
    ASSERT_TRUE((*session)->retire_samples(PortId{2U}, retained_records));
    if (direct_retained) {
      const auto promoted =
          RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
      ASSERT_TRUE(promoted) << runtime_error_message(promoted.error());
      EXPECT_EQ(promoted->occupied_samples, 2U);
      EXPECT_EQ(promoted->pending_direct_ingress_samples, 0U);
    }

    std::array<RuntimeEvent, 2U> survivor_events{};
    std::array<TemporalSampleRecord, 2U> survivor_records{};
    for (std::size_t index = 0U; index < survivor_events.size(); ++index) {
      SCOPED_TRACE(index);
      auto event = wait_for_event(**session, **binding);
      ASSERT_TRUE(event) << runtime_error_message(event.error());
      survivor_events[index] = *event;
      auto record = (*session)->temporal_sample(event->port, event->source_sequence);
      ASSERT_TRUE(record) << runtime_error_message(record.error());
      survivor_records[index] = *record;
    }
    EXPECT_EQ(survivor_records[0].source_sequence, 4U);
    EXPECT_EQ(survivor_records[1].source_sequence, 5U);
    const std::array survivor_selection{RuntimeChannelActivationSelection{
        .port = PortId{2U},
        .selection = {InputState::kSelected, SelectionReason::kNone},
        .samples = survivor_records,
        .batch = true,
        .destructive = true,
    }};
    ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 2U, survivor_selection));
    auto survivors = fixture.sink.input.selected_batch();
    ASSERT_TRUE(survivors) << runtime_error_message(survivors.error());
    EXPECT_EQ(batch_values(*survivors), (std::vector<std::int32_t>{22, 23}));
    for (const RuntimeEvent& event : survivor_events) {
      ASSERT_TRUE((*binding)->retire(event.source_sequence));
    }
    ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 2U, true));
    ASSERT_TRUE((*session)->retire_samples(PortId{2U}, survivor_records));
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
    (*session)->cancel();
    EXPECT_TRUE((*session)->join());
  }
}

// EOS is the immediate successor operation: an empty generation starts at one, and a retained
// data sequence permits only predecessor-plus-one terminal authority.
// REQ: CHAN-136
TEST(RuntimeChannelBinding, EndOfStreamRejectsAProducerSequenceGapWithoutMutation) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());

  auto gap = RuntimeChannelSessionTestAccess::store_end_of_stream(**session, PortId{2U}, 2U);
  ASSERT_FALSE(gap);
  EXPECT_EQ(gap.error().code, ErrorCode::kProtocolError);
  auto unchanged = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(unchanged) << runtime_error_message(unchanged.error());
  EXPECT_FALSE(unchanged->end_of_stream_received);

  ASSERT_TRUE(RuntimeChannelSessionTestAccess::store_end_of_stream(**session, PortId{2U}, 1U));
}

// Terminal selector retirement may release ReuseLast only after canonical EOS, and a malformed
// partial batch must not release any member of the all-or-nothing terminal claim set.
// REQ: CHAN-136
TEST(RuntimeChannelBinding,
     TerminalSelectorHistoryRetirementAtomicallyReleasesReuseLastAfterEndOfStream) {
  SessionFixture fixture;
  fixture.enable_reuse_last();
  fixture.use_direct_retained_delivery();
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 3U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());

  const auto inject = [&](const std::int32_t value, const std::uint64_t sequence) {
    return RuntimeChannelSessionTestAccess::inject_direct_inline_payload(
        **session,
        PortId{2U},
        std::as_bytes(std::span{&value, 1U}),
        SampleMetadata{.sequence = sequence,
                       .byte_size = sizeof(value),
                       .payload_kind = PayloadKind::kInlineTyped});
  };
  ASSERT_TRUE(inject(11, 1U));
  ASSERT_TRUE(inject(22, 2U));

  std::array<RuntimeEvent, 2U> events{};
  std::array<TemporalSampleRecord, 2U> records{};
  for (std::size_t index = 0U; index < records.size(); ++index) {
    auto event = wait_for_event(**session, **binding);
    ASSERT_TRUE(event) << runtime_error_message(event.error());
    events[index] = *event;
    auto record = (*session)->temporal_sample(event->port, event->source_sequence);
    ASSERT_TRUE(record) << runtime_error_message(record.error());
    records[index] = *record;
  }
  const std::array selected{records.front()};
  const std::array selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = selected,
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selection));
  for (const RuntimeEvent& event : events) {
    ASSERT_TRUE((*binding)->retire(event.source_sequence));
  }
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));

  auto premature =
      (*session)->retire_terminal_selector_history_samples(VertexId{2U}, PortId{2U}, records);
  ASSERT_FALSE(premature);
  EXPECT_EQ(premature.error().code, ErrorCode::kBackpressured);
  auto ordinary = (*session)->retire_samples(PortId{2U}, records);
  ASSERT_FALSE(ordinary);
  EXPECT_EQ(ordinary.error().code, ErrorCode::kBackpressured);
  auto unchanged = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(unchanged) << runtime_error_message(unchanged.error());
  EXPECT_EQ(unchanged->occupied_samples, 2U);

  ASSERT_TRUE(RuntimeChannelSessionTestAccess::store_end_of_stream(**session, PortId{2U}, 3U));
  auto terminal = wait_for_event(**session, **binding);
  ASSERT_TRUE(terminal) << runtime_error_message(terminal.error());
  EXPECT_EQ(terminal->kind, RuntimeEventKind::kInputEndOfStream);
  ASSERT_TRUE((*binding)->retire(terminal->source_sequence));

  auto partial =
      (*session)->retire_terminal_selector_history_samples(VertexId{2U}, PortId{2U}, selected);
  ASSERT_FALSE(partial);
  EXPECT_EQ(partial.error().code, ErrorCode::kProtocolError);
  unchanged = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(unchanged) << runtime_error_message(unchanged.error());
  EXPECT_EQ(unchanged->occupied_samples, 2U);

  ASSERT_TRUE(
      (*session)->retire_terminal_selector_history_samples(VertexId{2U}, PortId{2U}, records));
  auto released = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(released) << runtime_error_message(released.error());
  EXPECT_EQ(released->occupied_samples, 0U);
  EXPECT_EQ(released->uncleared_sample_slots, 0U);
  auto generation_retired = (*session)->retire_terminal_input_generation(VertexId{2U});
  ASSERT_TRUE(generation_retired) << runtime_error_message(generation_retired.error());
  EXPECT_TRUE(*generation_retired);
}

// Direct replacement accepts forward producer-sequence jumps, rejects repeats/regressions, and
// keeps ordered EOS in separate terminal state after the last retained replacement.
// REQ: CHAN-078
// REQ: CHAN-116
// REQ: CHAN-132
TEST(RuntimeChannelBinding, DirectDropOldestScalarPreservesSequenceAndEndOfStreamRules) {
  SessionFixture fixture;
  fixture.use_one_slot_drop_oldest_queue();
  fixture.use_direct_retained_delivery();
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());

  const auto inject = [&](const std::int32_t value, const std::uint64_t sequence) {
    const SampleMetadata metadata{
        .sequence = sequence,
        .byte_size = sizeof(value),
        .payload_kind = PayloadKind::kInlineTyped,
    };
    return RuntimeChannelSessionTestAccess::inject_direct_inline_payload(
        **session, PortId{2U}, std::as_bytes(std::span{&value, 1U}), metadata);
  };

  ASSERT_TRUE(inject(11, 1U));
  ASSERT_TRUE(inject(33, 3U));
  auto repeated = inject(44, 3U);
  ASSERT_FALSE(repeated);
  EXPECT_EQ(repeated.error().code, ErrorCode::kProtocolError);
  auto regressed = inject(22, 2U);
  ASSERT_FALSE(regressed);
  EXPECT_EQ(regressed.error().code, ErrorCode::kProtocolError);

  auto terminal_gap =
      RuntimeChannelSessionTestAccess::store_end_of_stream(**session, PortId{2U}, 5U);
  ASSERT_FALSE(terminal_gap);
  EXPECT_EQ(terminal_gap.error().code, ErrorCode::kProtocolError);
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::store_end_of_stream(**session, PortId{2U}, 4U));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::store_end_of_stream(**session, PortId{2U}, 4U));
  auto conflicting =
      RuntimeChannelSessionTestAccess::store_end_of_stream(**session, PortId{2U}, 5U);
  ASSERT_FALSE(conflicting);
  EXPECT_EQ(conflicting.error().code, ErrorCode::kProtocolError);
  const auto retained = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(retained) << runtime_error_message(retained.error());
  EXPECT_EQ(retained->occupied_samples, 1U);
  EXPECT_EQ(retained->next_arrival_sequence, 3U);
  EXPECT_TRUE(retained->end_of_stream_received);
}

// A transport close follows the last producer sequence without displacing its retained data.
// In particular, once the sole typed slot holds data, the terminal head must still be classified
// and recorded; neither borrowed nor hook-capable copy fallback may turn EOS into a data-capacity
// retry. Legacy copy-only endpoints conservatively backpressure until they implement the bounded
// operation-classification hook.
// REQ: CHAN-136
TEST(RuntimeChannelBinding, InlineTransportCloseRetainsDataThenPublishesOneEndOfStream) {
  constexpr std::int32_t kValue = 37;
  constexpr std::uint64_t kDataSequence = 41U;
  for (const InlineReceivePath path :
       {InlineReceivePath::kBorrowed, InlineReceivePath::kCopyFallback}) {
    SCOPED_TRACE(path == InlineReceivePath::kBorrowed ? "borrowed" : "copy fallback");
    SessionFixture fixture;
    fixture.use_one_slot_blocking_transport();
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    auto replacement = std::make_unique<ScriptedInlineSubscriber>(kValue, kDataSequence, path);
    auto* const observed = replacement.get();
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_input_subscriber(
        **session, PortId{2U}, std::move(replacement)));
    auto binding =
        (*session)->create_readiness_binding(VertexId{2U},
                                             RuntimePortBindingRequest{
                                                 .plan_generation = fixture.plan.plan_generation,
                                                 .run_generation = 1U,
                                                 .kind = RuntimeEventKind::kInputArrival,
                                                 .source = make_source_id(1U),
                                                 .source_clock = ClockId{1U},
                                                 .reference_clock = ClockId{1U},
                                                 .capacity = 2U,
                                             });
    ASSERT_TRUE(binding) << runtime_error_message(binding.error());

    // The first receive occupies the only typed destination. The next receive observes the
    // queued terminal without needing a data reservation, and the third proves idempotence.
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
    const auto retained = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    ASSERT_TRUE(retained) << runtime_error_message(retained.error());
    EXPECT_EQ(retained->occupied_samples, 1U);
    EXPECT_EQ(retained->next_arrival_sequence, 2U);
    EXPECT_TRUE(retained->end_of_stream_received);

    // The first kClosed stored exactly predecessor+1; repeating kClosed did not advance it.
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::store_end_of_stream(
        **session, PortId{2U}, kDataSequence + 1U));
    auto conflicting = RuntimeChannelSessionTestAccess::store_end_of_stream(
        **session, PortId{2U}, kDataSequence + 2U);
    ASSERT_FALSE(conflicting);
    EXPECT_EQ(conflicting.error().code, ErrorCode::kProtocolError);

    EXPECT_EQ(observed->borrowed_receive_count, 1U);
    EXPECT_EQ(observed->copy_receive_count, path == InlineReceivePath::kCopyFallback ? 1U : 0U);
    EXPECT_EQ(observed->loan_release_count, path == InlineReceivePath::kBorrowed ? 1U : 0U);
    EXPECT_EQ(observed->staged_data_query_count, 0U);

    ASSERT_TRUE((*binding)->poll(TimePoint{1, ClockId{1U}}));
    std::array<RuntimeEvent, 3U> events{};
    auto drained = (*binding)->drain(events);
    ASSERT_TRUE(drained) << runtime_error_message(drained.error());
    ASSERT_EQ(*drained, 2U);
    EXPECT_EQ(events[0].kind, RuntimeEventKind::kInputArrival);
    EXPECT_EQ(events[1].kind, RuntimeEventKind::kInputEndOfStream);
    EXPECT_EQ(events[0].source_sequence, 1U);
    EXPECT_EQ(events[1].source_sequence, 2U);

    auto record = (*session)->temporal_sample(events[0].port, events[0].source_sequence);
    ASSERT_TRUE(record) << runtime_error_message(record.error());
    EXPECT_EQ(record->source_sequence, kDataSequence);
    const std::array selection{RuntimeChannelActivationSelection{
        .port = PortId{2U},
        .selection = {InputState::kSelected, SelectionReason::kNone},
        .samples = std::span<const TemporalSampleRecord>{&*record, 1U},
        .destructive = true,
    }};
    ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selection));
    auto received = fixture.sink.input.receive();
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    EXPECT_EQ(received->data, kValue);
    EXPECT_EQ(received->metadata.sequence, kDataSequence);
    ASSERT_TRUE((*binding)->retire(events[0].source_sequence));
    ASSERT_TRUE((*binding)->retire(events[1].source_sequence));
    ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
    ASSERT_TRUE((*session)->retire_samples(PortId{2U},
                                           std::span<const TemporalSampleRecord>{&*record, 1U}));

    EXPECT_FALSE((*binding)->poll(TimePoint{2, ClockId{1U}}));
    auto repeated = (*binding)->drain(events);
    ASSERT_TRUE(repeated) << runtime_error_message(repeated.error());
    EXPECT_EQ(*repeated, 0U);
  }
}

// Graph-local Output has no semantic close verb. RunSession teardown must therefore remain
// non-EOS for both the general HoloMQ route and the direct-retained optimization; only an explicit
// catalog-publisher close may create an end-of-stream witness.
// REQ: CHAN-078
TEST(RuntimeChannelBinding, ProcessLocalCancellationDoesNotInventEndOfStream) {
  for (const bool direct_retained : {false, true}) {
    SCOPED_TRACE(direct_retained ? "direct_retained" : "holomq");
    SessionFixture fixture;
    if (direct_retained) {
      fixture.use_direct_retained_delivery();
    }
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    ASSERT_TRUE((*session)->refresh_connections());
    ASSERT_TRUE((*session)->arm());

    // Key setup: cancel and join the process-local transport without an explicit catalog close.
    (*session)->cancel();
    ASSERT_TRUE((*session)->join());
    const auto input = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    ASSERT_TRUE(input) << runtime_error_message(input.error());

    // Key assertion: neither endpoint teardown nor direct-store teardown fabricates semantic EOS.
    EXPECT_FALSE(input->end_of_stream_received);
  }
}

// Source-tree phase counters observe the accepted component boundaries of one complete scalar
// Channel cycle without changing the installed Runtime or Channel API.
// REQ: RUNTIME-011
// REQ: RUNTIME-029
TEST(RuntimeChannelBinding, PrivatePhaseCountersTrackOneCompleteScalarCycle) {
  SessionFixture fixture;
  const auto counters = std::make_shared<RuntimePhaseCounters>();
  fixture.plan.executable_factories.phase_counters = counters;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 2U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  // Key setup: publish, retain, select, read, and retire exactly one process-local scalar sample.
  constexpr std::array value{71};
  constexpr std::array<std::int64_t, 1U> capture_time{17};
  auto script = publish_scalar_script(**session, **binding, fixture.source, value, capture_time);
  ASSERT_TRUE(script) << runtime_error_message(script.error());
  ASSERT_TRUE((*session)->flush_process_local_activation_cut());
  const std::array selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = InputSelection{InputState::kSelected, SelectionReason::kNone},
      .samples = script->records,
      .batch = false,
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selection));
  {
    auto received = fixture.sink.input.receive();
    ASSERT_TRUE(received);
    EXPECT_EQ(received->data, value.front());
  }
  ASSERT_TRUE((*binding)->retire(script->events.front().source_sequence));
  ASSERT_TRUE((*session)->retire_samples(PortId{2U}, script->records));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());

  const auto snapshot = counters->snapshot();
  const auto count = [&](RuntimePhaseCounter counter) {
    return snapshot[RuntimePhaseCounters::index(counter)];
  };
  // Key assertion: each accepted component boundary is counted once, while source and sink
  // activation preparation/retirement are each counted twice and no backpressure is invented.
  EXPECT_EQ(count(RuntimePhaseCounter::kChannelPrepareCalls), 2U);
  EXPECT_EQ(count(RuntimePhaseCounter::kChannelEmitAttempts), 1U);
  EXPECT_EQ(count(RuntimePhaseCounter::kChannelEmitAccepted), 1U);
  EXPECT_EQ(count(RuntimePhaseCounter::kChannelSamplesReceived), 1U);
  EXPECT_EQ(count(RuntimePhaseCounter::kChannelReadinessPublished), 1U);
  EXPECT_EQ(count(RuntimePhaseCounter::kChannelInputReads), 1U);
  EXPECT_EQ(count(RuntimePhaseCounter::kChannelWitnessRetired), 1U);
  EXPECT_EQ(count(RuntimePhaseCounter::kChannelRetireCalls), 2U);
  EXPECT_EQ(count(RuntimePhaseCounter::kChannelActivationCuts), 1U);
  EXPECT_EQ(count(RuntimePhaseCounter::kChannelActivationCutDrains), 0U);
  EXPECT_EQ(count(RuntimePhaseCounter::kChannelActivationCutSkips), 1U);
  EXPECT_EQ(count(RuntimePhaseCounter::kChannelEmitBackpressured), 0U);
  EXPECT_EQ(count(RuntimePhaseCounter::kChannelReceiveBackpressured), 0U);

  const auto usage = counters->thread_usage_snapshot();
  const auto& pump_usage =
      usage[RuntimePhaseCounters::thread_usage_index(RuntimeThreadUsageRole::kChannelPump)];
  EXPECT_EQ(pump_usage.supported, runtime_thread_usage_detail::kSupported);
  EXPECT_EQ(pump_usage.samples, runtime_thread_usage_detail::kSupported ? 1U : 0U);
}

// Direct-retained sessions have no transport subscriber to drain at an activation boundary.
// Preserve the armed/status checks while eliding transport-cut accounting and synchronization.
// REQ: RUNTIME-029
TEST(RuntimeChannelBinding, DirectRetainedActivationCutElidesTransportPhaseAccounting) {
  SessionFixture fixture;
  fixture.use_direct_retained_delivery();
  const auto counters = std::make_shared<RuntimePhaseCounters>();
  fixture.plan.executable_factories.phase_counters = counters;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  ASSERT_TRUE((*session)->flush_process_local_activation_cut());

  const auto snapshot = counters->snapshot();
  const auto count = [&](RuntimePhaseCounter counter) {
    return snapshot[RuntimePhaseCounters::index(counter)];
  };
  EXPECT_EQ(count(RuntimePhaseCounter::kChannelActivationCuts), 0U);
  EXPECT_EQ(count(RuntimePhaseCounter::kChannelActivationCutDrains), 0U);
  EXPECT_EQ(count(RuntimePhaseCounter::kChannelActivationCutSkips), 0U);

  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// A receive overcount cannot prove that every accepted local publication reached typed storage.
// Exact progress mismatch must retain the original bounded drain even when no head is blocked.
// REQ: RUNTIME-029
TEST(RuntimeChannelBinding, ActivationCutCountOverrunFallsBackToBoundedDrain) {
  SessionFixture fixture;
  const auto counters = std::make_shared<RuntimePhaseCounters>();
  fixture.plan.executable_factories.phase_counters = counters;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  // Key setup: model an unmatched receive after one accepted process-local delivery.
  ASSERT_TRUE(
      RuntimeChannelSessionTestAccess::set_process_local_progress_counts(**session, 1U, 2U));
  ASSERT_TRUE((*session)->flush_process_local_activation_cut());

  const auto snapshot = counters->snapshot();
  // Key assertion: received > published is a mismatch, not a completed-cut proof.
  EXPECT_EQ(snapshot[RuntimePhaseCounters::index(RuntimePhaseCounter::kChannelActivationCutDrains)],
            1U);
  EXPECT_EQ(snapshot[RuntimePhaseCounters::index(RuntimePhaseCounter::kChannelActivationCutSkips)],
            0U);
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// A backend may make one local delivery visible before its publishing lane returns from send.
// Another partition's equal progress snapshot must not elide its cut during that attribution gap.
// REQ: RUNTIME-029
TEST(RuntimeChannelBinding, InFlightPublicationCannotPrepayAnotherActivationCut) {
  SessionFixture fixture;
  const auto counters = std::make_shared<RuntimePhaseCounters>();
  fixture.plan.executable_factories.phase_counters = counters;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto publisher = std::make_unique<HeldAcceptedInlinePublisher>();
  auto* const held_publisher = publisher.get();
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::move(publisher));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
      **session, PortId{1U}, std::move(publishers)));
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

  auto publication = std::async(
      std::launch::async, [&fixture] { return fixture.source.output.emit(std::int32_t{41}); });
  ScopeExit release_publication{[&]() noexcept { held_publisher->resume.release(); }};
  ASSERT_TRUE(held_publisher->entered.try_acquire_for(2s));
  const auto progress = RuntimeChannelSessionTestAccess::process_local_progress_state(**session);
  ASSERT_TRUE(progress) << runtime_error_message(progress.error());
  ASSERT_EQ(progress->publications_in_flight, 1U);

  // Key setup: model another lane's otherwise equal aggregate progress while this send remains
  // between backend visibility and Runtime publication accounting.
  ASSERT_TRUE(
      RuntimeChannelSessionTestAccess::set_process_local_progress_counts(**session, 1U, 1U));
  ASSERT_TRUE((*session)->flush_process_local_activation_cut());
  const auto snapshot = counters->snapshot();

  // Key assertion: the in-flight publication forces the bounded drain despite equal counts.
  EXPECT_EQ(snapshot[RuntimePhaseCounters::index(RuntimePhaseCounter::kChannelActivationCutDrains)],
            1U);
  EXPECT_EQ(snapshot[RuntimePhaseCounters::index(RuntimePhaseCounter::kChannelActivationCutSkips)],
            0U);
  held_publisher->resume.release();
  release_publication.release();
  const auto published = publication.get();
  ASSERT_TRUE(published) << runtime_error_message(published.error());
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// An accepted delivery blocked behind a retained typed slot keeps the exact progress counts
// unequal. The controller must run the ordinary bounded drain until retirement admits that head.
// REQ: RUNTIME-029
TEST(RuntimeChannelBinding, ActivationCutSkipWaitsForBlockedTransportHeadToMaterialize) {
  SessionFixture fixture;
  fixture.use_one_slot_blocking_transport();
  const auto counters = std::make_shared<RuntimePhaseCounters>();
  fixture.plan.executable_factories.phase_counters = counters;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 2U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  // Key setup: retain the first sample's only typed slot, then queue a second accepted sample.
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{41}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  auto first_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(first_event) << runtime_error_message(first_event.error());
  auto first_record = (*session)->temporal_sample(first_event->port, first_event->source_sequence);
  ASSERT_TRUE(first_record) << runtime_error_message(first_record.error());

  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 2U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{42}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 2U, true));
  ASSERT_TRUE((*session)->flush_process_local_activation_cut());
  auto snapshot = counters->snapshot();
  const auto count = [&](RuntimePhaseCounter counter) {
    return snapshot[RuntimePhaseCounters::index(counter)];
  };
  EXPECT_EQ(count(RuntimePhaseCounter::kChannelActivationCutDrains), 1U);
  EXPECT_EQ(count(RuntimePhaseCounter::kChannelActivationCutSkips), 0U);
  EXPECT_GE(count(RuntimePhaseCounter::kChannelReceiveBackpressured), 1U);

  // Release the typed slot. Either drain owner may materialize the retained transport head.
  ASSERT_TRUE((*binding)->retire(first_event->source_sequence));
  const std::array first_records{*first_record};
  ASSERT_TRUE((*session)->retire_samples(PortId{2U}, first_records));
  ASSERT_TRUE((*session)->flush_process_local_activation_cut());
  auto second_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(second_event) << runtime_error_message(second_event.error());
  auto second_record =
      (*session)->temporal_sample(second_event->port, second_event->source_sequence);
  ASSERT_TRUE(second_record) << runtime_error_message(second_record.error());
  ASSERT_TRUE((*session)->flush_process_local_activation_cut());

  snapshot = counters->snapshot();
  // Key assertion: only exact accepted-delivery/typed-store parity authorizes a user-space skip.
  EXPECT_EQ(count(RuntimePhaseCounter::kChannelActivationCuts), 3U);
  EXPECT_GE(count(RuntimePhaseCounter::kChannelActivationCutDrains), 1U);
  EXPECT_GE(count(RuntimePhaseCounter::kChannelActivationCutSkips), 1U);
  EXPECT_EQ(count(RuntimePhaseCounter::kChannelActivationCutDrains) +
                count(RuntimePhaseCounter::kChannelActivationCutSkips),
            3U);

  ASSERT_TRUE((*binding)->retire(second_event->source_sequence));
  const std::array second_records{*second_record};
  ASSERT_TRUE((*session)->retire_samples(PortId{2U}, second_records));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// A distributed relay can receive one inbound sample and publish one outbound sample while a
// second inbound transport head remains blocked. Coincident session-wide progress totals must not
// hide the pump's blocked-head retry after activation retirement frees typed storage.
// REQ: RUNTIME-033
TEST(RuntimeChannelBinding, BlockedTransportRetryOverridesCoincidentProgressCounts) {
  SessionFixture fixture;
  fixture.use_one_slot_blocking_transport();
  const auto counters = std::make_shared<RuntimePhaseCounters>();
  fixture.plan.executable_factories.phase_counters = counters;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
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

  // Key setup: retain the only typed slot, queue another sample, and make the controller observe
  // the blocked head before modeling the equal inbound/outbound totals of a relay process.
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{41}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  auto first_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(first_event) << runtime_error_message(first_event.error());
  auto first_record = (*session)->temporal_sample(first_event->port, first_event->source_sequence);
  ASSERT_TRUE(first_record) << runtime_error_message(first_record.error());

  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 2U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{42}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 2U, true));
  ASSERT_TRUE((*session)->flush_process_local_activation_cut());
  auto snapshot = counters->snapshot();
  EXPECT_GE(
      snapshot[RuntimePhaseCounters::index(RuntimePhaseCounter::kChannelReceiveBackpressured)], 1U);
  ASSERT_TRUE(
      RuntimeChannelSessionTestAccess::set_process_local_progress_counts(**session, 1U, 1U));

  ASSERT_TRUE((*binding)->retire(first_event->source_sequence));
  const std::array first_records{*first_record};
  ASSERT_TRUE((*session)->retire_samples(PortId{2U}, first_records));
  ASSERT_TRUE((*session)->flush_process_local_activation_cut());
  auto second_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(second_event) << runtime_error_message(second_event.error());
  auto second_record =
      (*session)->temporal_sample(second_event->port, second_event->source_sequence);
  ASSERT_TRUE(second_record) << runtime_error_message(second_record.error());

  const std::array selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&*second_record, 1U},
      .batch = false,
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selection));
  auto received = fixture.sink.input.receive();
  ASSERT_TRUE(received) << runtime_error_message(received.error());
  EXPECT_EQ(received->data, 42);
  ASSERT_TRUE((*binding)->retire(second_event->source_sequence));
  ASSERT_TRUE((*session)->retire_samples(PortId{2U}, selection.front().samples));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));

  snapshot = counters->snapshot();
  // Key assertion: the independent blocked-head latch forces both cuts through the bounded drain;
  // equal aggregate progress cannot defer the second sample to the 10 ms pump timeout.
  EXPECT_EQ(snapshot[RuntimePhaseCounters::index(RuntimePhaseCounter::kChannelActivationCutDrains)],
            2U);
  EXPECT_EQ(snapshot[RuntimePhaseCounters::index(RuntimePhaseCounter::kChannelActivationCutSkips)],
            0U);

  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// A cross-process receive is not evidence that an eventless process-local publication has reached
// typed storage. The relay's post-compute cut must synchronously materialize its local output even
// while the only pump wake attachment belongs to the cross-process input.
// REQ: RUNTIME-029
TEST(RuntimeChannelBinding, CrossProcessRelayReceiveCannotSkipLocalActivationCut) {
  MixedDomainRelayFixture fixture;
  const auto counters = std::make_shared<RuntimePhaseCounters>();
  fixture.base.plan.executable_factories.phase_counters = counters;
  struct WakeProbe {
    std::binary_semaphore cross_process_receive{0};
    std::binary_semaphore release_pump{0};
    std::atomic<bool> held{};
  } wake;

  auto producer =
      RuntimeChannelSession::create(fixture.base.plan, fixture.replay, ProcessGroupId{1U});
  auto worker = RuntimeChannelSession::create(
      fixture.base.plan,
      fixture.replay,
      ProcessGroupId{2U},
      RuntimeChannelReadinessWake{
          .state = &wake,
          .signal = [](void*) noexcept {},
          .signal_owner_progress =
              [](void* state, const VertexId owner) noexcept {
                auto* const probe = static_cast<WakeProbe*>(state);
                const VertexId relay_owner{2U};
                if (owner == relay_owner &&
                    !probe->held.exchange(true, std::memory_order_acq_rel)) {
                  probe->cross_process_receive.release();
                  probe->release_pump.acquire();
                }
              },
      });
  ASSERT_TRUE(producer) << runtime_error_message(producer.error());
  ASSERT_TRUE(worker) << runtime_error_message(worker.error());
  auto relay_binding =
      (*worker)->create_readiness_binding(VertexId{2U},
                                          RuntimePortBindingRequest{
                                              .plan_generation = fixture.base.plan.plan_generation,
                                              .run_generation = 1U,
                                              .kind = RuntimeEventKind::kInputArrival,
                                              .source = make_source_id(1U),
                                              .source_clock = ClockId{1U},
                                              .reference_clock = ClockId{1U},
                                              .capacity = 1U,
                                          });
  auto sink_binding =
      (*worker)->create_readiness_binding(VertexId{3U},
                                          RuntimePortBindingRequest{
                                              .plan_generation = fixture.base.plan.plan_generation,
                                              .run_generation = 1U,
                                              .kind = RuntimeEventKind::kInputArrival,
                                              .source = make_source_id(2U),
                                              .source_clock = ClockId{1U},
                                              .reference_clock = ClockId{1U},
                                              .capacity = 1U,
                                          });
  ASSERT_TRUE(relay_binding) << runtime_error_message(relay_binding.error());
  ASSERT_TRUE(sink_binding) << runtime_error_message(sink_binding.error());
  ASSERT_TRUE((*worker)->refresh_connections());
  ASSERT_TRUE((*producer)->refresh_connections());
  ASSERT_TRUE((*worker)->refresh_connections());
  ASSERT_TRUE((*worker)->arm());
  ASSERT_TRUE((*producer)->arm());

  std::atomic<bool> pump_released{};
  const auto unblock_pump = [&]() noexcept {
    if (!pump_released.exchange(true, std::memory_order_acq_rel)) {
      wake.release_pump.release();
    }
  };
  ScopeExit release_pump_on_failure{unblock_pump};

  // Key setup: retain one cross-process input after the pump has already scanned the empty local
  // subscriber, then publish the relay output while that pump scan remains held open.
  ASSERT_TRUE((*producer)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.base.source.output.emit(std::int32_t{42}));
  ASSERT_TRUE((*producer)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_TRUE(wake.cross_process_receive.try_acquire_for(2s));
  auto relay_event = wait_for_event(**worker, **relay_binding);
  ASSERT_TRUE(relay_event) << runtime_error_message(relay_event.error());
  auto relay_record = (*worker)->temporal_sample(relay_event->port, relay_event->source_sequence);
  ASSERT_TRUE(relay_record) << runtime_error_message(relay_record.error());
  const std::array relay_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&*relay_record, 1U},
      .batch = false,
      .destructive = true,
  }};
  ASSERT_TRUE((*worker)->prepare_activation(VertexId{2U}, 1U, relay_selection));
  auto relayed = fixture.base.sink.input.receive();
  ASSERT_TRUE(relayed) << runtime_error_message(relayed.error());
  ASSERT_TRUE(fixture.relay_output.output.emit(relayed->data));
  ASSERT_TRUE((*relay_binding)->retire(relay_event->source_sequence));
  ASSERT_TRUE((*worker)->retire_samples(PortId{2U}, relay_selection.front().samples));
  ASSERT_TRUE((*worker)->retire_activation(VertexId{2U}, 1U, true));

  unblock_pump();
  release_pump_on_failure.release();
  ASSERT_TRUE((*worker)->flush_process_local_activation_cut());
  const auto snapshot = counters->snapshot();
  EXPECT_EQ(snapshot[RuntimePhaseCounters::index(RuntimePhaseCounter::kChannelActivationCuts)], 1U);
  EXPECT_EQ(snapshot[RuntimePhaseCounters::index(RuntimePhaseCounter::kChannelActivationCutDrains)],
            1U);
  EXPECT_EQ(snapshot[RuntimePhaseCounters::index(RuntimePhaseCounter::kChannelActivationCutSkips)],
            0U);

  ASSERT_TRUE((*sink_binding)->poll(TimePoint{.timestamp_ns = 1, .clock_id = ClockId{1U}}));
  RuntimeEvent sink_event;
  auto sink_events = (*sink_binding)->drain(std::span<RuntimeEvent>{&sink_event, 1U});
  ASSERT_TRUE(sink_events) << runtime_error_message(sink_events.error());
  ASSERT_EQ(*sink_events, 1U);
  auto sink_record = (*worker)->temporal_sample(sink_event.port, sink_event.source_sequence);
  ASSERT_TRUE(sink_record) << runtime_error_message(sink_record.error());
  const std::array sink_selection{RuntimeChannelActivationSelection{
      .port = PortId{4U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&*sink_record, 1U},
      .batch = false,
      .destructive = true,
  }};
  ASSERT_TRUE((*worker)->prepare_activation(VertexId{3U}, 1U, sink_selection));
  auto received = fixture.base.second_sink.input.receive();
  ASSERT_TRUE(received) << runtime_error_message(received.error());
  // Key assertion: the local sink observes the relayed value in the same post-compute cut, without
  // a 10 ms pump timeout or another cross-process event.
  EXPECT_EQ(received->data, 42);
  ASSERT_TRUE((*sink_binding)->retire(sink_event.source_sequence));
  ASSERT_TRUE((*worker)->retire_samples(PortId{4U}, sink_selection.front().samples));
  ASSERT_TRUE((*worker)->retire_activation(VertexId{3U}, 1U, true));

  (*producer)->cancel();
  (*worker)->cancel();
  ASSERT_TRUE((*producer)->join());
  ASSERT_TRUE((*worker)->join());
}

// A clean process-local cache needs no execution-lane poll. One accepted typed arrival makes the
// prebound target visible, and only a complete poll acknowledges that cache revision.
// REQ: RUNTIME-016
// REQ: RUNTIME-026
TEST(RuntimeChannelBinding, PollsProcessLocalReadinessOnlyAfterInputRevisionChanges) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 2U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());

  // No ingress revision changed, so the controller can skip the synchronous target-lane poll.
  EXPECT_FALSE((*binding)->poll_target().has_value());
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::inject_received_metadata(
      **session,
      PortId{2U},
      SampleMetadata{
          .sequence = 1U,
          .byte_size = sizeof(std::int32_t),
          .payload_kind = PayloadKind::kInlineTyped,
      }));
  EXPECT_TRUE((*binding)->poll_target().has_value());

  // A complete poll publishes the witness and acknowledges exactly the captured revision.
  ASSERT_TRUE((*binding)->poll(TimePoint{1, ClockId{1U}}));
  EXPECT_FALSE((*binding)->poll_target().has_value());
  RuntimeEvent event;
  auto drained = (*binding)->drain(std::span<RuntimeEvent>{&event, 1U});
  ASSERT_TRUE(drained) << runtime_error_message(drained.error());
  ASSERT_EQ(*drained, 1U);
  auto sample = (*session)->temporal_sample(event.port, event.source_sequence);
  ASSERT_TRUE(sample) << runtime_error_message(sample.error());
  ASSERT_TRUE((*binding)->retire(event.source_sequence));
  ASSERT_TRUE(
      (*session)->retire_samples(event.port, std::span<const TemporalSampleRecord>{&*sample, 1U}));
}

// A potentially slow first-use transport import retains sole-consumer and scratch authority but
// must not block readiness or retirement from observing the independently protected typed cache.
// REQ: RUNTIME-023
// REQ: RUNTIME-042
// REQ: RUNTIME-044
TEST(RuntimeChannelBinding, TransportReceiveDoesNotHoldTypedCacheMutex) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());

  auto replacement = std::make_unique<BlockingInlineSubscriber>();
  auto* const blocking = replacement.get();
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_input_subscriber(
      **session, PortId{2U}, std::move(replacement)));

  expected<void, Error> receive_result = make_unexpected(Error{ErrorCode::kNotReady});
  std::thread receiver([&] {
    receive_result = RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U});
  });
  const auto receive_deadline = std::chrono::steady_clock::now() + 2s;
  while (!blocking->receive_entered.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < receive_deadline) {
    std::this_thread::yield();
  }

  std::atomic<bool> cache_observed{};
  expected<void, Error> cache_result = make_unexpected(Error{ErrorCode::kNotReady});
  std::thread cache_reader([&] {
    cache_result = RuntimeChannelSessionTestAccess::with_input_lock(
        **session, PortId{2U}, &cache_observed, &observe_input_lock);
  });
  const auto cache_deadline = std::chrono::steady_clock::now() + 2s;
  while (!cache_observed.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < cache_deadline) {
    std::this_thread::yield();
  }
  const bool cache_available_while_receive_blocked = cache_observed.load(std::memory_order_acquire);
  const auto reservations =
      RuntimeChannelSessionTestAccess::typed_ingress_reservations(**session, PortId{2U});
  blocking->release_receive.store(true, std::memory_order_release);
  cache_reader.join();
  receiver.join();

  ASSERT_TRUE(blocking->receive_entered.load(std::memory_order_acquire));
  EXPECT_TRUE(cache_available_while_receive_blocked);
  ASSERT_TRUE(reservations) << runtime_error_message(reservations.error());
  EXPECT_EQ(*reservations, 1U);
  ASSERT_TRUE(cache_result) << runtime_error_message(cache_result.error());
  ASSERT_TRUE(receive_result) << runtime_error_message(receive_result.error());
  const auto retained = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(retained) << runtime_error_message(retained.error());
  EXPECT_EQ(retained->occupied_samples, 1U);
  const auto released =
      RuntimeChannelSessionTestAccess::typed_ingress_reservations(**session, PortId{2U});
  ASSERT_TRUE(released) << runtime_error_message(released.error());
  EXPECT_EQ(*released, 0U);
}

// A lossy capacity-one cache may reserve its occupied pre-witness incumbent, but readiness must
// not acknowledge that temporarily hidden sample until receive either replaces it or fails and
// exposes it again.
// REQ: RUNTIME-026
// REQ: RUNTIME-042
// REQ: RUNTIME-044
TEST(RuntimeChannelBinding, LossyInlineReservationPreservesSuccessAndFailureReadiness) {
  // Key setup: pause a replacement receive after reserving the occupied one-slot incumbent and
  // exercise both commit and rollback outcomes.
  for (const ErrorCode receive_error : {ErrorCode::kSuccess, ErrorCode::kBackpressured}) {
    SCOPED_TRACE(receive_error == ErrorCode::kSuccess ? "successful receive"
                                                      : "backpressured receive");
    SessionFixture fixture;
    fixture.use_one_slot_blocking_transport();
    fixture.use_drop_oldest_queue_policy();
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());

    auto replacement = std::make_unique<BlockingInlineSubscriber>();
    auto* const blocking = replacement.get();
    blocking->block_receive.store(false, std::memory_order_release);
    blocking->value.store(11, std::memory_order_release);
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_input_subscriber(
        **session, PortId{2U}, std::move(replacement)));
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_pre_witness_replacement_allowed(
        **session, PortId{2U}, true));

    auto binding =
        (*session)->create_readiness_binding(VertexId{2U},
                                             RuntimePortBindingRequest{
                                                 .plan_generation = fixture.plan.plan_generation,
                                                 .run_generation = 1U,
                                                 .kind = RuntimeEventKind::kInputArrival,
                                                 .source = make_source_id(1U),
                                                 .source_clock = ClockId{1U},
                                                 .reference_clock = ClockId{1U},
                                                 .capacity = 1U,
                                             });
    ASSERT_TRUE(binding) << runtime_error_message(binding.error());
    ASSERT_TRUE((*binding)->poll_target().has_value());

    blocking->receive_entered.store(false, std::memory_order_release);
    blocking->release_receive.store(false, std::memory_order_release);
    blocking->block_receive.store(true, std::memory_order_release);
    blocking->receive_error.store(receive_error, std::memory_order_release);
    blocking->value.store(22, std::memory_order_release);
    blocking->sequence.store(2U, std::memory_order_release);
    expected<void, Error> receive_result = make_unexpected(Error{ErrorCode::kNotReady});
    std::thread receiver([&] {
      receive_result = RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U});
    });
    const auto receive_deadline = std::chrono::steady_clock::now() + 2s;
    while (!blocking->receive_entered.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < receive_deadline) {
      std::this_thread::yield();
    }

    // Key assertion: readiness and ordinary reclamation cannot expose or reclaim the incumbent
    // until the exact reservation commits the replacement or restores the prior sample.
    const auto reservations =
        RuntimeChannelSessionTestAccess::typed_ingress_reservations(**session, PortId{2U});
    const auto retained_while_blocked =
        RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    const auto reclaimed =
        RuntimeChannelSessionTestAccess::reclaim_reserved_input_slot(**session, PortId{2U});
    const auto retained_after_reclaim =
        RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    const auto blocked_poll = (*binding)->poll(TimePoint{1, ClockId{1U}});
    const bool retry_remains_visible = (*binding)->poll_target().has_value();
    std::array<RuntimeEvent, 1U> blocked_events{};
    const auto blocked_drain = (*binding)->drain(blocked_events);
    blocking->release_receive.store(true, std::memory_order_release);
    receiver.join();

    ASSERT_TRUE(blocking->receive_entered.load(std::memory_order_acquire));
    ASSERT_TRUE(reservations) << runtime_error_message(reservations.error());
    EXPECT_EQ(*reservations, 1U);
    ASSERT_TRUE(retained_while_blocked) << runtime_error_message(retained_while_blocked.error());
    EXPECT_EQ(retained_while_blocked->occupied_samples, 1U);
    EXPECT_EQ(retained_while_blocked->next_arrival_sequence, 2U);
    ASSERT_TRUE(reclaimed) << runtime_error_message(reclaimed.error());
    ASSERT_TRUE(retained_after_reclaim) << runtime_error_message(retained_after_reclaim.error());
    EXPECT_EQ(retained_after_reclaim->occupied_samples, 1U);
    EXPECT_EQ(retained_after_reclaim->next_arrival_sequence, 2U);
    ASSERT_TRUE(blocked_poll) << runtime_error_message(blocked_poll.error());
    EXPECT_TRUE(retry_remains_visible);
    ASSERT_TRUE(blocked_drain) << runtime_error_message(blocked_drain.error());
    EXPECT_EQ(*blocked_drain, 0U);

    if (receive_error == ErrorCode::kSuccess) {
      ASSERT_TRUE(receive_result) << runtime_error_message(receive_result.error());
    } else {
      ASSERT_FALSE(receive_result);
      EXPECT_EQ(receive_result.error().code, ErrorCode::kBackpressured);
    }
    const auto released =
        RuntimeChannelSessionTestAccess::typed_ingress_reservations(**session, PortId{2U});
    ASSERT_TRUE(released) << runtime_error_message(released.error());
    EXPECT_EQ(*released, 0U);
    ASSERT_TRUE((*binding)->poll_target().has_value());
    ASSERT_TRUE((*binding)->poll(TimePoint{2, ClockId{1U}}));
    RuntimeEvent event;
    auto drained = (*binding)->drain(std::span<RuntimeEvent>{&event, 1U});
    ASSERT_TRUE(drained) << runtime_error_message(drained.error());
    ASSERT_EQ(*drained, 1U);
    auto record = (*session)->temporal_sample(event.port, event.source_sequence);
    ASSERT_TRUE(record) << runtime_error_message(record.error());
    const std::uint64_t expected_sequence = receive_error == ErrorCode::kSuccess ? 2U : 1U;
    EXPECT_EQ(record->source_sequence, expected_sequence);
    EXPECT_EQ(record->arrival_sequence, expected_sequence);
    const std::array selection{RuntimeChannelActivationSelection{
        .port = PortId{2U},
        .selection = {InputState::kSelected, SelectionReason::kNone},
        .samples = std::span<const TemporalSampleRecord>{&*record, 1U},
        .destructive = true,
    }};
    ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selection));
    auto received = fixture.sink.input.receive();
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    EXPECT_EQ(received->data, receive_error == ErrorCode::kSuccess ? 22 : 11);
    EXPECT_EQ(received->metadata.sequence, expected_sequence);
    ASSERT_TRUE((*binding)->retire(event.source_sequence));
    ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
    ASSERT_TRUE((*session)->retire_samples(PortId{2U},
                                           std::span<const TemporalSampleRecord>{&*record, 1U}));
  }
}

// REQ: RUNTIME-020
// REQ: RUNTIME-024
TEST(RuntimeChannelBinding, MalformedTransportMetadataFailsBeforeTypedStoreMutation) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  const PortId input{2U};
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_external_metadata_contract(
      **session, input, ClockId{}, ClockId{}));
  const auto before = RuntimeChannelSessionTestAccess::input_reset_state(**session, input);
  ASSERT_TRUE(before) << runtime_error_message(before.error());

  const auto expect_rejected_without_mutation = [&](const SampleMetadata metadata) {
    auto rejected =
        RuntimeChannelSessionTestAccess::inject_received_metadata(**session, input, metadata);
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, ErrorCode::kProtocolError);
    const auto after = RuntimeChannelSessionTestAccess::input_reset_state(**session, input);
    ASSERT_TRUE(after) << runtime_error_message(after.error());
    EXPECT_EQ(after->active_generation, before->active_generation);
    EXPECT_EQ(after->view_generation, before->view_generation);
    EXPECT_EQ(after->producer_epoch, before->producer_epoch);
    EXPECT_EQ(after->next_arrival_sequence, before->next_arrival_sequence);
    EXPECT_EQ(after->next_global_arrival_order, before->next_global_arrival_order);
    EXPECT_EQ(after->occupied_samples, before->occupied_samples);
  };

  const SampleMetadata canonical{
      .sequence = 1U,
      .byte_size = sizeof(std::int32_t),
      .payload_kind = PayloadKind::kInlineTyped,
  };
  auto malformed = canonical;
  malformed.sequence = 0U;
  expect_rejected_without_mutation(malformed);
  malformed = canonical;
  malformed.version.major = 2U;
  expect_rejected_without_mutation(malformed);
  malformed = canonical;
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  malformed.flags = static_cast<SampleFlags>(4U);
  // The analyzer reports the intentionally malformed enum when the lambda reads the copy.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  expect_rejected_without_mutation(malformed);
  malformed = canonical;
  malformed.byte_size = sizeof(std::int16_t);
  expect_rejected_without_mutation(malformed);
  malformed = canonical;
  malformed.payload_kind = PayloadKind::kSerializedBytes;
  expect_rejected_without_mutation(malformed);
  malformed = canonical;
  malformed.publish_timestamp_ns = 1;
  expect_rejected_without_mutation(malformed);
  malformed = canonical;
  malformed.publish_clock_id = ClockId{1U};
  expect_rejected_without_mutation(malformed);
  malformed = canonical;
  malformed.capture_timestamp_ns = 1;
  expect_rejected_without_mutation(malformed);
  malformed = canonical;
  malformed.source_clock_id = ClockId{1U};
  expect_rejected_without_mutation(malformed);
}

// The sibling case above covers an incoherent coordinate arriving at a session with no publish
// clock. This covers the other rejection: a session that does stamp publications receiving a
// coordinate attributed to some other clock. A process-local producer and its consumer share one
// plan clock catalog, so a foreign identity cannot be placed on the consumer's timeline at all.
// Admitting it would hand the consumer a plausible number in a domain it would read as its own.
// Canonical absence is equally inadmissible here, because a dropped stamp must not be able to
// present itself as a producer that legitimately has no publish clock.
// REQ: RUNTIME-020
// REQ: RUNTIME-024
TEST(RuntimeChannelBinding, ForeignProcessLocalPublishClockFailsBeforeTypedStoreMutation) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());

  ClockRegistryBuilder registry_builder;
  const ClockKey publish_key = ClockKey::named("publish-domain");
  ASSERT_TRUE(registry_builder.add_clock(publish_key,
                                         std::make_shared<ManualClock>(0),
                                         ClockCapabilityOptions{
                                             .can_read_now = true,
                                             .steady = true,
                                         }));
  ASSERT_TRUE(registry_builder.set_default(publish_key));
  auto registry = registry_builder.build();
  ASSERT_TRUE(registry) << runtime_error_message(registry.error());
  auto publish_clock = registry->resolve(publish_key);
  ASSERT_TRUE(publish_clock) << runtime_error_message(publish_clock.error());
  const ClockId bound = publish_clock->id();
  ASSERT_TRUE(bound.valid());
  ASSERT_TRUE((*session)->bind_publish_clock(*publish_clock));

  // The port is left process-local on purpose. An external import must carry canonical absence
  // whatever this session stamps, so declaring an external contract here would reject the foreign
  // coordinate merely for being populated and would exercise the same branch the sibling above
  // already covers, leaving the bound identity untested.
  const PortId input{2U};
  const auto before = RuntimeChannelSessionTestAccess::input_reset_state(**session, input);
  ASSERT_TRUE(before) << runtime_error_message(before.error());

  const auto expect_rejected_without_mutation = [&](const SampleMetadata metadata) {
    auto rejected =
        RuntimeChannelSessionTestAccess::inject_received_metadata(**session, input, metadata);
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, ErrorCode::kProtocolError);
    const auto after = RuntimeChannelSessionTestAccess::input_reset_state(**session, input);
    ASSERT_TRUE(after) << runtime_error_message(after.error());
    EXPECT_EQ(after->next_arrival_sequence, before->next_arrival_sequence);
    EXPECT_EQ(after->next_global_arrival_order, before->next_global_arrival_order);
    EXPECT_EQ(after->occupied_samples, before->occupied_samples);
  };

  const SampleMetadata canonical{
      .sequence = 1U,
      .byte_size = sizeof(std::int32_t),
      .payload_kind = PayloadKind::kInlineTyped,
  };
  // A clock this session never stamps from, coherent in itself and still inadmissible.
  auto foreign = canonical;
  foreign.publish_timestamp_ns = 7;
  foreign.publish_clock_id = ClockId{bound.value + 1U};
  expect_rejected_without_mutation(foreign);
  // The stamp is mandatory once a publish clock is bound, so absence is a dropped stamp.
  expect_rejected_without_mutation(canonical);
}

// REQ: RUNTIME-015
// REQ: RUNTIME-016
// REQ: RUNTIME-017
TEST(RuntimeChannelBinding, PublishesSelectsAndInvalidatesOneScalarSample) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
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

  // Key setup: emit a scalar and retire the producer activation to deliver the sample.
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  std::this_thread::sleep_for(20ms);
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{42}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  auto event = wait_for_event(**session, **binding);
  ASSERT_TRUE(event) << runtime_error_message(event.error());
  auto temporal = (*session)->temporal_sample(event->port, event->source_sequence);
  ASSERT_TRUE(temporal) << runtime_error_message(temporal.error());
  const std::array selected{*temporal};
  const std::array selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = selected,
      .batch = false,
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selection));
  auto received = fixture.sink.input.receive();
  ASSERT_TRUE(received) << runtime_error_message(received.error());
  // Key assertion: received value matches the emitted sample and carries its sequence number.
  EXPECT_EQ(received->data, 42);
  EXPECT_EQ(received->metadata.sequence, temporal->source_sequence);
  ASSERT_TRUE((*binding)->retire(event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  auto stale = fixture.sink.input.receive_data();
  ASSERT_FALSE(stale);
  EXPECT_EQ(stale.error().code, ErrorCode::kStaleEndpoint);

  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
  // The copied Sample and its finite read claim deliberately outlive callback retirement.
  EXPECT_EQ(received->data, 42);
}

// REQ: RUNTIME-019
TEST(RuntimeChannelBinding, PublishesAConstructedGenericLoan) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 8U,
                                           });
  ASSERT_TRUE(binding);
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  std::this_thread::sleep_for(20ms);
  auto loan = fixture.source.output.allocate();
  ASSERT_TRUE(loan) << runtime_error_message(loan.error());
  loan->value() = 73;
  ASSERT_TRUE(fixture.source.output.emit(std::move(*loan)));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  auto event = wait_for_event(**session, **binding);
  ASSERT_TRUE(event) << runtime_error_message(event.error());
  auto temporal = (*session)->temporal_sample(event->port, event->source_sequence);
  ASSERT_TRUE(temporal);
  const std::array selected{*temporal};
  const std::array selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = selected,
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selection));
  auto value = fixture.sink.input.receive_data();
  ASSERT_TRUE(value);
  EXPECT_EQ(*value, 73);
  ASSERT_TRUE((*binding)->retire(event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// A proven pre-submit rejection leaves the exact caller-owned loan armed. The Runtime does not
// allocate a replacement or retry it; the caller may correct the condition and explicitly submit
// the same authority again during the same activation.
// REQ: RUNTIME-019
TEST(RuntimeChannelBinding, PreservesGenericLoanAcrossPreSubmitBackpressure) {
  Graph graph;
  SessionFixture fixture;
  (void)graph.add_operator<SourceOperator>("source");
  (void)graph.add_operator<SinkOperator>("sink");
  fixture.plan.graph = graph.view();
  fixture.plan.generic_output_loans.front().max_live_loans = 1U;
  (void)profiler::operator_profile_domain();
  ASSERT_TRUE(test::nvtx_emit_marker_capture::reset("source.output:emit"));
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto publisher = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{
          channel_internal::InlinePublicationResult{
              .phase = channel_internal::InlinePublicationPhase::kPreSubmitRejected,
              .recipient_count = 0U,
              .error = Error{ErrorCode::kBackpressured},
          },
          channel_internal::InlinePublicationResult{
              .phase = channel_internal::InlinePublicationPhase::kAccepted,
              .recipient_count = 1U,
              .error = Error{ErrorCode::kSuccess},
          },
      });
  auto* observed = publisher.get();
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::move(publisher));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
      **session, PortId{1U}, std::move(publishers)));
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

  auto loan = fixture.source.output.allocate();
  ASSERT_TRUE(loan) << runtime_error_message(loan.error());
  loan->value() = 22;
  auto rejected = fixture.source.output.emit(std::move(*loan));
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kBackpressured);
  EXPECT_EQ(test::nvtx_emit_marker_capture::emitted(), 0U);
  auto still_held = fixture.source.output.allocate();
  ASSERT_FALSE(still_held);
  EXPECT_EQ(still_held.error().code, ErrorCode::kResourceExhausted);

  ASSERT_TRUE(fixture.source.output.emit(std::move(*loan)));
  EXPECT_EQ(test::nvtx_emit_marker_capture::emitted(), 1U);
  auto replacement = fixture.source.output.allocate();
  ASSERT_TRUE(replacement) << runtime_error_message(replacement.error());
  replacement->cancel();
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  EXPECT_EQ(observed->sequences, (std::vector<std::uint64_t>{1U, 1U}));
  EXPECT_EQ(observed->values, (std::vector<std::int32_t>{22, 22}));
}

// Nonconsuming and consuming outcomes retain their existing accounting across a legal cutover.
// REQ: RUNTIME-052
TEST(RuntimeChannelBinding, PublicationOutcomeAccountingSurvivesRestart) {
  // Key setup: source.output -> scripted publisher; reject, consume uncertain authority, then
  // accept, with a committed restart between each publication attempt.
  // Key assertion: observed sequences are 1, 1, 2 and next is 3; only consuming outcomes advance.
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session);
  auto publisher = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{
          {.phase = channel_internal::InlinePublicationPhase::kPreSubmitRejected,
           .recipient_count = 0U,
           .error = Error{ErrorCode::kBackpressured}},
          {.phase = channel_internal::InlinePublicationPhase::kPostSubmitUncertain,
           .recipient_count = 0U,
           .error = Error{ErrorCode::kPublicationIndeterminate}},
          {.phase = channel_internal::InlinePublicationPhase::kAccepted,
           .recipient_count = 1U,
           .error = Error{ErrorCode::kSuccess}},
      });
  auto* const observed = publisher.get();
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::move(publisher));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
      **session, PortId{1U}, std::move(publishers)));
  for (const std::uint64_t generation : {1U, 2U, 3U}) {
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, generation, {}));
    auto emitted = fixture.source.output.emit(std::int32_t{31});
    if (generation < 3U) {
      ASSERT_FALSE(emitted);
      EXPECT_EQ(
          emitted.error().code,
          generation == 1U ? ErrorCode::kBackpressured : ErrorCode::kPublicationIndeterminate);
    } else {
      ASSERT_TRUE(emitted);
    }
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, generation, true));
    const std::array affected{VertexId{1U}, VertexId{2U}};
    ASSERT_TRUE((*session)->commit_restart(VertexId{1U}, affected));
  }
  EXPECT_EQ(observed->sequences, (std::vector<std::uint64_t>{1U, 1U, 2U}));
  auto counter = RuntimeChannelSessionTestAccess::output_next_sequence(**session, PortId{1U});
  ASSERT_TRUE(counter);
  EXPECT_EQ(*counter, 3U);
}

// A pre-submit refresh failure owns no close authority; retry publishes one terminal operation.
// REQ: CHAN-136
TEST(RuntimeChannelBinding, CloseOwnerOutputsRefreshesBeforePublishingEndOfStream) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto publisher = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{},
      std::vector{ErrorCode::kBackpressured, ErrorCode::kSuccess},
      std::vector{ErrorCode::kSuccess});
  auto* const observed = publisher.get();
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::move(publisher));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
      **session, PortId{1U}, std::move(publishers)));

  auto pressured = (*session)->close_owner_outputs(VertexId{1U});
  ASSERT_FALSE(pressured);
  EXPECT_EQ(pressured.error().code, ErrorCode::kBackpressured);
  EXPECT_EQ(observed->close_count, 0U);

  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  EXPECT_EQ(observed->close_count, 1U);
  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  EXPECT_EQ(observed->close_count, 1U);
}

// Bounded catalog-publisher loan exhaustion happens before terminal submission. Transport
// locality does not change that authority fact: retry must reuse the same unresolved close.
// REQ: CHAN-136
TEST(RuntimeChannelBinding, NonLocalPreSubmitResourceExhaustionRemainsRetryable) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto publisher = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{},
      std::vector<ErrorCode>{},
      std::vector{ErrorCode::kResourceExhausted, ErrorCode::kSuccess});
  auto* const observed = publisher.get();
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::move(publisher));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
      **session, PortId{1U}, std::move(publishers), {}, true));

  auto pressured = (*session)->close_owner_outputs(VertexId{1U});
  ASSERT_FALSE(pressured);
  EXPECT_EQ(pressured.error().code, ErrorCode::kBackpressured);
  EXPECT_EQ(observed->close_count, 1U);

  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  EXPECT_EQ(observed->close_count, 2U);
  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  EXPECT_EQ(observed->close_count, 2U);
}

// A retry preserves every accepted fanout arm and submits only the still-unresolved suffix.
// REQ: CHAN-136
// REQ: RUNTIME-052
TEST(RuntimeChannelBinding, CloseOwnerOutputsCompletesAfterPartialFanoutRetry) {
  // Key setup: source.output --+-> accepting arm
  //                          +-> backpressured-then-accepting arm
  // Start at one and at UINT64_MAX - 1; retry the partially completed fanout.
  // Key assertion: the resolved arm is never refreshed or closed again, and final completion
  // charges one coordinate while subsequent completed closes remain idempotent at exhaustion.
  for (const std::uint64_t sequence :
       {std::uint64_t{1U}, std::numeric_limits<std::uint64_t>::max() - 1U}) {
    SCOPED_TRACE(sequence);
    SessionFixture fixture;
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    ASSERT_TRUE(
        RuntimeChannelSessionTestAccess::set_output_next_sequence(**session, PortId{1U}, sequence));
    auto first = std::make_unique<ScriptedInlinePublisher>(
        std::vector<channel_internal::InlinePublicationResult>{},
        std::vector{ErrorCode::kSuccess, ErrorCode::kProtocolError},
        std::vector{ErrorCode::kSuccess, ErrorCode::kClosed});
    auto second = std::make_unique<ScriptedInlinePublisher>(
        std::vector<channel_internal::InlinePublicationResult>{},
        std::vector{ErrorCode::kSuccess, ErrorCode::kSuccess},
        std::vector{ErrorCode::kBackpressured, ErrorCode::kSuccess});
    auto* const first_observed = first.get();
    auto* const second_observed = second.get();
    std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
    publishers.push_back(std::move(first));
    publishers.push_back(std::move(second));
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
        **session, PortId{1U}, std::move(publishers), {}, true));

    auto pressured = (*session)->close_owner_outputs(VertexId{1U});
    ASSERT_FALSE(pressured);
    EXPECT_EQ(pressured.error().code, ErrorCode::kBackpressured);
    EXPECT_EQ(first_observed->close_count, 1U);
    EXPECT_EQ(second_observed->close_count, 1U);

    ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
    EXPECT_EQ(first_observed->close_count, 1U);
    EXPECT_EQ(second_observed->close_count, 2U);

    ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
    EXPECT_EQ(first_observed->close_count, 1U);
    EXPECT_EQ(second_observed->close_count, 2U);
    auto counter = RuntimeChannelSessionTestAccess::output_next_sequence(**session, PortId{1U});
    ASSERT_TRUE(counter);
    EXPECT_EQ(*counter, sequence + 1U);
  }
}

// Observe cold terminal accounting without conflating arm resolution with published EOS.
// Representative caller: ExhaustedOptionalNoPeerCloseResolvesLocally.
void expect_output_close_state(const RuntimeChannelSession& session,
                               const std::uint64_t next_sequence, const std::size_t resolved_arms) {
  auto sequence = RuntimeChannelSessionTestAccess::output_next_sequence(session, PortId{1U});
  ASSERT_TRUE(sequence) << runtime_error_message(sequence.error());
  EXPECT_EQ(*sequence, next_sequence);
  auto resolved =
      RuntimeChannelSessionTestAccess::resolved_output_terminal_arms(session, PortId{1U});
  ASSERT_TRUE(resolved) << runtime_error_message(resolved.error());
  EXPECT_EQ(*resolved, resolved_arms);
}

// REQ: RUNTIME-052
// REQ: CHAN-136
TEST(RuntimeChannelBinding, ExhaustedOptionalNoPeerCloseResolvesLocally) {
  // Key setup: an optional exported arm has neither a peer nor accepted recipient data.
  // Fixture topology: source.output -> scripted optional export (no receiver).
  for (const std::uint64_t sequence :
       {std::uint64_t{1U}, std::numeric_limits<std::uint64_t>::max()}) {
    SCOPED_TRACE(sequence);
    SessionFixture fixture;
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session);
    auto publisher = std::make_unique<ScriptedInlinePublisher>(
        std::vector<channel_internal::InlinePublicationResult>{});
    auto* const observed = publisher.get();
    observed->subscribers_present = false;
    std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
    publishers.push_back(std::move(publisher));
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
        **session, PortId{1U}, std::move(publishers), {}, true, false));
    ASSERT_TRUE(
        RuntimeChannelSessionTestAccess::set_output_next_sequence(**session, PortId{1U}, sequence));

    // Key assertion: local completion resolves the arm without EOS or sequence consumption,
    // even when the next counter is exhausted; a repeated completed close performs no work.
    ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
    EXPECT_EQ(observed->refresh_count, 1U);
    EXPECT_EQ(observed->close_count, 0U);
    expect_output_close_state(**session, sequence, 1U);
    ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
    EXPECT_EQ(observed->refresh_count, 1U);
    EXPECT_EQ(observed->close_count, 0U);
    expect_output_close_state(**session, sequence, 1U);
    auto reset = (*session)->reset();
    ASSERT_FALSE(reset);
    EXPECT_EQ(reset.error().code, ErrorCode::kClosed);
  }
}

// REQ: RUNTIME-052
// REQ: CHAN-136
// REQ: CHAN-144
TEST(RuntimeChannelBinding, ExhaustedRequiredExportAbandonmentReturnsClosed) {
  // Key setup: source.output -> required export -> recipient; accept data before the
  // recipient disappears at terminal refresh, with the Runtime next counter exhausted.
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session);
  auto publisher = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{});
  auto* const observed = publisher.get();
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::move(publisher));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
      **session, PortId{1U}, std::move(publishers), {}, true));
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{41}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_EQ(observed->sequences, (std::vector<std::uint64_t>{1U}));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_output_next_sequence(
      **session, PortId{1U}, std::numeric_limits<std::uint64_t>::max()));
  const auto refreshes = observed->refresh_count;
  observed->remove_subscribers_on_refresh = true;

  // Key assertion: abandonment resolves the arm with kClosed, not exhaustion, and never
  // submits/counts EOS. Neither reset nor a close retry can reopen the abandoned arm.
  auto closed = (*session)->close_owner_outputs(VertexId{1U});
  ASSERT_FALSE(closed);
  EXPECT_EQ(closed.error().code, ErrorCode::kClosed);
  EXPECT_EQ(observed->refresh_count, refreshes + 1U);
  EXPECT_EQ(observed->close_count, 0U);
  expect_output_close_state(**session, std::numeric_limits<std::uint64_t>::max(), 1U);
  auto reset = (*session)->reset();
  ASSERT_FALSE(reset);
  EXPECT_EQ(reset.error().code, ErrorCode::kClosed);
  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  EXPECT_EQ(observed->refresh_count, refreshes + 1U);
  EXPECT_EQ(observed->close_count, 0U);
  expect_output_close_state(**session, std::numeric_limits<std::uint64_t>::max(), 1U);
}

// REQ: RUNTIME-052
// REQ: CHAN-136
TEST(RuntimeChannelBinding, ExhaustedOptionalAcceptedRecipientRemainsObligated) {
  // Key setup: source.output -> optional export -> recipient; accept data before the
  // recipient disappears at terminal refresh, with the Runtime next counter exhausted.
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session);
  auto publisher = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{});
  auto* const observed = publisher.get();
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::move(publisher));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
      **session, PortId{1U}, std::move(publishers), {}, true, false));
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{41}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_EQ(observed->sequences, (std::vector<std::uint64_t>{1U}));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_output_next_sequence(
      **session, PortId{1U}, std::numeric_limits<std::uint64_t>::max()));
  const auto refreshes = observed->refresh_count;
  observed->remove_subscribers_on_refresh = true;

  // Key assertion: accepted recipient data preserves the optional arm's terminal obligation;
  // losing that peer cannot turn exhausted EOS into successful local resolution.
  auto closed = (*session)->close_owner_outputs(VertexId{1U});
  ASSERT_FALSE(closed);
  EXPECT_EQ(closed.error().code, ErrorCode::kResourceExhausted);
  EXPECT_EQ(observed->refresh_count, refreshes + 1U);
  EXPECT_EQ(observed->close_count, 0U);
  expect_output_close_state(**session, std::numeric_limits<std::uint64_t>::max(), 0U);
}

// REQ: RUNTIME-052
// REQ: CHAN-136
TEST(RuntimeChannelBinding, ExhaustedClosePreservesRefreshFailure) {
  // Key setup: source.output -> scripted connected export; terminal refresh fails before
  // the exhausted output can classify its recipient.
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session);
  auto publisher = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{},
      std::vector{ErrorCode::kBackpressured});
  auto* const observed = publisher.get();
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::move(publisher));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
      **session, PortId{1U}, std::move(publishers), {}, true));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_output_next_sequence(
      **session, PortId{1U}, std::numeric_limits<std::uint64_t>::max()));
  // Key assertion: the existing refresh error survives without EOS or terminal resolution.
  auto closed = (*session)->close_owner_outputs(VertexId{1U});
  ASSERT_FALSE(closed);
  EXPECT_EQ(closed.error().code, ErrorCode::kBackpressured);
  EXPECT_EQ(observed->refresh_count, 1U);
  EXPECT_EQ(observed->close_count, 0U);
  expect_output_close_state(**session, std::numeric_limits<std::uint64_t>::max(), 0U);
}

// REQ: RUNTIME-052
// REQ: CHAN-136
TEST(RuntimeChannelBinding, ExhaustedClosePreservesEarlierLocalResolution) {
  // Key setup: source.output -> optional absent arm
  //                         -> optional connected arm requiring EOS.
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session);
  auto first = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{});
  auto second = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{});
  auto* const local = first.get();
  auto* const connected = second.get();
  local->subscribers_present = false;
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::move(first));
  publishers.push_back(std::move(second));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
      **session, PortId{1U}, std::move(publishers), {}, true, false));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_output_next_sequence(
      **session, PortId{1U}, std::numeric_limits<std::uint64_t>::max()));
  // Key assertion: exhaustion on the connected arm preserves the earlier local resolution;
  // retries skip that resolved arm and neither arm submits EOS.
  for (std::size_t attempt = 0U; attempt < 2U; ++attempt) {
    auto closed = (*session)->close_owner_outputs(VertexId{1U});
    ASSERT_FALSE(closed);
    EXPECT_EQ(closed.error().code, ErrorCode::kResourceExhausted);
    EXPECT_EQ(local->refresh_count, 1U);
    EXPECT_EQ(local->close_count, 0U);
    EXPECT_EQ(connected->close_count, 0U);
    expect_output_close_state(**session, std::numeric_limits<std::uint64_t>::max(), 1U);
  }
}

// REQ: RUNTIME-052
// REQ: CHAN-136
TEST(RuntimeChannelBinding, TerminalCompletionRetryRemembersConsumedAuthority) {
  // Key setup: source.output --+-> optional accepting arm (second scenario only)
  //                          +-> authority-consuming uncertain arm
  // Exercise uncertainty alone and after an accepted arm so either can establish accounting.
  for (const bool accepted_prefix : {false, true}) {
    SCOPED_TRACE(accepted_prefix);
    SessionFixture fixture;
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session);
    auto publisher = std::make_unique<ScriptedInlinePublisher>(
        std::vector<channel_internal::InlinePublicationResult>{},
        std::vector<ErrorCode>{},
        std::vector{ErrorCode::kPublicationIndeterminate});
    auto* const observed = publisher.get();
    std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
    auto accepted = std::make_unique<ScriptedInlinePublisher>(
        std::vector<channel_internal::InlinePublicationResult>{});
    auto* const first = accepted.get();
    if (accepted_prefix) {
      publishers.push_back(std::move(accepted));
    }
    publishers.push_back(std::move(publisher));
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
        **session, PortId{1U}, std::move(publishers), {}, true));
    const auto last_sequence = std::numeric_limits<std::uint64_t>::max() - 1U;
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_output_next_sequence(
        **session, PortId{1U}, last_sequence));
    auto uncertain = (*session)->close_owner_outputs(VertexId{1U});
    ASSERT_FALSE(uncertain);
    EXPECT_EQ(uncertain.error().code, ErrorCode::kPublicationIndeterminate);
    expect_output_close_state(**session, last_sequence, accepted_prefix ? 2U : 1U);
    // Key assertion: a retry which submits no EOS still charges the earlier consumed operation
    // exactly once; a completed retry at exhaustion remains idempotent.
    ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
    ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
    if (accepted_prefix) {
      EXPECT_EQ(first->close_count, 1U);
    }
    EXPECT_EQ(observed->close_count, 1U);
    expect_output_close_state(
        **session, std::numeric_limits<std::uint64_t>::max(), accepted_prefix ? 2U : 1U);
  }
}

// REQ: RUNTIME-052
// REQ: CHAN-136
TEST(RuntimeChannelBinding, ExhaustedDirectCloseDoesNotStoreEndOfStream) {
  // Key setup: source.output -> direct-retained sink.input, with exhausted output capacity.
  // Key assertion: no terminal storage, delivery count, or sequence changes on repeated close.
  SessionFixture fixture;
  fixture.use_direct_retained_delivery();
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session);
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_output_next_sequence(
      **session, PortId{1U}, std::numeric_limits<std::uint64_t>::max()));
  for (std::size_t attempt = 0U; attempt < 2U; ++attempt) {
    auto closed = (*session)->close_owner_outputs(VertexId{1U});
    ASSERT_FALSE(closed);
    EXPECT_EQ(closed.error().code, ErrorCode::kResourceExhausted);
    expect_output_close_state(**session, std::numeric_limits<std::uint64_t>::max(), 0U);
    auto input = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    ASSERT_TRUE(input);
    EXPECT_FALSE(input->end_of_stream_received);
    auto progress = RuntimeChannelSessionTestAccess::process_local_progress_state(**session);
    ASSERT_TRUE(progress);
    EXPECT_EQ(progress->published_deliveries, 0U);
    EXPECT_EQ(progress->publications_in_flight, 0U);
  }
}

// A newly exhausted close may refresh its recipient but must not submit EOS.
// REQ: RUNTIME-052
TEST(RuntimeChannelBinding, ExhaustedCloseDoesNotSubmitEndOfStream) {
  // Key setup: source.output -> scripted connected export; refresh succeeds at exhaustion.
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session);
  auto publisher = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{}, std::vector{ErrorCode::kSuccess});
  auto* const observed = publisher.get();
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::move(publisher));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
      **session, PortId{1U}, std::move(publishers), {}, true));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_output_next_sequence(
      **session, PortId{1U}, std::numeric_limits<std::uint64_t>::max()));
  auto closed = (*session)->close_owner_outputs(VertexId{1U});
  // Key assertion: classification refresh runs, but exhausted EOS consumes no authority.
  ASSERT_FALSE(closed);
  EXPECT_EQ(closed.error().code, ErrorCode::kResourceExhausted);
  EXPECT_EQ(observed->close_count, 0U);
  EXPECT_EQ(observed->refresh_count, 1U);
  EXPECT_TRUE(observed->sequences.empty());
  expect_output_close_state(**session, std::numeric_limits<std::uint64_t>::max(), 0U);
}

// Terminal authority is monotonic within one session; an administrative reset cannot reopen an
// endpoint after any arm has committed EOS.
// REQ: CHAN-136
TEST(RuntimeChannelBinding, FullResetCannotReopenClosedOwnerOutputs) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto publisher = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{});
  auto* const observed = publisher.get();
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::move(publisher));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
      **session, PortId{1U}, std::move(publishers)));

  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  auto reset = (*session)->reset();
  ASSERT_FALSE(reset);
  EXPECT_EQ(reset.error().code, ErrorCode::kClosed);
  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));

  EXPECT_EQ(observed->close_count, 1U);
}

// An authority-consuming close result resolves only that fanout arm. Lifecycle control cannot
// reopen the consumed authority, while an owner-close retry skips it and closes the unresolved
// suffix exactly once.
// REQ: CHAN-136
TEST(RuntimeChannelBinding, AuthorityConsumingCloseFailureResolvesOnlyThatArmAcrossRetry) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto first = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{},
      std::vector<ErrorCode>{},
      std::vector{ErrorCode::kPublicationIndeterminate});
  auto second = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{},
      std::vector<ErrorCode>{},
      std::vector{ErrorCode::kSuccess});
  auto* const first_observed = first.get();
  auto* const second_observed = second.get();
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::move(first));
  publishers.push_back(std::move(second));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
      **session, PortId{1U}, std::move(publishers), {}, true));

  auto closed = (*session)->close_owner_outputs(VertexId{1U});
  ASSERT_FALSE(closed);
  EXPECT_EQ(closed.error().code, ErrorCode::kPublicationIndeterminate);
  EXPECT_EQ(first_observed->close_count, 1U);
  EXPECT_EQ(second_observed->close_count, 0U);

  const std::array affected{VertexId{1U}, VertexId{2U}};
  auto full_reset = (*session)->reset();
  ASSERT_FALSE(full_reset);
  EXPECT_EQ(full_reset.error().code, ErrorCode::kClosed);
  auto scoped_reset = (*session)->reset(VertexId{1U});
  ASSERT_FALSE(scoped_reset);
  EXPECT_EQ(scoped_reset.error().code, ErrorCode::kClosed);
  auto restart = (*session)->commit_restart(VertexId{1U}, affected);
  ASSERT_FALSE(restart);
  EXPECT_EQ(restart.error().code, ErrorCode::kClosed);

  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  EXPECT_EQ(first_observed->close_count, 1U);
  EXPECT_EQ(second_observed->close_count, 1U);
  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  EXPECT_EQ(first_observed->close_count, 1U);
  EXPECT_EQ(second_observed->close_count, 1U);
}

// A throwable backend boundary is classified as indeterminate instead of escaping this noexcept
// owner-close transaction. The affected arm and process-local accounting are consumed exactly
// once, while a retry may close only the unresolved suffix.
// REQ: CHAN-136
TEST(RuntimeChannelBinding, ThrowingCloseResolvesOnlyThatArmAndInvalidatesLocalAccounting) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto first = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{},
      std::vector<ErrorCode>{},
      std::vector<ErrorCode>{},
      true);
  auto second = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{},
      std::vector<ErrorCode>{},
      std::vector{ErrorCode::kSuccess});
  auto* const first_observed = first.get();
  auto* const second_observed = second.get();
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::move(first));
  publishers.push_back(std::move(second));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
      **session, PortId{1U}, std::move(publishers)));

  auto closed = (*session)->close_owner_outputs(VertexId{1U});
  ASSERT_FALSE(closed);
  EXPECT_EQ(closed.error().code, ErrorCode::kPublicationIndeterminate);
  EXPECT_EQ(first_observed->close_count, 1U);
  EXPECT_EQ(second_observed->close_count, 0U);
  const auto progress_state =
      RuntimeChannelSessionTestAccess::process_local_progress_state(**session);
  ASSERT_TRUE(progress_state) << runtime_error_message(progress_state.error());
  EXPECT_FALSE(progress_state->tracking_valid);

  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  EXPECT_EQ(first_observed->close_count, 1U);
  EXPECT_EQ(second_observed->close_count, 1U);
  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  EXPECT_EQ(first_observed->close_count, 1U);
  EXPECT_EQ(second_observed->close_count, 1U);
#else
  GTEST_SKIP() << "exception containment requires language exception support";
#endif
}

// Catalog and inline publishers share one refresh-failure classifier. A failure before any arm
// submits preserves the current sequence, while a failure after an earlier arm submits consumes
// that authority and reports an indeterminate fanout.
// REQ: SYSTEM-003
TEST(RuntimeChannelBinding, RefreshFailureClassifiesPublicationAuthority) {
  const auto accepted = [] {
    return channel_internal::InlinePublicationResult{
        .phase = channel_internal::InlinePublicationPhase::kAccepted,
        .recipient_count = 1U,
        .error = Error{ErrorCode::kSuccess},
    };
  };

  {
    SessionFixture fixture;
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    auto publisher = std::make_unique<ScriptedInlinePublisher>(
        std::vector{accepted()}, std::vector{ErrorCode::kBackpressured, ErrorCode::kSuccess});
    auto* const observed = publisher.get();
    std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
    publishers.push_back(std::move(publisher));
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
        **session, PortId{1U}, std::move(publishers), {}, true));
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

    auto rejected = fixture.source.output.emit(std::int32_t{50});
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, ErrorCode::kBackpressured);
    ASSERT_TRUE(fixture.source.output.emit(std::int32_t{51}));
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
    EXPECT_EQ(observed->sequences, (std::vector<std::uint64_t>{1U}));
    EXPECT_EQ(observed->values, (std::vector<std::int32_t>{51}));
  }

  {
    SessionFixture fixture;
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    auto first = std::make_unique<ScriptedInlinePublisher>(std::vector{accepted(), accepted()});
    auto second = std::make_unique<ScriptedInlinePublisher>(
        std::vector{accepted()}, std::vector{ErrorCode::kBackpressured, ErrorCode::kSuccess});
    auto* const first_observed = first.get();
    auto* const second_observed = second.get();
    std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
    publishers.push_back(std::move(first));
    publishers.push_back(std::move(second));
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
        **session, PortId{1U}, std::move(publishers), {}, true));
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

    auto uncertain = fixture.source.output.emit(std::int32_t{60});
    ASSERT_FALSE(uncertain);
    EXPECT_EQ(uncertain.error().code, ErrorCode::kPublicationIndeterminate);
    ASSERT_TRUE(fixture.source.output.emit(std::int32_t{61}));
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
    EXPECT_EQ(first_observed->sequences, (std::vector<std::uint64_t>{1U, 2U}));
    EXPECT_EQ(second_observed->sequences, (std::vector<std::uint64_t>{2U}));
  }
}

// A malformed backend phase after another fanout arm submitted is indeterminate, but the
// accepted publication still consumes the sample sequence just like every other post-submit path.
// REQ: RUNTIME-019
TEST(RuntimeChannelBinding, MalformedPostSubmitPhaseConsumesPublicationSequence) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  const auto accepted = channel_internal::InlinePublicationResult{
      .phase = channel_internal::InlinePublicationPhase::kAccepted,
      .recipient_count = 1U,
      .error = Error{ErrorCode::kSuccess},
  };
  auto first = std::make_unique<ScriptedInlinePublisher>(std::vector{accepted, accepted});
  auto second = std::make_unique<ScriptedInlinePublisher>(
      std::vector{channel_internal::InlinePublicationResult{
                      // The out-of-range phase exercises malformed backend handling.
                      // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
                      .phase = static_cast<channel_internal::InlinePublicationPhase>(0xffU),
                      .recipient_count = 0U,
                      .error = Error{ErrorCode::kSuccess},
                  },
                  accepted});
  auto* const first_observed = first.get();
  auto* const second_observed = second.get();
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::move(first));
  publishers.push_back(std::move(second));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
      **session, PortId{1U}, std::move(publishers)));
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

  // Key setup: the first arm submits before the second arm returns an invalid private phase.
  auto uncertain = fixture.source.output.emit(std::int32_t{70});
  ASSERT_FALSE(uncertain);
  EXPECT_EQ(uncertain.error().code, ErrorCode::kPublicationIndeterminate);
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{71}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  // Key assertion: the retry uses sequence two on every fanout arm.
  EXPECT_EQ(first_observed->sequences, (std::vector<std::uint64_t>{1U, 2U}));
  EXPECT_EQ(second_observed->sequences, (std::vector<std::uint64_t>{1U, 2U}));
}

// Unknown post-submit attribution permanently disables delivery-count cut elision. A later
// numerical equality cannot prove that the uncertain transport observation was fully ingested.
// REQ: RUNTIME-029
TEST(RuntimeChannelBinding, PublicationIndeterminateDisablesActivationCutSkip) {
  SessionFixture fixture;
  const auto counters = std::make_shared<RuntimePhaseCounters>();
  fixture.plan.executable_factories.phase_counters = counters;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto publisher = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{
          channel_internal::InlinePublicationResult{
              .phase = channel_internal::InlinePublicationPhase::kPostSubmitUncertain,
              .recipient_count = 0U,
              .error = Error{ErrorCode::kPublicationIndeterminate},
          },
      });
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::move(publisher));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
      **session, PortId{1U}, std::move(publishers)));
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

  // Key setup: the backend cannot prove whether this process-local sample became observable.
  auto uncertain = fixture.source.output.emit(std::int32_t{61});
  ASSERT_FALSE(uncertain);
  EXPECT_EQ(uncertain.error().code, ErrorCode::kPublicationIndeterminate);
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, false));
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->flush_process_local_activation_cut());

  const auto snapshot = counters->snapshot();
  const auto count = [&](RuntimePhaseCounter counter) {
    return snapshot[RuntimePhaseCounters::index(counter)];
  };
  // Key assertion: uncertain attribution forces the ordinary serialized drain for the session.
  EXPECT_EQ(count(RuntimePhaseCounter::kChannelActivationCutDrains), 1U);
  EXPECT_EQ(count(RuntimePhaseCounter::kChannelActivationCutSkips), 0U);
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// An accepted backend phase is still post-submit indeterminate unless its exact delivered count
// matches the compiler-frozen fanout. Both zero delivery and partial delivery consume the output
// sequence because Runtime cannot prove that no recipient observed the attempted publication.
// REQ: CHAN-067
// REQ: CHAN-108
// REQ: CHAN-076
// REQ: RUNTIME-019
// REQ: RUNTIME-037
TEST(RuntimeChannelBinding, RejectsAcceptedPublicationWithIncompleteRecipientCount) {
  for (const std::uint32_t delivered : {0U, 1U}) {
    SCOPED_TRACE(delivered);
    SessionFixture fixture;
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    auto publisher = std::make_unique<ScriptedInlinePublisher>(
        std::vector<channel_internal::InlinePublicationResult>{
            channel_internal::InlinePublicationResult{
                .phase = channel_internal::InlinePublicationPhase::kAccepted,
                .recipient_count = delivered,
                .error = Error{ErrorCode::kSuccess},
            },
            channel_internal::InlinePublicationResult{
                .phase = channel_internal::InlinePublicationPhase::kAccepted,
                .recipient_count = 2U,
                .error = Error{ErrorCode::kSuccess},
            },
        });
    auto* const observed = publisher.get();
    std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
    publishers.push_back(std::move(publisher));
    const std::array<std::uint32_t, 1U> expected_recipients{2U};
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
        **session, PortId{1U}, std::move(publishers), expected_recipients));
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

    auto incomplete = fixture.source.output.emit(std::int32_t{40});
    ASSERT_FALSE(incomplete);
    // Key assertion: the accepted strict subset is quantified and still consumes sequence
    // authority, so the next call advances rather than retrying sequence one.
    EXPECT_EQ(incomplete.error().code, ErrorCode::kPublicationPartial);
    EXPECT_EQ(incomplete.error().message,
              fmt::format("PUBLICATION_PARTIAL delivered={} expected=2", delivered));
    ASSERT_TRUE(fixture.source.output.emit(std::int32_t{41}));
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

    EXPECT_EQ(observed->sequences, (std::vector<std::uint64_t>{1U, 2U}));
    EXPECT_EQ(observed->values, (std::vector<std::int32_t>{40, 41}));
  }
}

// Once one fanout endpoint accepts a publication, a later endpoint's rejection is indeterminate
// to the caller. Runtime consumes sequence and loan authority exactly once and never retries.
// REQ: CHAN-076
// REQ: CHAN-101
// REQ: RUNTIME-019
TEST(RuntimeChannelBinding, ConsumesGenericLoanAfterIndeterminateFanoutPublication) {
  SessionFixture fixture;
  fixture.plan.generic_output_loans.front().max_live_loans = 1U;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto first_publisher = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{
          channel_internal::InlinePublicationResult{
              .phase = channel_internal::InlinePublicationPhase::kAccepted,
              .recipient_count = 1U,
              .error = Error{ErrorCode::kSuccess},
          },
          channel_internal::InlinePublicationResult{
              .phase = channel_internal::InlinePublicationPhase::kAccepted,
              .recipient_count = 1U,
              .error = Error{ErrorCode::kSuccess},
          },
      });
  auto second_publisher = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{
          channel_internal::InlinePublicationResult{
              .phase = channel_internal::InlinePublicationPhase::kPreSubmitRejected,
              .recipient_count = 0U,
              .error = Error{ErrorCode::kBackpressured},
          },
          channel_internal::InlinePublicationResult{
              .phase = channel_internal::InlinePublicationPhase::kAccepted,
              .recipient_count = 1U,
              .error = Error{ErrorCode::kSuccess},
          },
      });
  auto* first_observed = first_publisher.get();
  auto* second_observed = second_publisher.get();
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::move(first_publisher));
  publishers.push_back(std::move(second_publisher));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
      **session, PortId{1U}, std::move(publishers)));
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

  auto loan = fixture.source.output.allocate();
  ASSERT_TRUE(loan) << runtime_error_message(loan.error());
  loan->value() = 32;
  auto uncertain = fixture.source.output.emit(std::move(*loan));
  ASSERT_FALSE(uncertain);
  EXPECT_EQ(uncertain.error().code, ErrorCode::kPublicationIndeterminate);
  auto replacement = fixture.source.output.allocate();
  ASSERT_TRUE(replacement) << runtime_error_message(replacement.error());
  replacement->cancel();
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{33}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  EXPECT_EQ(first_observed->sequences, (std::vector<std::uint64_t>{1U, 2U}));
  EXPECT_EQ(first_observed->values, (std::vector<std::int32_t>{32, 33}));
  EXPECT_EQ(second_observed->sequences, (std::vector<std::uint64_t>{1U, 2U}));
  EXPECT_EQ(second_observed->values, (std::vector<std::int32_t>{32, 33}));
}

// REQ: RUNTIME-019
TEST(RuntimeChannelBinding, EnforcesUnpublishedLoanCreditIndependentlyOfSlotCount) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

  auto first = fixture.source.output.allocate();
  auto second = fixture.source.output.allocate();
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  auto exhausted = fixture.source.output.allocate();
  ASSERT_FALSE(exhausted);
  EXPECT_EQ(exhausted.error().code, ErrorCode::kResourceExhausted);

  first->cancel();
  auto replacement = fixture.source.output.allocate();
  ASSERT_TRUE(replacement);
  replacement->cancel();
  second->cancel();
  EXPECT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
}

// REQ: RUNTIME-019
TEST(RuntimeChannelBinding, PublishesOneAuthorityAcrossMixedPolicyFanout) {
  SessionFixture fixture;
  fixture.add_mixed_policy_fanout();
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto first_binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 8U,
                                           });
  auto second_binding =
      (*session)->create_readiness_binding(VertexId{3U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(2U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 8U,
                                           });
  ASSERT_TRUE(first_binding);
  ASSERT_TRUE(second_binding);
  EXPECT_EQ(*(*session)->readiness_capacity(VertexId{2U}), 8U);
  EXPECT_EQ(*(*session)->readiness_capacity(VertexId{3U}), 8U);
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{17}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  auto first_event = wait_for_event(**session, **first_binding);
  auto second_event = wait_for_event(**session, **second_binding);
  ASSERT_TRUE(first_event) << runtime_error_message(first_event.error());
  ASSERT_TRUE(second_event) << runtime_error_message(second_event.error());
  EXPECT_EQ(first_event->port, PortId{2U});
  EXPECT_EQ(second_event->port, PortId{3U});
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// REQ: CHAN-085
TEST(InterprocessPartitionBoundaryTest, LocalityComesFromAcceptedPlan) {
  SessionFixture fixture;
  fixture.enable_scalar(InputSelectorKind::kExact, TemporalRecordMissingKind::kSkip);
  fixture.use_cross_process_edge();

  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());

  auto readiness =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 2U,
                                           });
  ASSERT_TRUE(readiness) << runtime_error_message(readiness.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  constexpr std::array value{71};
  constexpr std::array<std::int64_t, 1U> capture_time{17};
  auto published =
      publish_scalar_script(**session, **readiness, fixture.source, value, capture_time);
  ASSERT_TRUE(published) << runtime_error_message(published.error());

  const std::array selected{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{published->records},
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selected));
  auto received = fixture.sink.input.receive_data();
  ASSERT_TRUE(received);
  EXPECT_EQ(*received, 71);
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, false));
  for (const RuntimeEvent& event : published->events) {
    ASSERT_TRUE((*readiness)->retire(event.source_sequence));
  }
  ASSERT_TRUE((*session)->retire_samples(PortId{2U}, published->records));
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// REQ: CHAN-085
TEST(InterprocessPartitionBoundaryTest, MalformedCrossProcessPlanFailsSessionConstruction) {
  {
    SessionFixture missing_subscriber;
    missing_subscriber.enable_scalar(InputSelectorKind::kExact, TemporalRecordMissingKind::kSkip);
    missing_subscriber.use_cross_process_edge();
    missing_subscriber.plan.cross_process_channel_subscribers.clear();
    auto rejected =
        RuntimeChannelSession::create(missing_subscriber.plan, missing_subscriber.replay);
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, ErrorCode::kInvalidArgument);
  }
  {
    SessionFixture shared_groups;
    shared_groups.enable_scalar(InputSelectorKind::kExact, TemporalRecordMissingKind::kSkip);
    shared_groups.use_cross_process_edge();
    shared_groups.plan.cross_process_channel_services.front().target_process_group =
        shared_groups.plan.cross_process_channel_services.front().source_process_group;
    auto rejected = RuntimeChannelSession::create(shared_groups.plan, shared_groups.replay);
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, ErrorCode::kInvalidArgument);
  }
  {
    SessionFixture nonzero_history;
    nonzero_history.enable_scalar(InputSelectorKind::kExact, TemporalRecordMissingKind::kSkip);
    nonzero_history.use_cross_process_edge();
    nonzero_history.plan.cross_process_channel_services.front().publisher_history_depth = 1U;
    auto rejected = RuntimeChannelSession::create(nonzero_history.plan, nonzero_history.replay);
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, ErrorCode::kInvalidArgument);
  }
}

}  // namespace
}  // namespace holoscan::detail
