/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <semaphore>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <holomq/errors.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/graph.hpp>
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

namespace holoscan::detail {
namespace {

using namespace std::chrono_literals;

using namespace runtime_channel_binding_test;

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

// Process ids and per-process session ordinals repeat across PID namespaces that share IPC, so
// they alone cannot keep two sessions' HoloMQ names apart; the incarnation token must. Two
// sessions of one plan in this process must also use disjoint process-local transport names.
// REQ: CHAN-151
TEST(RuntimeChannelBinding, ProcessLocalTransportScopeSeparatesEqualProcessAndOrdinal) {
  const std::string first =
      RuntimeChannelSessionTestAccess::process_local_transport_scope(42U, 1U, 0x1234U);
  const std::string second =
      RuntimeChannelSessionTestAccess::process_local_transport_scope(42U, 1U, 0x5678U);
  EXPECT_EQ(first, "session.42.1.1234");
  EXPECT_NE(first, second);
  EXPECT_EQ(RuntimeChannelSessionTestAccess::process_local_transport_scope(42U, 1U, 0x1234U),
            first);

  SessionFixture fixture;
  ASSERT_FALSE(fixture.plan.process_local_channel_services.empty());
  const std::string& service_id = fixture.plan.process_local_channel_services.front().service_id;
  auto one = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(one) << runtime_error_message(one.error());
  auto two = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(two) << runtime_error_message(two.error());
  const auto one_name =
      RuntimeChannelSessionTestAccess::process_local_transport_name(**one, service_id);
  const auto two_name =
      RuntimeChannelSessionTestAccess::process_local_transport_name(**two, service_id);
  ASSERT_TRUE(one_name) << runtime_error_message(one_name.error());
  ASSERT_TRUE(two_name) << runtime_error_message(two_name.error());
  EXPECT_NE(*one_name, *two_name);
  EXPECT_TRUE(one_name->starts_with(service_id + ".session."));
  EXPECT_TRUE(two_name->starts_with(service_id + ".session."));
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

  SessionFixture oversized_process_local;
  auto& oversized_service = oversized_process_local.plan.process_local_channel_services.front();
  oversized_service.payload_size_bytes = std::numeric_limits<std::uint64_t>::max();
  auto oversized_payload = process_local_service_options(oversized_service);
  ASSERT_FALSE(oversized_payload);
  EXPECT_EQ(oversized_payload.error().code, ErrorCode::kInvalidArgument);

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
  auto terminal = binding->terminal_status();
  ASSERT_FALSE(terminal);
  EXPECT_EQ(terminal.error().code, ErrorCode::kClosed);
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

  EXPECT_TRUE((*binding)->terminal_status());
  (*binding)->cancel();

  EXPECT_TRUE((*binding)->closed());
  EXPECT_TRUE((*binding)->terminal_status());
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
  // The observed receives below read each sample from transport, so the pump delivers instead of
  // emit().
  const ExecutionPlan accepted_plan = compile_with_process_local_pump_delivery(graph, {});
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

// A drop-oldest producer never waits for capacity while it publishes, so a consumer's retirement
// posts it no wake. Once the producer starts its terminal close, the same retirement wakes it at
// once: the close can wait for consumers to release what the producer published, and no fallback
// timeout is needed to notice the release. Each retirement step may post the wake; the owner's
// wake state coalesces them.
// REQ: RUNTIME-025
TEST(RuntimeChannelBinding, DropOldestRetirementWakesTheProducerOnlyOnceItIsClosing) {
  SessionFixture fixture;
  fixture.use_one_slot_drop_oldest_queue();
  fixture.use_direct_retained_delivery();
  struct WakeProbe {
    std::vector<VertexId> owners;
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
                static_cast<WakeProbe*>(state)->owners.push_back(owner);
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
  const auto producer_wakes = [&]() { return std::ranges::count(wake.owners, VertexId{1U}); };
  // Consume the published sample in sink activation `generation`, then retire it.
  const auto consume_and_retire = [&](const std::int32_t value, const std::uint64_t generation) {
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
    ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, generation, selection));
    {
      auto received = fixture.sink.input.receive();
      ASSERT_TRUE(received) << runtime_error_message(received.error());
      EXPECT_EQ(received->data, value);
    }
    ASSERT_TRUE((*binding)->retire(event->source_sequence));
    ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, generation, true));
    ASSERT_TRUE((*session)->retire_samples(PortId{2U}, selected));
  };

  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{41}));
  ASSERT_NO_FATAL_FAILURE(consume_and_retire(41, 1U));
  EXPECT_EQ(producer_wakes(), 0);
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{42}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  (*session)->note_owner_terminal_close(VertexId{1U});
  ASSERT_NO_FATAL_FAILURE(consume_and_retire(42, 2U));
  // Key assertion: once the close started, the consumer's retirement wakes the producer.
  EXPECT_GT(producer_wakes(), 0);
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

// This session-level test selects cross-process transport within one OS process and checks
// locality-driven withdrawal. The local sibling proves that the script stamped the coordinate.
// PublicationCoordinateTest covers compiled default clocks in separate exec-isolated workers.
// REQ: RUNTIME-057
TEST(RuntimeChannelBinding, CrossProcessPublicationCoordinateIsWithdrawnBeforeTypedReceive) {
  SessionFixture fixture;
  fixture.enable_scalar(InputSelectorKind::kExact, TemporalRecordMissingKind::kSkip);
  fixture.use_cross_process_edge();

  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());

  // A manual clock reads a fixed nonzero instant, which keeps the stamped coordinate
  // distinguishable from the zero the withdrawal leaves behind.
  constexpr std::int64_t kPublishInstant = 4242;
  ClockRegistryBuilder registry_builder;
  const ClockKey publish_key = ClockKey::named("publish-domain");
  ASSERT_TRUE(registry_builder.add_clock(publish_key,
                                         std::make_shared<ManualClock>(kPublishInstant),
                                         ClockCapabilityOptions{
                                             .can_read_now = true,
                                             .steady = true,
                                         }));
  ASSERT_TRUE(registry_builder.set_default(publish_key));
  auto registry = registry_builder.build();
  ASSERT_TRUE(registry) << runtime_error_message(registry.error());
  auto publish_clock = registry->resolve(publish_key);
  ASSERT_TRUE(publish_clock) << runtime_error_message(publish_clock.error());
  ASSERT_TRUE((*session)->bind_publish_clock(*publish_clock));

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
  constexpr std::int64_t kCaptureInstant = 17;
  constexpr std::array<std::int64_t, 1U> capture_time{kCaptureInstant};
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
  auto received = fixture.sink.input.receive();
  ASSERT_TRUE(received) << runtime_error_message(received.error());
  EXPECT_EQ(received->data, 71);

  // Key assertion: the publication coordinate is withdrawn as a pair.
  EXPECT_FALSE(received->metadata.publish_time().valid());
  EXPECT_EQ(received->metadata.publish_timestamp_ns, 0);
  EXPECT_FALSE(received->metadata.publish_clock_id.valid());
  // Capture time survives the same hop, which is what shows the withdrawal is confined to the
  // publication pair instead of clearing the sample's timing evidence wholesale.
  EXPECT_TRUE(received->metadata.capture_time().valid());
  EXPECT_EQ(received->metadata.capture_timestamp_ns, kCaptureInstant);
  EXPECT_EQ(received->metadata.source_clock_id, ClockId{1U});

  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, false));
  for (const RuntimeEvent& event : published->events) {
    ASSERT_TRUE((*readiness)->retire(event.source_sequence));
  }
  ASSERT_TRUE((*session)->retire_samples(PortId{2U}, published->records));
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// The sibling of the cross-process case above, holding everything except the edge's locality
// fixed. A process-local producer and its consumer share one live clock generation, so the
// coordinate is comparable on the consumer's timeline and is retained.
// REQ: RUNTIME-057
TEST(RuntimeChannelBinding, ProcessLocalPublicationCoordinateSurvivesTypedReceive) {
  SessionFixture fixture;
  fixture.enable_scalar(InputSelectorKind::kExact, TemporalRecordMissingKind::kSkip);

  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());

  constexpr std::int64_t kPublishInstant = 4242;
  ClockRegistryBuilder registry_builder;
  const ClockKey publish_key = ClockKey::named("publish-domain");
  ASSERT_TRUE(registry_builder.add_clock(publish_key,
                                         std::make_shared<ManualClock>(kPublishInstant),
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
  constexpr std::int64_t kCaptureInstant = 17;
  constexpr std::array<std::int64_t, 1U> capture_time{kCaptureInstant};
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
  auto received = fixture.sink.input.receive();
  ASSERT_TRUE(received) << runtime_error_message(received.error());
  EXPECT_EQ(received->data, 71);

  // Key assertion: the same script that reads absence across processes reads the bound domain and
  // the instant the publish clock supplied when the hop stays process-local.
  EXPECT_TRUE(received->metadata.publish_time().valid());
  EXPECT_EQ(received->metadata.publish_timestamp_ns, kPublishInstant);
  EXPECT_EQ(received->metadata.publish_clock_id, bound);
  EXPECT_TRUE(received->metadata.capture_time().valid());
  EXPECT_EQ(received->metadata.capture_timestamp_ns, kCaptureInstant);
  EXPECT_EQ(received->metadata.source_clock_id, ClockId{1U});

  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, false));
  for (const RuntimeEvent& event : published->events) {
    ASSERT_TRUE((*readiness)->retire(event.source_sequence));
  }
  ASSERT_TRUE((*session)->retire_samples(PortId{2U}, published->records));
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// A capture-less sample on a side-only input takes its publication coordinate as its readiness
// coordinate, whether its owner polls it late or with a time snapshot read before it was stored.
// An independent trigger at T therefore observes exactly the samples published at or before T.
// Without the ordering the poll time is the coordinate.
// REQ: CHAN-131
// REQ: RUNTIME-065
TEST(RuntimeChannelBinding, PublicationOrderedCaptureAbsentArrivalTakesItsPublicationCoordinate) {
  constexpr std::int64_t kPublishInstant = 4242;
  struct Vector {
    bool ordered{};
    std::int64_t poll_instant{};
  };
  constexpr std::array vectors{
      Vector{false, 10'000},
      Vector{true, 10'000},  // polled long after the publication
      Vector{false, 1'000},
      Vector{true, 1'000},  // the poll's snapshot predates the publication
  };
  for (const Vector& vector : vectors) {
    const bool ordered = vector.ordered;
    SCOPED_TRACE(testing::Message() << "ordered=" << ordered << " poll=" << vector.poll_instant);
    ClockRegistryBuilder registry_builder;
    const ClockKey publish_key = ClockKey::named("publish-domain");
    ASSERT_TRUE(registry_builder.add_clock(publish_key,
                                           std::make_shared<ManualClock>(kPublishInstant),
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

    SessionFixture fixture;
    fixture.enable_drain_all(1U);
    fixture.use_direct_retained_delivery();
    fixture.plan.input_selections.front().reference_clock = bound;
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    ASSERT_TRUE((*session)->bind_publish_clock(*publish_clock));
    if (ordered) {
      auto applied = (*session)->order_capture_absent_arrivals_by_publication(PortId{2U});
      ASSERT_TRUE(applied) << runtime_error_message(applied.error());
      ASSERT_TRUE(*applied);
    }
    auto binding =
        (*session)->create_readiness_binding(VertexId{2U},
                                             RuntimePortBindingRequest{
                                                 .plan_generation = fixture.plan.plan_generation,
                                                 .run_generation = 1U,
                                                 .kind = RuntimeEventKind::kInputArrival,
                                                 .source = make_source_id(1U),
                                                 .source_clock = bound,
                                                 .reference_clock = bound,
                                                 .capacity = 2U,
                                             });
    ASSERT_TRUE(binding) << runtime_error_message(binding.error());
    ASSERT_TRUE((*session)->refresh_connections());
    ASSERT_TRUE((*session)->arm());

    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
    ASSERT_TRUE(fixture.source.output.emit(std::int32_t{5}));
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

    ASSERT_TRUE(
        (*binding)->poll(TimePoint{.timestamp_ns = vector.poll_instant, .clock_id = bound}));
    RuntimeEvent event;
    auto drained = (*binding)->drain(std::span<RuntimeEvent>{&event, 1U});
    ASSERT_TRUE(drained) << runtime_error_message(drained.error());
    ASSERT_EQ(*drained, 1U);
    // Key assertion: an ordered input publishes the publication coordinate, not the poll time,
    // on either side of it.
    EXPECT_EQ(event.reference_time.timestamp_ns, ordered ? kPublishInstant : vector.poll_instant);
    EXPECT_EQ(event.reference_time.clock_id, bound);

    ASSERT_TRUE((*binding)->retire(event.source_sequence));
    (*session)->cancel();
    ASSERT_TRUE((*session)->join());
  }
}

// A trigger at T drains the side samples published at or before T and none published after it,
// even when its owner polls them late with a time snapshot read before either was stored.
// REQ: CHAN-131
// REQ: RUNTIME-065
TEST(RuntimeChannelBinding, PublicationOrderedArrivalsStayOnTheirSideOfATrigger) {
  constexpr std::int64_t kTriggerInstant = 200;
  constexpr std::int64_t kBeforeTrigger = 190;
  constexpr std::int64_t kAfterTrigger = 210;
  constexpr std::int64_t kPollSnapshot = 180;
  ClockRegistryBuilder registry_builder;
  const ClockKey publish_key = ClockKey::named("publish-domain");
  const auto manual = std::make_shared<ManualClock>(kBeforeTrigger);
  ASSERT_TRUE(registry_builder.add_clock(publish_key,
                                         manual,
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

  SessionFixture fixture;
  fixture.enable_drain_all(2U);
  fixture.use_two_slot_drop_oldest_queue();
  fixture.use_direct_retained_delivery();
  fixture.plan.input_selections.front().reference_clock = bound;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE((*session)->bind_publish_clock(*publish_clock));
  auto applied = (*session)->order_capture_absent_arrivals_by_publication(PortId{2U});
  ASSERT_TRUE(applied) << runtime_error_message(applied.error());
  ASSERT_TRUE(*applied);
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = bound,
                                               .reference_clock = bound,
                                               .capacity = 4U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{1}));
  manual->advance_to(kAfterTrigger);
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{2}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  // The owner read its clock before either publication, then polled after both.
  ASSERT_TRUE((*binding)->poll(TimePoint{.timestamp_ns = kPollSnapshot, .clock_id = bound}));
  std::array<RuntimeEvent, 2> events{};
  auto drained = (*binding)->drain(std::span<RuntimeEvent>{events});
  ASSERT_TRUE(drained) << runtime_error_message(drained.error());
  ASSERT_EQ(*drained, 2U);
  // Key assertion: the witnesses keep their publication coordinates, one on each side of the
  // trigger, so the trigger at T admits the earlier sample and not the later one.
  EXPECT_EQ(events[0].reference_time.timestamp_ns, kBeforeTrigger);
  EXPECT_EQ(events[1].reference_time.timestamp_ns, kAfterTrigger);
  EXPECT_LE(events[0].reference_time.timestamp_ns, kTriggerInstant);
  EXPECT_GT(events[1].reference_time.timestamp_ns, kTriggerInstant);

  for (const RuntimeEvent& event : events) {
    ASSERT_TRUE((*binding)->retire(event.source_sequence));
  }
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
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
