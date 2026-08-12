/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <fmt/format.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <future>
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
#include <holoscan/time/manual_clock.hpp>

#include "common/allocation_counter.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "core/source_id_internal.hpp"
#include "runtime/endpoint_clock_health_runtime.hpp"
#include "runtime/runtime_channel_binding.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "time/clock_registry_internal.hpp"

#include "runtime_channel_binding_test_support.hpp"

namespace holoscan::detail {
namespace {

using namespace std::chrono_literals;
using namespace runtime_channel_binding_test;

// Hold one transport receive open so the lock-boundary regression can observe whether the
// independent typed-retention cache remains available to another controller thread.
class BlockingInlineSubscriber final : public channel_internal::InlinePayloadSubscriber {
 public:
  [[nodiscard]] expected<bool, Error> has_samples() const override { return true; }

  [[nodiscard]] bool has_publishers() const noexcept override { return true; }

  [[nodiscard]] expected<SampleMetadata, Error> receive(
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
    return SampleMetadata{
        .sequence = sequence.load(std::memory_order_acquire),
        .byte_size = sizeof(received_value),
        .payload_kind = PayloadKind::kInlineTyped,
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
    // Pending ingress has not entered typed Runtime retention and owns no arrival identity yet.
    EXPECT_EQ(retained->next_arrival_sequence, 2U);

    // Key assertion: capacity-one kDropOldest replaces the pending unread survivor while the
    // independently claimed typed sample remains in place, without creating a direct-only arrival
    // gap or spuriously backpressuring the publication.
    ASSERT_TRUE(fixture.source.output.emit(std::int32_t{33}));
    const auto replaced = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    ASSERT_TRUE(replaced) << runtime_error_message(replaced.error());
    EXPECT_EQ(replaced->occupied_samples, 1U);
    EXPECT_EQ(replaced->pending_direct_ingress_samples, 1U);
    EXPECT_EQ(replaced->next_arrival_sequence, 2U);
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
    (*session)->cancel();
    EXPECT_TRUE((*session)->join());
  }
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
    EXPECT_EQ(survivor_record->arrival_sequence, 2U);
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
  SessionFixture fixture;
  fixture.plan.generic_output_loans.front().max_live_loans = 1U;
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
  auto still_held = fixture.source.output.allocate();
  ASSERT_FALSE(still_held);
  EXPECT_EQ(still_held.error().code, ErrorCode::kResourceExhausted);

  ASSERT_TRUE(fixture.source.output.emit(std::move(*loan)));
  auto replacement = fixture.source.output.allocate();
  ASSERT_TRUE(replacement) << runtime_error_message(replacement.error());
  replacement->cancel();
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  EXPECT_EQ(observed->sequences, (std::vector<std::uint64_t>{1U, 1U}));
  EXPECT_EQ(observed->values, (std::vector<std::int32_t>{22, 22}));
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
