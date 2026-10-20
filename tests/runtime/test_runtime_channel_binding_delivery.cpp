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
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <semaphore>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holomq/errors.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "channel/channel_common.hpp"
#include "common/allocation_counter.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/process_placement_internal.hpp"
#include "core/operator_spec_compiler_access.hpp"
#include "core/source_id_internal.hpp"
#include "runtime/bounded_lock.hpp"
#include "runtime/endpoint_clock_health_runtime.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/runtime_channel_binding.hpp"
#include "runtime/runtime_channel_storage.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_rate_state.hpp"
#include "time/clock_registry_internal.hpp"

#include "runtime_channel_binding_test_support.hpp"

namespace holoscan::detail {
namespace {

using namespace std::chrono_literals;

using namespace runtime_channel_binding_test;

// Wait, for a few seconds at most, until an instrumented session has counted `target` events.
// Returns whether it did.
[[nodiscard]] bool wait_for_phase_count(const RuntimePhaseCounters& counters,
                                        const RuntimePhaseCounter counter,
                                        const std::uint64_t target = 1U) noexcept {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
  while (counters.snapshot()[RuntimePhaseCounters::index(counter)] < target &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::microseconds{100});
  }
  return counters.snapshot()[RuntimePhaseCounters::index(counter)] >= target;
}

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

// Concurrent admission keeps one authoritative count of blocked heads. The reviewed schedule
// (one subscriber's clear overlapping another's block and clear) and concurrent transitions on two
// subscribers, each under its own transport mutex, must leave no retry signal once every head is
// admitted: terminal input retirement refuses to complete while that signal remains set.
// REQ: CHAN-136
TEST(RuntimeChannelBinding, ScopedBlockedHeadSignalClearsAfterConcurrentTransitions) {
  SessionFixture fixture;
  fixture.add_blocking_fanout();
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::enable_concurrent_admission(**session));
  const PortId first{2U};
  const PortId second{3U};
  const auto set_blocked = [&](const PortId input, const bool blocked) {
    return RuntimeChannelSessionTestAccess::set_transport_head_blocked(**session, input, blocked);
  };

  // Key setup: the first head blocks and clears while the second blocks and clears.
  ASSERT_TRUE(set_blocked(first, true));
  EXPECT_TRUE(RuntimeChannelSessionTestAccess::blocked_transport_retry_pending(**session));
  ASSERT_TRUE(set_blocked(first, false));
  ASSERT_TRUE(set_blocked(second, true));
  EXPECT_TRUE(RuntimeChannelSessionTestAccess::blocked_transport_retry_pending(**session));
  ASSERT_TRUE(set_blocked(second, false));
  // Key assertion: repeated clear transitions are idempotent and leave no stale signal.
  ASSERT_TRUE(set_blocked(first, false));
  EXPECT_FALSE(RuntimeChannelSessionTestAccess::blocked_transport_retry_pending(**session));

  // Key setup: race block/clear transitions on both subscribers from two threads.
  constexpr std::uint32_t kIterations = 100'000U;
  std::atomic<bool> start{false};
  std::atomic<std::uint32_t> failures{0U};
  const auto toggle = [&](const PortId input) {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    for (std::uint32_t iteration = 0U; iteration < kIterations; ++iteration) {
      if (!set_blocked(input, true) || !set_blocked(input, false)) {
        failures.fetch_add(1U, std::memory_order_relaxed);
      }
    }
  };
  std::thread first_thread{toggle, first};
  std::thread second_thread{toggle, second};
  start.store(true, std::memory_order_release);
  first_thread.join();
  second_thread.join();
  EXPECT_EQ(failures.load(), 0U);
  // Key assertion: both heads are clear, so no retry or terminal-retirement decision may see one.
  EXPECT_FALSE(RuntimeChannelSessionTestAccess::blocked_transport_retry_pending(**session));
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

// A failed session closes every binding, but a parked owner sees that only when it next runs. The
// first failure therefore wakes every owner once; a later failure neither replaces the cause nor
// signals again.
// REQ: RUNTIME-041
// REQ: RUNTIME-087
TEST(RuntimeChannelBinding, SessionFailureWakesEveryOwnerOnce) {
  SessionFixture fixture;
  struct WakeProbe {
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
          .signal_owner_progress = [](void*, VertexId) noexcept {},
      });
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  const std::uint32_t before = wake.generic_signals.load(std::memory_order_relaxed);

  // Key assertion: the first failure wakes every owner once, and a second one does not.
  RuntimeChannelSessionTestAccess::fail(**session, ErrorCode::kProtocolError);
  EXPECT_EQ(wake.generic_signals.load(std::memory_order_relaxed) - before, 1U);
  RuntimeChannelSessionTestAccess::fail(**session, ErrorCode::kFailure);
  EXPECT_EQ(wake.generic_signals.load(std::memory_order_relaxed) - before, 1U);
  const auto status = (*session)->status();
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().code, ErrorCode::kProtocolError);

  (*session)->cancel();
  static_cast<void>((*session)->join());
}

// A direct edge stores its terminal record straight into the consumer's cache, and no receive
// follows to wake the consumer. Closing the producer must wake the consumer's owner itself rather
// than leave the record to the owner's bounded fallback poll.
// REQ: RUNTIME-025
// REQ: RUNTIME-068
TEST(RuntimeChannelBinding, DirectEndOfStreamSignalsItsConsumerOwner) {
  SessionFixture fixture;
  fixture.use_direct_retained_delivery();
  struct WakeProbe {
    std::atomic<std::uint32_t> consumer_signals{};
    std::atomic<std::uint32_t> other_signals{};
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
                auto* const probe = static_cast<WakeProbe*>(state);
                auto& signals = owner.value == 2U ? probe->consumer_signals : probe->other_signals;
                signals.fetch_add(1U, std::memory_order_relaxed);
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
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{42}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  auto event = wait_for_event(**session, **binding);
  ASSERT_TRUE(event) << runtime_error_message(event.error());
  ASSERT_TRUE((*binding)->retire(event->source_sequence));
  // Retiring the blocking edge's sample may already have woken the producer for capacity.
  const std::uint32_t data_signals = wake.consumer_signals.load(std::memory_order_relaxed);
  const std::uint32_t other_signals = wake.other_signals.load(std::memory_order_relaxed);

  // Key assertion: the terminal store itself signals the consumer's owner exactly once.
  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  EXPECT_EQ(wake.consumer_signals.load(std::memory_order_relaxed) - data_signals, 1U);
  EXPECT_EQ(wake.other_signals.load(std::memory_order_relaxed), other_signals);
  auto terminal = wait_for_event(**session, **binding);
  ASSERT_TRUE(terminal) << runtime_error_message(terminal.error());
  EXPECT_EQ(terminal->kind, RuntimeEventKind::kInputEndOfStream);

  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// Data on a side-only input cannot activate its owner. Admission still advances the input's
// readiness revision, so the owner's next scan ingests the sample, but neither a direct store nor
// a transported receive wakes the owner for it. End of stream still wakes it. A blocking edge is
// never side-only, because ingesting its sample can be what frees the producer.
// REQ: RUNTIME-068
TEST(RuntimeChannelBinding, SideOnlyDataAdvancesRevisionWithoutOwnerHint) {
  struct Vector {
    bool direct{};
    bool drop_oldest{};
  };
  constexpr std::array vectors{
      Vector{true, true},
      Vector{false, true},
      Vector{true, false},
      Vector{false, false},
  };
  for (const Vector& vector : vectors) {
    SCOPED_TRACE(testing::Message()
                 << "direct=" << vector.direct << " drop_oldest=" << vector.drop_oldest);
    SessionFixture fixture;
    fixture.enable_drain_all(2U);
    if (vector.drop_oldest) {
      fixture.use_two_slot_drop_oldest_queue();
    } else {
      fixture.use_two_slot_blocking_queue();
    }
    if (vector.direct) {
      fixture.use_direct_retained_delivery();
    }
    const auto counters = std::make_shared<RuntimePhaseCounters>();
    fixture.plan.executable_factories.phase_counters = counters;
    struct WakeProbe {
      std::atomic<std::uint32_t> consumer_signals{};
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
                  if (owner.value == 2U) {
                    static_cast<WakeProbe*>(state)->consumer_signals.fetch_add(
                        1U, std::memory_order_acq_rel);
                  }
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
    auto marked = (*session)->mark_side_only_input(PortId{2U}, false);
    ASSERT_TRUE(marked) << runtime_error_message(marked.error());
    EXPECT_EQ(*marked, vector.drop_oldest);
    const auto elided = [&] {
      return counters->snapshot()[RuntimePhaseCounters::index(
          RuntimePhaseCounter::kChannelSideOnlyDataHintsElided)];
    };
    // The pump may finish its receive after the binding already observes the sample.
    const auto eventually = [](const auto& done) {
      const auto deadline = std::chrono::steady_clock::now() + 2s;
      while (!done() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(1ms);
      }
      return done();
    };

    // Key setup: one data sample, stored directly or received by the pump or activation cut.
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
    ASSERT_TRUE(fixture.source.output.emit(std::int32_t{42}));
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
    ASSERT_TRUE((*session)->flush_process_local_activation_cut());
    auto event = wait_for_event(**session, **binding);
    ASSERT_TRUE(event) << runtime_error_message(event.error());
    EXPECT_EQ(event->kind, RuntimeEventKind::kInputArrival);
    ASSERT_TRUE((*binding)->retire(event->source_sequence));

    // Key assertion: the owner's scan saw the sample, but only a blocking edge woke it for it.
    if (vector.drop_oldest) {
      EXPECT_TRUE(eventually([&] { return elided() == 1U; }));
      EXPECT_EQ(wake.consumer_signals.load(std::memory_order_acquire), 0U);
    } else {
      EXPECT_TRUE(
          eventually([&] { return wake.consumer_signals.load(std::memory_order_acquire) == 1U; }));
      EXPECT_EQ(elided(), 0U);
    }
    const std::uint32_t data_signals = wake.consumer_signals.load(std::memory_order_acquire);

    // Key assertion: end of stream wakes the owner exactly once on every edge.
    ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
    ASSERT_TRUE((*session)->flush_process_local_activation_cut());
    auto terminal = wait_for_event(**session, **binding);
    ASSERT_TRUE(terminal) << runtime_error_message(terminal.error());
    EXPECT_EQ(terminal->kind, RuntimeEventKind::kInputEndOfStream);
    EXPECT_TRUE(eventually([&] {
      return wake.consumer_signals.load(std::memory_order_acquire) == data_signals + 1U;
    }));
    EXPECT_EQ(elided(), vector.drop_oldest ? 1U : 0U);

    (*session)->cancel();
    ASSERT_TRUE((*session)->join());
  }
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
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelActivationCutsPostCompute), 1U);
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelActivationCutDrains) +
                phase(RuntimePhaseCounter::kChannelActivationCutSkips),
            1U);
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelActivationCutDrains),
            phase(RuntimePhaseCounter::kChannelActivationCutDrainsPostCompute));
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
  // HoloMQ's own blocking pressure is under test, so turn off the credit check that prevents it.
  fixture.disable_publication_credits();
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

// Source-tree phase counters observe the accepted component boundaries of one complete scalar
// Channel cycle without changing the installed Runtime or Channel API.
// REQ: RUNTIME-011
// REQ: RUNTIME-029
// REQ: RUNTIME-063
// REQ: RUNTIME-078
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
// REQ: RUNTIME-063
// REQ: RUNTIME-078
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
// REQ: RUNTIME-063
// REQ: RUNTIME-078
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
// REQ: RUNTIME-063
// REQ: RUNTIME-078
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

// A transport head that typed retention refuses is staged in its endpoint, so the pump's wait sees
// only the samples queued behind it, and each of those would wake the pump for a head it still
// cannot store. Once the pump's wait has reported such a sample and its spin has passed, Runtime
// hides those samples from the wait until a release on the input wakes the pump. A head still
// within its spin, and the private A/B control, keep the input visible on every wait.
// REQ: RUNTIME-090
TEST(RuntimeChannelBinding, PumpWaitHidesABlockedHeadUntilAReleaseWakesIt) {
  enum class Arm : std::uint8_t { kSerial, kConcurrent, kSpinning, kWithoutWake };
  for (const Arm arm : {Arm::kSerial, Arm::kConcurrent, Arm::kSpinning, Arm::kWithoutWake}) {
    SCOPED_TRACE(::testing::Message() << "arm=" << static_cast<int>(arm));
    SessionFixture fixture;
    fixture.use_two_slot_blocking_queue();
    const bool hides = arm == Arm::kSerial || arm == Arm::kConcurrent;
    if (arm == Arm::kWithoutWake) {
      fixture.disable_blocked_head_wake();
    } else {
      fixture.set_blocked_head_spin(hides ? 0ns : std::chrono::nanoseconds{1h});
    }
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
                                                 .capacity = 4U,
                                             });
    ASSERT_TRUE(binding) << runtime_error_message(binding.error());
    ASSERT_TRUE((*session)->refresh_connections());
    if (arm == Arm::kConcurrent) {
      ASSERT_TRUE(RuntimeChannelSessionTestAccess::enable_concurrent_admission(**session));
    }
    const auto phase = [&](const RuntimePhaseCounter counter) {
      return counters->snapshot()[RuntimePhaseCounters::index(counter)];
    };
    const auto emit = [&](const std::int32_t value) {
      EXPECT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
      auto emitted = fixture.source.output.emit(value);
      EXPECT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
      return emitted;
    };
    const auto masked = [&] {
      auto state = RuntimeChannelSessionTestAccess::blocked_head_masked(**session, PortId{2U});
      return state.has_value() && *state;
    };

    // Key setup: fill both typed slots, stage a third sample that cannot be stored, and queue a
    // fourth behind it.
    for (const std::int32_t value : {1, 2}) {
      ASSERT_TRUE(emit(value));
      ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
    }
    ASSERT_TRUE(emit(3));
    auto blocked = RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U});
    ASSERT_FALSE(blocked);
    ASSERT_EQ(blocked.error().code, ErrorCode::kBackpressured);
    // Nothing is queued behind the head yet, so the pump never woke for it and it stays visible.
    EXPECT_FALSE(masked());
    ASSERT_TRUE(emit(4));

    // Key setup: the pump's wait reports the fourth sample, and the pump probes the refused head.
    auto reported = RuntimeChannelSessionTestAccess::wait_as_pump(**session, PortId{2U}, 0ns);
    ASSERT_TRUE(reported) << runtime_error_message(reported.error());
    EXPECT_TRUE(reported->input_ready);
    auto probed = RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U});
    ASSERT_FALSE(probed);
    ASSERT_EQ(probed.error().code, ErrorCode::kBackpressured);
    EXPECT_EQ(masked(), hides);
    EXPECT_EQ(phase(RuntimePhaseCounter::kChannelBlockedHeadsMasked), hides ? 1U : 0U);

    // Wait as the pump does until the waits run out. Each publication's notification can end one
    // wait, so four publications bound the count.
    std::size_t waits = 0U;
    std::size_t input_reports = 0U;
    bool quiet = false;
    while (waits < 8U && !quiet) {
      auto waited = RuntimeChannelSessionTestAccess::wait_as_pump(**session, PortId{2U}, 0ns);
      ASSERT_TRUE(waited) << runtime_error_message(waited.error());
      ++waits;
      input_reports += waited->input_ready ? 1U : 0U;
      quiet = !waited->ready;
    }
    if (!hides) {
      // Key assertion (control and spin): the fourth sample reports the input on every wait, so
      // the waits never run out.
      EXPECT_FALSE(quiet);
      EXPECT_EQ(input_reports, waits);
      (*session)->cancel();
      ASSERT_TRUE((*session)->join());
      continue;
    }
    // Key assertion: the hidden input is never reported, and the waits run out, so the pump would
    // block.
    EXPECT_EQ(input_reports, 0U);
    EXPECT_TRUE(quiet);
    EXPECT_LE(waits, 5U);

    // Key setup: a release on the input shows it again and wakes the pump, though its typed slots
    // are still full. A second release in the same hold finds nothing hidden and posts no wake.
    const std::array released{PortId{2U}};
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::retry_released_inputs(**session, released));
    EXPECT_FALSE(masked());
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::retry_released_inputs(**session, released));
    EXPECT_EQ(phase(RuntimePhaseCounter::kChannelPumpWakesSignaled), 1U);
    const auto started = std::chrono::steady_clock::now();
    auto woken = RuntimeChannelSessionTestAccess::wait_as_pump(**session, PortId{2U}, 10s);
    const auto elapsed = std::chrono::steady_clock::now() - started;
    ASSERT_TRUE(woken) << runtime_error_message(woken.error());
    // Key assertion: the release shows the queued fourth sample again, so a long wait reports the
    // input at once. Its readiness check ends this wait before the posted wake is read.
    EXPECT_TRUE(woken->ready);
    EXPECT_TRUE(woken->input_ready);
    EXPECT_LT(elapsed, 5s);

    // The head is still refused, so the pump's retry hides it again. That wait returned through the
    // readiness check, leaving the wake posted, so one more wait may end without the input; after
    // that the pump would block again.
    auto retried = RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U});
    ASSERT_FALSE(retried);
    EXPECT_EQ(retried.error().code, ErrorCode::kBackpressured);
    EXPECT_TRUE(masked());
    EXPECT_EQ(phase(RuntimePhaseCounter::kChannelBlockedHeadsMasked), 2U);
    std::size_t rehidden_waits = 0U;
    bool asleep = false;
    while (rehidden_waits < 3U && !asleep) {
      auto waited = RuntimeChannelSessionTestAccess::wait_as_pump(**session, PortId{2U}, 0ns);
      ASSERT_TRUE(waited) << runtime_error_message(waited.error());
      ++rehidden_waits;
      EXPECT_FALSE(waited->input_ready);
      asleep = !waited->ready;
    }
    EXPECT_TRUE(asleep);
    EXPECT_LE(rehidden_waits, 2U);
    (*session)->cancel();
    ASSERT_TRUE((*session)->join());
  }
}

// With a running pump, a consumer that holds both typed slots must not keep the pump waking for
// the head it cannot store, however long the hold lasts. Retiring one sample then wakes the pump,
// which stores the head, and the head after it is hidden in turn.
// REQ: RUNTIME-090
TEST(RuntimeChannelBinding, PumpSleepsThroughAConsumerHoldAndWakesOnItsRelease) {
  SessionFixture fixture;
  fixture.use_two_slot_blocking_queue();
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
                                               .capacity = 4U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  auto indexes = RuntimeChannelSessionTestAccess::pump_index_state(**session);
  ASSERT_TRUE(indexes) << runtime_error_message(indexes.error());
  ASSERT_TRUE(indexes->background_thread_running);
  const auto phase = [&](const RuntimePhaseCounter counter) {
    return counters->snapshot()[RuntimePhaseCounters::index(counter)];
  };
  const auto emit = [&](const std::int32_t value) {
    EXPECT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
    auto emitted = fixture.source.output.emit(value);
    EXPECT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
    return emitted;
  };
  const auto wait_for_masks = [&](const std::uint64_t count) {
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (phase(RuntimePhaseCounter::kChannelBlockedHeadsMasked) < count &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(1ms);
    }
    return phase(RuntimePhaseCounter::kChannelBlockedHeadsMasked) >= count;
  };

  // Key setup: the pump stores the first two samples, then hides the third behind the fourth.
  ASSERT_TRUE(emit(1));
  auto first_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(first_event) << runtime_error_message(first_event.error());
  ASSERT_TRUE(emit(2));
  auto second_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(second_event) << runtime_error_message(second_event.error());
  ASSERT_TRUE(emit(3));
  ASSERT_TRUE(emit(4));
  ASSERT_TRUE(wait_for_masks(1U));

  // Key assertion: holding the slots for 20 ms, then 40 ms more, adds no ready pump wakes beyond
  // a late notification of the fourth publication.
  const auto ready_before = phase(RuntimePhaseCounter::kChannelPumpWaitsReady);
  std::this_thread::sleep_for(20ms);
  const auto ready_after_short = phase(RuntimePhaseCounter::kChannelPumpWaitsReady);
  std::this_thread::sleep_for(40ms);
  const auto ready_after_long = phase(RuntimePhaseCounter::kChannelPumpWaitsReady);
  EXPECT_LE(ready_after_short - ready_before, 2U);
  EXPECT_LE(ready_after_long - ready_after_short, 2U);

  auto first_record = (*session)->temporal_sample(first_event->port, first_event->source_sequence);
  ASSERT_TRUE(first_record) << runtime_error_message(first_record.error());
  const std::array selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&*first_record, 1U},
      .batch = false,
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selection));
  auto received = fixture.sink.input.receive();
  ASSERT_TRUE(received) << runtime_error_message(received.error());
  EXPECT_EQ(received->data, 1);
  ASSERT_TRUE((*binding)->retire(first_event->source_sequence));
  ASSERT_TRUE((*session)->retire_samples(PortId{2U}, selection.front().samples));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));

  // Key assertion: the release woke the pump, which stored the third sample. The fourth is refused
  // in turn, but nothing is queued behind it to wake the pump, so it is not hidden.
  auto third_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(third_event) << runtime_error_message(third_event.error());
  EXPECT_EQ(third_event->source_sequence, 3U);
  EXPECT_GE(phase(RuntimePhaseCounter::kChannelPumpWakesSignaled), 1U);
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelBlockedHeadsMasked), 1U);

  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// A cross-process receive is not evidence that an eventless process-local publication has reached
// typed storage. The relay's post-compute cut must synchronously materialize its local output even
// while the only pump wake attachment belongs to the cross-process input.
// REQ: RUNTIME-029
// REQ: RUNTIME-063
// REQ: RUNTIME-078
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

// Unknown post-submit attribution permanently disables delivery-count cut elision. A later
// numerical equality cannot prove that the uncertain transport observation was fully ingested.
// REQ: RUNTIME-029
// REQ: RUNTIME-063
// REQ: RUNTIME-078
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

// A producer's direct store waits on the consumer's input cache mutex whenever the consumer lane
// holds it. Instrumented sessions count that wait per admission site, which is how a real-time
// producer blocked by another lane's cache shows up.
// REQ: RUNTIME-063
// REQ: RUNTIME-078
// REQ: RUNTIME-082
TEST(RuntimeChannelBinding, DirectStoreCountsAContendedInputCacheLock) {
  SessionFixture fixture;
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
                                               .capacity = 2U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

  struct Holder {
    SessionFixture* fixture{};
    const RuntimePhaseCounters* counters{};
    std::chrono::milliseconds hold{20};
    std::thread producer;
    std::atomic<bool> emitted{};
    bool contended{};
  } holder{.fixture = &fixture};
  // Hold the consumer's cache mutex while the producer's direct store runs on another thread. The
  // hold starts only once the producer has failed its first attempt, which is also when its wait
  // timing starts, so the test does not depend on when the new thread is scheduled.
  holder.counters = counters.get();
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::with_input_lock(
      **session, PortId{2U}, &holder, [](void* state) noexcept {
        auto& held = *static_cast<Holder*>(state);
        held.producer = std::thread{[&held] {
          held.emitted.store(held.fixture->source.output.emit(std::int32_t{7}).has_value());
        }};
        held.contended = wait_for_phase_count(
            *held.counters, RuntimePhaseCounter::kChannelInputCacheLockContendedDirect);
        std::this_thread::sleep_for(held.hold);
      }));
  holder.producer.join();
  // The producer counted its wait while the test still held the lock.
  EXPECT_TRUE(holder.contended);
  EXPECT_TRUE(holder.emitted.load());

  const auto snapshot = counters->snapshot();
  const auto count = [&](RuntimePhaseCounter counter) {
    return snapshot[RuntimePhaseCounters::index(counter)];
  };
  // Key assertion: exactly the direct site recorded one contended acquisition and its wait.
  EXPECT_EQ(count(RuntimePhaseCounter::kChannelInputCacheLockContendedDirect), 1U);
  EXPECT_GE(count(RuntimePhaseCounter::kChannelInputCacheLockWaitNsDirect),
            static_cast<std::uint64_t>(std::chrono::nanoseconds{holder.hold}.count()));
  EXPECT_EQ(count(RuntimePhaseCounter::kChannelInputCacheLockContendedTransport), 0U);

  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// The activation cut settles what one activation published. An owner that transported nothing
// skips it without reading session-wide progress, even while another owner's publication is still
// undelivered; the publishing owner's own cut still drains that publication.
// REQ: RUNTIME-063
// REQ: RUNTIME-078
// REQ: RUNTIME-082
TEST(RuntimeChannelBinding, ActivationCutSkipsAnOwnerThatTransportedNothing) {
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
  ASSERT_TRUE(fixture.source.output.emit(Sample<std::int32_t>{
      42, SampleMetadata{.capture_timestamp_ns = 10, .source_clock_id = ClockId{1U}}}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  const auto phase = [&](RuntimePhaseCounter counter) {
    return counters->snapshot()[RuntimePhaseCounters::index(counter)];
  };
  // Key assertion: the consumer owner, which transported nothing, skips while the producer's
  // sample is still in transport.
  ASSERT_TRUE((*session)->flush_process_local_activation_cut(VertexId{2U}));
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelActivationCutSkipsNoTransport), 1U);
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelActivationCutDrains), 0U);
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelSamplesReceived), 0U);

  // The producer's own cut drains its publication, and the skip flag was consumed by it.
  ASSERT_TRUE((*session)->flush_process_local_activation_cut(VertexId{1U}));
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelActivationCutDrains), 1U);
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelSamplesReceived), 1U);
  ASSERT_TRUE((*session)->flush_process_local_activation_cut(VertexId{1U}));
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelActivationCutSkipsNoTransport), 2U);
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelActivationCuts),
            phase(RuntimePhaseCounter::kChannelActivationCutDrains) +
                phase(RuntimePhaseCounter::kChannelActivationCutSkips));

  // A terminal close publishes outside any activation, so no owner cut settles its end of stream.
  // Key assertion: the next cut of any owner, even one that transported nothing, takes the full
  // path and drains it.
  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  const auto drains_before_close = phase(RuntimePhaseCounter::kChannelActivationCutDrains);
  ASSERT_TRUE((*session)->flush_process_local_activation_cut(VertexId{2U}));
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelActivationCutSkipsNoTransport), 2U);
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelActivationCutDrains), drains_before_close + 1U);
  // Settled: a later idle cut skips again.
  ASSERT_TRUE((*session)->flush_process_local_activation_cut(VertexId{2U}));
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelActivationCutSkipsNoTransport), 3U);

  auto event = wait_for_event(**session, **binding);
  ASSERT_TRUE(event) << runtime_error_message(event.error());
  ASSERT_TRUE((*binding)->retire(event->source_sequence));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// A transport receive reserves its typed destination under the consumer's cache mutex before it
// stores. A receive that waits there is counted at the transport site, even though the store then
// takes the mutex at once.
// REQ: RUNTIME-063
// REQ: RUNTIME-078
// REQ: RUNTIME-082
TEST(RuntimeChannelBinding, TransportReservationCountsAContendedInputCacheLock) {
  SessionFixture fixture;
  fixture.use_one_slot_blocking_transport();
  // No pump: the producer's activation cut is the receive that reserves the destination.
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
                                               .capacity = 2U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{7}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  struct Holder {
    RuntimeChannelSession* session{};
    const RuntimePhaseCounters* counters{};
    std::chrono::milliseconds hold{20};
    std::thread cut;
    std::atomic<bool> flushed{};
    bool contended{};
  } holder{.session = session->get(), .counters = counters.get()};
  // Hold the consumer's cache mutex while the cut receives on another thread. The hold starts only
  // once the receive has failed its first attempt at the mutex.
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::with_input_lock(
      **session, PortId{2U}, &holder, [](void* state) noexcept {
        auto& held = *static_cast<Holder*>(state);
        held.cut = std::thread{[&held] {
          held.flushed.store(
              held.session->flush_process_local_activation_cut(VertexId{1U}).has_value());
        }};
        held.contended = wait_for_phase_count(
            *held.counters, RuntimePhaseCounter::kChannelInputCacheLockContendedTransport);
        std::this_thread::sleep_for(held.hold);
      }));
  holder.cut.join();
  // The receive counted its wait while the test still held the lock.
  EXPECT_TRUE(holder.contended);
  EXPECT_TRUE(holder.flushed.load());

  const auto snapshot = counters->snapshot();
  const auto count = [&](RuntimePhaseCounter counter) {
    return snapshot[RuntimePhaseCounters::index(counter)];
  };
  // Key assertion: the reservation's wait is counted once, at the transport site, with its time.
  EXPECT_EQ(count(RuntimePhaseCounter::kChannelInputCacheLockContendedTransport), 1U);
  EXPECT_GE(count(RuntimePhaseCounter::kChannelInputCacheLockWaitNsTransport),
            static_cast<std::uint64_t>(std::chrono::nanoseconds{holder.hold}.count()));
  EXPECT_EQ(count(RuntimePhaseCounter::kChannelInputCacheLockContendedDirect), 0U);
  EXPECT_EQ(count(RuntimePhaseCounter::kChannelSamplesReceived), 1U);

  auto event = wait_for_event(**session, **binding);
  ASSERT_TRUE(event) << runtime_error_message(event.error());
  ASSERT_TRUE((*binding)->retire(event->source_sequence));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// A waiter that must not sleep untimed gives up the lock as soon as the work it would protect is
// done, and otherwise takes the lock once the holder releases it.
// REQ: RUNTIME-064
// REQ: RUNTIME-079
TEST(RuntimeChannelBinding, LockUnlessSettledStopsWhenSettledAndOtherwiseAcquires) {
  std::timed_mutex mutex;
  std::atomic<bool> locked{};
  std::atomic<bool> release{};
  // Another thread holds the lock. It gives it up after a few seconds at most, so a waiter that
  // ignores settled() fails the first check instead of hanging.
  std::thread holder{[&] {
    const std::unique_lock held{mutex};
    locked.store(true);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (!release.load() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::microseconds{100});
    }
  }};
  while (!locked.load()) {
    std::this_thread::yield();
  }

  {
    std::atomic<bool> settled{};
    std::thread settler{[&] {
      std::this_thread::sleep_for(std::chrono::milliseconds{5});
      settled.store(true);
    }};
    // Key assertion: the waiter returns without the lock while the holder still has it.
    const auto unowned = lock_unless_settled(mutex, [&] { return settled.load(); });
    EXPECT_FALSE(unowned.owns_lock());
    EXPECT_TRUE(settled.load());
    settler.join();
  }

  std::thread releaser{[&] {
    std::this_thread::sleep_for(std::chrono::milliseconds{5});
    release.store(true);
  }};
  auto owned = lock_unless_settled(mutex, [] { return false; });
  EXPECT_TRUE(owned.owns_lock());
  releaser.join();
  holder.join();
}

// Holds the pump scan lock while another thread runs one activation cut for producer 1.
struct HeldScanLockCut {
  RuntimeChannelSession* session{};
  const RuntimePhaseCounters* counters{};
  // Receive the cut's delivery as the pump would, while still holding the lock.
  bool receive_while_held{};
  std::thread cut;
  std::atomic<bool> cut_returned{};
  std::atomic<bool> flushed{};
  bool received{};
  // Observed while the lock was held: the cut's contended wait, and its return.
  bool contended{};
  bool returned_while_held{};

  // Wait until `done()`, for a few seconds at most. Returns whether it became true.
  template <typename Done>
  [[nodiscard]] static bool wait_until(const Done& done) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (!done() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::microseconds{100});
    }
    return done();
  }

  static void run(void* state) noexcept {
    auto& held = *static_cast<HeldScanLockCut*>(state);
    held.cut = std::thread{[&held] {
      held.flushed.store(
          held.session->flush_process_local_activation_cut(VertexId{1U}).has_value());
      held.cut_returned.store(true);
    }};
    // The cut counts a contended acquisition once its first attempt fails, when its wait timing
    // starts, so from here on it is waiting.
    held.contended =
        wait_for_phase_count(*held.counters, RuntimePhaseCounter::kChannelScanLockContendedCut);
    if (held.receive_while_held) {
      held.received = RuntimeChannelSessionTestAccess::receive_as_pump(*held.session).has_value();
      // Keep the lock until the cut returns, so it can return only by seeing the receive.
      held.returned_while_held = wait_until([&held] { return held.cut_returned.load(); });
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds{20});
    }
  }
};

// Runs one producer activation whose transported publication stays undelivered, then one cut for
// it while the test holds the pump scan lock.
void run_cut_against_held_scan_lock(const bool receive_while_held,
                                    const std::function<void(const RuntimePhaseCounters&)>& check) {
  SessionFixture fixture;
  fixture.use_one_slot_blocking_transport();
  // No pump thread runs, so only the test and the cut receive.
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
  ASSERT_TRUE(fixture.source.output.emit(Sample<std::int32_t>{
      42, SampleMetadata{.capture_timestamp_ns = 10, .source_clock_id = ClockId{1U}}}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  HeldScanLockCut held{
      .session = session->get(),
      .counters = counters.get(),
      .receive_while_held = receive_while_held,
  };
  ASSERT_TRUE(
      RuntimeChannelSessionTestAccess::with_pump_scan_lock(**session, &held, HeldScanLockCut::run));
  held.cut.join();
  EXPECT_TRUE(held.contended);
  EXPECT_EQ(held.returned_while_held, receive_while_held);
  EXPECT_TRUE(held.flushed.load());
  EXPECT_EQ(held.received, receive_while_held);
  check(*counters);

  auto event = wait_for_event(**session, **binding);
  ASSERT_TRUE(event) << runtime_error_message(event.error());
  ASSERT_TRUE((*binding)->retire(event->source_sequence));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

[[nodiscard]] std::uint64_t phase_count(const RuntimePhaseCounters& counters,
                                        const RuntimePhaseCounter counter) {
  return counters.snapshot()[RuntimePhaseCounters::index(counter)];
}

// While the pump holds the scan lock, a cut with undelivered publications waits for it in bounded
// slices and then drains them; instrumented sessions record the contended cut acquisition.
// REQ: RUNTIME-064
// REQ: RUNTIME-079
TEST(RuntimeChannelBinding, ActivationCutWaitsForAHeldScanLockAndThenDrains) {
  run_cut_against_held_scan_lock(false, [](const RuntimePhaseCounters& counters) {
    // Key assertion: the cut waited for the lock, then drained the undelivered publication.
    EXPECT_EQ(phase_count(counters, RuntimePhaseCounter::kChannelScanLockContendedCut), 1U);
    EXPECT_GE(phase_count(counters, RuntimePhaseCounter::kChannelScanLockWaitNsCut),
              static_cast<std::uint64_t>(
                  std::chrono::nanoseconds{std::chrono::milliseconds{20}}.count()));
    EXPECT_EQ(phase_count(counters, RuntimePhaseCounter::kChannelActivationCutDrains), 1U);
    EXPECT_EQ(phase_count(counters, RuntimePhaseCounter::kChannelActivationCutSkipsWhileWaiting),
              0U);
    EXPECT_EQ(phase_count(counters, RuntimePhaseCounter::kChannelSamplesReceived), 1U);
  });
}

// A cut waiting for the scan lock stops waiting once the holder has received its delivery, and
// returns without taking the lock or scanning again.
// REQ: RUNTIME-064
// REQ: RUNTIME-079
TEST(RuntimeChannelBinding, ActivationCutReturnsWithoutTheScanLockOnceTheHolderReceives) {
  run_cut_against_held_scan_lock(true, [](const RuntimePhaseCounters& counters) {
    // Key assertion: the cut returned while the lock was still held, without draining.
    EXPECT_EQ(phase_count(counters, RuntimePhaseCounter::kChannelActivationCutSkipsWhileWaiting),
              1U);
    EXPECT_EQ(phase_count(counters, RuntimePhaseCounter::kChannelActivationCutDrains), 0U);
    EXPECT_EQ(phase_count(counters, RuntimePhaseCounter::kChannelScanLockContendedCut), 1U);
    EXPECT_EQ(phase_count(counters, RuntimePhaseCounter::kChannelSamplesReceived), 1U);
  });
}

// HoloMQ replaces an unreceived drop-oldest sample when a newer one reaches the same full queue.
// That delivery was published and will never be received, so it must count as settled. Otherwise
// published would exceed received forever, and every later activation cut would drain the whole
// inventory under the scan lock.
// REQ: RUNTIME-029
// REQ: RUNTIME-082
TEST(RuntimeChannelBinding, TransportReplacementCountsAsASettledDelivery) {
  SessionFixture fixture;
  fixture.use_one_slot_drop_oldest_queue();
  // Eventless and in one partition: no pump runs, so only the activation cut drains the queue.
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
                                               .capacity = 2U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  const auto phase = [&](const RuntimePhaseCounter counter) {
    return counters->snapshot()[RuntimePhaseCounters::index(counter)];
  };

  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{41}));
  // The depth-one queue still holds 41, so this send replaces it.
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{42}));
  auto progress = RuntimeChannelSessionTestAccess::process_local_progress_state(**session);
  ASSERT_TRUE(progress);
  EXPECT_EQ(progress->published_deliveries, 2U);
  EXPECT_EQ(progress->received_deliveries, 0U);
  EXPECT_EQ(progress->replaced_deliveries, 1U);
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelTransportReplacements), 1U);
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_TRUE((*session)->flush_process_local_activation_cut());
  progress = RuntimeChannelSessionTestAccess::process_local_progress_state(**session);
  ASSERT_TRUE(progress);
  EXPECT_EQ(progress->received_deliveries, 1U);

  // Key assertion: received plus replaced settles every published delivery, so the next cut skips
  // without draining.
  const std::uint64_t drains = phase(RuntimePhaseCounter::kChannelActivationCutDrains);
  const std::uint64_t skips = phase(RuntimePhaseCounter::kChannelActivationCutSkips);
  ASSERT_TRUE((*session)->flush_process_local_activation_cut());
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelActivationCutSkips), skips + 1U);
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelActivationCutDrains), drains);

  auto event = wait_for_event(**session, **binding);
  ASSERT_TRUE(event) << runtime_error_message(event.error());
  auto sample = (*session)->temporal_sample(event->port, event->source_sequence);
  ASSERT_TRUE(sample);
  EXPECT_EQ(sample->source_sequence, 2U);
  ASSERT_TRUE((*binding)->retire(event->source_sequence));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

}  // namespace
}  // namespace holoscan::detail
