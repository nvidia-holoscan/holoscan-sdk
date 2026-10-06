/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
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

#include <holoscan/channel/runtime.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "channel/endpoint_catalog_state.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "core/operator_spec_compiler_access.hpp"
#include "core/source_id_internal.hpp"
#include "runtime/dispatcher.hpp"
#include "runtime/endpoint_clock_health_runtime.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/runtime_channel_binding.hpp"

#include "runtime_channel_binding_test_support.hpp"

namespace holoscan::detail {
namespace {

using namespace std::chrono_literals;
using namespace runtime_channel_binding_test;

struct MonitoredImportSinkParams {
  std::int32_t restart_token{};
};

HOLOSCAN_PARAM_SCHEMA(MonitoredImportSinkParams,
                      param_schema("nvidia.holoscan.test.MonitoredImportSinkParams", 1),
                      field<&MonitoredImportSinkParams::restart_token>("restart_token")
                          .lifecycle_mutable(LifecycleStage::kConfigure));

class MonitoredImportSinkOperator final : public Operator<MonitoredImportSinkParams> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnAnyInput{})
        .input(input, LatestWithin{.max_age = 5s})
        .on_missing(input, Skip{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
};

// REQ: CHAN-069
// REQ: CHAN-078
// REQ: CHAN-079
// REQ: CHAN-087
// REQ: RUNTIME-017
// REQ: RUNTIME-020
// REQ: RUNTIME-022
// REQ: RUNTIME-024
// REQ: RUNTIME-026
// REQ: SYSTEM-024
TEST(RuntimeChannelBinding,
     MonitoredImportPreservesProtectedDataAndExplicitEosAcrossGenerationFencedRebind) {
  enum class RebindForm : std::uint8_t { kScopedReset, kFullReset, kLifecycleRestart };
  constexpr std::array rebind_forms{
      RebindForm::kScopedReset,
      RebindForm::kFullReset,
      RebindForm::kLifecycleRestart,
  };
  for (const RebindForm rebind_form : rebind_forms) {
    SCOPED_TRACE(static_cast<std::uint32_t>(rebind_form));
    const std::string endpoint_name = unique_service();
    auto capture_clock = std::make_shared<ManualClock>(0);
    auto age_clock = std::make_shared<RealtimeClock>();

    Graph graph{"monitored-protected-witness"};
    (void)graph.bind_shared_clock("a-unrelated-clock", std::make_shared<ManualClock>(0));
    const ClockKey capture_key = graph.bind_shared_clock("z-capture-clock", capture_clock);
    const ClockKey age_key = graph.bind_shared_clock("m-health-age-clock", age_clock);
    graph.set_default_clock(age_key);
    graph.add_clock_conversion(capture_key, age_key, AffineClockConversion{});

    TopicOptions options;
    options.endpoint_name = endpoint_name;
    options.capture_time = RequireCaptureTime{.clock = capture_key};
    options.clock_health = ClockHealthOptions{
        .capability = ClockHealthCapability::kStatusReports,
        .requirement = ClockHealthRequirement::kFreshHealthyReport,
        .max_report_age = 5s,
    };
    const auto imported =
        graph.import_topic<std::uint32_t>("holomq://" + endpoint_name, std::move(options));
    const auto sink_handle = graph.op<MonitoredImportSinkOperator>("sink");
    graph.lifecycle().allow_self_restart(sink_handle);
    graph.add_flow(imported.output(),
                   sink_handle->input,
                   ConnectionOptions{
                       .queue_policy = QueuePolicy::kDropOldest,
                       .queue_depth = 1U,
                   });

    CompileOptions compile_options;
    compile_options.lifecycle_transactions = LifecycleTransactionOptions{
        .max_pending_requests = 64U,
        .max_pending_transactions = 64U,
        .max_receipts = 128U,
        .max_fenced_operators = 64U,
        .max_candidate_generations = 64U,
        .max_events = 1024U,
        .max_audit_records = 256U,
        .max_quiesce_records = 256U,
        .max_rollback_records = 256U,
    };
    const ExecutionPlan plan = compile(graph, std::move(compile_options));
    ASSERT_TRUE(plan.ok()) << plan.json();
    const auto state = ExecutionPlanAccess::retain(plan);
    ASSERT_NE(state, nullptr);
    const auto generation_capacity = state->imported_generation_capacities.begin();
    ASSERT_EQ(state->imported_generation_capacities.size(), 1U);
    ASSERT_TRUE(generation_capacity->restart_capable_minimum_two);
    ASSERT_EQ(generation_capacity->endpoint_max_generations, 2U);
    ASSERT_EQ(generation_capacity->candidate_reservation_capacity, 1U);
    auto exec = GraphExec::reconstruct(state, RunGenerationId{1U});
    ASSERT_TRUE(exec) << runtime_error_message(exec.error());

    std::vector<RuntimeChannelPortReplayRecord> replay_ports;
    MonitoredImportSinkOperator* runtime_sink = nullptr;
    VertexId sink_vertex{};
    PortId sink_input{};
    for (RuntimeOperatorInstance& instance : (*exec)->operators()) {
      if (auto* candidate = dynamic_cast<MonitoredImportSinkOperator*>(instance.op.get())) {
        runtime_sink = candidate;
        sink_vertex = instance.vertex;
      }
      ASSERT_NE(instance.replay_spec, nullptr);
      for (std::size_t index = 0U;
           index < OperatorSpecCompilerAccess::port_count(*instance.replay_spec);
           ++index) {
        const auto replay = OperatorSpecCompilerAccess::port(*instance.replay_spec, index);
        const auto planned = std::find_if(
            state->ports.begin(), state->ports.end(), [&](const PortPlanRecord& candidate) {
              return candidate.owner == instance.vertex &&
                     candidate.owner_kind == PortOwnerKind::kOperator &&
                     candidate.direction == replay.direction && candidate.local_name == replay.name;
            });
        ASSERT_NE(planned, state->ports.end());
        replay_ports.push_back(RuntimeChannelPortReplayRecord{
            .port = planned->id,
            .owner = instance.vertex,
            .direction = replay.direction,
            .declaration = const_cast<void*>(replay.declaration),
            .type_ops = replay.runtime_type_ops,
        });
        if (instance.vertex == sink_vertex && replay.direction == PortDirection::kInput) {
          sink_input = planned->id;
        }
      }
    }
    ASSERT_NE(runtime_sink, nullptr);
    ASSERT_TRUE(sink_vertex);
    ASSERT_TRUE(sink_input);

    auto session = RuntimeChannelSession::create(*state, replay_ports);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    auto health = EndpointClockHealthRuntimeCoordinator::create(state->endpoint_clock_health,
                                                                state->clocks,
                                                                state->plan_generation,
                                                                RunGenerationId{1U},
                                                                (*exec)->clock_registry());
    ASSERT_TRUE(health) << (health ? std::string_view{} : health.error().token);
    std::mutex clock_health_serialization;
    ASSERT_TRUE((*session)->bind_clock_health(**health, clock_health_serialization));

    const auto selection = std::find_if(
        state->input_selections.begin(),
        state->input_selections.end(),
        [&](const InputSelectionPlanRecord& candidate) { return candidate.input == sink_input; });
    ASSERT_NE(selection, state->input_selections.end());
    const auto endpoint = plan.endpoint_catalog().endpoint(endpoint_name);
    ASSERT_TRUE(endpoint);
    const auto* endpoint_record = ChannelEndpointCatalogAccess::record(*endpoint);
    ASSERT_NE(endpoint_record, nullptr);
    const auto health_plan =
        std::find_if(state->endpoint_clock_health.begin(),
                     state->endpoint_clock_health.end(),
                     [&](const EndpointClockHealthPlanRecord& candidate) {
                       return candidate.endpoint == endpoint_record->boundary_port &&
                              candidate.capability == ClockHealthCapability::kStatusReports;
                     });
    ASSERT_NE(health_plan, state->endpoint_clock_health.end());
    const ClockId catalog_capture_clock = ChannelEndpointCatalogAccess::capture_clock_id(*endpoint);
    ASSERT_TRUE(catalog_capture_clock.valid());
    ASSERT_TRUE(health_plan->capture_clock.valid());
    // The portable endpoint catalog canonicalizes its own dense identity order independently of the
    // ExecutionPlan. This fixture intentionally produces different numeric ids for the same named
    // capture domain and proves the Runtime attachment translates across that identity boundary.
    ASSERT_NE(catalog_capture_clock, health_plan->capture_clock);
    auto attached_clock_binding =
        RuntimeChannelSessionTestAccess::input_reset_state(**session, sink_input);
    ASSERT_TRUE(attached_clock_binding) << runtime_error_message(attached_clock_binding.error());
    ASSERT_EQ(attached_clock_binding->catalog_capture_clock, catalog_capture_clock);
    ASSERT_EQ(attached_clock_binding->plan_capture_clock, health_plan->capture_clock);
    auto capacity = (*session)->readiness_capacity(sink_vertex);
    ASSERT_TRUE(capacity);
    // The retained POD candidate and its ordered EOS witness require two independent readiness
    // credits. Tensor receive-control geometry is not involved in this scalar import.
    ASSERT_EQ(*capacity, 2U);
    auto binding = (*session)->create_readiness_binding(
        sink_vertex,
        RuntimePortBindingRequest{
            .plan_generation = state->plan_generation,
            .run_generation = RunGenerationId{1U},
            .kind = RuntimeEventKind::kInputArrival,
            .source = make_source_id(1U),
            .source_clock = health_plan->capture_clock,
            .reference_clock = selection->reference_clock,
            .source_to_reference_conversion_id = selection->conversion_id,
            .capacity = *capacity,
        });
    ASSERT_TRUE(binding) << runtime_error_message(binding.error());
    // External transport and clock-health heads are consumed by poll itself, so an import remains
    // lane-polled even before a typed cache revision is visible.
    EXPECT_TRUE((*binding)->poll_target().has_value());

    const std::array affected{sink_vertex};
    const auto apply_rebind = [&]() -> expected<void, Error> {
      switch (rebind_form) {
        case RebindForm::kScopedReset:
          return (*session)->reset(sink_vertex);
        case RebindForm::kFullReset:
          return (*session)->reset();
        case RebindForm::kLifecycleRestart:
          return (*session)->commit_restart(sink_vertex, affected);
      }
      return make_unexpected(Error{ErrorCode::kProtocolError});
    };

    // Peer absence alone is not an incarnation fence. Every reset form rejects before changing
    // the input state when no accepted operation and explicit EOS identify the old generation.
    const auto unattached_before =
        RuntimeChannelSessionTestAccess::input_reset_state(**session, sink_input);
    ASSERT_TRUE(unattached_before) << runtime_error_message(unattached_before.error());
    auto unsupported = apply_rebind();
    ASSERT_FALSE(unsupported);
    EXPECT_EQ(unsupported.error().code, ErrorCode::kNotSupported);
    const auto unattached_after =
        RuntimeChannelSessionTestAccess::input_reset_state(**session, sink_input);
    ASSERT_TRUE(unattached_after) << runtime_error_message(unattached_after.error());
    EXPECT_EQ(unattached_after->active_generation, unattached_before->active_generation);
    EXPECT_EQ(unattached_after->view_generation, unattached_before->view_generation);
    EXPECT_EQ(unattached_after->producer_epoch, unattached_before->producer_epoch);
    EXPECT_EQ(unattached_after->latched_producer_generation,
              unattached_before->latched_producer_generation);
    EXPECT_EQ(unattached_after->end_of_stream_received, unattached_before->end_of_stream_received);

    auto external_runtime = ChannelRuntime::create();
    ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
    auto created_publisher =
        external_runtime->create_publisher<std::uint32_t>(plan.endpoint_catalog(), endpoint_name);
    ASSERT_TRUE(created_publisher) << runtime_error_message(created_publisher.error());
    std::optional<ChannelPublisher<std::uint32_t>> publisher{std::move(*created_publisher)};
    ASSERT_TRUE((*session)->arm());

    const auto connection_deadline = std::chrono::steady_clock::now() + 2s;
    bool connected = false;
    while (std::chrono::steady_clock::now() < connection_deadline) {
      ASSERT_TRUE((*binding)->poll(TimePoint{.timestamp_ns = age_clock->timestamp(),
                                             .clock_id = selection->reference_clock}));
      auto refreshed = publisher->update_connections();
      ASSERT_TRUE(refreshed) << runtime_error_message(refreshed.error());
      if (publisher->has_subscribers()) {
        connected = true;
        break;
      }
      std::this_thread::sleep_for(1ms);
    }
    ASSERT_TRUE(connected);

    ASSERT_TRUE(publisher->report_clock_health(ClockHealthReport::healthy()));  // H1
    ASSERT_TRUE(publisher->emit_captured(11U, 101));                            // S1
    auto first_event = wait_for_event(**session, **binding, selection->reference_clock);
    ASSERT_TRUE(first_event) << runtime_error_message(first_event.error());
    auto first_temporal =
        (*session)->temporal_sample(first_event->port, first_event->source_sequence);
    ASSERT_TRUE(first_temporal);
    EXPECT_EQ(first_temporal->source_sequence, 2U);
    // Temporal cache records are normalized to the selector's reference clock when readiness is
    // emitted. The typed Sample retains the translated plan-local capture clock separately.
    EXPECT_EQ(first_temporal->capture_time,
              (TimePoint{.timestamp_ns = 101, .clock_id = selection->reference_clock}));

    // H2 is allowed to advance while S1's emitted witness protects the only typed slot. S2 follows
    // H2 in the publisher order but must remain a single staged claim until S1 is retired.
    ASSERT_TRUE(publisher->report_clock_health(ClockHealthReport::healthy()));  // H2
    ASSERT_TRUE(publisher->emit_captured(22U, 202));                            // S2
    EndpointClockHealthReceiver* const receiver = health->get()->receiver(health_plan->endpoint);
    ASSERT_NE(receiver, nullptr);
    const auto health_deadline = std::chrono::steady_clock::now() + 2s;
    while (receiver->state().report_sequence != 2U &&
           std::chrono::steady_clock::now() < health_deadline) {
      ASSERT_TRUE((*binding)->poll(TimePoint{.timestamp_ns = age_clock->timestamp(),
                                             .clock_id = selection->reference_clock}));
      std::this_thread::sleep_for(1ms);
    }
    ASSERT_EQ(receiver->state().report_sequence, 2U);
    EXPECT_EQ(receiver->state().publisher_operation_sequence, 3U);
    EXPECT_EQ(receiver->state().health_generation, 3U);

    for (std::size_t attempt = 0U; attempt < 8U; ++attempt) {
      ASSERT_TRUE((*binding)->poll(TimePoint{.timestamp_ns = age_clock->timestamp(),
                                             .clock_id = selection->reference_clock}));
      std::this_thread::sleep_for(1ms);
    }
    RuntimeEvent blocked_event;
    auto blocked_count = (*binding)->drain(std::span<RuntimeEvent>{&blocked_event, 1U});
    ASSERT_TRUE(blocked_count);
    EXPECT_EQ(*blocked_count, 0U);
    EXPECT_TRUE((*session)->temporal_sample(first_event->port, first_event->source_sequence));

    // Selector acceptance releases the temporal-cache claim exactly once, but the unretired
    // readiness witness remains an independent payload-lifetime claim. S2 must stay staged until
    // that witness is retired as well.
    const std::array first_record{*first_temporal};
    ASSERT_TRUE((*session)->retire_samples(sink_input, first_record));
    EXPECT_TRUE((*session)->temporal_sample(first_event->port, first_event->source_sequence));
    ASSERT_TRUE((*binding)->poll(
        TimePoint{.timestamp_ns = age_clock->timestamp(), .clock_id = selection->reference_clock}));
    auto still_blocked_count = (*binding)->drain(std::span<RuntimeEvent>{&blocked_event, 1U});
    ASSERT_TRUE(still_blocked_count);
    EXPECT_EQ(*still_blocked_count, 0U);

    ASSERT_TRUE((*binding)->retire(first_event->source_sequence));
    auto second_event = wait_for_event(**session, **binding, selection->reference_clock);
    ASSERT_TRUE(second_event) << runtime_error_message(second_event.error());
    auto second_temporal =
        (*session)->temporal_sample(second_event->port, second_event->source_sequence);
    ASSERT_TRUE(second_temporal);
    EXPECT_EQ(second_temporal->source_sequence, 4U);
    EXPECT_EQ(second_temporal->capture_time,
              (TimePoint{.timestamp_ns = 202, .clock_id = selection->reference_clock}));

    const std::array selected{RuntimeChannelActivationSelection{
        .port = sink_input,
        .selection = {InputState::kSelected, SelectionReason::kNone},
        .samples = std::span<const TemporalSampleRecord>{&*second_temporal, 1U},
        .destructive = true,
    }};
    ASSERT_TRUE((*session)->prepare_activation(sink_vertex, 1U, selected));
    auto value = runtime_sink->input.receive_data();
    ASSERT_TRUE(value) << runtime_error_message(value.error());
    EXPECT_EQ(*value, 22U);
    ASSERT_TRUE((*binding)->retire(second_event->source_sequence));
    const std::array second_record{*second_temporal};
    ASSERT_TRUE((*session)->retire_samples(sink_input, second_record));
    ASSERT_TRUE((*session)->retire_activation(sink_vertex, 1U, true));

    // Quiesce the readiness pump while the old producer remains attached. Empty queues do not
    // override that live-peer evidence: all reset forms must remain retryable and preserve the
    // accepted generation until a backend-enforced generation cut exists.
    (*session)->cancel();
    ASSERT_TRUE((*session)->join());
    const auto live_peer_input_before =
        RuntimeChannelSessionTestAccess::input_reset_state(**session, sink_input);
    ASSERT_TRUE(live_peer_input_before) << runtime_error_message(live_peer_input_before.error());
    auto live_peer_reset = (*session)->reset(sink_vertex);
    ASSERT_FALSE(live_peer_reset);
    EXPECT_EQ(live_peer_reset.error().code, ErrorCode::kBackpressured);
    auto live_peer_full_reset = (*session)->reset();
    ASSERT_FALSE(live_peer_full_reset);
    EXPECT_EQ(live_peer_full_reset.error().code, ErrorCode::kBackpressured);
    auto live_peer_restart = (*session)->commit_restart(sink_vertex, affected);
    ASSERT_FALSE(live_peer_restart);
    EXPECT_EQ(live_peer_restart.error().code, ErrorCode::kBackpressured);
    const auto live_peer_input_after =
        RuntimeChannelSessionTestAccess::input_reset_state(**session, sink_input);
    ASSERT_TRUE(live_peer_input_after) << runtime_error_message(live_peer_input_after.error());
    EXPECT_EQ(live_peer_input_after->active_generation, live_peer_input_before->active_generation);
    EXPECT_EQ(live_peer_input_after->view_generation, live_peer_input_before->view_generation);
    EXPECT_EQ(live_peer_input_after->producer_epoch, live_peer_input_before->producer_epoch);
    ASSERT_TRUE((*session)->arm());

    // Explicit close is an ordered control operation. It creates one EOS witness without a typed
    // payload or temporal-cache record; peer disappearance alone remains semantically silent.
    ASSERT_TRUE(publisher->close());
    auto end_of_stream = wait_for_event(**session, **binding, selection->reference_clock);
    ASSERT_TRUE(end_of_stream) << runtime_error_message(end_of_stream.error());
    EXPECT_EQ(end_of_stream->kind, RuntimeEventKind::kInputEndOfStream);
    EXPECT_EQ(end_of_stream->port, sink_input);
    EXPECT_GT(end_of_stream->source_sequence, second_event->source_sequence);
    auto no_payload = (*session)->temporal_sample(sink_input, end_of_stream->source_sequence);
    ASSERT_FALSE(no_payload);
    EXPECT_EQ(no_payload.error().code, ErrorCode::kNotFound);
    ASSERT_TRUE((*binding)->retire(end_of_stream->source_sequence));
    (*session)->cancel();
    ASSERT_TRUE((*session)->join());

    const auto input_before =
        RuntimeChannelSessionTestAccess::input_reset_state(**session, sink_input);
    ASSERT_TRUE(input_before) << runtime_error_message(input_before.error());
    const EndpointClockHealthState health_before = receiver->state();
    const std::size_t pending_reports_before = receiver->pending_report_count();
    // The trace/replay consumer owns the fixed structural-update ring. Drain the earlier accepted
    // report transitions so this test can attribute the next record exactly to the generation
    // cutover and so that later replacement health evidence cannot be rejected by unrelated
    // retained audit records.
    while ((*health)->pop_update(endpoint_record->boundary_port).has_value()) {
    }
    const std::size_t pending_updates_before = receiver->pending_update_count();

    std::binary_semaphore reset_waiting{0};
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_clock_health_lock_observer(
        **session, &reset_waiting, [](void* state) noexcept {
          static_cast<std::binary_semaphore*>(state)->release();
        }));
    std::unique_lock serialization_hold{clock_health_serialization};
    auto rebound_future = std::async(std::launch::async, apply_rebind);
    const bool reset_reached_serialization = reset_waiting.try_acquire_for(2s);
    const std::future_status status_before_release = rebound_future.wait_for(0ms);
    serialization_hold.unlock();
    EXPECT_TRUE(reset_reached_serialization);
    EXPECT_EQ(status_before_release, std::future_status::timeout);
    auto rebound = rebound_future.get();
    ASSERT_TRUE(rebound) << runtime_error_message(rebound.error());

    // Channel reset/restart publishes the transport, input, and producer epochs. The enclosing
    // Runtime transaction advances its independently owned scheduler source afterward; that
    // distinct cut reopens an EOS-retired readiness binding without exposing successor data under
    // the predecessor source generation.
    auto advanced_source_generation = (*binding)->advance_source_generation();
    ASSERT_TRUE(advanced_source_generation)
        << runtime_error_message(advanced_source_generation.error());
    EXPECT_EQ(*advanced_source_generation, 2U);
    const auto input_after =
        RuntimeChannelSessionTestAccess::input_reset_state(**session, sink_input);
    ASSERT_TRUE(input_after) << runtime_error_message(input_after.error());
    if (rebind_form == RebindForm::kFullReset) {
      // A whole-session reset reconstructs the input's baseline activation/view epochs; the
      // external producer epoch below is the generation fence that proves the catalog cutover.
      EXPECT_EQ(input_after->active_generation, input_before->active_generation);
      EXPECT_EQ(input_after->view_generation, input_before->view_generation);
    } else {
      EXPECT_EQ(input_after->active_generation, input_before->active_generation + 1U);
      EXPECT_EQ(input_after->view_generation, input_before->view_generation + 1U);
    }
    EXPECT_EQ(input_after->producer_epoch, input_before->producer_epoch + 1U);
    EXPECT_EQ(input_after->occupied_samples, 0U);
    EXPECT_EQ(input_after->latched_producer_generation, 0U);
    EXPECT_FALSE(input_after->end_of_stream_received);
    EXPECT_EQ(input_after->catalog_capture_clock, input_before->catalog_capture_clock);
    EXPECT_EQ(input_after->plan_capture_clock, input_before->plan_capture_clock);
    EXPECT_EQ(receiver->pending_report_count(), pending_reports_before);
    EXPECT_EQ(receiver->pending_update_count(), pending_updates_before + 1U);
    EXPECT_GT(receiver->state().binding.producer_epoch, health_before.binding.producer_epoch);
    EXPECT_EQ(receiver->state().report_sequence, 0U);
    EXPECT_EQ(receiver->state().publisher_operation_sequence, 0U);
    EXPECT_EQ(receiver->state().health_generation, health_before.health_generation + 1U);
    EXPECT_EQ(receiver->state().health_update_sequence, health_before.health_update_sequence + 1U);
    auto reset_update = (*health)->pop_update(endpoint_record->boundary_port);
    ASSERT_TRUE(reset_update.has_value());
    EXPECT_EQ(reset_update->endpoint, endpoint_record->boundary_port);
    EXPECT_EQ(reset_update->binding, receiver->state().binding);
    EXPECT_EQ(receiver->pending_update_count(), pending_updates_before);

    // Destroy the closed incarnation and attach a fresh publisher. The replacement starts its own
    // operation/report sequence at one, while Runtime preserves its local monotonic generation and
    // accepts the new sample only through the freshly rebound endpoint.
    publisher.reset();
    auto replacement_created =
        external_runtime->create_publisher<std::uint32_t>(plan.endpoint_catalog(), endpoint_name);
    ASSERT_TRUE(replacement_created) << runtime_error_message(replacement_created.error());
    std::optional<ChannelPublisher<std::uint32_t>> replacement{std::move(*replacement_created)};
    ASSERT_TRUE((*session)->arm());
    const auto replacement_connection_deadline = std::chrono::steady_clock::now() + 2s;
    bool replacement_connected = false;
    while (std::chrono::steady_clock::now() < replacement_connection_deadline) {
      ASSERT_TRUE((*binding)->poll(TimePoint{.timestamp_ns = age_clock->timestamp(),
                                             .clock_id = selection->reference_clock}));
      ASSERT_TRUE(replacement->update_connections());
      if (replacement->has_subscribers()) {
        replacement_connected = true;
        break;
      }
      std::this_thread::sleep_for(1ms);
    }
    ASSERT_TRUE(replacement_connected);
    ASSERT_TRUE(replacement->report_clock_health(ClockHealthReport::healthy()));
    ASSERT_TRUE(replacement->emit_captured(33U, 303));
    auto replacement_event = wait_for_event(**session, **binding, selection->reference_clock);
    ASSERT_TRUE(replacement_event) << runtime_error_message(replacement_event.error());
    EXPECT_EQ(replacement_event->kind, RuntimeEventKind::kInputArrival);
    auto replacement_sample =
        (*session)->temporal_sample(sink_input, replacement_event->source_sequence);
    ASSERT_TRUE(replacement_sample) << runtime_error_message(replacement_sample.error());
    EXPECT_EQ(replacement_sample->source_sequence, 2U);
    const std::array replacement_selection{RuntimeChannelActivationSelection{
        .port = sink_input,
        .selection = {InputState::kSelected, SelectionReason::kNone},
        .samples = std::span<const TemporalSampleRecord>{&*replacement_sample, 1U},
        .destructive = true,
    }};
    ASSERT_TRUE((*session)->prepare_activation(
        sink_vertex, input_after->view_generation, replacement_selection));
    auto replacement_value = runtime_sink->input.receive_data();
    ASSERT_TRUE(replacement_value) << runtime_error_message(replacement_value.error());
    EXPECT_EQ(*replacement_value, 33U);
    ASSERT_TRUE((*binding)->retire(replacement_event->source_sequence));
    const std::array replacement_record{*replacement_sample};
    ASSERT_TRUE((*session)->retire_samples(sink_input, replacement_record));
    ASSERT_TRUE((*session)->retire_activation(sink_vertex, input_after->view_generation, true));

    // Close and retire the replacement too. The plan admitted exactly two retained generations, so
    // another cutover is rejected before detach and leaves every published generation unchanged.
    ASSERT_TRUE(replacement->close());
    auto replacement_eos = wait_for_event(**session, **binding, selection->reference_clock);
    ASSERT_TRUE(replacement_eos) << runtime_error_message(replacement_eos.error());
    ASSERT_EQ(replacement_eos->kind, RuntimeEventKind::kInputEndOfStream);
    ASSERT_TRUE((*binding)->retire(replacement_eos->source_sequence));
    (*session)->cancel();
    ASSERT_TRUE((*session)->join());
    const auto exhausted_before =
        RuntimeChannelSessionTestAccess::input_reset_state(**session, sink_input);
    ASSERT_TRUE(exhausted_before) << runtime_error_message(exhausted_before.error());
    auto exhausted = apply_rebind();
    ASSERT_FALSE(exhausted);
    EXPECT_EQ(exhausted.error().code, ErrorCode::kResourceExhausted);
    const auto exhausted_after =
        RuntimeChannelSessionTestAccess::input_reset_state(**session, sink_input);
    ASSERT_TRUE(exhausted_after) << runtime_error_message(exhausted_after.error());
    EXPECT_EQ(exhausted_after->active_generation, exhausted_before->active_generation);
    EXPECT_EQ(exhausted_after->view_generation, exhausted_before->view_generation);
    EXPECT_EQ(exhausted_after->producer_epoch, exhausted_before->producer_epoch);
    EXPECT_EQ(exhausted_after->latched_producer_generation,
              exhausted_before->latched_producer_generation);
    EXPECT_EQ(exhausted_after->end_of_stream_received, exhausted_before->end_of_stream_received);
  }
}

// Share the reset/restart matrix across sequence continuity and exhaustion regressions, including
// OutputSequenceSurvivesEveryCutoverAndOrderedClose.
enum class SequenceCutover : std::uint8_t { kFull, kSource, kReceiver, kRestart };

// Apply a boundary to the fixture's source (owner 1) and receiver (owner 2); restart affects both.
// OutputSequenceSurvivesEveryCutoverAndOrderedClose uses this to repeat each cutover consistently.
expected<void, Error> apply_sequence_cutover(RuntimeChannelSession& session,
                                             const SequenceCutover cutover) {
  const std::array affected{VertexId{1U}, VertexId{2U}};
  switch (cutover) {
    case SequenceCutover::kFull:
      return session.reset();
    case SequenceCutover::kSource:
      return session.reset(VertexId{1U});
    case SequenceCutover::kReceiver:
      return session.reset(VertexId{2U});
    case SequenceCutover::kRestart:
      return session.commit_restart(VertexId{1U}, affected);
  }
  return make_unexpected(Error{ErrorCode::kInvalidArgument});
}

// Two outputs in one session have independent histories. Re-emitting a stamped sample uses the
// destination output counter, while a new session starts both counters afresh.
// REQ: RUNTIME-052
TEST(RuntimeChannelBinding, OutputSequencesAreIndependentAndFreshPerSession) {
  // Planned graph: first  --output--> input-- first-sink
  //                second --output--> input-- second-sink
  // Key setup: create two sessions from this plan and capture each output with a scripted
  // publisher; reset one source, then re-emit a sample stamped with sequence 99.
  Graph graph{"independent-output-sequences"};
  graph.set_default_clock(graph.bind_shared_clock("clock", std::make_shared<ManualClock>(0)));
  const auto first = graph.op<SourceOperator>("first");
  const auto second = graph.op<SourceOperator>("second");
  const auto first_sink = graph.op<SinkOperator>("first-sink");
  const auto second_sink = graph.op<SinkOperator>("second-sink");
  graph.add_flow(first->output, first_sink->input);
  graph.add_flow(second->output, second_sink->input);
  const auto plan = compile_with_process_local_holomq(graph, CompileOptions{});
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  for (const std::uint64_t run : {1U, 2U}) {
    SCOPED_TRACE(run);
    std::array<SourceOperator, 2U> source_operators;
    std::array<SinkOperator, 2U> sink_operators;
    std::vector<RuntimeChannelPortReplayRecord> ports;
    std::vector<SourceOperator*> sources;
    std::vector<PortId> outputs;
    std::vector<VertexId> owners;
    std::size_t input_index = 0U;
    for (const auto& planned : state->ports) {
      void* declaration = nullptr;
      if (planned.direction == PortDirection::kOutput) {
        ASSERT_LT(sources.size(), source_operators.size());
        auto& source = source_operators[sources.size()];
        declaration = &source.output;
        sources.push_back(&source);
        outputs.push_back(planned.id);
        owners.push_back(planned.owner);
      } else {
        ASSERT_LT(input_index, sink_operators.size());
        declaration = &sink_operators[input_index++].input;
      }
      ports.push_back(
          RuntimeChannelPortReplayRecord{.port = planned.id,
                                         .owner = planned.owner,
                                         .direction = planned.direction,
                                         .declaration = declaration,
                                         .type_ops = runtime_port_type_ops<std::int32_t>()});
    }
    ASSERT_EQ(sources.size(), 2U);
    auto session = RuntimeChannelSession::create(*state, ports);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    std::array<ScriptedInlinePublisher*, 2U> observed{};
    for (std::size_t index = 0U; index < sources.size(); ++index) {
      auto publisher = std::make_unique<ScriptedInlinePublisher>(
          std::vector<channel_internal::InlinePublicationResult>{});
      observed[index] = publisher.get();
      std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
      publishers.push_back(std::move(publisher));
      ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
          **session, outputs[index], std::move(publishers)));
      ASSERT_TRUE((*session)->prepare_activation(owners[index], 1U, {}));
      for (std::size_t count = 0U; count <= index; ++count) {
        ASSERT_TRUE(sources[index]->output.emit(std::int32_t{17}));
      }
      ASSERT_TRUE((*session)->retire_activation(owners[index], 1U, true));
    }
    ASSERT_TRUE((*session)->reset(owners[0]));
    ASSERT_TRUE((*session)->prepare_activation(owners[0], 2U, {}));
    ASSERT_TRUE(sources[0]->output.emit(
        Sample<std::int32_t>{19, SampleMetadata{.sequence = 99U, .trace_id = 0xA11CEU}}));
    ASSERT_TRUE((*session)->retire_activation(owners[0], 2U, true));
    ASSERT_TRUE((*session)->prepare_activation(owners[1], 1U, {}));
    ASSERT_TRUE(sources[1]->output.emit(std::int32_t{20}));
    ASSERT_TRUE((*session)->retire_activation(owners[1], 1U, true));
    // Key assertion: each session starts both independent histories at 1; re-emission uses the
    // reset source's next output sequence while preserving the payload and trace_id.
    ASSERT_EQ(observed[0]->sequences, (std::vector<std::uint64_t>{1U, 2U}));
    ASSERT_EQ(observed[1]->sequences, (std::vector<std::uint64_t>{1U, 2U, 3U}));
    EXPECT_EQ(observed[0]->metadata_records.back().trace_id, 0xA11CEU);
    EXPECT_EQ(observed[0]->values.back(), 19);
  }
}

// Resetting one fanout receiver advances only its epoch; the source still publishes one shared
// sequence to both receivers and counts the operation once.
// REQ: RUNTIME-052
// REQ: RUNTIME-024
TEST(RuntimeChannelBinding, ReceiverOnlyResetPreservesFanoutPublicationAuthority) {
  // Source --output--+--blocking----> input-- sink (owner 2)
  //                 +--drop-oldest-> input-- second_sink (owner 3)
  // Key setup: publish twice, reset only owner 2, then publish once more to both receivers.
  SessionFixture fixture;
  fixture.add_mixed_policy_fanout();
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session);
  ASSERT_TRUE((*session)->refresh_connections());
  for (const std::uint64_t sequence : {1U, 2U, 3U}) {
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
    ASSERT_TRUE(fixture.source.output.emit(static_cast<std::int32_t>(sequence)));
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
    // Key assertion: both receivers share each publication sequence, and only the reset
    // receiver advances its epoch; the final counter counts each fanout publication once.
    for (const PortId input : {PortId{2U}, PortId{3U}}) {
      ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, input));
      auto state = RuntimeChannelSessionTestAccess::input_reset_state(**session, input);
      ASSERT_TRUE(state);
      EXPECT_EQ(state->last_producer_sequence, sequence);
      EXPECT_EQ(state->producer_epoch, input.value() == 2U && sequence == 3U ? 2U : 1U);
    }
    if (sequence == 2U) {
      ASSERT_TRUE((*session)->reset(VertexId{2U}));
    }
  }
  auto counter = RuntimeChannelSessionTestAccess::output_next_sequence(**session, PortId{1U});
  ASSERT_TRUE(counter);
  EXPECT_EQ(*counter, 4U);
}

// Publication identity survives each administrative boundary, including repeated empty cutovers.
// Terminal order must work with and without a replacement data record on both delivery routes.
// REQ: RUNTIME-052
// REQ: RUNTIME-024
// REQ: CHAN-136
TEST(RuntimeChannelBinding, OutputSequenceSurvivesEveryCutoverAndOrderedClose) {
  // Source --output--[transport or direct retained]--> input-- sink
  // Key setup: receive sequences 1 and 2, repeat each cutover twice without data, then close
  // with or without replacement data on both delivery routes.
  for (const bool direct : {false, true}) {
    for (const auto reset_kind : {SequenceCutover::kFull,
                                  SequenceCutover::kSource,
                                  SequenceCutover::kReceiver,
                                  SequenceCutover::kRestart}) {
      for (const bool replacement_data : {false, true}) {
        SCOPED_TRACE(::testing::Message()
                     << direct << "/" << static_cast<int>(reset_kind) << "/" << replacement_data);
        SessionFixture fixture;
        if (direct) {
          fixture.use_direct_retained_delivery();
        }
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
                                      .capacity = 8U});
        ASSERT_TRUE(binding) << runtime_error_message(binding.error());
        ASSERT_TRUE((*session)->refresh_connections());
        ASSERT_TRUE((*session)->arm());
        ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
        ASSERT_TRUE(fixture.source.output.emit(std::int32_t{101}));
        ASSERT_TRUE(fixture.source.output.emit(std::int32_t{102}));
        ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
        for (const std::uint64_t sequence : {1U, 2U}) {
          auto event = wait_for_event(**session, **binding);
          ASSERT_TRUE(event) << runtime_error_message(event.error());
          auto record = (*session)->temporal_sample(event->port, event->source_sequence);
          ASSERT_TRUE(record) << runtime_error_message(record.error());
          EXPECT_EQ(record->source_sequence, sequence);
          EXPECT_EQ(record->producer_epoch, 1U);
          ASSERT_TRUE((*binding)->retire(event->source_sequence));
        }
        (*session)->cancel();
        ASSERT_TRUE((*session)->join());
        ASSERT_TRUE(apply_sequence_cutover(**session, reset_kind));
        ASSERT_TRUE(apply_sequence_cutover(**session, reset_kind));
        // Key assertion: empty cutovers advance the receiver epoch while preserving output
        // sequence 3 as the next coordinate for replacement data or immediate EOS.
        auto counter = RuntimeChannelSessionTestAccess::output_next_sequence(**session, PortId{1U});
        ASSERT_TRUE(counter);
        EXPECT_EQ(*counter, 3U);
        auto state = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
        ASSERT_TRUE(state);
        EXPECT_EQ(state->producer_epoch, 3U);
        EXPECT_EQ(state->last_producer_sequence, 0U);
        EXPECT_TRUE(state->subscriber_sequence_state_reset);
        // A cutover floor authorizes only its immediate successor, never an arbitrary first EOS.
        auto gap = RuntimeChannelSessionTestAccess::store_end_of_stream(**session, PortId{2U}, 4U);
        ASSERT_FALSE(gap);
        EXPECT_EQ(gap.error().code, ErrorCode::kProtocolError);
        ASSERT_TRUE((*session)->refresh_connections());
        auto active_generation =
            RuntimeChannelSessionTestAccess::output_active_generation(**session, PortId{1U});
        ASSERT_TRUE(active_generation);
        const auto generation = *active_generation;
        ASSERT_TRUE((*session)->arm());
        if (replacement_data) {
          ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, generation, {}));
          ASSERT_TRUE(fixture.source.output.emit(std::int32_t{103}));
          ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, generation, true));
          auto event = wait_for_event(**session, **binding);
          ASSERT_TRUE(event) << runtime_error_message(event.error());
          auto record = (*session)->temporal_sample(event->port, event->source_sequence);
          ASSERT_TRUE(record) << runtime_error_message(record.error());
          EXPECT_EQ(record->source_sequence, 3U);
          EXPECT_EQ(record->producer_epoch, 3U);
          ASSERT_TRUE((*binding)->retire(event->source_sequence));
        }
        // Key assertion: close emits one terminal at the next sequence, and repeated close
        // does not enqueue a duplicate terminal.
        auto closed = (*session)->close_owner_outputs(VertexId{1U});
        ASSERT_TRUE(closed) << runtime_error_message(closed.error());
        auto terminal = wait_for_event(**session, **binding);
        ASSERT_TRUE(terminal) << runtime_error_message(terminal.error());
        EXPECT_EQ(terminal->kind, RuntimeEventKind::kInputEndOfStream);
        ASSERT_TRUE((*binding)->retire(terminal->source_sequence));
        ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
        auto terminal_poll = (*binding)->poll(TimePoint{1, ClockId{1U}});
        ASSERT_FALSE(terminal_poll);
        EXPECT_EQ(terminal_poll.error().code, ErrorCode::kClosed);
        RuntimeEvent duplicate;
        auto drained = (*binding)->drain(std::span<RuntimeEvent>{&duplicate, 1U});
        ASSERT_TRUE(drained);
        EXPECT_EQ(*drained, 0U);
        (*session)->cancel();
        ASSERT_TRUE((*session)->join());
        state = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
        ASSERT_TRUE(state);
        EXPECT_EQ(state->end_of_stream_transport_sequence, replacement_data ? 4U : 3U);
        counter = RuntimeChannelSessionTestAccess::output_next_sequence(**session, PortId{1U});
        ASSERT_TRUE(counter);
        EXPECT_EQ(*counter, replacement_data ? 5U : 4U);
      }
    }
  }
}

// Exhaustion is a session-lifetime bound, even after otherwise valid resets. Reject close before
// any terminal endpoint effect and preserve the last permitted data coordinate.
// REQ: RUNTIME-052
TEST(RuntimeChannelBinding, OutputSequenceExhaustionSurvivesEveryCutover) {
  // Source --output--[direct retained]--> input-- sink
  // Key setup: publish the last permitted data sequence, then apply a full reset, source reset,
  // or lifecycle restart while keeping the same session.
  constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
  for (const auto reset_kind :
       {SequenceCutover::kFull, SequenceCutover::kSource, SequenceCutover::kRestart}) {
    SessionFixture fixture;
    fixture.use_direct_retained_delivery();
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session);
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_output_next_sequence(
        **session, PortId{1U}, maximum - 1U));
    ASSERT_TRUE((*session)->refresh_connections());
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
    ASSERT_TRUE(fixture.source.output.emit(std::int32_t{17}));
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
    auto before = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    ASSERT_TRUE(before);
    EXPECT_EQ(before->last_producer_sequence, maximum - 1U);
    auto reset = apply_sequence_cutover(**session, reset_kind);
    ASSERT_TRUE(reset) << runtime_error_message(reset.error());
    auto counter = RuntimeChannelSessionTestAccess::output_next_sequence(**session, PortId{1U});
    ASSERT_TRUE(counter);
    EXPECT_EQ(*counter, maximum);
    const std::uint64_t generation = reset_kind == SequenceCutover::kFull ? 1U : 2U;
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, generation, {}));
    // Key assertion: emit and close still report exhaustion after the cutover; neither wraps
    // the output counter nor delivers EOS to the receiver.
    auto emit = fixture.source.output.emit(std::int32_t{18});
    ASSERT_FALSE(emit);
    EXPECT_EQ(emit.error().code, ErrorCode::kResourceExhausted);
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, generation, true));
    auto close = (*session)->close_owner_outputs(VertexId{1U});
    ASSERT_FALSE(close);
    EXPECT_EQ(close.error().code, ErrorCode::kResourceExhausted);
    auto state = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    ASSERT_TRUE(state);
    EXPECT_FALSE(state->end_of_stream_received);
    counter = RuntimeChannelSessionTestAccess::output_next_sequence(**session, PortId{1U});
    ASSERT_TRUE(counter);
    EXPECT_EQ(*counter, maximum);
  }
}

// A full reset is one session-wide generation transition. Exhaustion in a later subscriber must
// be discovered before an earlier subscriber loses its retained payload or advances its epoch.
// REQ: RUNTIME-020
// REQ: RUNTIME-024
// REQ: RUNTIME-052
TEST(RuntimeChannelBinding, FullResetPreflightsEveryProducerEpochBeforeMutation) {
  // Source --output--+--blocking----> input-- sink (owner 2)
  //                 +--drop-oldest-> input-- second_sink (owner 3)
  // Key setup: retain one publication in both receivers, then exhaust the later receiver's
  // producer epoch before requesting a full reset.
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
  ASSERT_TRUE(first_binding) << runtime_error_message(first_binding.error());
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
  ASSERT_TRUE(second_binding) << runtime_error_message(second_binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{71}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  auto first_event = wait_for_event(**session, **first_binding);
  ASSERT_TRUE(first_event) << runtime_error_message(first_event.error());
  auto second_event = wait_for_event(**session, **second_binding);
  ASSERT_TRUE(second_event) << runtime_error_message(second_event.error());
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());

  auto second_before = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{3U});
  ASSERT_TRUE(second_before) << runtime_error_message(second_before.error());
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_input_reset_counters(
      **session,
      PortId{3U},
      second_before->active_generation,
      second_before->view_generation,
      std::numeric_limits<std::uint64_t>::max()));
  auto first_before = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(first_before) << runtime_error_message(first_before.error());
  second_before = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{3U});
  ASSERT_TRUE(second_before) << runtime_error_message(second_before.error());
  ASSERT_EQ(first_before->occupied_samples, 1U);
  ASSERT_EQ(second_before->occupied_samples, 1U);

  // Key assertion: epoch exhaustion rejects the whole reset without changing either receiver's
  // generations or retained sample, and the output's next sequence remains 2.
  auto reset = (*session)->reset();
  ASSERT_FALSE(reset);
  EXPECT_EQ(reset.error().code, ErrorCode::kResourceExhausted);
  auto first_after = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  auto second_after = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{3U});
  ASSERT_TRUE(first_after) << runtime_error_message(first_after.error());
  ASSERT_TRUE(second_after) << runtime_error_message(second_after.error());
  EXPECT_EQ(first_after->active_generation, first_before->active_generation);
  EXPECT_EQ(first_after->view_generation, first_before->view_generation);
  EXPECT_EQ(first_after->producer_epoch, first_before->producer_epoch);
  EXPECT_EQ(first_after->occupied_samples, first_before->occupied_samples);
  EXPECT_EQ(second_after->active_generation, second_before->active_generation);
  EXPECT_EQ(second_after->view_generation, second_before->view_generation);
  EXPECT_EQ(second_after->producer_epoch, second_before->producer_epoch);
  EXPECT_EQ(second_after->occupied_samples, second_before->occupied_samples);
  EXPECT_TRUE((*session)->temporal_sample(first_event->port, first_event->source_sequence));
  EXPECT_TRUE((*session)->temporal_sample(second_event->port, second_event->source_sequence));
  auto output_counter =
      RuntimeChannelSessionTestAccess::output_next_sequence(**session, PortId{1U});
  ASSERT_TRUE(output_counter);
  EXPECT_EQ(*output_counter, 2U);
}

// A frozen cutover predecessor cannot incorporate post-cutover data the receiver has not seen.
// REQ: RUNTIME-052
// REQ: CHAN-136
TEST(RuntimeChannelBinding, CutoverFloorCannotHideMissingReplacementData) {
  // Source --output--[transport]--> input-- sink
  // Key setup: publish sequences 1 and 2, restart, then leave replacement sequence 3 queued
  // while injecting EOS at sequence 4 directly into receiver bookkeeping.
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session);
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{1}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{2}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_TRUE(apply_sequence_cutover(**session, SequenceCutover::kRestart));
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 2U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{3}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 2U, true));
  // Key assertion: the frozen predecessor rejects EOS 4 until the receiver consumes data 3;
  // the source's newer output counter cannot conceal the missing replacement record.
  auto gap = RuntimeChannelSessionTestAccess::store_end_of_stream(**session, PortId{2U}, 4U);
  ASSERT_FALSE(gap);
  EXPECT_EQ(gap.error().code, ErrorCode::kProtocolError);
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::store_end_of_stream(**session, PortId{2U}, 4U));
}

// Whole-session cleanup and ordinary slot retirement must erase the same complete set of slot
// claims. This regression saturates every independently reset field before exercising the public
// administrative reset.
// REQ: RUNTIME-020
// REQ: RUNTIME-024
TEST(RuntimeChannelBinding, FullResetUsesCanonicalResetForEveryRetainedSlotField) {
  // Source --output--> input-- sink
  // Key setup: retain one typed sample, then saturate every private slot and subscriber reset
  // field through the source-tree conformance seam.
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
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{71}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  auto event = wait_for_event(**session, **binding);
  ASSERT_TRUE(event) << runtime_error_message(event.error());
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
  ASSERT_TRUE(
      RuntimeChannelSessionTestAccess::saturate_single_occupied_reset_state(**session, PortId{2U}));
  const auto before = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(before) << runtime_error_message(before.error());
  ASSERT_EQ(before->uncleared_sample_slots, 1U);
  ASSERT_FALSE(before->subscriber_sequence_state_reset);

  ASSERT_TRUE((*session)->reset());

  // Key assertion: the canonical slot primitive clears every payload/claim/temporal field, while
  // clear_samples() retains its subscriber-level sequence and EOS reset obligations.
  const auto after = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(after) << runtime_error_message(after.error());
  EXPECT_EQ(after->occupied_samples, 0U);
  EXPECT_EQ(after->pending_direct_ingress_samples, 0U);
  EXPECT_EQ(after->uncleared_sample_slots, 0U);
  EXPECT_EQ(after->next_arrival_sequence, 1U);
  EXPECT_FALSE(after->end_of_stream_received);
  EXPECT_TRUE(after->subscriber_sequence_state_reset);
}

// Source-tree reset-counter injection is a cold-session seam. Rejecting a live session prevents
// test-only generation writes from racing the pump or an armed activation.
TEST(RuntimeChannelBinding, ResetCounterInjectionRejectsLiveSession) {
  // Source --output--> input-- sink
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  // Key setup: attempt to mutate input reset counters after startup has armed the session.
  const auto injected =
      RuntimeChannelSessionTestAccess::set_input_reset_counters(**session, PortId{2U}, 1U, 1U, 1U);

  // Key assertion: the live session rejects the source-tree-only mutation as backpressured.
  ASSERT_FALSE(injected);
  EXPECT_EQ(injected.error().code, ErrorCode::kBackpressured);
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// Scoped reset/restart classify retryable quiescence failures separately from terminal bounded-
// counter exhaustion. Both operations must make this classification before advancing any counter.
// REQ: RUNTIME-020
// REQ: RUNTIME-024
TEST(RuntimeChannelBinding, ScopedResetAndRestartDistinguishCounterExhaustion) {
  constexpr auto exhausted = std::numeric_limits<std::uint64_t>::max();
  const std::array affected{VertexId{1U}, VertexId{2U}};

  const auto verify_input_case = [&](const std::uint64_t active_generation,
                                     const std::uint64_t view_generation,
                                     const std::uint64_t producer_epoch,
                                     const bool restart) {
    SessionFixture fixture;
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_input_reset_counters(
        **session, PortId{2U}, active_generation, view_generation, producer_epoch));
    auto first = restart ? (*session)->commit_restart(VertexId{1U}, affected)
                         : (*session)->reset(VertexId{2U});
    ASSERT_FALSE(first);
    EXPECT_EQ(first.error().code, ErrorCode::kResourceExhausted);
    auto second = restart ? (*session)->commit_restart(VertexId{1U}, affected)
                          : (*session)->reset(VertexId{2U});
    ASSERT_FALSE(second);
    EXPECT_EQ(second.error().code, ErrorCode::kResourceExhausted);
  };
  for (const bool restart : {false, true}) {
    verify_input_case(exhausted, 1U, 1U, restart);
    verify_input_case(1U, exhausted, 1U, restart);
    verify_input_case(1U, 1U, exhausted, restart);

    SessionFixture fixture;
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_output_active_generation(
        **session, PortId{1U}, exhausted));
    auto first = restart ? (*session)->commit_restart(VertexId{1U}, affected)
                         : (*session)->reset(VertexId{1U});
    ASSERT_FALSE(first);
    EXPECT_EQ(first.error().code, ErrorCode::kResourceExhausted);
    auto second = restart ? (*session)->commit_restart(VertexId{1U}, affected)
                          : (*session)->reset(VertexId{1U});
    ASSERT_FALSE(second);
    EXPECT_EQ(second.error().code, ErrorCode::kResourceExhausted);
  }
}

// A coalesced lifecycle transaction may name an Operator in both the restarted-owner set and the
// affected-receiver set. The overlap must not double-advance the Operator's input view, a shared
// subscriber epoch, or an output generation owned by another restarted member.
// REQ: RUNTIME-020
// REQ: RUNTIME-024
TEST(RuntimeChannelBinding, CommitsOverlappingMultiOwnerRestartExactlyOnce) {
  SessionFixture fixture;
  fixture.add_mixed_policy_fanout();
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());

  const auto first_before =
      RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  const auto second_before =
      RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{3U});
  const auto output_before =
      RuntimeChannelSessionTestAccess::output_active_generation(**session, PortId{1U});
  ASSERT_TRUE(first_before) << runtime_error_message(first_before.error());
  ASSERT_TRUE(second_before) << runtime_error_message(second_before.error());
  ASSERT_TRUE(output_before) << runtime_error_message(output_before.error());

  const std::array restarted{VertexId{1U}, VertexId{2U}};
  const std::array affected{VertexId{1U}, VertexId{2U}, VertexId{3U}};
  ASSERT_TRUE((*session)->commit_restart(restarted, affected));

  const auto first_after =
      RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  const auto second_after =
      RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{3U});
  const auto output_after =
      RuntimeChannelSessionTestAccess::output_active_generation(**session, PortId{1U});
  ASSERT_TRUE(first_after) << runtime_error_message(first_after.error());
  ASSERT_TRUE(second_after) << runtime_error_message(second_after.error());
  ASSERT_TRUE(output_after) << runtime_error_message(output_after.error());
  EXPECT_EQ(first_after->active_generation, first_before->active_generation + 1U);
  EXPECT_EQ(first_after->view_generation, first_before->view_generation + 1U);
  EXPECT_EQ(first_after->producer_epoch, first_before->producer_epoch + 1U);
  EXPECT_EQ(second_after->active_generation, second_before->active_generation + 1U);
  EXPECT_EQ(second_after->view_generation, second_before->view_generation + 1U);
  EXPECT_EQ(second_after->producer_epoch, second_before->producer_epoch + 1U);
  EXPECT_EQ(*output_after, *output_before + 1U);
}

// Restart-set validation is a cold transaction boundary. Empty, invalid, duplicate, or
// non-subset owner sets must be rejected before any input, subscriber, or output generation is
// published.
// REQ: RUNTIME-020
// REQ: RUNTIME-024
// REQ: RUNTIME-052
TEST(RuntimeChannelBinding, RejectsMalformedMultiOwnerRestartSetsWithoutMutation) {
  // Source (owner 1) --output--> input-- sink (owner 2)
  // Key setup: use a fresh session to submit empty, duplicate, invalid, and non-subset
  // restart/affected owner sets.
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());

  const auto input_before =
      RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  const auto output_before =
      RuntimeChannelSessionTestAccess::output_active_generation(**session, PortId{1U});
  ASSERT_TRUE(input_before) << runtime_error_message(input_before.error());
  ASSERT_TRUE(output_before) << runtime_error_message(output_before.error());

  // Key assertion: every malformed set reports invalid argument and preserves input state and
  // output generation; the final output counter still starts at sequence 1.
  const auto expect_unchanged = [&] {
    const auto input_after =
        RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    const auto output_after =
        RuntimeChannelSessionTestAccess::output_active_generation(**session, PortId{1U});
    ASSERT_TRUE(input_after) << runtime_error_message(input_after.error());
    ASSERT_TRUE(output_after) << runtime_error_message(output_after.error());
    EXPECT_EQ(input_after->active_generation, input_before->active_generation);
    EXPECT_EQ(input_after->view_generation, input_before->view_generation);
    EXPECT_EQ(input_after->producer_epoch, input_before->producer_epoch);
    EXPECT_EQ(input_after->next_arrival_sequence, input_before->next_arrival_sequence);
    EXPECT_EQ(input_after->next_global_arrival_order, input_before->next_global_arrival_order);
    EXPECT_EQ(input_after->occupied_samples, input_before->occupied_samples);
    EXPECT_EQ(*output_after, *output_before);
  };
  const auto expect_invalid =
      [&](const std::span<const VertexId> restarted, const std::span<const VertexId> affected) {
        auto result = (*session)->commit_restart(restarted, affected);
        ASSERT_FALSE(result);
        EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
        expect_unchanged();
      };

  const std::array valid_restarted{VertexId{1U}};
  const std::array valid_affected{VertexId{1U}, VertexId{2U}};
  const std::array duplicate_restarted{VertexId{1U}, VertexId{1U}};
  const std::array duplicate_affected{VertexId{1U}, VertexId{2U}, VertexId{2U}};
  const std::array invalid_restarted{VertexId{}};
  const std::array missing_restarted_owner{VertexId{3U}};
  expect_invalid({}, valid_affected);
  expect_invalid(valid_restarted, {});
  expect_invalid(duplicate_restarted, valid_affected);
  expect_invalid(valid_restarted, duplicate_affected);
  expect_invalid(invalid_restarted, valid_affected);
  expect_invalid(missing_restarted_owner, valid_affected);
  auto output_counter =
      RuntimeChannelSessionTestAccess::output_next_sequence(**session, PortId{1U});
  ASSERT_TRUE(output_counter);
  EXPECT_EQ(*output_counter, 1U);
}

// The restart union is one transaction even when a later affected receiver owns the exhausted
// counter. Preflight must discover that failure before advancing an earlier input, either
// subscriber epoch, or the restarted producer's output generation.
// REQ: RUNTIME-020
// REQ: RUNTIME-024
TEST(RuntimeChannelBinding, PreflightsCompleteMultiOwnerRestartUnionBeforeMutation) {
  SessionFixture fixture;
  fixture.add_mixed_policy_fanout();
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());

  const auto second_initial =
      RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{3U});
  ASSERT_TRUE(second_initial) << runtime_error_message(second_initial.error());
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_input_reset_counters(
      **session,
      PortId{3U},
      std::numeric_limits<std::uint64_t>::max(),
      second_initial->view_generation,
      second_initial->producer_epoch));
  const auto first_before =
      RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  const auto second_before =
      RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{3U});
  const auto output_before =
      RuntimeChannelSessionTestAccess::output_active_generation(**session, PortId{1U});
  ASSERT_TRUE(first_before) << runtime_error_message(first_before.error());
  ASSERT_TRUE(second_before) << runtime_error_message(second_before.error());
  ASSERT_TRUE(output_before) << runtime_error_message(output_before.error());

  const std::array restarted{VertexId{1U}, VertexId{2U}};
  const std::array affected{VertexId{1U}, VertexId{2U}, VertexId{3U}};
  auto restart = (*session)->commit_restart(restarted, affected);
  ASSERT_FALSE(restart);
  EXPECT_EQ(restart.error().code, ErrorCode::kResourceExhausted);

  const auto first_after =
      RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  const auto second_after =
      RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{3U});
  const auto output_after =
      RuntimeChannelSessionTestAccess::output_active_generation(**session, PortId{1U});
  ASSERT_TRUE(first_after) << runtime_error_message(first_after.error());
  ASSERT_TRUE(second_after) << runtime_error_message(second_after.error());
  ASSERT_TRUE(output_after) << runtime_error_message(output_after.error());
  EXPECT_EQ(first_after->active_generation, first_before->active_generation);
  EXPECT_EQ(first_after->view_generation, first_before->view_generation);
  EXPECT_EQ(first_after->producer_epoch, first_before->producer_epoch);
  EXPECT_EQ(second_after->active_generation, second_before->active_generation);
  EXPECT_EQ(second_after->view_generation, second_before->view_generation);
  EXPECT_EQ(second_after->producer_epoch, second_before->producer_epoch);
  EXPECT_EQ(*output_after, *output_before);
}

// Administrative reset has no lifecycle fence authorizing transport consumption. A queued sample
// therefore keeps reset retryable and remains available to the ordinary pump afterward.
// REQ: RUNTIME-020
// REQ: RUNTIME-024
TEST(RuntimeChannelBinding, PendingTransportBackpressuresAdministrativeResetWithoutConsumption) {
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
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{83}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  auto reset = (*session)->reset(VertexId{1U});
  ASSERT_FALSE(reset);
  EXPECT_EQ(reset.error().code, ErrorCode::kBackpressured);

  ASSERT_TRUE((*session)->arm());
  auto event = wait_for_event(**session, **binding);
  ASSERT_TRUE(event) << runtime_error_message(event.error());
  auto temporal = (*session)->temporal_sample(event->port, event->source_sequence);
  ASSERT_TRUE(temporal) << runtime_error_message(temporal.error());
  EXPECT_EQ(temporal->source_sequence, 1U);
  ASSERT_TRUE((*binding)->retire(event->source_sequence));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// Full reset applies one session-wide epoch transition, but queued transport is outside the
// resettable typed cache. It must preflight that transport before clearing any subscriber or
// advancing any counter, and a retryable refusal must leave the payload for the ordinary pump.
// REQ: RUNTIME-020
// REQ: RUNTIME-024
// REQ: RUNTIME-052
TEST(RuntimeChannelBinding, FullResetPendingTransportBackpressuresBeforeAnyMutation) {
  // Source --output--[transport]--> input-- sink
  // Key setup: leave one publication queued with the pump unarmed, then request a full reset.
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
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{89}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  auto before = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(before) << runtime_error_message(before.error());
  // Key assertion: backpressure preserves receiver state and the output counter; the ordinary
  // pump can still deliver the queued sequence 1 under the original producer epoch.
  auto reset = (*session)->reset();
  ASSERT_FALSE(reset);
  EXPECT_EQ(reset.error().code, ErrorCode::kBackpressured);
  auto after = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(after) << runtime_error_message(after.error());
  EXPECT_EQ(after->active_generation, before->active_generation);
  EXPECT_EQ(after->view_generation, before->view_generation);
  EXPECT_EQ(after->producer_epoch, before->producer_epoch);
  EXPECT_EQ(after->next_arrival_sequence, before->next_arrival_sequence);
  EXPECT_EQ(after->next_global_arrival_order, before->next_global_arrival_order);
  EXPECT_EQ(after->occupied_samples, before->occupied_samples);

  ASSERT_TRUE((*session)->arm());
  auto event = wait_for_event(**session, **binding);
  ASSERT_TRUE(event) << runtime_error_message(event.error());
  auto temporal = (*session)->temporal_sample(event->port, event->source_sequence);
  ASSERT_TRUE(temporal) << runtime_error_message(temporal.error());
  EXPECT_EQ(temporal->source_sequence, 1U);
  EXPECT_EQ(temporal->producer_epoch, before->producer_epoch);
  ASSERT_TRUE((*binding)->retire(event->source_sequence));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
  auto output_counter =
      RuntimeChannelSessionTestAccess::output_next_sequence(**session, PortId{1U});
  ASSERT_TRUE(output_counter);
  EXPECT_EQ(*output_counter, 2U);
}

// Accepted lifecycle restart owns a joined process-local pump and a fenced publisher. It may
// consume exactly the compiler-frozen subscriber depth before publishing the new producer epoch;
// the first post-restart activation must therefore observe only the new-generation payload.
// REQ: RUNTIME-019
// REQ: RUNTIME-020
// REQ: RUNTIME-024
// REQ: RUNTIME-052
TEST(RuntimeChannelBinding, RestartDiscardsExactFrozenTransportBoundBeforeGenerationAdvance) {
  // Source --output--[blocking transport queue, depth 8]--> input-- sink
  // Key setup: fill the eight-record transport bound before restart, then emit a distinct
  // payload in the next generation.
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
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  for (std::int32_t value = 100; value < 108; ++value) {
    ASSERT_TRUE(fixture.source.output.emit(value));
  }
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  const std::array affected{VertexId{1U}, VertexId{2U}};
  ASSERT_TRUE((*session)->commit_restart(VertexId{1U}, affected));
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 2U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{999}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 2U, true));

  // Key assertion: the first received record is payload 999 at epoch 2 and sequence 9, proving
  // restart discarded the old queue without restarting the source's sequence history.
  auto event = wait_for_event(**session, **binding);
  ASSERT_TRUE(event) << runtime_error_message(event.error());
  auto temporal = (*session)->temporal_sample(event->port, event->source_sequence);
  ASSERT_TRUE(temporal) << runtime_error_message(temporal.error());
  EXPECT_EQ(temporal->source_sequence, 9U);
  EXPECT_EQ(temporal->producer_epoch, 2U);
  const std::array records{*temporal};
  const std::array selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = records,
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 2U, selection));
  auto value = fixture.sink.input.receive_data();
  ASSERT_TRUE(value) << runtime_error_message(value.error());
  EXPECT_EQ(*value, 999);
  ASSERT_TRUE((*binding)->retire(event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 2U, true));
  ASSERT_TRUE((*session)->retire_samples(PortId{2U}, records));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// Once restart consumes an old-generation transport prefix, a later state-transition failure is
// irreversible. The session must retain the first terminal error and reject every recovery entry
// point before any input, producer-epoch, or output generation is published.
// REQ: RUNTIME-020
// REQ: RUNTIME-024
TEST(RuntimeChannelBinding, RestartPostDiscardFailureTerminalizesBeforePublishingGenerations) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{67}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  auto before = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(before) << runtime_error_message(before.error());
  auto output_before =
      RuntimeChannelSessionTestAccess::output_active_generation(**session, PortId{1U});
  ASSERT_TRUE(output_before) << runtime_error_message(output_before.error());

  ASSERT_TRUE(RuntimeChannelSessionTestAccess::fail_next_restart_after_transport_discard(
      **session, ErrorCode::kProtocolError));
  auto duplicate_fault = RuntimeChannelSessionTestAccess::fail_next_restart_after_transport_discard(
      **session, ErrorCode::kFailure);
  ASSERT_FALSE(duplicate_fault);
  EXPECT_EQ(duplicate_fault.error().code, ErrorCode::kBackpressured);
  const std::array affected{VertexId{1U}, VertexId{2U}};
  auto restart = (*session)->commit_restart(VertexId{1U}, affected);
  ASSERT_FALSE(restart);
  EXPECT_EQ(restart.error().code, ErrorCode::kProtocolError);
  auto terminal = (*session)->status();
  ASSERT_FALSE(terminal);
  EXPECT_EQ(terminal.error().code, ErrorCode::kProtocolError);

  auto retry = (*session)->commit_restart(VertexId{1U}, affected);
  ASSERT_FALSE(retry);
  EXPECT_EQ(retry.error().code, ErrorCode::kProtocolError);
  auto scoped_reset = (*session)->reset(VertexId{1U});
  ASSERT_FALSE(scoped_reset);
  EXPECT_EQ(scoped_reset.error().code, ErrorCode::kProtocolError);
  auto full_reset = (*session)->reset();
  ASSERT_FALSE(full_reset);
  EXPECT_EQ(full_reset.error().code, ErrorCode::kProtocolError);
  auto rearm = (*session)->arm();
  ASSERT_FALSE(rearm);
  EXPECT_EQ(rearm.error().code, ErrorCode::kProtocolError);

  auto after = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(after) << runtime_error_message(after.error());
  auto output_after =
      RuntimeChannelSessionTestAccess::output_active_generation(**session, PortId{1U});
  ASSERT_TRUE(output_after) << runtime_error_message(output_after.error());
  EXPECT_EQ(after->active_generation, before->active_generation);
  EXPECT_EQ(after->view_generation, before->view_generation);
  EXPECT_EQ(after->producer_epoch, before->producer_epoch);
  EXPECT_EQ(after->next_arrival_sequence, before->next_arrival_sequence);
  EXPECT_EQ(after->next_global_arrival_order, before->next_global_arrival_order);
  EXPECT_EQ(after->occupied_samples, before->occupied_samples);
  EXPECT_EQ(*output_after, *output_before);
}

// A nonempty lifecycle scope that names no materialized Channel state is an inconsistent
// Runtime/plan join. The caller must not treat this as a successful empty restart and publish new
// selector or readiness generations.
// REQ: RUNTIME-020
// REQ: RUNTIME-024
TEST(RuntimeChannelBinding, RestartRejectsAffectedScopeWithoutMaterializedChannelState) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());

  const std::array affected{VertexId{3U}};
  auto restart = (*session)->commit_restart(VertexId{3U}, affected);
  ASSERT_FALSE(restart);
  EXPECT_EQ(restart.error().code, ErrorCode::kNotFound);
  EXPECT_TRUE((*session)->status());
}

// REQ: RUNTIME-018
// REQ: RUNTIME-020
// REQ: RUNTIME-024
// REQ: RUNTIME-034
TEST(RuntimeChannelBinding, RetainsReuseLastUntilTargetScopedReset) {
  SessionFixture fixture;
  fixture.enable_reuse_last();
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
  const SampleMetadata metadata{.capture_timestamp_ns = 10, .source_clock_id = ClockId{1U}};
  ASSERT_TRUE(fixture.source.output.emit(Sample<std::int32_t>{55, metadata}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  auto event = wait_for_event(**session, **binding);
  ASSERT_TRUE(event) << runtime_error_message(event.error());
  auto temporal = (*session)->temporal_sample(event->port, event->source_sequence);
  ASSERT_TRUE(temporal);
  auto direct = RuntimeInputSelectionState::create(fixture.plan.input_selections.front());
  ASSERT_TRUE(direct);
  ASSERT_TRUE(direct->push(*temporal));
  auto first_decision =
      direct->begin(temporal->capture_time, temporal->arrival_sequence, temporal->producer_epoch);
  ASSERT_TRUE(first_decision);
  ASSERT_EQ(first_decision->selected().size(), 1U);
  EXPECT_EQ(first_decision->selected().front(), *temporal);
  const std::array records{*temporal};
  const std::array selected{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = first_decision->selection(),
      .samples = first_decision->selected(),
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selected));
  auto first = fixture.sink.input.receive_data();
  ASSERT_TRUE(first);
  EXPECT_EQ(*first, 55);
  ASSERT_TRUE((*binding)->retire(event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  ASSERT_TRUE(first_decision->finish(true));

  auto reuse_decision = direct->begin(
      TimePoint{20, ClockId{1U}}, temporal->arrival_sequence, temporal->producer_epoch);
  ASSERT_TRUE(reuse_decision);
  EXPECT_EQ(reuse_decision->selection().state, InputState::kReused);
  ASSERT_EQ(reuse_decision->selected().size(), 1U);
  EXPECT_EQ(reuse_decision->selected().front(), *temporal);
  const InputSelection reuse_selection = reuse_decision->selection();
  const std::array reused{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = reuse_selection,
      .samples = reuse_decision->selected(),
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, reused));
  auto second = fixture.sink.input.receive_data();
  ASSERT_TRUE(second);
  EXPECT_EQ(*second, 55);
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  ASSERT_TRUE(reuse_decision->finish(true));

  auto premature_retirement = (*session)->retire_samples(PortId{2U}, records);
  ASSERT_FALSE(premature_retirement);
  EXPECT_EQ(premature_retirement.error().code, ErrorCode::kBackpressured);
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
  ASSERT_TRUE((*session)->reset(VertexId{2U}));
  const std::array stale_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = reuse_selection,
      .samples = records,
      .destructive = true,
  }};
  auto stale_reuse = (*session)->prepare_activation(VertexId{2U}, 3U, stale_selection);
  ASSERT_FALSE(stale_reuse);
  EXPECT_EQ(stale_reuse.error().code, ErrorCode::kStaleEndpoint);
}

// REQ: RUNTIME-018
// REQ: RUNTIME-020
// REQ: RUNTIME-024
// REQ: RUNTIME-034
TEST(RuntimeChannelBinding, SourceScopedResetPurgesDownstreamReuseLastStorage) {
  SessionFixture fixture;
  fixture.enable_reuse_last();
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
  const SampleMetadata metadata{.capture_timestamp_ns = 10, .source_clock_id = ClockId{1U}};
  ASSERT_TRUE(fixture.source.output.emit(Sample<std::int32_t>{77, metadata}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  auto event = wait_for_event(**session, **binding);
  ASSERT_TRUE(event) << runtime_error_message(event.error());
  auto temporal = (*session)->temporal_sample(event->port, event->source_sequence);
  ASSERT_TRUE(temporal);
  const std::array records{*temporal};
  const std::array selected{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = records,
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selected));
  auto value = fixture.sink.input.receive_data();
  ASSERT_TRUE(value);
  EXPECT_EQ(*value, 77);
  ASSERT_TRUE((*binding)->retire(event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  auto retained = (*session)->retire_samples(PortId{2U}, records);
  ASSERT_FALSE(retained);
  EXPECT_EQ(retained.error().code, ErrorCode::kBackpressured);

  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
  const std::array affected{VertexId{1U}, VertexId{2U}};
  ASSERT_TRUE((*session)->commit_restart(VertexId{1U}, affected));
  auto purged = (*session)->temporal_sample(event->port, event->source_sequence);
  ASSERT_FALSE(purged);
  EXPECT_EQ(purged.error().code, ErrorCode::kNotFound);
  auto stale_reuse = (*session)->prepare_activation(VertexId{2U}, 1U, selected);
  ASSERT_FALSE(stale_reuse);
  EXPECT_EQ(stale_reuse.error().code, ErrorCode::kStaleEndpoint);
}

// An owning scalar sample may outlive its activation, but its finite claim fences every reset form
// that would invalidate the claimed generation. Each operation preflights the complete affected
// set and returns before changing cache or generation state; dropping the claim makes it retryable.
// REQ: RUNTIME-017
// REQ: RUNTIME-020
// REQ: RUNTIME-024
TEST(RuntimeChannelBinding, EscapedReadClaimFencesResetAndRestartBeforeMutation) {
  enum class ResetKind : std::uint8_t { kFull, kSourceScoped, kRestart };
  for (const ResetKind kind : {ResetKind::kFull, ResetKind::kSourceScoped, ResetKind::kRestart}) {
    SCOPED_TRACE(static_cast<std::uint32_t>(kind));
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
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
    ASSERT_TRUE(fixture.source.output.emit(std::int32_t{47}));
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
    auto event = wait_for_event(**session, **binding);
    ASSERT_TRUE(event) << runtime_error_message(event.error());
    auto temporal = (*session)->temporal_sample(event->port, event->source_sequence);
    ASSERT_TRUE(temporal) << runtime_error_message(temporal.error());
    const std::array records{*temporal};
    const std::array selected{RuntimeChannelActivationSelection{
        .port = PortId{2U},
        .selection = {InputState::kSelected, SelectionReason::kNone},
        .samples = records,
        .destructive = true,
    }};
    ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selected));
    auto received = fixture.sink.input.receive();
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    // Key setup: escape the received sample so its read claim outlives the activation boundary.
    std::optional<Sample<std::int32_t>> escaped{std::move(*received)};
    ASSERT_TRUE((*binding)->retire(event->source_sequence));
    ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
    (*session)->cancel();
    ASSERT_TRUE((*session)->join());

    const std::array affected{VertexId{1U}, VertexId{2U}};
    const auto apply_reset = [&]() -> expected<void, Error> {
      switch (kind) {
        case ResetKind::kFull:
          return (*session)->reset();
        case ResetKind::kSourceScoped:
          return (*session)->reset(VertexId{1U});
        case ResetKind::kRestart:
          return (*session)->commit_restart(VertexId{1U}, affected);
      }
      return make_unexpected(Error{ErrorCode::kProtocolError});
    };
    auto before = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    ASSERT_TRUE(before) << runtime_error_message(before.error());
    auto fenced = apply_reset();
    // Key assertion: all reset/restart forms are backpressured while the read claim is live.
    ASSERT_FALSE(fenced);
    EXPECT_EQ(fenced.error().code, ErrorCode::kBackpressured);
    auto after = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    ASSERT_TRUE(after) << runtime_error_message(after.error());
    EXPECT_EQ(after->active_generation, before->active_generation);
    EXPECT_EQ(after->view_generation, before->view_generation);
    EXPECT_EQ(after->producer_epoch, before->producer_epoch);
    EXPECT_EQ(after->occupied_samples, before->occupied_samples);

    escaped.reset();
    ASSERT_TRUE(apply_reset());
  }
}

// A live generic loan is an activation-scoped authority. Restart cannot advance the output or
// receiver generations while that authority could still publish into the old generation. Once
// the owner cancels it and retires the activation, restart re-arms both sides. The first actual
// publication still uses one because the canceled loan consumed no sequence authority.
// REQ: RUNTIME-020
// REQ: RUNTIME-024
TEST(RuntimeChannelBinding, LiveGenericLoanFencesRestartUntilExplicitCancellation) {
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
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

  auto loan = fixture.source.output.allocate();
  ASSERT_TRUE(loan) << runtime_error_message(loan.error());
  loan->value() = 41;
  auto retire_with_live_loan = (*session)->retire_activation(VertexId{1U}, 1U, true);
  ASSERT_FALSE(retire_with_live_loan);
  EXPECT_EQ(retire_with_live_loan.error().code, ErrorCode::kBackpressured);

  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
  const std::array affected{VertexId{1U}, VertexId{2U}};
  auto reset_with_live_loan = (*session)->reset(VertexId{1U});
  ASSERT_FALSE(reset_with_live_loan);
  EXPECT_EQ(reset_with_live_loan.error().code, ErrorCode::kBackpressured);
  auto restart_with_live_loan = (*session)->commit_restart(VertexId{1U}, affected);
  ASSERT_FALSE(restart_with_live_loan);
  EXPECT_EQ(restart_with_live_loan.error().code, ErrorCode::kBackpressured);

  loan->cancel();
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_TRUE((*session)->commit_restart(VertexId{1U}, affected));
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 2U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{42}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 2U, true));

  auto event = wait_for_event(**session, **binding);
  ASSERT_TRUE(event) << runtime_error_message(event.error());
  auto temporal = (*session)->temporal_sample(event->port, event->source_sequence);
  ASSERT_TRUE(temporal) << runtime_error_message(temporal.error());
  EXPECT_EQ(temporal->source_sequence, 1U);
  EXPECT_EQ(temporal->producer_epoch, 2U);
  ASSERT_TRUE((*binding)->retire(event->source_sequence));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// REQ: RUNTIME-020
// REQ: RUNTIME-024
TEST(RuntimeChannelBinding, DestructorDestroysEndpointChildrenBeforeChannelRuntime) {
  SessionFixture fixture;
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
    ASSERT_TRUE((*session)->arm());
  }

  const std::vector expected{
      RuntimeChannelTeardownPhase::kPumpJoined,
      RuntimeChannelTeardownPhase::kEndpointChildrenDestroyed,
      RuntimeChannelTeardownPhase::kRuntimeDestroyed,
  };
  EXPECT_EQ(phases, expected);
}

// REQ: RUNTIME-018
// REQ: RUNTIME-020
// REQ: RUNTIME-024
TEST(RuntimeChannelBinding, DestructorJoinsPumpAndAllowsSameServiceReconstruction) {
  SessionFixture fixture;
  std::optional<Sample<std::int32_t>> escaped;
  {
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
    ASSERT_TRUE(fixture.source.output.emit(std::int32_t{9}));
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

    auto event = wait_for_event(**session, **binding);
    ASSERT_TRUE(event) << runtime_error_message(event.error());
    auto temporal = (*session)->temporal_sample(event->port, event->source_sequence);
    ASSERT_TRUE(temporal);
    const std::array records{*temporal};
    const std::array selected{RuntimeChannelActivationSelection{
        .port = PortId{2U},
        .selection = {InputState::kSelected, SelectionReason::kNone},
        .samples = records,
        .destructive = true,
    }};
    ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selected));
    auto sample = fixture.sink.input.receive();
    ASSERT_TRUE(sample);
    escaped.emplace(std::move(*sample));
    ASSERT_TRUE((*binding)->retire(event->source_sequence));
    ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  }

  ASSERT_TRUE(escaped);
  EXPECT_EQ(escaped->data, 9);

  auto reconstructed = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(reconstructed) << runtime_error_message(reconstructed.error());
  ASSERT_TRUE((*reconstructed)->refresh_connections());
  ASSERT_TRUE((*reconstructed)->arm());
  (*reconstructed)->cancel();
  EXPECT_TRUE((*reconstructed)->join());
}

}  // namespace
}  // namespace holoscan::detail
