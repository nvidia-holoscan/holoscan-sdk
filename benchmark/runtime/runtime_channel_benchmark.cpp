/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
#include <intrin.h>
#endif

#include <benchmark/benchmark.h>

#include <holoscan/channel/endpoint_interfaces.hpp>
#include <holoscan/channel/runtime.hpp>
#include <holoscan/channel/types.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "../common/benchmark_pod.hpp"
#include "compiler/execution_plan_state.hpp"
#include "core/operator_spec_compiler_access.hpp"
#include "core/source_id_internal.hpp"
#include "runtime/channel_runtime_binding_access.hpp"
#include "runtime/execution_lane.hpp"
#include "runtime/runtime_channel_binding.hpp"
#include "runtime/runtime_channel_binding_test_access.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "time/clock_registry_internal.hpp"

namespace {

template <typename PayloadT>
class ChannelSourceOperator final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(output).max_emits_per_compute(1024U);
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Output<PayloadT> output{"output"};
};

template <typename PayloadT>
class ChannelSinkOperator final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(input).queue_depth(1024U); }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Input<PayloadT> input{"input"};
};

template <typename PayloadT>
class ChannelBenchmarkFixtureT {
 public:
  using CreateResult =
      holoscan::expected<std::unique_ptr<ChannelBenchmarkFixtureT>, holoscan::Error>;

  [[nodiscard]] static CreateResult create(
      std::uint32_t input_count, std::uint32_t queue_depth, bool shared_sink_owner,
      holoscan::detail::ProcessLocalChannelDeliveryKind delivery_kind =
          holoscan::detail::ProcessLocalChannelDeliveryKind::kHoloMQ) {
    if (input_count == 0U || queue_depth == 0U) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kInvalidArgument});
    }
    auto fixture = std::unique_ptr<ChannelBenchmarkFixtureT>{
        new ChannelBenchmarkFixtureT{input_count, queue_depth, shared_sink_owner, delivery_kind}};
    auto session =
        holoscan::detail::RuntimeChannelSession::create(fixture->plan_, fixture->replay_);
    if (!session) {
      return holoscan::make_unexpected(std::move(session).error());
    }
    fixture->session_ = std::move(*session);
    // Production binds the plan default clock before arm, so every publication reads it inside the
    // timed emit interval. Leaving the session unbound would take the inert branch and silently
    // exclude that clock read from every measurement here.
    if (auto bound = fixture->bind_publish_clock(); !bound) {
      return holoscan::make_unexpected(std::move(bound).error());
    }
    return fixture;
  }

  [[nodiscard]] holoscan::detail::RuntimeChannelSession& session() noexcept { return *session_; }

  [[nodiscard]] holoscan::VertexId sink_owner() const noexcept { return holoscan::VertexId{2U}; }

  [[nodiscard]] holoscan::VertexId sink_owner(std::size_t index) const noexcept {
    return holoscan::VertexId{static_cast<std::uint64_t>(index) + 2U};
  }

  [[nodiscard]] holoscan::VertexId last_sink_owner() const noexcept { return last_sink_owner_; }

  [[nodiscard]] holoscan::PortId last_input_port() const noexcept { return input_ports_.back(); }

  [[nodiscard]] holoscan::PlanGenerationId plan_generation() const noexcept {
    return plan_.plan_generation;
  }

  [[nodiscard]] std::size_t burst_size() const noexcept {
    return input_ports_.size() * queue_depth_;
  }

  [[nodiscard]] std::uint64_t readiness_poll_count() const noexcept {
    return readiness_poll_count_;
  }

  void reset_readiness_poll_count() noexcept { readiness_poll_count_ = 0U; }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> arm() noexcept {
    auto refreshed = session_->refresh_connections();
    if (!refreshed) {
      return holoscan::make_unexpected(std::move(refreshed).error());
    }
    return session_->arm();
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> prepare_typed_pod_source(
      std::uint64_t generation) noexcept {
    auto prepared = session_->prepare_activation(holoscan::VertexId{1U}, generation, {});
    if (!prepared) {
      return holoscan::make_unexpected(holoscan::Error{prepared.error().code, "prepare_source"});
    }
    return {};
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> emit_typed_pod(PayloadT value) noexcept {
    auto emitted = source_.output.emit(value);
    if (!emitted) {
      return holoscan::make_unexpected(holoscan::Error{emitted.error().code, "emit"});
    }
    return {};
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> retire_typed_pod_source(
      std::uint64_t generation) noexcept {
    auto retired = session_->retire_activation(holoscan::VertexId{1U}, generation, true);
    if (!retired) {
      return holoscan::make_unexpected(holoscan::Error{retired.error().code, "retire_source"});
    }
    return {};
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> publish_typed_pod_before_cut(
      std::uint64_t generation, PayloadT value) noexcept {
    auto prepared = prepare_typed_pod_source(generation);
    if (!prepared) {
      return prepared;
    }
    auto emitted = emit_typed_pod(value);
    if (!emitted) {
      return emitted;
    }
    return retire_typed_pod_source(generation);
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> flush_process_local_cut() noexcept {
    auto flushed = session_->flush_process_local_activation_cut();
    if (!flushed) {
      return holoscan::make_unexpected(holoscan::Error{flushed.error().code, "flush_after_source"});
    }
    return {};
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> publish_typed_pod(
      std::uint64_t generation, PayloadT value) noexcept {
    auto published = publish_typed_pod_before_cut(generation, value);
    if (!published) {
      return published;
    }
    // Match the controller's post-source cut so the sample reaches typed Runtime storage before
    // downstream readiness is frozen, regardless of whether the asynchronous pump won the race.
    return flush_process_local_cut();
  }

  [[nodiscard]] holoscan::expected<PayloadT, holoscan::Error> consume_typed_pod(
      holoscan::detail::PortBinding& binding, std::uint64_t generation) noexcept {
    return consume_typed_pod_impl(binding, generation, 0U, true);
  }

  [[nodiscard]] holoscan::expected<PayloadT, holoscan::Error> consume_typed_pod(
      holoscan::detail::PortBinding& binding, std::uint64_t generation,
      std::size_t sink_index) noexcept {
    if (sink_index >= sinks_.size()) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "sink_index"});
    }
    return consume_typed_pod_impl(binding, generation, sink_index, true);
  }

  [[nodiscard]] holoscan::expected<PayloadT, holoscan::Error> consume_typed_pod_after_poll(
      holoscan::detail::PortBinding& binding, std::uint64_t generation) noexcept {
    return consume_typed_pod_impl(binding, generation, 0U, false);
  }

  [[nodiscard]] holoscan::expected<PayloadT, holoscan::Error> consume_typed_pod_after_poll(
      holoscan::detail::PortBinding& binding, std::uint64_t generation,
      std::size_t sink_index) noexcept {
    if (sink_index >= sinks_.size()) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "sink_index"});
    }
    return consume_typed_pod_impl(binding, generation, sink_index, false);
  }

 private:
  [[nodiscard]] holoscan::expected<PayloadT, holoscan::Error> consume_typed_pod_impl(
      holoscan::detail::PortBinding& binding, std::uint64_t generation, std::size_t sink_index,
      bool poll_before_drain) noexcept {
    const auto failure = [](const holoscan::Error& error,
                            const char* stage) -> holoscan::expected<PayloadT, holoscan::Error> {
      return holoscan::make_unexpected(holoscan::Error{error.code, stage});
    };
    holoscan::detail::RuntimeEvent event;
    bool event_ready = false;
    for (std::uint32_t attempt = 0U; attempt < 4096U; ++attempt) {
      if (poll_before_drain || attempt != 0U) {
        ++readiness_poll_count_;
        auto polled = binding.poll(
            holoscan::TimePoint{static_cast<std::int64_t>(generation), holoscan::ClockId{1U}});
        if (!polled) {
          return failure(std::move(polled).error(), "poll_readiness");
        }
      }
      auto drained = binding.drain(std::span<holoscan::detail::RuntimeEvent>{&event, 1U});
      if (!drained) {
        return failure(std::move(drained).error(), "drain_readiness");
      }
      if (*drained == 1U) {
        event_ready = true;
        break;
      }
      if (auto status = session_->status(); !status) {
        return failure(std::move(status).error(), "session_status");
      }
      std::this_thread::yield();
    }
    if (!event_ready) {
      return failure(holoscan::Error{holoscan::ErrorCode::kNotReady}, "readiness_not_observed");
    }
    auto record = session_->temporal_sample(event.port, event.source_sequence);
    if (!record) {
      return failure(std::move(record).error(), "resolve_temporal_sample");
    }
    // Match the production arrival-pin fast path: the readiness witness retains storage through
    // callback preparation, so the selector layer can release its otherwise-unused cache claim.
    auto samples_retired = session_->retire_samples(
        event.port, std::span<const holoscan::detail::TemporalSampleRecord>{&*record, 1U});
    if (!samples_retired) {
      return failure(std::move(samples_retired).error(), "retire_temporal_claim");
    }
    const std::array selection{holoscan::detail::RuntimeChannelActivationSelection{
        .port = event.port,
        .selection = {holoscan::InputState::kSelected, holoscan::SelectionReason::kNone},
        .samples = std::span<const holoscan::detail::TemporalSampleRecord>{&*record, 1U},
        .destructive = true,
    }};
    const holoscan::VertexId owner = sink_owner(sink_index);
    auto prepared = session_->prepare_activation(owner, generation, selection);
    if (!prepared) {
      return failure(std::move(prepared).error(), "prepare_sink");
    }
    auto received = sinks_[sink_index]->input.receive_data();
    if (!received) {
      return failure(std::move(received).error(), "receive_data");
    }
    auto witness_retired = binding.retire(event.source_sequence);
    if (!witness_retired) {
      return failure(std::move(witness_retired).error(), "retire_witness");
    }
    auto retired = session_->retire_activation(owner, generation, true);
    if (!retired) {
      return failure(std::move(retired).error(), "retire_sink");
    }
    // Production cuts after every successful callback, including a sink with no output.
    auto flushed = session_->flush_process_local_activation_cut();
    if (!flushed) {
      return failure(std::move(flushed).error(), "flush_after_sink");
    }
    return *received;
  }

 public:
  [[nodiscard]] holoscan::expected<PayloadT, holoscan::Error> run_typed_pod_cycle(
      holoscan::detail::PortBinding& binding, std::uint64_t generation, PayloadT value) noexcept {
    auto published = publish_typed_pod(generation, value);
    if (!published) {
      return holoscan::make_unexpected(std::move(published).error());
    }
    return consume_typed_pod(binding, generation);
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> inject_burst(
      std::uint64_t first_sequence) noexcept {
    for (std::uint32_t offset = 0U; offset < queue_depth_; ++offset) {
      holoscan::SampleMetadata metadata{};
      metadata.sequence = first_sequence + offset;
      metadata.publish_clock_id = holoscan::ClockId{1U};
      metadata.byte_size = sizeof(PayloadT);
      metadata.payload_kind = holoscan::PayloadKind::kInlineTyped;
      for (const holoscan::PortId port : input_ports_) {
        auto injected = holoscan::detail::RuntimeChannelSessionTestAccess::inject_received_metadata(
            *session_, port, metadata);
        if (!injected) {
          return holoscan::make_unexpected(std::move(injected).error());
        }
      }
    }
    return {};
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> inject_one(
      holoscan::PortId port, std::uint64_t sequence) noexcept {
    holoscan::SampleMetadata metadata{};
    metadata.sequence = sequence;
    metadata.publish_clock_id = holoscan::ClockId{1U};
    metadata.byte_size = sizeof(PayloadT);
    metadata.payload_kind = holoscan::PayloadKind::kInlineTyped;
    return holoscan::detail::RuntimeChannelSessionTestAccess::inject_received_metadata(
        *session_, port, metadata);
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> retire_burst(
      holoscan::detail::PortBinding& binding,
      std::span<holoscan::detail::RuntimeEvent> events) noexcept {
    auto drained = binding.drain(events);
    if (!drained) {
      return holoscan::make_unexpected(std::move(drained).error());
    }
    if (*drained != events.size()) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kProtocolError});
    }
    for (const holoscan::detail::RuntimeEvent& event : events) {
      auto record = session_->temporal_sample(event.port, event.source_sequence);
      if (!record) {
        return holoscan::make_unexpected(std::move(record).error());
      }
      auto retired = binding.retire(event.source_sequence);
      if (!retired) {
        return holoscan::make_unexpected(std::move(retired).error());
      }
      retired = session_->retire_samples(
          event.port, std::span<const holoscan::detail::TemporalSampleRecord>{&*record, 1U});
      if (!retired) {
        return holoscan::make_unexpected(std::move(retired).error());
      }
    }
    return {};
  }

 private:
  ChannelBenchmarkFixtureT(std::uint32_t input_count, std::uint32_t queue_depth,
                           bool shared_sink_owner,
                           holoscan::detail::ProcessLocalChannelDeliveryKind delivery_kind)
      : queue_depth_(queue_depth) {
    source_.setup(source_spec_);
    const auto output = holoscan::detail::OperatorSpecCompilerAccess::port(source_spec_, 0U);

    plan_.admitted = true;
    plan_.plan_generation =
        1000U + static_cast<std::uint64_t>(input_count) * 100U + queue_depth +
        (shared_sink_owner ? 1U : 0U) +
        (delivery_kind == holoscan::detail::ProcessLocalChannelDeliveryKind::kDirectRetained
             ? 10'000U
             : 0U);
    const std::string service =
        "process-local-channel." + std::to_string(plan_.plan_generation) + ".source.1.policy." +
        std::to_string(static_cast<std::uint32_t>(holoscan::QueuePolicy::kBlocking));

    plan_.ports.reserve(static_cast<std::size_t>(input_count) + 1U);
    plan_.payload_edges.reserve(input_count);
    plan_.channel_locality.reserve(input_count);
    plan_.process_local_channel_subscribers.reserve(input_count);
    replay_.reserve(static_cast<std::size_t>(input_count) + 1U);
    sinks_.reserve(input_count);
    sink_specs_.reserve(input_count);
    input_ports_.reserve(input_count);

    plan_.ports.push_back(holoscan::PortPlanRecord{
        .id = holoscan::PortId{1U},
        .owner = holoscan::VertexId{1U},
        .direction = holoscan::PortDirection::kOutput,
        .local_name = "output",
        .payload_identity = output.payload_identity,
        .max_emits_per_compute = queue_depth,
        .effective_profile = holoscan::Profile::kStandard,
    });
    replay_.push_back(holoscan::detail::RuntimeChannelPortReplayRecord{
        .port = holoscan::PortId{1U},
        .owner = holoscan::VertexId{1U},
        .direction = holoscan::PortDirection::kOutput,
        .declaration = const_cast<void*>(output.declaration),
        .type_ops = output.runtime_type_ops,
    });

    for (std::uint32_t index = 0U; index < input_count; ++index) {
      auto sink = std::make_unique<ChannelSinkOperator<PayloadT>>();
      auto spec = std::make_unique<holoscan::OperatorSpec>();
      sink->setup(*spec);
      const auto input = holoscan::detail::OperatorSpecCompilerAccess::port(*spec, 0U);
      const holoscan::PortId port{static_cast<std::uint64_t>(index) + 2U};
      const holoscan::VertexId owner{shared_sink_owner ? 2U
                                                       : static_cast<std::uint64_t>(index) + 2U};
      const holoscan::EdgeId edge{static_cast<std::uint64_t>(index) + 1U};
      input_ports_.push_back(port);
      last_sink_owner_ = owner;
      plan_.ports.push_back(holoscan::PortPlanRecord{
          .id = port,
          .owner = owner,
          .direction = holoscan::PortDirection::kInput,
          .local_name = "input-" + std::to_string(index),
          .payload_identity = input.payload_identity,
          .queue_depth = queue_depth,
          .effective_profile = holoscan::Profile::kStandard,
      });
      plan_.payload_edges.push_back(holoscan::PayloadEdgePlanRecord{
          .edge = edge,
          .source = holoscan::PortId{1U},
          .target = port,
          .payload_identity = output.payload_identity,
          .queue_policy = holoscan::QueuePolicy::kBlocking,
          .queue_depth = queue_depth,
          .max_inflight_samples = queue_depth,
          .producer_max_emits_per_compute = queue_depth,
          .fanout = input_count,
          .cache_capacity = queue_depth,
          .locality = holoscan::ChannelLocalityKind::kProcessLocal,
          .disposition = holoscan::PlanDisposition::kAccepted,
      });
      plan_.channel_locality.push_back(holoscan::ChannelLocalityRecord{
          .edge = edge,
          .requested = holoscan::ChannelLocalityKind::kProcessLocal,
          .accepted = holoscan::ChannelLocalityKind::kProcessLocal,
          .requires_runtime_attach = false,
      });
      plan_.process_local_channel_subscribers.push_back(
          holoscan::detail::ProcessLocalChannelSubscriberPlanRecord{
              .edge = edge,
              .source = holoscan::PortId{1U},
              .target = port,
              .service_id = service,
              .requested_queue_depth = queue_depth,
              .max_borrowed_samples = queue_depth,
              .cache_capacity = queue_depth,
              .readiness_capacity = queue_depth,
              .event_wake =
                  delivery_kind == holoscan::detail::ProcessLocalChannelDeliveryKind::kHoloMQ,
          });
      replay_.push_back(holoscan::detail::RuntimeChannelPortReplayRecord{
          .port = port,
          .owner = owner,
          .direction = holoscan::PortDirection::kInput,
          .declaration = const_cast<void*>(input.declaration),
          .type_ops = input.runtime_type_ops,
      });
      sinks_.push_back(std::move(sink));
      sink_specs_.push_back(std::move(spec));
    }

    plan_.process_local_channel_services = {holoscan::detail::ProcessLocalChannelServicePlanRecord{
        .source = holoscan::PortId{1U},
        .service_id = service,
        .payload_identity = output.payload_identity,
        .payload_size_bytes = sizeof(PayloadT),
        .delivery_kind = delivery_kind,
        .queue_policy = holoscan::QueuePolicy::kBlocking,
        .max_subscribers = input_count,
        .max_subscriber_queue_depth = queue_depth,
        .max_subscriber_borrowed_samples = queue_depth,
        .publisher_history_depth = 0U,
        .event_wake = delivery_kind == holoscan::detail::ProcessLocalChannelDeliveryKind::kHoloMQ,
    }};
    plan_.process_local_channel_publishers = {
        holoscan::detail::ProcessLocalChannelPublisherPlanRecord{
            .source = holoscan::PortId{1U},
            .service_id = service,
            .max_loaned_samples = queue_depth,
        }};
  }

  // Build one readable clock generation and bind it as this session's publication domain. The
  // registry is retained because ClockRef borrows from the generation that produced it.
  [[nodiscard]] holoscan::expected<void, holoscan::Error> bind_publish_clock() {
    const holoscan::ClockKey key = holoscan::ClockKey::named("benchmark-publish-clock");
    holoscan::detail::ClockRegistryBuilder builder;
    if (auto added = builder.add_clock(key,
                                       std::make_shared<holoscan::RealtimeClock>(),
                                       holoscan::detail::ClockCapabilityOptions{
                                           .can_read_now = true,
                                           .can_wait_until = true,
                                           .steady = true,
                                       });
        !added) {
      return holoscan::make_unexpected(std::move(added).error());
    }
    if (auto selected = builder.set_default(key); !selected) {
      return holoscan::make_unexpected(std::move(selected).error());
    }
    auto registry = builder.build();
    if (!registry) {
      return holoscan::make_unexpected(std::move(registry).error());
    }
    clock_registry_ = std::move(*registry);
    auto resolved = clock_registry_.resolve(holoscan::ClockKey::default_clock());
    if (!resolved) {
      return holoscan::make_unexpected(std::move(resolved).error());
    }
    return session_->bind_publish_clock(std::move(*resolved));
  }

  std::uint32_t queue_depth_{};
  ChannelSourceOperator<PayloadT> source_;
  holoscan::OperatorSpec source_spec_;
  std::vector<std::unique_ptr<ChannelSinkOperator<PayloadT>>> sinks_;
  std::vector<std::unique_ptr<holoscan::OperatorSpec>> sink_specs_;
  holoscan::detail::ExecutionPlanState plan_;
  std::vector<holoscan::detail::RuntimeChannelPortReplayRecord> replay_;
  std::vector<holoscan::PortId> input_ports_;
  holoscan::VertexId last_sink_owner_{};
  std::uint64_t readiness_poll_count_{};
  // Declared before the session so the borrowed publish ClockRef outlives every publication.
  holoscan::detail::ClockRegistry clock_registry_;
  std::unique_ptr<holoscan::detail::RuntimeChannelSession> session_;
};

using ChannelBenchmarkFixture = ChannelBenchmarkFixtureT<std::int32_t>;

void skip_with_error(benchmark::State& state, const char* operation, const holoscan::Error& error) {
  std::string message =
      std::string{operation} + ": " + std::string{holoscan::error_code_name(error.code)};
  if (!error.message.empty()) {
    message += " at ";
    message += error.message;
  }
  state.SkipWithError(message.c_str());
}

template <typename PayloadT>
struct TypedPODCycleContextT {
  std::unique_ptr<ChannelBenchmarkFixtureT<PayloadT>> fixture;
  std::unique_ptr<holoscan::detail::PortBinding> binding;
};

using TypedPODCycleContext = TypedPODCycleContextT<std::int32_t>;

struct TypedPODFanoutContext {
  std::unique_ptr<ChannelBenchmarkFixture> fixture;
  std::vector<std::unique_ptr<holoscan::detail::PortBinding>> bindings;
};

struct TypedPODFanoutConsumeInvocation {
  ChannelBenchmarkFixture* fixture{};
  holoscan::detail::PortBinding* binding{};
  std::uint64_t generation{};
  std::size_t sink_index{};
  std::int32_t received{};
};

struct ReadinessPollInvocation {
  holoscan::detail::PortBinding* binding{};
  holoscan::TimePoint observed_at{};
};

struct DirectRetainedConsumeInvocation {
  ChannelBenchmarkFixture* fixture{};
  holoscan::detail::PortBinding* binding{};
  std::uint64_t generation{};
  std::int32_t received{};
  bool readiness_already_polled{};
};

enum class DirectRetainedActivationCeilingMode : std::uint8_t {
  kOrdinary,
  kReceiptElided,
  kComposite,
};

struct DirectRetainedActivationCeilingInvocation {
  ChannelBenchmarkFixture* fixture{};
  holoscan::detail::PortBinding* binding{};
  DirectRetainedActivationCeilingMode mode{DirectRetainedActivationCeilingMode::kOrdinary};
  std::chrono::nanoseconds callback_work{};
  std::uint64_t generation{};
  std::int32_t value{};
  std::int32_t received{};
  std::int64_t source_entry_ns{};
  std::int64_t source_publication_ns{};
  std::int64_t source_exit_ns{};
  std::int64_t successor_entry_ns{};
  std::int64_t successor_exit_ns{};
  std::uint32_t callback_stage{};
  std::uint32_t callback_count{};
  std::uint32_t active_callbacks{};
  std::uint32_t max_active_callbacks{};
  bool callback_overlap{};
  bool callback_order_valid{true};
  bool receipt_validated{};
  std::optional<holoscan::detail::RuntimeChannelDirectContinuationReceipt> receipt;
};

inline void benchmark_busy_wait_pause() noexcept;

class CallbackConcurrencyScope final {
 public:
  explicit CallbackConcurrencyScope(DirectRetainedActivationCeilingInvocation& invocation) noexcept
      : invocation_(invocation) {
    ++invocation_.callback_count;
    ++invocation_.active_callbacks;
    invocation_.max_active_callbacks =
        std::max(invocation_.max_active_callbacks, invocation_.active_callbacks);
    invocation_.callback_overlap =
        invocation_.callback_overlap || invocation_.active_callbacks > 1U;
  }

  CallbackConcurrencyScope(const CallbackConcurrencyScope&) = delete;
  CallbackConcurrencyScope& operator=(const CallbackConcurrencyScope&) = delete;
  CallbackConcurrencyScope(CallbackConcurrencyScope&&) = delete;
  CallbackConcurrencyScope& operator=(CallbackConcurrencyScope&&) = delete;

  ~CallbackConcurrencyScope() { --invocation_.active_callbacks; }

 private:
  DirectRetainedActivationCeilingInvocation& invocation_;
};

[[nodiscard]] std::int64_t steady_timestamp_ns() noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void run_controlled_callback_work(std::chrono::nanoseconds duration) noexcept {
  if (duration <= std::chrono::nanoseconds::zero()) {
    return;
  }
  const auto deadline = std::chrono::steady_clock::now() + duration;
  std::uint64_t iterations = 0U;
  while (std::chrono::steady_clock::now() < deadline) {
    ++iterations;
    benchmark_busy_wait_pause();
  }
  benchmark::DoNotOptimize(iterations);
}

[[nodiscard]] holoscan::expected<void, holoscan::Error> invoke_direct_retained_ceiling_successor(
    void* opaque) noexcept {
  auto& invocation = *static_cast<DirectRetainedActivationCeilingInvocation*>(opaque);
  const CallbackConcurrencyScope callback{invocation};
  invocation.callback_order_valid =
      invocation.callback_order_valid && invocation.callback_stage == 1U;
  invocation.callback_stage = 2U;
  invocation.successor_entry_ns = steady_timestamp_ns();
  auto received =
      invocation.fixture->consume_typed_pod_after_poll(*invocation.binding, invocation.generation);
  if (!received) {
    return holoscan::make_unexpected(std::move(received).error());
  }
  invocation.received = *received;
  run_controlled_callback_work(invocation.callback_work);
  invocation.successor_exit_ns = steady_timestamp_ns();
  invocation.callback_stage = 3U;
  return {};
}

[[nodiscard]] holoscan::expected<void, holoscan::Error> invoke_direct_retained_ceiling_source(
    void* opaque) noexcept {
  auto& invocation = *static_cast<DirectRetainedActivationCeilingInvocation*>(opaque);
  {
    const CallbackConcurrencyScope callback{invocation};
    invocation.callback_order_valid =
        invocation.callback_order_valid && invocation.callback_stage == 0U;
    invocation.source_entry_ns = steady_timestamp_ns();
    run_controlled_callback_work(invocation.callback_work);
    auto published = invocation.fixture->emit_typed_pod(invocation.value);
    invocation.source_publication_ns = steady_timestamp_ns();
    if (!published) {
      return holoscan::make_unexpected(std::move(published).error());
    }
    invocation.source_exit_ns = steady_timestamp_ns();
    invocation.callback_stage = 1U;
  }
  if (invocation.mode != DirectRetainedActivationCeilingMode::kOrdinary) {
    auto receipt = invocation.fixture->session().poll_direct_continuation(
        holoscan::VertexId{1U},
        0U,
        holoscan::TimePoint{static_cast<std::int64_t>(invocation.generation),
                            holoscan::ClockId{1U}});
    if (!receipt) {
      return holoscan::make_unexpected(std::move(receipt).error());
    }
    if (!receipt->has_value()) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kProtocolError, "continuation_receipt"});
    }
    invocation.receipt = std::move(**receipt);
  }
  if (invocation.mode == DirectRetainedActivationCeilingMode::kComposite) {
    auto retired = invocation.fixture->retire_typed_pod_source(invocation.generation);
    if (!retired) {
      return retired;
    }
    auto flushed = invocation.fixture->flush_process_local_cut();
    if (!flushed) {
      return flushed;
    }
    invocation.receipt_validated =
        invocation.receipt.has_value() &&
        invocation.fixture->session().validate_direct_continuation_receipt(*invocation.receipt);
    if (!invocation.receipt_validated) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kProtocolError,
                                                       "continuation_receipt_after_retirement"});
    }
    return invoke_direct_retained_ceiling_successor(opaque);
  }
  return {};
}

[[nodiscard]] holoscan::expected<void, holoscan::Error> invoke_readiness_poll(
    void* opaque) noexcept {
  auto& invocation = *static_cast<ReadinessPollInvocation*>(opaque);
  return invocation.binding->poll(invocation.observed_at);
}

[[nodiscard]] holoscan::expected<void, holoscan::Error> invoke_direct_retained_consume(
    void* opaque) noexcept {
  auto& invocation = *static_cast<DirectRetainedConsumeInvocation*>(opaque);
  auto received =
      invocation.readiness_already_polled
          ? invocation.fixture->consume_typed_pod_after_poll(*invocation.binding,
                                                             invocation.generation)
          : invocation.fixture->consume_typed_pod(*invocation.binding, invocation.generation);
  if (!received) {
    return holoscan::make_unexpected(std::move(received).error());
  }
  invocation.received = *received;
  return {};
}

[[nodiscard]] holoscan::expected<void, holoscan::Error> invoke_fanout_consume_after_poll(
    void* opaque) noexcept {
  auto& invocation = *static_cast<TypedPODFanoutConsumeInvocation*>(opaque);
  auto received = invocation.fixture->consume_typed_pod_after_poll(
      *invocation.binding, invocation.generation, invocation.sink_index);
  if (!received) {
    return holoscan::make_unexpected(std::move(received).error());
  }
  invocation.received = *received;
  return {};
}

[[nodiscard]] holoscan::expected<void, holoscan::Error> invoke_fanout_poll_consume(
    void* opaque) noexcept {
  auto& invocation = *static_cast<TypedPODFanoutConsumeInvocation*>(opaque);
  auto received = invocation.fixture->consume_typed_pod(
      *invocation.binding, invocation.generation, invocation.sink_index);
  if (!received) {
    return holoscan::make_unexpected(std::move(received).error());
  }
  invocation.received = *received;
  return {};
}

inline void benchmark_busy_wait_pause() noexcept {
#if defined(__i386__) || defined(__x86_64__)
  __builtin_ia32_pause();
#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
  _mm_pause();
#elif defined(__arm__) || defined(__aarch64__)
  __asm__ __volatile__("yield");
#else
  std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

// Benchmark-local split-phase worker. This estimates the handoff mechanism without proposing a
// Runtime API or bypassing controller authorization in production.
class DirectRetainedContinuationProbe {
 public:
  DirectRetainedContinuationProbe(ChannelBenchmarkFixture& fixture,
                                  holoscan::detail::PortBinding& binding)
      : fixture_(fixture), binding_(binding), worker_([this] { run(); }) {}

  DirectRetainedContinuationProbe(const DirectRetainedContinuationProbe&) = delete;
  DirectRetainedContinuationProbe& operator=(const DirectRetainedContinuationProbe&) = delete;
  DirectRetainedContinuationProbe(DirectRetainedContinuationProbe&&) = delete;
  DirectRetainedContinuationProbe& operator=(DirectRetainedContinuationProbe&&) = delete;

  ~DirectRetainedContinuationProbe() {
    state_.store(State::kStop, std::memory_order_release);
    worker_.join();
  }

  [[nodiscard]] holoscan::expected<std::int32_t, holoscan::Error> poll_authorize_consume(
      std::uint64_t generation) noexcept {
    if (state_.load(std::memory_order_acquire) != State::kIdle) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kBackpressured, "continuation_busy"});
    }
    generation_ = generation;
    received_ = 0;
    error_.reset();
    state_.store(State::kPollPending, std::memory_order_release);
    wait_for(State::kPollComplete);
    if (error_) {
      holoscan::Error error = std::move(*error_);
      error_.reset();
      state_.store(State::kIdle, std::memory_order_release);
      return holoscan::make_unexpected(std::move(error));
    }

    // This store models the controller's authorization after it observes the accepted poll.
    state_.store(State::kContinue, std::memory_order_release);
    wait_for(State::kComplete);
    if (error_) {
      holoscan::Error error = std::move(*error_);
      error_.reset();
      state_.store(State::kIdle, std::memory_order_release);
      return holoscan::make_unexpected(std::move(error));
    }
    const std::int32_t received = received_;
    state_.store(State::kIdle, std::memory_order_release);
    return received;
  }

 private:
  enum class State : std::uint8_t {
    kIdle,
    kPollPending,
    kPollComplete,
    kContinue,
    kComplete,
    kStop,
  };

  void wait_for(State expected) const noexcept {
    while (state_.load(std::memory_order_acquire) != expected) {
      benchmark_busy_wait_pause();
    }
  }

  void run() noexcept {
    for (;;) {
      const State state = state_.load(std::memory_order_acquire);
      if (state == State::kStop) {
        return;
      }
      if (state == State::kPollPending) {
        auto polled = binding_.poll(
            holoscan::TimePoint{static_cast<std::int64_t>(generation_), holoscan::ClockId{1U}});
        if (!polled) {
          error_ = std::move(polled).error();
        }
        state_.store(State::kPollComplete, std::memory_order_release);
        continue;
      }
      if (state == State::kContinue) {
        auto received = fixture_.consume_typed_pod_after_poll(binding_, generation_);
        if (received) {
          received_ = *received;
        } else {
          error_ = std::move(received).error();
        }
        state_.store(State::kComplete, std::memory_order_release);
        continue;
      }
      benchmark_busy_wait_pause();
    }
  }

  ChannelBenchmarkFixture& fixture_;
  holoscan::detail::PortBinding& binding_;
  std::atomic<State> state_{State::kIdle};
  std::uint64_t generation_{};
  std::int32_t received_{};
  std::optional<holoscan::Error> error_;
  std::thread worker_;
};

[[nodiscard]] holoscan::SchedulerPlanRecord single_busy_lane_scheduler_plan() {
  return {
      .ready_policy = holoscan::AlgorithmKey{"holoscan.ready.deterministic_serial", 1U},
      .execution_lane_count = 1U,
      .max_concurrent_activations = 1U,
      .activation_overlap = false,
      .worker_count = 1U,
      .intra_path_parallelism = 1U,
  };
}

[[nodiscard]] holoscan::expected<std::unique_ptr<holoscan::detail::ExecutionLaneSet>,
                                 holoscan::Error>
create_started_lane_set(holoscan::WakeMode wake_mode,
                        holoscan::detail::RuntimePhaseCounters* phase_counters = nullptr) {
  const std::array lanes{holoscan::ExecutionLanePlanRecord{
      .execution_lane = 0U,
      .partition = holoscan::ExecutionPartitionId{1U},
      .accepted_placement = holoscan::ThreadPlacement{.wake_mode = wake_mode},
  }};
  auto lane_set = holoscan::detail::ExecutionLaneSet::create(
      single_busy_lane_scheduler_plan(), lanes, {}, phase_counters);
  if (!lane_set) {
    return holoscan::make_unexpected(std::move(lane_set).error());
  }
  auto started = (*lane_set)->start();
  if (!started) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure, "start_lane"});
  }
  return lane_set;
}

[[nodiscard]] holoscan::expected<std::unique_ptr<holoscan::detail::ExecutionLaneSet>,
                                 holoscan::Error>
create_started_busy_lane_set() {
  return create_started_lane_set(holoscan::WakeMode::kBusyPoll);
}

struct NoopOutputBindingState {
  static holoscan::expected<void, holoscan::Error> emit_sample(void* opaque,
                                                               const void* sample) noexcept {
    if (opaque == nullptr || sample == nullptr) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kInvalidArgument});
    }
    auto& state = *static_cast<NoopOutputBindingState*>(opaque);
    const auto& typed = *static_cast<const holoscan::Sample<std::int32_t>*>(sample);
    ++state.accepted;
    state.checksum += static_cast<std::uint64_t>(typed.data);
    return {};
  }

  inline static const holoscan::detail::RuntimeOutputBindingOps kOps{
      .emit_sample = &emit_sample,
      .allocate = nullptr,
      .emit_loan = nullptr,
  };

  std::uint64_t accepted{};
  std::uint64_t checksum{};
};

struct InlinePublishContext {
  holoscan::ChannelRuntime runtime;
  std::unique_ptr<holoscan::channel_internal::InlinePayloadSubscriber> subscriber;
  std::unique_ptr<holoscan::channel_internal::InlinePayloadPublisher> publisher;
  std::unique_ptr<holoscan::channel_internal::InlinePayloadWaitSet> wait_set;
  std::uint64_t wait_handle{};
};

[[nodiscard]] bool wait_for_atomic_sequence(
    const std::atomic<std::uint64_t>& observed, std::uint64_t expected,
    const std::atomic<holoscan::ErrorCode>& error_code) noexcept {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (observed.load(std::memory_order_acquire) < expected) {
    if (error_code.load(std::memory_order_acquire) != holoscan::ErrorCode::kSuccess ||
        std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::yield();
  }
  return true;
}

class AsyncInlineConsumer {
 public:
  AsyncInlineConsumer(InlinePublishContext& context, bool defer_receive)
      : context_(context),
        defer_receive_(defer_receive),
        worker_([this](const std::stop_token& stop) { run(stop); }) {}

  AsyncInlineConsumer(const AsyncInlineConsumer&) = delete;
  AsyncInlineConsumer& operator=(const AsyncInlineConsumer&) = delete;
  AsyncInlineConsumer(AsyncInlineConsumer&&) = delete;
  AsyncInlineConsumer& operator=(AsyncInlineConsumer&&) = delete;
  ~AsyncInlineConsumer() = default;

  [[nodiscard]] bool wait_until_waiting(std::uint64_t sequence) const noexcept {
    return wait_for_atomic_sequence(waiting_sequence_, sequence, error_code_);
  }

  [[nodiscard]] bool wait_until_consumed(std::uint64_t sequence) const noexcept {
    return wait_for_atomic_sequence(consumed_sequence_, sequence, error_code_);
  }

  [[nodiscard]] bool wait_until_woken(std::uint64_t sequence) const noexcept {
    return wait_for_atomic_sequence(woken_sequence_, sequence, error_code_);
  }

  void release_receive(std::uint64_t sequence) noexcept {
    receive_release_sequence_.store(sequence, std::memory_order_release);
  }

  [[nodiscard]] holoscan::ErrorCode error_code() const noexcept {
    return error_code_.load(std::memory_order_acquire);
  }

 private:
  void fail(holoscan::ErrorCode code) noexcept {
    error_code_.store(code, std::memory_order_release);
  }

  void run(const std::stop_token& stop) noexcept {
    std::array<std::uint64_t, 1U> ready{};
    std::int32_t received_value{};
    auto destination = std::as_writable_bytes(std::span<std::int32_t, 1U>{&received_value, 1U});
    std::uint64_t expected_sequence = 1U;

    while (!stop.stop_requested()) {
      // Publish the handshake before entering the bounded kernel wait. The producer yields after
      // observing it so this row models Runtime's normally parked asynchronous input pump.
      waiting_sequence_.store(expected_sequence, std::memory_order_release);
      auto waited = context_.wait_set->wait_for(std::chrono::milliseconds{1});
      if (!waited) {
        fail(waited.error().code);
        return;
      }
      if (!*waited) {
        continue;
      }
      auto count = context_.wait_set->ready_handles(ready);
      if (!count) {
        fail(count.error().code);
        return;
      }
      if (*count == 0U) {
        continue;
      }
      if (*count != 1U || ready[0] != context_.wait_handle) {
        fail(holoscan::ErrorCode::kProtocolError);
        return;
      }
      woken_sequence_.store(expected_sequence, std::memory_order_release);
      while (defer_receive_ &&
             receive_release_sequence_.load(std::memory_order_acquire) < expected_sequence) {
        if (stop.stop_requested()) {
          return;
        }
        std::this_thread::yield();
      }
      auto received = context_.subscriber->receive(destination);
      if (!received) {
        fail(received.error().code);
        return;
      }
      if (received->metadata.sequence != expected_sequence ||
          received->payload_size != destination.size() || received_value != 7) {
        fail(holoscan::ErrorCode::kProtocolError);
        return;
      }
      consumed_sequence_.store(expected_sequence, std::memory_order_release);
      ++expected_sequence;
    }
  }

  InlinePublishContext& context_;
  bool defer_receive_{};
  std::atomic<std::uint64_t> waiting_sequence_{};
  std::atomic<std::uint64_t> woken_sequence_{};
  std::atomic<std::uint64_t> receive_release_sequence_{};
  std::atomic<std::uint64_t> consumed_sequence_{};
  std::atomic<holoscan::ErrorCode> error_code_{holoscan::ErrorCode::kSuccess};
  std::jthread worker_;
};

class PollingInlineConsumer {
 public:
  PollingInlineConsumer(InlinePublishContext& context, std::uint64_t first_sequence)
      : context_(context),
        first_sequence_(first_sequence),
        worker_([this](const std::stop_token& stop) { run(stop); }) {}

  PollingInlineConsumer(const PollingInlineConsumer&) = delete;
  PollingInlineConsumer& operator=(const PollingInlineConsumer&) = delete;
  PollingInlineConsumer(PollingInlineConsumer&&) = delete;
  PollingInlineConsumer& operator=(PollingInlineConsumer&&) = delete;
  ~PollingInlineConsumer() = default;

  [[nodiscard]] bool wait_until_polling(std::uint64_t sequence) const noexcept {
    return wait_for_atomic_sequence(polling_sequence_, sequence, error_code_);
  }

  [[nodiscard]] bool wait_until_consumed(std::uint64_t sequence) const noexcept {
    return wait_for_atomic_sequence(consumed_sequence_, sequence, error_code_);
  }

  [[nodiscard]] holoscan::ErrorCode error_code() const noexcept {
    return error_code_.load(std::memory_order_acquire);
  }

 private:
  void run(const std::stop_token& stop) noexcept {
    std::int32_t received_value{};
    auto destination = std::as_writable_bytes(std::span<std::int32_t, 1U>{&received_value, 1U});
    std::uint64_t expected_sequence = first_sequence_;
    while (!stop.stop_requested()) {
      polling_sequence_.store(expected_sequence, std::memory_order_release);
      auto received = context_.subscriber->receive(destination);
      if (!received) {
        if (received.error().code == holoscan::ErrorCode::kNotReady) {
          continue;
        }
        error_code_.store(received.error().code, std::memory_order_release);
        return;
      }
      if (received->metadata.sequence != expected_sequence ||
          received->payload_size != destination.size() || received_value != 7) {
        error_code_.store(holoscan::ErrorCode::kProtocolError, std::memory_order_release);
        return;
      }
      consumed_sequence_.store(expected_sequence, std::memory_order_release);
      ++expected_sequence;
    }
  }

  InlinePublishContext& context_;
  std::uint64_t first_sequence_{};
  std::atomic<std::uint64_t> polling_sequence_{};
  std::atomic<std::uint64_t> consumed_sequence_{};
  std::atomic<holoscan::ErrorCode> error_code_{holoscan::ErrorCode::kSuccess};
  std::jthread worker_;
};

[[nodiscard]] holoscan::expected<InlinePublishContext, holoscan::Error>
create_inline_publish_context(bool enable_event_wake = true) {
  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime) {
    return holoscan::make_unexpected(std::move(runtime).error());
  }
  holoscan::ChannelOptions options{};
  options.enable_event_wake = enable_event_wake;
  options.max_subscriber_queue_depth = 2U;
  options.subscriber_queue_depth = 2U;
  options.max_subscriber_borrowed_samples = 2U;
  options.max_borrowed_samples = 2U;
  options.max_loaned_samples = 2U;
  options.expected_recipient_count = 1U;
  constexpr std::string_view kService = "runtime-channel-inline-publish-benchmark";
  auto publisher = holoscan::detail::RuntimeChannelBindingAccess::create_publisher(
      *runtime, kService, sizeof(std::int32_t), options);
  if (!publisher) {
    return holoscan::make_unexpected(std::move(publisher).error());
  }
  auto subscriber = holoscan::detail::RuntimeChannelBindingAccess::create_subscriber(
      *runtime, kService, sizeof(std::int32_t), options);
  if (!subscriber) {
    return holoscan::make_unexpected(std::move(subscriber).error());
  }
  std::unique_ptr<holoscan::channel_internal::InlinePayloadWaitSet> wait_set;
  std::uint64_t wait_handle{};
  if (enable_event_wake) {
    auto created = holoscan::detail::RuntimeChannelBindingAccess::create_wait_set(*runtime);
    if (!created) {
      return holoscan::make_unexpected(std::move(created).error());
    }
    auto attached = (*created)->attach(**subscriber);
    if (!attached) {
      return holoscan::make_unexpected(std::move(attached).error());
    }
    wait_handle = *attached;
    wait_set = std::move(*created);
  }
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (!(*publisher)->has_subscribers()) {
    auto refreshed = (*publisher)->update_connections();
    if (!refreshed) {
      return holoscan::make_unexpected(std::move(refreshed).error());
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kNotReady});
    }
    std::this_thread::yield();
  }
  return InlinePublishContext{
      .runtime = std::move(*runtime),
      .subscriber = std::move(*subscriber),
      .publisher = std::move(*publisher),
      .wait_set = std::move(wait_set),
      .wait_handle = wait_handle,
  };
}

[[nodiscard]] holoscan::expected<void, holoscan::Error> drain_inline_publish(
    InlinePublishContext& context, std::span<std::byte> destination) {
  auto waited = context.wait_set->wait_for(std::chrono::seconds{1});
  if (!waited) {
    return holoscan::make_unexpected(std::move(waited).error());
  }
  if (!*waited) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kNotReady});
  }
  std::array<std::uint64_t, 1U> ready{};
  auto count = context.wait_set->ready_handles(ready);
  if (!count) {
    return holoscan::make_unexpected(std::move(count).error());
  }
  if (*count != 1U || ready[0] != context.wait_handle) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kProtocolError});
  }
  auto received = context.subscriber->receive(destination);
  if (!received) {
    return holoscan::make_unexpected(std::move(received).error());
  }
  return {};
}

[[nodiscard]] holoscan::expected<void, holoscan::Error> drain_inline_publish_polling(
    InlinePublishContext& context, std::span<std::byte> destination) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{1};
  while (true) {
    auto received = context.subscriber->receive(destination);
    if (received) {
      return {};
    }
    if (received.error().code != holoscan::ErrorCode::kNotReady) {
      return holoscan::make_unexpected(std::move(received).error());
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kNotReady});
    }
    std::this_thread::yield();
  }
}

[[nodiscard]] holoscan::expected<void, holoscan::Error> warm_inline_publish_polling(
    InlinePublishContext& context, std::span<const std::byte> payload,
    std::span<std::byte> destination, holoscan::SampleMetadata metadata) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (std::chrono::steady_clock::now() < deadline) {
    auto published = context.publisher->emit_phase(payload, metadata);
    if (published.phase == holoscan::channel_internal::InlinePublicationPhase::kAccepted &&
        published.recipient_count == 1U) {
      return drain_inline_publish_polling(context, destination);
    }
    if (published.phase != holoscan::channel_internal::InlinePublicationPhase::kPreSubmitRejected ||
        published.error.code != holoscan::ErrorCode::kNotReady) {
      return holoscan::make_unexpected(std::move(published.error));
    }
    auto refreshed = context.publisher->update_connections();
    if (!refreshed) {
      return holoscan::make_unexpected(std::move(refreshed).error());
    }
    std::this_thread::yield();
  }
  return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kNotReady});
}

template <typename PayloadT>
[[nodiscard]] bool has_equal_object_representation(const PayloadT& left,
                                                   const PayloadT& right) noexcept {
  static_assert(std::has_unique_object_representations_v<PayloadT>,
                "bytewise benchmark validation requires unique object representations");
  return std::memcmp(std::addressof(left), std::addressof(right), sizeof(PayloadT)) == 0;
}

template <typename PayloadT>
[[nodiscard]] holoscan::expected<TypedPODCycleContextT<PayloadT>, holoscan::Error>
create_typed_pod_cycle_context_for(holoscan::detail::ProcessLocalChannelDeliveryKind delivery_kind,
                                   std::uint32_t queue_depth, const PayloadT& warmup_value) {
  auto fixture = ChannelBenchmarkFixtureT<PayloadT>::create(1U, queue_depth, false, delivery_kind);
  if (!fixture) {
    return holoscan::make_unexpected(std::move(fixture).error());
  }
  auto binding = (*fixture)->session().create_readiness_binding(
      (*fixture)->sink_owner(),
      holoscan::detail::RuntimePortBindingRequest{
          .plan_generation = (*fixture)->plan_generation(),
          .run_generation = 1U,
          .kind = holoscan::detail::RuntimeEventKind::kInputArrival,
          .source = holoscan::detail::make_source_id(1U),
          .source_clock = holoscan::ClockId{1U},
          .reference_clock = holoscan::ClockId{1U},
          .capacity = queue_depth,
      });
  if (!binding) {
    return holoscan::make_unexpected(std::move(binding).error());
  }
  auto armed = (*fixture)->arm();
  if (!armed) {
    return holoscan::make_unexpected(std::move(armed).error());
  }
  // Endpoint discovery is cold setup. Match the established Runtime tests, then prove one full
  // excluded cycle before Google Benchmark starts a measured loop.
  std::this_thread::sleep_for(std::chrono::milliseconds{20});
  auto warmup = (*fixture)->run_typed_pod_cycle(**binding, 1U, warmup_value);
  if (!warmup) {
    return holoscan::make_unexpected(std::move(warmup).error());
  }
  if (!has_equal_object_representation(*warmup, warmup_value)) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kProtocolError, "warmup_value"});
  }
  (*fixture)->reset_readiness_poll_count();
  return TypedPODCycleContextT<PayloadT>{.fixture = std::move(*fixture),
                                         .binding = std::move(*binding)};
}

[[nodiscard]] holoscan::expected<TypedPODCycleContext, holoscan::Error>
create_typed_pod_cycle_context(holoscan::detail::ProcessLocalChannelDeliveryKind delivery_kind =
                                   holoscan::detail::ProcessLocalChannelDeliveryKind::kHoloMQ,
                               std::uint32_t queue_depth = 2U) {
  return create_typed_pod_cycle_context_for<std::int32_t>(delivery_kind, queue_depth, 1);
}

[[nodiscard]] holoscan::expected<TypedPODFanoutContext, holoscan::Error>
create_typed_pod_fanout_context(std::uint32_t fanout) {
  auto fixture = ChannelBenchmarkFixture::create(
      fanout, 2U, false, holoscan::detail::ProcessLocalChannelDeliveryKind::kHoloMQ);
  if (!fixture) {
    return holoscan::make_unexpected(std::move(fixture).error());
  }

  std::vector<std::unique_ptr<holoscan::detail::PortBinding>> bindings;
  bindings.reserve(fanout);
  for (std::uint32_t index = 0U; index < fanout; ++index) {
    auto binding = (*fixture)->session().create_readiness_binding(
        (*fixture)->sink_owner(index),
        holoscan::detail::RuntimePortBindingRequest{
            .plan_generation = (*fixture)->plan_generation(),
            .run_generation = 1U,
            .kind = holoscan::detail::RuntimeEventKind::kInputArrival,
            .source = holoscan::detail::make_source_id(static_cast<std::uint64_t>(index) + 1U),
            .source_clock = holoscan::ClockId{1U},
            .reference_clock = holoscan::ClockId{1U},
            .capacity = 1U,
        });
    if (!binding) {
      return holoscan::make_unexpected(std::move(binding).error());
    }
    bindings.push_back(std::move(*binding));
  }

  auto armed = (*fixture)->arm();
  if (!armed) {
    return holoscan::make_unexpected(std::move(armed).error());
  }
  // Discovery and one complete fanout transaction are cold setup, not measured work.
  std::this_thread::sleep_for(std::chrono::milliseconds{20});
  auto warmup = (*fixture)->publish_typed_pod(1U, 1);
  if (!warmup) {
    return holoscan::make_unexpected(std::move(warmup).error());
  }
  for (std::size_t index = 0U; index < bindings.size(); ++index) {
    auto received = (*fixture)->consume_typed_pod(*bindings[index], 1U, index);
    if (!received) {
      return holoscan::make_unexpected(std::move(received).error());
    }
    if (*received != 1) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kProtocolError, "fanout_warmup_value"});
    }
  }
  (*fixture)->reset_readiness_poll_count();
  return TypedPODFanoutContext{.fixture = std::move(*fixture), .bindings = std::move(bindings)};
}

[[nodiscard]] holoscan::expected<void, holoscan::Error> consume_typed_pod_fanout(
    TypedPODFanoutContext& context, std::uint64_t generation, std::int32_t expected_value) {
  for (std::size_t index = 0U; index < context.bindings.size(); ++index) {
    auto received = context.fixture->consume_typed_pod(*context.bindings[index], generation, index);
    if (!received) {
      return holoscan::make_unexpected(std::move(received).error());
    }
    if (*received != expected_value) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kProtocolError, "fanout_value"});
    }
  }
  return {};
}

// REQ: RUNTIME-016
// REQ: SYSTEM-005
void BM_ChannelActivationCycleByGraphOwners(benchmark::State& state) {
  const auto total_owner_count = static_cast<std::uint32_t>(state.range(0));
  auto fixture = ChannelBenchmarkFixture::create(total_owner_count - 1U, 1U, false);
  if (!fixture) {
    skip_with_error(state, "failed to create Channel activation fixture", fixture.error());
    return;
  }

  std::uint64_t generation = 1U;
  for (auto _ : state) {
    auto prepared =
        (*fixture)->session().prepare_activation(holoscan::VertexId{1U}, generation, {});
    if (!prepared) {
      skip_with_error(state, "failed to prepare Channel activation", prepared.error());
      return;
    }
    auto retired =
        (*fixture)->session().retire_activation(holoscan::VertexId{1U}, generation, true);
    if (!retired) {
      skip_with_error(state, "failed to retire Channel activation", retired.error());
      return;
    }
    benchmark::DoNotOptimize(generation);
    ++generation;
  }
  state.counters["total_graph_owners"] = static_cast<double>(total_owner_count);
  state.SetItemsProcessed(state.iterations());
}

// REQ: RUNTIME-016
// REQ: RUNTIME-018
void BM_ChannelReadinessBurst(benchmark::State& state) {
  const auto input_count = static_cast<std::uint32_t>(state.range(0));
  const auto queue_depth = static_cast<std::uint32_t>(state.range(1));
  auto fixture = ChannelBenchmarkFixture::create(input_count, queue_depth, true);
  if (!fixture) {
    skip_with_error(state, "failed to create Channel readiness fixture", fixture.error());
    return;
  }
  if ((*fixture)->burst_size() > std::numeric_limits<std::uint32_t>::max()) {
    state.SkipWithError("Channel readiness burst exceeds Runtime request capacity");
    return;
  }
  const auto capacity = static_cast<std::uint32_t>((*fixture)->burst_size());
  auto binding = (*fixture)->session().create_readiness_binding(
      (*fixture)->sink_owner(),
      holoscan::detail::RuntimePortBindingRequest{
          .plan_generation = (*fixture)->plan_generation(),
          .run_generation = 1U,
          .kind = holoscan::detail::RuntimeEventKind::kInputArrival,
          .source = holoscan::detail::make_source_id(1U),
          .source_clock = holoscan::ClockId{1U},
          .reference_clock = holoscan::ClockId{1U},
          .capacity = capacity,
      });
  if (!binding) {
    skip_with_error(state, "failed to bind Channel readiness", binding.error());
    return;
  }
  std::vector<holoscan::detail::RuntimeEvent> events((*fixture)->burst_size());
  std::uint64_t first_sequence = 1U;

  for (auto _ : state) {
    state.PauseTiming();
    auto injected = (*fixture)->inject_burst(first_sequence);
    state.ResumeTiming();
    if (!injected) {
      skip_with_error(state, "failed to inject Channel burst", injected.error());
      return;
    }

    auto polled = (*binding)->poll(holoscan::TimePoint{1, holoscan::ClockId{1U}});
    benchmark::DoNotOptimize(polled);
    state.PauseTiming();
    if (!polled) {
      skip_with_error(state, "failed to poll Channel burst", polled.error());
      return;
    }
    auto retired = (*fixture)->retire_burst(**binding, events);
    if (!retired) {
      skip_with_error(state, "failed to retire Channel burst", retired.error());
      return;
    }
    first_sequence += queue_depth;
    state.ResumeTiming();
  }
  state.counters["input_count"] = static_cast<double>(input_count);
  state.counters["queue_depth"] = static_cast<double>(queue_depth);
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>((*fixture)->burst_size()));
}

// Measure the production-shaped target-lane handoff for one revision-gated process-local input.
// Sample injection and witness retirement remain outside the timed interval.
// REQ: RUNTIME-011
// REQ: RUNTIME-016
// REQ: RUNTIME-026
void BM_ChannelReadinessPollViaBusyLane(benchmark::State& state) {
  auto fixture = ChannelBenchmarkFixture::create(1U, 1U, false);
  if (!fixture) {
    skip_with_error(state, "failed to create Channel readiness fixture", fixture.error());
    return;
  }
  auto binding = (*fixture)->session().create_readiness_binding(
      (*fixture)->sink_owner(),
      holoscan::detail::RuntimePortBindingRequest{
          .plan_generation = (*fixture)->plan_generation(),
          .run_generation = 1U,
          .kind = holoscan::detail::RuntimeEventKind::kInputArrival,
          .source = holoscan::detail::make_source_id(1U),
          .source_clock = holoscan::ClockId{1U},
          .reference_clock = holoscan::ClockId{1U},
          .capacity = 1U,
      });
  if (!binding) {
    skip_with_error(state, "failed to bind Channel readiness", binding.error());
    return;
  }
  auto lane_set = create_started_busy_lane_set();
  if (!lane_set) {
    skip_with_error(state, "failed to create Runtime lane set", lane_set.error());
    return;
  }

  std::array<holoscan::detail::RuntimeEvent, 1U> events;
  std::uint64_t sequence = 1U;
  for (auto _ : state) {
    state.PauseTiming();
    auto injected = (*fixture)->inject_one((*fixture)->last_input_port(), sequence);
    state.ResumeTiming();
    if (!injected) {
      skip_with_error(state, "failed to inject Channel readiness sample", injected.error());
      return;
    }

    const std::optional<holoscan::detail::PortBindingPollTarget> target = (*binding)->poll_target();
    if (!target) {
      state.SkipWithError("revision-gated Channel readiness did not advertise a poll target");
      return;
    }
    ReadinessPollInvocation invocation{
        .binding = binding->get(),
        .observed_at =
            holoscan::TimePoint{static_cast<std::int64_t>(sequence), holoscan::ClockId{1U}},
    };
    auto polled = (*lane_set)->dispatch(
        target->execution_lane,
        holoscan::detail::ExecutionLaneTask{
            .context = &invocation,
            .invoke = invoke_readiness_poll,
            .callback_context = holoscan::detail::RuntimeCallbackContext::kDispatcher,
        });
    benchmark::DoNotOptimize(polled);

    state.PauseTiming();
    if (!polled) {
      state.ResumeTiming();
      skip_with_error(state, "failed to poll Channel readiness on Runtime lane", polled.error());
      return;
    }
    auto retired = (*fixture)->retire_burst(**binding, events);
    if (!retired) {
      state.ResumeTiming();
      skip_with_error(state, "failed to retire Channel readiness witness", retired.error());
      return;
    }
    ++sequence;
    state.ResumeTiming();
  }
  state.counters["poll_lane_dispatches_per_cycle"] = 1.0;
  state.SetItemsProcessed(state.iterations());

  auto joined = (*lane_set)->join();
  if (!joined) {
    state.SkipWithError("failed to stop Runtime lane set");
  }
}

constexpr std::size_t kActivationCeilingLatencyCapacity = 8192U;

void report_activation_ceiling_percentiles(benchmark::State& state,
                                           std::span<std::uint64_t> samples,
                                           std::string_view prefix) {
  if (samples.empty()) {
    return;
  }
  std::sort(samples.begin(), samples.end());
  const auto nearest_rank = [&](std::size_t percentile) {
    const std::size_t rank = (samples.size() * percentile + 99U) / 100U;
    return samples[std::clamp(rank, std::size_t{1U}, samples.size()) - 1U];
  };
  state.counters[std::string(prefix) + "_p50_ns"] = static_cast<double>(nearest_rank(50U));
  state.counters[std::string(prefix) + "_p90_ns"] = static_cast<double>(nearest_rank(90U));
  state.counters[std::string(prefix) + "_p99_ns"] = static_cast<double>(nearest_rank(99U));
}

void report_activation_ceiling_thread_usage(benchmark::State& state,
                                            const holoscan::detail::RuntimeThreadUsageRecord& usage,
                                            std::string_view prefix, std::int64_t completed) {
  state.counters[std::string(prefix) + "_usage_supported"] = usage.supported ? 1.0 : 0.0;
  if (!usage.supported || completed <= 0) {
    return;
  }
  const double denominator = static_cast<double>(completed);
  state.counters[std::string(prefix) + "_cpu_ns_per_cycle"] =
      static_cast<double>(usage.user_cpu_ns + usage.system_cpu_ns) / denominator;
  state.counters[std::string(prefix) + "_context_switches_per_cycle"] =
      static_cast<double>(usage.voluntary_context_switches + usage.involuntary_context_switches) /
      denominator;
}

// Measure an optimistic benchmark-only activation ceiling using real Runtime lane handoffs and
// direct-retained Channel transactions. The three arms execute the same source and successor
// bodies; only their physical lane-task geometry differs. This is not a production continuation
// or correctness-evidence path.
// This benchmark intentionally keeps all three execution modes in one routine so their setup and
// measured bodies cannot drift apart.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void run_direct_retained_activation_ceiling(benchmark::State& state,
                                            DirectRetainedActivationCeilingMode mode,
                                            holoscan::WakeMode wake_mode) {
  const auto callback_work_us = static_cast<std::uint32_t>(state.range(0));
  const auto queue_depth = static_cast<std::uint32_t>(state.range(1));
  if ((callback_work_us != 0U && callback_work_us != 1U && callback_work_us != 5U &&
       callback_work_us != 20U && callback_work_us != 100U) ||
      (queue_depth != 1U && queue_depth != 2U)) {
    state.SkipWithError("activation ceiling arguments are outside the benchmark bounds");
    return;
  }

  auto context = create_typed_pod_cycle_context(
      holoscan::detail::ProcessLocalChannelDeliveryKind::kDirectRetained, queue_depth);
  if (!context) {
    skip_with_error(state, "failed to create activation ceiling fixture", context.error());
    return;
  }
  holoscan::detail::RuntimePhaseCounters phase_counters;
  auto lane_set = create_started_lane_set(wake_mode, &phase_counters);
  if (!lane_set) {
    skip_with_error(state, "failed to create activation ceiling lane", lane_set.error());
    return;
  }

  std::array<std::uint64_t, kActivationCeilingLatencyCapacity> source_to_successor_samples{};
  std::array<std::uint64_t, kActivationCeilingLatencyCapacity> total_transaction_samples{};
  std::size_t sample_count = 0U;
  std::uint64_t generation = 2U;
  std::int64_t completed = 0;
  std::uint32_t maximum_callback_concurrency = 0U;
  bool callback_overlap = false;
  bool failed = false;
  {
    const holoscan::detail::RuntimeThreadUsageScope controller_usage{
        &phase_counters, holoscan::detail::RuntimeThreadUsageRole::kController};
    for (auto _ : state) {
      DirectRetainedActivationCeilingInvocation invocation{
          .fixture = context->fixture.get(),
          .binding = context->binding.get(),
          .mode = mode,
          .callback_work = std::chrono::microseconds{callback_work_us},
          .generation = generation,
          .value = static_cast<std::int32_t>(generation),
      };
      auto prepared = context->fixture->prepare_typed_pod_source(generation);
      if (!prepared) {
        skip_with_error(state, "failed to prepare activation ceiling source", prepared.error());
        failed = true;
        break;
      }
      if (mode != DirectRetainedActivationCeilingMode::kOrdinary) {
        const auto target =
            context->fixture->session().arm_direct_continuation(holoscan::VertexId{1U}, 0U);
        if (!target) {
          state.SkipWithError("failed to arm activation ceiling continuation");
          failed = true;
          break;
        }
      }

      const std::int64_t transaction_start_ns = steady_timestamp_ns();
      auto source = (*lane_set)->dispatch(
          0U,
          holoscan::detail::ExecutionLaneTask{
              .context = &invocation,
              .invoke = invoke_direct_retained_ceiling_source,
              .callback_context = holoscan::detail::RuntimeCallbackContext::kCompute,
          });
      holoscan::expected<void, holoscan::Error> retired;
      holoscan::expected<void, holoscan::Error> flushed;
      if (source && mode != DirectRetainedActivationCeilingMode::kComposite) {
        retired = context->fixture->retire_typed_pod_source(generation);
        if (retired) {
          flushed = context->fixture->flush_process_local_cut();
        }
        if (mode == DirectRetainedActivationCeilingMode::kReceiptElided && retired && flushed) {
          invocation.receipt_validated =
              invocation.receipt.has_value() &&
              context->fixture->session().validate_direct_continuation_receipt(*invocation.receipt);
        }
      }
      holoscan::expected<void, holoscan::Error> polled;
      if (source && retired && flushed && mode == DirectRetainedActivationCeilingMode::kOrdinary) {
        ReadinessPollInvocation poll{
            .binding = context->binding.get(),
            .observed_at =
                holoscan::TimePoint{static_cast<std::int64_t>(generation), holoscan::ClockId{1U}},
        };
        polled = (*lane_set)->dispatch(
            0U,
            holoscan::detail::ExecutionLaneTask{
                .context = &poll,
                .invoke = invoke_readiness_poll,
                .callback_context = holoscan::detail::RuntimeCallbackContext::kDispatcher,
            });
      }
      holoscan::expected<void, holoscan::Error> successor;
      const bool successor_ready = mode == DirectRetainedActivationCeilingMode::kOrdinary
                                       ? static_cast<bool>(polled)
                                       : invocation.receipt_validated;
      if (source && retired && flushed && successor_ready &&
          mode != DirectRetainedActivationCeilingMode::kComposite) {
        successor = (*lane_set)->dispatch(
            0U,
            holoscan::detail::ExecutionLaneTask{
                .context = &invocation,
                .invoke = invoke_direct_retained_ceiling_successor,
                .callback_context = holoscan::detail::RuntimeCallbackContext::kCompute,
            });
      }
      const std::int64_t transaction_finish_ns = steady_timestamp_ns();

      if (!source) {
        skip_with_error(state, "activation ceiling source failed", source.error());
        failed = true;
        break;
      }
      if (mode != DirectRetainedActivationCeilingMode::kComposite && !retired) {
        skip_with_error(state, "activation ceiling source retirement failed", retired.error());
        failed = true;
        break;
      }
      if (mode != DirectRetainedActivationCeilingMode::kComposite && !flushed) {
        skip_with_error(state, "activation ceiling source cut failed", flushed.error());
        failed = true;
        break;
      }
      if (mode == DirectRetainedActivationCeilingMode::kOrdinary && !polled) {
        skip_with_error(state, "activation ceiling readiness poll failed", polled.error());
        failed = true;
        break;
      }
      if (mode != DirectRetainedActivationCeilingMode::kComposite && !successor) {
        skip_with_error(state, "activation ceiling successor failed", successor.error());
        failed = true;
        break;
      }
      if (invocation.received != invocation.value || invocation.callback_stage != 3U ||
          invocation.callback_count != 2U || invocation.active_callbacks != 0U ||
          invocation.max_active_callbacks != 1U || invocation.callback_overlap ||
          !invocation.callback_order_valid ||
          (mode != DirectRetainedActivationCeilingMode::kOrdinary &&
           !invocation.receipt_validated) ||
          invocation.successor_entry_ns < invocation.source_publication_ns ||
          transaction_finish_ns < transaction_start_ns) {
        state.SkipWithError("activation ceiling invariants failed");
        failed = true;
        break;
      }

      maximum_callback_concurrency =
          std::max(maximum_callback_concurrency, invocation.max_active_callbacks);
      callback_overlap = callback_overlap || invocation.callback_overlap;
      const auto source_to_successor_ns = static_cast<std::uint64_t>(
          invocation.successor_entry_ns - invocation.source_publication_ns);
      const auto total_transaction_ns =
          static_cast<std::uint64_t>(transaction_finish_ns - transaction_start_ns);
      if (sample_count < source_to_successor_samples.size()) {
        source_to_successor_samples[sample_count] = source_to_successor_ns;
        total_transaction_samples[sample_count] = total_transaction_ns;
        ++sample_count;
      }
      state.SetIterationTime(static_cast<double>(total_transaction_ns) * 1.0e-9);
      ++generation;
      ++completed;
    }
  }

  auto joined = (*lane_set)->join();
  if (!joined) {
    state.SkipWithError("failed to stop activation ceiling lane");
    return;
  }
  if (failed) {
    return;
  }

  const auto phase_snapshot = phase_counters.snapshot();
  const auto phase_count = [&](holoscan::detail::RuntimePhaseCounter counter) {
    return phase_snapshot[holoscan::detail::RuntimePhaseCounters::index(counter)];
  };
  const std::uint64_t lane_tasks_per_cycle =
      mode == DirectRetainedActivationCeilingMode::kOrdinary
          ? 3U
          : (mode == DirectRetainedActivationCeilingMode::kReceiptElided ? 2U : 1U);
  const std::uint64_t expected_lane_tasks =
      static_cast<std::uint64_t>(completed) * lane_tasks_per_cycle;
  const std::uint64_t expected_compute_tasks =
      static_cast<std::uint64_t>(completed) *
      (mode == DirectRetainedActivationCeilingMode::kComposite ? 1U : 2U);
  const std::uint64_t expected_dispatcher_tasks =
      mode == DirectRetainedActivationCeilingMode::kOrdinary ? static_cast<std::uint64_t>(completed)
                                                             : 0U;
  if (phase_count(holoscan::detail::RuntimePhaseCounter::kLaneDispatches) != expected_lane_tasks ||
      phase_count(holoscan::detail::RuntimePhaseCounter::kLaneCompletionsObserved) !=
          expected_lane_tasks ||
      phase_count(holoscan::detail::RuntimePhaseCounter::kLaneComputeDispatches) !=
          expected_compute_tasks ||
      phase_count(holoscan::detail::RuntimePhaseCounter::kLaneDispatcherDispatches) !=
          expected_dispatcher_tasks) {
    state.SkipWithError("activation ceiling lane-task cardinality failed");
    return;
  }

  report_activation_ceiling_percentiles(state,
                                        std::span{source_to_successor_samples}.first(sample_count),
                                        "source_publication_to_successor_entry");
  report_activation_ceiling_percentiles(
      state, std::span{total_transaction_samples}.first(sample_count), "total_transaction");
  const auto thread_usage = phase_counters.thread_usage_snapshot();
  report_activation_ceiling_thread_usage(
      state,
      thread_usage[holoscan::detail::RuntimePhaseCounters::thread_usage_index(
          holoscan::detail::RuntimeThreadUsageRole::kController)],
      "controller",
      completed);
  report_activation_ceiling_thread_usage(
      state,
      thread_usage[holoscan::detail::RuntimePhaseCounters::thread_usage_index(
          holoscan::detail::RuntimeThreadUsageRole::kExecutionLane)],
      "execution_lane",
      completed);
  state.counters["callback_work_us"] = static_cast<double>(callback_work_us);
  state.counters["queue_depth"] = static_cast<double>(queue_depth);
  state.counters["lane_tasks_per_cycle"] = static_cast<double>(lane_tasks_per_cycle);
  state.counters["logical_callbacks_per_cycle"] = 2.0;
  state.counters["maximum_callback_concurrency"] =
      static_cast<double>(maximum_callback_concurrency);
  state.counters["callback_overlap"] = callback_overlap ? 1.0 : 0.0;
  state.counters["receipt_validated"] =
      mode == DirectRetainedActivationCeilingMode::kOrdinary ? 0.0 : 1.0;
  state.SetItemsProcessed(completed);
}

void BM_ChannelDirectRetainedActivationCeiling(benchmark::State& state,
                                               DirectRetainedActivationCeilingMode mode,
                                               holoscan::WakeMode wake_mode) {
  run_direct_retained_activation_ceiling(state, mode, wake_mode);
}

void apply_activation_ceiling_arguments(benchmark::Benchmark* benchmark) {
  for (const std::int64_t callback_work_us : {0, 1, 5, 20, 100}) {
    for (const std::int64_t queue_depth : {1, 2}) {
      benchmark->Args({callback_work_us, queue_depth});
    }
  }
}

// Measure the actual direct-retained poll after synchronous publication has already installed
// typed storage and advanced its owner-local readiness revision, with an optional target-lane
// handoff matching production dispatch.
// REQ: CHAN-096
// REQ: RUNTIME-011
// REQ: RUNTIME-016
// REQ: RUNTIME-026
void run_direct_retained_readiness_poll_benchmark(benchmark::State& state, bool via_busy_lane) {
  auto context = create_typed_pod_cycle_context(
      holoscan::detail::ProcessLocalChannelDeliveryKind::kDirectRetained);
  if (!context) {
    skip_with_error(state, "failed to create direct-retained Channel fixture", context.error());
    return;
  }
  std::unique_ptr<holoscan::detail::ExecutionLaneSet> lane_set;
  if (via_busy_lane) {
    auto created = create_started_busy_lane_set();
    if (!created) {
      skip_with_error(state, "failed to create Runtime lane set", created.error());
      return;
    }
    lane_set = std::move(*created);
  }

  std::uint64_t generation = 2U;
  for (auto _ : state) {
    state.PauseTiming();
    const auto value = static_cast<std::int32_t>(generation);
    auto published = context->fixture->publish_typed_pod(generation, value);
    state.ResumeTiming();
    if (!published) {
      skip_with_error(state, "failed to publish direct-retained Channel POD", published.error());
      return;
    }

    const std::optional<holoscan::detail::PortBindingPollTarget> target =
        context->binding->poll_target();
    if (!target) {
      state.SkipWithError("direct-retained readiness did not advertise a poll target");
      return;
    }
    ReadinessPollInvocation invocation{
        .binding = context->binding.get(),
        .observed_at =
            holoscan::TimePoint{static_cast<std::int64_t>(generation), holoscan::ClockId{1U}},
    };
    holoscan::expected<void, holoscan::Error> polled;
    if (via_busy_lane) {
      polled = lane_set->dispatch(
          target->execution_lane,
          holoscan::detail::ExecutionLaneTask{
              .context = &invocation,
              .invoke = invoke_readiness_poll,
              .callback_context = holoscan::detail::RuntimeCallbackContext::kDispatcher,
          });
    } else {
      polled = context->binding->poll(invocation.observed_at);
    }
    benchmark::DoNotOptimize(polled);

    state.PauseTiming();
    if (!polled) {
      state.ResumeTiming();
      skip_with_error(state, "failed to poll direct-retained readiness", polled.error());
      return;
    }
    auto received = context->fixture->consume_typed_pod_after_poll(*context->binding, generation);
    if (!received || *received != value) {
      state.ResumeTiming();
      if (!received) {
        skip_with_error(state, "failed to clean up direct-retained poll", received.error());
      } else {
        state.SkipWithError("direct-retained poll cleanup returned the wrong value");
      }
      return;
    }
    ++generation;
    state.ResumeTiming();
  }
  state.counters["poll_lane_dispatches_per_cycle"] = via_busy_lane ? 1.0 : 0.0;
  state.SetItemsProcessed(state.iterations());

  if (lane_set) {
    auto joined = lane_set->join();
    if (!joined) {
      state.SkipWithError("failed to stop Runtime lane set");
    }
  }
}

void BM_ChannelDirectRetainedReadinessPoll(benchmark::State& state) {
  run_direct_retained_readiness_poll_benchmark(state, false);
}

void BM_ChannelDirectRetainedReadinessPollViaBusyLane(benchmark::State& state) {
  run_direct_retained_readiness_poll_benchmark(state, true);
}

// Bound an optimistic target-lane continuation by combining readiness polling, typed receive,
// claim retirement, and the sink activation cut in one task. Production still returns to the
// controller for global candidate ordering and activation selection between poll and compute.
// REQ: CHAN-096
// REQ: RUNTIME-011
// REQ: RUNTIME-016
// REQ: RUNTIME-026
void BM_ChannelDirectRetainedFusedPollConsumeCeiling(benchmark::State& state) {
  auto context = create_typed_pod_cycle_context(
      holoscan::detail::ProcessLocalChannelDeliveryKind::kDirectRetained);
  if (!context) {
    skip_with_error(state, "failed to create direct-retained Channel fixture", context.error());
    return;
  }
  auto lane_set = create_started_busy_lane_set();
  if (!lane_set) {
    skip_with_error(state, "failed to create Runtime lane set", lane_set.error());
    return;
  }

  std::uint64_t generation = 2U;
  for (auto _ : state) {
    state.PauseTiming();
    const auto value = static_cast<std::int32_t>(generation);
    auto published = context->fixture->publish_typed_pod(generation, value);
    state.ResumeTiming();
    if (!published) {
      skip_with_error(state, "failed to publish direct-retained Channel POD", published.error());
      return;
    }

    DirectRetainedConsumeInvocation invocation{
        .fixture = context->fixture.get(),
        .binding = context->binding.get(),
        .generation = generation,
    };
    auto consumed = (*lane_set)->dispatch(
        0U,
        holoscan::detail::ExecutionLaneTask{
            .context = &invocation,
            .invoke = invoke_direct_retained_consume,
            .callback_context = holoscan::detail::RuntimeCallbackContext::kCompute,
        });
    benchmark::DoNotOptimize(consumed);

    state.PauseTiming();
    if (!consumed) {
      state.ResumeTiming();
      skip_with_error(state, "failed to run fused direct-retained consumption", consumed.error());
      return;
    }
    if (invocation.received != value) {
      state.ResumeTiming();
      state.SkipWithError("fused direct-retained consumption returned the wrong value");
      return;
    }
    ++generation;
    state.ResumeTiming();
  }
  state.counters["fused_poll_consume_tasks_per_cycle"] = 1.0;
  state.SetItemsProcessed(state.iterations());

  auto joined = (*lane_set)->join();
  if (!joined) {
    state.SkipWithError("failed to stop Runtime lane set");
  }
}

// Measure the current two-task mechanism without the controller's selection work between tasks.
// Publication remains outside timing; the first task polls readiness and the second drains and
// consumes the already-polled sample through the production Channel state.
// REQ: CHAN-096
// REQ: RUNTIME-011
// REQ: RUNTIME-016
// REQ: RUNTIME-026
void BM_ChannelDirectRetainedTwoDispatchPollConsume(benchmark::State& state) {
  auto context = create_typed_pod_cycle_context(
      holoscan::detail::ProcessLocalChannelDeliveryKind::kDirectRetained);
  if (!context) {
    skip_with_error(state, "failed to create direct-retained Channel fixture", context.error());
    return;
  }
  auto lane_set = create_started_busy_lane_set();
  if (!lane_set) {
    skip_with_error(state, "failed to create Runtime lane set", lane_set.error());
    return;
  }
  std::uint64_t generation = 2U;
  for (auto _ : state) {
    state.PauseTiming();
    const auto value = static_cast<std::int32_t>(generation);
    auto published = context->fixture->publish_typed_pod(generation, value);
    state.ResumeTiming();
    if (!published) {
      skip_with_error(state, "failed to publish direct-retained Channel POD", published.error());
      return;
    }

    const std::optional<holoscan::detail::PortBindingPollTarget> target =
        context->binding->poll_target();
    if (!target) {
      state.SkipWithError("direct-retained readiness did not advertise a poll target");
      return;
    }

    ReadinessPollInvocation poll{
        .binding = context->binding.get(),
        .observed_at =
            holoscan::TimePoint{static_cast<std::int64_t>(generation), holoscan::ClockId{1U}},
    };
    auto polled = (*lane_set)->dispatch(
        target->execution_lane,
        holoscan::detail::ExecutionLaneTask{
            .context = &poll,
            .invoke = invoke_readiness_poll,
            .callback_context = holoscan::detail::RuntimeCallbackContext::kDispatcher,
        });
    DirectRetainedConsumeInvocation consume{
        .fixture = context->fixture.get(),
        .binding = context->binding.get(),
        .generation = generation,
        .readiness_already_polled = true,
    };
    holoscan::expected<void, holoscan::Error> consumed;
    if (polled) {
      consumed = (*lane_set)->dispatch(
          target->execution_lane,
          holoscan::detail::ExecutionLaneTask{
              .context = &consume,
              .invoke = invoke_direct_retained_consume,
              .callback_context = holoscan::detail::RuntimeCallbackContext::kCompute,
          });
    } else {
      consumed = holoscan::make_unexpected(polled.error());
    }
    benchmark::DoNotOptimize(consumed);

    state.PauseTiming();
    if (!polled) {
      state.ResumeTiming();
      skip_with_error(state, "failed to poll direct-retained readiness", polled.error());
      return;
    }
    if (!consumed) {
      state.ResumeTiming();
      skip_with_error(state, "failed to consume direct-retained Channel POD", consumed.error());
      return;
    }
    if (consume.received != value) {
      state.ResumeTiming();
      state.SkipWithError("two-dispatch consumption returned the wrong value");
      return;
    }
    ++generation;
    state.ResumeTiming();
  }
  state.counters["lane_tasks_per_cycle"] = 2.0;
  state.SetItemsProcessed(state.iterations());

  auto joined = (*lane_set)->join();
  if (!joined) {
    state.SkipWithError("failed to stop Runtime lane set");
  }
}

// Estimate a semantic-preserving split-phase mechanism: the lane reports that polling completed,
// waits for controller authorization, then consumes without returning and publishing a second task.
// The probe intentionally contains no Dispatcher or activation-policy work.
// REQ: CHAN-096
// REQ: RUNTIME-011
// REQ: RUNTIME-016
// REQ: RUNTIME-026
void BM_ChannelDirectRetainedResumablePollConsumeProbe(benchmark::State& state) {
  auto context = create_typed_pod_cycle_context(
      holoscan::detail::ProcessLocalChannelDeliveryKind::kDirectRetained);
  if (!context) {
    skip_with_error(state, "failed to create direct-retained Channel fixture", context.error());
    return;
  }
  DirectRetainedContinuationProbe probe{*context->fixture, *context->binding};

  std::uint64_t generation = 2U;
  for (auto _ : state) {
    state.PauseTiming();
    const auto value = static_cast<std::int32_t>(generation);
    auto published = context->fixture->publish_typed_pod(generation, value);
    state.ResumeTiming();
    if (!published) {
      skip_with_error(state, "failed to publish direct-retained Channel POD", published.error());
      return;
    }

    auto received = probe.poll_authorize_consume(generation);
    benchmark::DoNotOptimize(received);

    state.PauseTiming();
    if (!received) {
      state.ResumeTiming();
      skip_with_error(state, "failed to resume direct-retained consumption", received.error());
      return;
    }
    if (*received != value) {
      state.ResumeTiming();
      state.SkipWithError("resumable consumption returned the wrong value");
      return;
    }
    ++generation;
    state.ResumeTiming();
  }
  state.counters["controller_authorizations_per_cycle"] = 1.0;
  state.counters["lane_tasks_per_cycle"] = 1.0;
  state.SetItemsProcessed(state.iterations());
}

// REQ: RUNTIME-016
// REQ: RUNTIME-018
void BM_ChannelLastWitnessRetirementByGraphInputs(benchmark::State& state) {
  const auto input_count = static_cast<std::uint32_t>(state.range(0));
  auto fixture = ChannelBenchmarkFixture::create(input_count, 1U, false);
  if (!fixture) {
    skip_with_error(state, "failed to create Channel retirement fixture", fixture.error());
    return;
  }
  // Target the final owner so a session-wide lookup must traverse every preceding subscriber.
  auto binding = (*fixture)->session().create_readiness_binding(
      (*fixture)->last_sink_owner(),
      holoscan::detail::RuntimePortBindingRequest{
          .plan_generation = (*fixture)->plan_generation(),
          .run_generation = 1U,
          .kind = holoscan::detail::RuntimeEventKind::kInputArrival,
          .source = holoscan::detail::make_source_id(1U),
          .source_clock = holoscan::ClockId{1U},
          .reference_clock = holoscan::ClockId{1U},
          .capacity = 1U,
      });
  if (!binding) {
    skip_with_error(state, "failed to bind Channel retirement", binding.error());
    return;
  }
  const holoscan::PortId input = (*fixture)->last_input_port();
  std::array<holoscan::detail::RuntimeEvent, 1U> events;
  std::uint64_t sequence = 1U;

  for (auto _ : state) {
    state.PauseTiming();
    auto injected = (*fixture)->inject_one(input, sequence);
    if (!injected) {
      state.ResumeTiming();
      skip_with_error(state, "failed to inject Channel retirement sample", injected.error());
      return;
    }
    auto polled = (*binding)->poll(holoscan::TimePoint{1, holoscan::ClockId{1U}});
    if (!polled) {
      state.ResumeTiming();
      skip_with_error(state, "failed to poll Channel retirement sample", polled.error());
      return;
    }
    auto drained = (*binding)->drain(events);
    if (!drained || *drained != 1U) {
      state.ResumeTiming();
      state.SkipWithError("failed to drain Channel retirement witness");
      return;
    }
    auto record = (*fixture)->session().temporal_sample(input, events[0].source_sequence);
    if (!record) {
      state.ResumeTiming();
      skip_with_error(state, "failed to resolve Channel retirement sample", record.error());
      return;
    }
    state.ResumeTiming();

    auto retired = (*binding)->retire(events[0].source_sequence);
    benchmark::DoNotOptimize(retired);

    state.PauseTiming();
    if (!retired) {
      state.ResumeTiming();
      skip_with_error(state, "failed to retire Channel witness", retired.error());
      return;
    }
    auto released = (*fixture)->session().retire_samples(
        input, std::span<const holoscan::detail::TemporalSampleRecord>{&*record, 1U});
    if (!released) {
      state.ResumeTiming();
      skip_with_error(state, "failed to release Channel retirement sample", released.error());
      return;
    }
    ++sequence;
    state.ResumeTiming();
  }
  state.counters["total_graph_inputs"] = static_cast<double>(input_count);
  state.SetItemsProcessed(state.iterations());
}

// Measure one complete production Runtime Channel edge without Dispatcher or execution-lane work.
// REQ: RUNTIME-016
// REQ: RUNTIME-018
void run_typed_pod_cycle_benchmark(
    benchmark::State& state, holoscan::detail::ProcessLocalChannelDeliveryKind delivery_kind) {
  auto context = create_typed_pod_cycle_context(delivery_kind);
  if (!context) {
    skip_with_error(state, "failed to create typed Channel cycle fixture", context.error());
    return;
  }

  std::uint64_t generation = 2U;
  for (auto _ : state) {
    const auto value = static_cast<std::int32_t>(generation);
    auto received = context->fixture->run_typed_pod_cycle(*context->binding, generation, value);
    if (!received || *received != value) {
      if (!received) {
        skip_with_error(state, "failed to run typed Channel cycle", received.error());
      } else {
        state.SkipWithError("typed Channel cycle returned the wrong value");
      }
      return;
    }
    benchmark::DoNotOptimize(*received);
    ++generation;
  }
  state.counters["activation_cuts_per_cycle"] = 2.0;
  state.counters["payload_bytes"] = static_cast<double>(sizeof(std::int32_t));
  state.counters["direct_retained"] =
      delivery_kind == holoscan::detail::ProcessLocalChannelDeliveryKind::kDirectRetained ? 1.0
                                                                                          : 0.0;
  state.counters["readiness_polls_per_cycle"] =
      static_cast<double>(context->fixture->readiness_poll_count()) /
      static_cast<double>(state.iterations());
  const std::uint64_t expected_polls = static_cast<std::uint64_t>(state.iterations());
  const std::uint64_t extra_polls = context->fixture->readiness_poll_count() > expected_polls
                                        ? context->fixture->readiness_poll_count() - expected_polls
                                        : 0U;
  state.counters["readiness_poll_retries_per_cycle"] =
      static_cast<double>(extra_polls) / static_cast<double>(state.iterations());
  state.counters["comparison_uncontaminated"] = extra_polls == 0U ? 1.0 : 0.0;
  state.SetItemsProcessed(state.iterations());
}

void BM_ChannelTypedPODCycle(benchmark::State& state) {
  run_typed_pod_cycle_benchmark(state, holoscan::detail::ProcessLocalChannelDeliveryKind::kHoloMQ);
}

// Measure the same complete Runtime Channel edge with compiler-authorized direct retention.
// The synchronous publication bypasses HoloMQ, event notification, and pump work.
// REQ: CHAN-096
// REQ: RUNTIME-016
// REQ: RUNTIME-018
void BM_ChannelTypedPODDirectRetainedCycle(benchmark::State& state) {
  run_typed_pod_cycle_benchmark(state,
                                holoscan::detail::ProcessLocalChannelDeliveryKind::kDirectRetained);
}

[[nodiscard]] holoscan::BenchmarkPOD64 make_channel_pod64_payload(
    const std::uint64_t sequence) noexcept {
  holoscan::BenchmarkPOD64 payload{
      .sequence = sequence,
      .auxiliary0 = static_cast<std::uint32_t>(sequence) ^ 0xA5A5'5A5AU,
      .auxiliary1 = sequence * 3U,
  };
  payload.checksum = holoscan::benchmark_pod_checksum(payload);
  return payload;
}

// Match the production Graph's 64-byte POD, blocking depth-one, direct-retained edge while
// deliberately excluding Dispatcher selection and the two Operator callback transactions.
// REQ: CHAN-096
// REQ: RUNTIME-016
// REQ: RUNTIME-018
void BM_ChannelTypedPOD64DirectRetainedCycle(benchmark::State& state) {
  constexpr std::uint32_t kQueueDepth = 1U;
  auto context = create_typed_pod_cycle_context_for<holoscan::BenchmarkPOD64>(
      holoscan::detail::ProcessLocalChannelDeliveryKind::kDirectRetained,
      kQueueDepth,
      make_channel_pod64_payload(1U));
  if (!context) {
    skip_with_error(
        state, "failed to create 64-byte direct-retained Channel fixture", context.error());
    return;
  }

  std::uint64_t generation = 2U;
  holoscan::BenchmarkPOD64 final_received;
  holoscan::BenchmarkPOD64 final_expected;
  for (auto _ : state) {
    final_expected = make_channel_pod64_payload(generation);
    auto received =
        context->fixture->run_typed_pod_cycle(*context->binding, generation, final_expected);
    if (!received) {
      skip_with_error(
          state, "failed to run 64-byte direct-retained Channel cycle", received.error());
      return;
    }
    final_received = *received;
    benchmark::DoNotOptimize(final_received.sequence);
    ++generation;
  }
  if (!holoscan::benchmark_pod_valid(final_received) ||
      !has_equal_object_representation(final_received, final_expected)) {
    state.SkipWithError("64-byte direct-retained Channel cycle returned the wrong final value");
    return;
  }

  state.counters["activation_cuts_per_cycle"] = 2.0;
  state.counters["comparison_uncontaminated"] =
      context->fixture->readiness_poll_count() == state.iterations() ? 1.0 : 0.0;
  state.counters["direct_retained"] = 1.0;
  state.counters["final_validation"] = 1.0;
  state.counters["payload_bytes"] = static_cast<double>(sizeof(holoscan::BenchmarkPOD64));
  state.counters["queue_depth"] = kQueueDepth;
  state.counters["readiness_polls_per_cycle"] =
      static_cast<double>(context->fixture->readiness_poll_count()) /
      static_cast<double>(state.iterations());
  state.SetItemsProcessed(state.iterations());
}

// Time producer activation, publication, retirement, and the post-source Channel cut. The
// downstream receive and cleanup run outside the manual interval so the bounded edge remains
// reusable.
// REQ: RUNTIME-016
// REQ: RUNTIME-018
void BM_ChannelTypedPODPublishCut(benchmark::State& state) {
  auto context = create_typed_pod_cycle_context();
  if (!context) {
    skip_with_error(state, "failed to create typed Channel publish fixture", context.error());
    return;
  }

  std::uint64_t generation = 2U;
  for (auto _ : state) {
    const auto value = static_cast<std::int32_t>(generation);
    const auto start = std::chrono::steady_clock::now();
    auto published = context->fixture->publish_typed_pod(generation, value);
    const auto finish = std::chrono::steady_clock::now();
    if (!published) {
      skip_with_error(state, "failed to publish typed Channel POD", published.error());
      return;
    }
    state.SetIterationTime(std::chrono::duration<double>{finish - start}.count());

    auto received = context->fixture->consume_typed_pod(*context->binding, generation);
    if (!received || *received != value) {
      if (!received) {
        skip_with_error(state, "failed to clean up typed Channel publication", received.error());
      } else {
        state.SkipWithError("typed Channel publication cleanup returned the wrong value");
      }
      return;
    }
    ++generation;
  }
  state.counters["activation_cuts_per_cycle"] = 1.0;
  state.counters["payload_bytes"] = static_cast<double>(sizeof(std::int32_t));
  state.SetItemsProcessed(state.iterations());
}

// Time producer activation, accepted emit, and retirement before the controller establishes the
// process-local cut. Cut and sink cleanup remain outside the manual interval.
// REQ: RUNTIME-016
// REQ: RUNTIME-018
void BM_ChannelTypedPODPublishRetire(benchmark::State& state) {
  auto context = create_typed_pod_cycle_context();
  if (!context) {
    skip_with_error(state, "failed to create typed Channel pre-cut fixture", context.error());
    return;
  }

  std::uint64_t generation = 2U;
  for (auto _ : state) {
    const auto value = static_cast<std::int32_t>(generation);
    const auto start = std::chrono::steady_clock::now();
    auto published = context->fixture->publish_typed_pod_before_cut(generation, value);
    const auto finish = std::chrono::steady_clock::now();
    if (!published) {
      skip_with_error(state, "failed to publish typed Channel POD before cut", published.error());
      return;
    }
    state.SetIterationTime(std::chrono::duration<double>{finish - start}.count());

    auto flushed = context->fixture->flush_process_local_cut();
    if (!flushed) {
      skip_with_error(state, "failed to clean up typed Channel source cut", flushed.error());
      return;
    }
    auto received = context->fixture->consume_typed_pod(*context->binding, generation);
    if (!received || *received != value) {
      if (!received) {
        skip_with_error(state, "failed to clean up typed Channel pre-cut row", received.error());
      } else {
        state.SkipWithError("typed Channel pre-cut cleanup returned the wrong value");
      }
      return;
    }
    ++generation;
  }
  state.counters["activation_cuts_per_cycle"] = 0.0;
  state.counters["payload_bytes"] = static_cast<double>(sizeof(std::int32_t));
  state.SetItemsProcessed(state.iterations());
}

// Time only Output<T>::emit() on an already prepared source activation. Source retirement, the
// process-local cut, and downstream cleanup remain outside the manual interval.
// REQ: RUNTIME-016
// REQ: RUNTIME-018
void run_typed_pod_emit_benchmark(benchmark::State& state,
                                  holoscan::detail::ProcessLocalChannelDeliveryKind delivery_kind) {
  auto context = create_typed_pod_cycle_context(delivery_kind);
  if (!context) {
    skip_with_error(state, "failed to create typed Channel emit fixture", context.error());
    return;
  }

  std::uint64_t generation = 2U;
  for (auto _ : state) {
    const auto value = static_cast<std::int32_t>(generation);
    auto prepared = context->fixture->prepare_typed_pod_source(generation);
    if (!prepared) {
      skip_with_error(state, "failed to prepare typed Channel emit", prepared.error());
      return;
    }

    const auto start = std::chrono::steady_clock::now();
    auto emitted = context->fixture->emit_typed_pod(value);
    const auto finish = std::chrono::steady_clock::now();
    if (!emitted) {
      skip_with_error(state, "failed to emit typed Channel POD", emitted.error());
      return;
    }
    state.SetIterationTime(std::chrono::duration<double>{finish - start}.count());

    auto retired = context->fixture->retire_typed_pod_source(generation);
    if (!retired) {
      skip_with_error(state, "failed to retire typed Channel emit", retired.error());
      return;
    }
    auto flushed = context->fixture->flush_process_local_cut();
    if (!flushed) {
      skip_with_error(state, "failed to clean up typed Channel emit cut", flushed.error());
      return;
    }
    auto received = context->fixture->consume_typed_pod(*context->binding, generation);
    if (!received || *received != value) {
      if (!received) {
        skip_with_error(state, "failed to clean up typed Channel emit row", received.error());
      } else {
        state.SkipWithError("typed Channel emit cleanup returned the wrong value");
      }
      return;
    }
    ++generation;
  }
  state.counters["activation_cuts_per_cycle"] = 0.0;
  state.counters["payload_bytes"] = static_cast<double>(sizeof(std::int32_t));
  state.counters["direct_retained"] =
      delivery_kind == holoscan::detail::ProcessLocalChannelDeliveryKind::kDirectRetained ? 1.0
                                                                                          : 0.0;
  state.SetItemsProcessed(state.iterations());
}

void BM_ChannelTypedPODEmit(benchmark::State& state) {
  run_typed_pod_emit_benchmark(state, holoscan::detail::ProcessLocalChannelDeliveryKind::kHoloMQ);
}

// Isolate the synchronous direct-retained Output emit after source activation preparation.
// REQ: CHAN-096
// REQ: RUNTIME-016
// REQ: RUNTIME-018
void BM_ChannelTypedPODDirectRetainedEmit(benchmark::State& state) {
  run_typed_pod_emit_benchmark(state,
                               holoscan::detail::ProcessLocalChannelDeliveryKind::kDirectRetained);
}

// Isolate one HoloMQ-backed Output emit while the service fanout varies. Source retirement,
// process-local ingestion, and all sink work remain outside the manual interval.
// REQ: RUNTIME-016
// REQ: RUNTIME-018
void BM_ChannelTypedPODFanoutEmit(benchmark::State& state) {
  const auto fanout = static_cast<std::uint32_t>(state.range(0));
  auto context = create_typed_pod_fanout_context(fanout);
  if (!context) {
    skip_with_error(state, "failed to create typed Channel fanout emit fixture", context.error());
    return;
  }

  std::uint64_t generation = 2U;
  for (auto _ : state) {
    const auto value = static_cast<std::int32_t>(generation);
    auto prepared = context->fixture->prepare_typed_pod_source(generation);
    if (!prepared) {
      skip_with_error(state, "failed to prepare typed Channel fanout emit", prepared.error());
      return;
    }

    const auto start = std::chrono::steady_clock::now();
    auto emitted = context->fixture->emit_typed_pod(value);
    const auto finish = std::chrono::steady_clock::now();
    if (!emitted) {
      skip_with_error(state, "failed to emit typed Channel fanout POD", emitted.error());
      return;
    }
    state.SetIterationTime(std::chrono::duration<double>{finish - start}.count());

    auto retired = context->fixture->retire_typed_pod_source(generation);
    if (!retired) {
      skip_with_error(state, "failed to retire typed Channel fanout source", retired.error());
      return;
    }
    auto flushed = context->fixture->flush_process_local_cut();
    if (!flushed) {
      skip_with_error(state, "failed to flush typed Channel fanout emit", flushed.error());
      return;
    }
    auto consumed = consume_typed_pod_fanout(*context, generation, value);
    if (!consumed) {
      skip_with_error(state, "failed to clean up typed Channel fanout emit", consumed.error());
      return;
    }
    ++generation;
  }
  state.counters["fanout"] = static_cast<double>(fanout);
  state.counters["payload_bytes"] = static_cast<double>(sizeof(std::int32_t));
  state.SetItemsProcessed(state.iterations());
}

// Time producer activation, fanout publication, retirement, and the post-source cut that makes
// every sink visible to Runtime typed storage. Sink consumption remains outside the interval.
// REQ: RUNTIME-016
// REQ: RUNTIME-018
void BM_ChannelTypedPODFanoutPublishCut(benchmark::State& state) {
  const auto fanout = static_cast<std::uint32_t>(state.range(0));
  auto context = create_typed_pod_fanout_context(fanout);
  if (!context) {
    skip_with_error(
        state, "failed to create typed Channel fanout publish-cut fixture", context.error());
    return;
  }

  std::uint64_t generation = 2U;
  for (auto _ : state) {
    const auto value = static_cast<std::int32_t>(generation);
    const auto start = std::chrono::steady_clock::now();
    auto published = context->fixture->publish_typed_pod(generation, value);
    const auto finish = std::chrono::steady_clock::now();
    if (!published) {
      skip_with_error(
          state, "failed to publish typed Channel fanout before cut", published.error());
      return;
    }
    state.SetIterationTime(std::chrono::duration<double>{finish - start}.count());

    auto consumed = consume_typed_pod_fanout(*context, generation, value);
    if (!consumed) {
      skip_with_error(
          state, "failed to clean up typed Channel fanout publish-cut row", consumed.error());
      return;
    }
    ++generation;
  }
  state.counters["activation_cuts_per_cycle"] = 1.0;
  state.counters["fanout"] = static_cast<double>(fanout);
  state.counters["payload_bytes"] = static_cast<double>(sizeof(std::int32_t));
  state.SetItemsProcessed(state.iterations());
}

// Time all sink readiness, receive, witness, sample, activation, and post-callback retirement
// transactions after one fanout publication has reached Runtime typed storage.
// REQ: RUNTIME-016
// REQ: RUNTIME-018
void BM_ChannelTypedPODFanoutConsumeRetire(benchmark::State& state) {
  const auto fanout = static_cast<std::uint32_t>(state.range(0));
  auto context = create_typed_pod_fanout_context(fanout);
  if (!context) {
    skip_with_error(
        state, "failed to create typed Channel fanout consume fixture", context.error());
    return;
  }

  std::uint64_t generation = 2U;
  for (auto _ : state) {
    const auto value = static_cast<std::int32_t>(generation);
    auto published = context->fixture->publish_typed_pod(generation, value);
    if (!published) {
      skip_with_error(
          state, "failed to prepare typed Channel fanout consumption", published.error());
      return;
    }

    const auto start = std::chrono::steady_clock::now();
    auto consumed = consume_typed_pod_fanout(*context, generation, value);
    const auto finish = std::chrono::steady_clock::now();
    if (!consumed) {
      skip_with_error(state, "failed to consume typed Channel fanout POD", consumed.error());
      return;
    }
    state.SetIterationTime(std::chrono::duration<double>{finish - start}.count());
    ++generation;
  }
  state.counters["fanout"] = static_cast<double>(fanout);
  state.counters["payload_bytes"] = static_cast<double>(sizeof(std::int32_t));
  state.counters["readiness_polls_per_delivery"] =
      static_cast<double>(context->fixture->readiness_poll_count()) /
      static_cast<double>(state.iterations() * fanout);
  state.SetItemsProcessed(state.iterations() * fanout);
}

// Bound an optimistic continuation by combining each sink's readiness poll and consume/retire
// transaction in one busy-lane task. Production global candidate ordering remains excluded.
// REQ: RUNTIME-011
// REQ: RUNTIME-016
// REQ: RUNTIME-018
// REQ: RUNTIME-026
void BM_ChannelTypedPODFanoutFusedPollConsumeCeiling(benchmark::State& state) {
  const auto fanout = static_cast<std::uint32_t>(state.range(0));
  auto context = create_typed_pod_fanout_context(fanout);
  if (!context) {
    skip_with_error(state, "failed to create typed Channel fanout fused fixture", context.error());
    return;
  }
  auto lane_set = create_started_busy_lane_set();
  if (!lane_set) {
    skip_with_error(state, "failed to create Runtime fanout lane", lane_set.error());
    return;
  }

  std::uint64_t generation = 2U;
  for (auto _ : state) {
    state.PauseTiming();
    const auto value = static_cast<std::int32_t>(generation);
    auto published = context->fixture->publish_typed_pod(generation, value);
    state.ResumeTiming();
    if (!published) {
      skip_with_error(state, "failed to prepare typed Channel fused fanout", published.error());
      return;
    }

    for (std::size_t index = 0U; index < context->bindings.size(); ++index) {
      holoscan::detail::PortBinding& binding = *context->bindings[index];
      const std::optional<holoscan::detail::PortBindingPollTarget> target = binding.poll_target();
      if (!target) {
        state.SkipWithError("fanout readiness did not advertise a poll target");
        return;
      }
      TypedPODFanoutConsumeInvocation consume{
          .fixture = context->fixture.get(),
          .binding = &binding,
          .generation = generation,
          .sink_index = index,
      };
      auto consumed = (*lane_set)->dispatch(
          target->execution_lane,
          holoscan::detail::ExecutionLaneTask{
              .context = &consume,
              .invoke = invoke_fanout_poll_consume,
              .callback_context = holoscan::detail::RuntimeCallbackContext::kCompute,
          });
      if (!consumed) {
        skip_with_error(state, "failed to run fused typed Channel fanout sink", consumed.error());
        return;
      }
      if (consume.received != value) {
        state.SkipWithError("fused fanout consumption returned the wrong value");
        return;
      }
    }
    ++generation;
  }
  state.counters["fanout"] = static_cast<double>(fanout);
  state.counters["fused_poll_consume_tasks_per_cycle"] = static_cast<double>(fanout);
  state.SetItemsProcessed(state.iterations() * fanout);

  auto joined = (*lane_set)->join();
  if (!joined) {
    state.SkipWithError("failed to stop Runtime fanout lane");
  }
}

// Compose two production-shaped busy-lane handoffs per sink around the already measured Channel
// consume transaction. Global candidate selection and controller-owned activation work remain
// outside this component boundary.
// REQ: RUNTIME-011
// REQ: RUNTIME-016
// REQ: RUNTIME-018
// REQ: RUNTIME-026
void BM_ChannelTypedPODFanoutTwoDispatchConsumeRetire(benchmark::State& state) {
  const auto fanout = static_cast<std::uint32_t>(state.range(0));
  auto context = create_typed_pod_fanout_context(fanout);
  if (!context) {
    skip_with_error(
        state, "failed to create typed Channel fanout dispatch fixture", context.error());
    return;
  }
  auto lane_set = create_started_busy_lane_set();
  if (!lane_set) {
    skip_with_error(state, "failed to create Runtime fanout lane", lane_set.error());
    return;
  }

  std::uint64_t generation = 2U;
  for (auto _ : state) {
    state.PauseTiming();
    const auto value = static_cast<std::int32_t>(generation);
    auto published = context->fixture->publish_typed_pod(generation, value);
    state.ResumeTiming();
    if (!published) {
      skip_with_error(state, "failed to prepare typed Channel fanout dispatch", published.error());
      return;
    }

    for (std::size_t index = 0U; index < context->bindings.size(); ++index) {
      holoscan::detail::PortBinding& binding = *context->bindings[index];
      const std::optional<holoscan::detail::PortBindingPollTarget> target = binding.poll_target();
      if (!target) {
        state.SkipWithError("fanout readiness did not advertise a poll target");
        return;
      }
      ReadinessPollInvocation poll{
          .binding = &binding,
          .observed_at =
              holoscan::TimePoint{static_cast<std::int64_t>(generation), holoscan::ClockId{1U}},
      };
      auto polled = (*lane_set)->dispatch(
          target->execution_lane,
          holoscan::detail::ExecutionLaneTask{
              .context = &poll,
              .invoke = invoke_readiness_poll,
              .callback_context = holoscan::detail::RuntimeCallbackContext::kDispatcher,
          });
      if (!polled) {
        skip_with_error(state, "failed to poll typed Channel fanout sink", polled.error());
        return;
      }

      TypedPODFanoutConsumeInvocation consume{
          .fixture = context->fixture.get(),
          .binding = &binding,
          .generation = generation,
          .sink_index = index,
      };
      auto consumed = (*lane_set)->dispatch(
          target->execution_lane,
          holoscan::detail::ExecutionLaneTask{
              .context = &consume,
              .invoke = invoke_fanout_consume_after_poll,
              .callback_context = holoscan::detail::RuntimeCallbackContext::kCompute,
          });
      if (!consumed) {
        skip_with_error(state, "failed to consume typed Channel fanout sink", consumed.error());
        return;
      }
      if (consume.received != value) {
        state.SkipWithError("fanout dispatch consumption returned the wrong value");
        return;
      }
    }
    ++generation;
  }
  state.counters["compute_lane_dispatches_per_cycle"] = static_cast<double>(fanout);
  state.counters["fanout"] = static_cast<double>(fanout);
  state.counters["poll_lane_dispatches_per_cycle"] = static_cast<double>(fanout);
  state.SetItemsProcessed(state.iterations() * fanout);

  auto joined = (*lane_set)->join();
  if (!joined) {
    state.SkipWithError("failed to stop Runtime fanout lane");
  }
}

// Bound the public Output facade and its two prebound dispatch layers without Runtime admission,
// fanout, HoloMQ loan/send, or event notification work.
// REQ: OPERATOR-014
void BM_OutputTypedPODNoopBinding(benchmark::State& state) {
  NoopOutputBindingState sink;
  holoscan::detail::RuntimeOutputBinding runtime_binding{
      .state = &sink,
      .ops = &NoopOutputBindingState::kOps,
  };
  holoscan::Output<std::int32_t> output{"output"};
  holoscan::detail::RuntimePortBindingAccess::bind_output(output, &runtime_binding);

  constexpr std::int32_t kValue = 7;
  for (auto _ : state) {
    auto emitted = output.emit(kValue);
    if (!emitted) {
      skip_with_error(state, "failed to emit through no-op Output binding", emitted.error());
      return;
    }
  }

  holoscan::detail::RuntimePortBindingAccess::unbind_output(output, &runtime_binding);
  const auto expected_count = static_cast<std::uint64_t>(state.iterations());
  if (sink.accepted != expected_count || sink.checksum != expected_count * kValue) {
    state.SkipWithError("no-op Output binding observed the wrong payload count or checksum");
    return;
  }
  state.counters["payload_bytes"] = static_cast<double>(sizeof(std::int32_t));
  state.SetItemsProcessed(state.iterations());
}

// Time the exact event-enabled private inline Channel publication used below Runtime OutputState.
// The companion wait drains the advisory event and data queue outside the manual interval.
// REQ: CHAN-066
// REQ: RUNTIME-016
void BM_ChannelInlineTypedPODPublish(benchmark::State& state) {
  auto context = create_inline_publish_context();
  if (!context) {
    skip_with_error(state, "failed to create inline Channel publish fixture", context.error());
    return;
  }
  std::int32_t value = 7;
  auto payload = std::as_bytes(std::span<const std::int32_t, 1U>{&value, 1U});
  std::int32_t received_value{};
  auto destination = std::as_writable_bytes(std::span<std::int32_t, 1U>{&received_value, 1U});
  holoscan::SampleMetadata metadata{
      .byte_size = sizeof(std::int32_t),
      .payload_kind = holoscan::PayloadKind::kInlineTyped,
  };

  for (auto _ : state) {
    ++metadata.sequence;
    const auto start = std::chrono::steady_clock::now();
    auto published = context->publisher->emit_phase(payload, metadata);
    const auto finish = std::chrono::steady_clock::now();
    if (published.phase != holoscan::channel_internal::InlinePublicationPhase::kAccepted ||
        published.recipient_count != 1U) {
      skip_with_error(state, "failed to publish inline Channel POD", published.error);
      return;
    }
    state.SetIterationTime(std::chrono::duration<double>{finish - start}.count());

    auto drained = drain_inline_publish(*context, destination);
    if (!drained) {
      skip_with_error(state, "failed to drain inline Channel POD", drained.error());
      return;
    }
    if (received_value != value) {
      state.SkipWithError("inline Channel publish returned the wrong payload");
      return;
    }
  }
  state.counters["event_wake"] = 1.0;
  state.counters["payload_bytes"] = static_cast<double>(sizeof(std::int32_t));
  state.SetItemsProcessed(state.iterations());
}

// Bound data publication without the parallel event service. Polling cleanup remains outside the
// timed interval; this row is diagnostic evidence, not a Runtime synchronization proposal.
// REQ: CHAN-066
void BM_ChannelInlineTypedPODPublishNoEvent(benchmark::State& state) {
  auto context = create_inline_publish_context(false);
  if (!context) {
    skip_with_error(state, "failed to create no-event inline Channel fixture", context.error());
    return;
  }
  std::int32_t value = 7;
  const auto payload = std::as_bytes(std::span<const std::int32_t, 1U>{&value, 1U});
  std::int32_t received_value{};
  auto destination = std::as_writable_bytes(std::span<std::int32_t, 1U>{&received_value, 1U});
  holoscan::SampleMetadata metadata{
      .byte_size = sizeof(std::int32_t),
      .payload_kind = holoscan::PayloadKind::kInlineTyped,
  };
  metadata.sequence = 1U;
  auto warmed = warm_inline_publish_polling(*context, payload, destination, metadata);
  if (!warmed) {
    skip_with_error(state, "failed to warm no-event inline Channel POD", warmed.error());
    return;
  }

  for (auto _ : state) {
    ++metadata.sequence;
    const auto start = std::chrono::steady_clock::now();
    auto published = context->publisher->emit_phase(payload, metadata);
    const auto finish = std::chrono::steady_clock::now();
    if (published.phase != holoscan::channel_internal::InlinePublicationPhase::kAccepted ||
        published.recipient_count != 1U) {
      skip_with_error(state, "failed to publish no-event inline Channel POD", published.error);
      return;
    }
    state.SetIterationTime(std::chrono::duration<double>{finish - start}.count());

    auto drained = drain_inline_publish_polling(*context, destination);
    if (!drained) {
      skip_with_error(state, "failed to poll no-event inline Channel POD", drained.error());
      return;
    }
    if (received_value != value) {
      state.SkipWithError("no-event inline Channel publish returned the wrong payload");
      return;
    }
  }
  state.counters["event_wake"] = 0.0;
  state.counters["payload_bytes"] = static_cast<double>(sizeof(std::int32_t));
  state.SetItemsProcessed(state.iterations());
}

// Keep a no-event consumer actively polling on another thread. This bounds userspace concurrent
// queue access but intentionally consumes one support-thread CPU while the benchmark is running.
// REQ: CHAN-066
void BM_ChannelInlineTypedPODPublishActivePolling(benchmark::State& state) {
  auto context = create_inline_publish_context(false);
  if (!context) {
    skip_with_error(state, "failed to create polling inline Channel fixture", context.error());
    return;
  }
  std::int32_t value = 7;
  const auto payload = std::as_bytes(std::span<const std::int32_t, 1U>{&value, 1U});
  std::int32_t warm_value{};
  auto warm_destination = std::as_writable_bytes(std::span<std::int32_t, 1U>{&warm_value, 1U});
  holoscan::SampleMetadata metadata{
      .sequence = 1U,
      .byte_size = sizeof(std::int32_t),
      .payload_kind = holoscan::PayloadKind::kInlineTyped,
  };
  auto warmed = warm_inline_publish_polling(*context, payload, warm_destination, metadata);
  if (!warmed) {
    skip_with_error(state, "failed to warm polling inline Channel POD", warmed.error());
    return;
  }
  const PollingInlineConsumer consumer{*context, 2U};

  for (auto _ : state) {
    ++metadata.sequence;
    if (!consumer.wait_until_polling(metadata.sequence)) {
      skip_with_error(state,
                      "inline Channel consumer did not begin polling",
                      holoscan::Error{consumer.error_code() == holoscan::ErrorCode::kSuccess
                                          ? holoscan::ErrorCode::kNotReady
                                          : consumer.error_code()});
      return;
    }

    const auto start = std::chrono::steady_clock::now();
    auto published = context->publisher->emit_phase(payload, metadata);
    const auto finish = std::chrono::steady_clock::now();
    if (published.phase != holoscan::channel_internal::InlinePublicationPhase::kAccepted ||
        published.recipient_count != 1U) {
      skip_with_error(state, "failed to publish to polling Channel consumer", published.error);
      return;
    }
    state.SetIterationTime(std::chrono::duration<double>{finish - start}.count());

    if (!consumer.wait_until_consumed(metadata.sequence)) {
      skip_with_error(state,
                      "polling inline Channel consumer did not drain the sample",
                      holoscan::Error{consumer.error_code() == holoscan::ErrorCode::kSuccess
                                          ? holoscan::ErrorCode::kNotReady
                                          : consumer.error_code()});
      return;
    }
  }
  state.counters["active_polling"] = 1.0;
  state.counters["event_wake"] = 0.0;
  state.counters["payload_bytes"] = static_cast<double>(sizeof(std::int32_t));
  state.SetItemsProcessed(state.iterations());
}

// Run one event-enabled publisher against a dedicated WaitSet consumer. The wrappers choose
// whether the consumer may take data before the timed publisher call returns.
// REQ: CHAN-066
// REQ: RUNTIME-016
void run_inline_publish_async_waiter(benchmark::State& state, bool defer_receive) {
  auto context = create_inline_publish_context();
  if (!context) {
    skip_with_error(state, "failed to create asynchronous inline Channel fixture", context.error());
    return;
  }
  AsyncInlineConsumer consumer{*context, defer_receive};
  std::int32_t value = 7;
  const auto payload = std::as_bytes(std::span<const std::int32_t, 1U>{&value, 1U});
  holoscan::SampleMetadata metadata{
      .byte_size = sizeof(std::int32_t),
      .payload_kind = holoscan::PayloadKind::kInlineTyped,
  };

  for (auto _ : state) {
    ++metadata.sequence;
    if (!consumer.wait_until_waiting(metadata.sequence)) {
      skip_with_error(state,
                      "asynchronous inline Channel consumer did not enter its wait",
                      holoscan::Error{consumer.error_code() == holoscan::ErrorCode::kSuccess
                                          ? holoscan::ErrorCode::kNotReady
                                          : consumer.error_code()});
      return;
    }
    std::this_thread::yield();

    const auto start = std::chrono::steady_clock::now();
    auto published = context->publisher->emit_phase(payload, metadata);
    const auto finish = std::chrono::steady_clock::now();
    if (published.phase != holoscan::channel_internal::InlinePublicationPhase::kAccepted ||
        published.recipient_count != 1U) {
      skip_with_error(state, "failed to publish to asynchronous Channel consumer", published.error);
      return;
    }
    state.SetIterationTime(std::chrono::duration<double>{finish - start}.count());

    if (defer_receive) {
      if (!consumer.wait_until_woken(metadata.sequence)) {
        skip_with_error(state,
                        "asynchronous inline Channel consumer did not observe the wake",
                        holoscan::Error{consumer.error_code() == holoscan::ErrorCode::kSuccess
                                            ? holoscan::ErrorCode::kNotReady
                                            : consumer.error_code()});
        return;
      }
      consumer.release_receive(metadata.sequence);
    }
    if (!consumer.wait_until_consumed(metadata.sequence)) {
      skip_with_error(state,
                      "asynchronous inline Channel consumer did not drain the sample",
                      holoscan::Error{consumer.error_code() == holoscan::ErrorCode::kSuccess
                                          ? holoscan::ErrorCode::kNotReady
                                          : consumer.error_code()});
      return;
    }
  }
  state.counters["async_waiter"] = 1.0;
  state.counters["concurrent_receive"] = defer_receive ? 0.0 : 1.0;
  state.counters["event_wake"] = 1.0;
  state.counters["payload_bytes"] = static_cast<double>(sizeof(std::int32_t));
  state.SetItemsProcessed(state.iterations());
}

void BM_ChannelInlineTypedPODPublishDeferredReceive(benchmark::State& state) {
  // Let the listener wake concurrently, but defer the subscriber data take until after publish.
  run_inline_publish_async_waiter(state, true);
}

void BM_ChannelInlineTypedPODPublishAsyncWaiter(benchmark::State& state) {
  // Match Runtime's pump by allowing the awakened consumer to take while publish is in flight.
  run_inline_publish_async_waiter(state, false);
}

// Time only the post-source process-local cut. The asynchronous pump remains active, so the cut
// may either join an already-completed scan or perform the transport take and typed-store write.
// REQ: RUNTIME-016
// REQ: RUNTIME-018
void BM_ChannelTypedPODSourceCut(benchmark::State& state) {
  auto context = create_typed_pod_cycle_context();
  if (!context) {
    skip_with_error(state, "failed to create typed Channel source-cut fixture", context.error());
    return;
  }

  std::uint64_t generation = 2U;
  for (auto _ : state) {
    const auto value = static_cast<std::int32_t>(generation);
    auto published = context->fixture->publish_typed_pod_before_cut(generation, value);
    if (!published) {
      skip_with_error(state, "failed to prepare typed Channel source cut", published.error());
      return;
    }

    const auto start = std::chrono::steady_clock::now();
    auto flushed = context->fixture->flush_process_local_cut();
    const auto finish = std::chrono::steady_clock::now();
    if (!flushed) {
      skip_with_error(state, "failed to run typed Channel source cut", flushed.error());
      return;
    }
    state.SetIterationTime(std::chrono::duration<double>{finish - start}.count());

    auto received = context->fixture->consume_typed_pod(*context->binding, generation);
    if (!received || *received != value) {
      if (!received) {
        skip_with_error(state, "failed to clean up typed Channel source-cut row", received.error());
      } else {
        state.SkipWithError("typed Channel source-cut cleanup returned the wrong value");
      }
      return;
    }
    ++generation;
  }
  state.counters["activation_cuts_per_cycle"] = 1.0;
  state.counters["payload_bytes"] = static_cast<double>(sizeof(std::int32_t));
  state.SetItemsProcessed(state.iterations());
}

// Time readiness through typed receive and complete sink-side claim retirement. Producer setup and
// publication run outside the manual interval so the row starts after the post-source cut.
// REQ: RUNTIME-016
// REQ: RUNTIME-018
void BM_ChannelTypedPODConsumeRetire(benchmark::State& state) {
  auto context = create_typed_pod_cycle_context();
  if (!context) {
    skip_with_error(state, "failed to create typed Channel consume fixture", context.error());
    return;
  }

  std::uint64_t generation = 2U;
  for (auto _ : state) {
    const auto value = static_cast<std::int32_t>(generation);
    auto published = context->fixture->publish_typed_pod(generation, value);
    if (!published) {
      skip_with_error(state, "failed to prepare typed Channel consumption", published.error());
      return;
    }

    const auto start = std::chrono::steady_clock::now();
    auto received = context->fixture->consume_typed_pod(*context->binding, generation);
    const auto finish = std::chrono::steady_clock::now();
    if (!received || *received != value) {
      if (!received) {
        skip_with_error(state, "failed to consume typed Channel POD", received.error());
      } else {
        state.SkipWithError("typed Channel consumption returned the wrong value");
      }
      return;
    }
    state.SetIterationTime(std::chrono::duration<double>{finish - start}.count());
    ++generation;
  }
  state.counters["activation_cuts_per_cycle"] = 1.0;
  state.counters["payload_bytes"] = static_cast<double>(sizeof(std::int32_t));
  state.counters["readiness_polls_per_cycle"] =
      static_cast<double>(context->fixture->readiness_poll_count()) /
      static_cast<double>(state.iterations());
  state.SetItemsProcessed(state.iterations());
}

// Quantify the two steady-clock reads used by each manual phase row.
void BM_ChannelPhaseClockBracket(benchmark::State& state) {
  for (auto _ : state) {
    const auto start = std::chrono::steady_clock::now();
    const auto finish = std::chrono::steady_clock::now();
    state.SetIterationTime(std::chrono::duration<double>{finish - start}.count());
  }
}

BENCHMARK(BM_ChannelActivationCycleByGraphOwners)->Arg(2)->Arg(8)->Arg(16);
BENCHMARK(BM_ChannelReadinessBurst)
    ->Args({1, 1})
    ->Args({1, 8})
    ->Args({8, 1})
    ->Args({8, 8})
    ->Args({16, 64});
BENCHMARK(BM_ChannelReadinessPollViaBusyLane)->UseRealTime();
BENCHMARK(BM_ChannelDirectRetainedReadinessPoll)->UseRealTime();
BENCHMARK(BM_ChannelDirectRetainedReadinessPollViaBusyLane)->UseRealTime();
BENCHMARK(BM_ChannelDirectRetainedFusedPollConsumeCeiling)->UseRealTime();
BENCHMARK(BM_ChannelDirectRetainedTwoDispatchPollConsume)->UseRealTime();
BENCHMARK(BM_ChannelDirectRetainedResumablePollConsumeProbe)->UseRealTime();
BENCHMARK_CAPTURE(BM_ChannelDirectRetainedActivationCeiling, ordinary_sleeping,
                  DirectRetainedActivationCeilingMode::kOrdinary,
                  holoscan::WakeMode::kSleepingWaitset)
    ->Apply(apply_activation_ceiling_arguments)
    ->UseManualTime();
BENCHMARK_CAPTURE(BM_ChannelDirectRetainedActivationCeiling, receipt_sleeping,
                  DirectRetainedActivationCeilingMode::kReceiptElided,
                  holoscan::WakeMode::kSleepingWaitset)
    ->Apply(apply_activation_ceiling_arguments)
    ->UseManualTime();
BENCHMARK_CAPTURE(BM_ChannelDirectRetainedActivationCeiling, composite_sleeping,
                  DirectRetainedActivationCeilingMode::kComposite,
                  holoscan::WakeMode::kSleepingWaitset)
    ->Apply(apply_activation_ceiling_arguments)
    ->UseManualTime();
BENCHMARK_CAPTURE(BM_ChannelDirectRetainedActivationCeiling, ordinary_busy,
                  DirectRetainedActivationCeilingMode::kOrdinary, holoscan::WakeMode::kBusyPoll)
    ->Apply(apply_activation_ceiling_arguments)
    ->UseManualTime();
BENCHMARK_CAPTURE(BM_ChannelDirectRetainedActivationCeiling, receipt_busy,
                  DirectRetainedActivationCeilingMode::kReceiptElided,
                  holoscan::WakeMode::kBusyPoll)
    ->Apply(apply_activation_ceiling_arguments)
    ->UseManualTime();
BENCHMARK_CAPTURE(BM_ChannelDirectRetainedActivationCeiling, composite_busy,
                  DirectRetainedActivationCeilingMode::kComposite, holoscan::WakeMode::kBusyPoll)
    ->Apply(apply_activation_ceiling_arguments)
    ->UseManualTime();
BENCHMARK(BM_ChannelLastWitnessRetirementByGraphInputs)
    ->Arg(1)
    ->Arg(2)
    ->Arg(3)
    ->Arg(8)
    ->Arg(16)
    ->Arg(64);
BENCHMARK(BM_ChannelTypedPODCycle)->UseRealTime();
BENCHMARK(BM_ChannelTypedPODDirectRetainedCycle)->UseRealTime();
BENCHMARK(BM_ChannelTypedPOD64DirectRetainedCycle)->UseRealTime();
BENCHMARK(BM_ChannelTypedPODPublishCut)->UseManualTime();
BENCHMARK(BM_ChannelTypedPODPublishRetire)->UseManualTime();
BENCHMARK(BM_ChannelTypedPODEmit)->UseManualTime();
BENCHMARK(BM_ChannelTypedPODDirectRetainedEmit)->UseManualTime();
BENCHMARK(BM_ChannelTypedPODFanoutEmit)->Arg(1)->Arg(2)->Arg(4)->UseManualTime();
BENCHMARK(BM_ChannelTypedPODFanoutPublishCut)->Arg(1)->Arg(2)->Arg(4)->UseManualTime();
BENCHMARK(BM_ChannelTypedPODFanoutConsumeRetire)->Arg(1)->Arg(2)->Arg(4)->UseManualTime();
BENCHMARK(BM_ChannelTypedPODFanoutFusedPollConsumeCeiling)->Arg(1)->Arg(2)->Arg(4)->UseRealTime();
BENCHMARK(BM_ChannelTypedPODFanoutTwoDispatchConsumeRetire)->Arg(1)->Arg(2)->Arg(4)->UseRealTime();
BENCHMARK(BM_OutputTypedPODNoopBinding);
BENCHMARK(BM_ChannelInlineTypedPODPublish)->UseManualTime();
BENCHMARK(BM_ChannelInlineTypedPODPublishNoEvent)->UseManualTime();
BENCHMARK(BM_ChannelInlineTypedPODPublishActivePolling)->UseManualTime();
BENCHMARK(BM_ChannelInlineTypedPODPublishDeferredReceive)->UseManualTime();
BENCHMARK(BM_ChannelInlineTypedPODPublishAsyncWaiter)->UseManualTime();
BENCHMARK(BM_ChannelTypedPODSourceCut)->UseManualTime();
BENCHMARK(BM_ChannelTypedPODConsumeRetire)->UseManualTime();
BENCHMARK(BM_ChannelPhaseClockBracket)->UseManualTime();

}  // namespace
