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
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator_spec.hpp>

#include "channel/process_identity.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "core/operator_spec_compiler_access.hpp"
#include "core/operator_test_binding.hpp"
#include "core/source_id_internal.hpp"
#include "runtime/runtime_channel_binding.hpp"

#include "runtime_channel_binding_test_support.hpp"

namespace holoscan::detail::runtime_channel_binding_test {
namespace {

using namespace std::chrono_literals;

[[nodiscard]] std::uint64_t unique_plan_generation() noexcept {
  static std::atomic<std::uint64_t> sequence{1U};
  // CTest launches individual cases in separate processes. Partition the identity by PID so their
  // compiler-derived HoloMQ service names cannot collide under parallel execution.
  const std::uint64_t process = channel_internal::current_process_identity();
  const std::uint64_t ordinal = sequence.fetch_add(1U, std::memory_order_relaxed);
  return (process << 32U) | (ordinal & std::numeric_limits<std::uint32_t>::max());
}

}  // namespace

ScriptedInlinePublisher::ScriptedInlinePublisher(
    std::vector<channel_internal::InlinePublicationResult> results,
    std::vector<ErrorCode> refresh_results, std::vector<ErrorCode> close_results,
    const bool throw_on_close)
    : results_(std::move(results)),
      refresh_results_(std::move(refresh_results)),
      close_results_(std::move(close_results)),
      throw_on_close_(throw_on_close) {}

channel_internal::InlinePublicationResult ScriptedInlinePublisher::emit_phase(
    const std::span<const std::byte> payload, const SampleMetadata& metadata) {
  payloads.emplace_back(payload.begin(), payload.end());
  metadata_records.push_back(metadata);
  sequences.push_back(metadata.sequence);
  std::int32_t value{};
  if (payload.size() == sizeof(value)) {
    std::memcpy(&value, payload.data(), sizeof(value));
  }
  values.push_back(value);
  if (next_result_ < results_.size()) {
    return results_[next_result_++];
  }
  return channel_internal::InlinePublicationResult{
      .phase = channel_internal::InlinePublicationPhase::kAccepted,
      .recipient_count = 1U,
      .error = Error{ErrorCode::kSuccess},
  };
}

expected<void, Error> ScriptedInlinePublisher::close_stream() {
  ++close_count;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  if (throw_on_close_) {
    throw std::runtime_error{"injected terminal close failure"};
  }
#endif
  if (next_close_result_ == close_results_.size()) {
    return {};
  }
  const ErrorCode result = close_results_[next_close_result_++];
  return result == ErrorCode::kSuccess ? expected<void, Error>{}
                                       : expected<void, Error>{make_unexpected(Error{result})};
}

expected<void, Error> ScriptedInlinePublisher::update_connections() noexcept {
  if (next_refresh_result_ == refresh_results_.size()) {
    return {};
  }
  const ErrorCode result = refresh_results_[next_refresh_result_++];
  return result == ErrorCode::kSuccess ? expected<void, Error>{}
                                       : expected<void, Error>{make_unexpected(Error{result})};
}

bool ScriptedInlinePublisher::has_subscribers() const noexcept {
  return true;
}

ScriptedInlineSubscriber::ScriptedInlineSubscriber(const std::int32_t value,
                                                   const std::uint64_t sequence,
                                                   const InlineReceivePath path)
    : metadata_{
          .sequence = sequence,
          .byte_size = sizeof(value),
          .payload_kind = PayloadKind::kInlineTyped,
      },
      path_(path) {
  std::memcpy(payload_.data(), &value, sizeof(value));
}

expected<bool, Error> ScriptedInlineSubscriber::has_samples() const {
  return true;
}

bool ScriptedInlineSubscriber::event_wake_enabled() const noexcept {
  return false;
}

void* ScriptedInlineSubscriber::wait_context_runtime() noexcept {
  return nullptr;
}

bool ScriptedInlineSubscriber::has_publishers() const noexcept {
  return true;
}

expected<channel_internal::InlinePayloadReceiveResult, Error> ScriptedInlineSubscriber::receive(
    const std::span<std::byte> payload) {
  ++copy_receive_count;
  if (path_ != InlineReceivePath::kCopyFallback) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  if (data_delivered_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  if (payload.size() != payload_.size()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  std::memcpy(payload.data(), payload_.data(), payload_.size());
  data_delivered_ = true;
  return channel_internal::InlinePayloadReceiveResult{.metadata = metadata_,
                                                      .payload_size = payload_.size()};
}

expected<channel_internal::InlinePayloadReceiveGuard, Error>
ScriptedInlineSubscriber::receive_borrowed() {
  ++borrowed_receive_count;
  if (path_ == InlineReceivePath::kCopyFallback) {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }
  if (loan_active_) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  if (data_delivered_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  data_delivered_ = true;
  loan_active_ = true;
  return channel_internal::InlinePayloadReceiveGuard{
      *this, std::span<const std::byte>{payload_}, metadata_};
}

expected<channel_internal::InlinePayloadReceiveGuard, Error>
ScriptedInlineSubscriber::receive_borrowed_if_data_accepted(const bool accept_data) {
  // Model the bounded endpoint transaction: a data head stays pending until Runtime has a typed
  // destination, while the stable terminal behind an already-consumed data head remains visible.
  if (data_delivered_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  if (!accept_data) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  return receive_borrowed();
}

void ScriptedInlineSubscriber::release_payload_loan() noexcept {
  if (loan_active_) {
    loan_active_ = false;
    ++loan_release_count;
  }
}

expected<std::uint64_t, Error> ScriptedInlineSubscriber::attach_wait_set(void*) {
  return make_unexpected(Error{ErrorCode::kNotSupported});
}

expected<void, Error> ScriptedInlineSubscriber::detach_wait_set() noexcept {
  return {};
}

std::optional<channel_internal::StagedDataState> ScriptedInlineSubscriber::staged_data_state()
    const noexcept {
  ++staged_data_query_count;
  return std::nullopt;
}

std::optional<std::uint64_t> ScriptedInlineSubscriber::staged_terminal_sequence() const noexcept {
  return std::nullopt;
}

[[nodiscard]] std::string unique_service() {
  static std::atomic<std::uint64_t> sequence{1U};
  const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
  return "holoscan-runtime-channel-" +
         std::to_string(channel_internal::current_process_identity()) + "-" +
         std::to_string(timestamp) + "-" +
         std::to_string(sequence.fetch_add(1U, std::memory_order_relaxed));
}

SessionFixture::SessionFixture() {
  source.setup(source_spec);
  sink.setup(sink_spec);
  second_sink.setup(second_sink_spec);
  const auto output = OperatorSpecCompilerAccess::port(source_spec, 0U);
  const auto input = OperatorSpecCompilerAccess::port(sink_spec, 0U);
  const auto second_input = OperatorSpecCompilerAccess::port(second_sink_spec, 0U);
  plan.admitted = true;
  plan.plan_generation = unique_plan_generation();
  const std::string service = "process-local-channel." + std::to_string(plan.plan_generation) +
                              ".source.1.policy." +
                              std::to_string(static_cast<std::uint32_t>(QueuePolicy::kBlocking));
  plan.ports = {
      PortPlanRecord{
          .id = PortId{1U},
          .owner = VertexId{1U},
          .direction = PortDirection::kOutput,
          .local_name = "output",
          .payload_identity = output.payload_identity,
          .max_emits_per_compute = 8U,
          .effective_profile = Profile::kStandard,
      },
      PortPlanRecord{
          .id = PortId{2U},
          .owner = VertexId{2U},
          .direction = PortDirection::kInput,
          .local_name = "input",
          .payload_identity = input.payload_identity,
          .queue_depth = 8U,
          .effective_profile = Profile::kStandard,
      },
  };
  plan.process_local_channel_services = {
      ProcessLocalChannelServicePlanRecord{
          .source = PortId{1U},
          .service_id = service,
          .payload_identity = output.payload_identity,
          .payload_size_bytes = sizeof(std::int32_t),
          .queue_policy = QueuePolicy::kBlocking,
          .max_subscribers = 1U,
          .max_subscriber_queue_depth = 8U,
          .max_subscriber_borrowed_samples = 8U,
          .publisher_history_depth = 0U,
          .event_wake = true,
      },
  };
  plan.process_local_channel_publishers = {
      ProcessLocalChannelPublisherPlanRecord{
          .source = PortId{1U},
          .service_id = service,
          .max_loaned_samples = 8U,
      },
  };
  plan.process_local_channel_subscribers = {
      ProcessLocalChannelSubscriberPlanRecord{
          .edge = EdgeId{1U},
          .source = PortId{1U},
          .target = PortId{2U},
          .service_id = service,
          .requested_queue_depth = 8U,
          .max_borrowed_samples = 8U,
          .cache_capacity = 8U,
          .readiness_capacity = 8U,
          .event_wake = true,
      },
  };
  plan.payload_edges = {
      PayloadEdgePlanRecord{
          .edge = EdgeId{1U},
          .source = PortId{1U},
          .target = PortId{2U},
          .payload_identity = output.payload_identity,
          .queue_policy = QueuePolicy::kBlocking,
          .queue_depth = 8U,
          .max_inflight_samples = 8U,
          .producer_max_emits_per_compute = 8U,
          .fanout = 1U,
          .cache_capacity = 8U,
          .locality = ChannelLocalityKind::kProcessLocal,
          .disposition = PlanDisposition::kAccepted,
      },
  };
  plan.channel_locality = {ChannelLocalityRecord{
      .edge = EdgeId{1U},
      .requested = ChannelLocalityKind::kProcessLocal,
      .accepted = ChannelLocalityKind::kProcessLocal,
      .requires_runtime_attach = false,
  }};
  plan.generic_output_loans = {
      GenericOutputLoanPlanRecord{
          .output = PortId{1U},
          .plan_generation = plan.plan_generation,
          .pool_id = "holoscan-runtime-channel-loans",
          .slot_size_bytes = sizeof(std::int32_t),
          .aggregate_byte_capacity = 4U * sizeof(std::int32_t),
          .slot_count = 4U,
          .max_live_loans = 2U,
          .max_emits_per_compute = 8U,
          .retirement_capacity = 1U,
          .effective_profile = Profile::kStandard,
          .disposition = PlanDisposition::kAccepted,
      },
  };
  replay = {
      RuntimeChannelPortReplayRecord{
          .port = PortId{1U},
          .owner = VertexId{1U},
          .direction = PortDirection::kOutput,
          .declaration = const_cast<void*>(output.declaration),
          .type_ops = output.runtime_type_ops,
      },
      RuntimeChannelPortReplayRecord{
          .port = PortId{2U},
          .owner = VertexId{2U},
          .direction = PortDirection::kInput,
          .declaration = const_cast<void*>(input.declaration),
          .type_ops = input.runtime_type_ops,
      },
      RuntimeChannelPortReplayRecord{
          .port = PortId{3U},
          .owner = VertexId{3U},
          .direction = PortDirection::kInput,
          .declaration = const_cast<void*>(second_input.declaration),
          .type_ops = second_input.runtime_type_ops,
      },
  };
}

[[nodiscard]] PortPlanRecord& SessionFixture::planned_port(PortId id) {
  const auto record = std::find_if(
      plan.ports.begin(), plan.ports.end(), [id](const auto& value) { return value.id == id; });
  if (record == plan.ports.end()) {
    ADD_FAILURE() << "missing planned port " << id.value();
    throw std::logic_error{"missing planned port"};
  }
  return *record;
}

[[nodiscard]] ProcessLocalChannelServicePlanRecord& SessionFixture::planned_service(
    std::string_view service_id) {
  const auto record =
      std::find_if(plan.process_local_channel_services.begin(),
                   plan.process_local_channel_services.end(),
                   [service_id](const auto& value) { return value.service_id == service_id; });
  if (record == plan.process_local_channel_services.end()) {
    ADD_FAILURE() << "missing process-local service " << service_id;
    throw std::logic_error{"missing process-local service"};
  }
  return *record;
}

[[nodiscard]] ProcessLocalChannelPublisherPlanRecord& SessionFixture::planned_publisher(
    std::string_view service_id) {
  const auto record =
      std::find_if(plan.process_local_channel_publishers.begin(),
                   plan.process_local_channel_publishers.end(),
                   [service_id](const auto& value) { return value.service_id == service_id; });
  if (record == plan.process_local_channel_publishers.end()) {
    ADD_FAILURE() << "missing process-local publisher for service " << service_id;
    throw std::logic_error{"missing process-local publisher"};
  }
  return *record;
}

[[nodiscard]] ProcessLocalChannelSubscriberPlanRecord& SessionFixture::planned_subscriber(
    std::size_t index) {
  if (index >= plan.process_local_channel_subscribers.size()) {
    ADD_FAILURE() << "missing process-local subscriber at index " << index;
    throw std::logic_error{"missing process-local subscriber"};
  }
  return plan.process_local_channel_subscribers[index];
}

[[nodiscard]] PayloadEdgePlanRecord& SessionFixture::planned_edge(EdgeId id) {
  const auto record = std::find_if(plan.payload_edges.begin(),
                                   plan.payload_edges.end(),
                                   [id](const auto& value) { return value.edge == id; });
  if (record == plan.payload_edges.end()) {
    ADD_FAILURE() << "missing payload edge " << id.value;
    throw std::logic_error{"missing payload edge"};
  }
  return *record;
}

void SessionFixture::expect_queue_depth(std::size_t subscriber_index, std::uint32_t depth) {
  const auto& subscriber = planned_subscriber(subscriber_index);
  const auto& port = planned_port(subscriber.target);
  const auto& service = planned_service(subscriber.service_id);
  const auto& edge = planned_edge(subscriber.edge);
  std::uint32_t service_depth{};
  for (const auto& candidate : plan.process_local_channel_subscribers) {
    if (candidate.service_id == subscriber.service_id) {
      service_depth = std::max(service_depth, candidate.requested_queue_depth);
    }
  }
  EXPECT_EQ(port.queue_depth, depth);
  EXPECT_EQ(subscriber.requested_queue_depth, depth);
  EXPECT_EQ(subscriber.max_borrowed_samples, depth);
  EXPECT_EQ(subscriber.cache_capacity, depth);
  EXPECT_EQ(subscriber.readiness_capacity, depth);
  EXPECT_EQ(edge.queue_depth, depth);
  EXPECT_EQ(edge.cache_capacity, depth);
  EXPECT_EQ(service.max_subscriber_queue_depth, service_depth);
  EXPECT_EQ(service.max_subscriber_borrowed_samples, service_depth);
}

void SessionFixture::set_queue_depth(std::size_t subscriber_index, std::uint32_t depth) {
  if (depth == 0U) {
    ADD_FAILURE() << "queue depth must be nonzero";
    throw std::logic_error{"zero queue depth"};
  }
  auto& subscriber = planned_subscriber(subscriber_index);
  planned_port(subscriber.target).queue_depth = depth;
  subscriber.requested_queue_depth = depth;
  subscriber.max_borrowed_samples = depth;
  subscriber.cache_capacity = depth;
  subscriber.readiness_capacity = depth;
  auto& edge = planned_edge(subscriber.edge);
  edge.queue_depth = depth;
  edge.cache_capacity = depth;

  auto& service = planned_service(subscriber.service_id);
  service.max_subscriber_queue_depth = 0U;
  service.max_subscriber_borrowed_samples = 0U;
  for (const auto& candidate : plan.process_local_channel_subscribers) {
    if (candidate.service_id == subscriber.service_id) {
      service.max_subscriber_queue_depth =
          std::max(service.max_subscriber_queue_depth, candidate.requested_queue_depth);
      service.max_subscriber_borrowed_samples =
          std::max(service.max_subscriber_borrowed_samples, candidate.max_borrowed_samples);
    }
  }
  expect_queue_depth(subscriber_index, depth);
}

void SessionFixture::set_primary_queue_policy(QueuePolicy policy) {
  auto& subscriber = planned_subscriber(0U);
  auto& service = planned_service(subscriber.service_id);
  auto& publisher = planned_publisher(subscriber.service_id);
  const std::string service_id = "process-local-channel." + std::to_string(plan.plan_generation) +
                                 ".source.1.policy." +
                                 std::to_string(static_cast<std::uint32_t>(policy));
  service.queue_policy = policy;
  service.service_id = service_id;
  publisher.service_id = service_id;
  publisher.replacement_attribution_required = policy != QueuePolicy::kBlocking;
  subscriber.service_id = service_id;
  auto& edge = planned_edge(subscriber.edge);
  edge.queue_policy = policy;

  EXPECT_EQ(service.queue_policy, policy);
  EXPECT_EQ(service.service_id, service_id);
  EXPECT_EQ(publisher.service_id, service_id);
  EXPECT_EQ(publisher.replacement_attribution_required, policy != QueuePolicy::kBlocking);
  EXPECT_EQ(subscriber.service_id, service_id);
  EXPECT_EQ(edge.queue_policy, policy);
}

void SessionFixture::use_direct_retained_delivery() {
  auto& service = plan.process_local_channel_services.front();
  service.delivery_kind = ProcessLocalChannelDeliveryKind::kDirectRetained;
  service.event_wake = false;
  plan.process_local_channel_subscribers.front().event_wake = false;
}

void SessionFixture::use_one_slot_direct_retained_delivery() {
  use_direct_retained_delivery();
  set_queue_depth(0U, 1U);
}

void SessionFixture::use_one_slot_blocking_transport() {
  set_queue_depth(0U, 1U);
}

void SessionFixture::use_same_partition_eventless_transport() {
  for (auto& service : plan.process_local_channel_services) {
    service.event_wake = false;
  }
  for (auto& subscriber : plan.process_local_channel_subscribers) {
    subscriber.event_wake = false;
  }
  std::vector<VertexId> members;
  members.reserve(plan.ports.size());
  for (const auto& port : plan.ports) {
    if (port.owner && std::find(members.begin(), members.end(), port.owner) == members.end()) {
      members.push_back(port.owner);
    }
  }
  plan.execution_partitions = {ExecutionPartitionRecord{
      .id = ExecutionPartitionId{1U},
      .members = std::move(members),
      .execution_lane = 0U,
  }};
}

void SessionFixture::use_single_partition_runner() {
  plan.partition_runners = {PartitionRunnerPlanRecord{
      .ordinal = 0U,
      .partition = ExecutionPartitionId{1U},
      .execution_lane = 0U,
      .process_group = ProcessGroupId{1U},
  }};
}

void SessionFixture::use_cross_process_edge() {
  const std::string service =
      fmt::format("cross-process-channel.{}.source.1.policy.0.anchor.1", plan.plan_generation);
  plan.cross_process_channel_services = {
      CrossProcessChannelServicePlanRecord{
          .edge = EdgeId{1U},
          .source = PortId{1U},
          .target = PortId{2U},
          .service_id = service,
          .payload_identity = plan.ports.front().payload_identity,
          .payload_size_bytes = sizeof(std::int32_t),
          .queue_policy = QueuePolicy::kBlocking,
          .max_subscribers = 1U,
          .max_subscriber_queue_depth = 8U,
          .max_subscriber_borrowed_samples = 8U,
          .publisher_history_depth = 0U,
          .event_wake = true,
          .source_process_group = ProcessGroupId{1U},
          .target_process_group = ProcessGroupId{2U},
          .target_process_groups = {ProcessGroupId{2U}},
      },
  };
  plan.cross_process_channel_publishers = {
      CrossProcessChannelPublisherPlanRecord{
          .edge = EdgeId{1U},
          .source = PortId{1U},
          .service_id = service,
          .max_loaned_samples = 8U,
      },
  };
  plan.cross_process_channel_subscribers = {
      CrossProcessChannelSubscriberPlanRecord{
          .edge = EdgeId{1U},
          .source = PortId{1U},
          .target = PortId{2U},
          .service_id = service,
          .requested_queue_depth = 8U,
          .max_borrowed_samples = 8U,
          .cache_capacity = 8U,
          .readiness_capacity = 8U,
          .event_wake = true,
      },
  };
  plan.process_local_channel_services.clear();
  plan.process_local_channel_publishers.clear();
  plan.process_local_channel_subscribers.clear();
  plan.vertex_process_placements = {
      VertexProcessPlacementPlanRecord{
          .vertex = VertexId{1U},
          .process_group = ProcessGroupId{1U},
      },
      VertexProcessPlacementPlanRecord{
          .vertex = VertexId{2U},
          .process_group = ProcessGroupId{2U},
      },
  };
  plan.payload_edges.front().locality = ChannelLocalityKind::kCrossProcess;
  plan.channel_locality.front().accepted = ChannelLocalityKind::kCrossProcess;
  plan.channel_locality.front().requires_runtime_attach = true;
}

void SessionFixture::add_mixed_policy_fanout() {
  const std::string service = "process-local-channel." + std::to_string(plan.plan_generation) +
                              ".source.1.policy." +
                              std::to_string(static_cast<std::uint32_t>(QueuePolicy::kDropOldest));
  plan.ports.push_back(PortPlanRecord{
      .id = PortId{3U},
      .owner = VertexId{3U},
      .direction = PortDirection::kInput,
      .local_name = "input",
      .payload_identity = plan.ports.front().payload_identity,
      .queue_depth = 8U,
      .effective_profile = Profile::kStandard,
  });
  plan.process_local_channel_services.push_back(ProcessLocalChannelServicePlanRecord{
      .source = PortId{1U},
      .service_id = service,
      .payload_identity = plan.ports.front().payload_identity,
      .payload_size_bytes = sizeof(std::int32_t),
      .queue_policy = QueuePolicy::kDropOldest,
      .max_subscribers = 1U,
      .max_subscriber_queue_depth = 8U,
      .max_subscriber_borrowed_samples = 8U,
      .publisher_history_depth = 0U,
      .event_wake = true,
  });
  plan.process_local_channel_publishers.push_back(ProcessLocalChannelPublisherPlanRecord{
      .source = PortId{1U},
      .service_id = service,
      .max_loaned_samples = 8U,
      .replacement_attribution_required = true,
  });
  plan.process_local_channel_subscribers.push_back(ProcessLocalChannelSubscriberPlanRecord{
      .edge = EdgeId{2U},
      .source = PortId{1U},
      .target = PortId{3U},
      .service_id = service,
      .requested_queue_depth = 8U,
      .max_borrowed_samples = 8U,
      .cache_capacity = 8U,
      .readiness_capacity = 8U,
      .event_wake = true,
  });
  plan.payload_edges.push_back(PayloadEdgePlanRecord{
      .edge = EdgeId{2U},
      .source = PortId{1U},
      .target = PortId{3U},
      .payload_identity = plan.ports.front().payload_identity,
      .queue_policy = QueuePolicy::kDropOldest,
      .queue_depth = 8U,
      .max_inflight_samples = 8U,
      .producer_max_emits_per_compute = 8U,
      .fanout = 2U,
      .cache_capacity = 8U,
      .locality = ChannelLocalityKind::kProcessLocal,
      .disposition = PlanDisposition::kAccepted,
  });
  plan.channel_locality.push_back(ChannelLocalityRecord{
      .edge = EdgeId{2U},
      .requested = ChannelLocalityKind::kProcessLocal,
      .accepted = ChannelLocalityKind::kProcessLocal,
      .requires_runtime_attach = false,
  });
}

void SessionFixture::add_blocking_fanout() {
  const std::string& service = plan.process_local_channel_services.front().service_id;
  plan.ports.push_back(PortPlanRecord{
      .id = PortId{3U},
      .owner = VertexId{3U},
      .direction = PortDirection::kInput,
      .local_name = "input",
      .payload_identity = plan.ports.front().payload_identity,
      .serialized_payload_capacity_bytes = plan.ports[1].serialized_payload_capacity_bytes,
      .queue_depth = 8U,
      .effective_profile = Profile::kStandard,
  });
  plan.process_local_channel_services.front().max_subscribers = 2U;
  plan.process_local_channel_subscribers.push_back(ProcessLocalChannelSubscriberPlanRecord{
      .edge = EdgeId{2U},
      .source = PortId{1U},
      .target = PortId{3U},
      .service_id = service,
      .requested_queue_depth = 8U,
      .max_borrowed_samples = 8U,
      .cache_capacity = 8U,
      .readiness_capacity = 8U,
      .event_wake = true,
  });
  plan.payload_edges.front().fanout = 2U;
  plan.payload_edges.push_back(PayloadEdgePlanRecord{
      .edge = EdgeId{2U},
      .source = PortId{1U},
      .target = PortId{3U},
      .payload_identity = plan.ports.front().payload_identity,
      .queue_policy = QueuePolicy::kBlocking,
      .queue_depth = 8U,
      .max_inflight_samples = 8U,
      .producer_max_emits_per_compute = 8U,
      .fanout = 2U,
      .cache_capacity = 8U,
      .locality = ChannelLocalityKind::kProcessLocal,
      .disposition = PlanDisposition::kAccepted,
  });
  plan.channel_locality.push_back(ChannelLocalityRecord{
      .edge = EdgeId{2U},
      .requested = ChannelLocalityKind::kProcessLocal,
      .accepted = ChannelLocalityKind::kProcessLocal,
      .requires_runtime_attach = false,
  });
}

void SessionFixture::use_one_slot_blocking_fanout() {
  use_one_slot_blocking_transport();
  add_blocking_fanout();
  set_queue_depth(1U, 1U);
}

void SessionFixture::use_one_slot_drop_oldest_fanout() {
  use_one_slot_blocking_fanout();
  set_primary_queue_policy(QueuePolicy::kDropOldest);
  const std::string& service = plan.process_local_channel_services.front().service_id;
  plan.process_local_channel_subscribers.back().service_id = service;
  plan.payload_edges.back().queue_policy = QueuePolicy::kDropOldest;
}

void SessionFixture::add_compute_only_input_for_sink() {
  const std::string& service = plan.process_local_channel_services.front().service_id;
  plan.ports.push_back(PortPlanRecord{
      .id = PortId{3U},
      .owner = VertexId{2U},
      .direction = PortDirection::kInput,
      .local_name = "compute_only",
      .payload_identity = plan.ports.front().payload_identity,
      .queue_depth = 8U,
      .effective_profile = Profile::kStandard,
  });
  plan.process_local_channel_services.front().max_subscribers = 2U;
  plan.process_local_channel_subscribers.push_back(ProcessLocalChannelSubscriberPlanRecord{
      .edge = EdgeId{2U},
      .source = PortId{1U},
      .target = PortId{3U},
      .service_id = service,
      .requested_queue_depth = 8U,
      .max_borrowed_samples = 8U,
      .cache_capacity = 8U,
      .readiness_capacity = 8U,
      .event_wake = true,
  });
  plan.payload_edges.push_back(PayloadEdgePlanRecord{
      .edge = EdgeId{2U},
      .source = PortId{1U},
      .target = PortId{3U},
      .payload_identity = plan.ports.front().payload_identity,
      .queue_policy = QueuePolicy::kBlocking,
      .queue_depth = 8U,
      .max_inflight_samples = 8U,
      .producer_max_emits_per_compute = 8U,
      .fanout = 2U,
      .cache_capacity = 8U,
      .locality = ChannelLocalityKind::kProcessLocal,
      .disposition = PlanDisposition::kAccepted,
  });
  plan.channel_locality.push_back(ChannelLocalityRecord{
      .edge = EdgeId{2U},
      .requested = ChannelLocalityKind::kProcessLocal,
      .accepted = ChannelLocalityKind::kProcessLocal,
      .requires_runtime_attach = false,
  });
  replay[2].owner = VertexId{2U};
}

void SessionFixture::use_drop_oldest_queue_policy() {
  set_primary_queue_policy(QueuePolicy::kDropOldest);
}

void SessionFixture::use_one_slot_drop_oldest_queue() {
  use_drop_oldest_queue_policy();
  set_queue_depth(0U, 1U);
}

void SessionFixture::use_two_slot_drop_oldest_queue() {
  use_drop_oldest_queue_policy();
  set_queue_depth(0U, 2U);
}

void SessionFixture::use_two_slot_blocking_queue() {
  set_queue_depth(0U, 2U);
}

void SessionFixture::require_direct_capture_conversion() {
  plan.input_selections = {InputSelectionPlanRecord{
      .input = PortId{2U},
      .selector = InputSelectorKind::kExact,
      .cardinality = InputSelectionCardinality::kSingleSample,
      .max_selected_samples = 1U,
      .max_selected_samples_origin = SelectedSampleBoundOrigin::kSingleSample,
      .batch_selection_mode = BatchSelectionMode::kNotApplicable,
      .cache_capacity = 8U,
      .reference_clock = ClockId{2U},
      .conversion_id = 1U,
  }};
  plan.clock_conversions = {ClockConversionRecord{
      .id = 1U,
      .plan_generation = plan.plan_generation,
      .from = ClockId{1U},
      .to = ClockId{2U},
      .scale = ClockRateRatio{.numerator = 2, .denominator = 1U},
      .offset_ns = 3,
  }};
}

void SessionFixture::enable_reuse_last() {
  plan.input_selections = {InputSelectionPlanRecord{
      .input = PortId{2U},
      .selector = InputSelectorKind::kExact,
      .cardinality = InputSelectionCardinality::kSingleSample,
      .max_selected_samples = 1U,
      .max_selected_samples_origin = SelectedSampleBoundOrigin::kSingleSample,
      .batch_selection_mode = BatchSelectionMode::kNotApplicable,
      .cache_capacity = 8U,
      .reference_clock = ClockId{1U},
      .missing = TemporalRecordMissingKind::kReuseLast,
      .reuse_max_age_ns = 100,
  }};
}

void SessionFixture::enable_scalar(InputSelectorKind selector, TemporalRecordMissingKind missing) {
  plan.input_selections = {InputSelectionPlanRecord{
      .input = PortId{2U},
      .selector = selector,
      .cardinality = InputSelectionCardinality::kSingleSample,
      .max_selected_samples = 1U,
      .max_selected_samples_origin = SelectedSampleBoundOrigin::kSingleSample,
      .batch_selection_mode = BatchSelectionMode::kNotApplicable,
      .cache_capacity = 8U,
      .reference_clock = ClockId{1U},
      .missing = missing,
      .max_age_ns = selector == InputSelectorKind::kLatestWithin ? 100 : 0,
  }};
}

void SessionFixture::enable_latest_within() {
  plan.input_selections = {InputSelectionPlanRecord{
      .input = PortId{2U},
      .selector = InputSelectorKind::kLatestWithin,
      .cardinality = InputSelectionCardinality::kSingleSample,
      .max_selected_samples = 1U,
      .max_selected_samples_origin = SelectedSampleBoundOrigin::kSingleSample,
      .batch_selection_mode = BatchSelectionMode::kNotApplicable,
      .cache_capacity = 8U,
      .reference_clock = ClockId{1U},
      .max_age_ns = 100,
  }};
}

void SessionFixture::enable_drain_all(std::uint32_t max_selected_samples) {
  plan.input_selections = {InputSelectionPlanRecord{
      .input = PortId{2U},
      .selector = InputSelectorKind::kDrainAll,
      .cardinality = InputSelectionCardinality::kSampleBatch,
      .max_selected_samples = max_selected_samples,
      .max_selected_samples_origin = SelectedSampleBoundOrigin::kResolvedRetainedCapacity,
      .batch_selection_mode = BatchSelectionMode::kConsumeSelected,
      .cache_capacity = 8U,
      .reference_clock = ClockId{1U},
  }};
}

void SessionFixture::enable_bracketed_window(std::uint32_t max_selected_samples) {
  plan.input_selections = {InputSelectionPlanRecord{
      .input = PortId{2U},
      .selector = InputSelectorKind::kBracketedWindow,
      .cardinality = InputSelectionCardinality::kSampleBatch,
      .max_selected_samples = max_selected_samples,
      .max_selected_samples_origin = SelectedSampleBoundOrigin::kSelectorMaxSamples,
      .batch_selection_mode = BatchSelectionMode::kNonDestructiveSnapshot,
      .cache_capacity = 8U,
      .reference_clock = ClockId{1U},
      .lookbehind = 1U,
      .lookahead = 1U,
  }};
  plan.bracketed_windows = {BracketedWindowPlanRecord{
      .owner = OperatorRef{2U, plan.plan_generation, 0U},
      .input = PortId{2U},
      .anchor = PortId{2U},
      .lookbehind = 1U,
      .lookahead = 1U,
      .max_samples = max_selected_samples,
      .event_clock = ClockId{1U},
      .queue_capacity = 8U,
      .cache_capacity = 8U,
      .producer_max_emits_per_compute = 8U,
      .retained_overlap_capacity = 3U,
  }};
}

[[nodiscard]] expected<RuntimeEvent, Error> wait_for_event(RuntimeChannelSession& session,
                                                           PortBinding& binding,
                                                           ClockId observed_clock) {
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  std::int64_t observed_ns = 1;
  while (std::chrono::steady_clock::now() < deadline) {
    auto polled =
        binding.poll(TimePoint{.timestamp_ns = observed_ns++, .clock_id = observed_clock});
    if (!polled) {
      return make_unexpected(std::move(polled).error());
    }
    RuntimeEvent event;
    auto drained = binding.drain(std::span<RuntimeEvent>{&event, 1U});
    if (!drained) {
      return make_unexpected(std::move(drained).error());
    }
    if (*drained == 1U) {
      return event;
    }
    if (auto status = session.status(); !status) {
      return make_unexpected(std::move(status).error());
    }
    std::this_thread::sleep_for(1ms);
  }
  return make_unexpected(Error{ErrorCode::kNotReady});
}

[[nodiscard]] expected<RuntimeChannelInputResetState, Error> wait_for_occupied_sample(
    RuntimeChannelSession& session, const PortId input) {
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (std::chrono::steady_clock::now() < deadline) {
    auto state = RuntimeChannelSessionTestAccess::input_reset_state(session, input);
    if (!state) {
      return make_unexpected(std::move(state).error());
    }
    if (state->occupied_samples != 0U) {
      return *state;
    }
    if (auto status = session.status(); !status) {
      return make_unexpected(std::move(status).error());
    }
    std::this_thread::sleep_for(1ms);
  }
  return make_unexpected(Error{ErrorCode::kNotReady});
}

[[nodiscard]] expected<PublishedScalarScript, Error> publish_scalar_script(
    RuntimeChannelSession& session, PortBinding& binding, SourceOperator& source,
    std::span<const std::int32_t> values, std::span<const std::int64_t> capture_times) {
  if (values.empty() || values.size() != capture_times.size()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto prepared = session.prepare_activation(VertexId{1U}, 1U, {});
  if (!prepared) {
    return make_unexpected(std::move(prepared).error());
  }
  for (std::size_t index = 0U; index < values.size(); ++index) {
    const SampleMetadata metadata{
        .capture_timestamp_ns = capture_times[index],
        .source_clock_id = ClockId{1U},
    };
    auto emitted = source.output.emit(Sample<std::int32_t>{values[index], metadata});
    if (!emitted) {
      const Error error = std::move(emitted).error();
      return make_unexpected(Error{
          error.code,
          "publish_scalar_script emit[" + std::to_string(index) +
              "]: " + runtime_error_message(error),
      });
    }
  }
  auto retired = session.retire_activation(VertexId{1U}, 1U, true);
  if (!retired) {
    return make_unexpected(std::move(retired).error());
  }

  PublishedScalarScript result;
  result.events.reserve(values.size());
  result.records.reserve(values.size());
  for (std::size_t index = 0U; index < values.size(); ++index) {
    auto event = wait_for_event(session, binding);
    if (!event) {
      const Error error = std::move(event).error();
      return make_unexpected(Error{
          error.code,
          "publish_scalar_script wait[" + std::to_string(index) +
              "]: " + runtime_error_message(error),
      });
    }
    auto record = session.temporal_sample(event->port, event->source_sequence);
    if (!record) {
      const Error error = std::move(record).error();
      return make_unexpected(Error{
          error.code,
          "publish_scalar_script record[" + std::to_string(index) +
              "]: " + runtime_error_message(error),
      });
    }
    result.events.push_back(*event);
    result.records.push_back(*record);
  }
  return result;
}

[[nodiscard]] std::vector<std::int32_t> batch_values(const SelectedBatchView<std::int32_t>& batch) {
  std::vector<std::int32_t> values;
  values.reserve(batch.size());
  for (const auto& sample : batch) {
    values.push_back(sample.data);
  }
  return values;
}

[[nodiscard]] ErrorCode observe_input_accessor(Input<std::int32_t>& input,
                                               InputAccessorKind accessor) {
  switch (accessor) {
    case InputAccessorKind::kSample: {
      auto result = input.receive();
      return result ? ErrorCode::kSuccess : result.error().code;
    }
    case InputAccessorKind::kData: {
      auto result = input.receive_data();
      return result ? ErrorCode::kSuccess : result.error().code;
    }
    case InputAccessorKind::kBatch: {
      auto result = input.selected_batch();
      return result ? ErrorCode::kSuccess : result.error().code;
    }
  }
  return ErrorCode::kFailure;
}

[[nodiscard]] ErrorCode observe_exhausted_scalar_claims(Input<std::int32_t>& input) {
  auto first = input.receive();
  if (!first) {
    return first.error().code;
  }
  auto second = input.receive();
  if (!second) {
    return second.error().code;
  }
  auto exhausted = input.receive();
  return exhausted ? ErrorCode::kSuccess : exhausted.error().code;
}

[[nodiscard]] expected<ErrorCode, Error> observe_channel_guard(
    const InputGuardExpectation& expectation) {
  if (expectation.channel != ChannelGuardApplicability::kAccessor) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (expectation.unbound) {
    SinkOperator unbound;
    return observe_input_accessor(unbound.input, expectation.accessor);
  }

  SessionFixture fixture;
  const bool batch = expectation.state.selector_shape == SyntheticSelectorShape::kBatch;
  if (batch) {
    fixture.enable_drain_all(2U);
  } else {
    fixture.enable_scalar(InputSelectorKind::kExact, TemporalRecordMissingKind::kSkip);
  }
  if (expectation.exhaust_claims) {
    fixture.use_two_slot_blocking_queue();
  }

  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  if (!session) {
    return make_unexpected(std::move(session).error());
  }

  std::unique_ptr<PortBinding> readiness;
  PublishedScalarScript script;
  const bool value_state = expectation.state.selection.state == InputState::kSelected ||
                           expectation.state.selection.state == InputState::kReused;
  if (value_state) {
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
    if (!created) {
      return make_unexpected(std::move(created).error());
    }
    readiness = std::move(*created);
    auto refreshed = (*session)->refresh_connections();
    if (!refreshed) {
      return make_unexpected(std::move(refreshed).error());
    }
    auto armed = (*session)->arm();
    if (!armed) {
      return make_unexpected(std::move(armed).error());
    }
    constexpr std::array value{71};
    constexpr std::array<std::int64_t, 1U> capture_time{17};
    auto published =
        publish_scalar_script(**session, *readiness, fixture.source, value, capture_time);
    if (!published) {
      return make_unexpected(std::move(published).error());
    }
    script = std::move(*published);
  }

  const std::span<const TemporalSampleRecord> selected_samples =
      value_state ? std::span<const TemporalSampleRecord>{script.records}
                  : std::span<const TemporalSampleRecord>{};
  const std::array selected{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = expectation.state.selection,
      .samples = selected_samples,
      .batch = batch,
      .destructive = value_state,
  }};
  auto prepared = (*session)->prepare_activation(VertexId{2U}, 1U, selected);
  if (!prepared) {
    return make_unexpected(std::move(prepared).error());
  }

  bool retired = false;
  if (expectation.state.stale_generation) {
    auto result = (*session)->retire_activation(VertexId{2U}, 1U, false);
    if (!result) {
      return make_unexpected(std::move(result).error());
    }
    retired = true;
  }
  const ErrorCode observed = expectation.exhaust_claims
                                 ? observe_exhausted_scalar_claims(fixture.sink.input)
                                 : observe_input_accessor(fixture.sink.input, expectation.accessor);

  if (!retired) {
    auto result = (*session)->retire_activation(VertexId{2U}, 1U, false);
    if (!result) {
      return make_unexpected(std::move(result).error());
    }
  }
  if (readiness) {
    for (const RuntimeEvent& event : script.events) {
      auto result = readiness->retire(event.source_sequence);
      if (!result) {
        return make_unexpected(std::move(result).error());
      }
    }
    auto result = (*session)->retire_samples(PortId{2U}, script.records);
    if (!result) {
      return make_unexpected(std::move(result).error());
    }
  }
  (*session)->cancel();
  auto joined = (*session)->join();
  if (!joined) {
    return make_unexpected(std::move(joined).error());
  }
  return observed;
}

}  // namespace holoscan::detail::runtime_channel_binding_test
