/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "runtime_channel_binding_test_access.hpp"
#include "runtime_channel_session_impl.hpp"

namespace holoscan::detail {
#if defined(HOLOSCAN_RUNTIME_CHANNEL_TEST_ACCESS)
expected<std::uint32_t, Error> RuntimeChannelSessionTestAccess::active_queue_depth(
    const RuntimeChannelSession& session, EdgeId edge) noexcept {
  if (session.impl_ == nullptr || !edge) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber = std::ranges::find_if(
      session.impl_->subscribers_, [&](const auto& candidate) { return candidate->edge == edge; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const std::scoped_lock lock{(*subscriber)->mutex};
  return (*subscriber)->active_typed_slot_capacity;
}

expected<void, Error> RuntimeChannelSessionTestAccess::bind_queue_observation(
    const RuntimeChannelSession& session, const ExecutionPlanState& plan,
    const RuntimeRateState& rates, const RuntimeRateSnapshotCapture capture, const EdgeId edge,
    const std::shared_ptr<RuntimeChannelQueueObservationTestControl>& control) noexcept {
  if (capture == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto snapshot = capture(rates);
  const auto* generation = snapshot ? snapshot->published() : nullptr;
  if (session.impl_ == nullptr || control == nullptr || !edge || plan.digest == PlanDigest() ||
      plan.plan_generation == 0U || session.impl_->plan_generation_ != plan.plan_generation ||
      !snapshot || snapshot->plan_digest() != plan.digest ||
      snapshot->plan_generation() != plan.plan_generation || snapshot->run_generation() == 0U ||
      session.impl_->run_generation_ == 0U ||
      session.impl_->run_generation_ != snapshot->run_generation() || generation == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const RateGenerationId rate_generation = generation->rate_generation;
  if (rate_generation == 0U || generation->disposition != RateGenerationDisposition::kPublished ||
      generation->run_generation != snapshot->run_generation()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }

  const auto recipe =
      std::ranges::find(plan.queue_sizing_recipes, edge, &QueueSizingRecipePlanRecord::edge);
  if (recipe == plan.queue_sizing_recipes.end() ||
      recipe->disposition != PlanDisposition::kAccepted ||
      std::ranges::count(plan.queue_sizing_recipes, edge, &QueueSizingRecipePlanRecord::edge) !=
          1) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const auto queue_rows = snapshot->queue_sizings(*generation);
  const auto queue =
      std::ranges::find(queue_rows, recipe->id, &QueueSizingGenerationRecord::recipe);
  if (queue == queue_rows.end() || queue->run_generation != snapshot->run_generation() ||
      queue->rate_generation != rate_generation ||
      queue->disposition != PlanDisposition::kAccepted || queue->effective_logical_depth == 0U ||
      std::ranges::count(queue_rows, recipe->id, &QueueSizingGenerationRecord::recipe) != 1) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }

  const auto subscriber = std::ranges::find_if(
      session.impl_->subscribers_, [&](const auto& candidate) { return candidate->edge == edge; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  {
    const std::scoped_lock subscriber_lock{(*subscriber)->mutex};
    if ((*subscriber)->active_typed_slot_capacity != queue->effective_logical_depth) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
  }

  const std::scoped_lock control_lock{control->mutex_};
  if (control->bound_) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  control->observation_ = RuntimeChannelQueueObservation{
      .plan_digest = plan.digest,
      .plan_generation = plan.plan_generation,
      .edge = edge,
      .run_generation = snapshot->run_generation(),
      .rate_generation = rate_generation,
      .selected_logical_depth = queue->effective_logical_depth,
      .rate_recommendation = queue->rate_recommendation,
  };
  control->rate_state_ = &rates;
  control->bound_ = true;
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::sample_queue_observation(
    const RuntimeChannelSession& session, const RuntimeRateState& rates,
    const RuntimeRateSnapshotCapture capture, const EdgeId edge,
    const std::shared_ptr<RuntimeChannelQueueObservationTestControl>& control) noexcept {
  if (session.impl_ == nullptr || control == nullptr || capture == nullptr || !edge) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber = std::ranges::find_if(
      session.impl_->subscribers_, [&](const auto& candidate) { return candidate->edge == edge; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }

  std::size_t occupied{};
  std::size_t staged{};
  std::uint64_t next_arrival_sequence{};
  std::uint64_t terminal_sequence{};
  std::uint32_t active_depth{};
  bool terminal_received{};
  {
    const std::scoped_lock subscriber_lock{(*subscriber)->mutex};
    occupied =
        static_cast<std::size_t>(std::ranges::count_if((*subscriber)->slots, [](const auto& slot) {
          return slot.occupied && !slot.direct_ingress_unread;
        }));
    staged =
        static_cast<std::size_t>(std::ranges::count_if((*subscriber)->slots, [](const auto& slot) {
          return slot.occupied && slot.direct_ingress_unread;
        }));
    next_arrival_sequence = (*subscriber)->next_arrival_sequence;
    terminal_sequence = (*subscriber)->end_of_stream_transport_sequence;
    active_depth = (*subscriber)->active_typed_slot_capacity;
    terminal_received = (*subscriber)->end_of_stream_received;
  }
  if (occupied > std::numeric_limits<std::uint32_t>::max() ||
      staged > std::numeric_limits<std::uint32_t>::max()) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  const auto observed_occupancy = static_cast<std::uint32_t>(occupied);
  const auto staged_occupancy = static_cast<std::uint32_t>(staged);

  auto live = capture(rates);
  const auto* published = live ? live->published() : nullptr;

  const std::scoped_lock control_lock{control->mutex_};
  auto& observation = control->observation_;
  if (!control->bound_ || control->rate_state_ != &rates || !live || published == nullptr ||
      live->plan_digest() != observation.plan_digest ||
      live->plan_generation() != observation.plan_generation ||
      live->run_generation() != observation.run_generation ||
      published->rate_generation != observation.rate_generation ||
      observation.plan_generation != session.impl_->plan_generation_ || observation.edge != edge ||
      observation.run_generation != session.impl_->run_generation_ ||
      active_depth != observation.selected_logical_depth ||
      observed_occupancy > observation.selected_logical_depth) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

  observation.observed_occupancy = observed_occupancy;
  observation.observed_occupancy_high_water =
      std::max(observation.observed_occupancy_high_water.value_or(0U), observed_occupancy);
  control->previous_occupancy_ = observed_occupancy;
  control->previous_staged_occupancy_ = staged_occupancy;
  control->previous_arrival_sequence_ = next_arrival_sequence;
  control->previous_terminal_sequence_ = terminal_sequence;
  control->previous_terminal_received_ = terminal_received;
  return {};
}

expected<void, Error>
RuntimeChannelSessionTestAccess::fail_fixed_tensor_shape_allocation_after_successes(
    RuntimeChannelSession& session, const std::size_t successful_allocations) noexcept {
  if (session.impl_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return session.impl_->fail_fixed_tensor_shape_allocation_after_successes_for_test(
      successful_allocations);
}

namespace {

template <typename Impl>
[[nodiscard]] auto cold_input_subscriber(Impl& impl, const PortId input) noexcept
    -> expected<decltype(impl.subscribers_.front().get()), Error> {
  if (!input || impl.armed_.load(std::memory_order_acquire) || impl.pump_thread_.joinable()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(impl.subscribers_.begin(), impl.subscribers_.end(), [&](const auto& candidate) {
        return candidate->target == input;
      });
  if (subscriber == impl.subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  return subscriber->get();
}

}  // namespace
expected<void, Error> RuntimeChannelSessionTestAccess::replace_output_publishers(
    RuntimeChannelSession& session, const PortId output,
    std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers,
    const std::span<const std::uint32_t> expected_recipient_counts, const bool refresh_before_emit,
    const bool peer_required) noexcept {
  if (!session.impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return session.impl_->replace_output_publishers_for_test(
      output, std::move(publishers), expected_recipient_counts, refresh_before_emit, peer_required);
}

expected<std::size_t, Error> RuntimeChannelSessionTestAccess::resolved_output_terminal_arms(
    const RuntimeChannelSession& session, const PortId output) noexcept {
  auto current = output_next_sequence(session, output);
  if (!current) {
    return make_unexpected(std::move(current).error());
  }
  const auto found = std::find_if(session.impl_->outputs_.begin(),
                                  session.impl_->outputs_.end(),
                                  [&](const auto& candidate) { return candidate->port == output; });
  const auto resolved = [](const auto& endpoints) {
    return std::ranges::count_if(endpoints,
                                 [](const auto& endpoint) { return endpoint.terminal_resolved; });
  };
  const auto& state = **found;
  return static_cast<std::size_t>(resolved(state.endpoints) + resolved(state.tensor_endpoints) +
                                  resolved(state.catalog_tensor_endpoints) +
                                  resolved(state.tensor_schema_endpoints));
}

expected<void, Error>
RuntimeChannelSessionTestAccess::replace_output_with_nonlocal_tensor_schema_publisher(
    RuntimeChannelSession& session, const PortId output,
    std::unique_ptr<channel_internal::TensorSchemaPayloadPublisher> publisher) noexcept {
  if (!session.impl_ || !output || publisher == nullptr ||
      session.impl_->armed_.load(std::memory_order_acquire) ||
      session.impl_->pump_thread_.joinable()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto found = std::find_if(session.impl_->outputs_.begin(),
                                  session.impl_->outputs_.end(),
                                  [&](const auto& candidate) { return candidate->port == output; });
  if (found == session.impl_->outputs_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  auto& state = **found;
  if (state.active || state.tensor_schema_endpoints.size() != 1U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto& endpoint = state.tensor_schema_endpoints.front();
  endpoint.publisher = std::move(publisher);
  endpoint.refresh_before_emit = true;
  endpoint.process_local_transport = false;
  endpoint.activation_cut_subscribers.clear();
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::throw_tensor_fanout_after_accepts(
    RuntimeChannelSession& session, const PortId output,
    const std::size_t accepted_count) noexcept {
  if (!session.impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return session.impl_->throw_tensor_fanout_after_accepts_for_test(output, accepted_count);
}

expected<void, Error> RuntimeChannelSessionTestAccess::fail_next_catalog_tensor_terminal_pre_submit(
    RuntimeChannelSession& session, const PortId output) noexcept {
  if (!session.impl_ || !output || session.impl_->armed_.load(std::memory_order_acquire) ||
      session.impl_->pump_thread_.joinable()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto found = std::find_if(session.impl_->outputs_.begin(),
                                  session.impl_->outputs_.end(),
                                  [&](const auto& candidate) { return candidate->port == output; });
  if (found == session.impl_->outputs_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  auto& endpoints = (*found)->catalog_tensor_endpoints;
  if (endpoints.size() != 1U || endpoints.front().terminal_resolved) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto& publisher = endpoints.front().publisher;
  if (!publisher.has_subscribers()) {
    return make_unexpected(Error{ErrorCode::kNotReady});
  }
  if (!channel_internal::TensorPublishLoanAccess::fail_next_catalog_terminal_pre_submit_for_test(
          publisher)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::exhaust_catalog_tensor_operation_sequence(
    RuntimeChannelSession& session, const PortId output) noexcept {
  if (!session.impl_ || !output || session.impl_->armed_.load(std::memory_order_acquire) ||
      session.impl_->pump_thread_.joinable()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto found = std::find_if(session.impl_->outputs_.begin(),
                                  session.impl_->outputs_.end(),
                                  [&](const auto& candidate) { return candidate->port == output; });
  if (found == session.impl_->outputs_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  auto& endpoints = (*found)->catalog_tensor_endpoints;
  if (endpoints.size() != 1U || endpoints.front().terminal_resolved) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto& publisher = endpoints.front().publisher;
  if (!publisher.has_subscribers()) {
    return make_unexpected(Error{ErrorCode::kNotReady});
  }
  if (!channel_internal::TensorPublishLoanAccess::exhaust_catalog_operation_sequence_for_test(
          publisher)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::replace_input_subscriber(
    RuntimeChannelSession& session, const PortId input,
    std::unique_ptr<channel_internal::InlinePayloadSubscriber> replacement) noexcept {
  if (!session.impl_ || replacement == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto subscriber = cold_input_subscriber(*session.impl_, input);
  if (!subscriber) {
    return make_unexpected(std::move(subscriber).error());
  }
  auto& state = **subscriber;
  const std::scoped_lock lock{state.transport_mutex, state.mutex};
  if (state.endpoint == nullptr || state.tensor_endpoint ||
      state.tensor_schema_endpoint != nullptr || state.externally_driven_import ||
      (state.wait_handle == 0U && state.terminal_wait_handle != 0U)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  // An eventless subscriber has no wait-set attachment to remove.
  if (state.wait_handle != 0U) {
    auto detached = state.endpoint->detach_wait_set();
    if (!detached) {
      return make_unexpected(std::move(detached).error());
    }
    const std::size_t expected_erasures = state.terminal_wait_handle == 0U ? 1U : 2U;
    std::size_t erased = session.impl_->wait_subscriber_by_handle_.erase(state.wait_handle);
    if (state.terminal_wait_handle != 0U) {
      erased += session.impl_->wait_subscriber_by_handle_.erase(state.terminal_wait_handle);
    }
    if (erased != expected_erasures || session.impl_->wait_attachment_count_ < expected_erasures) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    session.impl_->wait_attachment_count_ -= expected_erasures;
    state.wait_handle = 0U;
    state.terminal_wait_handle = 0U;
  }
  state.endpoint = std::move(replacement);
  state.refresh_staged_data_endpoint();
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::receive_one(RuntimeChannelSession& session,
                                                                   const PortId input) noexcept {
  if (!session.impl_) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto subscriber = cold_input_subscriber(*session.impl_, input);
  if (!subscriber) {
    return make_unexpected(std::move(subscriber).error());
  }
  auto received = session.impl_->receive_one(**subscriber);
  if (!received) {
    return make_unexpected(std::move(received).error());
  }
  if ((*subscriber)->process_local_transport) {
    session.impl_->record_process_local_receive();
  }
  return {};
}

expected<bool, Error> RuntimeChannelSessionTestAccess::blocked_head_masked(
    const RuntimeChannelSession& session, const PortId input) noexcept {
  if (!session.impl_) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto subscriber = cold_input_subscriber(*session.impl_, input);
  if (!subscriber) {
    return make_unexpected(std::move(subscriber).error());
  }
  const auto& mask = (*subscriber)->blocked_head_mask;
  return mask != nullptr && mask->load(std::memory_order_acquire);
}

expected<RuntimeChannelPumpWaitObservation, Error> RuntimeChannelSessionTestAccess::wait_as_pump(
    RuntimeChannelSession& session, const PortId input,
    const std::chrono::nanoseconds timeout) noexcept {
  if (!session.impl_ || session.impl_->pump_thread_.joinable() || !session.impl_->wait_set_) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto subscriber = cold_input_subscriber(*session.impl_, input);
  if (!subscriber) {
    return make_unexpected(std::move(subscriber).error());
  }
  const std::uint64_t handle = (*subscriber)->wait_handle;
  if (handle == 0U) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  auto waited = session.impl_->wait_set_->wait_for(timeout);
  if (!waited) {
    return make_unexpected(std::move(waited).error());
  }
  std::array<std::uint64_t, 64U> ready{};
  auto count = session.impl_->wait_set_->ready_handles(ready);
  if (!count) {
    return make_unexpected(std::move(count).error());
  }
  const auto reported = std::span{ready}.first(*count);
  session.impl_->mark_pump_reported_ready(reported);
  return RuntimeChannelPumpWaitObservation{
      .ready = *waited,
      .input_ready = std::ranges::find(reported, handle) != reported.end(),
  };
}

expected<void, Error> RuntimeChannelSessionTestAccess::enable_concurrent_admission(
    RuntimeChannelSession& session) noexcept {
  if (!session.impl_ || session.impl_->armed_.load(std::memory_order_acquire)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  session.impl_->concurrent_admission_ = true;
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_transport_head_blocked(
    RuntimeChannelSession& session, const PortId input, const bool blocked) noexcept {
  if (!session.impl_) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto subscriber = cold_input_subscriber(*session.impl_, input);
  if (!subscriber) {
    return make_unexpected(std::move(subscriber).error());
  }
  // Production sets the flag only on transport-inventory subscribers; keep tests to that state.
  if (!(*subscriber)->transport_inventory) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const std::scoped_lock transport_lock{(*subscriber)->transport_mutex};
  session.impl_->set_transport_head_blocked(**subscriber, blocked);
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::receive_end_of_stream_concurrently(
    RuntimeChannelSession& session, const PortId input,
    const std::uint64_t transport_sequence) noexcept {
  if (!session.impl_ || !input) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const std::scoped_lock transport_lock{(*subscriber)->transport_mutex};
  const std::scoped_lock cache_lock{(*subscriber)->mutex};
  return session.impl_->store_end_of_stream(**subscriber, transport_sequence);
}

expected<void, Error> RuntimeChannelSessionTestAccess::admit_recipients(
    RuntimeChannelSession& session, const std::span<const PortId> inputs) noexcept {
  if (!session.impl_) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  std::vector<RuntimeChannelSession::Impl::SubscriberState*> recipients;
  try {
    recipients.reserve(inputs.size());
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  for (const PortId input : inputs) {
    auto subscriber = cold_input_subscriber(*session.impl_, input);
    if (!subscriber) {
      return make_unexpected(std::move(subscriber).error());
    }
    recipients.push_back(*subscriber);
  }
  session.impl_->materialize_experimental_publication(
      RuntimeChannelSession::Impl::DrainScope::kRecipients, recipients, 0U);
  return session.impl_->status();
}

expected<void, Error> RuntimeChannelSessionTestAccess::retry_released_inputs(
    RuntimeChannelSession& session, const std::span<const PortId> inputs) noexcept {
  if (!session.impl_) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  std::vector<RuntimeChannelSession::Impl::SubscriberState*> released;
  try {
    released.reserve(inputs.size());
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  for (const PortId input : inputs) {
    auto subscriber = cold_input_subscriber(*session.impl_, input);
    if (!subscriber) {
      return make_unexpected(std::move(subscriber).error());
    }
    released.push_back(*subscriber);
  }
  return session.impl_->retry_blocked_process_local_transport(released);
}

bool RuntimeChannelSessionTestAccess::blocked_transport_retry_pending(
    const RuntimeChannelSession& session) noexcept {
  return session.impl_ && session.impl_->blocked_transport_retry_pending();
}

// Capture the complete cold-session queue transition for deterministic rate-observation tests.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> RuntimeChannelSessionTestAccess::receive_one_observed(
    RuntimeChannelSession& session, const RuntimeRateState& rates,
    const RuntimeRateSnapshotCapture capture, const PortId input, const EdgeId edge,
    const std::shared_ptr<RuntimeChannelQueueObservationTestControl>& control) noexcept {
  if (!session.impl_ || !input || !edge || control == nullptr || capture == nullptr ||
      session.impl_->armed_.load(std::memory_order_acquire) ||
      session.impl_->pump_thread_.joinable()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber = std::find_if(
      session.impl_->subscribers_.begin(),
      session.impl_->subscribers_.end(),
      [&](const auto& candidate) { return candidate->target == input && candidate->edge == edge; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }

  struct Inventory {
    std::uint32_t typed{};
    std::uint32_t staged{};
    std::uint32_t active_depth{};
    std::uint64_t next_arrival_sequence{};
    std::uint64_t terminal_sequence{};
    bool terminal_received{};
  };
  const auto inventory = [&]() -> expected<Inventory, Error> {
    const std::scoped_lock lock{(*subscriber)->mutex};
    const std::size_t typed =
        static_cast<std::size_t>(std::ranges::count_if((*subscriber)->slots, [](const auto& slot) {
          return slot.occupied && !slot.direct_ingress_unread;
        }));
    const std::size_t staged =
        static_cast<std::size_t>(std::ranges::count_if((*subscriber)->slots, [](const auto& slot) {
          return slot.occupied && slot.direct_ingress_unread;
        }));
    if (typed > std::numeric_limits<std::uint32_t>::max() ||
        staged > std::numeric_limits<std::uint32_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    return Inventory{
        .typed = static_cast<std::uint32_t>(typed),
        .staged = static_cast<std::uint32_t>(staged),
        .active_depth = (*subscriber)->active_typed_slot_capacity,
        .next_arrival_sequence = (*subscriber)->next_arrival_sequence,
        .terminal_sequence = (*subscriber)->end_of_stream_transport_sequence,
        .terminal_received = (*subscriber)->end_of_stream_received,
    };
  };

  auto before = inventory();
  auto live_before = capture(rates);
  const auto* published_before = live_before ? live_before->published() : nullptr;
  if (!before || !live_before || published_before == nullptr) {
    return make_unexpected(before ? Error{ErrorCode::kInvalidArgument} : std::move(before).error());
  }
  {
    const std::scoped_lock lock{control->mutex_};
    const auto& observation = control->observation_;
    if (!control->bound_ || control->rate_state_ != &rates ||
        live_before->plan_digest() != observation.plan_digest ||
        live_before->plan_generation() != observation.plan_generation ||
        live_before->run_generation() != observation.run_generation ||
        published_before->rate_generation != observation.rate_generation ||
        observation.plan_generation != session.impl_->plan_generation_ ||
        observation.run_generation != session.impl_->run_generation_ || observation.edge != edge ||
        before->active_depth != observation.selected_logical_depth ||
        !control->previous_occupancy_ || !control->previous_staged_occupancy_ ||
        !control->previous_arrival_sequence_ || !control->previous_terminal_sequence_ ||
        !control->previous_terminal_received_) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const Inventory previous{
        .typed = control->previous_occupancy_.value_or(0U),
        .staged = control->previous_staged_occupancy_.value_or(0U),
        .next_arrival_sequence = control->previous_arrival_sequence_.value_or(0U),
        .terminal_sequence = control->previous_terminal_sequence_.value_or(0U),
        .terminal_received = control->previous_terminal_received_.value_or(false),
    };
    if (previous.typed != before->typed || previous.staged != before->staged ||
        previous.next_arrival_sequence != before->next_arrival_sequence ||
        previous.terminal_sequence != before->terminal_sequence ||
        previous.terminal_received != before->terminal_received) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
  }

  auto received = session.impl_->receive_one(**subscriber);
  if (received && (*subscriber)->process_local_transport) {
    session.impl_->record_process_local_receive();
  }
  if (!received && received.error().code != ErrorCode::kBackpressured) {
    return make_unexpected(std::move(received).error());
  }

  auto after = inventory();
  auto live_after = capture(rates);
  const auto* published_after = live_after ? live_after->published() : nullptr;
  if (!after || !live_after || published_after == nullptr) {
    return make_unexpected(after ? Error{ErrorCode::kInvalidArgument} : std::move(after).error());
  }

  const std::scoped_lock lock{control->mutex_};
  auto& observation = control->observation_;
  if (!control->bound_ || control->rate_state_ != &rates ||
      live_after->plan_digest() != observation.plan_digest ||
      live_after->plan_generation() != observation.plan_generation ||
      live_after->run_generation() != observation.run_generation ||
      published_after->rate_generation != observation.rate_generation ||
      observation.plan_generation != session.impl_->plan_generation_ || observation.edge != edge ||
      observation.run_generation != session.impl_->run_generation_ ||
      after->active_depth != observation.selected_logical_depth || !control->previous_occupancy_ ||
      !control->previous_staged_occupancy_ || !control->previous_arrival_sequence_ ||
      !control->previous_terminal_sequence_ || !control->previous_terminal_received_) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  const Inventory previous{
      .typed = control->previous_occupancy_.value_or(0U),
      .staged = control->previous_staged_occupancy_.value_or(0U),
      .next_arrival_sequence = control->previous_arrival_sequence_.value_or(0U),
      .terminal_sequence = control->previous_terminal_sequence_.value_or(0U),
      .terminal_received = control->previous_terminal_received_.value_or(false),
  };
  if (previous.typed != before->typed || previous.staged != before->staged ||
      previous.next_arrival_sequence != before->next_arrival_sequence ||
      previous.terminal_sequence != before->terminal_sequence ||
      previous.terminal_received != before->terminal_received) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }

  if (received) {
    const bool data_arrived =
        before->next_arrival_sequence != std::numeric_limits<std::uint64_t>::max() &&
        after->next_arrival_sequence == before->next_arrival_sequence + 1U &&
        after->terminal_received == before->terminal_received &&
        after->terminal_sequence == before->terminal_sequence;
    const bool terminal_arrived =
        after->next_arrival_sequence == before->next_arrival_sequence &&
        after->typed == before->typed && after->staged == before->staged &&
        after->terminal_received && after->terminal_sequence != 0U &&
        (!before->terminal_received || after->terminal_sequence == before->terminal_sequence);
    if (!data_arrived && !terminal_arrived) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    if (data_arrived) {
      const bool appended = after->typed > before->typed && after->typed - before->typed == 1U &&
                            after->staged == before->staged;
      const bool staged = after->typed == before->typed && after->staged > before->staged &&
                          after->staged - before->staged == 1U;
      const bool replaced = after->typed == before->typed && after->staged == before->staged;
      if ((!appended && !staged && !replaced) ||
          (replaced && (*subscriber)->queue_policy != QueuePolicy::kDropOldest)) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      const std::uint64_t prior_replacements =
          observation.observed_replacement_or_drop_count.value_or(0U);
      if (replaced && prior_replacements == std::numeric_limits<std::uint64_t>::max()) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      observation.observed_backpressure_count =
          observation.observed_backpressure_count.value_or(0U);
      observation.observed_replacement_or_drop_count =
          prior_replacements + static_cast<std::uint64_t>(replaced);
    }
  } else {
    if (after->next_arrival_sequence != before->next_arrival_sequence ||
        after->typed != before->typed || after->staged != before->staged) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const std::uint64_t prior_backpressure = observation.observed_backpressure_count.value_or(0U);
    if (prior_backpressure == std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    observation.observed_backpressure_count = prior_backpressure + 1U;
    observation.observed_replacement_or_drop_count =
        observation.observed_replacement_or_drop_count.value_or(0U);
  }

  observation.observed_occupancy = after->typed;
  observation.observed_occupancy_high_water =
      std::max(observation.observed_occupancy_high_water.value_or(0U),
               std::max(before->typed, after->typed));
  control->previous_occupancy_ = after->typed;
  control->previous_staged_occupancy_ = after->staged;
  control->previous_arrival_sequence_ = after->next_arrival_sequence;
  control->previous_terminal_sequence_ = after->terminal_sequence;
  control->previous_terminal_received_ = after->terminal_received;
  if (!received) {
    return make_unexpected(std::move(received).error());
  }
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_observation_target_alias(
    RuntimeChannelSession& session, const EdgeId edge, const PortId target) noexcept {
  if (!session.impl_ || !edge || !target || session.impl_->armed_.load(std::memory_order_acquire) ||
      session.impl_->pump_thread_.joinable()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber = std::ranges::find_if(
      session.impl_->subscribers_, [&](const auto& candidate) { return candidate->edge == edge; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const std::scoped_lock lock{(*subscriber)->transport_mutex, (*subscriber)->mutex};
  (*subscriber)->target = target;
  return {};
}

expected<std::size_t, Error> RuntimeChannelSessionTestAccess::typed_ingress_reservations(
    const RuntimeChannelSession& session, const PortId input) noexcept {
  if (!session.impl_ || !input) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const std::scoped_lock lock{(*subscriber)->mutex};
  return static_cast<std::size_t>(
      std::count_if((*subscriber)->slots.begin(), (*subscriber)->slots.end(), [](const auto& slot) {
        return slot.typed_ingress_reserved;
      }));
}

expected<void, Error> RuntimeChannelSessionTestAccess::reclaim_reserved_input_slot(
    RuntimeChannelSession& session, const PortId input) noexcept {
  if (!session.impl_ || !input) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  RuntimeChannelSession::Impl::DeferredSampleReleases releases{*(*subscriber)->input->type_ops, 1U};
  const std::scoped_lock lock{(*subscriber)->mutex};
  RuntimeChannelSession::Impl::SampleSlot* reserved = nullptr;
  for (auto& slot : (*subscriber)->slots) {
    if (!slot.typed_ingress_reserved) {
      continue;
    }
    if (reserved != nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    reserved = &slot;
  }
  if (reserved == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  RuntimeChannelSession::Impl::reclaim_sample_slot_if_unclaimed(*reserved, releases);
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_typed_ingress_observer(
    RuntimeChannelSession& session, const PortId input, void* const state,
    void (*const observer)(void*) noexcept) noexcept {
  if (!session.impl_ || (state == nullptr) != (observer == nullptr)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto subscriber = cold_input_subscriber(*session.impl_, input);
  if (!subscriber) {
    return make_unexpected(std::move(subscriber).error());
  }
  const std::scoped_lock lock{(*subscriber)->transport_mutex, (*subscriber)->mutex};
  (*subscriber)->typed_ingress_observer_state = state;
  (*subscriber)->typed_ingress_observer = observer;
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::with_reuse_replacement_reservation(
    RuntimeChannelSession& session, const PortId input, void* const state,
    void (*callback)(void*) noexcept) noexcept {
  if (!session.impl_ || !input || callback == nullptr ||
      session.impl_->armed_.load(std::memory_order_acquire) ||
      session.impl_->pump_thread_.joinable()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  std::uint64_t successor_sequence = 0U;
  {
    const std::scoped_lock lock{(*subscriber)->mutex};
    const auto* const retained = (*subscriber)->input->reuse_slot.load(std::memory_order_acquire);
    if (retained == nullptr || retained->subscriber != subscriber->get() ||
        retained->temporal.source_sequence == std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    successor_sequence = retained->temporal.source_sequence + 1U;
  }
  auto reservation = session.impl_->reserve_staged_newer_data_replacement(
      **subscriber,
      std::nullopt,
      channel_internal::StagedDataState{.sequence = successor_sequence});
  if (!reservation) {
    return make_unexpected(Error{ErrorCode::kNotReady});
  }
  callback(state);
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::fail_next_reuse_replacement_receive(
    RuntimeChannelSession& session, const PortId input) noexcept {
  if (!session.impl_ || !input) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const std::scoped_lock lock{(*subscriber)->mutex};
  (*subscriber)->fail_next_reuse_replacement_receive_for_test = true;
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::stop_tensor_ingress_staging(
    RuntimeChannelSession& session, const PortId input) noexcept {
  if (!session.impl_ || !input) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const std::scoped_lock lock{(*subscriber)->mutex};
  if (!(*subscriber)->tensor_ingress_staging) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  (*subscriber)->tensor_ingress_staging = false;
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::reject_next_reuse_replacement_receive(
    RuntimeChannelSession& session, const PortId input, const ErrorCode error) noexcept {
  if (!session.impl_ || !input) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (error != ErrorCode::kBackpressured && error != ErrorCode::kNotReady) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const std::scoped_lock lock{(*subscriber)->mutex};
  (*subscriber)->retryable_reuse_replacement_receive_error_for_test = error;
  return {};
}

expected<std::optional<std::uint64_t>, Error> RuntimeChannelSessionTestAccess::staged_data_sequence(
    const RuntimeChannelSession& session, const PortId input) noexcept {
  if (!session.impl_ || !input) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const RuntimeChannelSession::Impl::InputPortState* const port = session.impl_->find_input(input);
  if (port == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  if (port->subscribers.size() != 1U || port->subscribers.front() == nullptr) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  RuntimeChannelSession::Impl::SubscriberState& subscriber = *port->subscribers.front();
  const std::scoped_lock transport_lock{subscriber.transport_mutex};
  const auto staged = subscriber.staged_data_state();
  return staged ? std::optional<std::uint64_t>{staged->sequence} : std::nullopt;
}

expected<void, Error> RuntimeChannelSessionTestAccess::discard_staged_data_head(
    RuntimeChannelSession& session, const PortId input) noexcept {
  if (!session.impl_ || !input) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto subscriber = cold_input_subscriber(*session.impl_, input);
  if (!subscriber) {
    return make_unexpected(std::move(subscriber).error());
  }
  auto& state = **subscriber;
  const std::scoped_lock transport_lock{state.transport_mutex};
  if (!state.process_local_transport || state.externally_driven_import ||
      state.endpoint == nullptr || !state.staged_data_state()) {
    return make_unexpected(Error{ErrorCode::kNotReady});
  }
  auto discarded = state.discard_one_process_local_transport(nullptr);
  if (!discarded) {
    return make_unexpected(std::move(discarded).error());
  }
  session.impl_->record_process_local_receive();
  // Inline HoloMQ materializes staged identity lazily. Rejecting data stages the next exact head
  // without consuming it, still under the serialization boundary that discarded its predecessor.
  auto next = state.endpoint->receive_borrowed_if_data_accepted(false);
  if (next || next.error().code != ErrorCode::kBackpressured || !state.staged_data_state()) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::fail_next_freshness_handoff_retry(
    RuntimeChannelSession& session) noexcept {
  if (!session.impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  session.impl_->fail_next_freshness_handoff_retry_.store(true, std::memory_order_release);
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::drain_external_inputs(
    RuntimeChannelSession& session, const VertexId owner) noexcept {
  if (!session.impl_ || !owner || session.impl_->armed_.load(std::memory_order_acquire) ||
      session.impl_->pump_thread_.joinable()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const RuntimeChannelSession::Impl::ActivationOwnerState* const activation_owner =
      session.impl_->find_activation_owner(owner);
  if (activation_owner == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  return session.impl_->drain_monitored_inputs(*activation_owner);
}

expected<void, Error> RuntimeChannelSessionTestAccess::store_end_of_stream(
    RuntimeChannelSession& session, const PortId input,
    const std::uint64_t transport_sequence) noexcept {
  if (!session.impl_) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto subscriber = cold_input_subscriber(*session.impl_, input);
  if (!subscriber) {
    return make_unexpected(std::move(subscriber).error());
  }
  const std::scoped_lock lock{(*subscriber)->mutex};
  return session.impl_->store_end_of_stream(**subscriber, transport_sequence);
}

expected<void, Error>
RuntimeChannelSessionTestAccess::corrupt_next_process_local_schema_publication(
    RuntimeChannelSession& session, const PortId output,
    const channel_internal::CatalogSchemaFrameCorruption corruption) noexcept {
  if (!session.impl_ || !output || session.impl_->armed_.load(std::memory_order_acquire)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto found = std::find_if(session.impl_->outputs_.begin(),
                                  session.impl_->outputs_.end(),
                                  [&](const auto& candidate) { return candidate->port == output; });
  if (found == session.impl_->outputs_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  auto& state = **found;
  if (state.active || !state.process_local_schema_identity ||
      state.next_process_local_schema_corruption) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  state.next_process_local_schema_corruption = corruption;
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_readiness_sequence(
    PortBinding& binding, const std::uint64_t sequence) noexcept {
  auto* const readiness = dynamic_cast<RuntimeChannelSession::Impl::ReadinessBinding*>(&binding);
  if (readiness == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  readiness->next_source_sequence_ = sequence;
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_single_occupied_witness_state(
    RuntimeChannelSession& session, const PortId input, const std::uint64_t readiness_sequence,
    const bool event_emitted, const bool witness_retired) noexcept {
  if (!session.impl_ || !input || readiness_sequence == 0U || (witness_retired && !event_emitted)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const std::scoped_lock lock{(*subscriber)->mutex};
  RuntimeChannelSession::Impl::SampleSlot* occupied = nullptr;
  for (auto& slot : (*subscriber)->slots) {
    if (!slot.occupied || slot.direct_ingress_unread) {
      continue;
    }
    if (occupied != nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    occupied = &slot;
  }
  if (occupied == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  occupied->readiness_sequence = readiness_sequence;
  occupied->event_emitted = event_emitted;
  occupied->witness_retired = witness_retired;
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_pre_witness_replacement_allowed(
    RuntimeChannelSession& session, const PortId input, const bool allowed) noexcept {
  if (!session.impl_ || !input || session.impl_->armed_.load(std::memory_order_acquire) ||
      session.impl_->pump_thread_.joinable()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const std::scoped_lock lock{(*subscriber)->mutex};
  (*subscriber)->pre_witness_replacement_allowed = allowed;
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_single_occupied_claim_state(
    RuntimeChannelSession& session, const PortId input, const bool active_selected,
    const bool reuse_retained, const bool temporal_retained,
    const bool retirement_selected) noexcept {
  if (!session.impl_ || !input) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const std::scoped_lock lock{(*subscriber)->mutex};
  RuntimeChannelSession::Impl::SampleSlot* occupied = nullptr;
  for (auto& slot : (*subscriber)->slots) {
    if (!slot.occupied || slot.direct_ingress_unread) {
      continue;
    }
    if (occupied != nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    occupied = &slot;
  }
  if (occupied == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  occupied->active_selected = active_selected;
  occupied->reuse_retained = reuse_retained;
  occupied->temporal_retained = temporal_retained;
  occupied->retirement_selected = retirement_selected;
  (*subscriber)
      ->input->reuse_slot.store(reuse_retained ? occupied : nullptr, std::memory_order_release);
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::make_staged_sample_older_than_typed_sample(
    RuntimeChannelSession& session, const PortId input) noexcept {
  if (!session.impl_ || !input || session.impl_->armed_.load(std::memory_order_acquire) ||
      session.impl_->pump_thread_.joinable()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const std::scoped_lock lock{(*subscriber)->mutex};
  RuntimeChannelSession::Impl::SampleSlot* typed = nullptr;
  RuntimeChannelSession::Impl::SampleSlot* staged = nullptr;
  for (auto& slot : (*subscriber)->slots) {
    if (!slot.occupied) {
      continue;
    }
    auto*& selected = slot.direct_ingress_unread ? staged : typed;
    if (selected != nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    selected = &slot;
  }
  if (typed == nullptr || staged == nullptr || typed->event_emitted || staged->event_emitted ||
      typed->typed_ingress_reserved || staged->typed_ingress_reserved ||
      staged->staged_arrival_evidence || staged->refill_reserved ||
      staged->refill_prefix_reserved ||
      typed->temporal.producer_epoch != staged->temporal.producer_epoch ||
      !is_strictly_newer_sample(staged->temporal, typed->temporal)) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  std::swap(typed->sample, staged->sample);
  std::swap(typed->temporal, staged->temporal);
  std::swap(typed->clock_health_disposition, staged->clock_health_disposition);
  std::swap(typed->clock_health_generation, staged->clock_health_generation);
  typed->arrival_order = typed->temporal.arrival_sequence;
  (*subscriber)->last_producer_sequence = typed->temporal.source_sequence;
  (*subscriber)->last_direct_ingress_sequence = staged->temporal.source_sequence;
  return {};
}

expected<void, Error>
RuntimeChannelSessionTestAccess::commit_staged_reservation_after_newer_typed_overtake(
    RuntimeChannelSession& session, const PortId input,
    const bool staged_is_committed_freshness_target) noexcept {
  if (!session.impl_ || !input || session.impl_->armed_.load(std::memory_order_acquire) ||
      session.impl_->pump_thread_.joinable()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  RuntimeChannelSession::Impl::DeferredSampleReleases releases{*(*subscriber)->input->type_ops,
                                                               (*subscriber)->slots.size() + 1U};
  std::unique_lock lock{(*subscriber)->mutex};
  RuntimeChannelSession::Impl::SampleSlot* typed = nullptr;
  RuntimeChannelSession::Impl::SampleSlot* staged = nullptr;
  for (auto& slot : (*subscriber)->slots) {
    if (!slot.occupied) {
      continue;
    }
    auto*& selected = slot.direct_ingress_unread ? staged : typed;
    if (selected != nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    selected = &slot;
  }
  if (typed == nullptr || staged == nullptr || typed->event_emitted || staged->event_emitted ||
      typed->typed_ingress_reserved || staged->typed_ingress_reserved || typed->active_selected ||
      typed->reuse_retained || typed->temporal_retained || typed->retirement_selected ||
      staged->staged_arrival_evidence || staged->refill_reserved ||
      staged->refill_prefix_reserved ||
      typed->temporal.producer_epoch != staged->temporal.producer_epoch ||
      !is_strictly_newer_sample(typed->temporal, staged->temporal)) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }

  auto& input_state = *(*subscriber)->input;
  if (staged_is_committed_freshness_target) {
    if (input_state.freshness_refill_target != nullptr ||
        input_state.freshness_reservation_generation != 0U) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    staged->refill_reserved = true;
    input_state.freshness_refill_target = staged;
    input_state.freshness_refill_target_reference = TimePoint{1, input_state.reference_clock};
    input_state.freshness_refill_prefix_committed = true;
  }
  const auto rollback_freshness_target = [&]() noexcept {
    if (!staged_is_committed_freshness_target) {
      return;
    }
    staged->refill_reserved = false;
    input_state.freshness_refill_target = nullptr;
    input_state.freshness_refill_target_reference = {};
    input_state.freshness_refill_prefix_committed = false;
  };

  staged->typed_ingress_reserved = true;
  RuntimeChannelSession::Impl::TypedIngressReservation reservation{**subscriber, *staged, true};
  auto deferred = session.impl_->promote_direct_ingress(**subscriber, releases);
  if (!deferred) {
    Error error = std::move(deferred).error();
    rollback_freshness_target();
    lock.unlock();
    return make_unexpected(std::move(error));
  }
  if (*deferred != 0U || !typed->occupied || !staged->occupied || !staged->direct_ingress_unread ||
      !staged->typed_ingress_reserved) {
    rollback_freshness_target();
    lock.unlock();
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  auto committed =
      session.impl_->commit_typed_ingress_reservation(**subscriber, reservation, releases);
  if (!committed) {
    rollback_freshness_target();
    return committed;
  }
  if (staged_is_committed_freshness_target &&
      (input_state.freshness_refill_target != nullptr ||
       input_state.freshness_refill_target_reference.valid() ||
       input_state.freshness_refill_prefix_committed)) {
    rollback_freshness_target();
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_freshness_subscriber_alias(
    RuntimeChannelSession& session, const PortId input, const PortId subscriber_input,
    const bool include) noexcept {
  if (!session.impl_ || !input || !subscriber_input || input == subscriber_input ||
      session.impl_->armed_.load(std::memory_order_acquire) ||
      session.impl_->pump_thread_.joinable()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto* const destination = session.impl_->find_input(input);
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == subscriber_input; });
  if (destination == nullptr || subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const auto alias = std::find(
      destination->subscribers.begin(), destination->subscribers.end(), subscriber->get());
  if (include) {
    if (alias != destination->subscribers.end()) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    destination->subscribers.push_back(subscriber->get());
  } else {
    if (alias == destination->subscribers.end()) {
      return make_unexpected(Error{ErrorCode::kNotFound});
    }
    destination->subscribers.erase(alias);
  }
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::saturate_single_occupied_reset_state(
    RuntimeChannelSession& session, const PortId input) noexcept {
  if (!session.impl_ || !input || session.impl_->armed_.load(std::memory_order_acquire) ||
      session.impl_->pump_thread_.joinable()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const std::scoped_lock lock{(*subscriber)->mutex};
  const auto occupied =
      std::find_if((*subscriber)->slots.begin(), (*subscriber)->slots.end(), [](const auto& slot) {
        return slot.occupied && !slot.direct_ingress_unread;
      });
  if (occupied == (*subscriber)->slots.end() ||
      std::any_of(std::next(occupied), (*subscriber)->slots.end(), [](const auto& slot) {
        return slot.occupied && !slot.direct_ingress_unread;
      })) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  occupied->event_emitted = true;
  occupied->witness_retired = true;
  occupied->active_selected = true;
  occupied->reuse_retained = true;
  occupied->temporal_retained = true;
  occupied->retirement_selected = true;
  occupied->typed_ingress_reserved = true;
  occupied->readiness_sequence = 17U;
  occupied->arrival_order = 19U;
  occupied->clock_health_disposition = EndpointClockHealthDisposition::kDegraded;
  occupied->clock_health_generation = 23U;
  occupied->direct_ingress_unread = true;
  occupied->refill_reserved = true;
  occupied->owed_to_released_claim = true;
  occupied->temporal = TemporalSampleRecord{
      .source_sequence = 29U,
      .producer_epoch = 31U,
      .arrival_sequence = 37U,
  };
  (*subscriber)->next_arrival_sequence = 41U;
  (*subscriber)->last_producer_sequence = 43U;
  (*subscriber)->last_direct_ingress_sequence = 47U;
  (*subscriber)->end_of_stream_arrival_order = 53U;
  (*subscriber)->end_of_stream_transport_sequence = 59U;
  (*subscriber)->end_of_stream_received = true;
  (*subscriber)->end_of_stream_event_emitted = true;
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::inject_direct_inline_payload(
    RuntimeChannelSession& session, const PortId input, const std::span<const std::byte> payload,
    SampleMetadata metadata) noexcept {
  if (!session.impl_ || !input || payload.empty() ||
      session.impl_->armed_.load(std::memory_order_acquire) ||
      session.impl_->pump_thread_.joinable()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  return session.impl_->retain_direct_inline_payload(**subscriber, payload, std::move(metadata));
}

expected<void, Error> RuntimeChannelSessionTestAccess::with_input_lock(
    RuntimeChannelSession& session, const PortId input, void* state,
    const RuntimeChannelInputLockObserver observer) noexcept {
  if (!session.impl_ || !input || observer == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const std::scoped_lock lock{(*subscriber)->mutex};
  observer(state);
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::with_pump_scan_lock(
    RuntimeChannelSession& session, void* state,
    const RuntimeChannelInputLockObserver observer) noexcept {
  if (!session.impl_ || observer == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const std::scoped_lock lock{session.impl_->pump_scan_mutex_};
  observer(state);
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::receive_as_pump(
    RuntimeChannelSession& session) noexcept {
  if (!session.impl_) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  PortId failed_port;
  VertexId failed_owner;
  std::optional<StructuralDiagnosticId> terminal_diagnostic;
  auto drained =
      session.impl_->drain_unmonitored_inputs(failed_port,
                                              failed_owner,
                                              terminal_diagnostic,
                                              RuntimeChannelSession::Impl::DrainScope::kPumpScan);
  if (!drained) {
    return make_unexpected(std::move(drained).error());
  }
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_input_poll_contention_observer(
    RuntimeChannelSession& session, const PortId input, void* const state,
    const RuntimeChannelInputPollContentionObserver observer) noexcept {
  if (!session.impl_ || !input || state == nullptr || observer == nullptr ||
      session.impl_->armed_.load(std::memory_order_acquire)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end() ||
      std::any_of(std::next(subscriber),
                  session.impl_->subscribers_.end(),
                  [&](const auto& candidate) { return candidate->target == input; }) ||
      (*subscriber)->input_poll_contention_observer != nullptr) {
    return make_unexpected(Error{subscriber == session.impl_->subscribers_.end()
                                     ? ErrorCode::kNotFound
                                     : ErrorCode::kProtocolError});
  }
  (*subscriber)->input_poll_contention_observer_state = state;
  (*subscriber)->input_poll_contention_observer = observer;
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_input_contention_retry_attempt_observer(
    RuntimeChannelSession& session, const PortId input, void* const state,
    const RuntimeChannelInputPollContentionObserver observer) noexcept {
  if (!session.impl_ || !input || state == nullptr || observer == nullptr ||
      session.impl_->armed_.load(std::memory_order_acquire)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end() ||
      std::any_of(std::next(subscriber),
                  session.impl_->subscribers_.end(),
                  [&](const auto& candidate) { return candidate->target == input; }) ||
      (*subscriber)->input_contention_retry_attempt_observer != nullptr) {
    return make_unexpected(Error{subscriber == session.impl_->subscribers_.end()
                                     ? ErrorCode::kNotFound
                                     : ErrorCode::kProtocolError});
  }
  (*subscriber)->input_contention_retry_attempt_observer_state = state;
  (*subscriber)->input_contention_retry_attempt_observer = observer;
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_contention_retry_budget(
    RuntimeChannelSession& session, const std::chrono::nanoseconds budget) noexcept {
  if (!session.impl_ || budget <= std::chrono::nanoseconds::zero() ||
      session.impl_->armed_.load(std::memory_order_acquire)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  session.impl_->contention_retry_budget_ = budget;
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_poll_selection_observer(
    RuntimeChannelSession& session, void* state,
    const RuntimeChannelPollSelectionObserver observer) noexcept {
  if (!session.impl_ || state == nullptr || observer == nullptr ||
      session.impl_->poll_selection_observer_ != nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  session.impl_->poll_selection_observer_state_ = state;
  session.impl_->poll_selection_observer_ = observer;
  return {};
}

expected<std::string, Error> RuntimeChannelSessionTestAccess::process_local_transport_name(
    const RuntimeChannelSession& session, const std::string_view service_id) {
  if (!session.impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return session.impl_->process_local_transport_name(service_id);
}

std::string RuntimeChannelSessionTestAccess::process_local_transport_scope(
    const std::uint64_t process, const std::uint64_t ordinal, const std::uint64_t incarnation) {
  return RuntimeChannelSession::Impl::process_local_transport_scope(process, ordinal, incarnation);
}

expected<void, Error> RuntimeChannelSessionTestAccess::fail_next_activation_retirement(
    RuntimeChannelSession& session) noexcept {
  return session.fail_next_activation_retirement_for_test();
}

void RuntimeChannelSessionTestAccess::fail(RuntimeChannelSession& session,
                                           const ErrorCode code) noexcept {
  if (session.impl_) {
    session.impl_->fail(code);
  }
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_activation_retirement_observer(
    RuntimeChannelSession& session, void* const state,
    const RuntimeChannelActivationRetirementObserver observer) noexcept {
  if (!session.impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return session.impl_->set_activation_retirement_observer_for_test(state, observer);
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_teardown_observer(
    RuntimeChannelSession& session, void* state,
    const RuntimeChannelTeardownObserver observer) noexcept {
  return session.set_teardown_observer_for_test(state, observer);
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_clock_health_lock_observer(
    RuntimeChannelSession& session, void* state,
    const RuntimeChannelClockHealthLockObserver observer) noexcept {
  if (!session.impl_ || state == nullptr || observer == nullptr ||
      session.impl_->clock_health_serialization_mutex_ == nullptr ||
      session.impl_->armed_.load(std::memory_order_acquire) ||
      session.impl_->pump_thread_.joinable() ||
      session.impl_->clock_health_lock_observer_ != nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  session.impl_->clock_health_lock_observer_state_ = state;
  session.impl_->clock_health_lock_observer_ = observer;
  return {};
}

bool RuntimeChannelSessionTestAccess::clock_health_serialization_bound(
    const RuntimeChannelSession& session) noexcept {
  return session.impl_ != nullptr && session.impl_->clock_health_serialization_mutex_ != nullptr;
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_input_reset_counters(
    RuntimeChannelSession& session, const PortId input, const std::uint64_t active_generation,
    const std::uint64_t view_generation, const std::uint64_t producer_epoch) noexcept {
  if (!session.impl_ || !input || active_generation == 0U || view_generation == 0U ||
      producer_epoch == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto* input_state = session.impl_->find_input(input);
  if (input_state == nullptr || input_state->active || input_state->predicate_view_constructed ||
      session.impl_->pump_thread_.joinable() ||
      session.impl_->armed_.load(std::memory_order_acquire)) {
    return make_unexpected(
        Error{input_state == nullptr ? ErrorCode::kNotFound : ErrorCode::kBackpressured});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  input_state->active_generation = active_generation;
  input_state->view_generation = view_generation;
  (*subscriber)->producer_epoch = producer_epoch;
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_arrival_counters(
    RuntimeChannelSession& session, const PortId input, const std::uint64_t next_arrival_sequence,
    const std::uint64_t next_global_arrival_order) noexcept {
  if (!session.impl_ || !input || next_arrival_sequence == 0U || next_global_arrival_order == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto* input_state = session.impl_->find_input(input);
  if (input_state == nullptr || input_state->active || input_state->predicate_view_constructed ||
      session.impl_->pump_thread_.joinable() ||
      session.impl_->armed_.load(std::memory_order_acquire)) {
    return make_unexpected(
        Error{input_state == nullptr ? ErrorCode::kNotFound : ErrorCode::kBackpressured});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  {
    const std::scoped_lock lock{(*subscriber)->mutex};
    (*subscriber)->next_arrival_sequence = next_arrival_sequence;
  }
  session.impl_->next_arrival_order_.store(next_global_arrival_order, std::memory_order_release);
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_process_local_progress_counts(
    RuntimeChannelSession& session, const std::uint64_t published,
    const std::uint64_t received) noexcept {
  if (!session.impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  session.impl_->process_local_published_deliveries_.store(published, std::memory_order_release);
  session.impl_->process_local_received_deliveries_.store(received, std::memory_order_release);
  return {};
}

expected<RuntimeChannelProcessLocalProgressState, Error>
RuntimeChannelSessionTestAccess::process_local_progress_state(
    const RuntimeChannelSession& session) noexcept {
  if (!session.impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return RuntimeChannelProcessLocalProgressState{
      .published_deliveries =
          session.impl_->process_local_published_deliveries_.load(std::memory_order_acquire),
      .received_deliveries =
          session.impl_->process_local_received_deliveries_.load(std::memory_order_acquire),
      .replaced_deliveries =
          session.impl_->process_local_replaced_deliveries_.load(std::memory_order_acquire),
      .discarded_deliveries =
          session.impl_->process_local_discarded_deliveries_.load(std::memory_order_acquire),
      .publications_in_flight =
          session.impl_->process_local_publications_in_flight_.load(std::memory_order_acquire),
      .tracking_valid =
          session.impl_->process_local_progress_tracking_valid_.load(std::memory_order_acquire),
  };
}

expected<void, Error> RuntimeChannelSessionTestAccess::inject_received_metadata(
    RuntimeChannelSession& session, const PortId input, const SampleMetadata metadata,
    std::optional<StructuralDiagnosticId>* const terminal_diagnostic) noexcept {
  if (!session.impl_ || !input || session.impl_->pump_thread_.joinable() ||
      session.impl_->armed_.load(std::memory_order_acquire)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  RuntimeChannelSession::Impl::DeferredSampleReleases releases{*(*subscriber)->input->type_ops,
                                                               (*subscriber)->slots.size() + 1U};
  const std::scoped_lock lock{(*subscriber)->mutex};
  const auto scratch = (*subscriber)->scratch_bytes();
  std::fill(scratch.begin(), scratch.end(), std::byte{});
  return session.impl_->store_received(**subscriber,
                                       releases,
                                       metadata,
                                       EndpointClockHealthDisposition::kNotApplicable,
                                       0U,
                                       {},
                                       terminal_diagnostic);
}

expected<void, Error> RuntimeChannelSessionTestAccess::inject_received_tensor(
    RuntimeChannelSession& session, const PortId input, Sample<Tensor> sample) noexcept {
  if (!session.impl_ || !input || session.impl_->pump_thread_.joinable() ||
      session.impl_->armed_.load(std::memory_order_acquire)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  if ((*subscriber)->input->type_ops != runtime_port_type_ops<Tensor>()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  RuntimeChannelSession::Impl::DeferredSampleReleases releases{*(*subscriber)->input->type_ops,
                                                               (*subscriber)->slots.size() + 1U};
  const std::scoped_lock lock{(*subscriber)->mutex};
  return session.impl_->store_received_tensor(**subscriber, releases, std::move(sample));
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_external_tensor_contract(
    RuntimeChannelSession& session, const PortId input,
    TensorBoundaryContractRecord requirements) noexcept {
  if (!session.impl_ || !input || !requirements.max_byte_span ||
      *requirements.max_byte_span == 0U || requirements.memory_kind != MemoryKind::kHost ||
      requirements.strides != TensorStridePolicy::kContiguousOnly ||
      session.impl_->pump_thread_.joinable() ||
      session.impl_->armed_.load(std::memory_order_acquire)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  if ((*subscriber)->input->type_ops != runtime_port_type_ops<Tensor>()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const std::scoped_lock lock{(*subscriber)->mutex};
  if (std::any_of((*subscriber)->slots.begin(), (*subscriber)->slots.end(), [](const auto& slot) {
        return slot.occupied;
      })) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  (*subscriber)->externally_driven_import = false;
  (*subscriber)->catalog_tensor_import = true;
  (*subscriber)->tensor_max_byte_span = *requirements.max_byte_span;
  (*subscriber)->tensor_memory_kind = MemoryKind::kHost;
  (*subscriber)->tensor_device = {};
  (*subscriber)->tensor_logical_requirements = std::move(requirements);
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_external_metadata_contract(
    RuntimeChannelSession& session, const PortId input, const ClockId catalog_capture_clock,
    const ClockId plan_capture_clock) noexcept {
  if (!session.impl_ || !input || catalog_capture_clock.valid() != plan_capture_clock.valid() ||
      session.impl_->pump_thread_.joinable() ||
      session.impl_->armed_.load(std::memory_order_acquire)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const std::scoped_lock lock{(*subscriber)->mutex};
  if (std::any_of((*subscriber)->slots.begin(), (*subscriber)->slots.end(), [](const auto& slot) {
        return slot.occupied;
      })) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  if ((*subscriber)->input->capture_time_required && !catalog_capture_clock.valid()) {
    // Production admission supplies a concrete catalog/plan clock pair whenever a selector
    // requires capture time. Keep this cold test seam from manufacturing an unreachable plan.
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  (*subscriber)->externally_driven_import = true;
  (*subscriber)->catalog_capture_clock = catalog_capture_clock;
  (*subscriber)->plan_capture_clock = plan_capture_clock;
  return {};
}

expected<RuntimeChannelSessionTestAccess::InputResourceState, Error>
RuntimeChannelSessionTestAccess::input_resource_state(const RuntimeChannelSession& session,
                                                      const PortId input) noexcept {
  if (!session.impl_ || !input || session.impl_->pump_thread_.joinable() ||
      session.impl_->armed_.load(std::memory_order_acquire)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto* input_state = session.impl_->find_input(input);
  if (input_state == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  InputResourceState result{
      .subscriber_count = input_state->subscribers.size(),
      .claim_pool_present = input_state->claim_pool != nullptr,
  };
  for (const auto* subscriber : input_state->subscribers) {
    result.payload_slot_count += subscriber->slots.size();
  }
  return result;
}

expected<RuntimeScalarReadBinding, Error> RuntimeChannelSessionTestAccess::selected_scalar_binding(
    const RuntimeChannelSession& session, const PortId input) noexcept {
  if (!session.impl_ || !input) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto* state = session.impl_->find_input(input);
  if (state == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  return state->binding.ops->scalar(state->binding.state, state->binding.generation, false);
}

expected<RuntimeChannelInputResetState, Error> RuntimeChannelSessionTestAccess::input_reset_state(
    const RuntimeChannelSession& session, const PortId input) noexcept {
  if (!session.impl_ || !input) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto* input_state = session.impl_->find_input(input);
  if (input_state == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const std::scoped_lock lock{(*subscriber)->mutex};
  std::uint64_t latched_producer_generation = 0U;
  if ((*subscriber)->endpoint != nullptr) {
    auto latched = channel_internal::CatalogChannelBindingAccess::latched_producer_generation(
        *(*subscriber)->endpoint);
    if (latched) {
      latched_producer_generation = *latched;
    }
  }
  const auto* const reuse_slot = (*subscriber)->input->reuse_slot.load(std::memory_order_acquire);
  return RuntimeChannelInputResetState{
      .active_generation = input_state->active_generation,
      .view_generation = input_state->view_generation,
      .producer_epoch = (*subscriber)->producer_epoch,
      .next_arrival_sequence = (*subscriber)->next_arrival_sequence,
      .next_global_arrival_order =
          session.impl_->next_arrival_order_.load(std::memory_order_acquire),
      .last_producer_sequence = (*subscriber)->last_producer_sequence,
      .end_of_stream_transport_sequence = (*subscriber)->end_of_stream_transport_sequence,
      .reuse_retained = std::any_of((*subscriber)->slots.begin(),
                                    (*subscriber)->slots.end(),
                                    [](const auto& slot) { return slot.reuse_retained; }),
      .temporal_retained = std::any_of((*subscriber)->slots.begin(),
                                       (*subscriber)->slots.end(),
                                       [](const auto& slot) { return slot.temporal_retained; }),
      .reuse_slot_matches_single_occupied =
          reuse_slot != nullptr && reuse_slot->subscriber == subscriber->get() &&
          reuse_slot->occupied &&
          std::count_if(
              (*subscriber)->slots.begin(),
              (*subscriber)->slots.end(),
              [](const auto& slot) { return slot.occupied && !slot.direct_ingress_unread; }) == 1,
      .occupied_samples = static_cast<std::size_t>(std::count_if(
          (*subscriber)->slots.begin(),
          (*subscriber)->slots.end(),
          [](const auto& slot) { return slot.occupied && !slot.direct_ingress_unread; })),
      .retired_temporal_samples = static_cast<std::size_t>(std::count_if(
          (*subscriber)->slots.begin(),
          (*subscriber)->slots.end(),
          [](const auto& slot) { return slot.temporal_retained && slot.witness_retired; })),
      .pending_direct_ingress_samples = static_cast<std::size_t>(std::count_if(
          (*subscriber)->slots.begin(),
          (*subscriber)->slots.end(),
          [](const auto& slot) { return slot.occupied && slot.direct_ingress_unread; })),
      .uncleared_sample_slots = static_cast<std::size_t>(std::count_if(
          (*subscriber)->slots.begin(),
          (*subscriber)->slots.end(),
          [](const auto& slot) {
            return slot.occupied || slot.event_emitted || slot.witness_retired ||
                   slot.active_selected || slot.reuse_retained || slot.temporal_retained ||
                   slot.retirement_selected || slot.typed_ingress_reserved ||
                   slot.readiness_sequence != 0U || slot.arrival_order != 0U ||
                   slot.clock_health_disposition !=
                       EndpointClockHealthDisposition::kNotApplicable ||
                   slot.clock_health_generation != 0U || slot.direct_ingress_unread ||
                   slot.refill_reserved || slot.refill_prefix_reserved ||
                   slot.owed_to_released_claim || slot.temporal != TemporalSampleRecord{};
          })),
      .catalog_capture_clock = (*subscriber)->catalog_capture_clock,
      .plan_capture_clock = (*subscriber)->plan_capture_clock,
      .latched_producer_generation = latched_producer_generation,
      .end_of_stream_received = (*subscriber)->end_of_stream_received,
      .subscriber_sequence_state_reset = (*subscriber)->next_arrival_sequence == 1U &&
                                         (*subscriber)->last_producer_sequence == 0U &&
                                         (*subscriber)->last_direct_ingress_sequence == 0U &&
                                         (*subscriber)->end_of_stream_arrival_order == 0U &&
                                         (*subscriber)->end_of_stream_transport_sequence == 0U &&
                                         !(*subscriber)->end_of_stream_received &&
                                         !(*subscriber)->end_of_stream_event_emitted,
  };
}

expected<RuntimeChannelPumpIndexState, Error> RuntimeChannelSessionTestAccess::pump_index_state(
    const RuntimeChannelSession& session) noexcept {
  if (!session.impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  const bool wait_index_exact =
      session.impl_->wait_subscriber_by_handle_.size() == session.impl_->wait_attachment_count_ &&
      std::all_of(session.impl_->wait_subscriber_by_handle_.begin(),
                  session.impl_->wait_subscriber_by_handle_.end(),
                  [](const auto& indexed) {
                    return indexed.first != 0U && indexed.second != nullptr &&
                           (indexed.second->wait_handle == indexed.first ||
                            indexed.second->terminal_wait_handle == indexed.first);
                  });
  const auto drain_eligible = std::count_if(
      session.impl_->subscribers_.begin(),
      session.impl_->subscribers_.end(),
      [](const auto& subscriber) {
        return !subscriber->direct_process_local && !subscriber->clock_health_endpoint &&
               (!subscriber->externally_driven_import ||
                (subscriber->catalog_tensor_import && subscriber->tensor_endpoint.has_value()));
      });
  const bool drain_index_exact =
      session.impl_->pump_subscribers_.size() == static_cast<std::size_t>(drain_eligible) &&
      std::all_of(
          session.impl_->pump_subscribers_.begin(),
          session.impl_->pump_subscribers_.end(),
          [](const auto* subscriber) {
            return subscriber != nullptr && !subscriber->direct_process_local &&
                   !subscriber->clock_health_endpoint &&
                   (!subscriber->externally_driven_import ||
                    (subscriber->catalog_tensor_import && subscriber->tensor_endpoint.has_value()));
          });
  return RuntimeChannelPumpIndexState{
      .wait_handle_count = session.impl_->wait_subscriber_by_handle_.size(),
      .drain_subscriber_count = session.impl_->pump_subscribers_.size(),
      .background_thread_required = session.impl_->background_pump_required(),
      .background_thread_running = session.impl_->pump_thread_.joinable(),
      .exact = wait_index_exact && drain_index_exact,
  };
}

expected<void, Error> RuntimeChannelSessionTestAccess::initialize_controller_descriptor_import(
    RuntimeChannelSession& session, const PortId input, const DeviceId device) noexcept {
  auto configured = configure_descriptor_import_device(session, input, device);
  if (!configured) {
    return make_unexpected(std::move(configured).error());
  }
  if (!session.impl_->initialize_controller_descriptor_import()) {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::configure_descriptor_import_device(
    RuntimeChannelSession& session, const PortId input, const DeviceId device) noexcept {
  if (!session.impl_ || !input || !device.valid() ||
      session.impl_->armed_.load(std::memory_order_acquire) ||
      session.impl_->pump_thread_.joinable()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  if (!(**subscriber).tensor_endpoint.has_value() ||
      (**subscriber).tensor_schema_endpoint != nullptr || (**subscriber).externally_driven_import) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  (**subscriber).tensor_device = device;
  try {
    session.impl_->freeze_descriptor_import_inventory();
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  return {};
}

expected<std::uint64_t, Error>
RuntimeChannelSessionTestAccess::controller_descriptor_import_initialization_count(
    const RuntimeChannelSession& session) noexcept {
  if (!session.impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return session.impl_->controller_descriptor_import_initialization_count_.load(
      std::memory_order_relaxed);
}

expected<void, Error> RuntimeChannelSessionTestAccess::initialize_descriptor_import_on_this_thread(
    RuntimeChannelSession& session) noexcept {
  if (!session.impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  if (!session.impl_->initialize_controller_descriptor_import()) {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }
  return {};
}

expected<DeviceId, Error> RuntimeChannelSessionTestAccess::cuda_device_after_import_selection(
    const DeviceId selected, const DeviceId fallback) noexcept {
  if (!selected.valid()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  CudaDeviceCursor cursor;
  if (!cursor.select(selected) || !cursor.select(fallback)) {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }
  int observed = -1;
  if (cudaGetDevice(&observed) != cudaSuccess) {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }
  return DeviceId{observed};
}

expected<RuntimeChannelTensorTransferState, Error>
RuntimeChannelSessionTestAccess::tensor_transfer_state(const RuntimeChannelSession& session,
                                                       const PortId output,
                                                       const PortId input) noexcept {
  if (!session.impl_ || !output || !input) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto found_output = std::find_if(
      session.impl_->outputs_.begin(), session.impl_->outputs_.end(), [&](const auto& candidate) {
        return candidate->port == output;
      });
  const auto found_input =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (found_output == session.impl_->outputs_.end() ||
      found_input == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  if (!(*found_output)->tensor_payload() ||
      (*found_input)->input->type_ops != runtime_port_type_ops<Tensor>()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return RuntimeChannelTensorTransferState{
      .buffer_handle_publishers = (*found_output)->tensor_endpoints.size(),
      .inline_publishers = (*found_output)->endpoints.size(),
      .buffer_handle_subscriber = (*found_input)->tensor_endpoint.has_value(),
      .inline_subscriber = (*found_input)->endpoint != nullptr,
  };
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_output_next_sequence(
    RuntimeChannelSession& session, const PortId output,
    const std::uint64_t next_sequence) noexcept {
  auto current = output_next_sequence(session, output);
  if (!current) {
    return make_unexpected(std::move(current).error());
  }
  if (next_sequence == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto found = std::find_if(session.impl_->outputs_.begin(),
                                  session.impl_->outputs_.end(),
                                  [&](const auto& candidate) { return candidate->port == output; });
  (*found)->next_sequence = next_sequence;
  return {};
}

expected<std::uint64_t, Error> RuntimeChannelSessionTestAccess::output_next_sequence(
    const RuntimeChannelSession& session, const PortId output) noexcept {
  if (!session.impl_ || !output || session.impl_->pump_thread_.joinable() ||
      session.impl_->armed_.load(std::memory_order_acquire)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto found = std::find_if(session.impl_->outputs_.begin(),
                                  session.impl_->outputs_.end(),
                                  [&](const auto& candidate) { return candidate->port == output; });
  if (found == session.impl_->outputs_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  if ((*found)->active || (*found)->live_loan_count.load(std::memory_order_acquire) != 0U) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  return (*found)->next_sequence;
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_output_active_generation(
    RuntimeChannelSession& session, const PortId output,
    const std::uint64_t active_generation) noexcept {
  if (!session.impl_ || !output || active_generation == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto found = std::find_if(session.impl_->outputs_.begin(),
                                  session.impl_->outputs_.end(),
                                  [&](const auto& candidate) { return candidate->port == output; });
  if (found == session.impl_->outputs_.end() || (*found)->active) {
    return make_unexpected(Error{
        found == session.impl_->outputs_.end() ? ErrorCode::kNotFound : ErrorCode::kBackpressured});
  }
  (*found)->active_generation = active_generation;
  return {};
}

expected<std::uint64_t, Error> RuntimeChannelSessionTestAccess::output_active_generation(
    const RuntimeChannelSession& session, const PortId output) noexcept {
  if (!session.impl_ || !output) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto found = std::find_if(session.impl_->outputs_.begin(),
                                  session.impl_->outputs_.end(),
                                  [&](const auto& candidate) { return candidate->port == output; });
  if (found == session.impl_->outputs_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  return (*found)->active_generation;
}

expected<void, Error> RuntimeChannelSessionTestAccess::fail_next_restart_after_transport_discard(
    RuntimeChannelSession& session, const ErrorCode code) noexcept {
  if (!session.impl_ || code == ErrorCode::kSuccess) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto current = session.impl_->status();
  if (!current) {
    return make_unexpected(std::move(current).error());
  }
  if (session.impl_->pump_thread_.joinable() ||
      session.impl_->armed_.load(std::memory_order_acquire) ||
      session.impl_->restart_post_discard_failure_for_test_ != ErrorCode::kSuccess) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  // One-shot fault injection exercises the otherwise backend-dependent transition between
  // irreversible process-local transport discard and generation publication.
  session.impl_->restart_post_discard_failure_for_test_ = code;
  return {};
}
#endif

}  // namespace holoscan::detail
