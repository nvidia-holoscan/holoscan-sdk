/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <utility>

#include <holoscan/channel/runtime.hpp>

#include "../channel/channel_catalog_binding.hpp"
#include "endpoint_clock_health_runtime.hpp"
#include "runtime_channel_session_impl.hpp"
#include "runtime_channel_storage.hpp"

namespace holoscan::detail {
// Never derive this floor during receive: the live output counter could include missing data
// published after cutover. External imports keep their endpoint-owned sequence authority.
std::uint64_t RuntimeChannelSession::Impl::cutover_sequence_floor(
    const SubscriberState& subscriber) const noexcept {
  if (subscriber.externally_driven_import) {
    return 0U;
  }
  const auto output = std::find_if(outputs_.begin(), outputs_.end(), [&](const auto& candidate) {
    return candidate->port == subscriber.source;
  });
  return output == outputs_.end() ? 0U : (*output)->next_sequence - 1U;
}

/// Gate one externally driven import for a generation-fenced rebind.
///
/// Caller holds the subscriber mutex with the pump joined. The gate stays kBackpressured while
/// pending transport of the current latched incarnation exists or a live publisher is present
/// without proof that the only observed traffic already belongs to a new incarnation. Advisory
/// absence alone never fabricates a replacement generation: the cutover is admitted only
/// because the wire generation fence rejects post-cut old-incarnation frames typed. A
/// non-restart-capable import keeps failing closed with kNotSupported, and exceeding the
/// plan-admitted retained-generation capacity fails with kResourceExhausted before any
/// destructive step (no eviction, wait, or growth).
[[nodiscard]] expected<void, Error> RuntimeChannelSession::Impl::import_rebind_gate(
    SubscriberState& subscriber) noexcept {
  auto quiescence = subscriber.catalog_import_quiescence();
  if (!quiescence) {
    return make_unexpected(std::move(quiescence).error());
  }
  const bool live_publisher = subscriber.catalog_has_publishers();
  if (quiescence->pending_current || (live_publisher && !quiescence->pending_foreign)) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  // Advisory peer absence is not a generation fence. Require one accepted operation (including
  // explicit EOS) to bind the old incarnation token before a replacement subscriber can mark it
  // superseded. Otherwise a racing old publisher could become the first operation of the new
  // attachment and acquire new-generation meaning.
  if (quiescence->latched_producer_generation == 0U || !subscriber.end_of_stream_received) {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }
  if (!subscriber.import_restart_capable || subscriber.import_generation_capacity < 2U) {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }
  if (subscriber.import_generations_used >= subscriber.import_generation_capacity) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  return {};
}

/// Atomically cut one gated import over to a replacement catalog attachment.
///
/// Caller holds the subscriber mutex with the pump joined and every preflight (gate, capacity,
/// epoch exhaustion, clock-health reset) already succeeded. The endpoint services admit one
/// subscriber, so the old attachment must be detached and dropped before the replacement is
/// created; any failure after that first destructive step cannot be rolled back and
/// terminalizes this session, mirroring commit_restart's discard semantics. The replacement
/// generation slot is consumed only after the cut completes.
[[nodiscard]] expected<void, Error> RuntimeChannelSession::Impl::rebind_import(
    SubscriberState& subscriber) noexcept {
  if (!runtime_.has_value()) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  ChannelRuntime& runtime = runtime_.value();
  auto endpoint_view = subscriber.rebind_catalog.endpoint(subscriber.rebind_endpoint_name);
  if (!endpoint_view) {
    return make_unexpected(std::move(endpoint_view).error());
  }
  auto superseded = subscriber.latched_catalog_producer_generation();
  if (!superseded) {
    return make_unexpected(std::move(superseded).error());
  }
  auto detached = subscriber.detach_catalog_wait_set();
  if (!detached) {
    Error error = std::move(detached).error();
    fail(error.code);
    return make_unexpected(std::move(error));
  }
  const std::size_t retired_wait_handles = subscriber.terminal_wait_handle == 0U ? 1U : 2U;
  if (wait_attachment_count_ < retired_wait_handles) {
    fail(ErrorCode::kProtocolError);
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  wait_attachment_count_ -= retired_wait_handles;
  subscriber.endpoint.reset();
  subscriber.tensor_endpoint.reset();
  subscriber.tensor_schema_endpoint.reset();
  subscriber.refresh_staged_data_endpoint();
  if (subscriber.catalog_tensor_import &&
      subscriber.input->type_ops == runtime_port_type_ops<Tensor>()) {
    // Keep one transient control slot beyond retained typed capacity so a lossy replacement can
    // materialize before releasing the evicted Tensor backing.
    const std::uint32_t receive_control_capacity = subscriber.typed_slot_capacity + 1U;
    auto replacement = RuntimeChannelBindingAccess::create_catalog_tensor_subscriber(
        runtime, *endpoint_view, receive_control_capacity);
    if (!replacement) {
      Error error = std::move(replacement).error();
      fail(error.code);
      return make_unexpected(std::move(error));
    }
    subscriber.tensor_endpoint.emplace(std::move(*replacement));
  } else if (subscriber.catalog_tensor_import) {
    if (!subscriber.catalog_schema_identity) {
      fail(ErrorCode::kProtocolError);
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    auto replacement = RuntimeChannelBindingAccess::create_catalog_tensor_schema_subscriber(
        runtime,
        *endpoint_view,
        *subscriber.catalog_schema_identity,
        subscriber.scratch_size,
        subscriber.input->type_ops->descriptor_tensor_static_count);
    if (!replacement) {
      Error error = std::move(replacement).error();
      fail(error.code);
      return make_unexpected(std::move(error));
    }
    subscriber.tensor_schema_endpoint = std::move(*replacement);
  } else {
    auto replacement = channel_internal::CatalogChannelBindingAccess::create_payload_subscriber(
        runtime, *endpoint_view, subscriber.scratch_size);
    if (!replacement) {
      Error error = std::move(replacement).error();
      fail(error.code);
      return make_unexpected(std::move(error));
    }
    subscriber.endpoint = std::move(*replacement);
  }
  subscriber.refresh_staged_data_endpoint();
  if (*superseded != 0U) {
    auto fenced = subscriber.supersede_catalog_producer_generation(*superseded);
    if (!fenced) {
      Error error = std::move(fenced).error();
      fail(error.code);
      return make_unexpected(std::move(error));
    }
  }
  auto attached = subscriber.attach_wait_set(*wait_set_);
  if (!attached) {
    Error error = std::move(attached).error();
    fail(error.code);
    return make_unexpected(std::move(error));
  }
  const auto terminal_attached = subscriber.secondary_wait_set_handle();
  if (!replace_wait_handles(subscriber, *attached, terminal_attached)) {
    fail(ErrorCode::kProtocolError);
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  subscriber.wait_handle = *attached;
  subscriber.terminal_wait_handle = terminal_attached.value_or(0U);
  wait_attachment_count_ += terminal_attached ? 2U : 1U;
  ++subscriber.import_generations_used;
  return {};
}

expected<void, Error> RuntimeChannelSession::retry_after_freshness_handoff() noexcept {
  if (!impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return impl_->retry_after_freshness_handoff();
}

// A whole-session reset is a two-phase transaction: preflight every endpoint and retained claim,
// then reset health and advance all producer epochs together. Keeping those phases adjacent makes
// the no-partial-reset guarantee reviewable across independently owned subscribers.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> RuntimeChannelSession::reset() noexcept {
  if (!impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
#if !defined(NDEBUG)
  Impl::ControlMutationGuard control_guard{impl_->controller_mutation_barrier_};
#endif
  auto current = impl_->status();
  if (!current) {
    return make_unexpected(std::move(current).error());
  }
  if (impl_->pump_thread_.joinable() || impl_->armed_.load(std::memory_order_acquire)) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  for (const auto& input : impl_->inputs_) {
    if (input->active || input->predicate_view_constructed ||
        input->claim_pool != nullptr && input->claim_pool->has_live_claims()) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
  }
  for (const auto& output : impl_->outputs_) {
    if (output->terminal_close_started()) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    if (output->active || std::any_of(output->loans.begin(),
                                      output->loans.end(),
                                      [](const auto* loan) { return loan->live; })) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
  }
  for (const auto& subscriber : impl_->subscribers_) {
    if (!subscriber->externally_driven_import) {
      continue;
    }
    // Externally driven imports pass through the generation-fenced rebind gate instead of the
    // former unconditional kNotSupported: kBackpressured while the current incarnation may still
    // publish, kNotSupported for a non-restart-capable import, kResourceExhausted beyond the
    // plan-admitted retained-generation capacity, all before any destructive step.
    const std::scoped_lock lock{subscriber->mutex};
    auto gated = impl_->import_rebind_gate(*subscriber);
    if (!gated) {
      return make_unexpected(std::move(gated).error());
    }
  }
  for (const auto& subscriber : impl_->subscribers_) {
    // Full reset has no lifecycle fence authorizing transport consumption. Observe the transport
    // head and typed cache under their owning locks so future endpoint implementations cannot
    // race this cold-path preflight. Do not clear cached samples or advance producer epochs until
    // every process-local endpoint is empty. Imports were classified by the rebind gate above; a
    // fenced foreign head must not read as pressure.
    const std::scoped_lock lock{subscriber->transport_mutex, subscriber->mutex};
    if (!subscriber->externally_driven_import) {
      auto pending = subscriber->has_pending_transport();
      if (!pending) {
        return make_unexpected(std::move(pending).error());
      }
      if (*pending) {
        return make_unexpected(Error{ErrorCode::kBackpressured});
      }
    }
    if (subscriber->producer_epoch == std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
  }
  // Epoch exhaustion is preflighted for the entire session before any retained sample is
  // destroyed. A failure therefore leaves every independently owned subscriber at the same
  // observable generation and with the same typed payload claims it had on entry.
  auto reset_health = impl_->reset_clock_health([](const auto&) { return true; });
  if (!reset_health) {
    return make_unexpected(std::move(reset_health).error());
  }
  for (const auto& subscriber : impl_->subscribers_) {
    if (!subscriber->externally_driven_import) {
      continue;
    }
    // Atomic detach+rebind under the fence. A failure inside the destructive window has already
    // terminalized the session; nothing below may run against a partially cut attachment.
    const std::scoped_lock lock{subscriber->mutex};
    auto rebound = impl_->rebind_import(*subscriber);
    if (!rebound) {
      return make_unexpected(std::move(rebound).error());
    }
  }
  for (const auto& subscriber : impl_->subscribers_) {
    subscriber->clear_samples(impl_->cutover_sequence_floor(*subscriber));
    ++subscriber->producer_epoch;
  }
  for (const auto& input : impl_->inputs_) {
    input->reuse_slot.store(nullptr, std::memory_order_release);
    input->retirement_slots.clear();
    input->selection_prepared = false;
    input->staged_now = false;
    input->freshness_reservation_subscriber = nullptr;
    input->freshness_reservation_retained = nullptr;
    input->freshness_reservation_successor = nullptr;
    input->freshness_refill_target = nullptr;
    input->freshness_refill_target_reference = {};
    input->freshness_refill_prefix_committed = false;
    input->freshness_reservation_generation = 0U;
    input->last_refill_outcome.store(0U, std::memory_order_relaxed);
  }
  for (const auto& output : impl_->outputs_) {
    output->reset_terminal_close_state();
  }
  impl_->next_arrival_order_.store(1U, std::memory_order_release);
  impl_->blocked_transport_retry_required_.store(false, std::memory_order_release);
  impl_->terminal_error_.store(ErrorCode::kSuccess, std::memory_order_release);
  impl_->terminal_owner_.store(0U, std::memory_order_relaxed);
  impl_->terminal_diagnostic_.store(0U, std::memory_order_relaxed);
  impl_->terminal_failure_claimed_.store(false, std::memory_order_release);
  impl_->cancel_requested_.store(false, std::memory_order_release);
  impl_->closed_.store(false, std::memory_order_release);
  return {};
}

// Owner-scoped reset fences transport, claims, generations, and health state in contractual order.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> RuntimeChannelSession::reset(VertexId owner) noexcept {
  if (!impl_ || !owner) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
#if !defined(NDEBUG)
  Impl::ControlMutationGuard control_guard{impl_->controller_mutation_barrier_};
#endif
  auto current = impl_->status();
  if (!current) {
    return make_unexpected(std::move(current).error());
  }
  if (impl_->pump_thread_.joinable() || impl_->armed_.load(std::memory_order_acquire)) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  bool found = false;
  for (const auto& input : impl_->inputs_) {
    if (input->owner != owner) {
      continue;
    }
    found = true;
    if (input->active || input->predicate_view_constructed ||
        input->claim_pool != nullptr && input->claim_pool->has_live_claims()) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
  }
  for (const auto& output : impl_->outputs_) {
    if (output->owner != owner) {
      continue;
    }
    found = true;
    if (output->terminal_close_started()) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    if (output->active || std::any_of(output->loans.begin(),
                                      output->loans.end(),
                                      [](const auto* loan) { return loan->live; })) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
  }
  for (const auto& subscriber : impl_->subscribers_) {
    if (!subscriber->externally_driven_import ||
        (subscriber->owner != owner && subscriber->source_owner != owner)) {
      continue;
    }
    found = true;
    // Generation-fenced rebind gate; see reset() for the typed outcome contract.
    const std::scoped_lock lock{subscriber->mutex};
    auto gated = impl_->import_rebind_gate(*subscriber);
    if (!gated) {
      return make_unexpected(std::move(gated).error());
    }
  }
  for (const auto& subscriber : impl_->subscribers_) {
    if (subscriber->owner != owner && subscriber->source_owner != owner) {
      continue;
    }
    found = true;
    if (subscriber->input->active || subscriber->input->claim_pool->has_live_claims()) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
    if (subscriber->externally_driven_import) {
      // Imports were classified by the rebind gate above; a fenced foreign head is not pressure.
      continue;
    }
    // Observe the endpoint and typed cache under their owning locks. This cold-path fence keeps a
    // future receive implementation from changing either half of the reset preflight between the
    // pending check and generation advance.
    const std::scoped_lock lock{subscriber->transport_mutex, subscriber->mutex};
    auto pending = subscriber->has_pending_transport();
    if (!pending) {
      return make_unexpected(std::move(pending).error());
    }
    if (*pending) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
  }
  if (!found) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }

  for (const auto& input : impl_->inputs_) {
    if (input->owner == owner &&
        (input->active_generation == std::numeric_limits<std::uint64_t>::max() ||
         input->view_generation == std::numeric_limits<std::uint64_t>::max())) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
  }
  for (const auto& output : impl_->outputs_) {
    if (output->owner == owner &&
        output->active_generation == std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
  }
  for (const auto& subscriber : impl_->subscribers_) {
    if ((subscriber->owner == owner || subscriber->source_owner == owner) &&
        subscriber->producer_epoch == std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
  }
  auto reset_health = impl_->reset_clock_health([&](const auto& subscriber) {
    return subscriber.owner == owner || subscriber.source_owner == owner;
  });
  if (!reset_health) {
    return make_unexpected(std::move(reset_health).error());
  }

  for (const auto& subscriber : impl_->subscribers_) {
    if (!subscriber->externally_driven_import ||
        (subscriber->owner != owner && subscriber->source_owner != owner)) {
      continue;
    }
    // Atomic detach+rebind under the fence; a destructive-window failure terminalized the session.
    const std::scoped_lock lock{subscriber->mutex};
    auto rebound = impl_->rebind_import(*subscriber);
    if (!rebound) {
      return make_unexpected(std::move(rebound).error());
    }
  }
  for (const auto& subscriber : impl_->subscribers_) {
    if (subscriber->owner == owner || subscriber->source_owner == owner) {
      subscriber->clear_samples(impl_->cutover_sequence_floor(*subscriber));
      ++subscriber->producer_epoch;
    }
  }
  for (const auto& input : impl_->inputs_) {
    const auto* const reuse_slot = input->reuse_slot.load(std::memory_order_acquire);
    if (reuse_slot != nullptr && !reuse_slot->occupied) {
      input->reuse_slot.store(nullptr, std::memory_order_release);
    }
    if (input->owner == owner) {
      input->retirement_slots.clear();
    } else {
      std::erase_if(input->retirement_slots, [&](const auto* slot) {
        return slot->subscriber->owner == owner || slot->subscriber->source_owner == owner;
      });
    }
    const bool incident = std::any_of(
        input->subscribers.begin(), input->subscribers.end(), [&](const auto* subscriber) {
          return subscriber->owner == owner || subscriber->source_owner == owner;
        });
    if (incident) {
      input->freshness_reservation_subscriber = nullptr;
      input->freshness_reservation_retained = nullptr;
      input->freshness_reservation_successor = nullptr;
      input->freshness_refill_target = nullptr;
      input->freshness_refill_target_reference = {};
      input->freshness_refill_prefix_committed = false;
      input->freshness_reservation_generation = 0U;
      input->last_refill_outcome.store(0U, std::memory_order_relaxed);
    }
  }
  for (const auto& input : impl_->inputs_) {
    if (input->owner == owner) {
      ++input->active_generation;
      ++input->view_generation;
      input->binding.generation = input->active_generation;
      input->type_ops->bind_input(input->declaration, &input->binding);
      input->active_selection = {InputState::kNotSelected,
                                 SelectionReason::kNotSelectedForActivation};
      input->selection_prepared = false;
      input->staged_now = false;
    }
  }
  for (const auto& output : impl_->outputs_) {
    if (output->owner == owner) {
      ++output->active_generation;
      output->emitted_this_activation = 0U;
    }
  }
  return {};
}

expected<bool, Error> RuntimeChannelSession::restart_affects(
    std::span<const VertexId> restarted_owners,
    std::span<const VertexId> affected_receivers) const noexcept {
  if (!impl_ || !unique_valid_vertices(restarted_owners) ||
      !unique_valid_vertices(affected_receivers)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto contains = [](std::span<const VertexId> vertices, VertexId owner) {
    return std::find(vertices.begin(), vertices.end(), owner) != vertices.end();
  };
  return std::any_of(
             impl_->outputs_.begin(),
             impl_->outputs_.end(),
             [&](const auto& output) { return contains(restarted_owners, output->owner); }) ||
         std::any_of(
             impl_->inputs_.begin(),
             impl_->inputs_.end(),
             [&](const auto& input) { return contains(affected_receivers, input->owner); }) ||
         std::any_of(
             impl_->subscribers_.begin(), impl_->subscribers_.end(), [&](const auto& subscriber) {
               return contains(restarted_owners, subscriber->source_owner) ||
                      contains(affected_receivers, subscriber->owner);
             });
}

bool RuntimeChannelSession::awaits_external_restart(const VertexId receiver) const noexcept {
  if (!impl_) {
    return false;
  }
  const Impl::ActivationOwnerState* const owner = impl_->find_activation_owner(receiver);
  if (owner == nullptr) {
    return false;
  }
  return std::any_of(
      owner->subscribers.begin(), owner->subscribers.end(), [&](const auto* const subscriber) {
        if (subscriber == nullptr || !subscriber->externally_driven_import) {
          return false;
        }
        const std::scoped_lock lock{subscriber->mutex};
        return subscriber->end_of_stream_received && subscriber->import_restart_capable &&
               subscriber->import_generation_capacity >= 2U &&
               subscriber->import_generations_used < subscriber->import_generation_capacity;
      });
}

expected<void, Error> RuntimeChannelSession::prepare_rate_candidate_active_depth(
    EdgeId edge, std::uint32_t depth) noexcept {
  if (impl_ == nullptr || !edge || depth == 0U || impl_->armed_.load(std::memory_order_acquire)) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"});
  }
  const auto subscriber = std::ranges::find_if(
      impl_->subscribers_, [&](const auto& candidate) { return candidate->edge == edge; });
  if (subscriber == impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"});
  }
  auto& state = **subscriber;
  const std::scoped_lock lock{state.mutex};
  if (depth > state.typed_slot_capacity ||
      (!state.direct_process_local && depth != state.typed_slot_capacity)) {
    return make_unexpected(Error{ErrorCode::kNotSupported, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"});
  }
  const std::uint32_t direct_depth = state.direct_ingress_capacity == 0U ? 0U : depth;
  if (direct_depth > state.direct_ingress_capacity) {
    return make_unexpected(
        Error{ErrorCode::kResourceExhausted, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"});
  }
  state.candidate_typed_slot_capacity = depth;
  state.candidate_direct_ingress_capacity = direct_depth;
  return {};
}

expected<void, Error> RuntimeChannelSession::commit_rate_candidate_active_depths() noexcept {
  if (impl_ == nullptr || impl_->armed_.load(std::memory_order_acquire)) {
    return make_unexpected(
        Error{ErrorCode::kLifecycleRunState, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"});
  }
  for (const auto& subscriber : impl_->subscribers_) {
    const std::scoped_lock lock{subscriber->mutex};
    const auto& typed_capacity = subscriber->candidate_typed_slot_capacity;
    const auto& direct_capacity = subscriber->candidate_direct_ingress_capacity;
    if (!typed_capacity.has_value()) {
      continue;
    }
    if (!direct_capacity.has_value()) {
      return make_unexpected(
          Error{ErrorCode::kBackpressured, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"});
    }
    if (typed_capacity.value() == 0U || typed_capacity.value() > subscriber->typed_slot_capacity ||
        direct_capacity.value() > subscriber->direct_ingress_capacity ||
        std::ranges::any_of(subscriber->slots, [](const auto& slot) {
          return slot.occupied || slot.typed_ingress_reserved;
        })) {
      return make_unexpected(
          Error{ErrorCode::kBackpressured, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"});
    }
  }
  for (const auto& subscriber : impl_->subscribers_) {
    const std::scoped_lock lock{subscriber->mutex};
    auto& typed_capacity = subscriber->candidate_typed_slot_capacity;
    auto& direct_capacity = subscriber->candidate_direct_ingress_capacity;
    if (!typed_capacity.has_value()) {
      continue;
    }
    if (!direct_capacity.has_value()) {
      return make_unexpected(
          Error{ErrorCode::kBackpressured, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"});
    }
    subscriber->active_typed_slot_capacity = typed_capacity.value();
    subscriber->active_direct_ingress_capacity = direct_capacity.value();
    typed_capacity.reset();
    direct_capacity.reset();
  }
  return {};
}

void RuntimeChannelSession::discard_rate_candidate_active_depths() noexcept {
  if (impl_ == nullptr) {
    return;
  }
  for (const auto& subscriber : impl_->subscribers_) {
    const std::scoped_lock lock{subscriber->mutex};
    subscriber->candidate_typed_slot_capacity.reset();
    subscriber->candidate_direct_ingress_capacity.reset();
  }
}

// Restart commit validates the complete lifecycle closure before crossing the transport discard
// boundary, then publishes every affected generation through an infallible commit suffix.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> RuntimeChannelSession::commit_restart(
    std::span<const VertexId> restarted_owners,
    std::span<const VertexId> affected_receivers) noexcept {
  if (!impl_ || !unique_valid_vertices(restarted_owners) ||
      !unique_valid_vertices(affected_receivers)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
#if !defined(NDEBUG)
  Impl::ControlMutationGuard control_guard{impl_->controller_mutation_barrier_};
#endif
  auto current = impl_->status();
  if (!current) {
    return make_unexpected(std::move(current).error());
  }
  if (impl_->pump_thread_.joinable() || impl_->armed_.load(std::memory_order_acquire)) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  const auto receiver_affected = [&](VertexId owner) {
    return std::find(affected_receivers.begin(), affected_receivers.end(), owner) !=
           affected_receivers.end();
  };
  const auto owner_restarted = [&](VertexId owner) {
    return std::find(restarted_owners.begin(), restarted_owners.end(), owner) !=
           restarted_owners.end();
  };
  if (std::any_of(restarted_owners.begin(), restarted_owners.end(), [&](VertexId owner) {
        return !receiver_affected(owner);
      })) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  bool found = false;
  for (const auto& input : impl_->inputs_) {
    if (!receiver_affected(input->owner)) {
      continue;
    }
    found = true;
    if (input->active || input->predicate_view_constructed ||
        input->claim_pool != nullptr && input->claim_pool->has_live_claims()) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
  }
  for (const auto& output : impl_->outputs_) {
    if (!owner_restarted(output->owner)) {
      continue;
    }
    found = true;
    if (output->terminal_close_started()) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    if (output->active || std::any_of(output->loans.begin(),
                                      output->loans.end(),
                                      [](const auto* loan) { return loan->live; })) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
  }
  for (const auto& subscriber : impl_->subscribers_) {
    if (!owner_restarted(subscriber->source_owner) && !receiver_affected(subscriber->owner)) {
      continue;
    }
    found = true;
    if (subscriber->input->active || subscriber->input->claim_pool->has_live_claims()) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
  }
  for (const auto& subscriber : impl_->subscribers_) {
    if (!subscriber->externally_driven_import ||
        (!owner_restarted(subscriber->source_owner) && !receiver_affected(subscriber->owner))) {
      continue;
    }
    found = true;
    // Generation-fenced rebind gate; see reset() for the typed outcome contract.
    const std::scoped_lock lock{subscriber->mutex};
    auto gated = impl_->import_rebind_gate(*subscriber);
    if (!gated) {
      return make_unexpected(std::move(gated).error());
    }
  }
  if (!found) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }

  for (const auto& input : impl_->inputs_) {
    if (receiver_affected(input->owner) &&
        (input->active_generation == std::numeric_limits<std::uint64_t>::max() ||
         input->view_generation == std::numeric_limits<std::uint64_t>::max())) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
  }
  for (const auto& output : impl_->outputs_) {
    if (owner_restarted(output->owner) &&
        output->active_generation == std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
  }
  for (const auto& subscriber : impl_->subscribers_) {
    if ((owner_restarted(subscriber->source_owner) || receiver_affected(subscriber->owner)) &&
        subscriber->producer_epoch == std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
  }

  // The lifecycle fence joined the only process-local ingress pump and fenced the plan-owned
  // publisher before entering this method. Consume at most the compiler-frozen transport depth
  // so old-generation data cannot survive the epoch publication. A transport error or excess
  // prefix terminalizes this session: the partially drained old attachment cannot be rolled back
  // or re-armed as if its pre-restart state were intact.
  std::optional<CudaDeviceCursor> discard_device;
  const bool affected_descriptor_import =
      std::ranges::any_of(impl_->subscribers_, [&](const auto& subscriber) {
        return !subscriber->externally_driven_import &&
               subscriber->uses_device_affine_descriptor_import() &&
               (owner_restarted(subscriber->source_owner) || receiver_affected(subscriber->owner));
      });
  if (affected_descriptor_import) {
    if (!impl_->initialize_controller_descriptor_import()) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    discard_device.emplace();
  }
  bool transport_consumed = false;
  for (const auto& subscriber : impl_->subscribers_) {
    if (subscriber->externally_driven_import ||
        (!owner_restarted(subscriber->source_owner) && !receiver_affected(subscriber->owner))) {
      // An externally driven import is not discardable process-local transport; its old
      // attachment is replaced wholesale by the generation-fenced rebind below.
      continue;
    }
    auto discarded = subscriber->discard_process_local_transport(
        discard_device.has_value() ? &*discard_device : nullptr);
    if (!discarded) {
      Error error = std::move(discarded).error();
      impl_->fail(error.code);
      return make_unexpected(std::move(error));
    }
    transport_consumed = transport_consumed || *discarded != 0U;
  }
  if (discard_device.has_value() && !discard_device->restore()) {
    impl_->fail(ErrorCode::kNotSupported);
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }

  auto reset_health = impl_->reset_clock_health([&](const auto& subscriber) {
    return owner_restarted(subscriber.source_owner) || receiver_affected(subscriber.owner);
  });
  if (reset_health && impl_->restart_post_discard_failure_for_test_ != ErrorCode::kSuccess) {
    const ErrorCode injected =
        std::exchange(impl_->restart_post_discard_failure_for_test_, ErrorCode::kSuccess);
    reset_health = make_unexpected(Error{injected});
  }
  if (!reset_health) {
    Error error = std::move(reset_health).error();
    if (transport_consumed) {
      // The consumed old-generation transport prefix cannot be restored. A later health reset
      // failure therefore crosses the rollback boundary and must leave the session terminal.
      impl_->fail(error.code);
    }
    return make_unexpected(std::move(error));
  }

  for (const auto& subscriber : impl_->subscribers_) {
    if (!subscriber->externally_driven_import ||
        (!owner_restarted(subscriber->source_owner) && !receiver_affected(subscriber->owner))) {
      continue;
    }
    // Atomic detach+rebind under the fence; a destructive-window failure terminalized the session.
    const std::scoped_lock lock{subscriber->mutex};
    auto rebound = impl_->rebind_import(*subscriber);
    if (!rebound) {
      return make_unexpected(std::move(rebound).error());
    }
  }
  // With the pump joined and the bounded old transport prefix discarded above, all remaining state
  // changes are infallible and each affected subscriber is cleared exactly once before publishing
  // any new source generation.
  for (const auto& subscriber : impl_->subscribers_) {
    if (!owner_restarted(subscriber->source_owner) && !receiver_affected(subscriber->owner)) {
      continue;
    }
    subscriber->clear_samples(impl_->cutover_sequence_floor(*subscriber));
    ++subscriber->producer_epoch;
  }
  for (const auto& input : impl_->inputs_) {
    if (!receiver_affected(input->owner)) {
      continue;
    }
    input->reuse_slot.store(nullptr, std::memory_order_release);
    input->retirement_slots.clear();
    input->freshness_reservation_subscriber = nullptr;
    input->freshness_reservation_retained = nullptr;
    input->freshness_reservation_successor = nullptr;
    input->freshness_refill_target = nullptr;
    input->freshness_refill_target_reference = {};
    input->freshness_refill_prefix_committed = false;
    input->freshness_reservation_generation = 0U;
    input->last_refill_outcome.store(0U, std::memory_order_relaxed);
    ++input->active_generation;
    ++input->view_generation;
    input->binding.generation = input->active_generation;
    input->type_ops->bind_input(input->declaration, &input->binding);
    input->active_selection = {InputState::kNotSelected,
                               SelectionReason::kNotSelectedForActivation};
    input->selection_prepared = false;
    input->staged_now = false;
  }
  for (const auto& output : impl_->outputs_) {
    if (!owner_restarted(output->owner)) {
      continue;
    }
    ++output->active_generation;
    output->emitted_this_activation = 0U;
  }
  return {};
}

}  // namespace holoscan::detail
