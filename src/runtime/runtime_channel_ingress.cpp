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
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "../channel/blocked_head_wait_set.hpp"
#include "../channel/payload_descriptor_wire.hpp"
#include "../channel/pod_codec.hpp"
#include "core/diagnostic_reporting.hpp"
#include "runtime_channel_session_impl.hpp"

namespace holoscan::detail {

// Transport ingress, retained-sample caches, pump progress, and blocked-head retry.

RuntimeChannelSession::Impl::SubscriberState::~SubscriberState() {
  clear_samples(0U);
  for (SampleSlot& slot : slots) {
    if (slot.sample_storage_prepared) {
      input->type_ops->destroy_sample_storage(slot.sample.data());
      slot.sample_storage_prepared = false;
    }
  }
  if (descriptor_replacement_sample_prepared) {
    input->type_ops->destroy_sample_storage(descriptor_replacement_sample.data());
    descriptor_replacement_sample_prepared = false;
  }
  if (replacement_sample_prepared) {
    input->type_ops->destroy_sample_storage(replacement_sample.data());
    replacement_sample_prepared = false;
  } else if (replacement_sample_constructed) {
    input->type_ops->destroy_sample(replacement_sample.data());
    replacement_sample_constructed = false;
  }
}

[[nodiscard]] bool RuntimeChannelSession::Impl::SubscriberState::has_pending_direct_ingress()
    const noexcept {
  return std::any_of(slots.begin(), slots.end(), [](const auto& slot) {
    return slot.occupied && slot.direct_ingress_unread;
  });
}

[[nodiscard]] bool
RuntimeChannelSession::Impl::SubscriberState::uses_device_affine_descriptor_import()
    const noexcept {
  return (tensor_endpoint.has_value() ||
          (tensor_schema_endpoint != nullptr && cross_process_transport)) &&
         tensor_device.valid();
}

[[nodiscard]] expected<bool, Error>
RuntimeChannelSession::Impl::SubscriberState::has_pending_transport() const noexcept {
  if (direct_process_local) {
    return has_pending_direct_ingress();
  }
  if (externally_driven_import || clock_health_endpoint) {
    if (tensor_endpoint.has_value()) {
      return RuntimeChannelBindingAccess::has_pending_catalog_tensor_operation(*tensor_endpoint);
    }
    if (tensor_schema_endpoint != nullptr) {
      return RuntimeChannelBindingAccess::has_pending_catalog_tensor_schema_operation(
          *tensor_schema_endpoint);
    }
    if (endpoint == nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    return channel_internal::CatalogChannelBindingAccess::has_pending_operation(*endpoint);
  }
  if (tensor_endpoint.has_value()) {
    return tensor_endpoint->has_samples();
  }
  if (tensor_schema_endpoint != nullptr) {
    return tensor_schema_endpoint->has_samples();
  }
  if (endpoint == nullptr) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  return endpoint->has_samples();
}

[[nodiscard]] expected<std::uint32_t, Error>
RuntimeChannelSession::Impl::SubscriberState::discard_process_local_transport(
    CudaDeviceCursor* const scan_device) noexcept {
  if (direct_process_local) {
    return 0U;
  }
  if (externally_driven_import || clock_health_endpoint || drain_capacity == 0U) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  const std::scoped_lock lock{mutex};
  std::uint32_t discarded_count = 0U;
  for (; discarded_count < drain_capacity; ++discarded_count) {
    auto pending = has_pending_transport();
    if (!pending) {
      return make_unexpected(std::move(pending).error());
    }
    if (!*pending) {
      return discarded_count;
    }
    auto discarded = discard_one_process_local_transport(scan_device);
    if (!discarded) {
      return make_unexpected(std::move(discarded).error());
    }
  }
  auto excess = has_pending_transport();
  if (!excess) {
    return make_unexpected(std::move(excess).error());
  }
  if (*excess) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  return discarded_count;
}

[[nodiscard]] expected<void, Error>
RuntimeChannelSession::Impl::SubscriberState::discard_one_process_local_transport(
    CudaDeviceCursor* const scan_device) noexcept {
  if (tensor_endpoint.has_value()) {
    if (uses_device_affine_descriptor_import() &&
        (scan_device == nullptr || !scan_device->select(tensor_device))) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    auto discarded = tensor_endpoint->receive_sample();
    if (!discarded) {
      return make_unexpected(std::move(discarded).error());
    }
    return {};
  }
  if (tensor_schema_endpoint != nullptr) {
    auto received = tensor_schema_endpoint->borrow_payload();
    if (!received) {
      return make_unexpected(std::move(received).error());
    }
    const std::span<const std::byte> wire_payload = received->payload();
    const RuntimePortTypeOps& ops = *input->type_ops;
    const std::size_t descriptor_bytes =
        static_cast<std::size_t>(ops.descriptor_tensor_static_count) * sizeof(PayloadDescriptor);
    if (ops.descriptor_tensor_static_count == 0U ||
        ops.descriptor_tensor_static_count > channel_internal::kMaximumDescriptorTensorFields ||
        ops.descriptor_offset > wire_payload.size() ||
        descriptor_bytes > wire_payload.size() - ops.descriptor_offset) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    for (std::uint32_t index = 0U; index < ops.descriptor_tensor_static_count; ++index) {
      auto descriptor = channel_internal::decode_payload_descriptor_wire(wire_payload.subspan(
          ops.descriptor_offset + static_cast<std::size_t>(index) * sizeof(PayloadDescriptor),
          sizeof(PayloadDescriptor)));
      if (!descriptor) {
        return make_unexpected(std::move(descriptor).error());
      }
      tensor_schema_endpoint->release_recipient_claim(*descriptor);
    }
    return {};
  }
  auto discarded = endpoint->receive(scratch_bytes());
  if (!discarded) {
    return make_unexpected(std::move(discarded).error());
  }
  return {};
}

void RuntimeChannelSession::Impl::SubscriberState::clear_samples(
    const std::uint64_t sequence_floor) noexcept {
  DeferredSampleReleases releases{*input->type_ops, slots.size()};
  const std::scoped_lock lock{mutex};
  for (SampleSlot& slot : slots) {
    reset_sample_slot(slot, releases);
  }
  cutover_sequence_floor = sequence_floor;
  next_arrival_sequence = 1U;
  last_producer_sequence = 0U;
  last_direct_ingress_sequence = 0U;
  publish_admitted_sequence();
  end_of_stream_arrival_order = 0U;
  end_of_stream_transport_sequence = 0U;
  end_of_stream_received = false;
  end_of_stream_event_emitted = false;
}

[[nodiscard]] expected<void, Error> RuntimeChannelSession::Impl::initialize_pump_indices() {
  if (!wait_subscriber_by_handle_.empty() || !pump_subscribers_.empty()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  wait_subscriber_by_handle_.reserve(wait_attachment_count_);
  pump_reported_ready_.reserve(wait_attachment_count_);
  pump_subscribers_.reserve(subscribers_.size());
  for (const auto& subscriber : subscribers_) {
    subscriber->transport_mutex.attach(phase_counters_,
                                       RuntimePhaseCounter::kChannelTransportMutexAcquisitions,
                                       RuntimePhaseCounter::kChannelTransportMutexContended);
    subscriber->mutex.attach(phase_counters_,
                             RuntimePhaseCounter::kChannelCacheMutexAcquisitions,
                             RuntimePhaseCounter::kChannelCacheMutexContended);
    if (subscriber->wait_handle != 0U) {
      const auto [iterator, inserted] =
          wait_subscriber_by_handle_.emplace(subscriber->wait_handle, subscriber.get());
      static_cast<void>(iterator);
      if (!inserted) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
    }
    if (subscriber->terminal_wait_handle != 0U) {
      const auto [iterator, inserted] =
          wait_subscriber_by_handle_.emplace(subscriber->terminal_wait_handle, subscriber.get());
      static_cast<void>(iterator);
      if (!inserted) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
    }
    if (!subscriber->direct_process_local && !subscriber->clock_health_endpoint &&
        (!subscriber->externally_driven_import ||
         (subscriber->catalog_tensor_import && subscriber->tensor_endpoint.has_value()))) {
      pump_subscribers_.push_back(subscriber.get());
      subscriber->transport_inventory = true;
    }
  }
  if (wait_subscriber_by_handle_.size() != wait_attachment_count_) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  return {};
}

[[nodiscard]] expected<void, Error> RuntimeChannelSession::Impl::bind_blocked_head_wake() {
  if (wait_set_ == nullptr || blocked_head_wait_set_ != nullptr ||
      (fanout_test_control_ != nullptr && fanout_test_control_->disable_blocked_head_wake)) {
    return {};
  }
  if (fanout_test_control_ != nullptr && fanout_test_control_->blocked_head_spin) {
    blocked_head_spin_ = *fanout_test_control_->blocked_head_spin;
  }
  auto* const capability = channel_internal::blocked_head_wait_set(*wait_set_);
  const bool maskable = std::ranges::any_of(pump_subscribers_, [](SubscriberState* subscriber) {
    return subscriber->wait_handle != 0U && !subscriber->supports_cold_sample_replacement();
  });
  if (capability == nullptr || !maskable) {
    return {};
  }
  auto attached = capability->attach_wake();
  if (!attached) {
    return make_unexpected(std::move(attached).error());
  }
  blocked_head_wait_set_ = capability;
  for (SubscriberState* const subscriber : pump_subscribers_) {
    // A head that can replace a retained sample progresses on a later attempt instead of waiting
    // for a retirement, so it is never hidden.
    if (subscriber->wait_handle == 0U || subscriber->supports_cold_sample_replacement()) {
      continue;
    }
    subscriber->blocked_head_mask = std::make_shared<std::atomic<bool>>(false);
    auto bound = bind_blocked_head_mask(*subscriber);
    if (!bound) {
      return make_unexpected(std::move(bound).error());
    }
  }
  return {};
}

[[nodiscard]] expected<void, Error> RuntimeChannelSession::Impl::bind_blocked_head_mask(
    SubscriberState& subscriber) noexcept {
  if (blocked_head_wait_set_ == nullptr || subscriber.blocked_head_mask == nullptr) {
    return {};
  }
  subscriber.unmask_blocked_head();
  return blocked_head_wait_set_->set_readiness_mask(subscriber.wait_handle,
                                                    subscriber.blocked_head_mask);
}

[[nodiscard]] bool RuntimeChannelSession::Impl::blocked_head_hide_due(
    SubscriberState& subscriber) noexcept {
  if (subscriber.blocked_head_mask == nullptr) {
    return false;
  }
  const std::int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now().time_since_epoch())
                               .count();
  if (subscriber.blocked_since_ns == 0) {
    subscriber.blocked_since_ns = now;
  }
  return subscriber.pump_reported_ready.load(std::memory_order_relaxed) &&
         now - subscriber.blocked_since_ns >= blocked_head_spin_.count();
}

void RuntimeChannelSession::Impl::mark_pump_reported_ready(
    const std::span<const std::uint64_t> handles) noexcept {
  for (SubscriberState* const subscriber : pump_reported_ready_) {
    subscriber->pump_reported_ready.store(false, std::memory_order_relaxed);
  }
  pump_reported_ready_.clear();
  for (const std::uint64_t handle : handles) {
    const auto found = wait_subscriber_by_handle_.find(handle);
    if (found == wait_subscriber_by_handle_.end() || found->second->wait_handle != handle ||
        pump_reported_ready_.size() == pump_reported_ready_.capacity()) {
      continue;
    }
    found->second->pump_reported_ready.store(true, std::memory_order_relaxed);
    pump_reported_ready_.push_back(found->second);
  }
}

[[nodiscard]] bool RuntimeChannelSession::Impl::mask_blocked_head(
    SubscriberState& subscriber) noexcept {
  if (subscriber.blocked_head_mask == nullptr ||
      subscriber.blocked_head_mask->exchange(true, std::memory_order_seq_cst)) {
    return false;
  }
  count_phase(phase_counters_, RuntimePhaseCounter::kChannelBlockedHeadsMasked);
  return true;
}

void RuntimeChannelSession::Impl::wake_pump_for_blocked_heads(
    const std::span<SubscriberState* const> released, const bool unknown_release) noexcept {
  if (blocked_head_wait_set_ == nullptr) {
    return;
  }
  // Releases free capacity under the subscriber's cache mutex before reaching here, and a receive
  // sets the mask before retrying its reservation under that mutex. So either that retry saw the
  // freed capacity, or this exchange sees the mask. The release that finds a head hidden shows it
  // again and wakes the pump once; later releases in the same hold find nothing hidden and post
  // nothing. A head that still does not fit is hidden again by the pump's next receive.
  // Emit-admitted subscribers are retried by the release itself, never by the pump.
  const std::span<SubscriberState* const> candidates =
      released.empty() && unknown_release ? std::span<SubscriberState* const>{pump_subscribers_}
                                          : released;
  bool wake = false;
  for (SubscriberState* const subscriber : candidates) {
    // Read before exchanging, so a release with nothing hidden does not write the mask's cache
    // line, which the pump also writes.
    if (!subscriber->emit_admitted && subscriber->blocked_head_mask != nullptr &&
        subscriber->blocked_head_mask->load(std::memory_order_seq_cst) &&
        subscriber->blocked_head_mask->exchange(false, std::memory_order_seq_cst)) {
      wake = true;
    }
  }
  if (!wake) {
    return;
  }
  count_phase(phase_counters_, RuntimePhaseCounter::kChannelPumpWakesSignaled);
  auto woken = blocked_head_wait_set_->wake();
  if (!woken) {
    // The pump still retries every head after its bounded wait; only the prompt wake is lost.
    HOLOSCAN_LOG_AUTOMATIC_ERROR("Channel pump wake failed with {}",
                                 error_code_name(woken.error().code));
  }
}

// Freeze the complete subscriber inventory used to establish activation cuts.
[[nodiscard]] expected<void, Error>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
RuntimeChannelSession::Impl::initialize_activation_cut_subscriber_index() {
  std::size_t pending_capacity = 0U;
  const auto index_endpoint =
      [&](const std::string& service,
          const std::uint32_t expected_recipient_count,
          std::vector<SubscriberState*>& indexed_subscribers) -> expected<void, Error> {
    if (service.empty()) {
      return {};
    }
    indexed_subscribers.reserve(expected_recipient_count);
    for (const auto& subscriber : subscribers_) {
      if (!subscriber->direct_process_local && !subscriber->cross_process_transport &&
          subscriber->queue_policy == QueuePolicy::kBlocking && subscriber->service == service &&
          subscriber->wait_handle == 0U && subscriber->terminal_wait_handle == 0U) {
        indexed_subscribers.push_back(subscriber.get());
      }
    }
    if (indexed_subscribers.size() != expected_recipient_count) {
      indexed_subscribers.clear();
      return {};
    }
    for (const auto* subscriber : indexed_subscribers) {
      if (pending_capacity > std::numeric_limits<std::size_t>::max() - subscriber->drain_capacity) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      pending_capacity += subscriber->drain_capacity;
    }
    return {};
  };
  for (auto& output : outputs_) {
    for (auto& endpoint : output->endpoints) {
      if (!endpoint.process_local_transport) {
        continue;
      }
      auto indexed = index_endpoint(endpoint.process_local_service,
                                    endpoint.expected_recipient_count,
                                    endpoint.activation_cut_subscribers);
      if (!indexed) {
        return make_unexpected(std::move(indexed).error());
      }
    }
    for (auto& endpoint : output->tensor_endpoints) {
      auto indexed = index_endpoint(endpoint.process_local_service,
                                    endpoint.expected_recipient_count,
                                    endpoint.activation_cut_subscribers);
      if (!indexed) {
        return make_unexpected(std::move(indexed).error());
      }
    }
    for (auto& endpoint : output->tensor_schema_endpoints) {
      if (!endpoint.process_local_transport) {
        continue;
      }
      auto indexed = index_endpoint(endpoint.process_local_service,
                                    endpoint.expected_recipient_count,
                                    endpoint.activation_cut_subscribers);
      if (!indexed) {
        return make_unexpected(std::move(indexed).error());
      }
    }
  }
  activation_cut_pending_subscribers_.reserve(pending_capacity);
  activation_cut_index_available_ = pending_capacity != 0U;
  return {};
}

// Emit admission delivers to every handle-less subscriber of the endpoint's service, of either
// queue policy. A count that differs from the cold recipient proof leaves the list empty, and the
// first emit then fails closed instead of scanning the inventory.
// Keep recipient proofs and the complementary pump inventory in one initialization pass so
// emit-admitted subscribers are excluded from autonomous scans using the same frozen facts.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<void, Error> RuntimeChannelSession::Impl::initialize_emit_recipient_index() {
  const auto index_endpoint = [&](const std::string& service,
                                  const std::uint32_t expected_recipient_count,
                                  std::vector<SubscriberState*>& recipients) {
    recipients.clear();
    if (service.empty()) {
      return;
    }
    recipients.reserve(expected_recipient_count);
    for (const auto& subscriber : subscribers_) {
      if (!subscriber->direct_process_local && !subscriber->cross_process_transport &&
          subscriber->service == service && subscriber->wait_handle == 0U &&
          subscriber->terminal_wait_handle == 0U) {
        recipients.push_back(subscriber.get());
      }
    }
    if (recipients.size() != expected_recipient_count) {
      recipients.clear();
    }
    for (SubscriberState* const recipient : recipients) {
      recipient->emit_admitted = true;
    }
  };
  for (auto& output : outputs_) {
    for (auto& endpoint : output->endpoints) {
      if (endpoint.emit_admitted) {
        index_endpoint(endpoint.process_local_service,
                       endpoint.expected_recipient_count,
                       endpoint.emit_recipients);
      }
    }
    for (auto& endpoint : output->tensor_endpoints) {
      if (endpoint.emit_admitted) {
        index_endpoint(endpoint.process_local_service,
                       endpoint.expected_recipient_count,
                       endpoint.emit_recipients);
      }
    }
    for (auto& endpoint : output->tensor_schema_endpoints) {
      if (endpoint.emit_admitted) {
        index_endpoint(endpoint.process_local_service,
                       endpoint.expected_recipient_count,
                       endpoint.emit_recipients);
      }
    }
  }
  // The pump serves every other inventory subscriber; emit-admitted ones are admitted by their
  // producers and retried on retirement.
  pump_scan_subscribers_.reserve(pump_subscribers_.size());
  for (SubscriberState* const subscriber : pump_subscribers_) {
    if (!subscriber->emit_admitted) {
      pump_scan_subscribers_.push_back(subscriber);
    }
  }
  return {};
}

// Keep all endpoint families on the same blocking-subscriber credit policy and allocation
// failure boundary before producers can use the published recipient index.
[[nodiscard]] expected<void, Error>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
RuntimeChannelSession::Impl::initialize_publication_credit_index() {
  if (!publication_credits_) {
    return {};
  }
  const auto index_endpoint = [&](auto& endpoint) -> expected<void, Error> {
    if (!endpoint.process_local_transport || endpoint.process_local_service.empty()) {
      return {};
    }
    endpoint.credit_subscribers.clear();
    for (const auto& subscriber : subscribers_) {
      if (!subscriber->direct_process_local && !subscriber->cross_process_transport &&
          subscriber->queue_policy == QueuePolicy::kBlocking &&
          subscriber->service == endpoint.process_local_service) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
        try {
#endif
          endpoint.credit_subscribers.push_back(subscriber.get());
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
        } catch (const std::bad_alloc&) {
          return make_unexpected(Error{ErrorCode::kResourceExhausted});
        }
#endif
      }
    }
    return {};
  };
  for (auto& output : outputs_) {
    for (auto& endpoint : output->endpoints) {
      if (auto indexed = index_endpoint(endpoint); !indexed) {
        return indexed;
      }
    }
    for (auto& endpoint : output->tensor_endpoints) {
      if (auto indexed = index_endpoint(endpoint); !indexed) {
        return indexed;
      }
    }
    for (auto& endpoint : output->tensor_schema_endpoints) {
      if (auto indexed = index_endpoint(endpoint); !indexed) {
        return indexed;
      }
    }
  }
  return {};
}

bool RuntimeChannelSession::publication_cancelled_by_stop(const VertexId owner,
                                                          const std::uint64_t generation,
                                                          const Error& error) const noexcept {
  return impl_ && generation != 0U && error.provenance != 0U &&
         std::ranges::any_of(impl_->outputs_, [&](const auto& output) {
           return output->owner == owner && output->stop_cancelled_generation == generation &&
                  output->stop_cancelled_error.provenance == error.provenance &&
                  output->stop_cancelled_error.code == error.code &&
                  output->stop_cancelled_error.message == error.message;
         });
}

[[nodiscard]] bool RuntimeChannelSession::Impl::background_pump_required() const noexcept {
  return !pump_subscribers_.empty() &&
         (force_process_local_channel_pump_ ||
          std::ranges::any_of(pump_subscribers_, [](const auto* subscriber) {
            return subscriber->wait_handle != 0U || subscriber->terminal_wait_handle != 0U;
          }));
}

bool RuntimeChannelSession::Impl::drain_cut_retains(const SampleSlot& slot) noexcept {
  const InputPortState& input = *slot.subscriber->input;
  const SampleSlot* target = input.freshness_refill_target;
  return input.freshness_refill_drain_cut && target != nullptr && slot.occupied &&
         slot.direct_ingress_unread &&
         slot.temporal.producer_epoch == target->temporal.producer_epoch &&
         (slot.temporal == target->temporal ||
          is_strictly_newer_sample(target->temporal, slot.temporal));
}

[[nodiscard]] RuntimeChannelSession::Impl::SampleSlot*
RuntimeChannelSession::Impl::writable_direct_ingress_slot(SubscriberState& subscriber) noexcept {
  if (subscriber.active_direct_ingress_capacity == 0U) {
    return nullptr;
  }
  // A reserved record is spoken for and lives in the headroom slot, so it does not count against
  // the authored staging depth. Counting it would let one reservation shrink the survivor set the
  // contract promises.
  const std::size_t ingress_occupied = static_cast<std::size_t>(
      std::count_if(subscriber.slots.begin(), subscriber.slots.end(), [](const auto& slot) {
        return slot.occupied && slot.direct_ingress_unread && !slot.refill_reserved &&
               !slot.staged_arrival_evidence;
      }));
  if (ingress_occupied < subscriber.active_direct_ingress_capacity) {
    const auto free =
        std::find_if(subscriber.slots.begin(), subscriber.slots.end(), [](const auto& slot) {
          return !slot.occupied && !slot.typed_ingress_reserved;
        });
    return free == subscriber.slots.end() ? nullptr : &*free;
  }
  if (ingress_occupied > subscriber.active_direct_ingress_capacity) {
    return nullptr;
  }
  const auto replaceable = [](const SampleSlot& slot) noexcept {
    return slot.occupied && slot.direct_ingress_unread && !slot.refill_reserved &&
           !slot.refill_prefix_reserved && !slot.staged_arrival_evidence &&
           !drain_cut_retains(slot);
  };
  // A tick-reconciled input keeps its newest staged survivor at or before the owner's next tick.
  // The extra staging slot outlives one publication racing that tick, but a scan delayed past
  // several publications would otherwise find only survivors after the tick, decline them, and
  // drain the older record already in the typed slot.
  const SampleSlot* tick_survivor = nullptr;
  if (subscriber.input->tick_reconciled) {
    const std::int64_t fence = subscriber.input->tick_fence_ns.load(std::memory_order_acquire);
    for (const SampleSlot& slot : subscriber.slots) {
      if (!replaceable(slot)) {
        continue;
      }
      const auto coordinate = tick_fence_coordinate(*subscriber.input, slot);
      if (coordinate && *coordinate <= fence &&
          (tick_survivor == nullptr ||
           is_strictly_newer_sample(slot.temporal, tick_survivor->temporal))) {
        tick_survivor = &slot;
      }
    }
  }
  SampleSlot* oldest = nullptr;
  for (SampleSlot& slot : subscriber.slots) {
    if (replaceable(slot) && &slot != tick_survivor &&
        (oldest == nullptr || slot.temporal.source_sequence < oldest->temporal.source_sequence)) {
      oldest = &slot;
    }
  }
  return oldest != nullptr && !oldest->typed_ingress_reserved ? oldest : nullptr;
}

bool RuntimeChannelSession::Impl::normalize_direct_ingress_after_freshness_release(
    SubscriberState& subscriber, DeferredSampleReleases& releases) noexcept {
  const std::size_t occupied = static_cast<std::size_t>(
      std::count_if(subscriber.slots.begin(), subscriber.slots.end(), [](const auto& slot) {
        return slot.occupied && slot.direct_ingress_unread && !slot.staged_arrival_evidence;
      }));
  if (occupied <= subscriber.active_direct_ingress_capacity) {
    return true;
  }
  // Materialization preallocates exactly one refill-headroom slot. A larger excess means the
  // staging ledger is already inconsistent, and must not be hidden by destructive recovery.
  if (occupied != static_cast<std::size_t>(subscriber.active_direct_ingress_capacity) + 1U) {
    return false;
  }
  const auto oldest = std::min_element(
      subscriber.slots.begin(), subscriber.slots.end(), [](const auto& lhs, const auto& rhs) {
        const std::uint64_t lhs_sequence =
            lhs.occupied && lhs.direct_ingress_unread && !lhs.staged_arrival_evidence
                ? lhs.temporal.source_sequence
                : std::numeric_limits<std::uint64_t>::max();
        const std::uint64_t rhs_sequence =
            rhs.occupied && rhs.direct_ingress_unread && !rhs.staged_arrival_evidence
                ? rhs.temporal.source_sequence
                : std::numeric_limits<std::uint64_t>::max();
        return lhs_sequence < rhs_sequence;
      });
  if (oldest == subscriber.slots.end() || !oldest->occupied || !oldest->direct_ingress_unread ||
      oldest->staged_arrival_evidence || oldest->typed_ingress_reserved) {
    return false;
  }
  reset_sample_slot(*oldest, releases);
  return true;
}

[[nodiscard]] std::optional<RuntimeChannelSession::Impl::TypedIngressReservation>
RuntimeChannelSession::Impl::reserve_typed_ingress_destination_locked(
    SubscriberState& subscriber) noexcept {
  // This is a claim, not a capacity probe: typed_ingress_reserved fences the selected slot and
  // the receive commits through that exact destination without asking writable_slot() again.
  SampleSlot* destination = writable_slot(subscriber, WritableSlotIntent::kClaim);
  bool staged = false;
  // A pinned typed slot must not stop the pump draining the transport, or the newer sample stays
  // where no query can observe it and drop-oldest freshness cannot apply. Staging holds it until
  // a promotion frees the typed slot.
  //
  // Tensor-bearing ingress stages only where the compiler budgeted it: a HoloMQ subscriber in
  // this process, whose staged backings come from its producer's pools, or in another process
  // group of this application, whose staged imports keep slots of the exporter's pools leased.
  // External imports do not stage.
  const bool tensor_backed =
      subscriber.tensor_endpoint.has_value() || subscriber.tensor_schema_endpoint != nullptr;
  if (destination == nullptr && (!tensor_backed || subscriber.tensor_ingress_staging)) {
    destination = writable_direct_ingress_slot(subscriber);
    staged = destination != nullptr;
  }
  if (destination == nullptr) {
    return std::nullopt;
  }
  destination->typed_ingress_reserved = true;
  return TypedIngressReservation{subscriber, *destination, staged};
}

[[nodiscard]] std::optional<RuntimeChannelSession::Impl::TypedIngressReservation>
RuntimeChannelSession::Impl::reserve_typed_ingress_destination(
    SubscriberState& subscriber) noexcept {
  const auto cache_lock =
      acquire_admission_cache_lock(subscriber, InputCacheAdmissionSite::kTransport);
  return reserve_typed_ingress_destination_locked(subscriber);
}

// Promotion validates and publishes one bounded direct-ingress survivor set atomically.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<std::uint32_t, Error> RuntimeChannelSession::Impl::promote_direct_ingress(
    SubscriberState& subscriber, DeferredSampleReleases& releases,
    std::optional<std::uint64_t> first_arrival_order, SampleSlot* freshness_refill,
    std::int64_t* promoted_capture_ns, SampleSlot* released_typed_capacity) noexcept {
  if (subscriber.direct_ingress_capacity == 0U) {
    return 0U;
  }
  count_phase(phase_counters_, RuntimePhaseCounter::kChannelDirectIngressPromotionPasses);
  if (released_typed_capacity != nullptr && freshness_refill == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  std::uint32_t promoted = 0U;
  while (true) {
    const bool ordinary_promotion =
        freshness_refill == nullptr && released_typed_capacity == nullptr;
    InputPortState& input = *subscriber.input;
    SampleSlot* const refill_target = input.freshness_refill_target;
    const bool refill_transaction_active = input.freshness_reservation_generation != 0U;
    const auto ordinary_eligible = [&](const SampleSlot& slot) noexcept {
      if (!slot.occupied || !slot.direct_ingress_unread || slot.refill_prefix_reserved ||
          slot.staged_arrival_evidence) {
        return false;
      }
      if (refill_target == nullptr) {
        return !slot.refill_reserved;
      }
      if (!input.freshness_refill_prefix_committed ||
          slot.temporal.producer_epoch != refill_target->temporal.producer_epoch) {
        return false;
      }
      const bool at_or_before_target =
          slot.temporal == refill_target->temporal ||
          is_strictly_newer_sample(refill_target->temporal, slot.temporal);
      return at_or_before_target &&
             (!slot.refill_reserved || &slot == input.freshness_refill_target);
    };
    auto pending = subscriber.slots.end();
    if (freshness_refill != nullptr) {
      pending = std::find_if(
          subscriber.slots.begin(), subscriber.slots.end(), [&](const SampleSlot& slot) {
            return &slot == freshness_refill && slot.occupied && slot.direct_ingress_unread &&
                   (slot.refill_reserved || slot.refill_prefix_reserved) &&
                   !slot.staged_arrival_evidence;
          });
    } else if (refill_target == nullptr || !refill_transaction_active) {
      // Ordinary promotion must not consume a record reserved for a later freshness handoff.
      // That reservation names both the subscriber and slot whose coordinate justified an
      // eviction; admitting it for unrelated capacity would leave the token pointing nowhere.
      // Once a prefix step has committed and no transaction is live, however, ordinary capacity
      // may drain the fenced prefix through the target. Records newer than the target remain
      // staged until the target publishes, so neither another activation nor terminal drain can
      // recreate the out-of-order admission this fence exists to prevent.
      pending = std::min_element(
          subscriber.slots.begin(), subscriber.slots.end(), [&](const auto& lhs, const auto& rhs) {
            const std::uint64_t lhs_sequence = ordinary_eligible(lhs)
                                                   ? lhs.temporal.source_sequence
                                                   : std::numeric_limits<std::uint64_t>::max();
            const std::uint64_t rhs_sequence = ordinary_eligible(rhs)
                                                   ? rhs.temporal.source_sequence
                                                   : std::numeric_limits<std::uint64_t>::max();
            return lhs_sequence < rhs_sequence;
          });
    }
    if (pending == subscriber.slots.end() || !pending->occupied ||
        !pending->direct_ingress_unread || (ordinary_promotion && !ordinary_eligible(*pending)) ||
        pending->staged_arrival_evidence) {
      if (input.freshness_refill_target == nullptr && subscriber.end_of_stream_received &&
          !subscriber.end_of_stream_event_emitted && subscriber.end_of_stream_arrival_order == 0U) {
        auto arrival_order = reserve_arrival_order();
        if (!arrival_order) {
          return make_unexpected(std::move(arrival_order).error());
        }
        subscriber.last_producer_sequence = subscriber.end_of_stream_transport_sequence;
        subscriber.publish_admitted_sequence();
        subscriber.end_of_stream_arrival_order = *arrival_order;
        count_phase(phase_counters_, RuntimePhaseCounter::kChannelEndOfStreamRetained);
        trace_phase(phase_counters_,
                    RuntimePhaseTracePoint::kChannelEndOfStreamRetained,
                    subscriber.end_of_stream_transport_sequence);
        static_cast<void>(subscriber.input->publish_readiness_change());
      }
      return promoted;
    }
    SampleSlot* destination = nullptr;
    if (released_typed_capacity != nullptr) {
      const auto released = std::find_if(
          subscriber.slots.begin(), subscriber.slots.end(), [&](const SampleSlot& slot) {
            return &slot == released_typed_capacity;
          });
      const bool released_slot_is_free =
          released != subscriber.slots.end() && &*released != &*pending && !released->occupied &&
          !released->event_emitted && !released->witness_retired && !released->active_selected &&
          !released->reuse_retained && !released->temporal_retained &&
          !released->retirement_selected && !released->typed_ingress_reserved &&
          !released->direct_ingress_unread && !released->refill_reserved &&
          !released->refill_prefix_reserved && !released->owed_to_released_claim;
      if (!released_slot_is_free) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      // This is a one-use capacity credit, not the payload destination. The successor already
      // lives in its stable staging slot; releasing this exact typed slot proves that promoting
      // the reserved successor preserves the authored depth even if another receive currently
      // owns a separate staging reservation.
      destination = &*released;
      released_typed_capacity = nullptr;
    } else {
      destination = ordinary_promotion
                        ? writable_slot<true>(subscriber, WritableSlotIntent::kClaim, &*pending)
                        : writable_slot(subscriber, WritableSlotIntent::kClaim);
    }
    if (destination == nullptr) {
      return promoted;
    }
    if (ordinary_promotion && destination->occupied && input.tick_reconciled &&
        is_strictly_newer_sample(pending->temporal, destination->temporal)) {
      // A long-parked owner witnesses this input only at its tick. Keep the newest record at or
      // before that tick; one whose coordinate is past it waits in staging for the next tick.
      const std::int64_t fence = input.tick_fence_ns.load(std::memory_order_acquire);
      const auto kept = tick_fence_coordinate(input, *destination);
      const auto incoming = tick_fence_coordinate(input, *pending);
      if (kept && incoming && *kept <= fence && *incoming > fence) {
        count_phase(phase_counters_, RuntimePhaseCounter::kChannelTickFenceDeferredPromotions);
        return promoted;
      }
    }
    if (ordinary_promotion && destination->occupied) {
      // A later receive can enter newly writable typed storage while an earlier survivor remains
      // staged. Merge the populations by producer identity so promotion cannot regress the
      // newest-N survivor set.
      const bool pending_is_committed_refill_target =
          &*pending == refill_target && input.freshness_refill_prefix_committed;
      if (is_strictly_newer_sample(destination->temporal, pending->temporal)) {
        // A transport receive can still own this staged slot while it materializes the payload.
        // Its reservation commits immediately after the store; defer destruction until then.
        if (pending->typed_ingress_reserved) {
          return promoted;
        }
        if (!pending_is_committed_refill_target) {
          reset_sample_slot(*pending, releases);
          continue;
        }
      } else if (!is_strictly_newer_sample(pending->temporal, destination->temporal)) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
    }
    std::uint64_t arrival_order = 0U;
    if (first_arrival_order) {
      arrival_order = *first_arrival_order;
      first_arrival_order.reset();
    } else {
      auto reserved = reserve_arrival_order();
      if (!reserved) {
        return make_unexpected(std::move(reserved).error());
      }
      arrival_order = *reserved;
    }
    if (destination->occupied) {
      reset_sample_slot(*destination, releases);
    }
    pending->temporal.producer_epoch = subscriber.producer_epoch;
    // Every staged record receives this identity when it is stored. Keeping it stable across
    // promotion preserves producer-arrival identity independently of readiness publication order.
    if (pending->temporal.arrival_sequence == 0U) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    pending->readiness_sequence = 0U;
    pending->arrival_order = arrival_order;
    pending->clock_health_disposition = EndpointClockHealthDisposition::kNotApplicable;
    pending->clock_health_generation = 0U;
    pending->event_emitted = false;
    pending->witness_retired = false;
    pending->active_selected = false;
    pending->reuse_retained = false;
    pending->temporal_retained = false;
    pending->retirement_selected = false;
    const bool completed_refill_prefix =
        ordinary_promotion && &*pending == input.freshness_refill_target;
    pending->direct_ingress_unread = false;
    pending->refill_reserved = false;
    pending->refill_prefix_reserved = false;
    if (completed_refill_prefix) {
      input.freshness_refill_target = nullptr;
      input.freshness_refill_target_reference = {};
      input.freshness_refill_prefix_committed = false;
    }
    const bool for_released_claim = &*pending == freshness_refill;
    pending->owed_to_released_claim = for_released_claim || completed_refill_prefix;
    if (pending->owed_to_released_claim) {
      subscriber.input->last_refill_outcome.store(0U, std::memory_order_relaxed);
    }
    if (for_released_claim) {
      freshness_refill = nullptr;
    }
    if (promoted_capture_ns != nullptr) {
      *promoted_capture_ns = pending->temporal.capture_time.timestamp_ns;
    }
    subscriber.last_producer_sequence = pending->temporal.source_sequence;
    subscriber.publish_admitted_sequence();
    ++promoted;
    count_phase(phase_counters_, RuntimePhaseCounter::kChannelDirectIngressPromotions);
    count_phase(phase_counters_, RuntimePhaseCounter::kChannelSamplesReceived);
    trace_phase(phase_counters_,
                RuntimePhaseTracePoint::kChannelIngressRetained,
                pending->temporal.source_sequence);
    static_cast<void>(subscriber.input->publish_readiness_change());
    if (for_released_claim || completed_refill_prefix) {
      // One released claim buys exactly the reserved successor. Do not let this transaction
      // promote unrelated staging or acquire another fallible arrival-order reservation after
      // the retained slot has already been reclaimed. The same boundary applies when ordinary
      // capacity reaches a fenced target: publish that target before any newer staged record.
      return promoted;
    }
  }
}

[[nodiscard]] expected<std::optional<std::uint64_t>, Error>
RuntimeChannelSession::Impl::reserve_immediate_direct_ingress_promotion(
    SubscriberState& subscriber, DeferredSampleReleases& releases) noexcept {
  auto promoted = promote_direct_ingress(subscriber, releases);
  if (!promoted) {
    return make_unexpected(std::move(promoted).error());
  }
  if (writable_slot(subscriber, WritableSlotIntent::kProbe) == nullptr) {
    return std::nullopt;
  }
  if (subscriber.next_arrival_sequence == std::numeric_limits<std::uint64_t>::max()) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  auto arrival_order = reserve_arrival_order();
  if (!arrival_order) {
    return make_unexpected(std::move(arrival_order).error());
  }
  return std::optional<std::uint64_t>{*arrival_order};
}

[[nodiscard]] expected<void, Error>
RuntimeChannelSession::Impl::validate_and_translate_catalog_capture(
    const SubscriberState& subscriber, SampleMetadata& metadata,
    std::optional<StructuralDiagnosticId>* const terminal_diagnostic) const noexcept {
  if (terminal_diagnostic != nullptr) {
    terminal_diagnostic->reset();
  }
  if (metadata.publish_clock_id.valid() || metadata.publish_timestamp_ns != 0) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR(
        "External Channel sample on port {} carried a noncanonical publication coordinate",
        subscriber.target.value());
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }

  const bool catalog_capture_present = subscriber.catalog_capture_clock.valid();
  const bool plan_capture_present = subscriber.plan_capture_clock.valid();
  if (catalog_capture_present != plan_capture_present) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR(
        "External Channel binding on port {} carried inconsistent catalog and plan capture "
        "clocks {} and {}",
        subscriber.target.value(),
        subscriber.catalog_capture_clock.value,
        subscriber.plan_capture_clock.value);
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  if (!catalog_capture_present) {
    if (metadata.source_clock_id.valid() || metadata.capture_timestamp_ns != 0) {
      HOLOSCAN_LOG_AUTOMATIC_ERROR(
          "External Channel sample on port {} carried a capture coordinate for an uncaptured "
          "endpoint",
          subscriber.target.value());
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    return {};
  }
  if (!metadata.source_clock_id.valid()) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR(
        "External Channel sample on port {} omitted promised capture clock {}",
        subscriber.target.value(),
        subscriber.catalog_capture_clock.value);
    if (terminal_diagnostic != nullptr) {
      *terminal_diagnostic = StructuralDiagnosticId::kTemporalCaptureTimeRequired;
    }
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  if (metadata.source_clock_id != subscriber.catalog_capture_clock) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR(
        "External Channel sample on port {} carried catalog capture clock {}; expected {} "
        "before translation to plan clock {}",
        subscriber.target.value(),
        metadata.source_clock_id.value,
        subscriber.catalog_capture_clock.value,
        subscriber.plan_capture_clock.value);
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }

  // Endpoint catalogs and execution plans own independently dense clock namespaces. Translate
  // only the metadata copy at the typed-retention boundary so every retained temporal identity
  // belongs to the enclosing plan.
  metadata.source_clock_id = subscriber.plan_capture_clock;
  return {};
}

// Decode, validate, stage, and signal one scalar/schema sample as one ingress transaction.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<void, Error> RuntimeChannelSession::Impl::store_received(
    SubscriberState& subscriber, DeferredSampleReleases& releases, SampleMetadata received,
    EndpointClockHealthDisposition clock_health_disposition, std::uint64_t clock_health_generation,
    std::span<const std::byte> borrowed_payload,
    std::optional<StructuralDiagnosticId>* const terminal_diagnostic,
    SampleSlot* const reserved_destination, const bool reserved_destination_staged,
    const void* const borrowed_value) noexcept {
#if HOLOSCAN_LOG_ACTIVE_LEVEL <= HOLOSCAN_LOG_LEVEL_TRACE
  // The logger formats before applying its runtime threshold. Avoid a per-message diagnostic
  // string allocation when Trace is compiled in but disabled, without adding a Release check.
  if (Logger::level() == LogLevel::TRACE) {
    HOLOSCAN_LOG_TRACE(
        "Store Channel sample on port {} with source clock {}; catalog clock {}, "
        "plan clock {}",
        subscriber.target.value(),
        received.source_clock_id.value,
        subscriber.catalog_capture_clock.value,
        subscriber.plan_capture_clock.value);
  }
#endif
  const RuntimePortTypeOps& type_ops = *subscriber.input->type_ops;
  const bool payload_size_valid = [&] {
    if (type_ops.codec_family == RuntimePayloadCodecFamily::kGeneratedSchema) {
      return received.byte_size != 0U && received.byte_size <= type_ops.wire_capacity;
    }
    if (type_ops.codec_family == RuntimePayloadCodecFamily::kProcessLocalObject) {
      return received.byte_size == 0U && borrowed_value != nullptr;
    }
    return received.byte_size == type_ops.wire_capacity;
  }();
  if (received.version.major != 1U || received.version.minor != 0U || received.sequence == 0U ||
      (static_cast<std::uint32_t>(received.flags) & ~0x3U) != 0U || !payload_size_valid ||
      received.payload_kind != type_ops.payload_kind) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR("Channel sample on port {} carried malformed semantic metadata",
                                 subscriber.target.value());
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  if (subscriber.externally_driven_import) {
    auto validated =
        validate_and_translate_catalog_capture(subscriber, received, terminal_diagnostic);
    if (!validated) {
      return make_unexpected(std::move(validated).error());
    }
  } else {
    const bool capture_coordinate_coherent =
        received.source_clock_id.valid() || received.capture_timestamp_ns == 0;
    const bool publish_coordinate_admitted = publication_coordinate_admitted(received, false);
    const bool capture_domain_admitted =
        !received.source_clock_id.valid() || !subscriber.input->capture_clock.valid() ||
        received.source_clock_id == subscriber.input->capture_clock;
    if (!capture_coordinate_coherent || !publish_coordinate_admitted || !capture_domain_admitted) {
      HOLOSCAN_LOG_AUTOMATIC_ERROR(
          "Process-local channel sample on port {} carried an invalid clock "
          "coordinate",
          subscriber.target.value());
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
  }
  withdraw_foreign_publication_coordinate(subscriber, received);
  const bool direct_ingress =
      subscriber.direct_process_local && subscriber.direct_ingress_capacity != 0U;
  if (direct_ingress && reserved_destination != nullptr) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  // Which population this slot belongs to, which is what the rest of this function needs. It
  // coincides with direct_ingress on the direct path, where the two cannot diverge, and differs
  // for a transported sample the pump staged because the typed slot was pinned. Reading the
  // delivery mode instead would hand a staged sample a typed slot's readiness accounting even
  // though its stable arrival identity is assigned here for both populations.
  const bool staged =
      reserved_destination != nullptr ? reserved_destination_staged : direct_ingress;
  const std::uint64_t prior_sequence =
      staged ? subscriber.last_direct_ingress_sequence : subscriber.last_producer_sequence;
  if (prior_sequence != 0U && received.sequence <= prior_sequence) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR(
        "Channel subscriber received non-monotonic producer sequence {} after {} on port {}",
        received.sequence,
        prior_sequence,
        subscriber.target.value());
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  std::optional<std::uint64_t> immediate_arrival_order;
  if (staged) {
    auto reserved = reserve_immediate_direct_ingress_promotion(subscriber, releases);
    if (!reserved) {
      return make_unexpected(std::move(reserved).error());
    }
    immediate_arrival_order = *reserved;
  }
  SampleSlot* const destination =
      reserved_destination != nullptr
          ? reserved_destination
          : (direct_ingress ? writable_direct_ingress_slot(subscriber)
                            : writable_slot(subscriber, WritableSlotIntent::kClaim));
  if (destination == nullptr) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  if (subscriber.next_arrival_sequence == std::numeric_limits<std::uint64_t>::max()) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  std::uint64_t arrival_order = 0U;
  if (!staged) {
    auto reserved = reserve_arrival_order();
    if (!reserved) {
      return make_unexpected(std::move(reserved).error());
    }
    arrival_order = *reserved;
  }
  std::optional<std::uint64_t> staged_evidence_arrival_order;
  if (staged) {
    const TemporalSampleRecord incoming{
        .capture_time = received.capture_time(),
        .source_sequence = received.sequence,
        .producer_epoch = subscriber.producer_epoch,
        .arrival_sequence = subscriber.next_arrival_sequence,
    };
    auto reserved = reserve_staged_arrival_evidence_order_if_eligible(
        subscriber,
        *destination,
        incoming,
        reserved_destination == destination && reserved_destination_staged);
    if (!reserved) {
      return make_unexpected(std::move(reserved).error());
    }
    staged_evidence_arrival_order = *reserved;
  }
  std::span<const std::byte> materialized_payload;
  if (!borrowed_payload.empty()) {
    if (borrowed_payload.size() != received.byte_size) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    materialized_payload = borrowed_payload;
  } else if (type_ops.codec_family == RuntimePayloadCodecFamily::kGeneratedSchema &&
             !subscriber.externally_driven_import && !subscriber.cross_process_transport) {
    if (!subscriber.process_local_schema_identity) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    auto payload_size =
        channel_internal::decode_schema_envelope(subscriber.scratch_bytes(),
                                                 *subscriber.process_local_schema_identity,
                                                 type_ops.wire_capacity);
    if (!payload_size || *payload_size != received.byte_size) {
      return make_unexpected(payload_size ? Error{ErrorCode::kProtocolError}
                                          : std::move(payload_size).error());
    }
    materialized_payload =
        subscriber.scratch_bytes().subspan(kSchemaEnvelopeHeaderSize, *payload_size);
  } else {
    materialized_payload = subscriber.scratch_bytes().first(received.byte_size);
  }
  if (type_ops.codec_family == RuntimePayloadCodecFamily::kCustomPod &&
      !subscriber.externally_driven_import) {
    if (!subscriber.pod_identity || subscriber.native_scratch.size() != type_ops.payload_size) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    auto decoded = channel_internal::PodFieldwiseCodec::decode_prevalidated(
        *subscriber.pod_identity, materialized_payload, subscriber.native_scratch);
    if (!decoded) {
      Error error = std::move(decoded).error();
      error.code = ErrorCode::kProtocolError;
      return make_unexpected(std::move(error));
    }
    materialized_payload = subscriber.native_scratch;
  }
  const bool replacing = destination->occupied;
  assert(!replacing || destination->subscriber == &subscriber);
  assert(!replacing || destination->sample.data() != nullptr);
  assert(!replacing || subscriber.replacement_sample.data() != nullptr);
  assert(!replacing || reserved_destination == nullptr || destination == reserved_destination);
  assert(!replacing || !destination->owed_to_released_claim);
  assert(!replacing || !destination->refill_reserved);
  assert(!replacing || !destination->refill_prefix_reserved);
  assert(!replacing || !destination->staged_arrival_evidence);
  if (replacing && subscriber.replacement_sample_constructed) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  void* const construction_storage =
      replacing ? subscriber.replacement_sample.data() : destination->sample.data();
  if (type_ops.persistent_sample_storage &&
      ((!replacing && !destination->sample_storage_prepared) ||
       (replacing && !subscriber.replacement_sample_prepared) ||
       type_ops.assign_sample_storage == nullptr)) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  expected<void, Error> constructed;
  if (type_ops.codec_family == RuntimePayloadCodecFamily::kProcessLocalObject) {
    if (type_ops.construct_sample_from_value == nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    constructed =
        type_ops.construct_sample_from_value(construction_storage, borrowed_value, received);
  } else if (type_ops.persistent_sample_storage) {
    constructed =
        type_ops.assign_sample_storage(construction_storage, materialized_payload, received);
  } else {
    if (type_ops.construct_sample == nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    constructed = type_ops.construct_sample(construction_storage, materialized_payload, received);
  }
  if (!constructed) {
    Error error = std::move(constructed).error();
    if (error.code == ErrorCode::kCodecError) {
      error.code = ErrorCode::kProtocolError;
    }
    return make_unexpected(std::move(error));
  }
  if (replacing) {
    std::swap(destination->sample, subscriber.replacement_sample);
    releases.retain(subscriber.replacement_sample.data());
    if (type_ops.persistent_sample_storage) {
      type_ops.reset_sample_storage(subscriber.replacement_sample.data());
    } else {
      subscriber.replacement_sample_constructed = true;
      type_ops.destroy_sample(subscriber.replacement_sample.data());
      subscriber.replacement_sample_constructed = false;
    }
  }
  destination->temporal = TemporalSampleRecord{
      .capture_time = received.capture_time(),
      .source_sequence = received.sequence,
      .producer_epoch = subscriber.producer_epoch,
      .arrival_sequence = subscriber.next_arrival_sequence,
  };
  destination->capture_absent_reference =
      stored_capture_absent_reference(*subscriber.input, received);
  ++subscriber.next_arrival_sequence;
  destination->readiness_sequence = 0U;
  destination->arrival_order = arrival_order;
  destination->clock_health_disposition = clock_health_disposition;
  destination->clock_health_generation = clock_health_generation;
  destination->occupied = true;
  destination->event_emitted = false;
  destination->witness_retired = false;
  destination->active_selected = false;
  destination->reuse_retained = false;
  destination->temporal_retained = false;
  destination->retirement_selected = false;
  destination->direct_ingress_unread = staged;
  if (staged) {
    subscriber.last_direct_ingress_sequence = received.sequence;
    subscriber.publish_admitted_sequence();
    auto promoted = promote_direct_ingress(subscriber, releases, immediate_arrival_order);
    if (!promoted) {
      return make_unexpected(std::move(promoted).error());
    }
    if (*promoted == 0U) {
      auto published = publish_staged_arrival_evidence_if_eligible(
          subscriber,
          *destination,
          reserved_destination == destination && reserved_destination_staged,
          staged_evidence_arrival_order);
      if (!published) {
        return make_unexpected(std::move(published).error());
      }
    }
  } else {
    subscriber.last_producer_sequence = received.sequence;
    subscriber.publish_admitted_sequence();
    count_phase(phase_counters_, RuntimePhaseCounter::kChannelSamplesReceived);
    trace_phase(
        phase_counters_, RuntimePhaseTracePoint::kChannelIngressRetained, received.sequence);
    static_cast<void>(subscriber.input->publish_readiness_change());
  }
  return {};
}

// Preserve Tensor backing ownership and typed-slot publication across every ingress mode.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool RuntimeChannelSession::Impl::tensor_recipient_admits(
    const SubscriberState& subscriber, const Tensor& tensor, const TensorAccessSnapshot& snapshot,
    const SampleMetadata& metadata) noexcept {
  const bool capture_domain_admitted = !metadata.source_clock_id.valid() ||
                                       !subscriber.input->capture_clock.valid() ||
                                       metadata.source_clock_id == subscriber.input->capture_clock;
  return capture_domain_admitted &&
         tensor_snapshot_matches_plan(snapshot,
                                      subscriber.tensor_max_byte_span,
                                      subscriber.tensor_memory_kind,
                                      subscriber.tensor_device) &&
         (!subscriber.pass_through_tensor_contract ||
          tensor_matches_port_contract(
              tensor, snapshot, *subscriber.pass_through_tensor_contract)) &&
         (subscriber.pass_through_tensor_contract || !subscriber.pass_through_tensor_layout ||
          tensor_matches_port_layout(tensor, snapshot, *subscriber.pass_through_tensor_layout)) &&
         (!subscriber.tensor_logical_requirements ||
          tensor_matches_logical_requirements(tensor, *subscriber.tensor_logical_requirements));
}

[[nodiscard]] bool RuntimeChannelSession::Impl::tensor_schema_recipient_admits(
    const SubscriberState& subscriber, const std::span<const Tensor> tensors,
    const SampleMetadata& metadata) noexcept {
  if (tensors.empty() ||
      (!subscriber.tensor_fields.empty() && subscriber.tensor_fields.size() != tensors.size()) ||
      (!subscriber.tensor_logical_fields.empty() &&
       subscriber.tensor_logical_fields.size() != tensors.size())) {
    return false;
  }
  for (std::size_t index = 0U; index < tensors.size(); ++index) {
    const TensorAccessSnapshot snapshot = TensorAccess::snapshot(tensors[index]);
    const bool matches = subscriber.tensor_fields.empty()
                             ? tensor_snapshot_matches_plan(snapshot,
                                                            subscriber.tensor_max_byte_span,
                                                            subscriber.tensor_memory_kind,
                                                            subscriber.tensor_device)
                             : subscriber.tensor_fields[index].tensor_index == index &&
                                   tensor_matches_field_contract(
                                       tensors[index], snapshot, subscriber.tensor_fields[index]);
    const bool logical_matches =
        subscriber.tensor_logical_fields.empty() ||
        (subscriber.tensor_logical_fields[index].tensor_index == index &&
         tensor_matches_logical_requirements(tensors[index],
                                             subscriber.tensor_logical_fields[index].tensor));
    if (!matches || !logical_matches) {
      return false;
    }
  }
  const bool capture_domain_admitted = !metadata.source_clock_id.valid() ||
                                       !subscriber.input->capture_clock.valid() ||
                                       metadata.source_clock_id == subscriber.input->capture_clock;
  const bool pass_through_layout_matches =
      subscriber.pass_through_tensor_contract
          ? tensors.size() == 1U &&
                tensor_matches_port_contract(tensors.front(),
                                             TensorAccess::snapshot(tensors.front()),
                                             *subscriber.pass_through_tensor_contract)
          : !subscriber.pass_through_tensor_layout ||
                (tensors.size() == 1U &&
                 tensor_matches_port_layout(tensors.front(),
                                            TensorAccess::snapshot(tensors.front()),
                                            *subscriber.pass_through_tensor_layout));
  return capture_domain_admitted && pass_through_layout_matches;
}

// Keep payload validation, destination reservation, and replacement preparation before slot
// publication so rejected arrivals preserve both the incumbent Tensor and its transport evidence.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<void, Error> RuntimeChannelSession::Impl::store_received_tensor(
    SubscriberState& subscriber, DeferredSampleReleases& releases, Sample<Tensor> received,
    std::optional<StructuralDiagnosticId>* const terminal_diagnostic,
    SampleSlot* const reserved_destination, const bool reserved_destination_staged,
    const EndpointClockHealthDisposition clock_health_disposition,
    const std::uint64_t clock_health_generation) noexcept {
  const TensorAccessSnapshot snapshot = TensorAccess::snapshot(received.data);
  SampleMetadata metadata = received.metadata;
  if (subscriber.catalog_tensor_import) {
    auto validated =
        validate_and_translate_catalog_capture(subscriber, metadata, terminal_diagnostic);
    if (!validated) {
      return make_unexpected(std::move(validated).error());
    }
  }
  const bool capture_coordinate_coherent =
      metadata.source_clock_id.valid() || metadata.capture_timestamp_ns == 0;
  const bool publish_coordinate_admitted =
      publication_coordinate_admitted(metadata, subscriber.catalog_tensor_import);
  if (metadata.version.major != 1U || metadata.version.minor != 0U || metadata.sequence == 0U ||
      (static_cast<std::uint32_t>(metadata.flags) & ~0x3U) != 0U ||
      metadata.payload_kind != PayloadKind::kDescriptorBacked ||
      metadata.byte_size != snapshot.byte_size || !capture_coordinate_coherent ||
      !publish_coordinate_admitted ||
      !tensor_recipient_admits(subscriber, received.data, snapshot, metadata)) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR(
        "Tensor Channel sample on port {} violated its frozen payload contract",
        subscriber.target.value());
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  withdraw_foreign_publication_coordinate(subscriber, metadata);
  received.metadata = metadata;
  const bool direct_ingress =
      subscriber.direct_process_local && subscriber.direct_ingress_capacity != 0U;
  if (direct_ingress && reserved_destination != nullptr) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  // Population of the slot actually written, exactly as store_received computes it. The pump
  // reserves through one shared entry point, so a tensor or schema sample reaches staging on the
  // same fallback a POD sample does, and inferring population from the delivery mode here would
  // record a staged slot as typed.
  const bool staged =
      reserved_destination != nullptr ? reserved_destination_staged : direct_ingress;
  const std::uint64_t prior_sequence =
      staged ? subscriber.last_direct_ingress_sequence : subscriber.last_producer_sequence;
  if (prior_sequence != 0U && metadata.sequence <= prior_sequence) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR(
        "Tensor Channel subscriber received non-monotonic producer sequence {} after {} on "
        "port {}",
        metadata.sequence,
        prior_sequence,
        subscriber.target.value());
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  std::optional<std::uint64_t> immediate_arrival_order;
  if (staged) {
    auto reserved = reserve_immediate_direct_ingress_promotion(subscriber, releases);
    if (!reserved) {
      return make_unexpected(std::move(reserved).error());
    }
    immediate_arrival_order = *reserved;
  }
  SampleSlot* const destination =
      reserved_destination != nullptr
          ? reserved_destination
          : (direct_ingress ? writable_direct_ingress_slot(subscriber)
                            : writable_slot(subscriber, WritableSlotIntent::kClaim));
  if (destination == nullptr) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  if (subscriber.next_arrival_sequence == std::numeric_limits<std::uint64_t>::max()) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  std::uint64_t arrival_order = 0U;
  if (!staged) {
    auto reserved = reserve_arrival_order();
    if (!reserved) {
      return make_unexpected(std::move(reserved).error());
    }
    arrival_order = *reserved;
  }
  std::optional<std::uint64_t> staged_evidence_arrival_order;
  if (staged) {
    const TemporalSampleRecord incoming{
        .capture_time = metadata.capture_time(),
        .source_sequence = metadata.sequence,
        .producer_epoch = subscriber.producer_epoch,
        .arrival_sequence = subscriber.next_arrival_sequence,
    };
    auto reserved = reserve_staged_arrival_evidence_order_if_eligible(
        subscriber,
        *destination,
        incoming,
        reserved_destination == destination && reserved_destination_staged);
    if (!reserved) {
      return make_unexpected(std::move(reserved).error());
    }
    staged_evidence_arrival_order = *reserved;
  }
  const bool replacing = destination->occupied;
  assert(!replacing || destination->subscriber == &subscriber);
  assert(!replacing || destination->sample.data() != nullptr);
  assert(!replacing || subscriber.replacement_sample.data() != nullptr);
  assert(!replacing || reserved_destination == nullptr || destination == reserved_destination);
  assert(!replacing || !destination->owed_to_released_claim);
  assert(!replacing || !destination->refill_reserved);
  assert(!replacing || !destination->refill_prefix_reserved);
  assert(!replacing || !destination->staged_arrival_evidence);
  if (replacing && subscriber.replacement_sample_constructed) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  void* const construction_storage =
      replacing ? subscriber.replacement_sample.data() : destination->sample.data();
  std::construct_at(static_cast<Sample<Tensor>*>(construction_storage), std::move(received));
  if (replacing) {
    subscriber.replacement_sample_constructed = true;
    std::swap(destination->sample, subscriber.replacement_sample);
    subscriber.input->type_ops->destroy_sample(subscriber.replacement_sample.data());
    subscriber.replacement_sample_constructed = false;
  }
  destination->temporal = TemporalSampleRecord{
      .capture_time = metadata.capture_time(),
      .source_sequence = metadata.sequence,
      .producer_epoch = subscriber.producer_epoch,
      .arrival_sequence = subscriber.next_arrival_sequence,
  };
  destination->capture_absent_reference =
      stored_capture_absent_reference(*subscriber.input, metadata);
  ++subscriber.next_arrival_sequence;
  destination->readiness_sequence = 0U;
  destination->arrival_order = arrival_order;
  destination->clock_health_disposition = clock_health_disposition;
  destination->clock_health_generation = clock_health_generation;
  destination->occupied = true;
  destination->event_emitted = false;
  destination->witness_retired = false;
  destination->active_selected = false;
  destination->reuse_retained = false;
  destination->temporal_retained = false;
  destination->retirement_selected = false;
  destination->direct_ingress_unread = staged;
  if (staged) {
    subscriber.last_direct_ingress_sequence = metadata.sequence;
    subscriber.publish_admitted_sequence();
    auto promoted = promote_direct_ingress(subscriber, releases, immediate_arrival_order);
    if (!promoted) {
      return make_unexpected(std::move(promoted).error());
    }
    if (*promoted == 0U) {
      auto published = publish_staged_arrival_evidence_if_eligible(
          subscriber,
          *destination,
          reserved_destination == destination && reserved_destination_staged,
          staged_evidence_arrival_order);
      if (!published) {
        return make_unexpected(std::move(published).error());
      }
    }
  } else {
    subscriber.last_producer_sequence = metadata.sequence;
    subscriber.publish_admitted_sequence();
    count_phase(phase_counters_, RuntimePhaseCounter::kChannelSamplesReceived);
    static_cast<void>(subscriber.input->publish_readiness_change());
  }
  return {};
}

// Validate all descriptor fields before publishing one schema-backed Tensor sample.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<void, Error> RuntimeChannelSession::Impl::store_received_tensor_schema(
    SubscriberState& subscriber, DeferredSampleReleases& releases,
    const std::span<const std::byte> encoded, const std::span<Tensor> received,
    SampleMetadata metadata, SampleSlot* const reserved_destination,
    const bool reserved_destination_staged,
    const EndpointClockHealthDisposition clock_health_disposition,
    const std::uint64_t clock_health_generation,
    std::optional<StructuralDiagnosticId>* const terminal_diagnostic) noexcept {
  const RuntimePortTypeOps& ops = *subscriber.input->type_ops;
  if (received.empty() || received.size() != ops.descriptor_tensor_static_count ||
      (!subscriber.tensor_fields.empty() && subscriber.tensor_fields.size() != received.size()) ||
      (!subscriber.tensor_logical_fields.empty() &&
       subscriber.tensor_logical_fields.size() != received.size())) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  const SampleMetadata wire_metadata = metadata;
  if (subscriber.catalog_tensor_import) {
    auto validated =
        validate_and_translate_catalog_capture(subscriber, metadata, terminal_diagnostic);
    if (!validated) {
      return make_unexpected(std::move(validated).error());
    }
  }
  std::uint64_t aggregate_byte_size = 0U;
  for (const Tensor& tensor : received) {
    const std::uint64_t byte_size = TensorAccess::snapshot(tensor).byte_size;
    if (byte_size > std::numeric_limits<std::uint64_t>::max() - aggregate_byte_size) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    aggregate_byte_size += byte_size;
  }
  if (!tensor_schema_recipient_admits(subscriber, received, metadata)) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR(
        "Tensor-bearing schema sample on port {} violated its frozen payload contract",
        subscriber.target.value());
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  const bool capture_coordinate_coherent =
      metadata.source_clock_id.valid() || metadata.capture_timestamp_ns == 0;
  const bool publish_coordinate_admitted =
      publication_coordinate_admitted(metadata, subscriber.catalog_tensor_import);
  if (metadata.version.major != 1U || metadata.version.minor != 0U || metadata.sequence == 0U ||
      (static_cast<std::uint32_t>(metadata.flags) & ~0x3U) != 0U ||
      metadata.payload_kind != PayloadKind::kDescriptorBacked ||
      metadata.byte_size != aggregate_byte_size || !capture_coordinate_coherent ||
      !publish_coordinate_admitted || ops.assign_descriptor_tensors_sample == nullptr) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR(
        "Tensor-bearing schema sample on port {} violated its frozen payload contract",
        subscriber.target.value());
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  withdraw_foreign_publication_coordinate(subscriber, metadata);
  const bool direct_ingress =
      subscriber.direct_process_local && subscriber.direct_ingress_capacity != 0U;
  if (direct_ingress && reserved_destination != nullptr) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  // Population of the slot actually written, exactly as store_received computes it. The pump
  // reserves through one shared entry point, so a tensor or schema sample reaches staging on the
  // same fallback a POD sample does, and inferring population from the delivery mode here would
  // record a staged slot as typed.
  const bool staged =
      reserved_destination != nullptr ? reserved_destination_staged : direct_ingress;
  const std::uint64_t prior_sequence =
      staged ? subscriber.last_direct_ingress_sequence : subscriber.last_producer_sequence;
  if (prior_sequence != 0U && metadata.sequence <= prior_sequence) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR(
        "Tensor-bearing schema subscriber received non-monotonic producer sequence {} after {} "
        "on port {}",
        metadata.sequence,
        prior_sequence,
        subscriber.target.value());
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  std::optional<std::uint64_t> immediate_arrival_order;
  if (staged) {
    auto reserved = reserve_immediate_direct_ingress_promotion(subscriber, releases);
    if (!reserved) {
      return make_unexpected(std::move(reserved).error());
    }
    immediate_arrival_order = *reserved;
  }
  SampleSlot* const destination =
      reserved_destination != nullptr
          ? reserved_destination
          : (direct_ingress ? writable_direct_ingress_slot(subscriber)
                            : writable_slot(subscriber, WritableSlotIntent::kClaim));
  if (destination == nullptr) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  if (subscriber.next_arrival_sequence == std::numeric_limits<std::uint64_t>::max()) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  std::uint64_t arrival_order = 0U;
  if (!staged) {
    auto reserved = reserve_arrival_order();
    if (!reserved) {
      return make_unexpected(std::move(reserved).error());
    }
    arrival_order = *reserved;
  }
  std::optional<std::uint64_t> staged_evidence_arrival_order;
  if (staged) {
    const TemporalSampleRecord incoming{
        .capture_time = metadata.capture_time(),
        .source_sequence = metadata.sequence,
        .producer_epoch = subscriber.producer_epoch,
        .arrival_sequence = subscriber.next_arrival_sequence,
    };
    auto reserved = reserve_staged_arrival_evidence_order_if_eligible(
        subscriber,
        *destination,
        incoming,
        reserved_destination == destination && reserved_destination_staged);
    if (!reserved) {
      return make_unexpected(std::move(reserved).error());
    }
    staged_evidence_arrival_order = *reserved;
  }
  // Generated assignment hooks may mutate ordinary fields before rejecting malformed Tensor
  // metadata. Stage occupied-slot replacements in a cold-prepared sample, then swap block
  // ownership only after the complete assignment succeeds. This preserves the incumbent and
  // keeps the steady-state receive path allocation-free.
  const bool replacing = destination->occupied;
  assert(!replacing || destination->subscriber == &subscriber);
  assert(!replacing || destination->sample.data() != nullptr);
  assert(!replacing || subscriber.descriptor_replacement_sample.data() != nullptr);
  assert(!replacing || subscriber.descriptor_replacement_sample_prepared);
  assert(!replacing || reserved_destination == nullptr || destination == reserved_destination);
  assert(!replacing || !destination->owed_to_released_claim);
  assert(!replacing || !destination->refill_reserved);
  assert(!replacing || !destination->refill_prefix_reserved);
  assert(!replacing || !destination->staged_arrival_evidence);
  void* const assignment_storage =
      replacing ? subscriber.descriptor_replacement_sample.data() : destination->sample.data();
  if (assignment_storage == nullptr ||
      (replacing && !subscriber.descriptor_replacement_sample_prepared)) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  auto assigned = ops.assign_descriptor_tensors_sample(assignment_storage,
                                                       encoded,
                                                       received,
                                                       DescriptorTensorWireMetadata{wire_metadata},
                                                       DescriptorTensorSampleMetadata{metadata});
  if (!assigned) {
    ops.reset_sample_storage(assignment_storage);
    return make_unexpected(std::move(assigned).error());
  }
  if (replacing) {
    std::swap(destination->sample, subscriber.descriptor_replacement_sample);
    ops.reset_sample_storage(subscriber.descriptor_replacement_sample.data());
  }
  destination->temporal = TemporalSampleRecord{
      .capture_time = metadata.capture_time(),
      .source_sequence = metadata.sequence,
      .producer_epoch = subscriber.producer_epoch,
      .arrival_sequence = subscriber.next_arrival_sequence,
  };
  destination->capture_absent_reference =
      stored_capture_absent_reference(*subscriber.input, metadata);
  ++subscriber.next_arrival_sequence;
  destination->readiness_sequence = 0U;
  destination->arrival_order = arrival_order;
  destination->clock_health_disposition = clock_health_disposition;
  destination->clock_health_generation = clock_health_generation;
  destination->occupied = true;
  destination->event_emitted = false;
  destination->witness_retired = false;
  destination->active_selected = false;
  destination->reuse_retained = false;
  destination->temporal_retained = false;
  destination->retirement_selected = false;
  destination->direct_ingress_unread = staged;
  if (staged) {
    subscriber.last_direct_ingress_sequence = metadata.sequence;
    subscriber.publish_admitted_sequence();
    auto promoted = promote_direct_ingress(subscriber, releases, immediate_arrival_order);
    if (!promoted) {
      return make_unexpected(std::move(promoted).error());
    }
    if (*promoted == 0U) {
      auto published = publish_staged_arrival_evidence_if_eligible(
          subscriber,
          *destination,
          reserved_destination == destination && reserved_destination_staged,
          staged_evidence_arrival_order);
      if (!published) {
        return make_unexpected(std::move(published).error());
      }
    }
  } else {
    subscriber.last_producer_sequence = metadata.sequence;
    subscriber.publish_admitted_sequence();
    count_phase(phase_counters_, RuntimePhaseCounter::kChannelSamplesReceived);
    static_cast<void>(subscriber.input->publish_readiness_change());
  }
  return {};
}

[[nodiscard]] bool RuntimeChannelSession::Impl::owner_progress_wake_enabled() const noexcept {
  return readiness_wake_.signal_owner_progress != nullptr;
}

void RuntimeChannelSession::Impl::signal_owner_progress(VertexId owner) noexcept {
  if (!owner_progress_wake_enabled()) {
    return;
  }
  trace_phase(phase_counters_, RuntimePhaseTracePoint::kChannelReadinessWakeSignaled);
  readiness_wake_.signal_owner_progress(readiness_wake_.state, owner);
}

void RuntimeChannelSession::Impl::signal_owner_data_progress(
    const SubscriberState& subscriber) noexcept {
  if (!owner_progress_wake_enabled()) {
    return;
  }
  if (subscriber.input->side_only) {
    // The readiness revision already advanced; the owner's next trigger scan ingests the sample.
    count_phase(phase_counters_, RuntimePhaseCounter::kChannelSideOnlyDataHintsElided);
    return;
  }
  signal_owner_progress(subscriber.owner);
}

void RuntimeChannelSession::Impl::signal_owner_receive_progress(
    const SubscriberState& subscriber, const ChannelReceiveKind received) noexcept {
  if (received == ChannelReceiveKind::kEndOfStream) {
    signal_owner_progress(subscriber.owner);
    return;
  }
  signal_owner_data_progress(subscriber);
}

[[nodiscard]] expected<void, Error> RuntimeChannelSession::Impl::store_end_of_stream(
    SubscriberState& subscriber, const std::uint64_t transport_sequence) noexcept {
  if (subscriber.end_of_stream_received) {
    if (subscriber.end_of_stream_transport_sequence == transport_sequence) {
      return {};
    }
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  const std::uint64_t prior_sequence = std::max({subscriber.cutover_sequence_floor,
                                                 subscriber.last_producer_sequence,
                                                 subscriber.last_direct_ingress_sequence});
  if (prior_sequence == std::numeric_limits<std::uint64_t>::max() ||
      transport_sequence != prior_sequence + 1U) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  std::uint64_t arrival_order = 0U;
  if (!subscriber.has_pending_direct_ingress()) {
    auto reserved = reserve_arrival_order();
    if (!reserved) {
      return make_unexpected(std::move(reserved).error());
    }
    arrival_order = *reserved;
  }
  if (arrival_order != 0U) {
    subscriber.last_producer_sequence = transport_sequence;
    subscriber.publish_admitted_sequence();
  }
  subscriber.end_of_stream_transport_sequence = transport_sequence;
  subscriber.end_of_stream_arrival_order = arrival_order;
  subscriber.end_of_stream_received = true;
  if (arrival_order != 0U) {
    count_phase(phase_counters_, RuntimePhaseCounter::kChannelEndOfStreamRetained);
    trace_phase(
        phase_counters_, RuntimePhaseTracePoint::kChannelEndOfStreamRetained, transport_sequence);
    static_cast<void>(subscriber.input->publish_readiness_change());
  }
  return {};
}

[[nodiscard]] expected<ChannelReceiveKind, Error> RuntimeChannelSession::Impl::receive_tensor(
    SubscriberState& subscriber, ChannelSubscriber<Tensor>& endpoint, CudaDeviceCursor* scan_device,
    TypedIngressReservation* const reservation,
    std::optional<StructuralDiagnosticId>* const terminal_diagnostic) noexcept {
  const bool device_affine = subscriber.uses_device_affine_descriptor_import();
  // Classify the bounded transport head before touching CUDA state. With accept_data=false the
  // Tensor endpoint retains a data operation and reports kBackpressured, but it can accept EOS
  // immediately even when the final typed slot is still retained by a temporal selector.
  auto received = RuntimeChannelBindingAccess::receive_tensor_sample(
      endpoint, device_affine ? false : reservation != nullptr);
  std::optional<CudaDeviceCursor> local_device;
  if (device_affine && !received && received.error().code == ErrorCode::kBackpressured &&
      reservation != nullptr) {
    CudaDeviceCursor* device = scan_device;
    if (device == nullptr) {
      local_device.emplace();
      device = &*local_device;
    }
    if (!device->select(subscriber.tensor_device)) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    received = RuntimeChannelBindingAccess::receive_tensor_sample(endpoint, true);
  }
  if (!received) {
    if (received.error().code == ErrorCode::kClosed) {
      const std::uint64_t end_of_stream_sequence =
          RuntimeChannelBindingAccess::accepted_tensor_end_of_stream_sequence(endpoint);
      if (end_of_stream_sequence != 0U) {
        if (reservation != nullptr) {
          reservation->release();
        }
        const std::scoped_lock cache_lock{subscriber.mutex};
        return received_as(store_end_of_stream(subscriber, end_of_stream_sequence),
                           ChannelReceiveKind::kEndOfStream);
      }
    }
    return make_unexpected(std::move(received).error());
  }
  if (reservation == nullptr) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  observe_typed_ingress_for_test(subscriber);
  DeferredSampleReleases releases{*subscriber.input->type_ops, subscriber.slots.size() + 1U};
  const auto cache_lock =
      acquire_admission_cache_lock(subscriber, InputCacheAdmissionSite::kTransport);
  const bool destination_staged = reservation->staged();
  SampleSlot* const destination = reservation->destination_locked();
  auto stored = store_received_tensor(subscriber,
                                      releases,
                                      std::move(*received),
                                      terminal_diagnostic,
                                      destination,
                                      destination_staged);
  if (!stored) {
    return make_unexpected(std::move(stored).error());
  }
  return received_as(commit_typed_ingress_reservation(subscriber, *reservation, releases),
                     ChannelReceiveKind::kData);
}

// Decode and stage every descriptor field before committing the reserved typed slot.
[[nodiscard]] expected<void, Error>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
RuntimeChannelSession::Impl::store_received_tensor_schema_payload(
    SubscriberState& subscriber, const std::span<const std::byte> wire_payload,
    std::optional<SampleMetadata> metadata, TypedIngressReservation& reservation,
    CudaDeviceCursor* scan_device, const EndpointClockHealthDisposition clock_health_disposition,
    const std::uint64_t clock_health_generation,
    std::optional<StructuralDiagnosticId>* const terminal_diagnostic) noexcept {
  std::optional<CudaDeviceCursor> local_device;
  CudaDeviceCursor* device = scan_device;
  if (subscriber.uses_device_affine_descriptor_import()) {
    if (device == nullptr) {
      local_device.emplace();
      device = &*local_device;
    }
    if (!device->select(subscriber.tensor_device)) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
  }
  const RuntimePortTypeOps& ops = subscriber.projected_schema_ops != nullptr
                                      ? *subscriber.projected_schema_ops
                                      : *subscriber.input->type_ops;
  const std::uint32_t tensor_count = ops.descriptor_tensor_static_count;
  if (tensor_count == 0U || tensor_count > channel_internal::kMaximumDescriptorTensorFields ||
      (subscriber.tensor_projection &&
       subscriber.tensor_projection->tensor_index >= tensor_count) ||
      ops.descriptor_offset > wire_payload.size() ||
      static_cast<std::size_t>(tensor_count) * sizeof(PayloadDescriptor) >
          wire_payload.size() - ops.descriptor_offset ||
      ops.descriptor_tensor_sample_metadata == nullptr ||
      ops.descriptor_tensor_view_metadata_at == nullptr ||
      ops.assign_descriptor_tensors_sample == nullptr) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  auto& descriptors = subscriber.descriptor_tensor_scratch;
  auto& tensors = subscriber.tensor_schema_scratch;
  if (descriptors.size() != tensor_count || tensors.size() != tensor_count) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  const TensorScratchReset reset{tensors};
  for (std::uint32_t index = 0U; index < tensor_count; ++index) {
    auto descriptor = channel_internal::decode_payload_descriptor_wire(wire_payload.subspan(
        ops.descriptor_offset + static_cast<std::size_t>(index) * sizeof(PayloadDescriptor),
        sizeof(PayloadDescriptor)));
    if (!descriptor) {
      for (std::uint32_t release_index = 0U; release_index < index; ++release_index) {
        subscriber.tensor_schema_endpoint->release_recipient_claim(descriptors[release_index]);
      }
      return make_unexpected(std::move(descriptor).error());
    }
    descriptors[index] = *descriptor;
  }
  const std::uint64_t carriage_mask = tensor_count == 64U
                                          ? std::numeric_limits<std::uint64_t>::max()
                                          : (std::uint64_t{1U} << tensor_count) - 1U;
  const auto release_claims_from = [&](const std::uint32_t first) noexcept {
    for (std::uint32_t index = first; index < tensor_count; ++index) {
      subscriber.tensor_schema_endpoint->release_recipient_claim(descriptors[index]);
    }
  };
  const std::uint64_t endpoint_generation =
      subscriber.catalog_tensor_import ? channel_internal::kCompatibilitySchemaEndpointGeneration
                                       : plan_generation_;
  if (!metadata) {
    auto decoded =
        ops.descriptor_tensor_sample_metadata(wire_payload, carriage_mask, endpoint_generation);
    if (!decoded) {
      release_claims_from(0U);
      return make_unexpected(std::move(decoded).error());
    }
    metadata = std::move(*decoded);
  }
  for (std::uint32_t index = 0U; index < tensor_count; ++index) {
    if (subscriber.tensor_projection && index != subscriber.tensor_projection->tensor_index) {
      // This recipient owns one claim per transported field, even when it only consumes one.
      // Retire unused claims without importing or retaining those fields' backing allocations.
      subscriber.tensor_schema_endpoint->release_recipient_claim(descriptors[index]);
      continue;
    }
    expected<Tensor, Error> tensor = make_unexpected(Error{ErrorCode::kProtocolError});
    if (descriptors[index].payload_kind == ChannelPayloadKind::kBufferHandle) {
      auto tensor_metadata =
          ops.descriptor_tensor_view_metadata_at(wire_payload, index, endpoint_generation);
      if (!tensor_metadata) {
        release_claims_from(index);
        return make_unexpected(std::move(tensor_metadata).error());
      }
      tensor = subscriber.tensor_schema_endpoint->import_buffer_handle_tensor(
          descriptors[index], *tensor_metadata, index);
    } else {
      tensor = subscriber.tensor_schema_endpoint->acquire_tensor(descriptors[index]);
    }
    if (!tensor) {
      release_claims_from(index);
      return make_unexpected(std::move(tensor).error());
    }
    tensors[index] = std::move(*tensor);
  }
  observe_typed_ingress_for_test(subscriber);
  DeferredSampleReleases releases{*subscriber.input->type_ops, subscriber.slots.size() + 1U};
  const auto cache_lock =
      acquire_admission_cache_lock(subscriber, InputCacheAdmissionSite::kTransport);
  const bool destination_staged = reservation.staged();
  SampleSlot* const destination = reservation.destination_locked();
  expected<void, Error> stored;
  if (subscriber.tensor_projection) {
    Tensor& selected = tensors[subscriber.tensor_projection->tensor_index];
    SampleMetadata projected_metadata = *metadata;
    projected_metadata.byte_size = TensorAccess::snapshot(selected).byte_size;
    stored = store_received_tensor(subscriber,
                                   releases,
                                   Sample<Tensor>{std::move(selected), projected_metadata},
                                   terminal_diagnostic,
                                   destination,
                                   destination_staged,
                                   clock_health_disposition,
                                   clock_health_generation);
  } else {
    stored = store_received_tensor_schema(subscriber,
                                          releases,
                                          wire_payload,
                                          std::span<Tensor>{tensors},
                                          *metadata,
                                          destination,
                                          destination_staged,
                                          clock_health_disposition,
                                          clock_health_generation,
                                          terminal_diagnostic);
  }
  if (!stored) {
    return stored;
  }
  return commit_typed_ingress_reservation(subscriber, reservation, releases);
}

// Keep endpoint-family decode and transport acknowledgement in one ordered receive operation.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<ChannelReceiveKind, Error> RuntimeChannelSession::Impl::receive_one(
    SubscriberState& subscriber, std::optional<StructuralDiagnosticId>* const terminal_diagnostic,
    CudaDeviceCursor* const scan_device, const std::optional<TemporalSampleRecord> reuse_handoff,
    const channel_internal::StagedDataState* const expected_staged_data,
    ReceiveReceipt* const receipt) noexcept {
  const std::scoped_lock transport_lock{subscriber.transport_mutex};
  return receive_one_with_transport_held(
      subscriber, terminal_diagnostic, scan_device, reuse_handoff, expected_staged_data, receipt);
}

[[nodiscard]] expected<ChannelReceiveKind, Error>
RuntimeChannelSession::Impl::receive_one_with_transport_held(
    SubscriberState& subscriber, std::optional<StructuralDiagnosticId>* const terminal_diagnostic,
    CudaDeviceCursor* const scan_device, const std::optional<TemporalSampleRecord> reuse_handoff,
    const channel_internal::StagedDataState* const expected_staged_data,
    ReceiveReceipt* const receipt) noexcept {
  // Receives write the attribution fields under the cache mutex while holding this transport
  // mutex, so reading them under both brackets exactly this receive. A freshness handoff can
  // receive outside the scan lock, so attribution must not read them after the locks are released.
  std::uint64_t prior_sequence = 0U;
  if (receipt != nullptr) {
    *receipt = {};
    const std::scoped_lock cache_lock{subscriber.mutex};
    prior_sequence = subscriber.last_producer_sequence;
  }
  auto received = receive_one_locked(
      subscriber, terminal_diagnostic, scan_device, reuse_handoff, expected_staged_data);
  // Only a plain receive hides a blocked head: a freshness handoff is a one-time trade that is not
  // retried here, and the plain receive that follows it hides the head if it is still blocked.
  const bool plain = !reuse_handoff && expected_staged_data == nullptr;
  if (plain && !received && received.error().code == ErrorCode::kBackpressured &&
      blocked_head_hide_due(subscriber) && mask_blocked_head(subscriber)) {
    received =
        receive_one_locked(subscriber, terminal_diagnostic, scan_device, std::nullopt, nullptr);
  }
  if (received || (plain && received.error().code != ErrorCode::kBackpressured)) {
    subscriber.unmask_blocked_head();
  }
  if (receipt != nullptr && received) {
    const std::scoped_lock cache_lock{subscriber.mutex};
    if (!subscriber.end_of_stream_received && subscriber.last_producer_sequence != prior_sequence) {
      receipt->data_sequence = subscriber.last_producer_sequence;
    }
  }
  if (concurrent_admission_) {
    // Record the outcome under the transport lock that ordered it; kNotReady changes nothing.
    if (received) {
      set_transport_head_blocked(subscriber, false);
    } else if (received.error().code == ErrorCode::kBackpressured) {
      set_transport_head_blocked(subscriber, true);
    }
  }
  return received;
}

// Keep transport-kind dispatch and reuse-replacement handling under one serialized receive
// boundary so probing, typed reservation, and terminal ordering follow the same protocol.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<ChannelReceiveKind, Error> RuntimeChannelSession::Impl::receive_one_locked(
    SubscriberState& subscriber, std::optional<StructuralDiagnosticId>* const terminal_diagnostic,
    CudaDeviceCursor* const scan_device, const std::optional<TemporalSampleRecord> reuse_handoff,
    const channel_internal::StagedDataState* const expected_staged_data) noexcept {
  if (expected_staged_data != nullptr) {
    const auto staged = subscriber.staged_data_state();
    if (!staged || staged->sequence != expected_staged_data->sequence ||
        staged->capture_time != expected_staged_data->capture_time) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
  }
  const auto reserve_reuse_replacement =
      [&](std::optional<TypedIngressReservation>& reservation,
          const std::optional<channel_internal::StagedDataState>& staged_data_before)
      -> expected<bool, Error> {
    if (reservation || !staged_data_before) {
      return false;
    }
    auto replacement =
        reserve_staged_newer_data_replacement(subscriber, reuse_handoff, staged_data_before);
    if (!replacement) {
      return false;
    }
    reservation.emplace(std::move(*replacement));
    if (std::exchange(subscriber.fail_next_reuse_replacement_receive_for_test, false)) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    return true;
  };
  if (subscriber.tensor_endpoint.has_value()) {
    auto reservation = reserve_typed_ingress_destination(subscriber);
    // Reuse replacement is a follow-up transition only. The first bounded probe must stage the
    // successor and report backpressure without turning every reuse-retained slot writable.
    const auto staged_data_before =
        subscriber.staged_data_before_reuse_replacement(reservation.has_value());
    // The source-private Tensor subscriber stages one bounded transport head. Even when typed
    // retention is full it can accept ordered EOS without consuming a data head; staged data
    // remains pending until activation retirement makes a typed slot writable.
    auto received = receive_tensor(subscriber,
                                   subscriber.tensor_endpoint.value(),
                                   scan_device,
                                   reservation.has_value() ? &reservation.value() : nullptr,
                                   terminal_diagnostic);
    if (!received && received.error().code == ErrorCode::kBackpressured && !reservation &&
        staged_data_before) {
      auto replacement = reserve_reuse_replacement(reservation, staged_data_before);
      if (!replacement) {
        return make_unexpected(std::move(replacement).error());
      }
      if (replacement.value() && reservation.has_value()) {
        received = receive_tensor(subscriber,
                                  subscriber.tensor_endpoint.value(),
                                  scan_device,
                                  &reservation.value(),
                                  terminal_diagnostic);
      }
    }
    return received;
  }
  if (subscriber.tensor_schema_endpoint != nullptr) {
    auto reservation = reserve_typed_ingress_destination(subscriber);
    const auto staged_data_before =
        subscriber.staged_data_before_reuse_replacement(reservation.has_value());
    // Tensor schemas retain sidecar claims in-process and import BufferHandles across a process
    // boundary. Stage at most one transport operation before requiring typed capacity so EOS
    // remains observable while the final data sample is retained by a temporal selector.
    auto payload_loan =
        subscriber.tensor_schema_endpoint->borrow_payload_if_data_accepted(reservation.has_value());
    if (!payload_loan && payload_loan.error().code == ErrorCode::kBackpressured && !reservation &&
        staged_data_before) {
      auto replacement = reserve_reuse_replacement(reservation, staged_data_before);
      if (!replacement) {
        return make_unexpected(std::move(replacement).error());
      }
      if (*replacement) {
        payload_loan = subscriber.tensor_schema_endpoint->borrow_payload_if_data_accepted(true);
      }
    }
    if (!payload_loan) {
      if (payload_loan.error().code == ErrorCode::kClosed) {
        if (reservation) {
          reservation->release();
        }
        const auto sequence = subscriber.tensor_schema_endpoint->end_of_stream_sequence();
        if (!sequence) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        const std::scoped_lock cache_lock{subscriber.mutex};
        return received_as(store_end_of_stream(subscriber, *sequence),
                           ChannelReceiveKind::kEndOfStream);
      }
      return make_unexpected(std::move(payload_loan).error());
    }
    if (!reservation) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    return received_as(
        store_received_tensor_schema_payload(subscriber,
                                             payload_loan->payload(),
                                             std::nullopt,
                                             *reservation,
                                             scan_device,
                                             EndpointClockHealthDisposition::kNotApplicable,
                                             0U,
                                             terminal_diagnostic),
        ChannelReceiveKind::kData);
  }
  if (subscriber.endpoint == nullptr) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  // Reserve exact typed capacity before accepting a data head. The endpoint may still stage one
  // bounded head and accept terminal control when capacity is unavailable.
  auto reservation = reserve_typed_ingress_destination(subscriber);
  const auto staged_data_before =
      subscriber.staged_data_before_reuse_replacement(reservation.has_value());
  const auto store_inline_end_of_stream = [&]() -> expected<void, Error> {
    if (reservation) {
      reservation->release();
    }
    const std::scoped_lock cache_lock{subscriber.mutex};
    if (subscriber.end_of_stream_received) {
      return store_end_of_stream(subscriber, subscriber.end_of_stream_transport_sequence);
    }
    const std::uint64_t prior_sequence = std::max({subscriber.cutover_sequence_floor,
                                                   subscriber.last_producer_sequence,
                                                   subscriber.last_direct_ingress_sequence});
    if (prior_sequence == std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    return store_end_of_stream(subscriber, prior_sequence + 1U);
  };
  const RuntimePayloadCodecFamily codec_family = subscriber.input->type_ops->codec_family;
  if (codec_family == RuntimePayloadCodecFamily::kGeneratedSchema ||
      !subscriber.externally_driven_import) {
    auto received = subscriber.endpoint->receive_borrowed_if_data_accepted(reservation.has_value());
    if (!received && received.error().code == ErrorCode::kBackpressured && !reservation &&
        staged_data_before) {
      auto replacement = reserve_reuse_replacement(reservation, staged_data_before);
      if (!replacement) {
        return make_unexpected(std::move(replacement).error());
      }
      if (*replacement) {
        received = subscriber.endpoint->receive_borrowed_if_data_accepted(true);
      }
    }
    if (!received) {
      if (received.error().code == ErrorCode::kClosed) {
        return received_as(store_inline_end_of_stream(), ChannelReceiveKind::kEndOfStream);
      }
      if (received.error().code != ErrorCode::kNotSupported) {
        return make_unexpected(std::move(received).error());
      }
      // Optional copy-only endpoints retain the established receive() contract below.
    } else {
      if (!reservation) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      std::span<const std::byte> payload = received->payload();
      if (subscriber.process_local_schema_identity) {
        auto payload_size =
            channel_internal::decode_schema_envelope(payload,
                                                     *subscriber.process_local_schema_identity,
                                                     subscriber.input->type_ops->wire_capacity);
        if (!payload_size || *payload_size != received->metadata().byte_size) {
          return make_unexpected(payload_size ? Error{ErrorCode::kProtocolError}
                                              : std::move(payload_size).error());
        }
        payload = payload.subspan(kSchemaEnvelopeHeaderSize, *payload_size);
      }
      observe_typed_ingress_for_test(subscriber);
      DeferredSampleReleases releases{*subscriber.input->type_ops, subscriber.slots.size() + 1U};
      const auto cache_lock =
          acquire_admission_cache_lock(subscriber, InputCacheAdmissionSite::kTransport);
      const bool destination_staged = reservation->staged();
      SampleSlot* const destination = reservation->destination_locked();
      auto stored = store_received(subscriber,
                                   releases,
                                   received->metadata(),
                                   EndpointClockHealthDisposition::kNotApplicable,
                                   0U,
                                   payload,
                                   terminal_diagnostic,
                                   destination,
                                   destination_staged);
      if (!stored) {
        return make_unexpected(std::move(stored).error());
      }
      return received_as(commit_typed_ingress_reservation(subscriber, *reservation, releases),
                         ChannelReceiveKind::kData);
    }
  }
  if (!reservation && staged_data_before) {
    auto replacement = reserve_reuse_replacement(reservation, staged_data_before);
    if (!replacement) {
      return make_unexpected(std::move(replacement).error());
    }
  }
  if (!reservation) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  auto received = subscriber.endpoint->receive(subscriber.scratch_bytes());
  if (!received) {
    if (received.error().code == ErrorCode::kClosed) {
      return received_as(store_inline_end_of_stream(), ChannelReceiveKind::kEndOfStream);
    }
    return make_unexpected(std::move(received).error());
  }
  // The scratch is subscriber-owned and reused across samples, so a frame that did not fill it
  // would leave the tail holding the previous sample. Admit only a complete typed frame.
  if (received->payload_size != subscriber.scratch_bytes().size()) {
    return make_unexpected(Error{ErrorCode::kProtocolError,
                                 "received payload byte size does not match its endpoint type"});
  }
  observe_typed_ingress_for_test(subscriber);
  DeferredSampleReleases releases{*subscriber.input->type_ops, subscriber.slots.size() + 1U};
  const auto cache_lock =
      acquire_admission_cache_lock(subscriber, InputCacheAdmissionSite::kTransport);
  const bool destination_staged = reservation->staged();
  SampleSlot* const destination = reservation->destination_locked();
  auto stored = store_received(subscriber,
                               releases,
                               received->metadata,
                               EndpointClockHealthDisposition::kNotApplicable,
                               0U,
                               {},
                               terminal_diagnostic,
                               destination,
                               destination_staged);
  if (!stored) {
    return make_unexpected(std::move(stored).error());
  }
  return received_as(commit_typed_ingress_reservation(subscriber, *reservation, releases),
                     ChannelReceiveKind::kData);
}

// Clock-health and data operations share one ordered monitored-input drain boundary.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<void, Error> RuntimeChannelSession::Impl::drain_monitored_inputs(
    const ActivationOwnerState& owner) noexcept {
  // Monitored external imports have no background pump to establish thread-local CUDA import
  // state. Initialize it on the serialized controller before any catalog BufferHandle is read.
  if (!initialize_controller_descriptor_import()) {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }
  std::optional<CudaDeviceCursor> controller_device;
  for (SubscriberState* const subscriber_pointer : owner.subscribers) {
    SubscriberState& subscriber = *subscriber_pointer;
    if (!subscriber.externally_driven_import) {
      continue;
    }
    const bool monitored = static_cast<bool>(subscriber.clock_health_endpoint);
    if (monitored && (subscriber.clock_health_coordinator == nullptr ||
                      subscriber.health_dispatch_budget == 0U)) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }

    CudaDeviceCursor* receive_device = nullptr;
    if (subscriber.uses_device_affine_descriptor_import()) {
      if (!controller_device.has_value()) {
        controller_device.emplace();
      }
      if (!controller_device->select(subscriber.tensor_device)) {
        return make_unexpected(Error{ErrorCode::kNotSupported});
      }
      receive_device = &*controller_device;
    }

    const std::unique_lock transport_lock{subscriber.transport_mutex, std::try_to_lock};
    if (!transport_lock.owns_lock()) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
    std::uint32_t data_count = 0U;
    while (data_count < subscriber.drain_capacity) {
      std::optional<TypedIngressReservation> reservation;
      {
        const std::scoped_lock cache_lock{subscriber.mutex};
        if (subscriber.end_of_stream_received) {
          break;
        }
        auto selected = reserve_typed_ingress_destination_locked(subscriber);
        if (selected) {
          reservation.emplace(std::move(*selected));
        }
      }
      auto operation = subscriber.receive_catalog_operation(reservation.has_value());
      if (!operation) {
        return make_unexpected(std::move(operation).error());
      }
      if (operation->kind == channel_internal::CatalogReceivedOperationKind::kNone) {
        break;
      }
      if (operation->kind == channel_internal::CatalogReceivedOperationKind::kDataPending) {
        // The catalog subscriber retains this contiguous head in its bounded staging slot. Leave
        // its sequence uncommitted until activation retirement makes typed storage writable.
        count_phase(phase_counters_, RuntimePhaseCounter::kChannelReceiveBackpressured);
        break;
      }
      if (operation->kind ==
          channel_internal::CatalogReceivedOperationKind::kPeerIncarnationChanged) {
        // The catalog subscriber fenced traffic from another producer incarnation: it delivered
        // no payload, kept EOS state unchanged, and mutated no receiver state. Keep the accepted
        // old-generation state fail-closed; only an explicit generation-fenced rebind (a reset
        // form) may cut over to the new incarnation.
        break;
      }

      const bool data_operation =
          operation->kind == channel_internal::CatalogReceivedOperationKind::kData;
      std::optional<channel_internal::TensorSchemaPayloadReceiveGuard> schema_payload_loan;
      if (data_operation && subscriber.tensor_schema_endpoint != nullptr) {
        schema_payload_loan.emplace(*subscriber.tensor_schema_endpoint, operation->schema_payload);
      }
      if (!data_operation && reservation) {
        reservation->release();
      }
      if (operation->kind == channel_internal::CatalogReceivedOperationKind::kClockHealth) {
        const std::scoped_lock cache_lock{subscriber.mutex};
        if (!monitored) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        if (clock_health_serialization_mutex_ == nullptr) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        const auto& health = operation->clock_health;
        // The wire producer field carries the producer-incarnation identity token, not an
        // ordinal. The catalog subscriber only surfaces operations of the latched incarnation,
        // so an applied report must match that token exactly; the Runtime-local producer_epoch
        // below stays a small local ordinal advanced by reset/rebind, never a wire value.
        auto latched = subscriber.latched_catalog_producer_generation();
        if (!latched) {
          return make_unexpected(std::move(latched).error());
        }
        if (*latched == 0U || health.producer_epoch != *latched) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        // Keep report enqueue and dispatch in one coordinator transaction. Expiry and reset use
        // the same cold-bound mutex, so neither can interleave between report validation and the
        // state transition that the following data operation observes. The transport/cache
        // locks are always acquired before this innermost lock; no clock-health path acquires
        // them in the reverse order.
        const std::lock_guard clock_health_lock{*clock_health_serialization_mutex_};
        auto submitted = subscriber.clock_health_coordinator->submit_report(
            subscriber.clock_health_endpoint,
            ClockHealthReport{.assessment = health.assessment, .reason = health.reason},
            subscriber.producer_epoch,
            health.publisher_operation_sequence,
            health.report_sequence);
        if (!submitted) {
          return make_unexpected(
              Error{submitted.error().code, std::string{submitted.error().token}});
        }
        auto applied =
            subscriber.clock_health_coordinator->dispatch_pending(subscriber.clock_health_endpoint);
        if (!applied || *applied != 1U) {
          return make_unexpected(
              applied ? Error{ErrorCode::kProtocolError}
                      : Error{applied.error().code, std::string{applied.error().token}});
        }
        // The baseline scalar plan admits one report per Dispatcher pass. Stop after applying it so
        // report bursts cannot starve data, expiry, lifecycle, or stop work; the next lane poll
        // resumes from the catalog subscriber's two retained service heads.
        break;
      }

      if (operation->kind == channel_internal::CatalogReceivedOperationKind::kEndOfStream) {
        const std::scoped_lock cache_lock{subscriber.mutex};
        auto stored_end_of_stream = store_end_of_stream(subscriber, operation->metadata.sequence);
        if (!stored_end_of_stream) {
          return make_unexpected(std::move(stored_end_of_stream).error());
        }
        break;
      }

      EndpointClockHealthDisposition health_disposition =
          EndpointClockHealthDisposition::kNotApplicable;
      std::uint64_t health_generation = 0U;
      if (monitored) {
        if (clock_health_serialization_mutex_ == nullptr) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        // Freeze the disposition and generation as one health snapshot. A concurrent expiry or
        // reset may proceed only after this sample has retained the complete prior snapshot.
        const std::lock_guard clock_health_lock{*clock_health_serialization_mutex_};
        EndpointClockHealthReceiver* const receiver =
            subscriber.clock_health_coordinator->receiver(subscriber.clock_health_endpoint);
        if (receiver == nullptr) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        health_disposition = receiver->current_disposition();
        health_generation = receiver->state().health_generation;
      }
      std::optional<StructuralDiagnosticId> terminal_diagnostic;
      if (!reservation) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      expected<void, Error> stored;
      if (subscriber.tensor_schema_endpoint != nullptr) {
        stored = store_received_tensor_schema_payload(subscriber,
                                                      operation->schema_payload,
                                                      operation->metadata,
                                                      reservation.value(),
                                                      receive_device,
                                                      health_disposition,
                                                      health_generation,
                                                      &terminal_diagnostic);
      } else {
        observe_typed_ingress_for_test(subscriber);
        DeferredSampleReleases releases{*subscriber.input->type_ops, subscriber.slots.size() + 1U};
        const std::scoped_lock cache_lock{subscriber.mutex};
        const bool destination_staged = reservation->staged();
        SampleSlot* const destination = reservation->destination_locked();
        if (subscriber.tensor_endpoint.has_value()) {
          auto& tensor_sample = operation->tensor_sample;
          if (!tensor_sample) {
            return make_unexpected(Error{ErrorCode::kProtocolError});
          }
          stored = store_received_tensor(subscriber,
                                         releases,
                                         std::move(tensor_sample.value()),
                                         &terminal_diagnostic,
                                         destination,
                                         destination_staged,
                                         health_disposition,
                                         health_generation);
        } else {
          stored = store_received(subscriber,
                                  releases,
                                  operation->metadata,
                                  health_disposition,
                                  health_generation,
                                  operation->schema_payload,
                                  &terminal_diagnostic,
                                  destination,
                                  destination_staged);
        }
        if (stored) {
          stored = commit_typed_ingress_reservation(subscriber, reservation.value(), releases);
        }
      }
      if (!stored) {
        // Every terminal transport/storage rejection closes the session. The optional
        // structural identity enriches attribution; it must not determine lifecycle state.
        fail(stored.error(), subscriber.owner, terminal_diagnostic);
        return make_unexpected(std::move(stored).error());
      }
      ++data_count;
    }
  }
  return {};
}

// Preserve fair subscriber scanning, blocked-head retry, and terminal ordering in one pass.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<bool, Error> RuntimeChannelSession::Impl::drain_unmonitored_inputs(
    PortId& failed_port, VertexId& failed_owner,
    std::optional<StructuralDiagnosticId>& terminal_diagnostic, const DrainScope scope,
    CudaDeviceCursor* scan_device, const bool activation_cut,
    const std::span<SubscriberState* const> recipients, const bool skip_known_blocked) noexcept {
  failed_port = {};
  failed_owner = {};
  terminal_diagnostic.reset();
  // Only the serialized controller activation cut consumes the prior retry request. Pump scans
  // preserve it so coincident inbound and outbound progress cannot hide a blocked transport head
  // before the controller establishes the cut. A still-blocked head relatches it below.
  if (activation_cut && scope == DrainScope::kInventory && !concurrent_admission_) {
    blocked_transport_retry_required_.store(false, std::memory_order_release);
  }
  bool readiness_published = false;
  if (activation_cut && scope == DrainScope::kInventory) {
    count_phase(phase_counters_, RuntimePhaseCounter::kChannelActivationCutInventoryScans);
  }
  const auto inventory =
      scope == DrainScope::kInventory  ? std::span<SubscriberState* const>{pump_subscribers_}
      : scope == DrainScope::kPumpScan ? std::span<SubscriberState* const>{pump_scan_subscribers_}
                                       : recipients;
  for (SubscriberState* const subscriber_pointer : inventory) {
    if (activation_cut) {
      count_phase(phase_counters_, RuntimePhaseCounter::kChannelActivationCutSubscriberProbes);
    }
    SubscriberState& subscriber = *subscriber_pointer;
    for (std::uint32_t count = 0U; count < subscriber.drain_capacity; ++count) {
      if (activation_cut) {
        count_phase(phase_counters_, RuntimePhaseCounter::kChannelActivationCutTransportProbes);
      }
      // Readiness is advisory—the endpoint transaction below remains authoritative—but probing
      // it first avoids capacity scans, locks, and false backpressure while transport is empty.
      // Endpoint staging preserves a concurrently observed data head or terminal record. The probe
      // reads that staging, which an owner's freshness trade can receive into outside the scan
      // lock, so it takes the subscriber's transport mutex like every receive. Concurrent admission
      // drains one subscriber from several lanes without the scan lock, and the backend port and
      // its staging are not thread-safe, so it keeps that mutex across the receive: a head another
      // lane consumes in between, typically EOS, would otherwise be received again, and a closed
      // stream would re-report its accepted EOS as a new delivery.
      std::unique_lock<CountedMutex> transport_lock{subscriber.transport_mutex};
      auto pending = subscriber.has_pending_transport();
      if (!concurrent_admission_) {
        transport_lock.unlock();
      }
      if (!pending) {
        return make_unexpected(pending.error());
      }
      if (!*pending) {
        break;
      }
      // This head stays blocked until its own subscriber releases typed capacity, and that release
      // retries it. Everything pending here is ordered behind it. Inputs that can replace a
      // retained sample instead progress on a later attempt, so they are never skipped. Only
      // concurrent admission skips, and it reads the flag under the transport mutex that wrote it.
      if (count == 0U && skip_known_blocked && transport_lock.owns_lock() &&
          subscriber.transport_head_blocked.load(std::memory_order_relaxed) &&
          !subscriber.supports_cold_sample_replacement()) {
        count_phase(phase_counters_, RuntimePhaseCounter::kChannelKnownBlockedSkips);
        break;
      }
      ReceiveReceipt receipt;
      ReceiveReceipt* const observed = fanout_observation_enabled() ? &receipt : nullptr;
      // Emit-time admission hints the owner only when the input's readiness changes, as direct
      // delivery does: a staged drop-oldest arrival publishes no revision and needs no scan. A
      // newly received terminal record always hints. Both reads sit inside the receive's own
      // serialization (the transport mutex, or the scan lock).
      const bool gate_hint = scope == DrainScope::kRecipients;
      const std::uint64_t prior_revision =
          gate_hint ? subscriber.input->readiness_revision.load(std::memory_order_acquire) : 0U;
      const bool prior_end_of_stream = gate_hint && subscriber.end_of_stream_received;
      auto received =
          transport_lock.owns_lock()
              ? receive_one_with_transport_held(
                    subscriber, &terminal_diagnostic, scan_device, std::nullopt, nullptr, observed)
              : receive_one(
                    subscriber, &terminal_diagnostic, scan_device, std::nullopt, nullptr, observed);
      const bool terminal_received =
          gate_hint && received && !prior_end_of_stream && subscriber.end_of_stream_received;
      if (transport_lock.owns_lock()) {
        transport_lock.unlock();
      }
      if (!received) {
        if (received.error().code == ErrorCode::kStaleEndpoint &&
            subscriber.catalog_tensor_import) {
          // A replacement producer is fenced without consuming its head. Preserve the accepted
          // incarnation until an explicit generation-fenced rebind cuts over to the new one.
          break;
        }
        if (received.error().code == ErrorCode::kNotReady) {
          // The endpoint arbiter owns the authoritative data/terminal transaction. A wait-set,
          // sibling-lane wake, or the device-affine readiness hint above can still precede a
          // commit-ready operation, so an empty staged receive is the bounded loop terminator
          // rather than a session failure.
          break;
        }
        if (received.error().code == ErrorCode::kBackpressured) {
          // Typed retention is full, but receive_one() has retained the transport head either in
          // the backend queue or in the raw-Tensor endpoint's one-record staging claim. Retry
          // after activation retirement frees a slot, while giving other process-local
          // subscribers their bounded opportunity now. Concurrent admission already counted this
          // block in receive_one(), and that count is their only retry signal.
          if (!concurrent_admission_) {
            blocked_transport_retry_required_.store(true, std::memory_order_release);
          }
          count_phase(phase_counters_, RuntimePhaseCounter::kChannelReceiveBackpressured);
          break;
        }
        failed_port = subscriber.target;
        failed_owner = subscriber.owner;
        return make_unexpected(std::move(received).error());
      }
      if (subscriber.process_local_transport) {
        record_process_local_receive();
      }
      if (activation_cut) {
        count_phase(phase_counters_, RuntimePhaseCounter::kChannelActivationCutMaterializations);
      }
      readiness_published = true;
      if (gate_hint && !terminal_received &&
          subscriber.input->readiness_revision.load(std::memory_order_acquire) == prior_revision) {
        count_phase(phase_counters_, RuntimePhaseCounter::kChannelEmitStagedHintsElided);
        continue;
      }
      signal_received_progress(subscriber, receipt, *received);
    }
  }
  return readiness_published;
}

void RuntimeChannelSession::Impl::signal_received_progress(
    SubscriberState& subscriber, const ReceiveReceipt& receipt,
    const ChannelReceiveKind received) noexcept {
  // Attribute only data this receive admitted; terminal/control records carry no receipt.
  const auto observe = [&](const RuntimeFanoutObservationPoint point) noexcept {
    if (receipt.data_sequence != 0U && fanout_observation_enabled()) {
      fanout_test_control_->observe(
          fanout_test_control_->context, subscriber.target, receipt.data_sequence, point);
    }
  };
  observe(RuntimeFanoutObservationPoint::kAdmissionCompleted);
  observe(RuntimeFanoutObservationPoint::kReadyHintStarted);
  signal_owner_receive_progress(subscriber, received);
  observe(RuntimeFanoutObservationPoint::kReadyHintReturned);
}

[[nodiscard]] bool RuntimeChannelSession::Impl::prepare_lane_descriptor_import(
    const std::span<SubscriberState* const> subscribers,
    std::optional<CudaDeviceCursor>& import_device) noexcept {
  if (!descriptor_import_present_) {
    return true;
  }
  if (!initialize_controller_descriptor_import()) {
    return false;
  }
  if (std::ranges::any_of(subscribers, [](const SubscriberState* subscriber) {
        return subscriber->uses_device_affine_descriptor_import();
      })) {
    import_device.emplace();
  }
  return true;
}

void RuntimeChannelSession::Impl::materialize_experimental_publication(
    const DrainScope scope, const std::span<SubscriberState* const> recipients,
    const std::uint64_t publication_sequence) noexcept {
  if (!emit_admission_) {
    return;
  }
  if (scope == DrainScope::kRecipients && recipients.empty()) {
    fail(ErrorCode::kProtocolError);
    return;
  }
  // The backend submission and publication accounting have already finished. Never hold this
  // lock across send: retirement on a consumer lane may need it to release transport pressure.
  // Without it, each recipient's probe and receive still serialize on that subscriber's transport
  // mutex; the retry latch, delivery accounting, and wake publication are atomic.
  std::unique_lock<std::timed_mutex> lock;
  if (!concurrent_admission_) {
    lock = acquire_pump_scan_lock(PumpScanLockSite::kEmit);
  }
  std::optional<CudaDeviceCursor> import_device;
  if (!prepare_lane_descriptor_import(scope == DrainScope::kInventory
                                          ? std::span<SubscriberState* const>{pump_subscribers_}
                                          : recipients,
                                      import_device)) {
    return;
  }
  PortId failed_port;
  VertexId failed_owner;
  std::optional<StructuralDiagnosticId> diagnostic;
  auto drained =
      drain_unmonitored_inputs(failed_port,
                               failed_owner,
                               diagnostic,
                               scope,
                               import_device ? &*import_device : nullptr,
                               true,
                               recipients,
                               concurrent_admission_ && scope == DrainScope::kRecipients);
  if (!drained) {
    // Accepted publication authority cannot become a retryable emit error. Contain admission
    // failure in session status, just as an independently running pump does after an emit.
    fail(drained.error(), failed_owner, diagnostic);
  }
}

[[nodiscard]] bool RuntimeChannelSession::Impl::activation_cut_index_ready() const noexcept {
  if (!activation_cut_index_enabled_ || !activation_cut_index_current_ ||
      activation_cut_pending_subscribers_.empty() || blocked_transport_retry_pending() ||
      !process_local_progress_tracking_valid_.load(std::memory_order_acquire)) {
    return false;
  }
  const std::uint64_t published =
      process_local_published_deliveries_.load(std::memory_order_acquire);
  const std::uint64_t settled = settled_process_local_deliveries();
  return published > settled && published - settled == activation_cut_pending_subscribers_.size();
}

[[nodiscard]] expected<bool, Error> RuntimeChannelSession::Impl::drain_indexed_activation_cut(
    PortId& failed_port, VertexId& failed_owner,
    std::optional<StructuralDiagnosticId>& terminal_diagnostic) noexcept {
  for (SubscriberState* const subscriber : activation_cut_pending_subscribers_) {
    count_phase(phase_counters_, RuntimePhaseCounter::kChannelActivationCutSubscriberProbes);
    ReceiveReceipt receipt;
    auto received = receive_one(*subscriber,
                                &terminal_diagnostic,
                                nullptr,
                                std::nullopt,
                                nullptr,
                                fanout_observation_enabled() ? &receipt : nullptr);
    if (!received) {
      if (received.error().code == ErrorCode::kBackpressured ||
          received.error().code == ErrorCode::kNotReady) {
        activation_cut_pending_subscribers_.clear();
        activation_cut_index_current_ = false;
        if (received.error().code == ErrorCode::kBackpressured) {
          if (!concurrent_admission_) {
            blocked_transport_retry_required_.store(true, std::memory_order_release);
          }
          count_phase(phase_counters_, RuntimePhaseCounter::kChannelReceiveBackpressured);
        }
        return false;
      }
      failed_port = subscriber->target;
      failed_owner = subscriber->owner;
      return make_unexpected(std::move(received).error());
    }
    record_process_local_receive();
    count_phase(phase_counters_, RuntimePhaseCounter::kChannelActivationCutMaterializations);
    signal_received_progress(*subscriber, receipt, *received);
  }
  activation_cut_pending_subscribers_.clear();
  return true;
}

void RuntimeChannelSession::Impl::rearm_activation_cut_index_if_drained() noexcept {
  if (!activation_cut_index_enabled_ || blocked_transport_retry_pending() ||
      !process_local_progress_tracking_valid_.load(std::memory_order_acquire)) {
    return;
  }
  const std::uint64_t published =
      process_local_published_deliveries_.load(std::memory_order_acquire);
  if (settled_process_local_deliveries() == published) {
    activation_cut_pending_subscribers_.clear();
    activation_cut_index_current_ = true;
  }
}

// Keep complete-cut fallback and bounded subscriber retries together so a racing blocked head
// cannot be skipped after retirement, regardless of the scratch-list locking path.
[[nodiscard]] expected<void, Error>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
RuntimeChannelSession::Impl::retry_blocked_process_local_transport(
    const std::span<SubscriberState* const> released, const bool unknown_release) noexcept {
  // A running pump owns retries of the heads it scans: after its bounded wait, or at once when
  // this release wakes it for a head it hides. Only the emit-admitted subscribers it never scans,
  // and the pump-free eventless path, need retirement to establish the follow-up cut synchronously.
  const bool pump_serves_others = background_pump_required();
  if (pump_serves_others) {
    wake_pump_for_blocked_heads(released, unknown_release);
  }
  if (pump_serves_others && !emit_admission_) {
    return {};
  }
  const bool latched = blocked_transport_retry_pending();
  if (released.empty()) {
    // An owner without inputs releases no typed capacity and keeps the latched complete cut. A
    // release on unknown subscribers under concurrent admission always takes that complete,
    // non-skipping cut: a racing block may not be counted yet, but its undelivered head forces
    // the drain.
    if (!latched && !(concurrent_admission_ && unknown_release)) {
      return {};
    }
    return ensure_process_local_activation_cut(RuntimeChannelActivationCutOrigin::kRetirementRetry);
  }
  if (!concurrent_admission_) {
    if (!latched) {
      return {};
    }
    return ensure_process_local_activation_cut(RuntimeChannelActivationCutOrigin::kRetirementRetry);
  }
  // Concurrent admission skips known-blocked heads everywhere else, so this retry of the
  // subscribers whose capacity it released is their only path forward. A block racing this
  // release may not be counted yet, so each released flag is read under the transport mutex that
  // wrote it: either that receive attempt followed this release and saw the freed capacity, or the
  // locked read sees its flag.
  auto current = status();
  if (!current) {
    return make_unexpected(std::move(current).error());
  }
  // The retry runs without the scan lock, as admission does: each receive serializes on its
  // subscriber's transport mutex, and the flags and count are atomic. The scratch list is then
  // private to this call. Released lists are one owner's or one input's subscribers; a larger
  // list keeps the locked scratch list.
  constexpr std::size_t kLocalRetryCapacity = 16U;
  const bool unlocked = released.size() <= kLocalRetryCapacity;
  std::array<SubscriberState*, kLocalRetryCapacity> local_retry{};
  std::size_t local_count = 0U;
  std::unique_lock<std::timed_mutex> lock;
  if (!unlocked) {
    lock = acquire_pump_scan_lock(PumpScanLockSite::kCut);
    scoped_retry_subscribers_.clear();
  }
  for (SubscriberState* const subscriber : released) {
    // Only a transport receive can block a head, so a subscriber outside the transport inventory
    // (direct-retained delivery) needs no locked read.
    if (!subscriber->transport_inventory || (pump_serves_others && !subscriber->emit_admitted) ||
        !transport_head_known_blocked(*subscriber)) {
      continue;
    }
    if (unlocked) {
      local_retry[local_count++] = subscriber;
    } else {
      scoped_retry_subscribers_.push_back(subscriber);
    }
  }
  const std::span<SubscriberState* const> retry =
      unlocked ? std::span<SubscriberState* const>{local_retry.data(), local_count}
               : std::span<SubscriberState* const>{scoped_retry_subscribers_};
  if (retry.empty()) {
    // Account only retirements that an outstanding blocked head sent here.
    if (latched) {
      count_phase(phase_counters_, RuntimePhaseCounter::kChannelActivationCuts);
      count_phase(phase_counters_, RuntimePhaseCounter::kChannelActivationCutsRetirementRetry);
      count_phase(phase_counters_, RuntimePhaseCounter::kChannelActivationCutSkips);
      count_phase(phase_counters_, RuntimePhaseCounter::kChannelActivationCutScopedSkips);
    }
    return {};
  }
  count_phase(phase_counters_, RuntimePhaseCounter::kChannelActivationCuts);
  count_phase(phase_counters_, RuntimePhaseCounter::kChannelActivationCutsRetirementRetry);
  count_phase(phase_counters_, RuntimePhaseCounter::kChannelActivationCutScopedRetries);
  std::optional<CudaDeviceCursor> import_device;
  if (!prepare_lane_descriptor_import(retry, import_device)) {
    return status();
  }
  PortId failed_port;
  VertexId failed_owner;
  std::optional<StructuralDiagnosticId> terminal_diagnostic;
  auto drained = drain_unmonitored_inputs(failed_port,
                                          failed_owner,
                                          terminal_diagnostic,
                                          DrainScope::kRecipients,
                                          import_device ? &*import_device : nullptr,
                                          true,
                                          retry);
  if (!drained) {
    fail(drained.error(), failed_owner, terminal_diagnostic);
    return make_unexpected(std::move(drained).error());
  }
  return status();
}

void RuntimeChannelSession::Impl::record_process_local_publication(
    const std::uint32_t deliveries,
    const std::span<SubscriberState* const> activation_cut_subscribers) noexcept {
  record_process_local_progress(process_local_published_deliveries_, deliveries);
  if (!activation_cut_index_enabled_) {
    return;
  }
  if (!activation_cut_index_current_ || activation_cut_subscribers.size() != deliveries ||
      activation_cut_pending_subscribers_.capacity() - activation_cut_pending_subscribers_.size() <
          activation_cut_subscribers.size()) {
    activation_cut_pending_subscribers_.clear();
    activation_cut_index_current_ = false;
    return;
  }
  activation_cut_pending_subscribers_.insert(activation_cut_pending_subscribers_.end(),
                                             activation_cut_subscribers.begin(),
                                             activation_cut_subscribers.end());
}

void RuntimeChannelSession::Impl::record_process_local_receive() noexcept {
  record_process_local_progress(process_local_received_deliveries_, 1U);
}

void RuntimeChannelSession::Impl::record_process_local_replacement() noexcept {
  record_process_local_progress(process_local_replaced_deliveries_, 1U);
}

void RuntimeChannelSession::Impl::record_process_local_discards(
    const std::uint32_t deliveries) noexcept {
  record_process_local_progress(process_local_discarded_deliveries_, deliveries);
}

void RuntimeChannelSession::Impl::invalidate_process_local_progress() noexcept {
  process_local_progress_tracking_valid_.store(false, std::memory_order_release);
  if (activation_cut_index_enabled_) {
    activation_cut_pending_subscribers_.clear();
    activation_cut_index_current_ = false;
  }
}

void RuntimeChannelSession::Impl::record_process_local_progress(
    std::atomic<std::uint64_t>& counter, const std::uint32_t delta) noexcept {
  std::uint64_t current = counter.load(std::memory_order_relaxed);
  while (current <= std::numeric_limits<std::uint64_t>::max() - delta) {
    if (counter.compare_exchange_weak(
            current, current + delta, std::memory_order_release, std::memory_order_relaxed)) {
      return;
    }
  }
  process_local_progress_tracking_valid_.store(false, std::memory_order_release);
}

[[nodiscard]] expected<void, Error>
RuntimeChannelSession::Impl::wait_for_transport_activity() noexcept {
  if (wait_attachment_count_ == 0U) {
    std::this_thread::sleep_for(kPumpWait);
    return {};
  }
  count_phase(phase_counters_, RuntimePhaseCounter::kChannelPumpWaits);
  auto waited = wait_set_->wait_for(kPumpWait);
  if (!waited) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR("Channel pump WaitSet wait failed with {}",
                                 error_code_name(waited.error().code));
    return make_unexpected(std::move(waited).error());
  }
  if (!*waited) {
    mark_pump_reported_ready({});
    count_phase(phase_counters_, RuntimePhaseCounter::kChannelPumpWaitTimeouts);
    if (blocked_transport_retry_pending()) {
      count_phase(phase_counters_, RuntimePhaseCounter::kChannelPumpWaitTimeoutsWithBlockedHead);
    }
    return {};
  }
  count_phase(phase_counters_, RuntimePhaseCounter::kChannelPumpWaitsReady);
  auto ready = wait_set_->ready_handles(ready_handles_);
  if (!ready) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR("Channel pump WaitSet ready-handle collection failed with {}",
                                 error_code_name(ready.error().code));
    return make_unexpected(std::move(ready).error());
  }
  for (std::size_t index = 0U; index < *ready; ++index) {
    if (!wait_subscriber_by_handle_.contains(ready_handles_[index])) {
      HOLOSCAN_LOG_AUTOMATIC_ERROR("Channel WaitSet returned unknown readiness handle {}",
                                   ready_handles_[index]);
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
  }
  mark_pump_reported_ready(std::span<const std::uint64_t>{ready_handles_.data(), *ready});
  trace_phase(phase_counters_, RuntimePhaseTracePoint::kChannelTransportWaitReturned);
  return {};
}

[[nodiscard]] bool RuntimeChannelSession::Impl::enter_pump_support_thread() noexcept {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    auto placement = apply_support_thread_affinity(support_thread_affinities_,
                                                   SupportThreadRole::kChannelPump,
                                                   &support_thread_placement_failure_);
    if (!placement) {
      HOLOSCAN_LOG_AUTOMATIC_ERROR("Channel pump thread placement failed with {}: {}",
                                   error_code_name(placement.error().code),
                                   placement.error().message);
      fail(placement.error(), {}, StructuralDiagnosticId::kRuntimeStartupFailed);
      return false;
    }
    if (support_thread_control_ == nullptr) {
      return true;
    }
    auto entered = support_thread_control_->enter(SupportThreadRole::kChannelPump);
    if (!entered) {
      HOLOSCAN_LOG_AUTOMATIC_ERROR("Channel pump thread setup failed with {}: {}",
                                   error_code_name(entered.error().code),
                                   entered.error().message);
      fail(entered.error(), {}, StructuralDiagnosticId::kRuntimeStartupFailed);
      return false;
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (const std::bad_alloc&) {
    fail(Error{ErrorCode::kResourceExhausted}, {}, StructuralDiagnosticId::kRuntimeStartupFailed);
    return false;
  } catch (...) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR("Channel pump thread setup raised an exception");
    fail(Error{ErrorCode::kFailure}, {}, StructuralDiagnosticId::kRuntimeStartupFailed);
    return false;
  }
#endif
  return true;
}

namespace {

// Process-unique descriptor-import inventory tokens; zero means none has been frozen.
[[nodiscard]] std::uint64_t allocate_descriptor_import_token() noexcept {
  static std::atomic<std::uint64_t> next_token{1U};
  return next_token.fetch_add(1U, std::memory_order_relaxed);
}

// CUDA binds its current context per thread. Owner polls and activation cuts on different lanes
// may import at the same time, so each thread records the inventories it has initialized instead of
// sharing one session field. An evicted token only repeats the idempotent initialization.
struct DescriptorImportThreadRecord {
  std::array<std::uint64_t, 8U> tokens{};
  std::size_t next{};
};

[[nodiscard]] DescriptorImportThreadRecord& descriptor_import_record_on_this_thread() noexcept {
  static thread_local DescriptorImportThreadRecord record;
  return record;
}

[[nodiscard]] bool descriptor_import_initialized_on_this_thread(
    const std::uint64_t token) noexcept {
  const auto& tokens = descriptor_import_record_on_this_thread().tokens;
  return std::ranges::find(tokens, token) != tokens.end();
}

void remember_descriptor_import_on_this_thread(const std::uint64_t token) noexcept {
  auto& record = descriptor_import_record_on_this_thread();
  record.tokens[record.next] = token;
  record.next = (record.next + 1U) % record.tokens.size();
}

}  // namespace

void RuntimeChannelSession::Impl::freeze_descriptor_import_inventory() {
  const bool previous_present = descriptor_import_present_;
  const std::vector<DeviceId> previous_devices = descriptor_import_devices_;
  descriptor_import_present_ = false;
  descriptor_import_devices_.clear();
  descriptor_import_devices_.reserve(subscribers_.size());
  for (const auto& subscriber : subscribers_) {
    if (!subscriber->requires_pump_vmm_import_context()) {
      continue;
    }
    descriptor_import_present_ = true;
    if (subscriber->uses_device_affine_descriptor_import()) {
      descriptor_import_devices_.push_back(subscriber->tensor_device);
    }
  }
  std::ranges::sort(descriptor_import_devices_, {}, &DeviceId::value);
  const auto unique = std::ranges::unique(descriptor_import_devices_);
  descriptor_import_devices_.erase(unique.begin(), unique.end());
  if (descriptor_import_token_ == 0U || descriptor_import_present_ != previous_present ||
      descriptor_import_devices_ != previous_devices) {
    descriptor_import_token_ = allocate_descriptor_import_token();
  }
}

[[nodiscard]] bool RuntimeChannelSession::Impl::initialize_descriptor_import(
    CudaDeviceCursor& device) noexcept {
  bool initialized = device.active();
  for (const DeviceId import_device : descriptor_import_devices_) {
    if (!initialized) {
      break;
    }
    initialized = device.initialize(import_device);
  }
  if (initialized) {
    initialized = descriptor_import_devices_.empty()
                      ? device.initialize()
                      : device.select(descriptor_import_devices_.front());
  }
  return initialized;
}

[[nodiscard]] bool RuntimeChannelSession::Impl::initialize_pump_descriptor_import(
    std::optional<CudaDeviceCursor>& pump_device) noexcept {
  if (!descriptor_import_present_) {
    return true;
  }

  pump_device.emplace();
  const bool initialized = initialize_descriptor_import(*pump_device);
  if (!initialized) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR("Channel pump could not initialize CUDA VMM descriptor import");
    fail(ErrorCode::kNotSupported);
    return false;
  }
  return true;
}

[[nodiscard]] bool RuntimeChannelSession::Impl::initialize_controller_descriptor_import() noexcept {
  if (!descriptor_import_present_) {
    return true;
  }
  if (descriptor_import_initialized_on_this_thread(descriptor_import_token_)) {
    return true;
  }

  // CUDA context state is thread-local. Activation cuts and owner polls may import on threads
  // other than the pump, so initialize every compiler-frozen raw-Tensor device once per thread
  // and frozen inventory before any descriptor import can occur.
  CudaDeviceCursor controller_device;
  if (!initialize_descriptor_import(controller_device) || !controller_device.restore()) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR(
        "Channel controller could not initialize CUDA VMM descriptor import");
    fail(ErrorCode::kNotSupported);
    return false;
  }
  remember_descriptor_import_on_this_thread(descriptor_import_token_);
  controller_descriptor_import_initialization_count_.fetch_add(1U, std::memory_order_relaxed);
  return true;
}

void RuntimeChannelSession::Impl::pump() noexcept {
  if (!enter_pump_support_thread()) {
    pump_startup_.release();
    return;
  }
  // HoloMQ descriptor import uses CUDA VMM even for pageable-host payload storage. CUDA Runtime
  // context state is thread-local, so cold startup attaches the pump to every compiler-frozen
  // raw-Tensor device and leaves the first one selected. Runtime owns that planned/current-device
  // selection; each HoloMQ handle carries the exporter's ordinal so the import layer can verify
  // the physical allocation location. The scan cursor changes devices only at an actual
  // plan-ordered transition and restores the thread's ambient device once.
  std::optional<CudaDeviceCursor> pump_device;
  if (!initialize_pump_descriptor_import(pump_device)) {
    pump_startup_.release();
    return;
  }
  // arm() owns the cold startup barrier. Release it only after support-thread control and every
  // frozen raw-Tensor import device are initialized so an unavailable device cannot race the
  // first activation.
  pump_startup_.release();
  while (!cancel_requested_.load(std::memory_order_acquire)) {
    auto waited = wait_for_transport_activity();
    if (!waited) {
      fail(waited.error().code);
      return;
    }
    bool readiness_published = false;
    {
      // Scan every finite process-local endpoint after each bounded wait, including a timeout.
      // The cut mutex makes the whole scan indivisible with respect to the serial controller's
      // post-compute flush, so the controller never freezes a prefix produced by one activation.
      const auto cut_lock = acquire_pump_scan_lock(PumpScanLockSite::kPump);
      PortId failed_port;
      VertexId failed_owner;
      std::optional<StructuralDiagnosticId> terminal_diagnostic;
      auto drained = drain_unmonitored_inputs(failed_port,
                                              failed_owner,
                                              terminal_diagnostic,
                                              DrainScope::kPumpScan,
                                              pump_device.has_value() ? &*pump_device : nullptr);
      if (!drained) {
        HOLOSCAN_LOG_AUTOMATIC_ERROR("Channel subscriber drain failed with {} on port {}",
                                     runtime_error_message(drained.error()),
                                     failed_port.value());
        fail(drained.error(), failed_owner, terminal_diagnostic);
        return;
      }
      readiness_published = *drained;
    }
    // Coalesce every typed sample retained by this finite scan into one durable controller hint.
    // Exact readiness cardinality remains owned by the revision-gated input binding.
    if (readiness_published && !owner_progress_wake_enabled() &&
        readiness_wake_.signal != nullptr) {
      trace_phase(phase_counters_, RuntimePhaseTracePoint::kChannelReadinessWakeSignaled);
      readiness_wake_.signal(readiness_wake_.state);
    }
  }
}

void RuntimeChannelSession::Impl::pump_with_thread_usage() noexcept {
  const RuntimeThreadUsageScope thread_usage{phase_counters_, RuntimeThreadUsageRole::kChannelPump};
  pump();
}

expected<void, Error> RuntimeChannelSession::refresh_connections() noexcept {
  if (!impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  for (const auto& output : impl_->outputs_) {
    auto refreshed = output->refresh_connections();
    if (!refreshed) {
      return make_unexpected(std::move(refreshed).error());
    }
  }
  return {};
}

expected<void, Error> RuntimeChannelSession::arm() noexcept {
  if (!impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  auto current = impl_->status();
  if (!current) {
    return make_unexpected(std::move(current).error());
  }
  if (impl_->armed_.exchange(true, std::memory_order_acq_rel) || impl_->pump_thread_.joinable()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  impl_->cancel_requested_.store(false, std::memory_order_release);
  impl_->closed_.store(false, std::memory_order_release);
  impl_->terminal_owner_.store(0U, std::memory_order_relaxed);
  impl_->terminal_diagnostic_.store(0U, std::memory_order_relaxed);
  impl_->terminal_error_.store(ErrorCode::kSuccess, std::memory_order_release);
  impl_->terminal_failure_claimed_.store(false, std::memory_order_release);
  // Keep a session with no autonomous pump work logically armed without starting a worker.
  // Direct-retained inputs need no transport drain, externally driven imports are controller
  // owned, and an all-eventless process-local inventory is drained by the successful-compute cut.
  impl_->concurrent_admission_ = false;
  // Delivery inside emit() admits on producer lanes without the scan lock whenever any service is
  // emit-admitted, whether or not a pump serves the others.
  if (impl_->emit_admission_) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      impl_->scoped_retry_subscribers_.reserve(impl_->pump_subscribers_.size());
      impl_->concurrent_admission_ = true;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (const std::bad_alloc&) {
      impl_->armed_.store(false, std::memory_order_release);
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
#endif
  }
  if (!impl_->background_pump_required()) {
    impl_->activation_cut_pending_subscribers_.clear();
    // Keep the mutable exact-cut scratch on the single serialized owner. Multi-partition
    // sessions retain the thread-safe complete-inventory scan; owner-progress routing is
    // orthogonal and may also be installed for serialized eventless sessions.
    impl_->activation_cut_index_enabled_ = impl_->activation_cut_index_available_ &&
                                           !impl_->track_concurrent_process_local_publications_;
    impl_->activation_cut_index_current_ = true;
    // The former background pump initialized every descriptor-import context before releasing its
    // startup barrier. Preserve that fail-fast behavior on the controller thread before any
    // eventless source callback can publish a descriptor for the synchronous cut.
    if (!impl_->initialize_controller_descriptor_import()) {
      impl_->armed_.store(false, std::memory_order_release);
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    return {};
  }
  impl_->activation_cut_index_enabled_ = false;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    // Keep ordinary sessions on the original entry point. Thread-usage accounting is selected
    // once during cold startup and adds no scope or sampling work to the uninstrumented loop.
    if (impl_->phase_counters_ != nullptr) {
      impl_->pump_thread_ = std::thread{[impl = impl_.get()] { impl->pump_with_thread_usage(); }};
    } else {
      impl_->pump_thread_ = std::thread{[impl = impl_.get()] { impl->pump(); }};
    }
    impl_->pump_startup_.acquire();
    auto started = impl_->status();
    if (!started) {
      impl_->pump_thread_.join();
      impl_->armed_.store(false, std::memory_order_release);
      return make_unexpected(std::move(started).error());
    }
    return started;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    impl_->armed_.store(false, std::memory_order_release);
    impl_->closed_.store(true, std::memory_order_release);
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
#endif
}

expected<void, Error> RuntimeChannelSession::flush_process_local_activation_cut(
    const RuntimeChannelActivationCutOrigin origin) noexcept {
  if (!impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return impl_->ensure_process_local_activation_cut(origin);
}

void RuntimeChannelSession::defer_activation_cut(const VertexId producer) noexcept {
  if (impl_) {
    impl_->defer_activation_cut(producer);
  }
}

expected<void, Error> RuntimeChannelSession::flush_process_local_activation_cut(
    const VertexId producer) noexcept {
  if (!impl_ || !producer) {
    return make_unexpected(Error{impl_ ? ErrorCode::kInvalidArgument : ErrorCode::kClosed});
  }
  return impl_->ensure_process_local_activation_cut(RuntimeChannelActivationCutOrigin::kPostCompute,
                                                    producer);
}

[[nodiscard]] expected<void, Error> RuntimeChannelSession::Impl::retain_direct_inline_payload(
    SubscriberState& subscriber, std::span<const std::byte> payload,
    SampleMetadata metadata) noexcept {
  // REQ: CHAN-096
  // REQ: CHAN-097
  const bool generated_schema =
      subscriber.input->type_ops->codec_family == RuntimePayloadCodecFamily::kGeneratedSchema;
  const bool payload_size_valid = generated_schema ? payload.size() >= kSchemaEnvelopeHeaderSize &&
                                                         payload.size() <= subscriber.scratch_size
                                                   : payload.size() == subscriber.scratch_size;
  if (!subscriber.direct_process_local || subscriber.endpoint != nullptr ||
      subscriber.tensor_endpoint.has_value() || subscriber.tensor_schema_endpoint != nullptr ||
      generated_schema != subscriber.process_local_schema_identity.has_value() ||
      !payload_size_valid) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  std::size_t schema_payload_size = 0U;
  if (generated_schema) {
    // Validate the exact incoming frame before copying it into capacity-sized scratch storage.
    // store_received() reuses the decoded payload span below instead of decoding the envelope
    // again.
    auto decoded =
        channel_internal::decode_schema_envelope(payload,
                                                 *subscriber.process_local_schema_identity,
                                                 subscriber.input->type_ops->wire_capacity);
    if (!decoded) {
      return make_unexpected(std::move(decoded).error());
    }
    if (*decoded != payload.size() - kSchemaEnvelopeHeaderSize) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    schema_payload_size = *decoded;
  }
  // The producer's span outlives this call and store_received() copies it into the typed sample,
  // so it is passed as is rather than through the subscriber's scratch.
  const std::span<const std::byte> validated_payload =
      generated_schema ? payload.subspan(kSchemaEnvelopeHeaderSize, schema_payload_size) : payload;
  expected<void, Error> stored;
  DeferredSampleReleases releases{*subscriber.input->type_ops, subscriber.slots.size() + 1U};
  {
    const auto cache_lock =
        acquire_admission_cache_lock(subscriber, InputCacheAdmissionSite::kDirect);
    stored = store_received(subscriber,
                            releases,
                            metadata,
                            EndpointClockHealthDisposition::kNotApplicable,
                            0U,
                            validated_payload);
  }
  return stored;
}

[[nodiscard]] expected<void, Error> RuntimeChannelSession::Impl::store_direct_object_payload(
    SubscriberState& subscriber, DeferredSampleReleases& releases, const void* payload,
    SampleMetadata metadata) noexcept {
  if (payload == nullptr || !subscriber.direct_process_local || subscriber.endpoint != nullptr ||
      subscriber.tensor_endpoint.has_value() || subscriber.tensor_schema_endpoint != nullptr ||
      subscriber.input->type_ops->codec_family != RuntimePayloadCodecFamily::kProcessLocalObject ||
      subscriber.scratch_size != 0U || subscriber.process_local_schema_identity.has_value()) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  expected<void, Error> stored;
  {
    const auto cache_lock =
        acquire_admission_cache_lock(subscriber, InputCacheAdmissionSite::kDirect);
    stored = store_received(subscriber,
                            releases,
                            metadata,
                            EndpointClockHealthDisposition::kNotApplicable,
                            0U,
                            {},
                            nullptr,
                            nullptr,
                            false,
                            payload);
  }
  return stored;
}

[[nodiscard]] expected<void, Error> RuntimeChannelSession::Impl::retain_direct_tensor_sample(
    SubscriberState& subscriber, Sample<Tensor> received) noexcept {
  // REQ: CHAN-098
  if (!subscriber.direct_process_local || subscriber.endpoint != nullptr ||
      subscriber.tensor_endpoint.has_value() || subscriber.tensor_schema_endpoint != nullptr ||
      subscriber.input->type_ops != runtime_port_type_ops<Tensor>()) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  expected<void, Error> stored;
  DeferredSampleReleases releases{*subscriber.input->type_ops, subscriber.slots.size() + 1U};
  {
    const auto cache_lock =
        acquire_admission_cache_lock(subscriber, InputCacheAdmissionSite::kDirect);
    stored = store_received_tensor(subscriber, releases, std::move(received));
  }
  return stored;
}

[[nodiscard]] expected<void, Error> RuntimeChannelSession::Impl::retain_direct_tensor_schema_sample(
    SubscriberState& subscriber, const std::span<const std::byte> encoded,
    const std::span<const Tensor> received) noexcept {
  // REQ: CHAN-099
  const RuntimePortTypeOps& ops = *subscriber.input->type_ops;
  if (!subscriber.direct_process_local || subscriber.endpoint != nullptr ||
      subscriber.tensor_endpoint.has_value() || subscriber.tensor_schema_endpoint != nullptr ||
      ops.codec_family != RuntimePayloadCodecFamily::kDescriptorTensorSchema ||
      ops.descriptor_tensor_sample_metadata == nullptr ||
      ops.assign_descriptor_tensors_sample == nullptr ||
      received.size() != ops.descriptor_tensor_static_count ||
      received.size() > channel_internal::kMaximumDescriptorTensorFields || encoded.empty() ||
      encoded.size() > ops.wire_capacity) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  auto& retained = subscriber.tensor_schema_scratch;
  if (retained.size() != received.size()) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  const TensorScratchReset reset{retained};
  std::copy(received.begin(), received.end(), retained.begin());
  expected<void, Error> stored;
  DeferredSampleReleases releases{*subscriber.input->type_ops, subscriber.slots.size() + 1U};
  {
    const auto cache_lock =
        acquire_admission_cache_lock(subscriber, InputCacheAdmissionSite::kDirect);
    auto metadata = ops.descriptor_tensor_sample_metadata(encoded, 0U, plan_generation_);
    if (!metadata) {
      return make_unexpected(std::move(metadata).error());
    }
    stored = store_received_tensor_schema(
        subscriber, releases, encoded, std::span<Tensor>{retained}, std::move(*metadata));
  }
  return stored;
}

}  // namespace holoscan::detail
