/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/diagnostic_reporting.hpp"
#include "endpoint_clock_health_runtime.hpp"
#include "runtime_channel_binding.hpp"
#include "runtime_channel_session_impl.hpp"

namespace holoscan::detail {

// RuntimeChannelSession facade, terminal status, cancellation, join, and teardown glue.

static_assert(std::is_trivially_copyable_v<RuntimeChannelDirectContinuationReceipt>);

void RuntimeChannelSession::Impl::cancel() noexcept {
  cancel_requested_.store(true, std::memory_order_release);
  closed_.store(true, std::memory_order_release);
}

[[nodiscard]] expected<void, Error> RuntimeChannelSession::Impl::join() noexcept {
  cancel();
  if (pump_thread_.joinable()) {
    if (pump_thread_.get_id() == std::this_thread::get_id()) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    pump_thread_.join();
  }
  armed_.store(false, std::memory_order_release);
  return status();
}

void RuntimeChannelSession::Impl::reset_sample_slot(SampleSlot& slot,
                                                    DeferredSampleReleases& releases) noexcept {
  if (slot.occupied) {
    releases.retain(slot.sample.data());
    if (slot.sample_storage_prepared) {
      slot.subscriber->input->type_ops->reset_sample_storage(slot.sample.data());
    } else {
      slot.subscriber->input->type_ops->destroy_sample(slot.sample.data());
    }
  }
  slot.occupied = false;
  slot.event_emitted = false;
  slot.witness_retired = false;
  slot.active_selected = false;
  slot.reuse_retained = false;
  slot.temporal_retained = false;
  slot.retirement_selected = false;
  slot.typed_ingress_reserved = false;
  slot.readiness_sequence = 0U;
  slot.arrival_order = 0U;
  slot.clock_health_disposition = EndpointClockHealthDisposition::kNotApplicable;
  slot.clock_health_generation = 0U;
  slot.direct_ingress_unread = false;
  slot.staged_arrival_evidence = false;
  // Cleared with the rest of the slot identity. Leaving it set would attribute the next
  // occupant's fate to a claim that was released for a record no longer here.
  slot.owed_to_released_claim = false;
  slot.refill_reserved = false;
  slot.refill_prefix_reserved = false;
  slot.temporal = {};
}

RuntimeChannelSession::RuntimeChannelSession(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

RuntimeChannelSession::~RuntimeChannelSession() = default;

expected<void, Error> RuntimeChannelSession::bind_clock_health(
    EndpointClockHealthRuntimeCoordinator& coordinator, std::mutex& serialization_mutex) noexcept {
  if (!impl_ || impl_->armed_.load(std::memory_order_acquire) || impl_->pump_thread_.joinable()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (impl_->clock_health_serialization_mutex_ != nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

  // Validate the complete cold binding before publishing any coordinator pointer. A partial
  // binding could otherwise let one external import advance while another fails closed.
  for (const auto& subscriber : impl_->subscribers_) {
    if (!subscriber->clock_health_endpoint) {
      if (subscriber->clock_health_coordinator != nullptr ||
          subscriber->health_dispatch_budget != 0U) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      continue;
    }
    if (subscriber->clock_health_coordinator != nullptr ||
        subscriber->health_dispatch_budget == 0U ||
        coordinator.receiver(subscriber->clock_health_endpoint) == nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
  }
  for (const auto& subscriber : impl_->subscribers_) {
    if (subscriber->clock_health_endpoint) {
      subscriber->clock_health_coordinator = &coordinator;
    }
  }
  if (coordinator.receiver_count() != 0U) {
    impl_->clock_health_serialization_mutex_ = &serialization_mutex;
  }
  return {};
}

expected<void, Error> RuntimeChannelSession::bind_publish_clock(ClockRef clock) noexcept {
  if (!impl_ || impl_->armed_.load(std::memory_order_acquire) || impl_->pump_thread_.joinable()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (impl_->publish_clock_.id().valid()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  // An inert reference is the clockless plan, which has no clock to attribute a publication to.
  // Leaving publication time absent is the correct answer there, so it is not an error.
  if (!clock.id().valid()) {
    return {};
  }
  // A resolved clock that cannot be read is refused here rather than at the first publication.
  // Discovering it later means the sample has already been accepted, leaving no outcome except a
  // terminal error on an otherwise valid graph.
  if (!clock.capabilities().can_read_now()) {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }
  // An exported coordinate is stamped from this clock but named by the catalog record the compiler
  // projected for the plan's root default clock. Those are two derivations of one identity, so a
  // caller that binds any other clock would ship a coordinate attributed to a domain it was not
  // measured on. GraphExec::reconstruct already refuses to start when its default clock disagrees
  // with the plan, which makes this unreachable today; it is checked here so a future composition
  // that resolves a different clock per executor fails at the binding rather than on the wire.
  if (impl_->plan_publish_clock_.valid() && clock.id() != impl_->plan_publish_clock_) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR(
        "Channel publish clock {} does not match the plan root default clock {} the endpoint "
        "catalog projected",
        clock.id().value,
        impl_->plan_publish_clock_.value);
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  impl_->publish_clock_ = std::move(clock);
  return {};
}

bool RuntimeChannelSession::restore_input_binding(const PortId input) noexcept {
  if (impl_ == nullptr || !input) {
    return false;
  }
  Impl::InputPortState* const state = impl_->find_input(input);
  if (state == nullptr || state->declaration == nullptr || state->type_ops == nullptr ||
      state->type_ops->bind_input == nullptr) {
    return false;
  }
  state->type_ops->bind_input(state->declaration, &state->binding);
  return true;
}

bool RuntimeChannelSession::restore_output_binding(const PortId output) noexcept {
  if (impl_ == nullptr || !output) {
    return false;
  }
  Impl::OutputState* const state = impl_->find_output(output);
  if (state == nullptr || state->declaration == nullptr || state->type_ops == nullptr ||
      state->type_ops->bind_output == nullptr) {
    return false;
  }
  state->type_ops->bind_output(state->declaration, &state->binding);
  return true;
}

bool RuntimeChannelSession::require_fixed_tensor(const PortId input, const void* address,
                                                 std::span<const std::uint64_t> shape) noexcept {
  if (impl_ == nullptr || !input || address == nullptr) {
    return false;
  }
  Impl::InputPortState* const state = impl_->find_input(input);
  if (state == nullptr || state->type_ops != runtime_port_type_ops<Tensor>() ||
      (state->required_tensor_address != nullptr && state->required_tensor_address != address)) {
    return false;
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    // This optional is set only through the source-tree Runtime test seam. Keeping the check in
    // the production translation unit lets real RunSession startup exercise this noexcept path.
    if (impl_->fixed_tensor_shape_allocations_before_failure_for_test_.has_value()) {
      auto& remaining = *impl_->fixed_tensor_shape_allocations_before_failure_for_test_;
      if (remaining == 0U) {
        impl_->fixed_tensor_shape_allocations_before_failure_for_test_.reset();
        throw std::bad_alloc{};
      }
      --remaining;
    }
#endif
    std::vector<std::uint64_t> required_shape{shape.begin(), shape.end()};
    state->required_tensor_shape.swap(required_shape);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (const std::bad_alloc&) {
    return false;
  }
#endif
  // Publish the address only after the owned shape is complete so rejection leaves no partial
  // fixed-input requirement for startup rollback to discover.
  state->required_tensor_address = address;
  return true;
}

bool RuntimeChannelSession::clear_fixed_tensor_requirement(const PortId input,
                                                           const void* address) noexcept {
  if (impl_ == nullptr || !input || address == nullptr) {
    return false;
  }
  Impl::InputPortState* const state = impl_->find_input(input);
  if (state == nullptr || state->required_tensor_address != address) {
    return false;
  }
  state->required_tensor_address = nullptr;
  state->required_tensor_shape.clear();
  return true;
}

expected<void, RuntimeFixtureObservationBindingFailure>
RuntimeChannelSession::observe_fixture_publications(
    std::shared_ptr<RuntimeFixtureObservation> observation) noexcept {
  if (!impl_ || observation == nullptr || impl_->fixture_observation_ != nullptr ||
      impl_->armed_.load(std::memory_order_acquire) || impl_->pump_thread_.joinable() ||
      impl_->closed_.load(std::memory_order_acquire)) {
    return make_unexpected(
        RuntimeFixtureObservationBindingFailure{.code = ErrorCode::kInvalidArgument});
  }
  for (const auto& output : impl_->outputs_) {
    if (output->type_ops->codec_family != RuntimePayloadCodecFamily::kInlinePod) {
      return make_unexpected(RuntimeFixtureObservationBindingFailure{
          .code = ErrorCode::kIncompatible, .owner = output->owner, .port = output->port});
    }
  }
  // Quarantine already retains this complete channel owner. Keep the observation in the same
  // closure so failed stale emits cannot dereference storage destroyed with the RunSession.
  impl_->fixture_observation_ = std::move(observation);
  for (const auto& output : impl_->outputs_) {
    output->binding.ops = &Impl::OutputState::kObservedBindingOps;
  }
  return {};
}

void RuntimeChannelSession::cancel() noexcept {
  if (impl_) {
    impl_->cancel();
  }
}

expected<void, Error> RuntimeChannelSession::join() noexcept {
  return impl_ ? impl_->join() : expected<void, Error>{make_unexpected(Error{ErrorCode::kClosed})};
}

expected<void, Error> RuntimeChannelSession::status() const noexcept {
  return impl_ ? impl_->status()
               : expected<void, Error>{make_unexpected(Error{ErrorCode::kClosed})};
}

std::optional<StructuralDiagnosticId> RuntimeChannelSession::terminal_diagnostic() const noexcept {
  if (!impl_) {
    return std::nullopt;
  }
  if (impl_->terminal_error_.load(std::memory_order_acquire) == ErrorCode::kSuccess) {
    return std::nullopt;
  }
  const std::uint32_t diagnostic = impl_->terminal_diagnostic_.load(std::memory_order_relaxed);
  return diagnostic == 0U ? std::nullopt
                          : std::optional{static_cast<StructuralDiagnosticId>(diagnostic)};
}

const RuntimeSupportThreadPlacementFailure*
RuntimeChannelSession::support_thread_placement_failure() const noexcept {
  if (!impl_ || impl_->terminal_error_.load(std::memory_order_acquire) == ErrorCode::kSuccess ||
      !impl_->support_thread_placement_failure_.present) {
    return nullptr;
  }
  return &impl_->support_thread_placement_failure_;
}

VertexId RuntimeChannelSession::terminal_owner() const noexcept {
  if (!impl_ || impl_->terminal_error_.load(std::memory_order_acquire) == ErrorCode::kSuccess) {
    return {};
  }
  return VertexId{impl_->terminal_owner_.load(std::memory_order_relaxed)};
}

expected<void, Error> RuntimeChannelSession::set_teardown_observer_for_test(
    void* const state, const RuntimeChannelTeardownObserver observer) noexcept {
  if (!impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return impl_->set_teardown_observer_for_test(state, observer);
}

expected<void, Error> RuntimeChannelSession::fail_next_activation_retirement_for_test() noexcept {
  if (!impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return impl_->fail_next_activation_retirement_for_test();
}

}  // namespace holoscan::detail
