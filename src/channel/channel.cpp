/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "channel_common.hpp"
#include "channel_inline_payload.hpp"
#include "channel_runtime_impl.hpp"
#include "channel_subscriber_wait_context.hpp"
#include "channel_tensor_schema_payload.hpp"

#include <holomq/context.hpp>
#include <holomq/context_builder.hpp>
#include <holomq/wait_trigger.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <new>
#include <optional>
#include <stop_token>
#include <string_view>
#include <utility>

namespace holoscan {
namespace channel_internal {
namespace {

/// Upper bound on one blocking pass, which makes this wait event-driven with a bounded recovery
/// poll rather than purely event-driven. Three things currently depend on that bound, and it may
/// not be raised or removed until all three are addressed:
///
/// - the publisher posts its transport notification best effort and discards the result
///   (publisher_port.hpp), so a dropped wake has to cost bounded latency rather than a stalled
///   wait;
/// - the cancellation callback below discards WaitTrigger::fire() failure. The latch is published
///   before the write, so the trigger->is_set() recheck at the bottom of the loop still catches it
///   -- but only once this bound expires;
/// - a subscriber's clock-health lane is reported by has_samples() but is not attached to the wait
///   set (channel_tensor_schema_payload.cpp), so health-only readiness cannot wake this thread and
///   is discovered on the fallback pass instead.
///
/// The cost of the fallback is one wakeup per second per idle waiting subscriber, which is what
/// keeps a fully idle process out of deep-idle residency. That is the documented contract for now;
/// see public/docs/examples/external_clients.mdx. It also keeps a deadline far in the future from
/// becoming an unbounded kernel wait.
constexpr std::chrono::nanoseconds kMaxIterationWait{std::chrono::seconds{1}};

/// Allocation-free stop callback. A capturing lambda would force std::function into the
/// std::stop_callback type and put the warmed wait path at the mercy of small-buffer optimization.
struct TriggerFirer {
  holomq::WaitTrigger* trigger;

  void operator()() const noexcept { static_cast<void>(trigger->fire()); }
};

}  // namespace

WaitSetSubscriber::WaitSetSubscriber() = default;
WaitSetSubscriber::~WaitSetSubscriber() = default;

expected<WaitSetSubscriber::WaitOwnerGuard, Error> WaitSetSubscriber::claim_wait_owner() noexcept {
  if (waiting_.test_and_set(std::memory_order_acquire)) {
    return make_unexpected_nothrow<WaitOwnerGuard>(
        ErrorCode::kInvalidArgument,
        "channel subscriber already has a wait or readiness acquisition in flight");
  }
  WaitOwnerGuard claim{waiting_};
  if (!wait_context_ && wait_set_attached()) {
    return readiness_handle_attached()
               ? make_unexpected_nothrow<WaitOwnerGuard>(
                     ErrorCode::kInvalidArgument,
                     "channel subscriber is already owned by a readiness handle; detach it before "
                     "using another wait owner")
               : make_unexpected_nothrow<WaitOwnerGuard>(
                     ErrorCode::kInvalidArgument,
                     "channel subscriber is owned by the Runtime readiness pump or another wait "
                     "set and cannot be claimed directly");
  }
  return claim;
}

expected<WaitSetSubscriber::WaitOwnerGuard, Error>
WaitSetSubscriber::claim_readiness_owner() noexcept {
  auto claim = claim_wait_owner();
  if (!claim) {
    return claim;
  }
  if (wait_context_) {
    auto detached = detach_wait_set();
    if (!detached) {
      return make_unexpected(std::move(detached).error());
    }
    wait_context_.reset();
  }
  return claim;
}

expected<void, Error> WaitSetSubscriber::wait_for_readiness(
    const std::chrono::steady_clock::time_point deadline,
    const std::stop_token* const stop_token) noexcept {
  // The noexcept is a promise this function keeps, not one it delegates to its callees. Setting a
  // wait up allocates in several places -- the diagnostic strings below, the whole of
  // SubscriberWaitContext::create on the first call, rearm_trigger after a cancellation, and the
  // backend's ready-handle vector -- and none of those paths has any other way to report a failed
  // allocation. Translating it here keeps the declared contract true; letting it cross the
  // noexcept boundary would terminate the process on a condition the caller can act on.
  //
  // The returned Error is code-only, so producing it cannot itself allocate and the failure path
  // cannot fail. Only std::bad_alloc is caught: anything else is a defect, and terminating on it
  // is the right outcome rather than reporting it as resource exhaustion.
  try {
    auto claim = claim_wait_owner();
    if (!claim) {
      return make_unexpected(std::move(claim).error());
    }
    return wait_for_readiness_checked(deadline, stop_token);
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
}

expected<void, Error> WaitSetSubscriber::wait_for_readiness_claimed(
    const std::chrono::steady_clock::time_point deadline,
    const std::stop_token* const stop_token) noexcept {
  try {
    return wait_for_readiness_checked(deadline, stop_token);
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
}

expected<void, Error> WaitSetSubscriber::wait_for_readiness_checked(
    const std::chrono::steady_clock::time_point deadline, const std::stop_token* const stop_token) {
  // Answer a stop that was already requested before doing any work, so a shutdown race cannot
  // build a wait context that is immediately thrown away.
  if (stop_token != nullptr && stop_token->stop_requested()) {
    return make_unexpected(Error{ErrorCode::kInterrupted});
  }

  // CHAN-066: a kernel wait on an endpoint with no event lane can only end at its deadline, which
  // would be a silent non-waking fallback. Refuse before any wait context exists.
  if (!event_wake_enabled()) {
    return make_unexpected(make_channel_error(
        ErrorCode::kNotSupported,
        "channel subscriber was created without event wake and cannot wait for readiness"));
  }

  if (!wait_context_) {
    auto* const runtime = wait_context_runtime();
    if (runtime == nullptr) {
      return make_unexpected(make_channel_error(
          ErrorCode::kNotSupported, "channel subscriber binding has no waitable transport"));
    }
    auto created = SubscriberWaitContext::create(*static_cast<holomq::Context*>(runtime), *this);
    if (!created) {
      return make_unexpected(std::move(created).error());
    }
    wait_context_ = std::move(created).value();
  }

  // A trigger latch is monotonic and this context outlives any single wait, so a cancellation
  // delivered to an earlier call would otherwise end every later wait immediately. Re-arm before
  // the trigger is observed below, so this wait blocks on the trigger it will actually be woken by.
  if (auto rearmed = wait_context_->rearm_trigger(); !rearmed) {
    return make_unexpected(std::move(rearmed).error());
  }

  // Arm cancellation for the whole wait. Constructing this on an already-stopped token runs the
  // callback inline on this thread, so the trigger is latched before the first block; a stop that
  // lands while blocked wakes the kernel wait; a stop that lands between waits is caught by the
  // stop_requested check below. None of the three has a window.
  //
  // The arming is scoped to one call, so a wait_and_receive_next retry that goes round more than
  // once builds and tears down one std::stop_callback per iteration, each taking the stop state's
  // lock. That is deliberate, not an oversight: it must not be hoisted above the retry loop while
  // TriggerFirer holds a raw pointer, because rearm_trigger above replaces both the attachment and
  // the context's shared_ptr and so drops the last owner of the previous WaitTrigger. A callback
  // that outlived a re-arm would fire on freed storage. Hoisting therefore needs the trigger to be
  // owned by the armed object, plus new surface on this exported class to reach it, which is a
  // larger and more ABI-sensitive change than the per-iteration cost justifies.
  holomq::WaitTrigger* const trigger = &wait_context_->trigger();
  std::optional<std::stop_callback<TriggerFirer>> arm;
  if (stop_token != nullptr) {
    arm.emplace(*stop_token, TriggerFirer{trigger});
  }

  while (true) {
    if (stop_token != nullptr && stop_token->stop_requested()) {
      return make_unexpected(Error{ErrorCode::kInterrupted});
    }

    // Recheck before every block. One transport notification can cover several queued samples, so
    // a sample may already be waiting with no notification left to wake this thread.
    auto ready = has_samples();
    if (!ready) {
      return make_unexpected(std::move(ready).error());
    }
    if (*ready) {
      return {};
    }

    const auto now = std::chrono::steady_clock::now();
    if (deadline <= now) {
      return make_unexpected(Error{ErrorCode::kTimeout});
    }
    // Clamped so an expired or far-future deadline cannot become a negative or effectively
    // infinite kernel timeout, and so a dropped transport notification costs latency, not a hang.
    const auto remaining = std::min(
        std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now), kMaxIterationWait);

    auto waited = wait_context_->wait_for(remaining);
    if (!waited) {
      return make_unexpected(std::move(waited).error());
    }
    if (trigger->is_set()) {
      return make_unexpected(Error{ErrorCode::kInterrupted});
    }
  }
}
InlinePayloadPublisher::~InlinePayloadPublisher() = default;
InlinePayloadSubscriber::~InlinePayloadSubscriber() = default;
InlinePayloadWaitSet::~InlinePayloadWaitSet() = default;

InlinePayloadReceiveGuard::InlinePayloadReceiveGuard(InlinePayloadSubscriber& subscriber,
                                                     const std::span<const std::byte> payload,
                                                     const SampleMetadata metadata) noexcept
    : subscriber_(&subscriber), payload_(payload), metadata_(metadata) {}
InlinePayloadReceiveGuard::InlinePayloadReceiveGuard(InlinePayloadReceiveGuard&& other) noexcept
    : subscriber_(std::exchange(other.subscriber_, nullptr)),
      payload_(std::exchange(other.payload_, {})),
      metadata_(other.metadata_) {}
InlinePayloadReceiveGuard& InlinePayloadReceiveGuard::operator=(
    InlinePayloadReceiveGuard&& other) noexcept {
  if (this != &other) {
    if (subscriber_ != nullptr) {
      subscriber_->release_payload_loan();
    }
    subscriber_ = std::exchange(other.subscriber_, nullptr);
    payload_ = std::exchange(other.payload_, {});
    metadata_ = other.metadata_;
  }
  return *this;
}
InlinePayloadReceiveGuard::~InlinePayloadReceiveGuard() noexcept {
  if (subscriber_ != nullptr) {
    subscriber_->release_payload_loan();
  }
}

expected<std::uint32_t, Error> InlinePayloadPublisher::emit(
    const std::span<const std::byte> payload, const SampleMetadata& metadata) {
  InlinePublicationResult result = emit_phase(payload, metadata);
  switch (result.phase) {
    case InlinePublicationPhase::kAccepted:
      if (result.recipient_count == 0U) {
        return make_unexpected(Error{ErrorCode::kPublicationIndeterminate});
      }
      return result.recipient_count;
    case InlinePublicationPhase::kPostSubmitUncertain:
      if (publication_error_consumes_authority(result.error.code) ||
          result.error.code == ErrorCode::kProtocolError ||
          result.error.code == ErrorCode::kInvalidArgument) {
        return make_unexpected(std::move(result.error));
      }
      return make_unexpected(Error{ErrorCode::kPublicationIndeterminate});
    case InlinePublicationPhase::kPreSubmitRejected:
    default:
      return make_unexpected(std::move(result.error));
  }
}

InlinePublicationResult InlinePayloadPublisher::emit_encoded_phase(const InlinePayloadEncoder&,
                                                                   const SampleMetadata&) {
  return InlinePublicationResult{
      .phase = InlinePublicationPhase::kPreSubmitRejected,
      .error = Error{ErrorCode::kNotSupported},
  };
}

expected<InlinePayloadReceiveGuard, Error> InlinePayloadSubscriber::receive_borrowed() {
  return make_unexpected(Error{ErrorCode::kNotSupported});
}

expected<InlinePayloadReceiveGuard, Error>
InlinePayloadSubscriber::receive_borrowed_if_data_accepted(const bool accept_data) {
  if (!accept_data) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  return receive_borrowed();
}

void InlinePayloadSubscriber::release_payload_loan() noexcept {}

expected<TimePoint, Error> InlinePayloadPublisher::make_capture_time(std::int64_t) const noexcept {
  return make_unexpected(Error{ErrorCode::kInvalidArgument});
}

expected<void, Error> InlinePayloadPublisher::report_clock_health(ClockHealthReport) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument});
}

expected<void, Error> InlinePayloadPublisher::close_stream() {
  return make_unexpected(Error{ErrorCode::kNotSupported});
}
TensorSchemaPayloadPublisher::~TensorSchemaPayloadPublisher() = default;
TensorSchemaPayloadSubscriber::~TensorSchemaPayloadSubscriber() = default;

InlinePublicationResult TensorSchemaPayloadPublisher::close_stream(SampleMetadata) {
  return InlinePublicationResult{
      .phase = InlinePublicationPhase::kPreSubmitRejected,
      .error = Error{ErrorCode::kNotSupported},
  };
}

std::uint64_t TensorSchemaPayloadPublisher::producer_generation() const noexcept {
  return 0U;
}

expected<std::uint64_t, Error> TensorSchemaPayloadPublisher::next_operation_sequence()
    const noexcept {
  return make_unexpected(Error{ErrorCode::kNotSupported});
}

expected<void, Error> TensorSchemaPayloadPublisher::report_clock_health(ClockHealthReport) {
  return make_unexpected(Error{ErrorCode::kNotSupported});
}

// The installed virtual declaration owns the default argument; this out-of-line fallback must
// retain the same signature.
// NOLINTNEXTLINE(google-default-arguments)
expected<std::uint32_t, Error> TensorSchemaPayloadPublisher::publish_encoded_payload(
    const TensorSchemaPayloadEncoder&, TensorSchemaPayloadReservation reservation, SampleMetadata) {
  release_tensor(std::move(reservation));
  return make_unexpected(Error{ErrorCode::kNotSupported});
}
expected<void, Error> TensorSchemaPayloadPublisher::validate_sample_metadata(
    const SampleMetadata& metadata) const noexcept {
  if (metadata.version.major != 1U || metadata.version.minor != 0U || metadata.sequence != 0U ||
      metadata.capture_timestamp_ns != 0 || metadata.source_clock_id.valid() ||
      metadata.publish_timestamp_ns != 0 || metadata.publish_clock_id.valid() ||
      metadata.trace_id != 0U || metadata.frame_id != 0U || metadata.byte_size != 0U ||
      metadata.payload_kind != PayloadKind::kInlineTyped || metadata.flags != SampleFlags::kNone) {
    return make_unexpected(make_channel_error(
        ErrorCode::kInvalidArgument,
        "Tensor schema sample metadata requires a catalog-bound Runtime envelope"));
  }
  return {};
}
expected<void, Error> TensorSchemaPayloadPublisher::validate_field_descriptor(
    std::uint32_t, const TensorViewMetadata&, const PayloadDescriptor&) const noexcept {
  return {};
}
expected<TimePoint, Error> TensorSchemaPayloadPublisher::make_capture_time(
    std::int64_t) const noexcept {
  return make_unexpected(Error{ErrorCode::kInvalidArgument});
}

expected<void, Error> TensorSchemaPayloadSubscriber::validate_field_descriptor(
    std::uint32_t, const TensorViewMetadata&, const PayloadDescriptor&) const noexcept {
  return {};
}
expected<TensorSchemaPayloadReceiveGuard, Error> TensorSchemaPayloadSubscriber::borrow_payload() {
  return make_unexpected(Error{ErrorCode::kNotSupported});
}

expected<TensorSchemaPayloadReceiveGuard, Error>
TensorSchemaPayloadSubscriber::borrow_payload_if_data_accepted(const bool accept_data) {
  if (!accept_data) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  return borrow_payload();
}

std::optional<std::uint64_t> TensorSchemaPayloadSubscriber::end_of_stream_sequence()
    const noexcept {
  return std::nullopt;
}
TensorSchemaPayloadReceiveGuard::TensorSchemaPayloadReceiveGuard(
    TensorSchemaPayloadSubscriber& subscriber, const std::span<const std::byte> payload) noexcept
    : subscriber_(&subscriber), payload_(payload) {}
TensorSchemaPayloadReceiveGuard::TensorSchemaPayloadReceiveGuard(
    TensorSchemaPayloadReceiveGuard&& other) noexcept
    : subscriber_(std::exchange(other.subscriber_, nullptr)),
      payload_(std::exchange(other.payload_, {})) {}
TensorSchemaPayloadReceiveGuard& TensorSchemaPayloadReceiveGuard::operator=(
    TensorSchemaPayloadReceiveGuard&& other) noexcept {
  if (this != &other) {
    if (subscriber_ != nullptr) {
      subscriber_->release_payload_loan();
    }
    subscriber_ = std::exchange(other.subscriber_, nullptr);
    payload_ = std::exchange(other.payload_, {});
  }
  return *this;
}
TensorSchemaPayloadReceiveGuard::~TensorSchemaPayloadReceiveGuard() noexcept {
  if (subscriber_ != nullptr) {
    subscriber_->release_payload_loan();
  }
}
}  // namespace channel_internal

using channel_internal::from_holomq_error;
using channel_internal::make_channel_error;
using channel_internal::replaces_oldest_unread_sample;
using channel_internal::validate_capabilities;
using channel_internal::validate_tensor_channel_options;

ChannelRuntime::ChannelRuntime(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
ChannelRuntime::ChannelRuntime(std::shared_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
ChannelRuntime::ChannelRuntime(ChannelRuntime&& other) noexcept = default;
ChannelRuntime& ChannelRuntime::operator=(ChannelRuntime&& other) noexcept = default;
ChannelRuntime::~ChannelRuntime() = default;

expected<ChannelRuntime, Error> ChannelRuntime::create() {
  // The HSDK5-012 base owns only the common context and deliberately leaves optional FD/VMM export
  // disabled. A later interprocess component may add private just-in-time realization; this path
  // must not imply that a public option or today's scalar/local endpoint factories enable it.
  auto context = holomq::ContextBuilder().with_vmm_fd_export(false).create();
  if (!context) {
    return make_unexpected(
        from_holomq_error(std::move(context).error(), "failed to create HoloMQ channel context"));
  }
  return ChannelRuntime(std::make_unique<Impl>(std::move(context).value()));
}

expected<std::unique_ptr<channel_internal::InlinePayloadPublisher>, Error>
ChannelRuntime::create_inline_payload_publisher(const std::string_view channel_name,
                                                const std::size_t payload_size,
                                                const ChannelOptions& options) {
  return create_inline_payload_publisher(
      channel_name, payload_size, options, channel_internal::PublisherPressureOptions{});
}

expected<std::unique_ptr<channel_internal::InlinePayloadPublisher>, Error>
ChannelRuntime::create_inline_payload_publisher(
    const std::string_view channel_name, const std::size_t payload_size,
    const ChannelOptions& options, const channel_internal::PublisherPressureOptions& pressure) {
  if (!impl_) {
    return make_unexpected(make_channel_error(ErrorCode::kClosed, "channel runtime is closed"));
  }
  return channel_internal::create_inline_payload_publisher(impl_->context_,
                                                           channel_name,
                                                           payload_size,
                                                           options,
                                                           impl_->context_.pubsub_capabilities(),
                                                           pressure);
}

expected<std::unique_ptr<channel_internal::InlinePayloadSubscriber>, Error>
ChannelRuntime::create_inline_payload_subscriber(const std::string_view channel_name,
                                                 const std::size_t payload_size,
                                                 const ChannelOptions& options) {
  if (!impl_) {
    return make_unexpected(make_channel_error(ErrorCode::kClosed, "channel runtime is closed"));
  }
  return channel_internal::create_inline_payload_subscriber(
      impl_->context_, channel_name, payload_size, options, impl_->context_.pubsub_capabilities());
}

expected<std::unique_ptr<channel_internal::InlinePayloadWaitSet>, Error>
ChannelRuntime::create_inline_payload_wait_set() {
  if (!impl_) {
    return make_unexpected(make_channel_error(ErrorCode::kClosed, "channel runtime is closed"));
  }
  return channel_internal::create_inline_payload_wait_set(impl_->context_);
}

expected<std::unique_ptr<channel_internal::TensorSchemaPayloadPublisher>, Error>
ChannelRuntime::create_tensor_schema_payload_publisher(const std::string_view channel_name,
                                                       const std::size_t payload_size,
                                                       const std::size_t descriptor_offset,
                                                       const ChannelOptions& options) {
  return create_tensor_schema_payload_publisher(channel_name,
                                                payload_size,
                                                descriptor_offset,
                                                options,
                                                channel_internal::PublisherPressureOptions{});
}

expected<std::unique_ptr<channel_internal::TensorSchemaPayloadPublisher>, Error>
ChannelRuntime::create_tensor_schema_payload_publisher(
    const std::string_view channel_name, const std::size_t payload_size,
    const std::span<const std::size_t> descriptor_offsets, const ChannelOptions& options) {
  return create_tensor_schema_payload_publisher(channel_name,
                                                payload_size,
                                                descriptor_offsets,
                                                options,
                                                channel_internal::PublisherPressureOptions{},
                                                {},
                                                nullptr);
}

expected<std::unique_ptr<channel_internal::TensorSchemaPayloadPublisher>, Error>
ChannelRuntime::create_tensor_schema_payload_publisher(
    const std::string_view channel_name, const std::size_t payload_size,
    const std::size_t descriptor_offset, const ChannelOptions& options,
    const channel_internal::PublisherPressureOptions& pressure) {
  const std::array descriptor_offsets{descriptor_offset};
  return create_tensor_schema_payload_publisher(
      channel_name, payload_size, descriptor_offsets, options, pressure, {}, nullptr);
}

expected<std::unique_ptr<channel_internal::TensorSchemaPayloadPublisher>, Error>
ChannelRuntime::create_tensor_schema_payload_publisher(
    const std::string_view channel_name, const std::size_t payload_size,
    const std::span<const std::size_t> descriptor_offsets, const ChannelOptions& options,
    const channel_internal::PublisherPressureOptions& pressure,
    const std::span<const channel_internal::TensorSchemaStagingOptions> staging,
    cudaStream_t staging_stream, const ClockId capture_clock,
    std::unique_ptr<channel_internal::InlinePayloadPublisher> health_publisher,
    const bool catalog_control) {
  if (!impl_) {
    return make_unexpected(make_channel_error(ErrorCode::kClosed, "channel runtime is closed"));
  }
  ChannelOptions effective_options = options;
  if (replaces_oldest_unread_sample(effective_options.queue_policy)) {
    effective_options.replacement_attribution_required = true;
  }
  if (auto valid = validate_tensor_channel_options(effective_options); !valid) {
    return make_unexpected(std::move(valid).error());
  }
  const auto capabilities = impl_->context_.pubsub_capabilities();
  if (auto valid = validate_capabilities(capabilities, effective_options); !valid) {
    return make_unexpected(std::move(valid).error());
  }
  // Reject invalid counts before they can seed the service's shared, fixed-capacity store.
  if (descriptor_offsets.empty() ||
      descriptor_offsets.size() > channel_internal::kMaximumDescriptorTensorFields) {
    return make_unexpected(
        make_channel_error(ErrorCode::kInvalidArgument,
                           "Tensor schema publisher field count is outside the endpoint range"));
  }
  // Publisher and subscribers of one service share this store; size it for every Tensor field.
  const auto tensor_fields = static_cast<std::uint32_t>(descriptor_offsets.size());
  auto sidecar_store = impl_->tensor_sidecar_store(channel_name, effective_options, tensor_fields);
  if (!sidecar_store) {
    return make_unexpected(std::move(sidecar_store).error());
  }
  return channel_internal::create_tensor_schema_payload_publisher(impl_->context_,
                                                                  channel_name,
                                                                  payload_size,
                                                                  descriptor_offsets,
                                                                  effective_options,
                                                                  std::move(sidecar_store).value(),
                                                                  staging,
                                                                  staging_stream,
                                                                  pressure,
                                                                  capture_clock,
                                                                  std::move(health_publisher),
                                                                  catalog_control);
}

expected<std::unique_ptr<channel_internal::TensorSchemaPayloadPublisher>, Error>
ChannelRuntime::create_tensor_schema_payload_publisher(
    const std::string_view channel_name, const std::size_t payload_size,
    const std::size_t descriptor_offset, const ChannelOptions& options,
    const channel_internal::PublisherPressureOptions& pressure,
    const channel_internal::TensorSchemaStagingOptions& staging) {
  const std::array descriptor_offsets{descriptor_offset};
  const std::array staging_fields{staging};
  return create_tensor_schema_payload_publisher(
      channel_name, payload_size, descriptor_offsets, options, pressure, staging_fields, nullptr);
}

expected<std::unique_ptr<channel_internal::TensorSchemaPayloadSubscriber>, Error>
ChannelRuntime::create_tensor_schema_payload_subscriber(
    const std::string_view channel_name, const std::size_t payload_size,
    const std::uint32_t tensor_field_count, const ChannelOptions& options,
    std::unique_ptr<channel_internal::InlinePayloadSubscriber> health_subscriber,
    const bool catalog_control,
    const std::span<const channel_internal::TensorSchemaStagingOptions> field_contracts,
    const std::uint64_t catalog_clock_count, const std::uint32_t receive_capacity) {
  if (!impl_) {
    return make_unexpected(make_channel_error(ErrorCode::kClosed, "channel runtime is closed"));
  }
  ChannelOptions effective_options = options;
  if (replaces_oldest_unread_sample(effective_options.queue_policy)) {
    effective_options.replacement_attribution_required = true;
  }
  if (auto valid = validate_tensor_channel_options(effective_options); !valid) {
    return make_unexpected(std::move(valid).error());
  }
  const auto capabilities = impl_->context_.pubsub_capabilities();
  if (auto valid = validate_capabilities(capabilities, effective_options); !valid) {
    return make_unexpected(std::move(valid).error());
  }
  // Reject invalid counts before they can seed the service's shared, fixed-capacity store.
  if (tensor_field_count == 0U ||
      tensor_field_count > channel_internal::kMaximumDescriptorTensorFields) {
    return make_unexpected(
        make_channel_error(ErrorCode::kInvalidArgument,
                           "Tensor schema subscriber field count is outside the endpoint range"));
  }
  // Publisher and subscribers of one service share this store; size it for every Tensor field.
  auto sidecar_store =
      impl_->tensor_sidecar_store(channel_name, effective_options, tensor_field_count);
  if (!sidecar_store) {
    return make_unexpected(std::move(sidecar_store).error());
  }
  return channel_internal::create_tensor_schema_payload_subscriber(impl_->context_,
                                                                   channel_name,
                                                                   payload_size,
                                                                   effective_options,
                                                                   std::move(sidecar_store).value(),
                                                                   impl_,
                                                                   tensor_field_count,
                                                                   std::move(health_subscriber),
                                                                   catalog_control,
                                                                   field_contracts,
                                                                   catalog_clock_count,
                                                                   receive_capacity);
}

}  // namespace holoscan
