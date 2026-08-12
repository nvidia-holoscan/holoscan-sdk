/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/channel/channel.hpp>

#include "channel_common.hpp"
#include "channel_inline_payload.hpp"
#include "channel_runtime_impl.hpp"
#include "channel_tensor_schema_payload.hpp"

#include <holomq/context_builder.hpp>
#include <holomq/pubsub_capabilities.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <string_view>
#include <utility>

namespace holoscan {
namespace channel_internal {
WaitSetSubscriber::~WaitSetSubscriber() = default;
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
expected<std::uint32_t, Error> TensorSchemaPayloadPublisher::publish_encoded_payload(
    const TensorSchemaPayloadEncoder&, TensorSchemaPayloadReservation reservation) {
  release_tensor(std::move(reservation));
  return make_unexpected(Error{ErrorCode::kNotSupported});
}
expected<TensorSchemaPayloadReceiveGuard, Error> TensorSchemaPayloadSubscriber::borrow_payload() {
  return make_unexpected(Error{ErrorCode::kNotSupported});
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
  // The EA1-012 base owns only the common context and deliberately leaves optional FD/VMM export
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
    const std::size_t descriptor_offset, const ChannelOptions& options,
    const channel_internal::PublisherPressureOptions& pressure) {
  const std::array descriptor_offsets{descriptor_offset};
  return create_tensor_schema_payload_publisher(
      channel_name, payload_size, descriptor_offsets, options, pressure, {});
}

expected<std::unique_ptr<channel_internal::TensorSchemaPayloadPublisher>, Error>
ChannelRuntime::create_tensor_schema_payload_publisher(
    const std::string_view channel_name, const std::size_t payload_size,
    const std::span<const std::size_t> descriptor_offsets, const ChannelOptions& options,
    const channel_internal::PublisherPressureOptions& pressure,
    const std::span<const channel_internal::TensorSchemaStagingOptions> staging) {
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
  auto sidecar_store = impl_->tensor_sidecar_store(channel_name, effective_options);
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
                                                                  pressure);
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
      channel_name, payload_size, descriptor_offsets, options, pressure, staging_fields);
}

expected<std::unique_ptr<channel_internal::TensorSchemaPayloadSubscriber>, Error>
ChannelRuntime::create_tensor_schema_payload_subscriber(const std::string_view channel_name,
                                                        const std::size_t payload_size,
                                                        const std::uint32_t tensor_field_count,
                                                        const ChannelOptions& options) {
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
  auto sidecar_store = impl_->tensor_sidecar_store(channel_name, effective_options);
  if (!sidecar_store) {
    return make_unexpected(std::move(sidecar_store).error());
  }
  return channel_internal::create_tensor_schema_payload_subscriber(impl_->context_,
                                                                   channel_name,
                                                                   payload_size,
                                                                   effective_options,
                                                                   std::move(sidecar_store).value(),
                                                                   impl_,
                                                                   tensor_field_count);
}

}  // namespace holoscan
