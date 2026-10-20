/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <holomq/inbound_message.hpp>
#include <holomq/outbound_message.hpp>
#include <holomq/publisher.hpp>
#include <holomq/subscriber.hpp>
#include <holoscan/channel/descriptor_endpoints.hpp>
#include <holoscan/channel/runtime.hpp>

#include "channel_common.hpp"
#include "channel_runtime_impl.hpp"

namespace holoscan {

using channel_internal::descriptor_core_payload_kind;
using channel_internal::from_holomq_error;
using channel_internal::from_holomq_publication_result;
using channel_internal::from_holomq_result;
using channel_internal::make_channel_error;
using channel_internal::to_holomq_publisher_options;
using channel_internal::to_holomq_service_options;
using channel_internal::to_holomq_subscriber_options;
using channel_internal::validate_capabilities;
using channel_internal::validate_channel_options;

class ChannelDescriptorPublisher::Impl {
 public:
  Impl(holomq::Publisher<PayloadDescriptor>&& publisher,
       const std::uint32_t expected_recipient_count) noexcept
      : publisher_(std::move(publisher)), expected_recipient_count_(expected_recipient_count) {}

  holomq::Publisher<PayloadDescriptor> publisher_;
  std::uint32_t expected_recipient_count_{};
};

class ChannelDescriptorSubscriber::Impl {
 public:
  explicit Impl(holomq::Subscriber<PayloadDescriptor>&& subscriber) noexcept
      : subscriber_(std::move(subscriber)) {}

  holomq::Subscriber<PayloadDescriptor> subscriber_;
};

ChannelDescriptorPublisher::ChannelDescriptorPublisher(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

ChannelDescriptorPublisher::ChannelDescriptorPublisher(
    ChannelDescriptorPublisher&& other) noexcept = default;

ChannelDescriptorPublisher& ChannelDescriptorPublisher::operator=(
    ChannelDescriptorPublisher&& other) noexcept = default;

ChannelDescriptorPublisher::~ChannelDescriptorPublisher() = default;

expected<std::uint32_t, Error> ChannelDescriptorPublisher::publish(
    const ChannelSampleDescriptor& descriptor) {
  if (!descriptor.has_supported_abi()) {
    return make_unexpected(
        make_channel_error(ErrorCode::kInvalidArgument, "channel descriptor ABI is not supported"));
  }
  return publish(payload_descriptor_from_descriptor(descriptor));
}

expected<std::uint32_t, Error> ChannelDescriptorPublisher::publish(
    const PayloadDescriptor& payload_descriptor) {
  if (!impl_) {
    return make_unexpected(make_channel_error(ErrorCode::kClosed, "channel publisher is closed"));
  }
  auto payload_kind = descriptor_core_payload_kind(payload_descriptor.payload_kind);
  if (!payload_kind) {
    return make_unexpected(std::move(payload_kind).error());
  }
  auto message = impl_->publisher_.loan();
  if (!message) {
    return make_unexpected(from_holomq_error(std::move(message).error()));
  }

  message->payload() = payload_descriptor;
  return from_holomq_publication_result(impl_->publisher_.send(std::move(message).value()),
                                        impl_->expected_recipient_count_);
}

expected<std::uint32_t, Error> ChannelDescriptorPublisher::publish(const SampleEnvelope& envelope) {
  return publish(envelope.payload);
}

expected<void, Error> ChannelDescriptorPublisher::update_connections() noexcept {
  if (!impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return from_holomq_result(impl_->publisher_.update_connections());
}

bool ChannelDescriptorPublisher::has_subscribers() const noexcept {
  return impl_ && impl_->publisher_.has_subscribers();
}

ChannelDescriptorSubscriber::ChannelDescriptorSubscriber(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

ChannelDescriptorSubscriber::ChannelDescriptorSubscriber(
    ChannelDescriptorSubscriber&& other) noexcept = default;

ChannelDescriptorSubscriber& ChannelDescriptorSubscriber::operator=(
    ChannelDescriptorSubscriber&& other) noexcept = default;

ChannelDescriptorSubscriber::~ChannelDescriptorSubscriber() = default;

expected<bool, Error> ChannelDescriptorSubscriber::has_samples() const {
  if (!impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return from_holomq_result(impl_->subscriber_.has_samples());
}

expected<ChannelSampleDescriptor, Error> ChannelDescriptorSubscriber::receive() {
  auto payload = receive_payload_descriptor();
  if (!payload) {
    return make_unexpected(std::move(payload).error());
  }
  if (payload->payload_kind != ChannelPayloadKind::kLocalBufferDescriptor) {
    return make_unexpected(make_channel_error(
        ErrorCode::kNotImplemented,
        "ChannelSampleDescriptor receive requires a local buffer descriptor payload"));
  }
  return channel_descriptor_from_payload_descriptor(payload.value());
}

expected<PayloadDescriptor, Error> ChannelDescriptorSubscriber::receive_payload_descriptor() {
  if (!impl_) {
    return make_unexpected(make_channel_error(ErrorCode::kClosed, "channel subscriber is closed"));
  }
  auto inbound = impl_->subscriber_.try_take();
  if (!inbound) {
    return make_unexpected(from_holomq_error(std::move(inbound).error()));
  }
  PayloadDescriptor payload = inbound->payload();
  auto payload_kind = descriptor_core_payload_kind(payload.payload_kind);
  if (!payload_kind) {
    return make_unexpected(std::move(payload_kind).error());
  }
  return payload;
}

expected<SampleEnvelope, Error> ChannelDescriptorSubscriber::receive_envelope() {
  auto payload = receive_payload_descriptor();
  if (!payload) {
    return make_unexpected(std::move(payload).error());
  }
  auto payload_kind = descriptor_core_payload_kind(payload->payload_kind);
  if (!payload_kind) {
    return make_unexpected(std::move(payload_kind).error());
  }
  return SampleEnvelope{.metadata = SampleMetadata{.sequence = payload->sequence,
                                                   .byte_size = payload->byte_size,
                                                   .payload_kind = payload_kind.value()},
                        .payload = payload.value()};
}

expected<ChannelDescriptorPublisher, Error> ChannelRuntime::create_descriptor_publisher(
    const std::string_view channel_name, const ChannelOptions& options) {
  if (!impl_) {
    return make_unexpected(make_channel_error(ErrorCode::kClosed, "channel runtime is closed"));
  }
  if (auto valid = validate_channel_options(options); !valid) {
    return make_unexpected(std::move(valid).error());
  }
  const auto capabilities = impl_->context_.pubsub_capabilities();
  if (auto valid = validate_capabilities(capabilities, options); !valid) {
    return make_unexpected(std::move(valid).error());
  }

  auto publisher_options = to_holomq_publisher_options(options);
  publisher_options.enable_sample_release_callbacks = false;
  auto publisher = impl_->context_.publisher_builder(std::string(channel_name))
                       .service_options(to_holomq_service_options(options))
                       .publisher_options(publisher_options)
                       .payload_type<PayloadDescriptor>()
                       .create();
  if (!publisher) {
    return make_unexpected(
        from_holomq_error(std::move(publisher).error(), "failed to create channel publisher"));
  }
  return ChannelDescriptorPublisher(std::make_unique<ChannelDescriptorPublisher::Impl>(
      std::move(publisher).value(), options.expected_recipient_count));
}

expected<ChannelDescriptorSubscriber, Error> ChannelRuntime::create_descriptor_subscriber(
    const std::string_view channel_name, const ChannelOptions& options) {
  if (!impl_) {
    return make_unexpected(make_channel_error(ErrorCode::kClosed, "channel runtime is closed"));
  }
  if (auto valid = validate_channel_options(options); !valid) {
    return make_unexpected(std::move(valid).error());
  }
  const auto capabilities = impl_->context_.pubsub_capabilities();
  if (auto valid = validate_capabilities(capabilities, options); !valid) {
    return make_unexpected(std::move(valid).error());
  }

  auto subscriber = impl_->context_.subscriber_builder(std::string(channel_name))
                        .service_options(to_holomq_service_options(options))
                        .subscriber_options(to_holomq_subscriber_options(options))
                        .payload_type<PayloadDescriptor>()
                        .create();
  if (!subscriber) {
    return make_unexpected(
        from_holomq_error(std::move(subscriber).error(), "failed to create channel subscriber"));
  }
  return ChannelDescriptorSubscriber(
      std::make_unique<ChannelDescriptorSubscriber::Impl>(std::move(subscriber).value()));
}
}  // namespace holoscan
