/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

#include <holoscan/channel/endpoint_interfaces.hpp>
#include <holoscan/channel/runtime.hpp>

namespace holoscan::channel_internal {

/// Provide source-tree tests access to Runtime-private inline Channel endpoints.
struct InlinePayloadTestAccess {
  template <typename T>
    requires(ChannelInlinePayload<T> || CatalogChannelPayload<T>)
  [[nodiscard]] static InlinePayloadPublisher* publisher_endpoint(
      ChannelPublisher<T>& publisher) noexcept {
    return publisher.publisher_.get();
  }

  template <typename T>
    requires(ChannelInlinePayload<T> || CatalogChannelPayload<T>)
  [[nodiscard]] static ChannelSubscriber<T> make_subscriber(
      std::unique_ptr<InlinePayloadSubscriber> subscriber) {
    return ChannelSubscriber<T>(std::move(subscriber));
  }

  template <typename T>
    requires(ChannelInlinePayload<T> || CatalogChannelPayload<T>)
  [[nodiscard]] static expected<std::uint64_t, Error> attach_wait_set(
      InlinePayloadWaitSet& wait_set, ChannelSubscriber<T>& subscriber) {
    if (!subscriber.subscriber_) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    return wait_set.attach(*subscriber.subscriber_);
  }

  template <typename T>
    requires(ChannelInlinePayload<T> || CatalogChannelPayload<T>)
  [[nodiscard]] static std::optional<std::uint64_t> secondary_wait_set_handle(
      ChannelSubscriber<T>& subscriber) noexcept {
    return subscriber.subscriber_ ? protected_terminal_wait_handle(*subscriber.subscriber_)
                                  : std::nullopt;
  }

  [[nodiscard]] static expected<std::unique_ptr<InlinePayloadPublisher>, Error> create_publisher(
      ChannelRuntime& runtime, const std::string_view service, const std::size_t payload_size,
      const ChannelOptions& options) {
    return runtime.create_inline_payload_publisher(service, payload_size, options);
  }

  [[nodiscard]] static expected<std::unique_ptr<InlinePayloadSubscriber>, Error> create_subscriber(
      ChannelRuntime& runtime, const std::string_view service, const std::size_t payload_size,
      const ChannelOptions& options) {
    return runtime.create_inline_payload_subscriber(service, payload_size, options);
  }

  [[nodiscard]] static expected<std::unique_ptr<InlinePayloadWaitSet>, Error> create_wait_set(
      ChannelRuntime& runtime) {
    return runtime.create_inline_payload_wait_set();
  }
};

}  // namespace holoscan::channel_internal
