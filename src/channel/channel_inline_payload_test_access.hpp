/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <memory>
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
