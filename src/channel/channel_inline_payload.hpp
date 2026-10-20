/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <memory>
#include <string_view>

#include <holomq/context.hpp>
#include <holoscan/channel/endpoint_interfaces.hpp>

#include "channel_common.hpp"

namespace holoscan::channel_internal {

[[nodiscard]] expected<std::unique_ptr<InlinePayloadPublisher>, Error>
create_inline_payload_publisher(holomq::Context& context, std::string_view channel_name,
                                std::size_t payload_size, const ChannelOptions& options,
                                const TransportCapabilities& capabilities,
                                PublisherPressureOptions pressure = {});

[[nodiscard]] expected<std::unique_ptr<InlinePayloadSubscriber>, Error>
create_inline_payload_subscriber(holomq::Context& context, std::string_view channel_name,
                                 std::size_t payload_size, const ChannelOptions& options,
                                 const TransportCapabilities& capabilities);

[[nodiscard]] expected<std::unique_ptr<InlinePayloadWaitSet>, Error> create_inline_payload_wait_set(
    holomq::Context& context);

}  // namespace holoscan::channel_internal
