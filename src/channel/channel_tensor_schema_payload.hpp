/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "channel_publisher_pressure.hpp"

#include <holoscan/channel/channel.hpp>

#include <holomq/context.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>

namespace holoscan {

class TensorSidecarStore;

namespace channel_internal {

[[nodiscard]] expected<std::unique_ptr<TensorSchemaPayloadPublisher>, Error>
create_tensor_schema_payload_publisher(
    holomq::Context& context, std::string_view channel_name, std::size_t payload_size,
    std::span<const std::size_t> descriptor_offsets, const ChannelOptions& options,
    std::shared_ptr<TensorSidecarStore> sidecar_store,
    std::span<const TensorSchemaStagingOptions> staging, cudaStream_t staging_stream = nullptr,
    PublisherPressureOptions pressure = {}, ClockId capture_clock = {},
    std::unique_ptr<InlinePayloadPublisher> health_publisher = nullptr,
    bool catalog_control = false);

[[nodiscard]] expected<std::unique_ptr<TensorSchemaPayloadSubscriber>, Error>
create_tensor_schema_payload_subscriber(
    holomq::Context& context, std::string_view channel_name, std::size_t payload_size,
    const ChannelOptions& options, std::shared_ptr<TensorSidecarStore> sidecar_store,
    std::shared_ptr<void> runtime_lifetime, std::uint32_t tensor_field_count,
    std::unique_ptr<InlinePayloadSubscriber> health_subscriber = nullptr,
    bool catalog_control = false, std::span<const TensorSchemaStagingOptions> field_contracts = {},
    std::uint64_t catalog_clock_count = 0U);

}  // namespace channel_internal
}  // namespace holoscan
