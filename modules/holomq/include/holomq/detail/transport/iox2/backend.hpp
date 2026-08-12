/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/detail/transport/iox2/config.hpp>
#include <holomq/detail/transport/iox2/inbound_message.hpp>
#include <holomq/detail/transport/iox2/outbound_loan.hpp>
#include <holomq/detail/transport/iox2/publisher_port.hpp>
#include <holomq/detail/transport/iox2/runtime.hpp>
#include <holomq/detail/transport/iox2/subscriber_port.hpp>
#include <holomq/detail/transport/validate_backend.hpp>

namespace holomq::detail::transport::contract {

inline constexpr bool backend_satisfied =
    validate_backend<Payload, OutboundLoan<Payload>, InboundMessage<Payload>,
                     PublisherPort<Payload>, SubscriberPort<Payload>, Runtime>();

static_assert(requires {
  { create_runtime() } -> std::same_as<holomq::expected<Runtime, holomq::Error>>;
});

}  // namespace holomq::detail::transport::contract
