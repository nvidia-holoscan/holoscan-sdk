/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <holomq/detail/transport/iox2/connection_error.hpp>
#include <holomq/detail/transport/iox2/inbound_message.hpp>
#include <holomq/detail/transport/iox2/listener_error.hpp>
#include <holomq/detail/transport/iox2/receive_error.hpp>
#include <holomq/detail/transport/iox2/subscriber_port.hpp>
#include <holomq/errors.hpp>
#include <holomq/expected.hpp>

#include "iox2/connection_failure.hpp"
#include "iox2/listener_error.hpp"
#include "iox2/port_error.hpp"

#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

namespace {

using Payload = std::uint64_t;
using InboundMessage = holomq::detail::transport::InboundMessage<Payload>;
using SubscriberPort = holomq::detail::transport::SubscriberPort<Payload>;

static_assert(std::is_same_v<decltype(std::declval<SubscriberPort&>().receive()),
                             holomq::expected<std::optional<InboundMessage>, holomq::ErrorCode>>);
static_assert(std::is_same_v<decltype(std::declval<const SubscriberPort&>().has_samples()),
                             holomq::expected<bool, holomq::ErrorCode>>);

TEST(Iox2ReceiveErrorTest, MapsReceiveErrors) {
  using holomq::ErrorCode;
  using holomq::detail::transport::map_receive_error_code;

  EXPECT_EQ(map_receive_error_code(iox2::ReceiveError::ExceedsMaxBorrows),
            ErrorCode::kBackpressured);
  EXPECT_EQ(map_receive_error_code(iox2::ReceiveError::FailedToEstablishConnection),
            ErrorCode::kNotReady);
  EXPECT_EQ(map_receive_error_code(iox2::ReceiveError::UnableToMapSendersDataSegment),
            ErrorCode::kStaleEndpoint);
}

TEST(Iox2ReceiveErrorTest, MapsConnectionFailures) {
  using holomq::ErrorCode;
  using holomq::detail::transport::map_connection_error_code;

  EXPECT_EQ(map_connection_error_code(iox2::ConnectionFailure::FailedToEstablishConnection),
            ErrorCode::kNotReady);
  EXPECT_EQ(map_connection_error_code(iox2::ConnectionFailure::UnableToMapSendersDataSegment),
            ErrorCode::kStaleEndpoint);
}

TEST(Iox2ReceiveErrorTest, MapsListenerWaitErrors) {
  using holomq::ErrorCode;
  using holomq::detail::transport::map_listener_wait_error_code;

  EXPECT_EQ(map_listener_wait_error_code(iox2::ListenerWaitError::ContractViolation),
            ErrorCode::kProtocolError);
  EXPECT_EQ(map_listener_wait_error_code(iox2::ListenerWaitError::InterruptSignal),
            ErrorCode::kInterrupted);
  EXPECT_EQ(map_listener_wait_error_code(iox2::ListenerWaitError::InternalFailure),
            ErrorCode::kFailure);
}

}  // namespace
