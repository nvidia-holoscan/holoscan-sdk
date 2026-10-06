/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <holomq/detail/transport/iox2/outbound_loan.hpp>
#include <holomq/detail/transport/iox2/publisher_port.hpp>
#include <holomq/detail/transport/iox2/send_error.hpp>
#include <holomq/detail/transport/iox2/unable_to_deliver_evidence.hpp>
#include <holomq/errors.hpp>
#include <holomq/expected.hpp>

#include <iox2/port_error.hpp>

#include <cstdint>
#include <type_traits>
#include <utility>

namespace {

using Payload = std::uint64_t;
using PublisherPort = holomq::detail::transport::PublisherPort<Payload>;
using OutboundLoan = holomq::detail::transport::OutboundLoan<Payload>;

static_assert(
    std::is_same_v<decltype(std::declval<PublisherPort&>().send(std::declval<OutboundLoan&&>())),
                   holomq::expected<int, holomq::Error>>);
static_assert(!std::is_move_assignable_v<PublisherPort>);

TEST(Iox2SendErrorTest, HandlerFreePublisherDoesNotConsumeStaleRecipientEvidence) {
  holomq::DeliveryRecipientId stale{};
  stale.bytes.front() = 0xABU;
  holomq::detail::transport::record_unable_to_deliver_recipient(stale);

  EXPECT_FALSE(holomq::detail::transport::take_unable_to_deliver_recipient_if(false));
  const auto retained = holomq::detail::transport::take_unable_to_deliver_recipient_if(true);
  ASSERT_TRUE(retained);
  EXPECT_EQ(*retained, stale);
}

struct SendErrorCase {
  iox2::SendError input;
  holomq::ErrorCode expected;
};

class Iox2SendErrorTest : public testing::TestWithParam<SendErrorCase> {};

TEST_P(Iox2SendErrorTest, MapsToHoloMQError) {
  const auto [input, expected] = GetParam();
  const holomq::Error error = holomq::detail::transport::map_send_error(input);

  EXPECT_EQ(error.code, expected);
  EXPECT_TRUE(error.message.empty());
}

INSTANTIATE_TEST_SUITE_P(
    SendErrors, Iox2SendErrorTest,
    testing::Values(
        SendErrorCase{iox2::SendError::ConnectionBrokenSinceSenderNoLongerExists,
                      holomq::ErrorCode::kClosed},
        SendErrorCase{iox2::SendError::ConnectionCorrupted, holomq::ErrorCode::kStaleEndpoint},
        SendErrorCase{iox2::SendError::LoanErrorOutOfMemory, holomq::ErrorCode::kBackpressured},
        SendErrorCase{iox2::SendError::LoanErrorExceedsMaxLoans, holomq::ErrorCode::kBackpressured},
        SendErrorCase{iox2::SendError::LoanErrorExceedsMaxLoanSize,
                      holomq::ErrorCode::kInvalidArgument},
        SendErrorCase{iox2::SendError::LoanErrorInternalFailure, holomq::ErrorCode::kFailure},
        SendErrorCase{iox2::SendError::ConnectionError, holomq::ErrorCode::kNotReady},
        SendErrorCase{iox2::SendError::UnableToDeliver, holomq::ErrorCode::kBackpressured},
        SendErrorCase{iox2::SendError::InternalError, holomq::ErrorCode::kFailure}));

}  // namespace
