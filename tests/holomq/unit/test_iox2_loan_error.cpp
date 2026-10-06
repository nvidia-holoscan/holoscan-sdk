/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include <holomq/detail/transport/iox2/loan_error.hpp>
#include <holomq/detail/transport/iox2/outbound_loan.hpp>
#include <holomq/detail/transport/iox2/publisher_port.hpp>
#include <holomq/errors.hpp>
#include <holomq/expected.hpp>

#include <iox2/port_error.hpp>

namespace {

using Payload = std::uint64_t;
using PublisherPort = holomq::detail::transport::PublisherPort<Payload>;
using OutboundLoan = holomq::detail::transport::OutboundLoan<Payload>;

static_assert(std::is_same_v<decltype(std::declval<PublisherPort&>().loan()),
                             holomq::expected<OutboundLoan, holomq::Error>>);

struct LoanErrorCase {
  iox2::LoanError input;
  holomq::ErrorCode expected;
};

class IOX2LoanErrorTest : public testing::TestWithParam<LoanErrorCase> {};

TEST_P(IOX2LoanErrorTest, MapsToHoloMQError) {
  const auto [input, expected] = GetParam();
  const holomq::Error error = holomq::detail::transport::map_loan_error(input);

  EXPECT_EQ(error.code, expected);
  EXPECT_TRUE(error.message.empty());
}

INSTANTIATE_TEST_SUITE_P(
    LoanErrors, IOX2LoanErrorTest,
    testing::Values(
        LoanErrorCase{iox2::LoanError::OutOfMemory, holomq::ErrorCode::kBackpressured},
        LoanErrorCase{iox2::LoanError::ExceedsMaxLoanedSamples, holomq::ErrorCode::kBackpressured},
        LoanErrorCase{iox2::LoanError::ExceedsMaxLoanSize, holomq::ErrorCode::kInvalidArgument},
        LoanErrorCase{iox2::LoanError::InternalFailure, holomq::ErrorCode::kFailure}));

}  // namespace
