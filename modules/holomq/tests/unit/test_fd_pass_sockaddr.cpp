/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fd_pass/protocol.hpp"
#include "fd_pass/sockaddr.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>
#include <string>

#include <sys/socket.h>
#include <sys/un.h>

namespace {

[[nodiscard]] std::string endpoint_of_max_length() {
  return std::string(holomq::fd_pass::endpoint_max_length(), 'a');
}

[[nodiscard]] std::string endpoint_over_max_length() {
  return std::string(holomq::fd_pass::endpoint_max_length() + 1U, 'a');
}

}  // namespace

TEST(FdPassProtocolTest, WireLayoutIsFixed) {
  using holomq::fd_pass::Request;
  using holomq::fd_pass::Response;

  EXPECT_EQ(sizeof(Request), 16U);
  EXPECT_EQ(sizeof(Response), 4U);
  EXPECT_EQ(Request{}.magic, holomq::fd_pass::k_magic);
  EXPECT_EQ(Request{}.version, holomq::fd_pass::k_version);
}

TEST(FdPassSockaddrTest, EndpointMaxFitsHandleField) {
  EXPECT_LE(holomq::fd_pass::endpoint_max_length(), holomq::fd_pass::socket_path_field_max() - 1U);
}

TEST(FdPassSockaddrTest, RejectsEmptyOrInvalidArguments) {
  ::sockaddr_un addr{};
  ::socklen_t len = 0;

  EXPECT_FALSE(holomq::fd_pass::fill_sockaddr_un("", &addr, &len));
  EXPECT_FALSE(holomq::fd_pass::fill_sockaddr_un("x", nullptr, &len));
  EXPECT_FALSE(holomq::fd_pass::fill_sockaddr_un("x", &addr, nullptr));
}

TEST(FdPassSockaddrTest, RejectsOverlongEndpoint) {
  ::sockaddr_un addr{};
  ::socklen_t len = 0;

  EXPECT_FALSE(holomq::fd_pass::fill_sockaddr_un(endpoint_over_max_length(), &addr, &len));
}

TEST(FdPassSockaddrTest, AcceptsMaxLengthEndpoint) {
  const std::string endpoint = endpoint_of_max_length();
  ::sockaddr_un addr{};
  ::socklen_t len = 0;

  ASSERT_TRUE(holomq::fd_pass::fill_sockaddr_un(endpoint, &addr, &len));
  EXPECT_EQ(addr.sun_family, AF_UNIX);

#if HOLOMQ_FD_PASS_USE_ABSTRACT
  EXPECT_EQ(addr.sun_path[0], '\0');
  EXPECT_EQ(std::memcmp(addr.sun_path + 1, endpoint.data(), endpoint.size()), 0);
  EXPECT_EQ(len,
            static_cast<::socklen_t>(offsetof(::sockaddr_un, sun_path) + 1U + endpoint.size()));
#else
  EXPECT_STREQ(addr.sun_path, endpoint.c_str());
  EXPECT_EQ(len,
            static_cast<::socklen_t>(offsetof(::sockaddr_un, sun_path) + endpoint.size() + 1U));
#endif
}

TEST(FdPassSockaddrTest, AcceptsTypicalFilesystemPath) {
  ::sockaddr_un addr{};
  ::socklen_t len = 0;

  ASSERT_TRUE(holomq::fd_pass::fill_sockaddr_un("/tmp/holomq_test.sock", &addr, &len));
  EXPECT_EQ(addr.sun_family, AF_UNIX);

#if HOLOMQ_FD_PASS_USE_ABSTRACT
  EXPECT_EQ(addr.sun_path[0], '\0');
  EXPECT_STREQ(addr.sun_path + 1, "/tmp/holomq_test.sock");
#else
  EXPECT_STREQ(addr.sun_path, "/tmp/holomq_test.sock");
#endif
}
