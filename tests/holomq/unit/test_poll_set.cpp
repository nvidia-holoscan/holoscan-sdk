/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fd_pass/poll_set.hpp"

#include <gtest/gtest.h>

#include <unistd.h>

TEST(PollSetTest, ZeroTimeoutReturnsImmediatelyWithoutEvents) {
  int fds[2] = {-1, -1};
  ASSERT_EQ(::pipe(fds), 0);

  holomq::fd_pass::PollSet poll_set({fds[0]});
  EXPECT_EQ(poll_set.wait(0), holomq::fd_pass::PollSet::PollStatus::completed);
  EXPECT_FALSE(poll_set.has_event(0, POLLIN));

  ::close(fds[0]);
  ::close(fds[1]);
}

TEST(PollSetTest, PositiveTimeoutReturnsCompletedWithoutEvents) {
  int fds[2] = {-1, -1};
  ASSERT_EQ(::pipe(fds), 0);

  holomq::fd_pass::PollSet poll_set({fds[0]});
  EXPECT_EQ(poll_set.wait(10), holomq::fd_pass::PollSet::PollStatus::completed);
  EXPECT_FALSE(poll_set.has_event(0, POLLIN));

  ::close(fds[0]);
  ::close(fds[1]);
}

TEST(PollSetTest, DetectsReadableFd) {
  int fds[2] = {-1, -1};
  ASSERT_EQ(::pipe(fds), 0);

  holomq::fd_pass::PollSet poll_set({fds[0]});
  ASSERT_EQ(::write(fds[1], "x", 1), 1);

  EXPECT_EQ(poll_set.wait(100), holomq::fd_pass::PollSet::PollStatus::completed);
  EXPECT_TRUE(poll_set.has_event(0, POLLIN));

  ::close(fds[0]);
  ::close(fds[1]);
}

TEST(PollSetTest, InvalidIndexHasNoEvent) {
  holomq::fd_pass::PollSet poll_set({});
  EXPECT_FALSE(poll_set.has_event(0, POLLIN));
  EXPECT_EQ(poll_set.fd(0), -1);
  EXPECT_EQ(poll_set.size(), 0U);
}
