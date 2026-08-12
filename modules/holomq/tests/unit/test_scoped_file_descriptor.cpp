/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "scoped_file_descriptor.hpp"

#include <gtest/gtest.h>

#include <utility>

#include <fcntl.h>
#include <unistd.h>

namespace {

[[nodiscard]] bool fd_is_open(const int fd) noexcept {
  return fd >= 0 && ::fcntl(fd, F_GETFD) != -1;
}

}  // namespace

TEST(ScopedFileDescriptorTest, DefaultIsEmpty) {
  const holomq::ScopedFileDescriptor guard;
  EXPECT_FALSE(guard);
  EXPECT_EQ(guard.get(), -1);
}

TEST(ScopedFileDescriptorTest, DestructorClosesOwnedFd) {
  int fds[2] = {-1, -1};
  ASSERT_EQ(::pipe(fds), 0);
  EXPECT_TRUE(fd_is_open(fds[0]));
  {
    const holomq::ScopedFileDescriptor guard(fds[0]);
    EXPECT_EQ(guard.get(), fds[0]);
  }
  EXPECT_FALSE(fd_is_open(fds[0]));
  ::close(fds[1]);
}

TEST(ScopedFileDescriptorTest, MoveTransfersOwnership) {
  int fds[2] = {-1, -1};
  ASSERT_EQ(::pipe(fds), 0);
  holomq::ScopedFileDescriptor source(fds[0]);
  holomq::ScopedFileDescriptor destination = std::move(source);
  EXPECT_FALSE(source);
  EXPECT_TRUE(destination);
  EXPECT_TRUE(fd_is_open(destination.get()));
  ::close(fds[1]);
}

TEST(ScopedFileDescriptorTest, MoveAssignClosesPreviousFd) {
  int first_pipe[2] = {-1, -1};
  int second_pipe[2] = {-1, -1};
  ASSERT_EQ(::pipe(first_pipe), 0);
  ASSERT_EQ(::pipe(second_pipe), 0);

  holomq::ScopedFileDescriptor first_guard(first_pipe[0]);
  holomq::ScopedFileDescriptor second_guard(second_pipe[0]);
  const int replaced_fd = second_pipe[0];

  second_guard = std::move(first_guard);
  EXPECT_FALSE(fd_is_open(replaced_fd));
  EXPECT_EQ(second_guard.get(), first_pipe[0]);

  ::close(first_pipe[1]);
  ::close(second_pipe[1]);
}

TEST(ScopedFileDescriptorTest, ReleaseTransfersWithoutClosing) {
  int fds[2] = {-1, -1};
  ASSERT_EQ(::pipe(fds), 0);

  holomq::ScopedFileDescriptor guard(fds[0]);
  const int released = guard.release();
  EXPECT_FALSE(guard);
  EXPECT_EQ(released, fds[0]);
  EXPECT_TRUE(fd_is_open(released));

  ::close(released);
  ::close(fds[1]);
}

TEST(ScopedFileDescriptorTest, ResetClosesAndClears) {
  int fds[2] = {-1, -1};
  ASSERT_EQ(::pipe(fds), 0);

  holomq::ScopedFileDescriptor guard(fds[0]);
  guard.reset();
  EXPECT_FALSE(guard);
  EXPECT_FALSE(fd_is_open(fds[0]));

  ::close(fds[1]);
}

TEST(ScopedFileDescriptorTest, SwapExchangesOwnership) {
  int first_pipe[2] = {-1, -1};
  int second_pipe[2] = {-1, -1};
  ASSERT_EQ(::pipe(first_pipe), 0);
  ASSERT_EQ(::pipe(second_pipe), 0);

  holomq::ScopedFileDescriptor first_guard(first_pipe[0]);
  holomq::ScopedFileDescriptor second_guard(second_pipe[0]);
  const int first_fd = first_guard.get();
  const int second_fd = second_guard.get();

  first_guard.swap(second_guard);
  EXPECT_EQ(first_guard.get(), second_fd);
  EXPECT_EQ(second_guard.get(), first_fd);

  ::close(first_pipe[1]);
  ::close(second_pipe[1]);
}

TEST(ScopedFileDescriptorTest, NegativeFdIsNoOp) {
  const holomq::ScopedFileDescriptor guard(-1);
  EXPECT_FALSE(guard);
  EXPECT_EQ(guard.get(), -1);
}
