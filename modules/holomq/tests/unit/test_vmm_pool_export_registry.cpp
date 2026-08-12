/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vmm_pool_export_registry.hpp"

#include <gtest/gtest.h>

#include <fcntl.h>
#include <unistd.h>

TEST(VmmPoolExportRegistryTest, DupAfterRegisterReturnsDistinctFd) {
  int fds[2] = {-1, -1};
  ASSERT_EQ(::pipe(fds), 0);

  holomq::VmmPoolExportRegistry registry;
  constexpr std::uint64_t kCookie = 42U;
  registry.register_vmm_fd_export(kCookie, [read_fd = fds[0]]() {
    const int dup_fd = ::dup(read_fd);
    return holomq::ScopedFileDescriptor(dup_fd >= 0 ? dup_fd : -1);
  });

  holomq::ScopedFileDescriptor exported = registry.dup_exported_fd(kCookie);
  ASSERT_TRUE(exported);
  EXPECT_NE(exported.get(), fds[0]);
  EXPECT_EQ(::fcntl(exported.get(), F_GETFD), 0);

  ::close(fds[1]);
}

TEST(VmmPoolExportRegistryTest, MissReturnsEmpty) {
  holomq::VmmPoolExportRegistry registry;
  EXPECT_FALSE(registry.dup_exported_fd(99U));
}

TEST(VmmPoolExportRegistryTest, UnregisterRemovesEntry) {
  int fds[2] = {-1, -1};
  ASSERT_EQ(::pipe(fds), 0);

  holomq::VmmPoolExportRegistry registry;
  constexpr std::uint64_t kCookie = 7U;
  registry.register_vmm_fd_export(kCookie, [read_fd = fds[0]]() {
    const int dup_fd = ::dup(read_fd);
    return holomq::ScopedFileDescriptor(dup_fd >= 0 ? dup_fd : -1);
  });
  registry.unregister_vmm_fd_export(kCookie);

  EXPECT_FALSE(registry.dup_exported_fd(kCookie));

  ::close(fds[1]);
}

TEST(VmmPoolExportRegistryTest, EmptyCallbackIsIgnored) {
  holomq::VmmPoolExportRegistry registry;
  registry.register_vmm_fd_export(1U, {});
  EXPECT_FALSE(registry.dup_exported_fd(1U));
}

TEST(VmmPoolExportRegistryTest, RegisterReplacesPriorCallback) {
  int first_pipe[2] = {-1, -1};
  int second_pipe[2] = {-1, -1};
  ASSERT_EQ(::pipe(first_pipe), 0);
  ASSERT_EQ(::pipe(second_pipe), 0);

  holomq::VmmPoolExportRegistry registry;
  constexpr std::uint64_t kCookie = 3U;
  registry.register_vmm_fd_export(kCookie, [read_fd = first_pipe[0]]() {
    const int dup_fd = ::dup(read_fd);
    return holomq::ScopedFileDescriptor(dup_fd >= 0 ? dup_fd : -1);
  });
  registry.register_vmm_fd_export(kCookie, [read_fd = second_pipe[0]]() {
    const int dup_fd = ::dup(read_fd);
    return holomq::ScopedFileDescriptor(dup_fd >= 0 ? dup_fd : -1);
  });

  holomq::ScopedFileDescriptor exported = registry.dup_exported_fd(kCookie);
  ASSERT_TRUE(exported);

  char byte = 0;
  ASSERT_EQ(::write(second_pipe[1], "x", 1), 1);
  ASSERT_EQ(::read(exported.get(), &byte, 1), 1);
  EXPECT_EQ(byte, 'x');

  ::close(first_pipe[1]);
  ::close(second_pipe[1]);
}
