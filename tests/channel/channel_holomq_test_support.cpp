/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "channel_holomq_test_support.hpp"

#include <holomq/context_builder.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <cuda.h>
#include <unistd.h>

namespace holoscan::channel_test {
namespace {

std::string make_holomq_socket_path(const std::string& prefix) {
  static std::atomic<std::uint64_t> next_socket_id{0U};
  const std::uint64_t socket_id = next_socket_id.fetch_add(1U, std::memory_order_relaxed);
  return "/tmp/" + prefix + "_" + std::to_string(static_cast<long long>(::getpid())) + "_" +
         std::to_string(socket_id) + ".sock";
}

void cleanup_holomq_socket_path(const std::string& socket_path) noexcept {
  static_cast<void>(::unlink(socket_path.c_str()));
}

}  // namespace

HoloMQPoolFixture::HoloMQPoolFixture(std::string socket_path_arg, holomq::Context context_arg,
                                     holomq::BufferPool pool_arg,
                                     test::ScopedCudaPrimaryContext cuda_context_arg)
    : cuda_context(std::move(cuda_context_arg)),
      socket_path(std::move(socket_path_arg)),
      context(std::move(context_arg)),
      pool(std::move(pool_arg)) {}

HoloMQPoolFixture::~HoloMQPoolFixture() {
  cleanup_holomq_socket_path(socket_path);
}

holomq::expected<holomq::BufferPool, holomq::Error> make_holomq_pinned_host_pool(
    holomq::Context& context, const std::size_t buffer_bytes, const std::uint32_t buffer_count) {
  return context.create_host_pinned_buffer_pool(
      buffer_bytes, buffer_count, CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP);
}

std::unique_ptr<HoloMQPoolFixture> make_holomq_pool_fixture(const HoloMQPoolOptions& options) {
  auto cuda_context = test::ScopedCudaPrimaryContext::create();
  if (!cuda_context) {
    return nullptr;
  }

  const std::string socket_path = make_holomq_socket_path(options.socket_prefix);
  holomq::ContextBuilder context_builder;
  context_builder.with_vmm_fd_export(options.fd_export);
  if (options.fd_export) {
    context_builder.with_fd_pass_socket(socket_path);
  }
  auto context = context_builder.create();
  if (!context) {
    cleanup_holomq_socket_path(socket_path);
    return nullptr;
  }

  holomq::expected<holomq::BufferPool, holomq::Error> pool =
      options.memory_kind == HoloMQTestMemoryKind::kPinnedHost
          ? make_holomq_pinned_host_pool(*context, options.buffer_bytes, options.buffer_count)
          : context->create_device_buffer_pool(options.buffer_bytes, options.buffer_count);
  if (!pool) {
    cleanup_holomq_socket_path(socket_path);
    return nullptr;
  }

  return std::make_unique<HoloMQPoolFixture>(
      socket_path, std::move(*context), std::move(*pool), std::move(*cuda_context));
}

bool try_acquire_buffer(holomq::BufferPool& pool, std::optional<holomq::Buffer>& result) {
  auto acquired = pool.acquire();
  if (!acquired) {
    return false;
  }
  result.emplace(std::move(*acquired));
  return true;
}

bool write_host_pattern(void* const ptr, const std::size_t size_bytes, const std::uint32_t seed) {
  auto* words = static_cast<std::uint32_t*>(ptr);
  const std::size_t word_count = size_bytes / sizeof(std::uint32_t);
  for (std::size_t index = 0U; index < word_count; ++index) {
    words[index] = seed + static_cast<std::uint32_t>(index);
  }
  return true;
}

bool verify_host_pattern(const void* const ptr, const std::size_t size_bytes,
                         const std::uint32_t seed) {
  const auto* words = static_cast<const std::uint32_t*>(ptr);
  const std::size_t word_count = size_bytes / sizeof(std::uint32_t);
  for (std::size_t index = 0U; index < word_count; ++index) {
    if (words[index] != seed + static_cast<std::uint32_t>(index)) {
      return false;
    }
  }
  return true;
}

}  // namespace holoscan::channel_test
