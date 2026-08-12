/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/buffer.hpp>
#include <holomq/buffer_pool.hpp>
#include <holomq/context.hpp>

#include "../common/cuda_test_utils.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace holoscan::channel_test {

enum class HoloMQTestMemoryKind { kPinnedHost, kCudaDevice };

struct HoloMQPoolOptions {
  std::size_t buffer_bytes{};
  std::uint32_t buffer_count{};
  std::string socket_prefix;
  HoloMQTestMemoryKind memory_kind{HoloMQTestMemoryKind::kPinnedHost};
  bool fd_export{false};
};

class HoloMQPoolFixture {
 public:
  HoloMQPoolFixture(std::string socket_path, holomq::Context context, holomq::BufferPool pool,
                    test::ScopedCudaPrimaryContext cuda_context);
  HoloMQPoolFixture(const HoloMQPoolFixture&) = delete;
  HoloMQPoolFixture& operator=(const HoloMQPoolFixture&) = delete;
  HoloMQPoolFixture(HoloMQPoolFixture&&) = delete;
  HoloMQPoolFixture& operator=(HoloMQPoolFixture&&) = delete;
  ~HoloMQPoolFixture();

  test::ScopedCudaPrimaryContext cuda_context;
  std::string socket_path;
  holomq::Context context;
  holomq::BufferPool pool;
};

[[nodiscard]] std::unique_ptr<HoloMQPoolFixture> make_holomq_pool_fixture(
    const HoloMQPoolOptions& options);

[[nodiscard]] holomq::expected<holomq::BufferPool, holomq::Error> make_holomq_pinned_host_pool(
    holomq::Context& context, std::size_t buffer_bytes, std::uint32_t buffer_count);

[[nodiscard]] bool try_acquire_buffer(holomq::BufferPool& pool,
                                      std::optional<holomq::Buffer>& result);

bool write_host_pattern(void* ptr, std::size_t size_bytes, std::uint32_t seed);

[[nodiscard]] bool verify_host_pattern(const void* ptr, std::size_t size_bytes, std::uint32_t seed);

}  // namespace holoscan::channel_test
