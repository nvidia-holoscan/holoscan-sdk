/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/buffer_handle.hpp>

#include <cstdint>
#include <optional>

namespace holomq {

struct RefCountKey {
  std::uint64_t owner_pid{0U};
  std::uint32_t shm_ref_count_id{0U};
};

[[nodiscard]] std::optional<RefCountKey> ref_count_key(const BufferHandle& handle) noexcept;

}  // namespace holomq
