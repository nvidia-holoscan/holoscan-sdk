/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <compare>
#include <cstdint>

namespace holomq {

class WaitSetHandle {
 public:
  WaitSetHandle() = default;

  explicit WaitSetHandle(const std::uint64_t id) noexcept : id_(id) {}

  [[nodiscard]] std::uint64_t id() const noexcept { return id_; }

  [[nodiscard]] friend auto operator<=>(const WaitSetHandle&, const WaitSetHandle&) = default;

 private:
  std::uint64_t id_{0};
};

}  // namespace holomq
